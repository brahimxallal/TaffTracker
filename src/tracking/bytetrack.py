from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.shared.types import Detection, Track


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    track_threshold: float = 0.45
    low_threshold: float = 0.10
    match_threshold: float = 0.55
    max_lost: int = 30
    birth_min_hits: int = 2


class ByteTracker:
    def __init__(
        self,
        track_thresh: float = 0.45,
        match_thresh: float = 0.55,
        max_lost: int = 30,
        low_thresh: float = 0.10,
        birth_min_hits: int = 2,
    ) -> None:
        self._config = TrackerConfig(
            track_threshold=track_thresh,
            low_threshold=low_thresh,
            match_threshold=match_thresh,
            max_lost=max_lost,
            birth_min_hits=birth_min_hits,
        )
        self._next_track_id = 1
        self._tracks: list[Track] = []
        self._tentative: list[Track] = []

    def reset(self) -> None:
        self._next_track_id = 1
        self._tracks.clear()
        self._tentative.clear()

    def set_track_threshold(self, threshold: float) -> None:
        """Update the high-confidence split threshold at runtime."""
        self._config = TrackerConfig(
            track_threshold=threshold,
            low_threshold=self._config.low_threshold,
            match_threshold=self._config.match_threshold,
            max_lost=self._config.max_lost,
            birth_min_hits=self._config.birth_min_hits,
        )

    def update(self, detections: list[Detection], timestamp_ns: int = 0) -> list[Track]:
        track_threshold = self._config.track_threshold
        high_conf = [detection for detection in detections if detection.score >= track_threshold]
        low_conf = [
            detection
            for detection in detections
            if self._config.low_threshold <= detection.score < track_threshold
        ]

        unmatched_track_indices = list(range(len(self._tracks)))
        unmatched_high_indices = list(range(len(high_conf)))

        matches, unmatched_track_indices, unmatched_high_indices = self._match_detections(
            self._tracks,
            high_conf,
            match_threshold=self._config.match_threshold,
        )
        self._apply_matches(matches, self._tracks, high_conf, timestamp_ns)

        remaining_tracks = [self._tracks[index] for index in unmatched_track_indices]
        low_matches, still_unmatched_tracks, _ = self._match_detections(
            remaining_tracks,
            low_conf,
            match_threshold=self._config.match_threshold,
        )
        self._apply_matches(low_matches, remaining_tracks, low_conf, timestamp_ns)

        still_unmatched_track_ids = {
            remaining_tracks[index].track_id for index in still_unmatched_tracks
        }
        for track in self._tracks:
            if track.track_id in still_unmatched_track_ids:
                track.lost_frames += 1
                track.age += 1

        # Birth hysteresis: new detections go to tentative first
        birth_min = self._config.birth_min_hits
        if birth_min <= 1:
            # No hysteresis — promote immediately (original behaviour)
            for detection_index in unmatched_high_indices:
                detection = high_conf[detection_index]
                self._tracks.append(
                    Track(
                        track_id=self._next_track_id,
                        bbox=detection.bbox.copy(),
                        score=float(detection.score),
                        keypoints=(
                            None if detection.keypoints is None else detection.keypoints.copy()
                        ),
                        age=1,
                        lost_frames=0,
                        last_timestamp_ns=timestamp_ns,
                    )
                )
                self._next_track_id += 1
        else:
            # Match tentative tracks against unmatched high-conf detections
            unmatched_new_detections = list(unmatched_high_indices)
            if self._tentative and unmatched_new_detections:
                tent_dets = [high_conf[i] for i in unmatched_new_detections]
                tent_matches, unmatched_tent, unmatched_det = self._match_detections(
                    self._tentative, tent_dets, match_threshold=self._config.match_threshold
                )
                for tent_idx, det_idx in tent_matches:
                    tent_track = self._tentative[tent_idx]
                    det = tent_dets[det_idx]
                    tent_track.bbox = det.bbox.copy()
                    tent_track.score = float(det.score)
                    tent_track.keypoints = None if det.keypoints is None else det.keypoints.copy()
                    tent_track.age += 1
                    tent_track.last_timestamp_ns = timestamp_ns
                # Promote tentative tracks that reached birth_min_hits
                promoted = set()
                for tent_idx, _ in tent_matches:
                    tent_track = self._tentative[tent_idx]
                    if tent_track.age >= birth_min:
                        self._tracks.append(tent_track)
                        promoted.add(tent_idx)
                # Remove unmatched tentative tracks (one miss = discard)
                surviving_tent = {idx for idx, _ in tent_matches} - promoted
                self._tentative = [
                    self._tentative[i] for i in range(len(self._tentative)) if i in surviving_tent
                ]
                # Remaining unmatched detections become new tentative
                unmatched_new_detections = [unmatched_new_detections[i] for i in unmatched_det]
            else:
                self._tentative = []

            for detection_index in unmatched_new_detections:
                detection = high_conf[detection_index]
                self._tentative.append(
                    Track(
                        track_id=self._next_track_id,
                        bbox=detection.bbox.copy(),
                        score=float(detection.score),
                        keypoints=(
                            None if detection.keypoints is None else detection.keypoints.copy()
                        ),
                        age=1,
                        lost_frames=0,
                        last_timestamp_ns=timestamp_ns,
                    )
                )
                self._next_track_id += 1

        self._tracks = [
            track for track in self._tracks if track.lost_frames <= self._config.max_lost
        ]
        return [self._copy_track(track) for track in self._tracks]

    def _apply_matches(
        self,
        matches: list[tuple[int, int]],
        tracks: list[Track],
        detections: list[Detection],
        timestamp_ns: int,
    ) -> None:
        for track_index, detection_index in matches:
            track = tracks[track_index]
            detection = detections[detection_index]
            track.bbox = detection.bbox.copy()
            track.score = float(detection.score)
            track.keypoints = None if detection.keypoints is None else detection.keypoints.copy()
            track.lost_frames = 0
            track.age += 1
            track.last_timestamp_ns = timestamp_ns

    def _match_detections(
        self,
        tracks: list[Track],
        detections: list[Detection],
        *,
        match_threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Fast path: skip Hungarian assignment for the common single-target case
        if len(tracks) == 1 and len(detections) == 1:
            iou = _scalar_iou(tracks[0].bbox, detections[0].bbox)
            if iou >= match_threshold:
                return [(0, 0)], [], []
            return [], [0], [0]

        track_boxes = np.stack([track.bbox for track in tracks], axis=0)
        detection_boxes = np.stack([detection.bbox for detection in detections], axis=0)
        iou_matrix = compute_iou_matrix(track_boxes, detection_boxes)
        cost_matrix = 1.0 - iou_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches: list[tuple[int, int]] = []
        matched_rows: set[int] = set()
        matched_cols: set[int] = set()
        for row_index, col_index in zip(row_indices, col_indices, strict=False):
            if iou_matrix[row_index, col_index] < match_threshold:
                continue
            matches.append((int(row_index), int(col_index)))
            matched_rows.add(int(row_index))
            matched_cols.add(int(col_index))

        unmatched_tracks = [index for index in range(len(tracks)) if index not in matched_rows]
        unmatched_detections = [
            index for index in range(len(detections)) if index not in matched_cols
        ]
        return matches, unmatched_tracks, unmatched_detections

    def _copy_track(self, track: Track) -> Track:
        return Track(
            track_id=track.track_id,
            bbox=track.bbox.copy(),
            score=float(track.score),
            keypoints=None if track.keypoints is None else track.keypoints.copy(),
            age=track.age,
            lost_frames=track.lost_frames,
            last_timestamp_ns=track.last_timestamp_ns,
        )


def _scalar_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.empty((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    overlap = np.clip(bottom_right - top_left, a_min=0.0, a_max=None)
    intersection = overlap[..., 0] * overlap[..., 1]

    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0.0, None) * np.clip(
        boxes_a[:, 3] - boxes_a[:, 1], 0.0, None
    )
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0.0, None) * np.clip(
        boxes_b[:, 3] - boxes_b[:, 1], 0.0, None
    )
    union = area_a[:, None] + area_b[None, :] - intersection
    safe_union = np.where(union <= 0.0, 1.0, union)
    return (intersection / safe_union).astype(np.float32)
