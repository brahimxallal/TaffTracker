from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class TrackAppearance:
    track_id: int
    histogram: np.ndarray
    timestamp_ns: int
    centroid: tuple[float, float] = (0.0, 0.0)  # bbox center for spatial check


class ReIDBuffer:
    """Lightweight re-identification using HSV color histograms + spatial proximity.

    When a track is lost, its appearance histogram and last centroid are stored.
    When a new track appears, it is compared against stored appearances using
    both histogram correlation AND spatial distance. Both must pass thresholds
    for a positive match.
    """

    def __init__(
        self,
        max_stored: int = 5,
        correlation_threshold: float = 0.5,
        max_age_ns: int = 10_000_000_000,  # 10 seconds
        max_spatial_distance_px: float = 200.0,  # max centroid shift for re-ID
    ) -> None:
        self._max_stored = max_stored
        self._threshold = correlation_threshold
        self._max_age_ns = max_age_ns
        self._max_spatial_dist = max_spatial_distance_px
        self._lost_tracks: OrderedDict[int, TrackAppearance] = OrderedDict()

    def compute_histogram(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        """Compute an 8x8x8 HSV histogram for the bbox region."""
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def store_lost_track(
        self, track_id: int, frame: np.ndarray, bbox: np.ndarray, timestamp_ns: int
    ) -> None:
        """Store appearance of a track that was just lost."""
        hist = self.compute_histogram(frame, bbox)
        if hist is None:
            return
        cx = float((bbox[0] + bbox[2]) * 0.5)
        cy = float((bbox[1] + bbox[3]) * 0.5)
        appearance = TrackAppearance(
            track_id=track_id,
            histogram=hist,
            timestamp_ns=timestamp_ns,
            centroid=(cx, cy),
        )
        self._lost_tracks[track_id] = appearance
        # Evict oldest if over capacity
        while len(self._lost_tracks) > self._max_stored:
            self._lost_tracks.popitem(last=False)

    def match(self, frame: np.ndarray, bbox: np.ndarray, timestamp_ns: int) -> int | None:
        """Try to match a new detection against stored lost tracks.

        Returns the old track_id if a match is found, None otherwise.
        """
        if not self._lost_tracks:
            return None
        hist = self.compute_histogram(frame, bbox)
        if hist is None:
            return None

        # Evict expired entries
        expired = [
            tid
            for tid, app in self._lost_tracks.items()
            if timestamp_ns - app.timestamp_ns > self._max_age_ns
        ]
        for tid in expired:
            del self._lost_tracks[tid]

        best_id: int | None = None
        best_score = self._threshold  # must exceed threshold to match
        # Compute new track centroid for spatial check
        new_cx = float((bbox[0] + bbox[2]) * 0.5)
        new_cy = float((bbox[1] + bbox[3]) * 0.5)
        for tid, appearance in self._lost_tracks.items():
            # Spatial gate: reject if centroid moved too far
            dx = new_cx - appearance.centroid[0]
            dy = new_cy - appearance.centroid[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > self._max_spatial_dist:
                continue
            score = float(cv2.compareHist(hist, appearance.histogram, cv2.HISTCMP_CORREL))
            if score > best_score:
                best_score = score
                best_id = tid

        if best_id is not None:
            del self._lost_tracks[best_id]
        return best_id

    def clear(self) -> None:
        self._lost_tracks.clear()
