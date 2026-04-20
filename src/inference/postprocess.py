from __future__ import annotations

from typing import Literal

import numpy as np

from src.config import PostprocessConfig
from src.shared.pose_schema import PoseSchema
from src.shared.types import Detection


# Default config instance for backward compatibility (used as defaults in function signatures).
_DEFAULT_PP = PostprocessConfig()


class KeypointStabilizer:
    """Temporal confidence smoother for per-keypoint confidence values.

    Maintains an EMA of each keypoint's confidence across frames so that
    a single bad frame doesn't yank the centroid.
    """

    def __init__(self, alpha: float = _DEFAULT_PP.conf_ema_alpha) -> None:
        self._alpha = alpha
        self._smoothed: np.ndarray | None = None

    def smooth(self, confidences: np.ndarray) -> np.ndarray:
        if self._smoothed is None or self._smoothed.shape != confidences.shape:
            self._smoothed = confidences.copy()
            return self._smoothed.copy()
        self._smoothed = self._alpha * confidences + (1.0 - self._alpha) * self._smoothed
        return self._smoothed.copy()

    def reset(self) -> None:
        self._smoothed = None

    def snapshot(self) -> np.ndarray | None:
        """Return an independent copy of the smoothed state."""
        return self._smoothed.copy() if self._smoothed is not None else None

    def restore(self, smoothed: np.ndarray | None) -> None:
        """Restore smoothed state from a previous snapshot."""
        self._smoothed = smoothed.copy() if smoothed is not None else None


def _mad_filter_keypoints(
    points: np.ndarray,
    confidences: np.ndarray,
    scale: float = _DEFAULT_PP.mad_scale,
    min_points: int = _DEFAULT_PP.mad_min_points,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject outlier keypoints using Median Absolute Deviation."""
    if len(points) < min_points:
        return points, confidences

    median = np.median(points, axis=0)
    distances = np.linalg.norm(points - median, axis=1)
    mad = np.median(distances)

    if mad < 1e-6:
        return points, confidences

    threshold = scale * mad
    keep = distances <= threshold
    if not np.any(keep):
        return points, confidences

    return points[keep], confidences[keep]


def compute_stabilized_centroid(
    detection: Detection,
    pose_schema: PoseSchema,
    stabilizer: KeypointStabilizer | None = None,
    pp_config: PostprocessConfig | None = None,
) -> tuple[float, float]:
    """Confidence-aware cascading centroid with per-target blend and smart bbox fallback.

    Cascade:
      1. Head keypoints (filtered by min_visibility, MAD, temporal smoothing)
      2. Body fallback keypoints (dogs: withers/throat)
      3. Smart bbox fallback (top-portion via bbox_fallback_y_ratio, not center)
    """
    cfg = pp_config or _DEFAULT_PP
    x1, y1, x2, y2 = detection.bbox
    bbox_cx = float((x1 + x2) * 0.5)
    bbox_ref_y = float(y1 + (y2 - y1) * pose_schema.bbox_fallback_y_ratio)

    if detection.keypoints is None:
        return bbox_cx, bbox_ref_y

    # --- Extract head keypoints above visibility threshold ---
    head_result = _extract_weighted_centroid(
        detection.keypoints,
        pose_schema.head_keypoint_indices,
        pose_schema.head_keypoint_weights,
        pose_schema.min_keypoint_visibility,
        stabilizer,
        cfg,
    )

    if head_result is not None:
        head_x, head_y, head_confs, n_head = head_result
        mean_conf = float(head_confs.mean()) if len(head_confs) > 0 else 0.0

        # Nose-priority boost: bias toward nose when it's high-confidence
        if pose_schema.nose_keypoint_index is not None:
            nose_idx = pose_schema.nose_keypoint_index
            if nose_idx < len(detection.keypoints):
                nose_vis = float(detection.keypoints[nose_idx, 2])
                if nose_vis > cfg.nose_priority_threshold:
                    nose_x = float(detection.keypoints[nose_idx, 0])
                    nose_y = float(detection.keypoints[nose_idx, 1])
                    nb = cfg.nose_priority_blend
                    head_x = (1.0 - nb) * head_x + nb * nose_x
                    head_y = (1.0 - nb) * head_y + nb * nose_y

        # Determine blend alpha based on keypoint quality
        if n_head >= cfg.high_quality_min_keypoints and mean_conf > cfg.high_quality_min_conf:
            alpha = pose_schema.head_blend_alpha
        else:
            alpha = max(0.0, pose_schema.head_blend_alpha - cfg.medium_quality_alpha_reduction)

        blended_x = alpha * head_x + (1.0 - alpha) * bbox_cx
        blended_y = alpha * head_y + (1.0 - alpha) * bbox_ref_y
        return blended_x, blended_y

    # --- Body fallback (dogs: withers/throat) ---
    if pose_schema.body_fallback_indices:
        body_result = _extract_weighted_centroid(
            detection.keypoints,
            pose_schema.body_fallback_indices,
            pose_schema.body_fallback_weights,
            pose_schema.min_keypoint_visibility,
            None,  # no temporal smoothing for body fallback
            cfg,
        )
        if body_result is not None:
            body_x, body_y, _, _ = body_result
            # Blend body centroid with bbox fallback position
            bfw = cfg.body_fallback_head_weight
            blended_x = bfw * body_x + (1.0 - bfw) * bbox_cx
            blended_y = bfw * body_y + (1.0 - bfw) * bbox_ref_y
            return blended_x, blended_y

    # --- Final fallback: smart bbox position (top-portion, not center) ---
    return bbox_cx, bbox_ref_y


def _extract_weighted_centroid(
    keypoints: np.ndarray,
    indices: tuple[int, ...],
    weights: dict[int, float],
    min_visibility: float,
    stabilizer: KeypointStabilizer | None,
    pp_config: PostprocessConfig | None = None,
) -> tuple[float, float, np.ndarray, int] | None:
    """Extract confidence-weighted centroid from a set of keypoint indices.

    Returns (cx, cy, filtered_confs, n_valid) or None if no valid keypoints.
    """
    valid_indices: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    raw_confs: list[float] = []

    for kp_idx in indices:
        if kp_idx >= len(keypoints):
            continue
        x = float(keypoints[kp_idx, 0])
        y = float(keypoints[kp_idx, 1])
        vis = float(keypoints[kp_idx, 2])
        if vis > min_visibility:
            valid_indices.append(kp_idx)
            xs.append(x)
            ys.append(y)
            raw_confs.append(vis * weights.get(kp_idx, 1.0))

    if not valid_indices:
        return None

    points = np.array(list(zip(xs, ys)), dtype=np.float32)
    confs = np.array(raw_confs, dtype=np.float32)

    # Temporal confidence smoothing
    if stabilizer is not None:
        full_confs = np.zeros(len(keypoints), dtype=np.float32)
        for i, kp_idx in enumerate(valid_indices):
            full_confs[kp_idx] = confs[i]
        smoothed_full = stabilizer.smooth(full_confs)
        confs = np.array([smoothed_full[kp_idx] for kp_idx in valid_indices], dtype=np.float32)

    # MAD outlier rejection
    _cfg = pp_config or _DEFAULT_PP
    points, confs = _mad_filter_keypoints(
        points, confs, scale=_cfg.mad_scale, min_points=_cfg.mad_min_points
    )
    if len(points) == 0:
        return None

    total_weight = confs.sum()
    if total_weight < 1e-9:
        return None

    weighted = (points * confs[:, np.newaxis]).sum(axis=0) / total_weight
    return float(weighted[0]), float(weighted[1]), confs, len(confs)


def parse_yolo_output(
    raw: np.ndarray,
    conf_threshold: float,
    num_keypoints: int,
    *,
    nms_threshold: float = 0.45,
) -> list[Detection]:
    predictions, box_format, class_index = _reshape_predictions(raw, num_keypoints)
    if predictions.size == 0:
        return []

    scores = predictions[:, 4]
    keep_mask = scores >= conf_threshold
    if not np.any(keep_mask):
        return []

    filtered = predictions[keep_mask]
    filtered_scores = filtered[:, 4]
    boxes_xyxy = filtered[:, :4] if box_format == "xyxy" else xywh_to_xyxy(filtered[:, :4])

    # If TRT already applied NMS (xyxy format with class_index), skip Python NMS
    if box_format == "xyxy" and class_index is not None:
        keep_indices = np.arange(len(filtered))
    else:
        keep_indices = non_max_suppression(boxes_xyxy, filtered_scores, nms_threshold)

    keypoint_start = 5 if class_index is None else class_index + 1
    keypoint_end = keypoint_start + (num_keypoints * 3)
    selected = filtered[keep_indices]
    selected_boxes = boxes_xyxy[keep_indices]
    keypoints = selected[:, keypoint_start:keypoint_end].reshape(-1, num_keypoints, 3)
    detections: list[Detection] = []
    for index, row in enumerate(selected):
        detections.append(
            Detection(
                bbox=selected_boxes[index],
                score=float(row[4]),
                keypoints=keypoints[index],
                class_id=0 if class_index is None else int(row[class_index]),
            )
        )
    return detections


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    boxes = boxes_xywh.astype(np.float32, copy=True)
    boxes[:, 0] = boxes_xywh[:, 0] - (boxes_xywh[:, 2] * 0.5)
    boxes[:, 1] = boxes_xywh[:, 1] - (boxes_xywh[:, 3] * 0.5)
    boxes[:, 2] = boxes_xywh[:, 0] + (boxes_xywh[:, 2] * 0.5)
    boxes[:, 3] = boxes_xywh[:, 1] + (boxes_xywh[:, 3] * 0.5)
    return boxes


def non_max_suppression(
    boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return np.empty((0,), dtype=np.int32)

    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        ious = _single_box_iou(boxes_xyxy[current], boxes_xyxy[remaining])
        order = remaining[ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int32)


def _single_box_iou(box: np.ndarray, other_boxes: np.ndarray) -> np.ndarray:
    top_left = np.maximum(box[:2], other_boxes[:, :2])
    bottom_right = np.minimum(box[2:], other_boxes[:, 2:])
    overlap = np.clip(bottom_right - top_left, a_min=0.0, a_max=None)
    intersection = overlap[:, 0] * overlap[:, 1]

    box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    other_areas = np.clip(other_boxes[:, 2] - other_boxes[:, 0], 0.0, None) * np.clip(
        other_boxes[:, 3] - other_boxes[:, 1], 0.0, None
    )
    union = box_area + other_areas - intersection
    safe_union = np.where(union <= 0.0, 1.0, union)
    return (intersection / safe_union).astype(np.float32)


def _reshape_predictions(
    raw: np.ndarray, num_keypoints: int
) -> tuple[np.ndarray, Literal["xywh", "xyxy"], int | None]:
    raw_attrs = 5 + (num_keypoints * 3)
    nms_attrs = 6 + (num_keypoints * 3)
    array = np.asarray(raw)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"expected 2D or batched 3D output, got {array.shape}")

    if array.shape[1] == raw_attrs:
        return array.astype(np.float32, copy=False), "xywh", None
    if array.shape[0] == raw_attrs:
        return array.T.astype(np.float32, copy=False), "xywh", None
    if array.shape[1] == nms_attrs:
        return array.astype(np.float32, copy=False), "xyxy", 5
    if array.shape[0] == nms_attrs:
        return array.T.astype(np.float32, copy=False), "xyxy", 5
    raise ValueError(
        f"unexpected TensorRT output shape {array.shape};"
        f" expected attribute dimensions {raw_attrs} or {nms_attrs}"
    )
