"""Pure helpers used by :class:`InferenceProcess` during startup.

These functions exist as module-level callables so they can be unit-tested
without instantiating the whole ``InferenceProcess`` shell and without
importing TensorRT / CUDA at test collection time. The ``InferenceProcess``
class keeps thin wrapper methods that delegate here, which preserves the
existing test surface while isolating the logic.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.calibration.camera_model import CameraModel
from src.config import CameraConfig, Mode, ModelConfig, RuntimePaths, TargetKind
from src.shared.pose_schema import PoseSchema, get_pose_schema


def resolve_engine_path(target: TargetKind, model_config: ModelConfig) -> Path:
    """Return the TensorRT engine path appropriate for the requested target."""
    return model_config.person_engine_path if target == "human" else model_config.dog_engine_path


def load_camera_model(
    mode: Mode,
    camera_config: CameraConfig,
    runtime_paths: RuntimePaths,
    logger: logging.Logger,
) -> CameraModel:
    """Resolve the camera model with calibration → FOV → identity fallback.

    Priority:
      1. ``calibration_data/intrinsics.npz`` if its image size matches runtime.
      2. ``camera.fov`` from config.yaml.
      3. Identity camera model (video mode only).

    Raises :class:`ValueError` when running in ``camera`` mode with no FOV and
    no usable calibration file, because downstream aiming math would silently
    produce garbage.
    """
    calibration_path = runtime_paths.calibration_file_path()
    if calibration_path.exists():
        try:
            model = CameraModel.load(calibration_path)
        except Exception as exc:
            logger.warning(
                "Failed to load camera calibration from %s: %s",
                calibration_path,
                exc,
            )
        else:
            if model.image_size == (camera_config.width, camera_config.height):
                logger.info("Using camera calibration from %s", calibration_path)
                return model
            logger.warning(
                "Calibration image size %s does not match runtime size (%d, %d); falling back",
                model.image_size,
                camera_config.width,
                camera_config.height,
            )
    if camera_config.fov is not None:
        logger.info("Using configured FOV %.1f deg", camera_config.fov)
        return CameraModel.from_fov(camera_config.fov, camera_config.width, camera_config.height)
    if mode == "camera":
        raise ValueError(
            "camera.fov must be set or calibration_data/intrinsics.npz must exist for camera mode"
        )
    logger.warning("No FOV configured, falling back to identity camera model")
    return CameraModel.identity(camera_config.width, camera_config.height)


def load_pose_schema(
    target: TargetKind,
    runtime_paths: RuntimePaths,
    logger: logging.Logger,
) -> PoseSchema:
    """Load the pose schema for the requested target and log its provenance."""
    schema_path = runtime_paths.resolved_dog_pose_schema_path() if target == "dog" else None
    pose_schema = get_pose_schema(target, schema_path)
    logger.info(
        "Using %s pose schema from %s with %s keypoints",
        pose_schema.target_kind,
        pose_schema.source,
        pose_schema.keypoint_count,
    )
    return pose_schema


def compute_dt(
    current_timestamp_ns: int,
    last_timestamp_ns: int | None,
    camera_fps: int,
) -> float:
    """Return the frame-to-frame dt in seconds with a 1 ms floor.

    The floor exists so Kalman prediction stays numerically stable on duplicate
    timestamps (e.g., if two frames share a ns tick after a hot-swap) and on
    the first frame (where we fall back to the nominal 1/fps).
    """
    if last_timestamp_ns is None:
        return 1.0 / max(camera_fps, 1)
    return max((current_timestamp_ns - last_timestamp_ns) / 1_000_000_000.0, 1e-3)
