from pathlib import Path

import pytest

from src.config import ModelConfig, PipelineConfig, RuntimePaths
from src.main import validate_environment


@pytest.fixture()
def engine_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "engines"
    directory.mkdir()
    (directory / "enhanced-dog-24pose.engine").write_bytes(b"engine")
    (directory / "yolo26n-person-17pose.engine").write_bytes(b"engine")
    return directory


@pytest.mark.unit
def test_validate_environment_rejects_missing_fov_in_camera_mode(
    engine_dir: Path,
    tmp_path: Path,
) -> None:
    config = PipelineConfig(
        mode="camera",
        target="dog",
        source="0",
        models=ModelConfig(engine_dir=engine_dir),
        paths=RuntimePaths(calibration_dir=tmp_path / "calibration_data"),
    )

    with pytest.raises(ValueError, match="camera.fov must be set"):
        validate_environment(config)


@pytest.mark.unit
def test_validate_environment_allows_video_mode_without_calibration(
    engine_dir: Path, tmp_path: Path
) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    config = PipelineConfig(
        mode="video",
        target="dog",
        source=str(video_path),
        models=ModelConfig(engine_dir=engine_dir),
        paths=RuntimePaths(calibration_dir=tmp_path / "calibration_data"),
    )

    validate_environment(config)
