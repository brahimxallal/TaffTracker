from __future__ import annotations

import argparse
from pathlib import Path
import logging
import shutil

from ultralytics import YOLO


LOGGER = logging.getLogger("export_engines")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO pose models to TensorRT engines")
    parser.add_argument(
        "--workspace",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Workspace root that contains the models directory",
    )
    parser.add_argument(
        "--target",
        choices=("human", "dog", "all"),
        default="all",
        help="Which engine to export",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Square inference size")
    parser.add_argument("--device", default="0", help="CUDA device index")
    parser.add_argument(
        "--keep-onnx",
        action="store_true",
        help="Keep the intermediate ONNX file after a successful TensorRT export",
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "int8"),
        default="fp16",
        help="TensorRT precision mode. INT8 requires a calibration dataset (auto-generated if missing).",
    )
    parser.add_argument(
        "--calib-images",
        type=Path,
        default=None,
        help="Directory of calibration images for INT8 quantization. "
        "If omitted, uses calibration_data/int8_calib/ or auto-generates from camera.",
    )
    parser.add_argument(
        "--calib-frames",
        type=int,
        default=200,
        help="Number of frames to capture for INT8 calibration dataset (default: 200)",
    )
    return parser.parse_args()


def _write_calib_yaml(calib_dir: Path) -> Path:
    """Write a YAML dataset file that Ultralytics uses for INT8 calibration."""
    yaml_path = calib_dir.parent / f"{calib_dir.name}.yaml"
    yaml_path.write_text(
        f"# Auto-generated INT8 calibration dataset\n"
        f"path: {calib_dir.resolve()}\n"
        f"train: images/train\nval: images/train\n"
        f"kpt_shape: [17, 3]\n"
        f"names:\n  0: object\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrote calibration YAML: %s", yaml_path)
    return yaml_path


def _ensure_calibration_images(
    calib_dir: Path, source: int | str = 0, num_frames: int = 200, imgsz: int = 640
) -> Path:
    """Ensure calibration images exist, capturing from camera if needed."""
    img_dir = calib_dir / "images" / "train"
    lbl_dir = calib_dir / "labels" / "train"
    existing = list(img_dir.glob("*.jpg")) if img_dir.exists() else []
    if len(existing) >= 50:
        LOGGER.info("Using existing calibration dataset: %s (%d images)", img_dir, len(existing))
        return _write_calib_yaml(calib_dir)

    LOGGER.info("Generating INT8 calibration dataset from camera (source=%s, frames=%d)", source, num_frames)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np

    cap = cv2.VideoCapture(source if isinstance(source, int) else int(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source {source} for calibration capture")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    captured = 0
    skip_interval = 3  # Skip frames to get temporal diversity
    frame_idx = 0

    while captured < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % skip_interval != 0:
            continue

        # Letterbox to inference size (match runtime preprocessing)
        h, w = frame.shape[:2]
        scale = min(imgsz / w, imgsz / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        pad_top = (imgsz - new_h) // 2
        pad_left = (imgsz - new_w) // 2
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        cv2.imwrite(str(img_dir / f"calib_{captured:04d}.jpg"), canvas)
        (lbl_dir / f"calib_{captured:04d}.txt").touch()
        captured += 1

        if captured % 50 == 0:
            LOGGER.info("Captured %d/%d calibration frames...", captured, num_frames)

    cap.release()
    LOGGER.info("Calibration dataset ready: %d images in %s", captured, calib_dir)

    # Generate YAML that Ultralytics expects for INT8 calibration
    yaml_path = _write_calib_yaml(calib_dir)
    return yaml_path


def export_model(
    model_path: Path,
    imgsz: int,
    device: str,
    precision: str = "fp16",
    calib_data: Path | None = None,
) -> Path:
    LOGGER.info("Exporting %s (precision=%s)", model_path.name, precision)
    model = YOLO(model_path)

    export_kwargs: dict = dict(
        format="engine",
        imgsz=imgsz,
        device=device,
        dynamic=False,
        nms=True,
        simplify=True,
        workspace=4,
    )

    if precision == "int8":
        export_kwargs["int8"] = True
        export_kwargs["half"] = True  # INT8 with FP16 fallback for non-quantizable layers
        if calib_data is not None and calib_data.exists():
            export_kwargs["data"] = str(calib_data)
            LOGGER.info("Using calibration data: %s", calib_data)
    else:
        export_kwargs["half"] = True

    engine_path = Path(model.export(**export_kwargs))
    LOGGER.info("Created engine %s", engine_path)
    return engine_path


def _engine_name(base_name: str, precision: str) -> str:
    """Generate engine filename with precision suffix for INT8, plain for FP16.

    FP16 keeps backward-compatible names:  yolo26n-person-17pose.engine
    INT8 gets explicit suffix:             yolo26n-person-17pose-int8.engine
    """
    stem = base_name.rsplit(".", 1)[0]
    if precision == "int8":
        return f"{stem}-int8.engine"
    return base_name


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    workspace = args.workspace.resolve()
    models_dir = workspace / "models"
    engines_dir = workspace / "engines"
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    engines_dir.mkdir(parents=True, exist_ok=True)

    _MODEL_MAP: dict[str, tuple[str, str]] = {
        "human": ("yolo26n-person-17pose.pt", "yolo26n-person-17pose.engine"),
        "dog": ("Enhanceddog/best.pt", "enhanced-dog-24pose.engine"),
    }

    # Resolve calibration data for INT8
    calib_data: Path | None = None
    if args.precision == "int8":
        import numpy as np  # noqa: F811 — needed for calibration capture

        if args.calib_images is not None:
            calib_data = args.calib_images.resolve()
            if not calib_data.exists():
                raise FileNotFoundError(f"Calibration image directory not found: {calib_data}")
            calib_data = _write_calib_yaml(calib_data)
        else:
            calib_dir = workspace / "calibration_data" / "int8_calib"
            calib_data = _ensure_calibration_images(calib_dir, num_frames=args.calib_frames, imgsz=args.imgsz)

    exports: list[Path] = []
    targets = [args.target] if args.target != "all" else ["human", "dog"]
    for target in targets:
        model_rel, engine_base = _MODEL_MAP[target]
        model_path = models_dir / model_rel
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        exported_path = export_model(
            model_path, imgsz=args.imgsz, device=args.device,
            precision=args.precision, calib_data=calib_data,
        )

        engine_name = _engine_name(engine_base, args.precision)
        destination = engines_dir / engine_name
        if exported_path.resolve() != destination.resolve():
            shutil.move(str(exported_path), str(destination))
        if not args.keep_onnx:
            # Clean up ONNX from the model's directory (may be a subdirectory)
            onnx_path = model_path.with_suffix(".onnx")
            if onnx_path.exists():
                onnx_path.unlink()
                LOGGER.info("Removed intermediate ONNX %s", onnx_path.name)
        exports.append(destination)

    LOGGER.info("Export complete (%s): %s", args.precision.upper(), ", ".join(path.name for path in exports))


if __name__ == "__main__":
    main()
