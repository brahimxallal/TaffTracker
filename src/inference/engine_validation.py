from __future__ import annotations

import json
from pathlib import Path


def _read_serialized_engine(engine_path: Path) -> bytes:
    with engine_path.open("rb") as engine_file:
        try:
            metadata_length = int.from_bytes(engine_file.read(4), byteorder="little")
            json.loads(engine_file.read(metadata_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            engine_file.seek(0)
        return engine_file.read()


def _undeserializable_engine_error(engine_path: Path, trt_version: str) -> RuntimeError:
    return RuntimeError(
        "Selected Python/TensorRT runtime cannot deserialize "
        f"{engine_path} (TensorRT {trt_version}). Run from the project virtual environment "
        "the engine was built with, or rebuild it with scripts/export_engines.py."
    )


def validate_tensorrt_engine_compatibility(engine_path: Path) -> None:
    """Verify the selected TensorRT runtime can deserialize an engine file.

    This intentionally stops after deserialization. Full ``TRTEngine`` startup
    allocates CUDA buffers and imports torch; the CLI preflight only needs the
    early incompatibility signal that catches wrong TensorRT/runtime versions.
    """
    try:
        import tensorrt as trt
    except Exception as exc:  # pragma: no cover - depends on host environment
        raise RuntimeError(
            "TensorRT is not importable in the selected Python environment; "
            f"cannot validate engine {engine_path}."
        ) from exc

    trt_version = getattr(trt, "__version__", "unknown")
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(_read_serialized_engine(engine_path))
    except Exception as exc:
        raise _undeserializable_engine_error(engine_path, trt_version) from exc

    if engine is None:
        raise _undeserializable_engine_error(engine_path, trt_version)
