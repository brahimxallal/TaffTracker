from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import logging

from cuda.bindings import runtime as cudart
import numpy as np
import tensorrt as trt


LOGGER = logging.getLogger("trt_engine")

# CUDA graph capture warmup: number of inferences before capturing the graph.
CUDA_GRAPH_WARMUP_FRAMES: int = 5


@dataclass(frozen=True, slots=True)
class TensorBinding:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    device_ptr: int
    host_array: np.ndarray
    host_ptr: int
    device_array: Any | None = None


def _status_code(status: Any) -> int:
    return int(getattr(status, "value", status))


def _checked_cuda(result: Any, label: str) -> Any:
    if isinstance(result, tuple):
        status = result[0]
        values = result[1:]
    else:
        status = result
        values = ()

    if _status_code(status) != 0:
        raise RuntimeError(f"{label} failed with CUDA status {status}")

    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


class TRTEngine:
    def __init__(
        self,
        engine_path: str | Path,
        input_shape: tuple[int, ...] = (1, 3, 640, 640),
        *,
        use_cuda_graph: bool = True,
    ) -> None:
        self._engine_path = Path(engine_path)
        if not self._engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self._engine_path}")

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        serialized_engine = self._read_serialized_engine(self._engine_path)

        self._engine = self._runtime.deserialize_cuda_engine(serialized_engine)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self._engine_path}")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self._stream = _checked_cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")
        self._bindings: dict[str, TensorBinding] = {}
        self._input_name = self._resolve_input_name()
        self._set_input_shape(input_shape)
        self._allocate_bindings()

        # PyTorch stream wrapping the TRT CUDA stream — all GPU work stays on one stream
        import torch

        self._torch_stream = torch.cuda.ExternalStream(int(self._stream))

        # Cache output tensor names (avoids dict iteration per frame)
        self._output_names_cached = self.output_names

        # CUDA graph state
        self._use_cuda_graph = use_cuda_graph
        self._cuda_graph = None
        self._cuda_graph_exec = None
        self._warmup_count = 0

    @property
    def input_name(self) -> str:
        return self._input_name

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in self._bindings
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        )

    def infer(self, frame: np.ndarray) -> np.ndarray | dict[str, np.ndarray]:
        import torch

        input_binding = self._bindings[self._input_name]

        # GPU Preprocessing via PyTorch (pre-compiled CUDA kernels — no JIT needed)
        with torch.cuda.stream(self._torch_stream):
            # Upload uint8 HWC BGR frame to GPU
            t = torch.from_numpy(frame).to(device="cuda", non_blocking=True)
            # uint8 → float32 + normalise to [0, 1]
            t = t.float().div_(255.0)
            # HWC (H,W,3) → CHW (3,H,W), BGR → RGB via flip, add batch dim
            t = t.permute(2, 0, 1).flip(0).unsqueeze(0).contiguous()

        # Device-to-device copy: torch tensor → TRT input binding (same stream)
        _checked_cuda(
            cudart.cudaMemcpyAsync(
                input_binding.device_ptr,
                t.data_ptr(),
                input_binding.host_array.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                self._stream,
            ),
            "cudaMemcpyAsync(preprocess→TRT)",
        )

        if self._cuda_graph_exec is not None:
            # Replay captured graph (inference + D2H)
            _checked_cuda(
                cudart.cudaGraphLaunch(self._cuda_graph_exec, self._stream),
                "cudaGraphLaunch",
            )
        else:
            # Normal execution path
            self._execute_inference_and_d2h()

            # Attempt graph capture after warmup
            if self._use_cuda_graph:
                self._warmup_count += 1
                if self._warmup_count == CUDA_GRAPH_WARMUP_FRAMES:
                    self._capture_cuda_graph()

        _checked_cuda(cudart.cudaStreamSynchronize(self._stream), "cudaStreamSynchronize")
        return self._collect_outputs()

    def close(self) -> None:
        if self._cuda_graph_exec is not None:
            _checked_cuda(
                cudart.cudaGraphExecDestroy(self._cuda_graph_exec), "cudaGraphExecDestroy"
            )
            self._cuda_graph_exec = None
        if self._cuda_graph is not None:
            _checked_cuda(cudart.cudaGraphDestroy(self._cuda_graph), "cudaGraphDestroy")
            self._cuda_graph = None
        for binding in self._bindings.values():
            _checked_cuda(cudart.cudaFree(binding.device_ptr), f"cudaFree({binding.name})")
            _checked_cuda(cudart.cudaFreeHost(binding.host_ptr), f"cudaFreeHost({binding.name})")
        self._bindings.clear()
        _checked_cuda(cudart.cudaStreamDestroy(self._stream), "cudaStreamDestroy")

    def _execute_inference_and_d2h(self) -> None:
        if not self._context.execute_async_v3(self._stream):
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        for name in self._output_names_cached:
            binding = self._bindings[name]
            _checked_cuda(
                cudart.cudaMemcpyAsync(
                    binding.host_array.ctypes.data,
                    binding.device_ptr,
                    binding.host_array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self._stream,
                ),
                "cudaMemcpyAsync(D2H)",
            )

    def _collect_outputs(self) -> np.ndarray | dict[str, np.ndarray]:
        outputs: dict[str, np.ndarray] = {
            name: self._bindings[name].host_array for name in self._output_names_cached
        }
        if len(outputs) == 1:
            return next(iter(outputs.values()))
        return outputs

    def _capture_cuda_graph(self) -> None:
        try:
            _checked_cuda(
                cudart.cudaStreamBeginCapture(
                    self._stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
                ),
                "cudaStreamBeginCapture",
            )
            self._execute_inference_and_d2h()
            self._cuda_graph = _checked_cuda(
                cudart.cudaStreamEndCapture(self._stream),
                "cudaStreamEndCapture",
            )
            self._cuda_graph_exec = _checked_cuda(
                cudart.cudaGraphInstantiate(self._cuda_graph, 0),
                "cudaGraphInstantiate",
            )
            LOGGER.info("CUDA graph captured successfully (warmup=%d)", self._warmup_count)
        except RuntimeError as exc:
            LOGGER.warning("CUDA graph capture failed (%s), falling back to eager execution", exc)
            self._use_cuda_graph = False
            self._cuda_graph = None
            self._cuda_graph_exec = None

    def _resolve_input_name(self) -> str:
        input_names = [
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
            if self._engine.get_tensor_mode(self._engine.get_tensor_name(index))
            == trt.TensorIOMode.INPUT
        ]
        if len(input_names) != 1:
            raise RuntimeError(f"Expected exactly one input tensor, found {input_names}")
        return input_names[0]

    def _read_serialized_engine(self, engine_path: Path) -> bytes:
        with engine_path.open("rb") as engine_file:
            try:
                metadata_length = int.from_bytes(engine_file.read(4), byteorder="little")
                json.loads(engine_file.read(metadata_length).decode("utf-8"))
            except UnicodeDecodeError:
                engine_file.seek(0)
            return engine_file.read()

    def _set_input_shape(self, input_shape: tuple[int, ...]) -> None:
        current_shape = tuple(self._engine.get_tensor_shape(self._input_name))
        if -1 in current_shape:
            self._context.set_input_shape(self._input_name, input_shape)

    def _allocate_bindings(self) -> None:
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            shape = tuple(int(dim) for dim in self._context.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
            nbytes = int(np.prod(shape)) * dtype.itemsize
            host_ptr = int(
                _checked_cuda(
                    cudart.cudaHostAlloc(nbytes, cudart.cudaHostAllocDefault),
                    f"cudaHostAlloc({name})",
                )
            )
            host_array = np.ndarray(
                shape, dtype=dtype, buffer=(ctypes.c_uint8 * nbytes).from_address(host_ptr)
            )
            device_ptr = int(_checked_cuda(cudart.cudaMalloc(nbytes), f"cudaMalloc({name})"))
            self._context.set_tensor_address(name, device_ptr)

            self._bindings[name] = TensorBinding(
                name=name,
                shape=shape,
                dtype=dtype,
                device_ptr=device_ptr,
                host_array=host_array,
                host_ptr=host_ptr,
            )
