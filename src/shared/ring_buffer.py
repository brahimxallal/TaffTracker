from __future__ import annotations

import ctypes
import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.sharedctypes import Synchronized
from typing import Self

import numpy as np

_META_DTYPE = np.dtype(
    [("timestamp_ns", np.int64), ("frame_id", np.int64), ("generation", np.uint64)]
)


@dataclass(frozen=True)
class RingBufferLayout:
    data_name: str
    meta_name: str
    frame_shape: tuple[int, ...]
    frame_dtype: str
    num_slots: int


@dataclass(frozen=True)
class FrameRecord:
    frame: np.ndarray
    timestamp_ns: int
    frame_id: int


class SharedRingBuffer:
    def __init__(
        self,
        layout: RingBufferLayout,
        write_index: Synchronized,
        data_shm: SharedMemory,
        meta_shm: SharedMemory,
        *,
        owner: bool,
    ) -> None:
        self._layout = layout
        self._write_index = write_index
        self._data_shm = data_shm
        self._meta_shm = meta_shm
        self._owner = owner
        self._dtype = np.dtype(layout.frame_dtype)
        self._frames = np.ndarray(
            (layout.num_slots, *layout.frame_shape), dtype=self._dtype, buffer=self._data_shm.buf
        )
        self._meta = np.ndarray((layout.num_slots,), dtype=_META_DTYPE, buffer=self._meta_shm.buf)

    @classmethod
    def create(
        cls,
        frame_shape: tuple[int, ...],
        *,
        num_slots: int = 3,
        frame_dtype: np.dtype | type[np.generic] = np.uint8,
    ) -> tuple[Self, Synchronized]:
        dtype = np.dtype(frame_dtype)
        frame_bytes = int(np.prod(frame_shape, dtype=np.int64)) * dtype.itemsize
        data_shm = SharedMemory(create=True, size=num_slots * frame_bytes)
        meta_shm = SharedMemory(create=True, size=num_slots * _META_DTYPE.itemsize)
        layout = RingBufferLayout(
            data_name=data_shm.name,
            meta_name=meta_shm.name,
            frame_shape=frame_shape,
            frame_dtype=dtype.str,
            num_slots=num_slots,
        )
        write_index: Synchronized = mp.Value(ctypes.c_ulonglong, 0)
        buffer = cls(layout, write_index, data_shm, meta_shm, owner=True)
        buffer._meta[:] = (0, 0, 0)
        buffer._frames[:] = 0
        return buffer, write_index

    @classmethod
    def attach(cls, layout: RingBufferLayout, write_index: Synchronized) -> Self:
        data_shm = SharedMemory(name=layout.data_name)
        meta_shm = SharedMemory(name=layout.meta_name)
        return cls(layout, write_index, data_shm, meta_shm, owner=False)

    @property
    def layout(self) -> RingBufferLayout:
        return self._layout

    def is_empty(self) -> bool:
        return self._write_index.value == 0

    def write(self, frame: np.ndarray, timestamp_ns: int) -> int:
        if frame.shape != self._layout.frame_shape:
            raise ValueError(f"expected frame shape {self._layout.frame_shape}, got {frame.shape}")
        if frame.dtype != self._dtype:
            raise ValueError(f"expected frame dtype {self._dtype}, got {frame.dtype}")

        next_frame_id = int(self._write_index.value) + 1
        slot_index = (next_frame_id - 1) % self._layout.num_slots
        current_generation = int(self._meta[slot_index]["generation"])
        write_generation = (
            current_generation + 1 if current_generation % 2 == 0 else current_generation + 2
        )
        self._meta[slot_index]["generation"] = write_generation
        np.copyto(self._frames[slot_index], frame, casting="no")
        self._meta[slot_index] = (int(timestamp_ns), next_frame_id, write_generation + 1)
        self._write_index.value = next_frame_id
        return next_frame_id

    def read_latest(
        self, *, after_frame_id: int | None = None, copy: bool = False
    ) -> FrameRecord | None:
        latest_frame_id = int(self._write_index.value)
        if latest_frame_id == 0:
            return None
        if after_frame_id is not None and latest_frame_id <= after_frame_id:
            return None

        return self.read_frame(latest_frame_id, copy=copy)

    def read_frame(self, frame_id: int, *, copy: bool = False) -> FrameRecord | None:
        latest_frame_id = int(self._write_index.value)
        if frame_id <= 0 or frame_id > latest_frame_id:
            return None
        if latest_frame_id - frame_id >= self._layout.num_slots:
            return None

        slot_index = (frame_id - 1) % self._layout.num_slots
        for _ in range(3):
            generation_before = int(self._meta[slot_index]["generation"])
            if generation_before % 2 == 1:
                continue

            timestamp_ns = int(self._meta[slot_index]["timestamp_ns"])
            meta_frame_id = int(self._meta[slot_index]["frame_id"])
            frame = self._frames[slot_index].copy() if copy else self._frames[slot_index]
            generation_after = int(self._meta[slot_index]["generation"])

            if (
                generation_before == generation_after
                and generation_after % 2 == 0
                and meta_frame_id == frame_id
            ):
                return FrameRecord(frame=frame, timestamp_ns=timestamp_ns, frame_id=meta_frame_id)
        return None

    def close(self, *, unlink: bool = False) -> None:
        self._data_shm.close()
        self._meta_shm.close()
        if unlink and self._owner:
            self._data_shm.unlink()
            self._meta_shm.unlink()

    def cleanup(self) -> None:
        self.close(unlink=True)
