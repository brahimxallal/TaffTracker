"""Single-slot shared-memory buffer for display frames.

Replaces ``mp.Queue`` (pickle overhead) with zero-copy shared memory for the
OutputProcess → main-process display path.  A generation counter provides
tear-free reads without locks.

Writer (OutputProcess): call ``write(frame)`` each time a new display frame
is ready.

Reader (main process): call ``read()`` to get the latest frame, or *None*
if no new frame has been written since the last read.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Self

import numpy as np


@dataclass(frozen=True)
class DisplayBufferLayout:
    data_name: str
    meta_name: str
    frame_shape: tuple[int, ...]
    frame_dtype: str


# Meta layout: [generation (uint64), frame_counter (uint64)]
_META_DTYPE = np.dtype([("generation", np.uint64), ("frame_counter", np.uint64)])


class SharedDisplayBuffer:
    """Single-slot shared-memory display frame buffer.

    Uses an even/odd generation counter protocol identical to
    ``SharedRingBuffer`` — writer bumps to odd before memcpy, then to
    even after.  Reader retries if the generation changed mid-read.
    """

    def __init__(
        self,
        layout: DisplayBufferLayout,
        data_shm: SharedMemory,
        meta_shm: SharedMemory,
        *,
        owner: bool,
    ) -> None:
        self._layout = layout
        self._data_shm = data_shm
        self._meta_shm = meta_shm
        self._owner = owner
        self._dtype = np.dtype(layout.frame_dtype)
        self._frame = np.ndarray(layout.frame_shape, dtype=self._dtype, buffer=self._data_shm.buf)
        self._meta = np.ndarray((1,), dtype=_META_DTYPE, buffer=self._meta_shm.buf)
        self._last_read_counter: int = 0

    @classmethod
    def create(cls, frame_shape: tuple[int, ...], *, frame_dtype: type[np.generic] = np.uint8) -> Self:
        dtype = np.dtype(frame_dtype)
        data_bytes = int(np.prod(frame_shape, dtype=np.int64)) * dtype.itemsize
        data_shm = SharedMemory(create=True, size=data_bytes)
        meta_shm = SharedMemory(create=True, size=_META_DTYPE.itemsize)
        layout = DisplayBufferLayout(
            data_name=data_shm.name,
            meta_name=meta_shm.name,
            frame_shape=frame_shape,
            frame_dtype=dtype.str,
        )
        buf = cls(layout, data_shm, meta_shm, owner=True)
        buf._meta[0] = (0, 0)
        buf._frame[:] = 0
        return buf

    @classmethod
    def attach(cls, layout: DisplayBufferLayout) -> Self:
        data_shm = SharedMemory(name=layout.data_name)
        meta_shm = SharedMemory(name=layout.meta_name)
        return cls(layout, data_shm, meta_shm, owner=False)

    @property
    def layout(self) -> DisplayBufferLayout:
        return self._layout

    def write(self, frame: np.ndarray) -> None:
        """Write a display frame (non-blocking, overwrites previous)."""
        if frame.shape != self._layout.frame_shape:
            raise ValueError(f"expected {self._layout.frame_shape}, got {frame.shape}")

        current_gen = int(self._meta[0]["generation"])
        write_gen = current_gen + 1 if current_gen % 2 == 0 else current_gen + 2
        self._meta[0]["generation"] = write_gen          # odd → write in progress
        np.copyto(self._frame, frame, casting="no")
        counter = int(self._meta[0]["frame_counter"]) + 1
        self._meta[0]["frame_counter"] = counter
        self._meta[0]["generation"] = write_gen + 1       # even → write complete

    def read(self) -> np.ndarray | None:
        """Read the latest display frame, or *None* if nothing new."""
        counter = int(self._meta[0]["frame_counter"])
        if counter <= self._last_read_counter:
            return None

        for _ in range(3):
            gen_before = int(self._meta[0]["generation"])
            if gen_before % 2 == 1:
                continue  # write in progress
            frame = self._frame.copy()
            gen_after = int(self._meta[0]["generation"])
            if gen_before == gen_after:
                self._last_read_counter = counter
                return frame
        return None  # torn read — skip frame

    def close(self, *, unlink: bool = False) -> None:
        self._data_shm.close()
        self._meta_shm.close()
        if unlink and self._owner:
            self._data_shm.unlink()
            self._meta_shm.unlink()

    def cleanup(self) -> None:
        self.close(unlink=True)
