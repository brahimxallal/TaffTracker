from __future__ import annotations

import ctypes
import logging
import sys
from types import TracebackType
from typing import Protocol

LOGGER = logging.getLogger(__name__)


class _WinMM(Protocol):
    def timeBeginPeriod(self, period_ms: int) -> int: ...

    def timeEndPeriod(self, period_ms: int) -> int: ...


class WindowsTimerResolution:
    """Temporarily request a higher Windows scheduler timer resolution."""

    def __init__(
        self,
        period_ms: int = 1,
        *,
        enabled: bool = True,
        winmm: _WinMM | None = None,
        is_windows: bool | None = None,
    ) -> None:
        self.period_ms = period_ms
        self.enabled = enabled
        self._winmm = winmm
        self._is_windows = sys.platform == "win32" if is_windows is None else is_windows
        self._active = False
        self._enter_depth = 0

    def __enter__(self) -> WindowsTimerResolution:
        if not self.enabled or not self._is_windows:
            return self
        if self._active:
            self._enter_depth += 1
            return self

        try:
            winmm = self._winmm if self._winmm is not None else ctypes.WinDLL("winmm")
            result = winmm.timeBeginPeriod(self.period_ms)
        except (AttributeError, OSError) as exc:
            LOGGER.warning("timeBeginPeriod(%d) unavailable: %s", self.period_ms, exc)
            return self

        if result != 0:
            LOGGER.warning("timeBeginPeriod(%d) failed with code %s", self.period_ms, result)
            return self

        self._winmm = winmm
        self._active = True
        self._enter_depth = 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._active and self._winmm is not None:
            if self._enter_depth > 1:
                self._enter_depth -= 1
                return
            try:
                result = self._winmm.timeEndPeriod(self.period_ms)
            except (AttributeError, OSError) as err:
                LOGGER.warning("timeEndPeriod(%d) unavailable: %s", self.period_ms, err)
            else:
                if result != 0:
                    LOGGER.warning("timeEndPeriod(%d) failed with code %s", self.period_ms, result)
            finally:
                self._active = False
                self._enter_depth = 0
