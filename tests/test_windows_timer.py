from __future__ import annotations

import pytest

from src.shared.windows_timer import WindowsTimerResolution


class _FakeWinMM:
    def __init__(self, *, begin_result: int = 0, end_result: int = 0) -> None:
        self.calls: list[tuple[str, int]] = []
        self.begin_result = begin_result
        self.end_result = end_result

    def timeBeginPeriod(self, period_ms: int) -> int:
        self.calls.append(("timeBeginPeriod", period_ms))
        return self.begin_result

    def timeEndPeriod(self, period_ms: int) -> int:
        self.calls.append(("timeEndPeriod", period_ms))
        return self.end_result


@pytest.mark.unit
def test_windows_timer_resolution_calls_begin_and_end_period() -> None:
    winmm = _FakeWinMM()

    with WindowsTimerResolution(period_ms=1, winmm=winmm, is_windows=True):
        assert winmm.calls == [("timeBeginPeriod", 1)]

    assert winmm.calls == [("timeBeginPeriod", 1), ("timeEndPeriod", 1)]


@pytest.mark.unit
def test_windows_timer_resolution_ends_period_when_context_raises() -> None:
    winmm = _FakeWinMM()

    with pytest.raises(RuntimeError, match="boom"):
        with WindowsTimerResolution(period_ms=1, winmm=winmm, is_windows=True):
            raise RuntimeError("boom")

    assert winmm.calls == [("timeBeginPeriod", 1), ("timeEndPeriod", 1)]


@pytest.mark.unit
def test_windows_timer_resolution_noops_when_disabled() -> None:
    winmm = _FakeWinMM()

    with WindowsTimerResolution(period_ms=1, enabled=False, winmm=winmm, is_windows=True):
        pass

    assert winmm.calls == []


@pytest.mark.unit
def test_windows_timer_resolution_noops_when_not_windows() -> None:
    winmm = _FakeWinMM()

    with WindowsTimerResolution(period_ms=1, winmm=winmm, is_windows=False):
        pass

    assert winmm.calls == []


@pytest.mark.unit
def test_windows_timer_resolution_does_not_end_failed_begin(caplog: pytest.LogCaptureFixture) -> None:
    winmm = _FakeWinMM(begin_result=97)

    with caplog.at_level("WARNING"):
        with WindowsTimerResolution(period_ms=1, winmm=winmm, is_windows=True):
            pass

    assert winmm.calls == [("timeBeginPeriod", 1)]
    assert "timeBeginPeriod(1) failed" in caplog.text


@pytest.mark.unit
def test_windows_timer_resolution_is_idempotent_for_nested_enter_on_same_instance() -> None:
    winmm = _FakeWinMM()
    timer = WindowsTimerResolution(period_ms=1, winmm=winmm, is_windows=True)

    timer.__enter__()
    timer.__enter__()
    timer.__exit__(None, None, None)
    assert winmm.calls == [("timeBeginPeriod", 1)]

    timer.__exit__(None, None, None)

    assert winmm.calls == [("timeBeginPeriod", 1), ("timeEndPeriod", 1)]


@pytest.mark.unit
def test_windows_timer_resolution_releases_only_after_outermost_exit() -> None:
    winmm = _FakeWinMM()
    timer = WindowsTimerResolution(period_ms=1, winmm=winmm, is_windows=True)

    with timer:
        with timer:
            assert winmm.calls == [("timeBeginPeriod", 1)]
        assert winmm.calls == [("timeBeginPeriod", 1)]

    assert winmm.calls == [("timeBeginPeriod", 1), ("timeEndPeriod", 1)]


@pytest.mark.unit
def test_windows_timer_resolution_warns_when_end_fails(caplog: pytest.LogCaptureFixture) -> None:
    winmm = _FakeWinMM(end_result=31)

    with caplog.at_level("WARNING"):
        with WindowsTimerResolution(period_ms=1, winmm=winmm, is_windows=True):
            pass

    assert winmm.calls == [("timeBeginPeriod", 1), ("timeEndPeriod", 1)]
    assert "timeEndPeriod(1) failed" in caplog.text
