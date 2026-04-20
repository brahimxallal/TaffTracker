"""Process-lifecycle helpers used by the main orchestrator.

Extracted from ``src/main.py`` so error draining, dead-process detection,
graceful-stop logic, and runtime cleanup live in one testable place. The
orchestrator keeps thin underscore-aliased wrappers so existing tests that
import the private names from ``src.main`` continue to work.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from queue import Empty

from src.shared.display_buffer import SharedDisplayBuffer
from src.shared.ring_buffer import SharedRingBuffer
from src.shared.types import ProcessErrorReport

LOGGER = logging.getLogger("main")


def drain_error_reports(error_queue: mp.Queue) -> list[ProcessErrorReport]:
    """Drain every report currently queued by child processes."""
    reports: list[ProcessErrorReport] = []
    while True:
        try:
            reports.append(error_queue.get_nowait())
        except Empty:
            return reports


def log_error_reports(reports: list[ProcessErrorReport]) -> None:
    """Emit one ERROR log line per process-failure report."""
    for report in reports:
        LOGGER.error(
            "%s failed: %s\n%s",
            report.process_name,
            report.summary,
            report.traceback_text,
        )


def find_unexpected_dead_processes(processes: list[mp.Process]) -> list[mp.Process]:
    """Return processes that have exited with a non-zero exit code."""
    return [
        process
        for process in processes
        if not process.is_alive() and process.exitcode not in (None, 0)
    ]


def all_processes_stopped(processes: list[mp.Process]) -> bool:
    """True when every tracked process has exited."""
    return all(not process.is_alive() for process in processes)


def check_runtime_failures(
    processes: list[mp.Process],
    error_queue: mp.Queue,
) -> tuple[list[ProcessErrorReport], list[mp.Process]]:
    """Return (error_reports, dead_processes) seen since the last check.

    If any process has died, drain the queue a second time to capture
    reports that landed between the first drain and the liveness probe.
    """
    reports = drain_error_reports(error_queue)
    dead_processes = find_unexpected_dead_processes(processes)
    if dead_processes:
        reports.extend(drain_error_reports(error_queue))
    return reports, dead_processes


def process_was_started(process: mp.Process) -> bool:
    """True when ``process.start()`` has been called at least once."""
    return (
        getattr(process, "pid", None) is not None or getattr(process, "exitcode", None) is not None
    )


def stop_processes(
    processes: list[mp.Process],
    join_timeout: float = 5.0,
    terminate_timeout: float = 2.0,
) -> list[str]:
    """Join each process; force-terminate any that overrun ``join_timeout``.

    Returns the names of processes that required SIGTERM.
    """
    forced_terminations: list[str] = []
    for process in processes:
        if not process_was_started(process):
            continue
        process.join(timeout=join_timeout)
        if process.is_alive():
            LOGGER.warning(
                "Process %s did not terminate gracefully, forcing termination", process.name
            )
            process.terminate()
            process.join(timeout=terminate_timeout)
            forced_terminations.append(process.name)
    return forced_terminations


def cleanup_runtime_resources(
    ring_buffer: SharedRingBuffer,
    pid_lock_path: Path = Path("logs/.tracker.pid"),
    *,
    display_buffer: SharedDisplayBuffer | None = None,
) -> None:
    """Tear down OpenCV windows, shared buffers, and the PID lockfile.

    Every step is best-effort — the caller is shutting down, so we never
    re-raise. Missing lockfile is not an error.
    """
    if display_buffer is not None:
        import cv2

        cv2.destroyAllWindows()
    if display_buffer is not None:
        try:
            display_buffer.cleanup()
        except Exception as exc:
            LOGGER.warning("Error during display buffer cleanup: %s", exc)
    try:
        ring_buffer.cleanup()
    except Exception as exc:
        LOGGER.warning("Error during ring buffer cleanup: %s", exc)
    pid_lock_path.unlink(missing_ok=True)
