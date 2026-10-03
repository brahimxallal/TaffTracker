from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.shared.logging_config import configure_process_logging


@pytest.mark.unit
def test_configure_process_logging_uses_env_log_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_file = tmp_path / "logs" / "tracker.log"
    monkeypatch.setenv("TAFFTRACKER_LOG_FILE", str(log_file))
    monkeypatch.setenv("TAFFTRACKER_LOG_LEVEL", "WARNING")

    configure_process_logging(force=True)
    try:
        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert any(
            isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_file
            for handler in root.handlers
        )
        assert log_file.parent.exists()
    finally:
        for handler in list(logging.getLogger().handlers):
            handler.close()
            logging.getLogger().removeHandler(handler)


@pytest.mark.unit
def test_configure_process_logging_ignores_non_level_logging_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TAFFTRACKER_LOG_FILE", raising=False)
    # ``logging.BASIC_FORMAT`` exists but is a format string, not a level.
    monkeypatch.setenv("TAFFTRACKER_LOG_LEVEL", "BASIC_FORMAT")

    configure_process_logging(default_level=logging.ERROR, force=True)
    try:
        assert logging.getLogger().level == logging.ERROR
    finally:
        for handler in list(logging.getLogger().handlers):
            handler.close()
            logging.getLogger().removeHandler(handler)
