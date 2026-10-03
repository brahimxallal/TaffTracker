from __future__ import annotations

import logging
import os
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(processName)s %(levelname)s %(message)s"
_ENV_LOG_FILE = "TAFFTRACKER_LOG_FILE"
_ENV_LOG_LEVEL = "TAFFTRACKER_LOG_LEVEL"


def configure_logging_environment(log_file: Path | None, log_level: str) -> None:
    """Publish logging settings for spawned child processes."""
    if log_file is None:
        os.environ.pop(_ENV_LOG_FILE, None)
    else:
        os.environ[_ENV_LOG_FILE] = str(log_file)
    os.environ[_ENV_LOG_LEVEL] = log_level


def configure_process_logging(
    *,
    default_level: int = logging.INFO,
    log_file: Path | str | None = None,
    level_name: str | None = None,
    force: bool = False,
) -> None:
    """Configure stream logging, plus optional shared file logging.

    Child processes read ``TAFFTRACKER_LOG_FILE``/``TAFFTRACKER_LOG_LEVEL`` so
    ``--log-file`` captures capture/inference/output diagnostics, not only the
    main process.
    """
    configured_level_name = level_name or os.environ.get(_ENV_LOG_LEVEL)
    level = getattr(logging, configured_level_name or "", default_level)
    if not isinstance(level, int):  # e.g. TAFFTRACKER_LOG_LEVEL=BASIC_FORMAT
        level = default_level

    configured_log_file = log_file
    if configured_log_file is None:
        env_log_file = os.environ.get(_ENV_LOG_FILE)
        configured_log_file = Path(env_log_file) if env_log_file else None
    else:
        configured_log_file = Path(configured_log_file)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if configured_log_file is not None:
        configured_log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(configured_log_file), encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        handlers=handlers,
        force=force,
    )
