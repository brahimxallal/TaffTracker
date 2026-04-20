from __future__ import annotations

import re
from pathlib import Path

import pytest


@pytest.mark.unit
def test_launch_batch_internal_call_labels_exist() -> None:
    launch_path = Path("launch.bat")
    content = launch_path.read_text(encoding="utf-8")

    called_labels = set(re.findall(r"(?im)^\s*call\s+:([a-z0-9_]+)\b", content))
    defined_labels = set(re.findall(r"(?im)^\s*:([a-z0-9_]+)\b", content))

    missing = sorted(called_labels - defined_labels)

    assert missing == []


@pytest.mark.unit
def test_launch_batch_does_not_use_broken_check_engine_call() -> None:
    content = Path("launch.bat").read_text(encoding="utf-8")

    assert "call :check_engine" not in content
