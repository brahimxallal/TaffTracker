from __future__ import annotations

import re
from pathlib import Path

import pytest


def _label_block(content: str, label: str, next_label: str) -> str:
    start = content.index(f"\n:{label}\n")
    end = content.index(f"\n:{next_label}\n", start)
    return content[start:end]


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


@pytest.mark.unit
def test_launch_batch_keeps_crlf_line_endings_for_cmd() -> None:
    data = Path("launch.bat").read_bytes()

    assert data.count(b"\n") == data.count(b"\r\n")


@pytest.mark.unit
def test_launch_batch_prefers_project_venv_over_active_virtualenv() -> None:
    content = Path("launch.bat").read_text(encoding="utf-8")

    project_venv = 'if exist "%ROOT%.venv\\Scripts\\python.exe"'
    active_venv = "else if defined VIRTUAL_ENV"

    assert project_venv in content
    assert active_venv in content
    assert content.index(project_venv) < content.index(active_venv)


@pytest.mark.unit
def test_quick_launch_uses_engine_compatibility_check() -> None:
    content = Path("launch.bat").read_text(encoding="utf-8")
    quick_track = _label_block(content, "quick_track", "qt_launch")

    assert 'call :prompt_engine_ready "%QT_TARGET%" "%QT_ENGINE%"' in quick_track
    assert 'scripts\\export_engines.py" --target %QT_TARGET%' not in quick_track


@pytest.mark.unit
def test_prompt_engine_ready_verifies_existing_engine_deserializes() -> None:
    content = Path("launch.bat").read_text(encoding="utf-8")
    prompt_engine_ready = _label_block(content, "prompt_engine_ready", "select_mode")

    assert 'call :verify_engine_compatible "%~2"' in prompt_engine_ready
    assert "TRTEngine" in content
