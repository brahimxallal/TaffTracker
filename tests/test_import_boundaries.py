from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_capture_preflight_shim_removed() -> None:
    assert not (ROOT / "src" / "capture" / "preflight.py").exists()


def test_src_does_not_import_capture_preflight() -> None:
    forbidden = ("src.capture.preflight", "capture.preflight")

    offenders: list[Path] = []
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            offenders.append(path.relative_to(ROOT))

    assert offenders == []
