"""OpenCV overlay renderers used by the main display loop.

Extracted from ``src/main.py`` so the orchestrator can stay focused on
process lifecycle. Pure rendering — no multiprocessing, no hardware.
"""

from __future__ import annotations

from typing import Any


def draw_help_overlay(frame: Any) -> None:
    """Draw a semi-transparent hotkey help panel on the frame."""
    import cv2

    h, w = frame.shape[:2]
    overlay = frame.copy()
    pw, ph = 260, 260
    px, py = (w - pw) // 2, (h - ph) // 2
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (200, 200, 200), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        "HOTKEYS",
        "",
        "M       Toggle auto/manual",
        "ZQSD    Manual fine (80 d/s)",
        "Arrows  Manual coarse (200 d/s)",
        "Tab     Cycle target",
        "H       Toggle this help",
        "L       Relock target",
        "P       Toggle laser output",
        "O       Relay pulse",
        "K       Laser boresight cal",
        "ESC     Quit tracker",
        "",
        "--- Auto mode ---",
        "All tracking params are",
        "auto-tuned at runtime.",
    ]
    for i, line in enumerate(lines):
        color = (255, 255, 0) if i == 0 else (200, 200, 200)
        scale = 0.5 if i == 0 else 0.38
        cv2.putText(frame, line, (px + 15, py + 22 + i * 16), font, scale, color, 1, cv2.LINE_AA)


def draw_laser_cal_hud(frame: Any, text: str) -> None:
    """Render the LASER CAL HUD as a bright yellow banner along the top edge."""
    import cv2

    _h, w = frame.shape[:2]
    bar_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(
        frame,
        text,
        (10, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
