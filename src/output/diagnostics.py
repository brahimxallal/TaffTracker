"""Frame-overlay diagnostics rendering for OutputProcess.

Split out of ``src/output/process.py`` so the process orchestrator stays
focused on packet encoding + transport dispatch. These helpers are pure
rendering and have no multiprocessing or transport concerns.
"""

from __future__ import annotations

from typing import Any

from src.output.serial_comm import SerialComm
from src.output.udp_comm import UDPComm


def draw_diagnostics(
    frame: Any,
    lock_frames: int,
    total_frames: int,
    latency_sum: float,
    latency_max: float,
    display_drops: int,
    display_total: int,
    transport_label: str,
    transport_color: tuple[int, int, int],
) -> None:
    """Minimal diagnostic panel (bottom-left) — lock %, latency, drops, link."""
    import cv2

    h, _w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.35, 1
    x = 12
    y_base = h - 60
    gap = 14
    green = (50, 255, 50)
    yellow = (0, 255, 255)
    red = (50, 50, 255)

    lock_pct = (lock_frames / max(1, total_frames)) * 100.0
    avg_lat = latency_sum / max(1, total_frames)
    drop_pct = (display_drops / max(1, display_total)) * 100.0

    color_lock = green if lock_pct > 80 else yellow if lock_pct > 50 else red
    color_lat = green if avg_lat < 20 else yellow if avg_lat < 40 else red
    color_drop = green if drop_pct < 2 else yellow if drop_pct < 10 else red

    cv2.putText(
        frame, f"LOCK {lock_pct:.0f}%", (x, y_base), font, scale, color_lock, thick, cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"LAT {avg_lat:.1f}/{latency_max:.0f}ms",
        (x, y_base + gap),
        font,
        scale,
        color_lat,
        thick,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"DROP {display_drops}/{max(1, display_total)} ({drop_pct:.1f}%)",
        (x, y_base + gap * 2),
        font,
        scale,
        color_drop,
        thick,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        transport_label,
        (x, y_base + gap * 3),
        font,
        scale,
        transport_color,
        thick,
        cv2.LINE_AA,
    )


def get_transport_status(sender: Any) -> tuple[str, tuple[int, int, int]]:
    """Return ('LINK ...', BGR-color) for the current transport state."""
    green = (50, 255, 50)
    red = (50, 50, 255)

    if sender is None:
        return "LINK OFFLINE", red

    connected = bool(getattr(sender, "is_connected", False))
    if not connected:
        return "LINK OFFLINE", red

    channel = getattr(sender, "active_channel", None)
    if channel not in {"serial", "udp"}:
        if isinstance(sender, SerialComm):
            channel = "serial"
        elif isinstance(sender, UDPComm):
            channel = "udp"

    if channel == "serial":
        return "LINK SERIAL", green
    if channel == "udp":
        return "LINK WIFI", green
    return "LINK OFFLINE", red
