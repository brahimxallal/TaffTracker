from __future__ import annotations

import cv2
import numpy as np

from src.shared.types import TrackingMessage

# ── Colours (BGR) ────────────────────────────────────────────────────────────
_WHITE = (240, 240, 240)
_GREEN = (50, 255, 50)
_YELLOW = (0, 255, 255)
_RED = (50, 50, 255)
_CYAN = (255, 255, 0)

_STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "measurement": _GREEN,
    "prediction": _YELLOW,
    "lost": _RED,
    "center": _RED,
    "open_loop": _CYAN,
}

_HANDOFF_ALPHA = 0.25  # EMA blend factor per frame (higher = faster snap)


class FrameSmoother:
    """Instance-isolated reticle position smoother (replaces module-global state)."""

    __slots__ = ("prev_track_id", "smooth_x", "smooth_y")

    def __init__(self) -> None:
        self.prev_track_id: int | None = None
        self.smooth_x: float | None = None
        self.smooth_y: float | None = None

    def update(
        self, track_id: int | None, tx: float, ty: float, alpha: float = _HANDOFF_ALPHA
    ) -> tuple[float, float]:
        if track_id != self.prev_track_id:
            if self.smooth_x is not None and track_id is not None:
                pass  # handoff: blend from old position
            else:
                self.smooth_x, self.smooth_y = tx, ty
            self.prev_track_id = track_id
        if self.smooth_x is None:
            self.smooth_x, self.smooth_y = tx, ty
        self.smooth_x = self.smooth_x + alpha * (tx - self.smooth_x)
        self.smooth_y = self.smooth_y + alpha * (ty - self.smooth_y)
        return self.smooth_x, self.smooth_y

    def clear(self) -> None:
        self.smooth_x = self.smooth_y = None
        self.prev_track_id = None


# Module-level default instance for backward compatibility
_default_smoother = FrameSmoother()


def draw_overlay(
    frame: np.ndarray,
    message: TrackingMessage,
    manual_mode: bool = False,
    laser_enabled: bool = True,
    smoother: FrameSmoother | None = None,
) -> np.ndarray:
    """Minimal HUD: crosshair, target reticle, FPS bar, mode indicator.

    Draws directly on the input frame (no copy) to eliminate 1.2MB/frame allocation.
    """
    sm = smoother if smoother is not None else _default_smoother

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Center crosshair
    _draw_crosshair(frame, cx, cy)

    # Target reticle (colored brackets) with smooth handoff
    state_color = _STATE_COLORS.get(message.state_source, _GREEN)
    if message.filtered_pixel is not None:
        tx, ty = message.filtered_pixel[0], message.filtered_pixel[1]
        sx, sy = sm.update(message.track_id, tx, ty)
        _draw_reticle(frame, (int(sx), int(sy)), state_color, message.confidence)

        # Raw model centroid — cyan dot showing the unfiltered pose point
        if message.raw_pixel is not None:
            rx, ry = int(message.raw_pixel[0]), int(message.raw_pixel[1])
            cv2.circle(frame, (rx, ry), 5, _CYAN, -1, cv2.LINE_AA)
    else:
        sm.clear()

    # Secondary target markers
    for cx_f, cy_f, _tid, _score in message.other_targets:
        _draw_secondary_marker(frame, int(cx_f), int(cy_f))

    # Target priority panel (top-right) — shows all tracked targets
    _draw_target_panel(frame, message)

    # Minimal FPS / timing info (top-left)
    _draw_fps_bar(frame, message)

    # Mode and output state indicators
    _draw_mode_indicator(frame, manual_mode, laser_enabled)

    return frame


def _draw_crosshair(canvas: np.ndarray, cx: int, cy: int) -> None:
    color = (120, 120, 120)
    cv2.ellipse(canvas, (cx, cy), (50, 50), 0, 0, 60, color, 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy), (50, 50), 0, 120, 180, color, 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy), (50, 50), 0, 240, 300, color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx - 10, cy), (cx - 2, cy), color, 1)
    cv2.line(canvas, (cx + 2, cy), (cx + 10, cy), color, 1)
    cv2.line(canvas, (cx, cy - 10), (cx, cy - 2), color, 1)
    cv2.line(canvas, (cx, cy + 2), (cx, cy + 10), color, 1)


def _draw_reticle(
    canvas: np.ndarray, pt: tuple[int, int], color: tuple[int, int, int], confidence: float
) -> None:
    x, y = pt
    s = 22 + int((1.0 - confidence) * 10)
    cv2.line(canvas, (x - s, y - s), (x - s + 8, y - s), color, 2)
    cv2.line(canvas, (x - s, y - s), (x - s, y - s + 8), color, 2)
    cv2.line(canvas, (x + s, y - s), (x + s - 8, y - s), color, 2)
    cv2.line(canvas, (x + s, y - s), (x + s, y - s + 8), color, 2)
    cv2.line(canvas, (x - s, y + s), (x - s + 8, y + s), color, 2)
    cv2.line(canvas, (x - s, y + s), (x - s, y + s - 8), color, 2)
    cv2.line(canvas, (x + s, y + s), (x + s - 8, y + s), color, 2)
    cv2.line(canvas, (x + s, y + s), (x + s, y + s - 8), color, 2)


def _draw_fps_bar(canvas: np.ndarray, msg: TrackingMessage) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.40, 1
    x, y = 12, 20
    gap = 16
    fps_color = _GREEN if msg.fps > 45 else _YELLOW
    cv2.putText(canvas, f"FPS {msg.fps:.0f}", (x, y), font, scale, fps_color, thick, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"INF {msg.inference_ms:.1f}ms",
        (x, y + gap),
        font,
        scale,
        _WHITE,
        thick,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"TRK {msg.tracking_ms:.1f}ms",
        (x, y + gap * 2),
        font,
        scale,
        _WHITE,
        thick,
        cv2.LINE_AA,
    )


def _draw_mode_indicator(canvas: np.ndarray, manual_mode: bool, laser_enabled: bool) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    if manual_mode:
        cv2.putText(canvas, "MANUAL", (12, 72), font, 0.45, _YELLOW, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "AUTO", (12, 72), font, 0.45, _GREEN, 1, cv2.LINE_AA)
    if not laser_enabled:
        cv2.putText(canvas, "LASER OFF", (12, 88), font, 0.45, _RED, 1, cv2.LINE_AA)


def _draw_secondary_marker(canvas: np.ndarray, x: int, y: int) -> None:
    """Small diamond marker for non-primary tracked targets."""
    s = 8
    pts = np.array([(x, y - s), (x + s, y), (x, y + s), (x - s, y)], dtype=np.int32)
    cv2.polylines(canvas, [pts], True, _CYAN, 1, cv2.LINE_AA)


def _draw_target_panel(canvas: np.ndarray, msg: TrackingMessage) -> None:
    """Compact target list panel (top-right). Shows locked + secondary targets."""
    if msg.track_id is None and not msg.other_targets:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.35, 1
    h, w = canvas.shape[:2]
    line_h = 14
    entries: list[tuple[str, tuple[int, int, int]]] = []
    if msg.track_id is not None:
        bar = int(msg.confidence * 10)
        entries.append((f">{msg.track_id:3d} {'|' * bar:<10s} {msg.confidence:.0%}", _GREEN))
    for _, _, tid, score in msg.other_targets:
        bar = int(score * 10)
        entries.append((f" {tid:3d} {'|' * bar:<10s} {score:.0%}", _CYAN))
    if not entries:
        return
    pw = 150
    ph = len(entries) * line_h + 8
    px = w - pw - 8
    py = 8
    sub = canvas[py : py + ph, px : px + pw]
    cv2.addWeighted(sub, 0.4, sub, 0, 0, sub)
    for i, (text, color) in enumerate(entries):
        cv2.putText(
            canvas, text, (px + 4, py + 12 + i * line_h), font, scale, color, thick, cv2.LINE_AA
        )
