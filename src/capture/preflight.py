"""Deprecated re-export shim.

``FrameHealthMonitor`` moved to :mod:`src.shared.preflight` because it is
consumed by the inference process, not the capture process. Import from
``src.shared.preflight`` in new code.
"""

from __future__ import annotations

from src.shared.preflight import FrameHealthMonitor

__all__ = ["FrameHealthMonitor"]
