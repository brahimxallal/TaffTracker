"""Hotkey helpers for the main display loop.

Extracted from ``src/main.py`` — keeping the dispatch rules in one place
makes it easier to add keys without bloating the orchestrator.
"""

from __future__ import annotations

QUIT_KEY_CODE = 27  # ESC


def is_quit_hotkey(key_code: int, no_quit: bool) -> bool:
    """Return True when the user pressed ESC and quit is allowed."""
    return not no_quit and key_code == QUIT_KEY_CODE
