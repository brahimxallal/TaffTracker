"""Factory for the output process's transport.

Extracted from ``src/output/process.py`` to keep the process class focused
on the hot path. The factory maps ``CommConfig.channel`` to the concrete
``SerialComm`` / ``UDPComm`` / ``AutoCommTransport`` implementation, with
best-effort error handling so a missing device falls back to visualization
only.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import CommConfig
from src.output.auto_comm import AutoCommTransport
from src.output.serial_comm import SerialComm
from src.output.udp_comm import UDPComm

LOGGER = logging.getLogger("output")


def create_sender(comm_config: CommConfig) -> Any | None:
    """Return the configured transport, or ``None`` when hardware/network fails.

    ``None`` is a valid result: the output process runs visualization-only
    when comms can't be opened.
    """
    channel = comm_config.channel
    if channel == "auto":
        try:
            return AutoCommTransport(comm_config)
        except Exception as exc:
            LOGGER.warning(
                "Could not initialize auto comm transport (%s). Visualization only.", exc
            )
            return None
    try:
        if channel == "serial":
            LOGGER.info("Opening serial output on %s", comm_config.serial_port)
            return SerialComm(comm_config.serial_port, comm_config.baud_rate)
        LOGGER.info(
            "Opening UDP output on %s:%s",
            comm_config.udp_host,
            comm_config.udp_port,
        )
        return UDPComm(
            comm_config.udp_host,
            comm_config.udp_port,
            redundancy=comm_config.udp_redundancy,
        )
    except Exception as exc:
        LOGGER.warning(
            "Could not open %s comm (%s). Running visualization only.",
            channel,
            exc,
        )
        return None
