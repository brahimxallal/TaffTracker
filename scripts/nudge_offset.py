"""Apply a relative nudge to the ESP32 servo offsets without the full calibrator.

Usage::

    python scripts/nudge_offset.py --pan -5
    python scripts/nudge_offset.py --pan 0 --tilt 2.5
    python scripts/nudge_offset.py --port COM4 --pan -5

Reads the current offsets from the firmware, adds the requested delta,
pushes the new values back to NVS, then mirrors them into
``calibration_data/servo_limits.json``. If the firmware doesn't ack the
write, nothing is changed on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import serial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shared.protocol import (  # noqa: E402
    CAL_CMD_GET_OFFSETS,
    CAL_CMD_SET_OFFSETS,
    CAL_PACKET_SIZE,
    HEADER_CAL,
    decode_cal_response,
    encode_cal_get_offsets,
    encode_cal_set_offsets,
)

JSON_PATH = Path(__file__).resolve().parent.parent / "calibration_data" / "servo_limits.json"


def _read_response(ser: serial.Serial, expected_command: int, timeout_s: float = 1.0) -> bytes:
    """Hunt for an HEADER_CAL byte in the serial stream, then read the rest
    of the calibration packet exactly. The firmware also sends telemetry
    on this port, so we have to skip non-cal bytes."""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        b = ser.read(1)
        if not b:
            continue
        if b[0] == HEADER_CAL:
            rest = ser.read(CAL_PACKET_SIZE - 1)
            if len(rest) != CAL_PACKET_SIZE - 1:
                continue
            # ARIA: Require CRC-valid packets for the expected command to avoid stale ACKs.
            packet = b + rest
            decoded = decode_cal_response(packet)
            if decoded is not None and decoded.command == expected_command:
                return packet
    return b""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument(
        "--pan", type=float, default=0.0, help="Delta degrees to add to current pan offset"
    )
    parser.add_argument(
        "--tilt", type=float, default=0.0, help="Delta degrees to add to current tilt offset"
    )
    parser.add_argument(
        "--set-pan", type=float, default=None, help="Set absolute pan offset (overrides --pan)"
    )
    parser.add_argument(
        "--set-tilt", type=float, default=None, help="Set absolute tilt offset (overrides --tilt)"
    )
    args = parser.parse_args()

    print(f"Opening {args.port} @ {args.baud}...")
    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    try:
        ser.reset_input_buffer()
        ser.write(encode_cal_get_offsets())
        raw = _read_response(ser, CAL_CMD_GET_OFFSETS)
        current = decode_cal_response(raw)
        if current is None:
            print(f"  ERROR: invalid response from firmware ({len(raw)} bytes)")
            return 1
        cur_pan = current.pan_offset_deg
        cur_tilt = current.tilt_offset_deg
        new_pan = args.set_pan if args.set_pan is not None else cur_pan + args.pan
        new_tilt = args.set_tilt if args.set_tilt is not None else cur_tilt + args.tilt
        print(f"  Current: pan={cur_pan:+.3f} deg  tilt={cur_tilt:+.3f} deg")
        print(f"  New:     pan={new_pan:+.3f} deg  tilt={new_tilt:+.3f} deg")

        ser.reset_input_buffer()
        ser.write(encode_cal_set_offsets(new_pan, new_tilt))
        raw = _read_response(ser, CAL_CMD_SET_OFFSETS, timeout_s=1.0)
        confirmed = decode_cal_response(raw)
        if confirmed is None:
            print(f"  ERROR: firmware did not ack the write ({len(raw)} bytes returned)")
            return 1
        if (
            abs(confirmed.pan_offset_deg - new_pan) > 0.02
            or abs(confirmed.tilt_offset_deg - new_tilt) > 0.02
        ):
            print(
                f"  ERROR: firmware reported pan={confirmed.pan_offset_deg:+.3f} "
                f"tilt={confirmed.tilt_offset_deg:+.3f} (expected pan={new_pan:+.3f} tilt={new_tilt:+.3f})"
            )
            return 1
        print(
            f"  OK: firmware confirmed pan={confirmed.pan_offset_deg:+.3f} "
            f"tilt={confirmed.tilt_offset_deg:+.3f}"
        )
    finally:
        ser.close()

    if JSON_PATH.exists():
        data = json.loads(JSON_PATH.read_text())
        data["center_pan_offset_deg"] = round(new_pan, 3)
        data["center_tilt_offset_deg"] = round(new_tilt, 3)
        JSON_PATH.write_text(json.dumps(data, indent=2) + "\n")
        print(f"  Wrote {JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
