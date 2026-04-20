from __future__ import annotations

from unittest.mock import MagicMock, patch

import serial as pyserial
import pytest

from src.output.serial_comm import SerialComm


@pytest.mark.unit
def test_serial_comm_sends_bytes() -> None:
    with patch("src.output.serial_comm.serial.Serial") as mock_serial_cls:
        mock_serial = MagicMock()
        mock_serial_cls.return_value = mock_serial
        comm = SerialComm(port="COM3", baud_rate=921600)

        comm.send(b"\xaa\x01\x02")

        mock_serial.write.assert_called_once_with(b"\xaa\x01\x02")


@pytest.mark.unit
def test_serial_comm_close_calls_serial_close() -> None:
    with patch("src.output.serial_comm.serial.Serial") as mock_serial_cls:
        mock_serial = MagicMock()
        mock_serial_cls.return_value = mock_serial
        comm = SerialComm(port="COM3", baud_rate=115200)

        comm.close()

        mock_serial.close.assert_called_once()


@pytest.mark.unit
def test_serial_comm_opens_with_correct_params() -> None:
    with patch("src.output.serial_comm.serial.Serial") as mock_serial_cls:
        SerialComm(port="COM5", baud_rate=460800)

        mock_serial_cls.assert_called_once_with(
            port="COM5", baudrate=460800, timeout=0, write_timeout=0
        )


# --- Reconnection tests ---


@pytest.mark.unit
def test_serial_is_connected_after_open() -> None:
    with patch("src.output.serial_comm.serial.Serial"):
        comm = SerialComm(port="COM3")
        assert comm.is_connected is True


@pytest.mark.unit
def test_serial_disconnects_on_send_failure() -> None:
    with patch("src.output.serial_comm.serial.Serial") as mock_cls:
        mock_serial = MagicMock()
        mock_serial.write.side_effect = pyserial.SerialException("device disconnected")
        mock_cls.return_value = mock_serial
        comm = SerialComm(port="COM3")

        comm.send(b"\x00")

        assert comm.is_connected is False
        assert comm.packets_failed == 1
        assert comm.packets_sent == 0


@pytest.mark.unit
def test_serial_send_skips_when_disconnected() -> None:
    with patch("src.output.serial_comm.serial.Serial") as mock_cls:
        mock_serial = MagicMock()
        mock_cls.return_value = mock_serial
        comm = SerialComm(port="COM3")
        comm.close()  # force disconnect

        comm.send(b"\x00")

        mock_serial.write.assert_not_called()  # closed before send
        assert comm.packets_failed == 1


@pytest.mark.unit
def test_serial_reconnect_succeeds() -> None:
    with patch("src.output.serial_comm.serial.Serial") as mock_cls:
        mock_serial = MagicMock()
        mock_serial.write.side_effect = pyserial.SerialException("fail")
        mock_cls.return_value = mock_serial
        comm = SerialComm(port="COM3")
        comm.send(b"\x00")  # triggers disconnect
        assert comm.is_connected is False

        # Reset mock for reconnection
        mock_serial2 = MagicMock()
        mock_cls.return_value = mock_serial2
        comm._last_reconnect_attempt = 0.0  # bypass backoff for test
        result = comm.reconnect()

        assert result is True
        assert comm.is_connected is True


@pytest.mark.unit
def test_serial_reconnect_respects_backoff() -> None:
    with patch("src.output.serial_comm.serial.Serial") as mock_cls:
        mock_cls.side_effect = pyserial.SerialException("no device")
        comm = SerialComm.__new__(SerialComm)
        comm._port = "COM3"
        comm._baud_rate = 921600
        comm._serial = None
        comm._connected = False
        comm._packets_sent = 0
        comm._packets_failed = 0
        comm._backoff = 0.5
        comm._last_reconnect_attempt = 0.0

        # First reconnect attempt — fails, increases backoff
        comm.reconnect()
        assert comm.is_connected is False

        # Immediate retry — should be skipped due to backoff
        result = comm.reconnect()
        assert result is False


@pytest.mark.unit
def test_serial_health_check_reports_state() -> None:
    with patch("src.output.serial_comm.serial.Serial"):
        comm = SerialComm(port="COM3")
        comm.send(b"\x01")
        comm.send(b"\x02")

        health = comm.health_check()

        assert health["connected"] is True
        assert health["port"] == "COM3"
        assert health["packets_sent"] == 2
        assert health["packets_failed"] == 0


@pytest.mark.unit
def test_serial_constructor_handles_open_failure() -> None:
    with patch("src.output.serial_comm.serial.Serial") as mock_cls:
        mock_cls.side_effect = pyserial.SerialException("no device")
        comm = SerialComm(port="COM99")

        assert comm.is_connected is False
        comm.send(b"\x00")  # should not raise
        assert comm.packets_failed == 1
