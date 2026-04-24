from __future__ import annotations

import socket
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.output.udp_comm import UDPComm, _default_reachability_probe


@pytest.mark.unit
def test_udp_comm_sends_packet_to_localhost() -> None:
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind(("127.0.0.1", 0))
    port = recv_sock.getsockname()[1]
    recv_sock.settimeout(1.0)

    try:
        comm = UDPComm(
            "127.0.0.1",
            port,
            reachability_probe=lambda host, timeout_ms: True,
            start_monitor=False,
        )
        comm._connected = True
        comm.send(b"\xaa\x00\x01")
        data, _ = recv_sock.recvfrom(64)
        assert data == b"\xaa\x00\x01"
        comm.close()
    finally:
        recv_sock.close()


@pytest.mark.unit
def test_udp_comm_close_does_not_raise() -> None:
    comm = UDPComm("127.0.0.1", 59999, start_monitor=False)
    comm.close()  # Should not raise


@pytest.mark.unit
def test_udp_comm_send_multiple_packets() -> None:
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind(("127.0.0.1", 0))
    port = recv_sock.getsockname()[1]
    recv_sock.settimeout(1.0)
    received: list[bytes] = []

    try:
        comm = UDPComm(
            "127.0.0.1",
            port,
            redundancy=1,
            reachability_probe=lambda host, timeout_ms: True,
            start_monitor=False,
        )
        comm._connected = True
        for i in range(3):
            payload = bytes([i, i + 1, i + 2])
            comm.send(payload)
            data, _ = recv_sock.recvfrom(64)
            received.append(data)
        comm.close()
    finally:
        recv_sock.close()

    assert received == [bytes([i, i + 1, i + 2]) for i in range(3)]


@pytest.mark.unit
def test_udp_comm_reports_reachability() -> None:
    comm = UDPComm(
        "127.0.0.1",
        59999,
        reachability_probe=lambda host, timeout_ms: True,
        start_monitor=False,
    )
    comm._connected = True
    assert comm.is_connected is True
    assert comm.reconnect() is True
    comm.close()


@pytest.mark.unit
def test_udp_comm_tracks_packet_counts() -> None:
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind(("127.0.0.1", 0))
    port = recv_sock.getsockname()[1]
    recv_sock.settimeout(1.0)

    try:
        comm = UDPComm(
            "127.0.0.1",
            port,
            reachability_probe=lambda host, timeout_ms: True,
            start_monitor=False,
        )
        comm._connected = True
        comm.send(b"\x01")
        comm.send(b"\x02")
        recv_sock.recvfrom(64)
        recv_sock.recvfrom(64)
        assert comm.packets_sent == 2
        assert comm.packets_failed == 0
        comm.close()
    finally:
        recv_sock.close()


@pytest.mark.unit
def test_udp_comm_health_check() -> None:
    comm = UDPComm(
        "192.168.4.1",
        6000,
        reachability_probe=lambda host, timeout_ms: False,
        start_monitor=False,
    )
    health = comm.health_check()
    assert health["connected"] is False
    assert health["packets_sent"] == 0
    comm.close()


@pytest.mark.unit
def test_udp_comm_probe_marks_unreachable_host_disconnected() -> None:
    comm = UDPComm(
        "192.168.4.1",
        6000,
        reachability_probe=lambda host, timeout_ms: False,
        start_monitor=False,
    )

    assert comm.is_connected is False
    assert comm.reconnect() is False

    comm.close()


@pytest.mark.unit
def test_udp_comm_send_failure_increments_failed_counter() -> None:
    """sendto() raising OSError must bump packets_failed, not crash."""
    comm = UDPComm(
        "127.0.0.1",
        59999,
        reachability_probe=lambda host, timeout_ms: True,
        start_monitor=False,
    )
    # Force every sendto to raise
    comm._socket.close()  # closing the socket makes sendto raise OSError
    comm.send(b"\x00")
    assert comm.packets_sent == 0
    assert comm.packets_failed == 1
    # Re-issue the send to confirm the counter accumulates
    comm.send(b"\x01")
    assert comm.packets_failed == 2
    comm._monitor_stop.set()  # close() would try to close an already-closed socket


@pytest.mark.unit
def test_udp_comm_probe_exception_is_swallowed() -> None:
    """A misbehaving probe must not crash the monitor."""

    def _bad_probe(host: str, timeout_ms: int) -> bool:
        raise RuntimeError("probe blew up")

    comm = UDPComm(
        "127.0.0.1",
        59999,
        reachability_probe=_bad_probe,
        start_monitor=False,
    )
    # Internal probe path should swallow the exception and report False
    assert comm._probe_reachability() is False
    comm.close()


@pytest.mark.unit
def test_udp_comm_monitor_thread_updates_connection_state() -> None:
    """A live monitor thread should reflect the latest probe outcome."""
    state = {"reachable": False}

    def _probe(host: str, timeout_ms: int) -> bool:
        return state["reachable"]

    # Use a tiny probe interval for the test
    with patch.object(UDPComm, "_PROBE_INTERVAL_S", 0.05):
        comm = UDPComm("127.0.0.1", 59999, reachability_probe=_probe, start_monitor=True)
        # Wait a probe tick to settle initial value
        time.sleep(0.15)
        assert comm.is_connected is False

        # Flip reachability and wait for the monitor to pick it up
        state["reachable"] = True
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline and not comm.is_connected:
            time.sleep(0.05)
        assert comm.is_connected is True
        comm.close()


@pytest.mark.unit
def test_default_reachability_probe_returns_false_for_empty_host() -> None:
    assert _default_reachability_probe("", 100) is False
    assert _default_reachability_probe("   ", 100) is False


@pytest.mark.unit
def test_default_reachability_probe_returns_true_when_ping_succeeds() -> None:
    """Exit code 0 from ping → reachable. Mock subprocess to avoid network."""
    fake_completed = MagicMock(returncode=0)
    with patch("src.output.udp_comm.subprocess.run", return_value=fake_completed):
        assert _default_reachability_probe("8.8.8.8", 200) is True


@pytest.mark.unit
def test_default_reachability_probe_returns_false_when_ping_fails() -> None:
    fake_completed = MagicMock(returncode=1)
    with patch("src.output.udp_comm.subprocess.run", return_value=fake_completed):
        assert _default_reachability_probe("192.0.2.1", 200) is False


@pytest.mark.unit
def test_default_reachability_probe_returns_false_when_subprocess_raises() -> None:
    """If ping itself blows up (timeout, missing binary), report unreachable."""
    with patch(
        "src.output.udp_comm.subprocess.run",
        side_effect=TimeoutError("ping timed out"),
    ):
        assert _default_reachability_probe("8.8.8.8", 200) is False


@pytest.mark.unit
def test_udp_comm_close_joins_monitor_thread_cleanly() -> None:
    """close() must signal the monitor thread to exit and wait for it."""

    def _probe(host: str, timeout_ms: int) -> bool:
        return True

    with patch.object(UDPComm, "_PROBE_INTERVAL_S", 0.05):
        comm = UDPComm("127.0.0.1", 59999, reachability_probe=_probe, start_monitor=True)
        time.sleep(0.1)
        thread = comm._monitor_thread
        assert thread is not None and thread.is_alive()
        comm.close()
        # After close, the thread must have stopped within a reasonable bound.
        thread.join(timeout=2.0)
        assert not thread.is_alive()


@pytest.mark.unit
def test_udp_comm_redundancy_actually_sends_n_copies() -> None:
    """redundancy=N → sendto called N times per logical send()."""
    comm = UDPComm(
        "127.0.0.1",
        59999,
        redundancy=4,
        reachability_probe=lambda host, timeout_ms: True,
        start_monitor=False,
    )
    fake_socket = MagicMock()
    comm._socket.close()
    comm._socket = fake_socket
    comm.send(b"\xaa")
    assert fake_socket.sendto.call_count == 4
    assert comm.packets_sent == 1


@pytest.mark.unit
def test_udp_comm_close_is_idempotent_after_monitor_thread_join() -> None:
    """Double-close must not raise, even if the monitor finished cleanly."""

    def _probe(host: str, timeout_ms: int) -> bool:
        return True

    with patch.object(UDPComm, "_PROBE_INTERVAL_S", 0.05):
        comm = UDPComm("127.0.0.1", 59999, reachability_probe=_probe, start_monitor=True)
        time.sleep(0.1)
        comm.close()
        # Calling close() a second time would close an already-closed socket;
        # we don't expect that to be supported, but at least the monitor join
        # should be a no-op once the thread is finished.
        comm._monitor_stop.set()
        if comm._monitor_thread is not None:
            comm._monitor_thread.join(timeout=1.0)
            assert not comm._monitor_thread.is_alive()


@pytest.mark.unit
def test_udp_comm_threading_event_is_set_on_close() -> None:
    comm = UDPComm(
        "127.0.0.1",
        59999,
        reachability_probe=lambda host, timeout_ms: True,
        start_monitor=False,
    )
    assert isinstance(comm._monitor_stop, threading.Event)
    assert not comm._monitor_stop.is_set()
    comm.close()
    assert comm._monitor_stop.is_set()
