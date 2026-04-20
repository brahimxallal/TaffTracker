from __future__ import annotations

import socket

import pytest

from src.output.udp_comm import UDPComm


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
