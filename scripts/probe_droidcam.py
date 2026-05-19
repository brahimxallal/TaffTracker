from __future__ import annotations

import argparse
import json
import platform
import socket
import struct
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

FRAME_HEADER_SIZE = 12
MAX_PACKET_BYTES = 16 * 1024 * 1024
NO_PTS = (1 << 64) - 1
STOP_OR_ERROR_PAYLOAD_LEN = (1 << 32) - 1


@dataclass
class HttpProbe:
    path: str
    status: int | None
    content_type: str | None
    body_preview: str
    error: str | None = None


@dataclass
class StreamProbe:
    video_format: str
    protocol_format: str
    width: int
    height: int
    connected: bool
    http_status: str | None
    config_payload_bytes: int | None
    config_payload_prefix_hex: str | None
    first_pts: int | None
    first_payload_bytes: int | None
    first_payload_prefix_hex: str | None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Black-box DroidCam probe for TaffTracker. It does not decompile or bypass "
            "DroidCam; it only observes HTTP/TCP endpoints exposed by your own running app."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4747)
    parser.add_argument("--timeout-s", type=float, default=1.5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--formats",
        default="jpg,avc,hevc",
        help="Comma-separated direct stream formats to probe: avc,jpg,hevc (mjpg aliases to jpg)",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def probe_http(host: str, port: int, path: str, timeout_s: float) -> HttpProbe:
    url = f"http://{host}:{port}{path}"
    request = urllib.request.Request(url, method="GET", headers={"Connection": "close"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read(1024)
            return HttpProbe(
                path=path,
                status=int(response.status),
                content_type=response.headers.get("Content-Type"),
                body_preview=_preview(body),
            )
    except urllib.error.HTTPError as exc:
        body = exc.read(1024)
        return HttpProbe(
            path=path,
            status=int(exc.code),
            content_type=exc.headers.get("Content-Type"),
            body_preview=_preview(body),
        )
    except Exception as exc:
        return HttpProbe(
            path=path,
            status=None,
            content_type=None,
            body_preview="",
            error=str(exc),
        )


def probe_stream(
    host: str,
    port: int,
    video_format: str,
    width: int,
    height: int,
    timeout_s: float,
) -> StreamProbe:
    protocol_format = _droidcam_protocol_format(video_format)
    forwarded_port = port if host.strip().lower() in {"127.0.0.1", "localhost", "::1"} else 0
    path = (
        f"/v5/video/{protocol_format}/{width}x{height}/port/{forwarded_port}/"
        f"os/{_os_name()}/obs/7.0.0/client/243/nonce/5912/"
    )
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
            sock.settimeout(timeout_s)
            request = f"GET {path}".encode("ascii")
            sock.sendall(request)
            recv_buffer = bytearray()
            first_packet = _read_packet(sock, recv_buffer)
            config_payload_bytes = None
            config_payload_prefix_hex = None
            if first_packet[0] == NO_PTS:
                config_payload_bytes = first_packet[1]
                config_payload_prefix_hex = first_packet[2].hex(" ")
                first_packet = _read_packet(sock, recv_buffer)
            return StreamProbe(
                video_format=video_format,
                protocol_format=protocol_format,
                width=width,
                height=height,
                connected=True,
                http_status="raw DroidCam packet stream",
                config_payload_bytes=config_payload_bytes,
                config_payload_prefix_hex=config_payload_prefix_hex,
                first_pts=first_packet[0],
                first_payload_bytes=first_packet[1],
                first_payload_prefix_hex=first_packet[2].hex(" "),
            )
    except Exception as exc:
        return StreamProbe(
            video_format=video_format,
            protocol_format=_safe_protocol_format(video_format),
            width=width,
            height=height,
            connected=False,
            http_status=None,
            config_payload_bytes=None,
            config_payload_prefix_hex=None,
            first_pts=None,
            first_payload_bytes=None,
            first_payload_prefix_hex=None,
            error=str(exc),
        )


def _read_packet(sock: socket.socket, recv_buffer: bytearray) -> tuple[int, int, bytes]:
    header = _read_exact_buffered(sock, recv_buffer, FRAME_HEADER_SIZE)
    pts, payload_bytes = struct.unpack(">QI", header)
    if payload_bytes == STOP_OR_ERROR_PAYLOAD_LEN:
        raise ConnectionError("DroidCam app signaled stream stop/error")
    if payload_bytes == 0 or payload_bytes > MAX_PACKET_BYTES:
        raise ConnectionError(f"invalid DroidCam packet size {payload_bytes}")
    payload = _read_exact_buffered(sock, recv_buffer, payload_bytes)
    return pts, payload_bytes, payload[:16]


def _read_exact_buffered(sock: socket.socket, recv_buffer: bytearray, size: int) -> bytes:
    while len(recv_buffer) < size:
        chunk = sock.recv(size - len(recv_buffer))
        if not chunk:
            raise ConnectionError("socket closed while reading frame")
        recv_buffer.extend(chunk)
    data = bytes(recv_buffer[:size])
    del recv_buffer[:size]
    return data


def _read_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("socket closed while reading frame")
        data.extend(chunk)
    return bytes(data)


def _droidcam_protocol_format(video_format: str) -> str:
    normalized = video_format.lower()
    if normalized in ("mjpg", "mjpeg"):
        return "jpg"
    if normalized in ("avc", "jpg", "hevc"):
        return normalized
    raise ValueError(f"unsupported DroidCam format: {video_format}")


def _safe_protocol_format(video_format: str) -> str:
    try:
        return _droidcam_protocol_format(video_format)
    except ValueError:
        return video_format


def _os_name() -> str:
    if platform.system().lower() == "windows":
        return f"win{platform.version() or '10.0.0'}"
    return platform.system().lower() or "linux"


def _preview(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")
    return " ".join(text.split())[:240]


def _print_human(result: dict[str, Any]) -> None:
    print(f"DroidCam probe {result['host']}:{result['port']} at {result['timestamp']}")
    print("\nHTTP endpoints:")
    for item in result["http"]:
        status = item["status"] if item["status"] is not None else "ERR"
        line = f"  {item['path']:<24} {status}"
        if item["error"]:
            line += f"  {item['error']}"
        elif item["body_preview"]:
            line += f"  {item['body_preview']}"
        print(line)
    print("\nDirect streams:")
    for item in result["streams"]:
        label = f"  {item['video_format']:<4} {item['width']}x{item['height']}:"
        if item["connected"]:
            config = ""
            if item["config_payload_bytes"] is not None:
                config = (
                    f" config_bytes={item['config_payload_bytes']},"
                    f" config_prefix={item['config_payload_prefix_hex']},"
                )
            print(
                f"{label} connected,{config} pts={item['first_pts']}, "
                f"bytes={item['first_payload_bytes']}, "
                f"prefix={item['first_payload_prefix_hex']}"
            )
        else:
            reason = item["error"] or item["http_status"] or "not connected"
            print(f"{label} {reason}")


def main() -> int:
    args = parse_args()
    endpoints = [
        "/",
        "/video",
        "/v1/phone/name",
        "/v1/phone/battery_info",
        "/v1/camera/camera_list",
        "/v1/camera/info",
    ]
    formats = [part.strip() for part in args.formats.split(",") if part.strip()]
    result = {
        "host": args.host,
        "port": args.port,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "http": [
            asdict(probe_http(args.host, args.port, path, args.timeout_s))
            for path in endpoints
        ],
        "streams": [
            asdict(
                probe_stream(
                    args.host,
                    args.port,
                    video_format,
                    args.width,
                    args.height,
                    args.timeout_s,
                )
            )
            for video_format in formats
        ],
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
