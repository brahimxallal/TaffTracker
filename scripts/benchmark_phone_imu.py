from __future__ import annotations

import argparse
import logging
import time

from src.sensors.phone_imu import PhoneImuReceiver, PhoneImuRuntimeConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark TaffCam phone IMU stream over ADB TCP.")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--sensor-port", type=int, default=27185)
    parser.add_argument("--rate-hz", type=int, default=200)
    parser.add_argument("--adb-path", default="adb")
    parser.add_argument("--adb-serial", default=None)
    parser.add_argument("--no-adb-reverse", action="store_true")
    parser.add_argument("--no-start-service", action="store_true")
    parser.add_argument("--allow-remote-clients", action="store_true")
    parser.add_argument("--read-timeout-s", type=float, default=0.03)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    receiver = PhoneImuReceiver(
        PhoneImuRuntimeConfig(
            sensor_host=args.host,
            sensor_port=args.sensor_port,
            requested_rate_hz=args.rate_hz,
            read_timeout_s=args.read_timeout_s,
            adb_reverse=not args.no_adb_reverse,
            adb_path=args.adb_path,
            adb_serial=args.adb_serial,
            allow_remote_clients=args.allow_remote_clients,
            start_android_service=not args.no_start_service,
        )
    )
    start_s = time.perf_counter()
    deadline_s = start_s + max(args.duration_s, 0.1)
    total_samples = 0
    per_sensor: dict[str, int] = {}
    first_timestamp_ns: dict[str, int] = {}
    last_timestamp_ns: dict[str, int] = {}

    try:
        while time.perf_counter() < deadline_s:
            snapshot = receiver.read_latest()
            if snapshot is None:
                time.sleep(0.001)
                continue
            total_samples += snapshot.samples_read
            for name, sample in snapshot.latest_by_sensor.items():
                per_sensor[name] = max(per_sensor.get(name, 0), sample.sequence)
                first_timestamp_ns.setdefault(name, sample.timestamp_ns)
                last_timestamp_ns[name] = sample.timestamp_ns
    finally:
        receiver.release()

    elapsed_s = max(time.perf_counter() - start_s, 1e-9)
    print(f"elapsed_s={elapsed_s:.3f}")
    print(f"total_samples={total_samples}")
    print(f"host_sample_rate={total_samples / elapsed_s:.1f}")
    for name in sorted(per_sensor):
        span_s = max(last_timestamp_ns[name] - first_timestamp_ns[name], 0) / 1_000_000_000.0
        print(f"sensor={name} last_sequence={per_sensor[name]} android_span_s={span_s:.3f}")


if __name__ == "__main__":
    main()
