from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

MAGIC = b"TAFFIMU1"
VERSION = 1

SENSOR_TYPE_ACCELEROMETER = 1
SENSOR_TYPE_MAGNETIC_FIELD = 2
SENSOR_TYPE_GYROSCOPE = 4
SENSOR_TYPE_GRAVITY = 9
SENSOR_TYPE_LINEAR_ACCELERATION = 10
SENSOR_TYPE_ROTATION_VECTOR = 11
SENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED = 14
SENSOR_TYPE_GAME_ROTATION_VECTOR = 15
SENSOR_TYPE_GYROSCOPE_UNCALIBRATED = 16
SENSOR_TYPE_GEOMAGNETIC_ROTATION_VECTOR = 20
SENSOR_TYPE_POSE_6DOF = 28
SENSOR_TYPE_ACCELEROMETER_UNCALIBRATED = 35
SENSOR_TYPE_HEAD_TRACKER = 37

SENSOR_TYPE_TO_NAME = {
    SENSOR_TYPE_ACCELEROMETER: "accelerometer",
    SENSOR_TYPE_MAGNETIC_FIELD: "magnetic_field",
    SENSOR_TYPE_GYROSCOPE: "gyroscope",
    SENSOR_TYPE_GRAVITY: "gravity",
    SENSOR_TYPE_LINEAR_ACCELERATION: "linear_acceleration",
    SENSOR_TYPE_ROTATION_VECTOR: "rotation_vector",
    SENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED: "magnetic_field_uncalibrated",
    SENSOR_TYPE_GAME_ROTATION_VECTOR: "game_rotation_vector",
    SENSOR_TYPE_GYROSCOPE_UNCALIBRATED: "gyroscope_uncalibrated",
    SENSOR_TYPE_GEOMAGNETIC_ROTATION_VECTOR: "geomagnetic_rotation_vector",
    SENSOR_TYPE_POSE_6DOF: "pose_6dof",
    SENSOR_TYPE_ACCELEROMETER_UNCALIBRATED: "accelerometer_uncalibrated",
    SENSOR_TYPE_HEAD_TRACKER: "head_tracker",
}
SUPPORTED_SENSOR_TYPES = frozenset(SENSOR_TYPE_TO_NAME)
MAX_VALUES = 16

HEADER_STRUCT = struct.Struct("<8sHHQqHhHH" + "f" * MAX_VALUES)
HEADER_SIZE = HEADER_STRUCT.size


@dataclass(frozen=True, slots=True)
class ImuSample:
    version: int
    header_size: int
    sequence: int
    timestamp_ns: int
    sensor_type: int
    accuracy: int
    values: tuple[float, ...]
    values_count: int
    flags: int

    @property
    def sensor_name(self) -> str:
        return SENSOR_TYPE_TO_NAME[self.sensor_type]

    @property
    def active_values(self) -> tuple[float, ...]:
        return self.values[: self.values_count]


def parse_imu_sample(data: bytes) -> ImuSample:
    if len(data) < HEADER_SIZE:
        raise ValueError(f"IMU packet too short: {len(data)} < {HEADER_SIZE}")

    (
        magic,
        version,
        header_size,
        sequence,
        timestamp_ns,
        sensor_type,
        accuracy,
        values_count,
        flags,
        *values,
    ) = HEADER_STRUCT.unpack_from(data)

    if magic != MAGIC:
        raise ValueError(f"invalid phone IMU magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported phone IMU version: {version}")
    if header_size < HEADER_SIZE:
        raise ValueError(f"invalid phone IMU header size: {header_size}")
    if sensor_type not in SUPPORTED_SENSOR_TYPES:
        raise ValueError(f"unsupported phone IMU sensor type: {sensor_type}")
    if values_count < 1 or values_count > MAX_VALUES:
        raise ValueError(f"invalid phone IMU values_count: {values_count}")

    return ImuSample(
        version=version,
        header_size=header_size,
        sequence=sequence,
        timestamp_ns=timestamp_ns,
        sensor_type=sensor_type,
        accuracy=accuracy,
        values=tuple(values[:values_count]),
        values_count=values_count,
        flags=flags,
    )


def pack_imu_sample(sample: ImuSample) -> bytes:
    if sample.version != VERSION:
        raise ValueError(f"unsupported phone IMU version: {sample.version}")
    if sample.header_size < HEADER_SIZE:
        raise ValueError(f"invalid phone IMU header size: {sample.header_size}")
    if sample.sensor_type not in SUPPORTED_SENSOR_TYPES:
        raise ValueError(f"unsupported phone IMU sensor type: {sample.sensor_type}")
    if sample.values_count < 1 or sample.values_count > MAX_VALUES:
        raise ValueError(f"invalid phone IMU values_count: {sample.values_count}")
    if len(sample.values) < sample.values_count:
        raise ValueError(
            f"phone IMU values shorter than values_count: {len(sample.values)} < "
            f"{sample.values_count}"
        )

    values = tuple(sample.values[: sample.values_count])
    values = values + (0.0,) * (MAX_VALUES - len(values))
    fixed_header = HEADER_STRUCT.pack(
        MAGIC,
        sample.version,
        sample.header_size,
        sample.sequence,
        sample.timestamp_ns,
        sample.sensor_type,
        sample.accuracy,
        sample.values_count,
        sample.flags,
        *values,
    )
    if sample.header_size == HEADER_SIZE:
        return fixed_header
    return fixed_header + bytes(sample.header_size - HEADER_SIZE)


def read_exact(sock: socket.socket, n: int) -> bytes:
    if n < 0:
        raise ValueError("read size must be non-negative")
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed while reading phone IMU sample")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_imu_sample(sock: socket.socket) -> ImuSample:
    return parse_imu_sample(read_exact(sock, HEADER_SIZE))
