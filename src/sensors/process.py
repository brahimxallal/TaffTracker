from __future__ import annotations

import logging
import multiprocessing as mp
import time
from traceback import format_exc

from src.config import PhoneImuConfig
from src.sensors.phone_imu import PhoneImuReceiver, PhoneImuRuntimeConfig
from src.shared.logging_config import configure_process_logging
from src.shared.types import ProcessErrorReport

LOGGER = logging.getLogger("phone_imu")
_LOG_INTERVAL_S = 2.0


def build_phone_imu_runtime_config(config: PhoneImuConfig) -> PhoneImuRuntimeConfig:
    return PhoneImuRuntimeConfig(
        sensor_host=config.bind_host,
        sensor_port=config.sensor_port,
        requested_rate_hz=config.requested_rate_hz,
        read_timeout_s=config.read_timeout_s,
        adb_reverse=config.adb_reverse_enabled,
        adb_path=config.adb_path,
        adb_serial=config.adb_serial,
        adb_reverse_timeout_s=config.adb_reverse_timeout_s,
        allow_remote_clients=config.allow_remote_clients,
        start_android_service=config.start_service_on_open,
        force_stop_app=config.force_stop_app_on_open,
        app_start_delay_s=config.app_start_delay_s,
    )


class PhoneImuProcess(mp.Process):
    """Sidecar process that keeps the phone motion-sensor stream alive."""

    def __init__(
        self,
        phone_imu_config: PhoneImuConfig,
        shutdown_event: mp.synchronize.Event,
        error_queue: mp.Queue,
    ) -> None:
        super().__init__(name="PhoneImuProcess")
        self._phone_imu_config = phone_imu_config
        self._shutdown_event = shutdown_event
        self._error_queue = error_queue

    def run(self) -> None:
        configure_process_logging()
        if not self._phone_imu_config.enabled:
            LOGGER.info("Phone IMU disabled.")
            return

        receiver: PhoneImuReceiver | None = None
        try:
            receiver = PhoneImuReceiver(build_phone_imu_runtime_config(self._phone_imu_config))
            last_log_s = time.perf_counter()
            last_samples = 0
            total_samples = 0
            latest_sensors: tuple[str, ...] = ()

            while not self._shutdown_event.is_set():
                snapshot = receiver.read_latest()
                if snapshot is None:
                    self._shutdown_event.wait(0.002)
                    continue

                total_samples += snapshot.samples_read
                latest_sensors = tuple(sorted(snapshot.latest_by_sensor))
                now_s = time.perf_counter()
                if now_s - last_log_s >= _LOG_INTERVAL_S:
                    interval = max(now_s - last_log_s, 1e-9)
                    rate = (total_samples - last_samples) / interval
                    LOGGER.info(
                        "Phone IMU: %.1f samples/s sensors=%s",
                        rate,
                        ",".join(latest_sensors) if latest_sensors else "none",
                    )
                    last_log_s = now_s
                    last_samples = total_samples
        except BaseException as exc:
            LOGGER.exception("Phone IMU process failed")
            self._report_error(exc)
            self._shutdown_event.set()
        finally:
            if receiver is not None:
                receiver.release()

    def _report_error(self, exc: BaseException) -> None:
        report = ProcessErrorReport(
            process_name=self.name,
            summary=str(exc) or exc.__class__.__name__,
            traceback_text=format_exc(),
            timestamp_ns=time.perf_counter_ns(),
            severity="error",
        )
        try:
            self._error_queue.put_nowait(report)
        except Exception:
            LOGGER.error("Failed to publish phone IMU error report")
