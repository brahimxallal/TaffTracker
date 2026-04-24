from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from dataclasses import replace
from math import degrees
from multiprocessing.sharedctypes import Synchronized
from queue import Empty
from traceback import format_exc

from src.config import (
    CameraConfig,
    CommConfig,
    GimbalConfig,
    Mode,
    RuntimeFlags,
    ServoControlConfig,
    TrackingConfig,
)
from src.output.auto_controller import (
    AutoControllerConfig,
    AutoControllerState,
    compute_auto_command,
)
from src.output.diagnostics import draw_diagnostics, get_transport_status
from src.output.manual_control import (
    ManualVelocityTracker,
    boost_manual_velocity,
)
from src.output.manual_loop import ManualLoopConfig, run_manual_tick_loop
from src.output.sender_factory import create_sender
from src.output.telemetry import write_metrics_summary
from src.output.visualizer import FrameSmoother, draw_overlay
from src.shared.display_buffer import DisplayBufferLayout, SharedDisplayBuffer
from src.shared.profiler import StageProfiler
from src.shared.protocol import (
    FLAG_LASER_ON,
    FLAG_RELAY_ON,
    QUALITY_FLAG_MANUAL,
    build_state_flags,
    encode_packet_v2,
)
from src.shared.ring_buffer import RingBufferLayout, SharedRingBuffer
from src.shared.types import ProcessErrorReport, TrackingMessage

LOGGER = logging.getLogger("output")


class OutputProcess(mp.Process):
    def __init__(
        self,
        layout: RingBufferLayout,
        write_index,
        result_queue: mp.Queue,
        shutdown_event: mp.synchronize.Event,
        error_queue: mp.Queue,
        mode: Mode,
        camera_config: CameraConfig,
        comm_config: CommConfig,
        tracking_config: TrackingConfig,
        flags: RuntimeFlags | None = None,
        gimbal_config: GimbalConfig | None = None,
        servo_control_config: ServoControlConfig | None = None,
        display_queue: mp.Queue | None = None,
        display_buffer_layout: DisplayBufferLayout | None = None,
        relay_flag: Synchronized | None = None,
        laser_enabled: Synchronized | None = None,
        manual_mode: Synchronized | None = None,
        manual_pan: Synchronized | None = None,
        manual_tilt: Synchronized | None = None,
        laser_boresight_pan: Synchronized | None = None,
        laser_boresight_tilt: Synchronized | None = None,
    ) -> None:
        super().__init__(name="OutputProcess")
        self._layout = layout
        self._write_index = write_index
        self._result_queue = result_queue
        self._shutdown_event = shutdown_event
        self._error_queue = error_queue
        self._display_queue = display_queue
        self._display_buffer_layout = display_buffer_layout
        self._relay_flag = relay_flag
        self._laser_enabled = laser_enabled
        self._manual_mode = manual_mode
        self._manual_pan = manual_pan
        self._manual_tilt = manual_tilt
        self._laser_boresight_pan = laser_boresight_pan
        self._laser_boresight_tilt = laser_boresight_tilt
        self._mode = mode
        self._camera_config = camera_config
        self._comm_config = comm_config
        self._tracking_config = tracking_config
        self._flags = flags or RuntimeFlags()
        self._gimbal_config = gimbal_config or GimbalConfig()
        self._servo_control_config = servo_control_config or ServoControlConfig()
        self._frames_processed = 0
        self._last_log_time = time.perf_counter()
        self._center_pixel = (camera_config.width / 2.0, camera_config.height / 2.0)
        # Auto-mode controller state lives in a dedicated dataclass so
        # the integrator + derivative history can be unit-tested as a
        # plain step function (see src/output/auto_controller.py).
        self._auto_controller_state = AutoControllerState()
        self._auto_controller_config = AutoControllerConfig.from_configs(
            self._gimbal_config, self._servo_control_config
        )
        # Manual-mode velocity history is owned by ManualVelocityTracker.
        self._manual_velocity_tracker = ManualVelocityTracker()

    # --- Backward-compat properties for existing tests that reach into
    # the per-frame controller state. New code should use
    # self._auto_controller_state directly.
    @property
    def _pi_integral_pan(self) -> float:
        return self._auto_controller_state.pi_integral_pan

    @_pi_integral_pan.setter
    def _pi_integral_pan(self, value: float) -> None:
        self._auto_controller_state.pi_integral_pan = value

    @property
    def _pi_integral_tilt(self) -> float:
        return self._auto_controller_state.pi_integral_tilt

    @_pi_integral_tilt.setter
    def _pi_integral_tilt(self, value: float) -> None:
        self._auto_controller_state.pi_integral_tilt = value

    def _read_laser_boresight_pan(self) -> float:
        if self._laser_boresight_pan is None:
            return 0.0
        return float(self._laser_boresight_pan.value)

    def _read_laser_boresight_tilt(self) -> float:
        if self._laser_boresight_tilt is None:
            return 0.0
        return float(self._laser_boresight_tilt.value)

    def _is_laser_enabled(self) -> bool:
        return self._laser_enabled is None or bool(self._laser_enabled.value)

    def _boost_manual_velocity(self, velocity_dps: float) -> float:
        # Thin wrapper around src.output.manual_control.boost_manual_velocity;
        # keeps existing call sites unchanged.
        return boost_manual_velocity(
            velocity_dps,
            self._servo_control_config.manual_response_velocity_floor_dps,
        )

    def _manual_tick_loop(
        self,
        sender,
        send_lock: threading.Lock,
        shutdown_event: mp.synchronize.Event,
    ) -> None:
        """100 Hz manual packet emitter, decoupled from inference frame rate.

        Thin delegate over :func:`src.output.manual_loop.run_manual_tick_loop`.
        ``send_lock`` protects the shared ``self._manual_sequence`` counter so
        the same counter remains consistent across start/restart cycles of
        this method.
        """

        def _get_next_sequence() -> int:
            with send_lock:
                seq = self._manual_sequence
                self._manual_sequence = (seq + 1) & 0xFFFF
            return seq

        run_manual_tick_loop(
            sender=sender,
            shutdown_event=shutdown_event,
            manual_mode=self._manual_mode,
            manual_pan=self._manual_pan,
            manual_tilt=self._manual_tilt,
            relay_flag=self._relay_flag,
            laser_enabled=self._laser_enabled,
            loop_config=ManualLoopConfig.from_configs(
                self._gimbal_config, self._servo_control_config
            ),
            get_next_sequence=_get_next_sequence,
        )

    def run(self) -> None:
        import sys

        def _excepthook(etype, value, tb):
            import traceback

            LOGGER.error(
                "Uncaught exception in OutputProcess: %s",
                "".join(traceback.format_exception(etype, value, tb)),
            )

        sys.excepthook = _excepthook
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(processName)s %(levelname)s %(message)s",
        )
        sender = (
            self._create_sender() if self._mode == "camera" and self._comm_config.enabled else None
        )
        ring_buffer = SharedRingBuffer.attach(self._layout, self._write_index)
        display_buffer: SharedDisplayBuffer | None = None
        if self._display_buffer_layout is not None:
            display_buffer = SharedDisplayBuffer.attach(self._display_buffer_layout)
        use_display = not self._flags.headless and (
            display_buffer is not None or self._display_queue is not None
        )
        profiler = StageProfiler(window_size=256, enabled=self._flags.profile)
        _frame_smoother = FrameSmoother()
        run_start = time.perf_counter()

        sequence = 0
        last_valid_message: TrackingMessage | None = None

        # Manual tick thread: emits packets at 100 Hz independent of inference
        self._manual_sequence = 0
        send_lock = threading.Lock()
        manual_thread = threading.Thread(
            target=self._manual_tick_loop,
            args=(sender, send_lock, self._shutdown_event),
            name="ManualTick",
            daemon=True,
        )
        manual_thread.start()
        _display_drops = 0
        _display_total = 0
        _display_drop_window_count = 0
        _display_drop_window_total = 0
        _display_drop_window_start = time.perf_counter()
        # Display throttle: render every Nth frame (2 = 30fps at 60fps camera rate)
        _DISPLAY_RENDER_DIVISOR = 2
        _display_frame_counter = -1  # starts at -1 so first frame (→0) always renders
        # Diagnostic HUD accumulators
        _diag_lock_frames = 0
        _diag_total_frames = 0
        _diag_latency_sum = 0.0
        _diag_latency_max = 0.0

        try:
            while not self._shutdown_event.is_set():
                try:
                    message = self._result_queue.get(timeout=0.05)
                except Empty:
                    continue

                if message is None:
                    break

                outbound = self._apply_fail_safe(message, last_valid_message)
                if outbound.target_acquired:
                    last_valid_message = outbound

                # Diagnostic accumulators
                _diag_total_frames += 1
                if outbound.target_acquired:
                    _diag_lock_frames += 1
                _diag_latency_sum += outbound.total_latency_ms
                _diag_latency_max = max(_diag_latency_max, outbound.total_latency_ms)

                if sender is not None:
                    is_manual_now = self._manual_mode is not None and bool(self._manual_mode.value)
                    if not is_manual_now:
                        if not sender.is_connected:
                            sender.reconnect()
                        send_start_ns = time.perf_counter_ns()
                        sender.send(self._encode_packet(outbound, sequence))
                        profiler.add_sample("packet_send", time.perf_counter_ns() - send_start_ns)
                        sequence = (sequence + 1) & 0xFFFF

                if use_display:
                    _display_frame_counter += 1
                    # Throttle: only render + enqueue every Nth frame
                    if _display_frame_counter % _DISPLAY_RENDER_DIVISOR == 0:
                        record = ring_buffer.read_frame(outbound.frame_id, copy=True)
                        if record is None:
                            record = ring_buffer.read_latest(copy=True)
                        if record is not None:
                            is_manual = self._manual_mode is not None and bool(
                                self._manual_mode.value
                            )
                            frame = draw_overlay(
                                record.frame,
                                outbound,
                                manual_mode=is_manual,
                                laser_enabled=self._is_laser_enabled(),
                                smoother=_frame_smoother,
                            )
                            # Diagnostic HUD (bottom-left)
                            transport_label, transport_color = get_transport_status(sender)
                            draw_diagnostics(
                                frame,
                                _diag_lock_frames,
                                _diag_total_frames,
                                _diag_latency_sum,
                                _diag_latency_max,
                                _display_drops,
                                _display_total,
                                transport_label,
                                transport_color,
                            )
                            if display_buffer is not None:
                                display_buffer.write(frame)
                            elif self._display_queue is not None:
                                try:
                                    self._display_queue.put_nowait(frame)
                                except Exception:
                                    try:
                                        self._display_queue.get_nowait()
                                    except Exception:
                                        pass
                                    try:
                                        self._display_queue.put_nowait(frame)
                                    except Exception:
                                        pass
                                    _display_drops += 1
                                    _display_drop_window_count += 1
                            _display_drop_window_total += 1
                            _display_total += 1

                            now = time.perf_counter()
                            if now - _display_drop_window_start >= 1.0:
                                if _display_drop_window_count > 0:
                                    drop_pct = (
                                        _display_drop_window_count
                                        / max(1, _display_drop_window_total)
                                    ) * 100.0
                                    LOGGER.warning(
                                        "Display backpressure: dropped %d/%d frames in the last %.1fs (%.1f%%)",
                                        _display_drop_window_count,
                                        _display_drop_window_total,
                                        now - _display_drop_window_start,
                                        drop_pct,
                                    )
                                _display_drop_window_count = 0
                                _display_drop_window_total = 0
                                _display_drop_window_start = now
                # Performance monitoring
                self._frames_processed += 1
                if self._frames_processed % 300 == 0:
                    current_time = time.perf_counter()
                    elapsed = current_time - self._last_log_time
                    fps = self._frames_processed / elapsed if elapsed > 0 else 0
                    drop_pct = (_display_drops / max(1, _display_total)) * 100.0
                    LOGGER.info(
                        "Output FPS: %.1f | display drops: %d/%d (%.1f%%)",
                        fps,
                        _display_drops,
                        _display_total,
                        drop_pct,
                    )
                    self._frames_processed = 0
                    self._last_log_time = current_time
                    _display_drops = 0
                    _display_total = 0
                    _diag_latency_max = 0.0
        except BaseException as exc:
            LOGGER.exception("Output process failed")
            self._report_error(exc)
            self._shutdown_event.set()
        finally:
            self._write_metrics_summary(profiler, sender, run_start, _display_drops, _display_total)
            if sender is not None:
                sender.close()
            if display_buffer is not None:
                display_buffer.close()
            ring_buffer.close()

    def _write_metrics_summary(
        self,
        profiler: StageProfiler,
        sender,
        run_start: float,
        display_drops: int,
        display_total: int,
    ) -> None:
        # Thin wrapper around src.output.telemetry.write_metrics_summary.
        # Kept as an instance method so existing call sites stay unchanged.
        write_metrics_summary(
            profiler,
            sender,
            run_start,
            display_drops,
            display_total,
        )

    def _reset_manual_command_history(self) -> None:
        self._manual_velocity_tracker.reset()

    def _record_auto_command(
        self,
        pan_deg: float,
        tilt_deg: float,
        pan_sign: float,
        tilt_sign: float,
    ) -> None:
        """Mirror the latest auto command into the shared manual values.

        Ensures a smooth handoff when the user toggles manual mode on —
        the gimbal resumes from where auto tracking left it.
        """
        if self._manual_pan is not None:
            self._manual_pan.value = pan_deg * pan_sign
        if self._manual_tilt is not None:
            self._manual_tilt.value = tilt_deg * tilt_sign

    def _compute_manual_velocity_dps(
        self,
        pan_deg: float,
        tilt_deg: float,
        timestamp_ns: int,
    ) -> tuple[float, float]:
        return self._manual_velocity_tracker.compute_velocity_dps(pan_deg, tilt_deg, timestamp_ns)

    def _create_sender(self):
        # Thin wrapper around src.output.sender_factory.create_sender.
        return create_sender(self._comm_config)

    def _apply_fail_safe(
        self, message: TrackingMessage, last_valid_message: TrackingMessage | None
    ) -> TrackingMessage:
        if message.target_acquired:
            return message
        if last_valid_message is None:
            return self._center_message(message)

        hold_elapsed_s = (message.timestamp_ns - last_valid_message.timestamp_ns) / 1_000_000_000.0
        hold_time = (
            message.hold_time_s
            if message.hold_time_s is not None
            else self._tracking_config.hold_time_s
        )
        if hold_elapsed_s <= hold_time and message.filtered_angles is not None:
            return message
        return self._center_message(message)

    def _center_message(self, message: TrackingMessage) -> TrackingMessage:
        return replace(
            message,
            state_source="center",
            track_id=None,
            confidence=0.0,
            raw_pixel=None,
            filtered_pixel=self._center_pixel,
            raw_angles=None,
            filtered_angles=(0.0, 0.0),
            servo_angles=(0.0, 0.0),
            servo_angular_velocity=(0.0, 0.0),
            filtered_velocity=None,
            angular_velocity=None,
        )

    def _encode_packet(self, message: TrackingMessage, sequence: int) -> bytes:
        pan_sign = -1.0 if self._gimbal_config.invert_pan else 1.0
        tilt_sign = -1.0 if self._gimbal_config.invert_tilt else 1.0
        pan_lim = self._gimbal_config.pan_limit_deg
        tilt_lim = self._gimbal_config.tilt_limit_deg
        is_manual = self._manual_mode is not None and bool(self._manual_mode.value)
        laser_enabled = self._is_laser_enabled()

        # --- Manual mode: bypass all controllers, send absolute angles ---
        if is_manual:
            manual_pan_deg = self._manual_pan.value if self._manual_pan is not None else 0.0
            manual_tilt_deg = self._manual_tilt.value if self._manual_tilt is not None else 0.0
            manual_pan_deg = max(-pan_lim, min(pan_lim, manual_pan_deg))
            manual_tilt_deg = max(-tilt_lim, min(tilt_lim, manual_tilt_deg))
            if self._manual_pan is not None:
                self._manual_pan.value = manual_pan_deg
            if self._manual_tilt is not None:
                self._manual_tilt.value = manual_tilt_deg

            pan_deg = manual_pan_deg * pan_sign
            tilt_deg = manual_tilt_deg * tilt_sign
            pan_vel_dps, tilt_vel_dps = self._compute_manual_velocity_dps(
                pan_deg,
                tilt_deg,
                message.timestamp_ns,
            )
            pan_vel_dps = self._boost_manual_velocity(pan_vel_dps)
            tilt_vel_dps = self._boost_manual_velocity(tilt_vel_dps)
            self._auto_controller_state.reset()
            confidence_value = 0.0
            vel_mag_dps = (pan_vel_dps * pan_vel_dps + tilt_vel_dps * tilt_vel_dps) ** 0.5
            state = build_state_flags(
                state_source="measurement",
                target_acquired=False,
                confidence=0.0,
                velocity_magnitude_dps=vel_mag_dps,
                is_occlusion_recovery=False,
            )
            quality = QUALITY_FLAG_MANUAL
            pan_vel_cdps = int(round(pan_vel_dps * 100.0))
            tilt_vel_cdps = int(round(tilt_vel_dps * 100.0))
        elif not is_manual:
            self._reset_manual_command_history()
            pan_deg, tilt_deg = compute_auto_command(
                message=message,
                state=self._auto_controller_state,
                config=self._auto_controller_config,
            )

        # Laser boresight offset: physical angular offset between the laser
        # emitter and the camera lens. Applied to auto-tracking commands only —
        # manual mode honours the user's explicit angle, not a tracking aim.
        if not is_manual:
            pan_deg += self._read_laser_boresight_pan() * pan_sign
            tilt_deg += self._read_laser_boresight_tilt() * tilt_sign

        pan_deg = max(-pan_lim, min(pan_lim, pan_deg))
        tilt_deg = max(-tilt_lim, min(tilt_lim, tilt_deg))
        if not is_manual:
            self._record_auto_command(pan_deg, tilt_deg, pan_sign, tilt_sign)

        pan_centideg = int(round(pan_deg * 100.0))
        tilt_centideg = int(round(tilt_deg * 100.0))
        confidence = int(
            max(0.0, min(1.0, confidence_value if is_manual else message.confidence)) * 255.0
        )

        if not is_manual:
            # Angular velocity → centideg/sec (with sign inversion) — use servo velocity (no EMA)
            ang_vel = message.servo_angular_velocity or message.angular_velocity or (0.0, 0.0)
            pan_vel_cdps = int(round(degrees(ang_vel[0]) * 100.0 * pan_sign))
            tilt_vel_cdps = int(round(degrees(ang_vel[1]) * 100.0 * tilt_sign))

            vel_mag_dps = (degrees(ang_vel[0]) ** 2 + degrees(ang_vel[1]) ** 2) ** 0.5
            state = build_state_flags(
                state_source=message.state_source,
                target_acquired=message.target_acquired,
                confidence=message.confidence,
                velocity_magnitude_dps=vel_mag_dps,
                is_occlusion_recovery=message.is_occlusion_recovery,
            )
            quality = confidence
        if laser_enabled and (is_manual or message.target_acquired):
            state |= FLAG_LASER_ON
        if self._relay_flag is not None and self._relay_flag.value:
            state |= FLAG_RELAY_ON
        latency = int(min(255, max(0, round(message.total_latency_ms))))

        return encode_packet_v2(
            sequence=sequence,
            timestamp_ms=(message.timestamp_ns // 1_000_000) & 0xFFFFFFFF,
            pan=pan_centideg,
            tilt=tilt_centideg,
            pan_vel=pan_vel_cdps,
            tilt_vel=tilt_vel_cdps,
            confidence=confidence,
            state=state,
            quality=quality,
            latency=latency,
        )

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
            LOGGER.error("Failed to publish output error report")
