from __future__ import annotations

import logging
from dataclasses import replace
from math import degrees
from time import perf_counter_ns

import numpy as np

from src.config import TrackingConfig
from src.inference.postprocess import (
    compute_stabilized_centroid,
)
from src.inference.stages.centroid import CentroidStage
from src.inference.stages.servo import ServoStage
from src.inference.stages.tracker import TrackerStage
from src.shared.pose_schema import PoseSchema
from src.shared.types import Detection, TrackingMessage
from src.tracking.adaptive import AdaptiveController

LOGGER = logging.getLogger("inference.pipeline")


class TrackingPipeline:
    """Orchestrates the full per-frame tracking pipeline.

    Composes TrackerStage, CentroidStage, ServoStage, and AdaptiveController
    into a single ``process_frame()`` call.
    """

    def __init__(
        self,
        *,
        tracker_stage: TrackerStage,
        centroid_stage: CentroidStage,
        servo_stage: ServoStage,
        adaptive: AdaptiveController,
        tracking_config: TrackingConfig,
        pose_schema: PoseSchema,
        ema_pixel,
        servo_ema_pixel,
    ) -> None:
        self.tracker_stage = tracker_stage
        self.centroid_stage = centroid_stage
        self.servo_stage = servo_stage
        self.adaptive = adaptive
        self._tracking_config = tracking_config
        self._pose_schema = pose_schema
        self._ema_pixel = ema_pixel
        self._servo_ema_pixel = servo_ema_pixel
        self._measurement_update_count = 0
        self._measurement_gated_count = 0

    @property
    def measurement_update_count(self) -> int:
        return self._measurement_update_count

    @property
    def measurement_gated_count(self) -> int:
        return self._measurement_gated_count

    def process_frame(
        self,
        *,
        record,
        undistorted: np.ndarray,
        tracks,
        prev_locked_id: int | None,
        was_lost: bool,
        dt: float,
        fps: float,
        wait_ms: float,
        inference_ms: float,
        postprocess_ms: float,
    ) -> tuple[TrackingMessage, bool, int | None, tuple[float, ...] | None]:
        """Run the full tracking pipeline for one frame.

        Returns (message, next_was_lost, current_locked_id, prev_locked_bbox).
        """
        ts = self.tracker_stage

        primary_track, reid_match = ts.select_primary_track(
            tracks, record.frame, record.timestamp_ns, prev_locked_id
        )

        current_locked = primary_track.track_id if primary_track is not None else None
        proximity_transfer = ts.consume_proximity_transfer()

        # Handle lock transitions
        if current_locked != prev_locked_id:
            if proximity_transfer:
                LOGGER.info(
                    "Lock transition: %s -> %s (proximity, no reset)",
                    prev_locked_id,
                    current_locked,
                )
            else:
                if prev_locked_id is not None and ts.kalman.initialized:
                    ts.save_track_state(
                        prev_locked_id,
                        self._ema_pixel.snapshot(),
                        self._servo_ema_pixel.snapshot(),
                        record.timestamp_ns,
                    )
                if prev_locked_id is not None:
                    prev_bbox_arr = None
                    # Store ReID appearance for outgoing track
                    for t in tracks:
                        if t.track_id == prev_locked_id:
                            prev_bbox_arr = t.bbox
                            break
                    if prev_bbox_arr is not None:
                        ts.reid_buffer.store_lost_track(
                            prev_locked_id,
                            record.frame,
                            np.array(prev_bbox_arr),
                            record.timestamp_ns,
                        )

                if not reid_match and primary_track is not None and prev_locked_id is not None:
                    matched_id = ts.reid_buffer.match(
                        record.frame,
                        primary_track.bbox,
                        record.timestamp_ns,
                    )
                    if matched_id == prev_locked_id:
                        reid_match = True
                        LOGGER.debug("Re-ID match: track %d → %d", current_locked, prev_locked_id)

                if not reid_match:
                    restored = (
                        ts.restore_track_state(
                            current_locked,
                            self._ema_pixel,
                            self._servo_ema_pixel,
                            current_timestamp_ns=record.timestamp_ns,
                        )
                        if current_locked is not None
                        else False
                    )
                    if not restored and prev_locked_id is not None and current_locked is not None:
                        restored = ts.restore_track_state(
                            prev_locked_id,
                            self._ema_pixel,
                            self._servo_ema_pixel,
                            current_timestamp_ns=record.timestamp_ns,
                        )
                    if not restored:
                        ts.stabilizer.reset()
                        ts.kalman.reset()
                        self.adaptive.reset()
                    LOGGER.info(
                        "Lock transition: %s -> %s (reid=%s, cache=%s)",
                        prev_locked_id,
                        current_locked,
                        "hit" if reid_match else "miss",
                        (
                            "hit"
                            if (not reid_match and restored)
                            else ("skip" if reid_match else "miss")
                        ),
                    )

        prev_locked_bbox = tuple(primary_track.bbox) if primary_track is not None else None

        # Build message
        message, next_was_lost = self._build_message(
            primary_track=primary_track,
            all_tracks=tracks,
            record=record,
            frame=undistorted,
            was_lost=was_lost,
            dt=dt,
            fps=fps,
            wait_ms=wait_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
        )

        # Feed adaptive controller
        detected = message.target_acquired
        speed = 0.0
        if message.filtered_velocity is not None:
            vx, vy = message.filtered_velocity
            speed = (vx * vx + vy * vy) ** 0.5
        self.adaptive.update(detected, speed)
        ts.tracker.set_track_threshold(self.adaptive.confidence_threshold)
        message = replace(message, hold_time_s=self.adaptive.hold_time_s)

        if message.state_source == "measurement":
            self._measurement_update_count += 1
            if ts.kalman.last_innovation_gated:
                self._measurement_gated_count += 1

        # Notify adaptive controller of camera motion (slew freeze)
        if message.servo_angular_velocity is not None:
            pan_dps = degrees(message.servo_angular_velocity[0])
            tilt_dps = degrees(message.servo_angular_velocity[1])
            cam_speed = (pan_dps**2 + tilt_dps**2) ** 0.5
            self.adaptive.notify_camera_motion(cam_speed)

        return message, next_was_lost, current_locked, prev_locked_bbox

    def _build_message(
        self,
        *,
        primary_track,
        all_tracks,
        record,
        frame: np.ndarray,
        was_lost: bool,
        dt: float,
        fps: float,
        wait_ms: float,
        inference_ms: float,
        postprocess_ms: float,
    ) -> tuple[TrackingMessage, bool]:

        ts = self.tracker_stage
        cs = self.centroid_stage
        kalman = ts.kalman
        stabilizer = ts.stabilizer

        tracking_start_ns = perf_counter_ns()
        is_occlusion_recovery = False
        egomotion_applied_px: tuple[float, float] | None = None

        if primary_track is not None and primary_track.lost_frames == 0:
            detection = Detection(
                bbox=primary_track.bbox,
                score=primary_track.score,
                keypoints=primary_track.keypoints,
            )
            raw_pixel = compute_stabilized_centroid(
                detection,
                self._pose_schema,
                stabilizer,
                pp_config=self._tracking_config.postprocess,
            )

            if was_lost:
                is_occlusion_recovery = True
                kalman.oru_re_update()

            compensation = cs.compensate_measurement_for_egomotion(
                raw_pixel,
                dt,
                record.timestamp_ns,
            )
            measurement_pixel = raw_pixel
            if compensation is not None:
                measurement_pixel = compensation.compensated_pixel
                egomotion_applied_px = compensation.applied_delta_px

            filtered_state = kalman.update(measurement_pixel, dt)
            filtered_pixel = filtered_state.position if filtered_state is not None else raw_pixel
            ts.last_locked_centroid = filtered_pixel

            if filtered_state is not None:
                filtered_velocity = (filtered_state.vx, filtered_state.vy)
            else:
                filtered_velocity = (0.0, 0.0)
            ts.last_locked_velocity = filtered_velocity
            ts.last_locked_timestamp_ns = record.timestamp_ns

            current_time_s = record.timestamp_ns / 1e9
            servo_filtered_pixel = self._servo_ema_pixel(filtered_pixel, current_time_s)
            servo_angles = cs.compute_angles(servo_filtered_pixel)
            servo_angular_velocity = cs.compute_angular_velocity(
                servo_filtered_pixel,
                filtered_state.vx if filtered_state is not None else 0.0,
                filtered_state.vy if filtered_state is not None else 0.0,
            )

            filtered_pixel = self._ema_pixel(filtered_pixel, current_time_s)
            raw_angles = cs.camera_model.pixel_to_angle(*raw_pixel)
            filtered_angles = cs.compute_angles(filtered_pixel)
            angular_velocity = cs.compute_angular_velocity(
                filtered_pixel,
                self._ema_pixel.dx,
                self._ema_pixel.dy,
            )

            state_source = "measurement"
            track_id = primary_track.track_id
            confidence = max(0.0, min(1.0, float(primary_track.score)))
            target_acquired = True
            next_was_lost = False
        else:
            predicted = kalman.predict(dt)
            raw_pixel = None
            filtered_pixel = predicted.position if predicted is not None else None
            raw_angles = None
            filtered_angles = (
                cs.compute_angles(filtered_pixel) if filtered_pixel is not None else None
            )
            servo_angles = filtered_angles
            state_source = "prediction" if predicted is not None else "lost"
            track_id = primary_track.track_id if primary_track is not None else None
            confidence = 0.0
            target_acquired = False
            next_was_lost = True
            if predicted is not None:
                filtered_velocity = (predicted.vx, predicted.vy)
                angular_velocity = cs.compute_angular_velocity(
                    filtered_pixel,
                    predicted.vx,
                    predicted.vy,
                )
                servo_angular_velocity = angular_velocity
            else:
                filtered_velocity = None
                angular_velocity = None
                servo_angular_velocity = None

        # Laser overlay (open-loop only — visual-servo PID removed)
        laser_pixel, servo_angles = self.servo_stage.process(
            frame,
            filtered_pixel,
            servo_angles,
            target_acquired,
            dt,
        )

        # Non-primary targets
        primary_id = primary_track.track_id if primary_track is not None else None
        other_targets = tuple(
            (
                float((t.bbox[0] + t.bbox[2]) * 0.5),
                float((t.bbox[1] + t.bbox[3]) * 0.5),
                t.track_id,
                float(t.score),
            )
            for t in all_tracks
            if t.track_id != primary_id and t.lost_frames == 0
        )

        tracking_ms = (perf_counter_ns() - tracking_start_ns) / 1_000_000.0
        total_latency_ms = (perf_counter_ns() - record.timestamp_ns) / 1_000_000.0
        msg = TrackingMessage(
            frame_id=record.frame_id,
            timestamp_ns=record.timestamp_ns,
            target_kind=self._pose_schema.target_kind,
            target_acquired=target_acquired,
            state_source=state_source,
            track_id=track_id,
            confidence=float(confidence),
            raw_pixel=raw_pixel,
            filtered_pixel=filtered_pixel,
            raw_angles=raw_angles,
            filtered_angles=filtered_angles,
            servo_angles=servo_angles,
            servo_angular_velocity=servo_angular_velocity,
            filtered_velocity=filtered_velocity,
            angular_velocity=angular_velocity,
            inference_ms=inference_ms,
            tracking_ms=tracking_ms,
            total_latency_ms=total_latency_ms,
            fps=fps,
            wait_ms=wait_ms,
            postprocess_ms=postprocess_ms,
            is_occlusion_recovery=is_occlusion_recovery,
            laser_pixel=laser_pixel,
            other_targets=other_targets,
            egomotion_applied_px=egomotion_applied_px,
        )
        return msg, next_was_lost
