from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.inference.postprocess import KeypointStabilizer
from src.tracking.botsort import BoTSORT
from src.tracking.kalman import KalmanFilter
from src.tracking.reid import ReIDBuffer

LOGGER = logging.getLogger("inference.tracker")

_MAX_CACHED_TRACKS = 8
_MAX_CACHE_AGE_NS = 10_000_000_000  # 10 seconds
_PROXIMITY_TRANSFER_PX = 80.0


@dataclass
class _TrackStateSnapshot:
    kalman_snap: dict
    stabilizer_snap: np.ndarray | None
    ema_snap: dict
    servo_ema_snap: dict
    timestamp_ns: int


class TrackerStage:
    """Manages BoTSORT tracking, target selection, Kalman state cache, and ReID."""

    def __init__(
        self,
        tracker: BoTSORT,
        kalman: KalmanFilter,
        stabilizer: KeypointStabilizer,
        reid_buffer: ReIDBuffer,
        max_lost_frames: int,
    ) -> None:
        self.tracker = tracker
        self.kalman = kalman
        self.stabilizer = stabilizer
        self.reid_buffer = reid_buffer
        self._max_lost_frames = max_lost_frames
        self._locked_track_id: int | None = None
        self._cycle_requested: bool = False
        self._proximity_transfer: bool = False
        self._last_locked_centroid: tuple[float, float] | None = None
        self._last_locked_velocity: tuple[float, float] | None = None
        self._last_locked_timestamp_ns: int | None = None
        self._track_state_cache: dict[int, _TrackStateSnapshot] = {}

    @property
    def locked_track_id(self) -> int | None:
        return self._locked_track_id

    @property
    def last_locked_centroid(self) -> tuple[float, float] | None:
        return self._last_locked_centroid

    @last_locked_centroid.setter
    def last_locked_centroid(self, value: tuple[float, float] | None) -> None:
        self._last_locked_centroid = value

    @property
    def last_locked_velocity(self) -> tuple[float, float] | None:
        return self._last_locked_velocity

    @last_locked_velocity.setter
    def last_locked_velocity(self, value: tuple[float, float] | None) -> None:
        self._last_locked_velocity = value

    @property
    def last_locked_timestamp_ns(self) -> int | None:
        return self._last_locked_timestamp_ns

    @last_locked_timestamp_ns.setter
    def last_locked_timestamp_ns(self, value: int | None) -> None:
        self._last_locked_timestamp_ns = value

    def request_cycle(self) -> None:
        self._cycle_requested = True

    def request_relock(self) -> None:
        self._last_locked_velocity = None
        self._last_locked_timestamp_ns = None
        self._locked_track_id = None
        self._last_locked_centroid = None

    def consume_proximity_transfer(self) -> bool:
        flag = self._proximity_transfer
        self._proximity_transfer = False
        return flag

    def select_primary_track(self, tracks, frame, timestamp_ns, prev_locked_id):
        if not tracks:
            self._locked_track_id = None
            self._cycle_requested = False
            return None, False

        # Cycle target
        if self._cycle_requested:
            self._cycle_requested = False
            candidates = [t for t in tracks if t.lost_frames <= self._max_lost_frames]
            if len(candidates) > 1:
                ids = sorted(t.track_id for t in candidates)
                current = self._locked_track_id
                idx = ids.index(current) if current is not None and current in ids else -1
                next_id = ids[(idx + 1) % len(ids)]
                for t in candidates:
                    if t.track_id == next_id:
                        self._locked_track_id = next_id
                        LOGGER.info("Cycled target: track_id=%d", next_id)
                        return t, False
            elif candidates:
                self._locked_track_id = candidates[0].track_id
                LOGGER.info(
                    "Cycle target: only 1 candidate, locked track_id=%d", candidates[0].track_id
                )
                return candidates[0], False

        # If we have a locked target, prefer it
        if self._locked_track_id is not None:
            for track in tracks:
                if track.track_id == self._locked_track_id:
                    if track.lost_frames <= self._max_lost_frames:
                        return track, False
                    break

            # Proximity transfer
            if self._last_locked_centroid is not None:
                best_prox = None
                best_dist = _PROXIMITY_TRANSFER_PX
                for track in tracks:
                    if track.lost_frames > 0:
                        continue
                    cx = (track.bbox[0] + track.bbox[2]) * 0.5
                    cy = (track.bbox[1] + track.bbox[3]) * 0.5
                    dist = (
                        (cx - self._last_locked_centroid[0]) ** 2
                        + (cy - self._last_locked_centroid[1]) ** 2
                    ) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_prox = track
                if best_prox is not None:
                    old_id = self._locked_track_id
                    self._locked_track_id = best_prox.track_id
                    self._proximity_transfer = True
                    LOGGER.debug(
                        "Proximity transfer: %d -> %d (%.1fpx)",
                        old_id,
                        best_prox.track_id,
                        best_dist,
                    )
                    return best_prox, False

            self._locked_track_id = None

        # Re-ID Lock-on
        if prev_locked_id is not None:
            for track in tracks:
                if track.lost_frames == 0:
                    matched_id = self.reid_buffer.match(frame, track.bbox, timestamp_ns)
                    if matched_id == prev_locked_id:
                        self._locked_track_id = track.track_id
                        LOGGER.info("Re-ID Lock-on: Target recovered track_id=%d", track.track_id)
                        return track, True

        # No lock — pick best visible track
        last_centroid = self._last_locked_centroid
        predicted_centroid = last_centroid
        if (
            last_centroid is not None
            and self._last_locked_velocity is not None
            and self._last_locked_timestamp_ns is not None
        ):
            dt_s = (timestamp_ns - self._last_locked_timestamp_ns) / 1e9
            if 0.0 < dt_s < 2.0:
                vx, vy = self._last_locked_velocity
                predicted_centroid = (
                    last_centroid[0] + vx * dt_s,
                    last_centroid[1] + vy * dt_s,
                )
        if predicted_centroid is not None:
            visible_tracks = [t for t in tracks if t.lost_frames == 0]
            if not visible_tracks:
                visible_tracks = tracks

            def _relock_score(t):
                cx = (t.bbox[0] + t.bbox[2]) * 0.5
                cy = (t.bbox[1] + t.bbox[3]) * 0.5
                dist_sq = (cx - predicted_centroid[0]) ** 2 + (cy - predicted_centroid[1]) ** 2
                age_penalty = 0 if t.age >= 2 else 1
                return (t.lost_frames, age_penalty, dist_sq, -t.score)

            best = min(visible_tracks, key=_relock_score)
        else:
            best = min(tracks, key=lambda t: (t.lost_frames, t.track_id))

        self._locked_track_id = best.track_id
        if LOGGER.isEnabledFor(logging.DEBUG):
            candidates_str = ", ".join(
                f"id={t.track_id} lost={t.lost_frames} score={t.score:.2f}" for t in tracks
            )
            LOGGER.debug(
                "Re-lock candidates: [%s] -> selected id=%d", candidates_str, best.track_id
            )
        LOGGER.info("Target locked: track_id=%d", best.track_id)
        return best, False

    def save_track_state(
        self,
        track_id: int,
        ema_snap: dict,
        servo_ema_snap: dict,
        timestamp_ns: int,
    ) -> None:
        self._track_state_cache[track_id] = _TrackStateSnapshot(
            kalman_snap=self.kalman.snapshot(),
            stabilizer_snap=self.stabilizer.snapshot(),
            servo_ema_snap=servo_ema_snap,
            ema_snap=ema_snap,
            timestamp_ns=timestamp_ns,
        )
        while len(self._track_state_cache) > _MAX_CACHED_TRACKS:
            oldest_id = min(
                self._track_state_cache, key=lambda k: self._track_state_cache[k].timestamp_ns
            )
            del self._track_state_cache[oldest_id]

    def restore_track_state(
        self,
        track_id: int,
        ema_pixel,
        servo_ema_pixel,
        current_timestamp_ns: int,
    ) -> bool:
        snap = self._track_state_cache.pop(track_id, None)
        if snap is None:
            return False
        if current_timestamp_ns and (current_timestamp_ns - snap.timestamp_ns) > _MAX_CACHE_AGE_NS:
            LOGGER.debug(
                "Rejected stale cache for track_id=%d (age=%.1fs)",
                track_id,
                (current_timestamp_ns - snap.timestamp_ns) / 1e9,
            )
            return False
        self.kalman.restore(snap.kalman_snap)
        self.stabilizer.restore(snap.stabilizer_snap)
        servo_ema_pixel.restore(snap.servo_ema_snap)
        ema_pixel.restore(snap.ema_snap)
        LOGGER.debug("Restored cached state for track_id=%d", track_id)
        return True
