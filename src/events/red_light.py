"""red_light event detector (PHASE 16).

Definition: a VEHICLE approaches a configured stop line while the traffic-light
signal covering that stop line is RED, crosses the line instead of stopping,
and continues through (post-crossing motion).

Evidence requires, per frame and per vehicle track (bottom-center in scene
pixels, the ONLY spatial convention used):

  * vehicle  : label in `vehicle_labels` (car/truck/bus/motorcycle).
  * motion   : MotionState speed >= `min_vehicle_speed_px_s`. A vehicle stopped
               before / at the line never produces evidence.
  * signal   : the traffic-light state for the stop line being
               approached/crossed must be exactly "RED". GREEN / YELLOW /
               UNKNOWN (and the geometry default) all REJECT.
  * approach : the vehicle must CLOSE on the line (distance decreasing inside
               `max_stop_line_distance_px` at >= `min_approach_speed_px_s`);
               the approach flag is latched while the vehicle stays near the
               line so the crossing frame itself cannot flicker it off.
  * crossing : the bottom-center must intersect the stop-line segment
               (existing `Geometry.crosses_stop_line` plus per-line segment
               checks to identify WHICH configured line was crossed).
  * continue : after crossing, evidence stays on while the vehicle keeps
               moving through the `crossing_evidence_window_sec` window and
               has `min_post_crossing_motion_sec` of real forward motion
               (crossing then stopping early -> short run -> no event). If the
               vehicle rolls back onto the approach side the crossing is
               invalidated (tracking jitter / stop).

CRITICAL UNKNOWN RULE:
  `Geometry.get_traffic_light_state(...)` may return "UNKNOWN" (it currently
  ALWAYS does - no traffic-light classifier exists). UNKNOWN (or missing, or
  any non-RED/GREEN/YELLOW value) MUST NEVER produce red_light evidence.
  This detector never guesses a color, never fakes a signal, never reads frame
  brightness as a signal proxy, and never edits scene_config.json. With the
  current geometry the detector therefore yields 0 events - that is CORRECT.

  The signal is read through `signal_fn(geometry, line_id, t_sec)` (constructor
  injectable for tests). The default consults the traffic-light ROI nearest to
  the stop line in question and calls the existing Geometry API; when the
  mapping line->signal is not determinable it stays UNKNOWN and rejects.

Each vehicle keeps a bounded, strictly causal per-track state (last position,
last distance to the line, latched approach, registered crossing/time/line/
signal, approach side, post-crossing motion). The state is pruned after
`max_track_gap_sec`; a track id that reappears after the gap starts FRESH (no
state bleed, id-reuse safe). All checks use only t <= current time - the signal
is captured AT the crossing frame, so RED-before-crossing is required and later
RED/GREEN changes neither create nor revoke a crossing.

Deterministic. Temporal confirmation reuses the shared TemporalEventEngine
(label "red_light").
"""

from __future__ import annotations

from ..scene.geometry import segments_intersect
from .temporal import EventSegment, TemporalEventEngine

LABEL = "red_light"
DEFAULT_VEHICLE_LABELS = ("car", "truck", "bus", "motorcycle")

# canonical ordered rejection reasons (CSV / debug reports order-friendly)
REASONS = ("no_stop_line_geometry", "insufficient_trajectory", "low_quality",
           "vehicle_stationary", "traffic_light_unknown", "light_green",
           "light_yellow", "moving_away", "outside_approach", "no_crossing",
           "crossing_rolled_back", "post_crossing_stop", "episode_ended")

RED, GREEN, YELLOW, UNKNOWN = "RED", "GREEN", "YELLOW", "UNKNOWN"


def _default_signal(geometry, line_id, t_sec):
    """Default signal source: existing Geometry API, ROI nearest to stop line."""
    roi = None
    try:
        if geometry is not None and geometry.traffic_light_rois and \
                line_id is not None and line_id < len(geometry.stop_lines):
            line = geometry.stop_lines[line_id]
            mx = (line[0][0] + line[1][0]) / 2.0
            my = (line[0][1] + line[1][1]) / 2.0
            best = 0
            best_d = float("inf")
            for i, r in enumerate(geometry.traffic_light_rois):
                rx = (r[0][0] + r[1][0]) / 2.0
                ry = (r[0][1] + r[1][1]) / 2.0
                d = (rx - mx) ** 2 + (ry - my) ** 2
                if d < best_d:
                    best_d, best = d, i
            roi = geometry.traffic_light_rois[best] if \
                geometry.traffic_light_rois else None
        if geometry is not None and hasattr(geometry, "get_traffic_light_state"):
            return str(geometry.get_traffic_light_state(None, roi)).upper()
    except Exception:
        pass
    return UNKNOWN


class RedLightDetector:
    """Per-frame red_light evidence for vehicle tracks.

    Signature mirrors the other detectors: `update(tracks, motion_states,
    geometry, t_sec)` -> report; `finalize()` -> confirmed segments; `reset()`.
    Geometry is required for stop lines; the signal is "UNKNOWN" by default
    (existing Geometry API), so evidence requires an authoritative RED source
    (injected `signal_fn` for tests / a future classifier).
    """

    def __init__(
        self,
        vehicle_labels: tuple = DEFAULT_VEHICLE_LABELS,
        min_vehicle_speed_px_s: float = 12.0,
        min_approach_speed_px_s: float = 10.0,
        max_stop_line_distance_px: float = 140.0,
        min_post_crossing_motion_sec: float = 0.4,
        crossing_evidence_window_sec: float = 0.8,
        max_track_gap_sec: float = 2.0,
        min_trajectory_points: int = 3,
        min_quality: float = 0.2,
        signal_fn=None,
        temporal: TemporalEventEngine | None = None,
        min_on_duration: float = 0.35,
        allowed_gap: float = 0.3,
        merge_gap: float = 1.0,
        min_duration: float = 0.3,
    ) -> None:
        assert min_vehicle_speed_px_s > 0.0
        assert min_approach_speed_px_s > 0.0
        assert max_stop_line_distance_px > 0.0
        assert min_post_crossing_motion_sec >= 0.0
        assert crossing_evidence_window_sec > 0.0
        assert max_track_gap_sec > 0.0
        assert min_trajectory_points >= 1
        self.vehicle_labels = frozenset(vehicle_labels)
        self.min_vehicle_speed = float(min_vehicle_speed_px_s)
        self.min_approach_speed = float(min_approach_speed_px_s)
        self.max_line_distance = float(max_stop_line_distance_px)
        self.min_post_motion = float(min_post_crossing_motion_sec)
        self.window = float(crossing_evidence_window_sec)
        self.max_track_gap = float(max_track_gap_sec)
        self.min_traj_points = int(min_trajectory_points)
        self.min_quality = float(min_quality)
        self.signal_fn = signal_fn if signal_fn is not None else _default_signal
        self.candidate_labels = self.vehicle_labels
        self.temporal = temporal if temporal is not None else TemporalEventEngine(
            min_on_duration=min_on_duration, allowed_gap=allowed_gap,
            merge_gap=merge_gap, min_duration=min_duration)
        # bounded per-track causal state, pruned by max_track_gap_sec
        self._state: dict[int, dict] = {}

    # ------------------------------------------------------------------ API
    def update(self, tracks, motion_states, geometry=None, t_sec: float = 0.0) -> dict:
        """Evaluate red_light evidence for every vehicle track at time t_sec.

        Args:
            tracks:        dict {track_id -> TrackTrajectory} (position + label).
            motion_states: dict {track_id -> MotionState} (smoothed motion).
            geometry:      Geometry with stop_lines (required).
            t_sec:         current time (causal).
        Returns per-frame report:
            {"t_sec", "evidence", "active_tracks", "tracks": {tid: record},
             "rejected": {tid: reason}, "signal": str}
        and feeds ("red_light", evidence) into the TemporalEventEngine.
        """
        has_geometry = geometry is not None and bool(geometry.stop_lines)
        records: dict[int, dict] = {}
        active: list[int] = []
        signal = self.signal_fn(geometry, None, t_sec) if geometry else UNKNOWN
        for tid, tr in tracks.items():
            if tr.last is None or tr.label not in self.vehicle_labels:
                continue
            if not has_geometry:
                rec = self._rec_base(tr, motion_states.get(tid), t_sec,
                                     signal=signal, reason="no_stop_line_geometry")
            else:
                rec = self._evaluate(tr, motion_states.get(tid), geometry, t_sec)
            records[tid] = rec
            if rec["evidence"]:
                active.append(tid)
        self._prune(t_sec)
        active.sort()
        evidence = bool(active)
        self.temporal.update(LABEL, t_sec, evidence=evidence)
        rejected = {tid: rec["reason"] for tid, rec in records.items()
                    if not rec["evidence"] and rec["reason"] is not None}
        return {"t_sec": t_sec, "evidence": evidence, "active_tracks": active,
                "tracks": records, "rejected": rejected, "signal": signal}

    def finalize(self) -> list[EventSegment]:
        return self.temporal.finalize()

    def reset(self) -> None:
        self.temporal.reset()
        self._state.clear()

    # ------------------------------------------------------------- evaluate
    def _evaluate(self, tr, st, geometry, t_sec: float) -> dict:
        tid = tr.track_id
        pos = (tr.last.x, tr.last.bottom_y)
        speed = st.speed if st is not None else 0.0
        moving = speed >= self.min_vehicle_speed

        s = self._state.get(tid)
        fresh = s is None
        if fresh:
            s = {"first_t": t_sec, "last_t": t_sec, "last_pos": pos,
                 "last_dist": None, "approaching": False,
                 "crossing_t": None, "crossing_line": None,
                 "approach_side": None, "light_at_crossing": None,
                 "post_motion": False, "last_moved": (t_sec if moving else None),
                 "reason": None}
            self._state[tid] = s
        else:
            s["last_t"] = t_sec
        if moving:
            s["last_moved"] = t_sec

        d, line_id = self._nearest_line(geometry, pos)
        raw_cross = bool(geometry.crosses_stop_line(s["last_pos"], pos))
        cross_line = self._crossing_line(geometry, s["last_pos"], pos)
        side_now = None
        if cross_line is not None:
            side_now = self._side(geometry, pos, cross_line)
        elif line_id is not None:
            side_now = self._side(geometry, pos, line_id)
        signal = self.signal_fn(geometry, line_id, t_sec)

        # ---- latched approach ----
        approaching = bool(s["approaching"])
        if d is not None and d <= self.max_line_distance and \
                s["last_dist"] is not None and d < s["last_dist"] and \
                speed >= self.min_approach_speed:
            approaching = True
            s["approaching"] = True
        elif approaching and d is not None and d > self.max_line_distance * 2.0:
            approaching = False
            s["approaching"] = False

        reason = None
        evidence = False
        received_light_at_cross = False

        if len(tr.points) < self.min_traj_points:
            reason = "insufficient_trajectory"
        elif st is None or st.quality < self.min_quality:
            reason = "low_quality"
        elif not moving:
            reason = "vehicle_stationary"
        elif s["crossing_t"] is not None:
            # ---- active crossing episode (signal captured AT crossing) ----
            cx_line = s["crossing_line"]
            now_side = self._side(geometry, pos, cx_line) \
                if cx_line is not None else None
            if now_side is not None and s["approach_side"] is not None and \
                    now_side == s["approach_side"] and t_sec > s["crossing_t"]:
                # rolled back onto the approach side: jitter / stop
                s["crossing_t"] = None
                s["crossing_line"] = None
                s["light_at_crossing"] = None
                reason = "crossing_rolled_back"
            else:
                window_end = s["crossing_t"] + self.window
                if t_sec <= window_end and moving:
                    reason = None
                    evidence = True
                    if t_sec - s["crossing_t"] >= self.min_post_motion:
                        s["post_motion"] = True
                        received_light_at_cross = True
                else:
                    reason = "post_crossing_stop" if not moving \
                        else "episode_ended"
                    if t_sec > window_end:
                        s["crossing_t"] = None
                        s["crossing_line"] = None
                        s["light_at_crossing"] = None
        elif signal == RED and cross_line is not None and approaching:
            # ---- fresh crossing under an authoritative RED ----
            s["crossing_t"] = t_sec
            s["crossing_line"] = cross_line
            s["approach_side"] = self._side(geometry, s["last_pos"], cross_line)
            s["light_at_crossing"] = signal
            reason = None
            evidence = True
            received_light_at_cross = True
        else:
            # ---- signal / approach gates (no active crossing) ----
            if signal == GREEN:
                reason = "light_green"
            elif signal == YELLOW:
                reason = "light_yellow"
            elif signal != RED:
                reason = "traffic_light_unknown"
            elif d is not None and s["last_dist"] is not None and \
                    d >= s["last_dist"]:
                reason = "moving_away"
            elif d is None or not approaching:
                reason = "outside_approach"
            else:
                reason = "no_crossing"

        if reason is not None:
            s["reason"] = reason

        rec = {
            "track_id": tid, "class": tr.label,
            "first_t": s["first_t"], "last_t": t_sec,
            "speed": speed, "heading_deg": st.heading_deg if st else None,
            "accel": st.accel if st else None,
            "stationary": st.stationary if st else None,
            "quality": st.quality if st else None,
            "signal": signal,
            "distance_to_stop_line": d,
            "stop_line_id": line_id,
            "crossing_line": s["crossing_line"],
            "crossing_time": s["crossing_t"],
            "light_at_crossing": s["light_at_crossing"],
            "crossing_received": received_light_at_cross,
            "approach": bool(s["approaching"]),
            "post_crossing_motion": bool(s["post_motion"]),
            "raw_crossing": bool(raw_cross),
            "evidence": bool(evidence), "reason": reason,
        }
        s["last_pos"] = pos
        s["last_dist"] = d
        return rec

    # ------------------------------------------------------------------ util
    def _nearest_line(self, geometry, pos):
        """(distance_px, line_index) to the closest stop line; scale to frame px."""
        a = geometry.to_ref(pos)
        best_d = float("inf")
        best_i = None
        if not geometry.stop_lines:
            return None, None
        scale = (geometry.sx + geometry.sy) / 2.0 or 1.0
        for i, line in enumerate(geometry.stop_lines):
            d_ref = _point_segment_distance(a, line[0], line[1])
            d_px = d_ref * scale
            if d_px < best_d:
                best_d, best_i = d_px, i
        return best_d, best_i

    def _crossing_line(self, geometry, prev, cur):
        """Index of the FIRST stop line whose segment the travel intersects."""
        a = geometry.to_ref(prev)
        b = geometry.to_ref(cur)
        for i, line in enumerate(geometry.stop_lines):
            if segments_intersect(a, b, line[0], line[1]):
                return i
        return None

    @staticmethod
    def _side(geometry, pos, line_index):
        line = geometry.stop_lines[line_index]
        a = geometry.to_ref(pos)
        l0, l1 = line
        cross = (l1[0] - l0[0]) * (a[1] - l0[1]) - \
            (l1[1] - l0[1]) * (a[0] - l0[0])
        return 1 if cross >= 0 else -1

    @staticmethod
    def _rec_base(tr, st, t_sec, signal, reason=None) -> dict:
        return {
            "track_id": tr.track_id, "class": tr.label,
            "first_t": t_sec, "last_t": t_sec,
            "speed": st.speed if st else None,
            "heading_deg": st.heading_deg if st else None,
            "accel": st.accel if st else None,
            "stationary": st.stationary if st else None,
            "quality": st.quality if st else None,
            "signal": signal,
            "distance_to_stop_line": None, "stop_line_id": None,
            "crossing_line": None, "crossing_time": None,
            "light_at_crossing": None, "crossing_received": False,
            "approach": False, "post_crossing_motion": False,
            "raw_crossing": False,
            "evidence": False, "reason": reason,
        }

    # ---------------------------------------------------------------- prune
    def _prune(self, t_sec: float) -> None:
        if t_sec < 0.0:
            return
        stale = [k for k, s in self._state.items()
                 if t_sec - s["last_t"] > self.max_track_gap]
        for k in stale:
            del self._state[k]


def _point_segment_distance(p, a, b):
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    px, py = p[0], p[1]
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def segments_to_events(segments: list[EventSegment]) -> list[list]:
    """EventSegments -> harness events [[start, end, label]]. (Part A glue.)"""
    return [s.to_list() for s in segments]