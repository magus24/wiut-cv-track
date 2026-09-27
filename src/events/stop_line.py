"""stop_line event detector (PHASE 17).

Definition: a VEHICLE fully crosses a configured stop line (bottom-center in
scene pixels, the ONLY spatial convention used) while coming from a genuine
approach (observed on a PRIOR frame, closing on the line) and continuing
through it. This is a pure geometry + trajectory fact.

CRITICALLY DIFFERENT FROM red_light (PHASE 16):
  * stop_line is SIGNAL-INDEPENDENT. It never reads the traffic-light state,
    never calls get_traffic_light_state(), and never cares whether the signal
    is RED, GREEN, YELLOW or UNKNOWN. RED-red_light is a DIFFERENT event that
    happens to reuse the same physical line crossing; stop_line fires for
    EVERY confirmed crossing regardless of the light.
  * state is per (vehicle_id, stop_line_id), not per vehicle, so one vehicle
    crossing several lines produces one episode per line and these episodes
    never bleed into each other.
  * the default policy is conservative for "born after the line": a track whose
    first observed frame is already at/beyond the line can never claim a
    crossing (no approach history -> no crossing). No post-hoc correction.

Crossing evidence requires, per vehicle track and stop line:
  * vehicle  : label in `vehicle_labels` (car/truck/bus/motorcycle).
  * motion   : MotionState speed >= `min_vehicle_speed_px_s`. A vehicle that
               stops before / at the line never produces evidence (the STOPPED
               case is a protection, not an event by itself).
  * approach : tracked per (vehicle, line). Latching requires the vehicle to be
               CLOSING on that line (distance decreasing) while inside
               `max_stop_line_distance_px` at >= `min_approach_speed_px_s` and
               observed on a PREVIOUS frame (no approach history on the very
               first frame). The latch holds until the vehicle leaves a 2x
               band; a `cooldown` after a completed/rolled-back crossing
               blocks re-latching until the vehicle leaves that 2x band, so
               oscillation around the line cannot re-trigger nonstop.
  * crossing : previous-frame bottom-center -> current bottom-center segment
               must intersect THE stop line (existing Geometry API + per-line
               segment tests to identify WHICH configured line was crossed).
               A one-frame intersection / bbox near the line is never enough.
  * continue : evidence stays on for `post_crossing_evidence_window_sec` after
               the crossing frame (post-crossing braking exception: a vehicle
               that decelerates right after the line does not erase the
               crossing). If the vehicle rolls back onto the approach side the
               episode is invalidated (tracking jitter).

Per-(vehicle, line) bounded causal state: last timestamp / previous position
(per track) / previous side, approach latch, crossing timestamp, window end,
post-crossing motion and cooldown. State is pruned after `max_track_gap_sec`; a
track id that reappears after the gap starts FRESH (no state bleed, id-reuse
safe).

Deterministic. Temporal confirmation reuses the shared TemporalEventEngine
(label "stop_line"). Signal is absent from every decision path: the report's
`"signal"` field is always None.
"""

from __future__ import annotations

from ..scene.geometry import segments_intersect
from .temporal import EventSegment, TemporalEventEngine

LABEL = "stop_line"
DEFAULT_VEHICLE_LABELS = ("car", "truck", "bus", "motorcycle")

# canonical ordered rejection reasons (CSV / debug reports order-friendly)
REASONS = ("no_stop_line_geometry", "insufficient_trajectory", "low_quality",
           "vehicle_stationary", "stopped_before_line", "moving_away",
           "outside_approach", "no_crossing", "born_after_crossing",
           "crossing_rolled_back", "episode_ended")


class StopLineDetector:
    """Per-frame stop_line evidence for vehicle tracks.

    Signature mirrors the other detectors: `update(tracks, motion_states,
    geometry, t_sec)` -> report; `finalize()` -> confirmed segments; `reset()`.
    Geometry provides the configured stop lines; the traffic-light state is
    intentionally NEVER consulted.
    """

    def __init__(
        self,
        vehicle_labels: tuple = DEFAULT_VEHICLE_LABELS,
        min_vehicle_speed_px_s: float = 12.0,
        min_approach_speed_px_s: float = 10.0,
        max_stop_line_distance_px: float = 140.0,
        min_trajectory_points: int = 3,
        min_quality: float = 0.2,
        max_track_gap_sec: float = 2.0,
        post_crossing_evidence_window_sec: float = 0.6,
        temporal: TemporalEventEngine | None = None,
        min_on_duration: float = 0.25,
        allowed_gap: float = 0.5,
        merge_gap: float = 1.0,
        min_duration: float = 0.25,
    ) -> None:
        assert min_vehicle_speed_px_s > 0.0
        assert min_approach_speed_px_s > 0.0
        assert max_stop_line_distance_px > 0.0
        assert post_crossing_evidence_window_sec > 0.0
        assert max_track_gap_sec > 0.0
        assert min_trajectory_points >= 1
        self.vehicle_labels = frozenset(vehicle_labels)
        self.min_vehicle_speed = float(min_vehicle_speed_px_s)
        self.min_approach_speed = float(min_approach_speed_px_s)
        self.max_line_distance = float(max_stop_line_distance_px)
        self.window = float(post_crossing_evidence_window_sec)
        self.min_traj_points = int(min_trajectory_points)
        self.min_quality = float(min_quality)
        self.max_track_gap = float(max_track_gap_sec)
        self.candidate_labels = self.vehicle_labels
        self.temporal = temporal if temporal is not None else TemporalEventEngine(
            min_on_duration=min_on_duration, allowed_gap=allowed_gap,
            merge_gap=merge_gap, min_duration=min_duration)
        # bounded causal state
        self._state: dict[tuple[int, int], dict] = {}   # (vehicle_id, line_id)
        self._last_pos: dict[int, tuple[float, float]] = {}
        self._first: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}

    # ------------------------------------------------------------------ API
    def update(self, tracks, motion_states, geometry=None, t_sec: float = 0.0) -> dict:
        """Evaluate stop_line evidence for every vehicle track at time t_sec.

        Args:
            tracks:        dict {track_id -> TrackTrajectory} (position + label).
            motion_states: dict {track_id -> MotionState} (smoothed motion).
            geometry:      Geometry with stop_lines (required).
            t_sec:         current time (causal).
        Returns per-frame report:
            {"t_sec", "evidence", "active_tracks", "tracks": {tid: record},
             "rejected": {tid: reason}, "signal": None}
        and feeds ("stop_line", evidence) into the TemporalEventEngine.
        """
        has_geometry = geometry is not None and bool(geometry.stop_lines)
        records: dict[int, dict] = {}
        active: list[int] = []
        for tid, tr in tracks.items():
            if tr.last is None or tr.label not in self.vehicle_labels:
                continue
            if not has_geometry:
                rec = self._rec_base(tr, motion_states.get(tid), t_sec,
                                     reason="no_stop_line_geometry")
            else:
                rec = self._evaluate(tid, tr, motion_states.get(tid), geometry,
                                     t_sec)
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
                "tracks": records, "rejected": rejected, "signal": None}

    def finalize(self) -> list[EventSegment]:
        return self.temporal.finalize()

    def reset(self) -> None:
        self.temporal.reset()
        self._state.clear()
        self._last_pos.clear()
        self._first.clear()
        self._last_seen.clear()

    # ------------------------------------------------------------- evaluate
    def _evaluate(self, tid, tr, st, geometry, t_sec: float) -> dict:
        pos = (tr.last.x, tr.last.bottom_y)
        speed = st.speed if st is not None else 0.0
        moving = speed >= self.min_vehicle_speed
        traj_ok = len(tr.points) >= self.min_traj_points
        quality_ok = st is not None and st.quality >= self.min_quality

        self._first.setdefault(tid, t_sec)
        self._last_seen[tid] = t_sec
        track_fresh = tid not in self._last_pos
        prev_pos = self._last_pos.get(tid)

        ref_cur = geometry.to_ref(pos)
        ref_prev = geometry.to_ref(prev_pos) if prev_pos is not None else None
        raw_cross = (not track_fresh) and \
            bool(geometry.crosses_stop_line(prev_pos, pos))

        line_statuses: dict[int, str] = {}
        active_lines: list[tuple[int, float]] = []    # (line_id, crossing_t)
        nearest_d = float("inf")
        nearest_i = None

        for i, line in enumerate(geometry.stop_lines):
            key = (tid, i)
            s = self._state.setdefault(key, {
                "approaching": False, "prev_dist": None, "prev_side": None,
                "crossing_t": None, "approach_side": None, "window_end": 0.0,
                "post_motion": False, "crossing_confirmed": False,
                "cooldown": False, "last_t": t_sec})
            s["last_t"] = t_sec
            d = _px(ref_cur, line, geometry)
            side = _side_ref(ref_cur, line)
            prev_d = s["prev_dist"]

            # ---- approach latch (per vehicle, per line) ----
            if speed >= self.min_approach_speed and d <= self.max_line_distance \
                    and prev_d is not None and d < prev_d and not s["cooldown"]:
                s["approaching"] = True
            elif s["approaching"] and (d > self.max_line_distance * 2.0
                                       or s["cooldown"]):
                s["approaching"] = False
            if s["cooldown"] and d > self.max_line_distance * 2.0:
                s["cooldown"] = False

            # ---- per-line status ----
            status = "outside_approach"
            if s["crossing_t"] is not None:
                if t_sec > s["window_end"]:
                    s["crossing_t"] = None
                    s["crossing_confirmed"] = True
                    s["cooldown"] = True
                    status = "episode_ended"
                elif t_sec > s["crossing_t"] and s["approach_side"] is not None \
                        and side == s["approach_side"]:
                    s["crossing_t"] = None
                    s["cooldown"] = True
                    status = "crossing_rolled_back"
                else:
                    status = "active"
                    active_lines.append((i, s["crossing_t"]))
                    if moving:
                        s["post_motion"] = True
            elif track_fresh:
                status = "born_after_crossing" if d <= 2 * self.max_line_distance \
                    else "outside_approach"
            else:
                cross = bool(ref_prev is not None and
                             segments_intersect(ref_prev, ref_cur,
                                                line[0], line[1]))
                if cross:
                    if moving and s["approaching"] and traj_ok and quality_ok:
                        s["crossing_t"] = t_sec
                        s["approach_side"] = _side_ref(ref_prev, line)
                        s["window_end"] = t_sec + self.window
                        s["post_motion"] = bool(moving)
                        status = "active"
                        active_lines.append((i, t_sec))
                    elif not s["approaching"]:
                        status = "born_after_crossing"
                    else:
                        status = "no_crossing"
                elif s["approaching"]:
                    status = "stopped_before_line" if not moving else "no_crossing"
                elif prev_d is not None and d >= prev_d:
                    status = "moving_away" if d <= 2 * self.max_line_distance \
                        else "outside_approach"
                else:
                    status = "outside_approach"

            line_statuses[i] = status
            if d < nearest_d:
                nearest_d, nearest_i = d, i
            s["prev_dist"] = d
            s["prev_side"] = side

        # ---- aggregate ----
        crossing_time = min((ct for _, ct in active_lines), default=None)
        crossing_line = next((i for i, ct in active_lines
                              if ct == crossing_time), None)
        evidence = bool(active_lines)
        approach_any = any(s["approaching"]
                           for k, s in self._state.items() if k[0] == tid)
        post_motion_any = any(s["post_motion"]
                              for k, s in self._state.items() if k[0] == tid)
        crossing_confirmed = any(s["crossing_confirmed"]
                                 for k, s in self._state.items() if k[0] == tid)

        reason = None
        if not evidence:
            if not traj_ok:
                reason = "insufficient_trajectory"
            elif not quality_ok:
                reason = "low_quality"
            elif not moving:
                reason = "stopped_before_line" if approach_any \
                    else "vehicle_stationary"
            else:
                # most informative non-trivial status among the lines
                cand = [ln for ln, st in line_statuses.items()
                        if st not in ("outside_approach", "born_after_crossing")]
                if not cand and nearest_i is not None and \
                        line_statuses.get(nearest_i) == "born_after_crossing":
                    cand = [nearest_i]
                if cand:
                    reason = line_statuses[cand[0]]
                else:
                    reason = line_statuses.get(nearest_i, "outside_approach") \
                        if nearest_i is not None else "outside_approach"

        self._last_pos[tid] = pos

        return {
            "track_id": tid, "class": tr.label,
            "first_t": self._first[tid], "last_t": t_sec,
            "speed": speed, "heading_deg": st.heading_deg if st else None,
            "accel": st.accel if st else None,
            "stationary": st.stationary if st else None,
            "quality": st.quality if st else None,
            "signal": None,
            "distance_to_stop_line": nearest_d if nearest_i is not None else None,
            "stop_line_id": nearest_i,
            "approach": bool(approach_any),
            "crossing_line": crossing_line,
            "crossing_time": crossing_time,
            "crossing_received": bool(active_lines and
                                      any(ct == t_sec for _, ct in active_lines)),
            "crossing_confirmed": bool(crossing_confirmed),
            "post_crossing_motion": bool(post_motion_any),
            "raw_crossing": bool(raw_cross),
            "evidence": bool(evidence), "reason": reason,
        }

    # ---------------------------------------------------------------- prune
    def _prune(self, t_sec: float) -> None:
        if t_sec < 0.0:
            return
        stale = [k for k, s in self._state.items()
                 if t_sec - s["last_t"] > self.max_track_gap]
        for k in stale:
            del self._state[k]
        stale_t = [tid for tid, lt in self._last_seen.items()
                   if t_sec - lt > self.max_track_gap]
        for tid in stale_t:
            self._last_pos.pop(tid, None)
            self._first.pop(tid, None)
            self._last_seen.pop(tid, None)

    # ------------------------------------------------------------------ util
    @staticmethod
    def _rec_base(tr, st, t_sec, reason=None) -> dict:
        return {
            "track_id": tr.track_id, "class": tr.label,
            "first_t": t_sec, "last_t": t_sec,
            "speed": st.speed if st else None,
            "heading_deg": st.heading_deg if st else None,
            "accel": st.accel if st else None,
            "stationary": st.stationary if st else None,
            "quality": st.quality if st else None,
            "signal": None,
            "distance_to_stop_line": None, "stop_line_id": None,
            "approach": False, "crossing_line": None, "crossing_time": None,
            "crossing_received": False, "crossing_confirmed": False,
            "post_crossing_motion": False, "raw_crossing": False,
            "evidence": False, "reason": reason,
        }


def _side_ref(ref_point, line) -> int:
    l0, l1 = line
    cross = (l1[0] - l0[0]) * (ref_point[1] - l0[1]) - \
        (l1[1] - l0[1]) * (ref_point[0] - l0[0])
    return 1 if cross >= 0 else -1


def _px(ref_point, line, geometry) -> float:
    """Distance from a REF-space point to a line, scaled to frame px."""
    a, b = line
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    px, py = ref_point[0], ref_point[1]
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        d_ref = ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    else:
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        cx, cy = ax + t * dx, ay + t * dy
        d_ref = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
    scale = (geometry.sx + geometry.sy) / 2.0 or 1.0
    return d_ref * scale


def segments_to_events(segments: list[EventSegment]) -> list[list]:
    """EventSegments -> harness events [[start, end, label]]. (Part A glue.)"""
    return [s.to_list() for s in segments]