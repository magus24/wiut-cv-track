"""solid_line_crossing event detector (PHASE 19).

Definition: a VEHICLE's bottom-center trajectory actually crosses a configured
solid line (from one clearly-defined side to the other), not merely touches or
approaches it. Pure geometry + trajectory fact, computed strictly causally
from the previous valid position and the current position of each track.

PRINCIPLE
  YOLO -> ByteTrack -> MotionEngine -> bottom-center trajectory
       -> Scene Geometry.solid_lines -> segment intersection
       -> side-change validation (+ speed / quality / born-after / cooldown)
       -> TemporalEventEngine -> "solid_line_crossing"

CROSSING DEFINITION (per vehicle track, per solid line)
  * p_prev (previous valid bottom-center) and p_curr (current one) must form a
    segment that INTERSECTS the solid line SEGMENT (geometry.segments_intersect,
    not an infinite-line distance test);
  * the vehicle must really move from one side of the line to the other. Sides
    are epsilon-aware: within `jitter_epsilon_px` of the line a point reports
    side 0 (the band). A crossing fires either directly (both endpoints clearly
    on opposite non-zero sides) or as a band pass-through: after the vehicle
    entered the band it re-emerges clearly on the side OPPOSITE to the last
    clear side (`clean_side`). Bouncing back onto the same side is "jitter",
    never a crossing; movement ALONG the line gives no side change -> no event.

PROTECTIONS
  * speed gate  : MotionState speed >= min_crossing_speed_px_s (a stationary
                  vehicle never crosses; perspective-safe, configurable).
  * quality     : MotionState quality >= min_quality.
  * born-after  : a track needs a VALID previous position on the other side;
                  appearing already beyond the line and driving on -> NO event.
  * track gap   : previous position invalidates after max_track_gap_sec
                  (track-id reuse is safe); a time regression is treated as a
                  discontinuity (track state + engine reset, no crash).
  * jitter      : epsilon side + one stable crossing per pass.
  * cooldown    : per (vehicle, line) `crossing_cooldown_sec`; a re-crossing
                  after the cooldown and a real side reversal may become a new
                  event, one pass never fires repeatedly.
  * endpoint    : a crossing whose intersection point is within
                  `endpoint_epsilon_px` of a line endpoint honours the
                  `endpoint_policy` ("reject" default -> endpoint_touch;
                  "accept" -> accepted with all other gates still applied).
  * multiple    : every configured line is evaluated independently; two lines
                  crossed by one vehicle are two independent crossing records
                  (non-overlapping in time -> two segments).

EVIDENCE per accepted crossing: line_id, crossing point (deterministic segment
intersection), p_prev / p_curr, side_prev / side_curr, interpolated crossing
timestamp (t_prev + alpha*(t_curr-t_prev), alpha from the position along the
trajectory segment), speed, heading, quality, class.

TEMPORAL: one TemporalEventEngine instance per track id (label
"solid_line_crossing"), so two vehicles crossing independently produce two
events. After an accepted crossing the evidence stays ON for a short
`post_crossing_evidence_window_sec` window (unless the vehicle rolls back onto
the approach side) -> a short segment around the crossing, never a multi-second
event. Deterministic (sorted iteration, no randomness); strictly causal; no
traffic-light state, no optical flow, no heavy ML.

ROAD GEOMETRY: not required - a configured solid line is already valid road
geometry. Missing solid_lines -> gracefully 0 events with reason
"no_solid_line_geometry" (never an error). No new geometry is ever created.
"""

from __future__ import annotations

from ..scene.geometry import segments_intersect
from .temporal import EventSegment, TemporalEventEngine

LABEL = "solid_line_crossing"
DEFAULT_VEHICLE_LABELS = ("car", "truck", "bus", "motorcycle")

# canonical ordered rejection reasons (CSV / debug reports order-friendly)
REASONS = ("no_solid_line_geometry", "invalid_vehicle", "no_motion_state",
           "no_previous_position", "track_gap", "low_quality", "stationary",
           "no_segment_intersection", "no_side_change", "jitter",
           "endpoint_touch", "cooldown")

# per-line reason precedence when reporting the overall track reason
_LINE_PREC = {r: i for i, r in enumerate(
    ("endpoint_touch", "cooldown", "jitter", "no_side_change",
     "no_segment_intersection"))}


def _side(p, a, b, eps: float) -> int:
    """Epsilon-aware side of point `p` relative to oriented line a->b.

    Returns +1 / -1 when the perpendicular distance is >= `eps`, else 0.
    Cross product sign (B-A)x(P-A) defines the side; movement along the line
    keeps the sign and is never a crossing.
    """
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    denom = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
    if denom <= 0.0:
        return 0
    if abs(cross) / denom < eps:
        return 0
    return 1 if cross > 0.0 else -1


def _segment_intersection(p1, p2, p3, p4):
    """Deterministic segment intersection point or None (parallel/collinear)."""
    r = (p2[0] - p1[0], p2[1] - p1[1])
    s = (p4[0] - p3[0], p4[1] - p3[1])
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return None
    qp = (p3[0] - p1[0], p3[1] - p1[1])
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom
    if t < -1e-9 or t > 1 + 1e-9 or u < -1e-9 or u > 1 + 1e-9:
        return None
    return (p1[0] + t * r[0], p1[1] + t * r[1])


def _alpha(p1, p2, pt) -> float:
    denom = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    if denom <= 0.0:
        return 0.0
    d1 = ((pt[0] - p1[0]) ** 2 + (pt[1] - p1[1]) ** 2) ** 0.5
    return min(1.0, max(0.0, d1 / denom))


def _dist(p, q) -> float:
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


class SolidLineCrossingDetector:
    """Per-frame solid_line_crossing evidence for vehicle tracks.

    Signature mirrors the other detectors: `update(tracks, motion_states,
    geometry, t_sec)` -> report; `finalize()` -> confirmed "solid_line_crossing"
    segments; `reset()`. Signal-free: "signal" is always None.
    """

    def __init__(
        self,
        vehicle_labels: tuple = DEFAULT_VEHICLE_LABELS,
        min_crossing_speed_px_s: float = 10.0,
        min_quality: float = 0.2,
        max_track_gap_sec: float = 2.0,
        post_crossing_evidence_window_sec: float = 0.3,
        crossing_cooldown_sec: float = 1.0,
        jitter_epsilon_px: float = 2.0,
        endpoint_epsilon_px: float = 6.0,
        endpoint_policy: str = "reject",
        min_on_duration: float = 0.05,
        allowed_gap: float = 0.3,
        merge_gap: float = 0.7,
        min_duration: float = 0.05,
    ) -> None:
        assert min_crossing_speed_px_s > 0.0
        assert min_quality > 0.0
        assert max_track_gap_sec > 0.0
        assert post_crossing_evidence_window_sec >= 0.0
        assert crossing_cooldown_sec >= 0.0
        assert jitter_epsilon_px >= 0.0
        assert endpoint_epsilon_px >= 0.0
        assert endpoint_policy in ("reject", "accept")
        self.vehicle_labels = frozenset(vehicle_labels)
        self.candidate_labels = self.vehicle_labels
        self.min_crossing_speed = float(min_crossing_speed_px_s)
        self.min_quality = float(min_quality)
        self.max_track_gap = float(max_track_gap_sec)
        self.window = float(post_crossing_evidence_window_sec)
        self.cooldown = float(crossing_cooldown_sec)
        self.jitter_eps = float(jitter_epsilon_px)
        self.endpoint_eps = float(endpoint_epsilon_px)
        self.endpoint_policy = endpoint_policy
        self._temp = dict(min_on_duration=min_on_duration,
                          allowed_gap=allowed_gap, merge_gap=merge_gap,
                          min_duration=min_duration)
        # bounded causal state
        self._lines: list[tuple] = []           # full-res scaled ((A),(B))
        self._lines_geom = None                 # geometry the cache was built from
        self._state: dict[int, dict] = {}       # tid -> per-track state
        self._engines: dict[int, TemporalEventEngine] = {}   # tid -> engine
        self._eng_t: dict[int, float] = {}      # tid -> last t fed to its engine
        self._finished: list[EventSegment] = []  # segments flushed on prune

    # ------------------------------------------------------------------ API
    def update(self, tracks, motion_states, geometry=None, t_sec: float = 0.0) -> dict:
        """Evaluate solid_line_crossing evidence at time t_sec (strictly causal).

        Args:
            tracks:        dict {track_id -> TrackTrajectory} (position + label).
            motion_states: dict {track_id -> MotionState} (smoothed motion).
            geometry:      Geometry with solid_lines (optional -> 0 events).
            t_sec:         current time.
        Returns per-frame report and feeds the per-track temporal engines.
        """
        lines = self._ensure_lines(geometry)
        records: dict[int, dict] = {}
        accepted_any = False
        for tid in sorted(tracks.keys()):
            tr = tracks[tid]
            if tr.last is None:
                continue
            rec = self._evaluate(tid, tr, motion_states.get(tid), geometry,
                                 lines, t_sec)
            records[tid] = rec
            accepted_any = accepted_any or rec["evidence"]

        # per-track engines: evidence True while any line is actively crossing.
        # A time regression for a track (discontinuity) resets ITS engine so the
        # guarded TemporalEventEngine never sees non-monotonic timestamps.
        fed = set()
        for tid, rec in records.items():
            engine = self._engine_for(tid, t_sec)
            if self._eng_t.get(tid) is not None and \
                    t_sec < self._eng_t[tid] - 1e-9:
                engine.reset()
            engine.update(LABEL, t_sec, evidence=rec["evidence"])
            self._eng_t[tid] = t_sec
            fed.add(tid)
        for tid in list(self._engines):
            if tid not in fed:
                self._engines[tid].update(LABEL, t_sec, evidence=False)
                self._eng_t[tid] = max(self._eng_t.get(tid, 0.0), t_sec)

        self._prune(t_sec)

        active = sorted(tid for tid, rec in records.items() if rec["evidence"])
        expected = set(self._state.keys()) | set(records.keys())
        rejected = {tid: rec["reason"] for tid, rec in records.items()
                    if not rec["evidence"] and rec["reason"] is not None}
        return {"t_sec": t_sec, "evidence": bool(active), "active_tracks": active,
                "tracks": records, "track_ids": sorted(expected),
                "rejected": rejected, "signal": None}

    def finalize(self) -> list[EventSegment]:
        """Close every still-active engine and return confirmed segments."""
        for tid in list(self._engines):
            self._flush_engine(tid)
        out = sorted(self._finished, key=lambda s: (s.label, s.start, s.end))
        return out

    def reset(self) -> None:
        for tid in list(self._engines):
            self._engines[tid].reset()
        self._lines = []
        self._lines_geom = None
        self._state.clear()
        self._engines.clear()
        self._eng_t.clear()
        self._finished.clear()

    # ------------------------------------------------------------- evaluate
    def _evaluate(self, tid, tr, st, geometry, lines, t_sec: float) -> dict:
        pos = (float(tr.last.x), float(tr.last.bottom_y))
        cls = tr.label
        base = self._state.setdefault(
            tid, {"last_pos": None, "last_t": t_sec, "line": {}})
        first_t = base.get("first_t", t_sec)
        base.setdefault("first_t", t_sec)

        speed = st.speed if st is not None else 0.0
        quality = st.quality if st is not None else None
        rec = {
            "track_id": tid, "class": cls,
            "first_t": first_t, "last_t": t_sec,
            "speed": speed, "heading_deg": st.heading_deg if st else None,
            "accel": st.accel if st else None,
            "stationary": st.stationary if st else None,
            "quality": quality, "signal": None,
            "crossing_line": None, "crossing_time": None,
            "crossing_received": False, "raw_crossing": False,
            "crossings": [], "rejected": {}, "evidence": False,
            "reason": None,
        }

        if cls not in self.vehicle_labels:
            self._remember(base, pos, t_sec)
            rec["reason"] = "invalid_vehicle"
            return rec
        if st is None:
            self._remember(base, pos, t_sec)
            rec["reason"] = "no_motion_state"
            return rec
        if quality is None or quality < self.min_quality:
            self._remember(base, pos, t_sec)
            rec["reason"] = "low_quality"
            return rec

        prev_pos = base["last_pos"]
        gap = t_sec - base["last_t"]
        if t_sec < base["last_t"] - 1e-9:
            gap = self.max_track_gap + 1.0          # time regression = gap
        if prev_pos is None:
            self._remember(base, pos, t_sec)
            rec["reason"] = "no_previous_position"
            return rec
        if gap > self.max_track_gap:
            self._reset_track_state(tid, t_sec)
            base = self._state[tid]
            base["first_t"] = first_t       # keep the ORIGINAL first_t
            self._remember(base, pos, t_sec)
            rec["reason"] = "track_gap"
            return rec
        if not lines:
            self._remember(base, pos, t_sec)
            rec["reason"] = "no_solid_line_geometry"
            return rec
        if speed < self.min_crossing_speed:
            self._remember(base, pos, t_sec)
            rec["reason"] = "stationary"
            return rec

        if geometry is not None:
            rec["raw_crossing"] = bool(geometry.crosses_solid_line(prev_pos, pos))

        active_crossings: list[tuple[int, float]] = []   # (line_id, crossing_t)
        rejected_lines: dict[int, str] = {}
        for i, (a, b) in enumerate(lines):
            ls = base["line"].setdefault(i, {
                "approach_side": None, "window_end": 0.0,
                "ready_after": 0.0, "last_crossing_t": None,
                "clean_side": None})
            s_prev = _side(prev_pos, a, b, self.jitter_eps)
            s_curr = _side(pos, a, b, self.jitter_eps)
            # clean_side = last position clearly OUTSIDE the band; band points
            # (s==0) never change it, so a pass-through is resolved when the
            # vehicle re-emerges clearly on the OPPOSITE side.
            clean_before = ls["clean_side"]
            if s_curr != 0:
                ls["clean_side"] = s_curr
                if clean_before is None:
                    clean_before = s_curr   # first clear observation counts

            # continuing evidence window from a recent crossing of this line
            if ls["approach_side"] is not None and t_sec <= ls["window_end"]:
                if s_curr != 0 and s_curr != ls["approach_side"]:
                    active_crossings.append((i, ls["last_crossing_t"]))
                    continue
                ls["approach_side"] = None   # rolled back -> close the window
                ls["window_end"] = 0.0

            if not segments_intersect(prev_pos, pos, a, b):
                rejected_lines[i] = "no_segment_intersection"
                continue
            if s_prev != 0 and s_curr != 0:
                if s_prev == s_curr:
                    rejected_lines[i] = "no_side_change"
                    continue
                approach = s_prev           # clear side change: direct crossing
            elif s_curr != 0:               # leaving the band -> pass-through?
                if s_curr == clean_before:
                    rejected_lines[i] = "jitter"    # band bounce, same side
                    continue
                approach = clean_before     # exited on the opposite side
            else:                           # entering / still inside the band
                rejected_lines[i] = "jitter"
                continue
            pt = _segment_intersection(prev_pos, pos, a, b)
            if pt is None:
                rejected_lines[i] = "no_side_change"   # collinear overlap safety
                continue
            if self.endpoint_policy == "reject" and (
                    _dist(pt, a) <= self.endpoint_eps or
                    _dist(pt, b) <= self.endpoint_eps):
                rejected_lines[i] = "endpoint_touch"
                continue
            if t_sec < ls["ready_after"] - 1e-9:
                rejected_lines[i] = "cooldown"
                continue

            alpha = _alpha(prev_pos, pos, pt)
            t_cross = base["last_t"] + alpha * (t_sec - base["last_t"])
            ls["ready_after"] = t_sec + self.cooldown
            ls["window_end"] = t_sec + self.window
            ls["approach_side"] = approach
            ls["last_crossing_t"] = t_cross
            active_crossings.append((i, t_cross))
            rec["crossings"].append({
                "line_id": i, "crossing_t": round(t_cross, 6),
                "crossing_point": (round(pt[0], 3), round(pt[1], 3)),
                "x_prev": round(prev_pos[0], 3), "y_prev": round(prev_pos[1], 3),
                "x_curr": round(pos[0], 3), "y_curr": round(pos[1], 3),
                "side_prev": s_prev, "side_curr": s_curr,
                "speed": round(speed, 2),
                "heading_deg": rec["heading_deg"],
                "quality": round(quality, 3),
            })

        self._remember(base, pos, t_sec)
        if active_crossings:
            crossing_time = min(ct for _, ct in active_crossings)
            rec["crossing_line"] = next(i for i, ct in active_crossings
                                        if ct == crossing_time)
            rec["crossing_time"] = crossing_time
            rec["crossing_received"] = bool(rec["crossings"])
            rec["evidence"] = True
            rec["rejected"] = rejected_lines
            return rec

        rec["rejected"] = rejected_lines
        rec["reason"] = self._overall_reason(rejected_lines)
        return rec

    def _overall_reason(self, rejected_lines: dict) -> str:
        if not rejected_lines:
            return "no_segment_intersection"
        return min(rejected_lines.values(),
                   key=lambda r: _LINE_PREC.get(r, len(_LINE_PREC)))

    # ----------------------------------------------------------------- util
    def _remember(self, base: dict, pos, t_sec: float) -> None:
        base["last_pos"] = pos
        base["last_t"] = t_sec

    def _ensure_lines(self, geometry) -> list:
        if geometry is None or not geometry.solid_lines:
            self._lines, self._lines_geom = [], None
            return []
        if self._lines is not None and self._lines_geom is geometry:
            return self._lines
        out = []
        for line in geometry.solid_lines:
            a = (line[0][0] * geometry.sx, line[0][1] * geometry.sy)
            b = (line[1][0] * geometry.sx, line[1][1] * geometry.sy)
            out.append((a, b))
        self._lines = out
        self._lines_geom = geometry
        return out

    def _reset_track_state(self, tid: int, t_sec: float) -> None:
        st = self._state.get(tid)
        self._state[tid] = {"last_pos": st["last_pos"] if st else None,
                            "last_t": st["last_t"] if st else t_sec,
                            "first_t": st.get("first_t", t_sec) if st else t_sec,
                            "line": {}}
        if tid in self._engines:          # fresh identity -> fresh temporal run
            self._flush_engine(tid)

    def _engine_for(self, tid: int, t_sec: float) -> TemporalEventEngine:
        eng = self._engines.get(tid)
        if eng is None:
            eng = TemporalEventEngine(**self._temp)
            self._engines[tid] = eng
        return eng

    def _flush_engine(self, tid: int) -> None:
        eng = self._engines.pop(tid, None)
        if eng is not None:
            self._finished.extend(eng.finalize())
        self._eng_t.pop(tid, None)

    def _prune(self, t_sec: float) -> None:
        if t_sec < 0.0:
            return
        stale = [tid for tid, base in self._state.items()
                 if t_sec - base["last_t"] > self.max_track_gap]
        for tid in stale:
            self._state.pop(tid, None)
            self._flush_engine(tid)


def segments_to_events(segments: list[EventSegment]) -> list[list]:
    """EventSegments -> harness events [[start, end, label]]. (Part A glue.)"""
    return [s.to_list() for s in segments]