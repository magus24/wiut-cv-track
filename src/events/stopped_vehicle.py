"""stopped_vehicle event detector (PHASE 20).

Definition: a VEHICLE on the road stays STATIONARY for a meaningful duration,
not as part of a normal queue / congestion, not a momentary brake, not
slow-creeping or idling alongside traffic, and without a track-id break.

PRINCIPLE
  YOLO -> ByteTrack -> MotionEngine -> road/lane geometry
       -> stationary episode -> queue/congestion context
       -> per-vehicle TemporalEventEngine -> "stopped_vehicle"

STATIONARY DEFINITION (motion is NEVER recomputed here)
  Per frame a vehicle is stationary iff MotionState.speed <
  `stationary_speed_px_s` (strict, so slow-moving traffic is NOT a stop).
      speed >= stationary_speed and < slow_speed_px_s  -> slow_not_stationary
      speed >= slow_speed_px_s                          -> moving
  MotionState.quality / stationary / accel / heading_deg are used as-is.
  Track history is implicitly gated by MotionEngine (no MotionState until the
  motion window has `min_points` samples -> `no_motion_state`).

EPISODE (per vehicle, bounded causal state - never an unbounded history)
  Kept per track id: first_stationary_time / last_stationary_time /
  last_seen_time / stationary_duration / last_position / stationary_anchor /
  previous_motion_state / quality / lane_id / road_state.
  An episode starts at the first stationary frame. It survives
    * short non-stationary frames while t - last_stationary_time <=
      `allowed_stationary_gap_sec` AND the box stayed near the anchor
      (tracking flicker, one jitter frame, a tiny creep);
    * paused frames on invalid_vehicle / no_motion_state / low_quality
      (nothing is decided on a frame we cannot judge).
  The episode ENDS when
    * the vehicle really moves away (noise gap exceeded, or drifted past the
      anchor) -> moving / slow_not_stationary;
    * it leaves the road polygon -> outside_road;
    * its bottom-center drifts > `stationary_anchor_distance_px` from the
      anchor -> unstable_position (tracking break / id swap);
    * the id was absent for > `max_track_gap_sec` -> track_gap (id-reuse safe,
      no state bleed) or the timeline regresses -> identity_reset.
  Episode duration = t - first_stationary_time. Death/reset keeps the engine's
  committed segments; no future information ever leaks in.

BORN-STATIONARY
  A track whose first-ever observation is already stationary cannot know the
  true stop moment -> `born_stationary_grace_sec` is ADDED to the engine's
  minimum on-duration for that vehicle (the event start stays the true episode
  start). Once the vehicle is ever clearly observed moving (`seen_moving`),
  later episodes are normal moving -> stopped transitions and lose the grace.

QUEUE / CONGESTION PROTECTION (lightweight, self-contained; no coupling to
CongestionDetector, no "if a congestion event exists then reject")
  Only LOCAL neighbours are examined, per candidate:
    * vehicles are partitioned by lane (geometry.get_lane) with the "<none>"
      road-zone bucket as fallback when lanes are not configured;
    * each lane group is grid-binned ONCE per frame, so a candidate inspects
      only the 3x3 cell window within `queue_neighbor_radius_px`
      (near-linear, not an O(N^2) all-pairs scan).
  Counters (self is added back to the group):
    near_count      = valid neighbours (vehicle label, motion state, quality,
                      on-road) inside the radius AND in the same lane-key;
    near_stationary = neighbours with speed <= queue_max_speed_px_s;
    group_total     = near_count + 1;  group_stationary = near_stationary +
                      (1 if this frame is stationary else 0);
    ratio           = group_stationary / group_total.
  Suppressions (evidence is withheld while either is in effect):
    queue_context      = group_stationary >= queue_min_vehicle_count AND
                         ratio >= queue_stationary_ratio
                         (a plausible traffic-light / ordinary queue);
    congestion_context = group_total >= congestion_min_vehicle_count AND
                         ratio >= congestion_stationary_ratio (dense stopped
                         mass - that is congestion, not an individual stop).
  An isolated stationary vehicle (no such group) is NEVER suppressed. A queue
  that persists forever keeps individuals suppressed: a permanently frozen mass
  is congestion, not a breakdown.

TRAFFIC LIGHT / STOP LINE / CROSSWALK
  The traffic-light state is intentionally NEVER read (always UNKNOWN);
  "signal" is None. Stop-line proximity and crosswalk membership are recorded
  as context features only (stop_line_distance / in_crosswalk) and never gate
  or trigger the event.

TEMPORAL: one TemporalEventEngine instance per vehicle track (label
"stopped_vehicle"), so two stationary vehicles produce two independent events
and one vehicle never produces overlapping ones. The engine's min_on_duration
is set per vehicle to `min_stationary_duration_sec` (+ born grace) so the
EVENT START is the real first stationary frame, a single stationary frame never
confirms, and a long isolated stop becomes one [start, end] segment that closes
when the episode ends (short `allowed_gap` confirmation only).

QUALIFICATION THRESHOLD = 10 s, matching the official annotation convention
("stationary on the carriageway for 10 s or more, not in a queue at a signal").
This is a QUALIFICATION gate, not the event length: a stop that never reaches
10 s is not labelled at all, while a qualifying stop is reported over its full
[first stationary frame, last stationary frame] extent. The default was 8.0,
which admitted 2-second-short stops the annotators would not have labelled and
therefore cost precision on this class. `born_stationary_grace_sec` is separate
and unchanged: it covers a vehicle that is already stopped when it first appears,
and does not lower the 10 s bar.

Deterministic (sorted iteration, no randomness), strictly causal, signal-free,
no heavy ML, no new tracker, no pairwise engine.
"""

from __future__ import annotations

import math

from .temporal import EventSegment, TemporalEventEngine

LABEL = "stopped_vehicle"
DEFAULT_VEHICLE_LABELS = ("car", "truck", "bus", "motorcycle")

# canonical ordered rejection reasons (CSV / debug reports order-friendly)
REASONS = ("no_valid_geometry", "invalid_vehicle", "no_motion_state",
           "low_quality", "outside_road", "track_gap", "identity_reset",
           "moving", "slow_not_stationary", "unstable_position",
           "queue_context", "congestion_context",
           "insufficient_stationary_duration", "duplicate")

# 3x3 cell window around the candidate's bin
_CELL_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1),
                 (1, -1), (1, 0), (1, 1))


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _cell(pos, size: float) -> tuple[int, int]:
    return (int(pos[0] // size), int(pos[1] // size))


def _point_line_dist_px(ref_point, line, scale: float) -> float:
    """Distance (frame px) from a REF-space point to a REF-space segment."""
    a, b = line
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    px, py = ref_point[0], ref_point[1]
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        d = ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    else:
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        cx, cy = ax + t * dx, ay + t * dy
        d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
    return d * scale


class StoppedVehicleDetector:
    """Per-frame stopped_vehicle evidence for vehicle tracks.

    Signature mirrors the other detectors: `update(tracks, motion_states,
    geometry, t_sec)` -> report; `finalize()` -> confirmed "stopped_vehicle"
    segments; `reset()`. The temporal engine is per vehicle track. "signal" is
    always None (the traffic-light state is never consulted).
    """

    def __init__(
        self,
        vehicle_labels: tuple = DEFAULT_VEHICLE_LABELS,
        stationary_speed_px_s: float = 6.0,
        slow_speed_px_s: float = 20.0,
        min_stationary_duration_sec: float = 10.0,
        born_stationary_grace_sec: float = 4.0,
        allowed_stationary_gap_sec: float = 1.5,
        max_track_gap_sec: float = 2.0,
        stationary_anchor_distance_px: float = 40.0,
        min_quality: float = 0.2,
        queue_neighbor_radius_px: float = 150.0,
        queue_min_vehicle_count: int = 3,
        queue_stationary_ratio: float = 0.66,
        queue_max_speed_px_s: float = 10.0,
        congestion_min_vehicle_count: int = 6,
        congestion_stationary_ratio: float = 0.75,
        min_on_duration: float | None = None,
        allowed_gap: float = 0.5,
        merge_gap: float = 1.0,
        min_duration: float = 0.05,
    ) -> None:
        assert stationary_speed_px_s > 0.0
        assert slow_speed_px_s >= stationary_speed_px_s
        assert min_stationary_duration_sec > 0.0
        assert born_stationary_grace_sec >= 0.0
        assert allowed_stationary_gap_sec >= 0.0
        assert max_track_gap_sec > 0.0
        assert stationary_anchor_distance_px >= 0.0
        assert min_quality > 0.0
        assert queue_neighbor_radius_px > 0.0
        assert queue_min_vehicle_count >= 1
        assert 0.0 <= queue_stationary_ratio <= 1.0
        assert queue_max_speed_px_s >= 0.0
        assert congestion_min_vehicle_count >= 1
        assert 0.0 <= congestion_stationary_ratio <= 1.0
        self.vehicle_labels = frozenset(vehicle_labels)
        self.candidate_labels = self.vehicle_labels
        self.stationary_speed = float(stationary_speed_px_s)
        self.slow_speed = float(slow_speed_px_s)
        self.min_stationary_duration = float(min_stationary_duration_sec)
        self.born_stationary_grace = float(born_stationary_grace_sec)
        self.allowed_stationary_gap = float(allowed_stationary_gap_sec)
        self.max_track_gap = float(max_track_gap_sec)
        self.anchor_distance = float(stationary_anchor_distance_px)
        self.min_quality = float(min_quality)
        self.queue_radius = float(queue_neighbor_radius_px)
        self.queue_min_count = int(queue_min_vehicle_count)
        self.queue_ratio = float(queue_stationary_ratio)
        self.queue_max_speed = float(queue_max_speed_px_s)
        self.cong_min_count = int(congestion_min_vehicle_count)
        self.cong_ratio = float(congestion_stationary_ratio)
        self._min_on_override = (float(min_on_duration)
                                 if min_on_duration is not None else None)
        self._temp = dict(allowed_gap=float(allowed_gap),
                          merge_gap=float(merge_gap),
                          min_duration=float(min_duration))
        # bounded causal state
        self._state: dict[int, dict] = {}        # tid -> episode/continuity
        self._engines: dict[int, TemporalEventEngine] = {}  # tid -> engine
        self._eng_min: dict[int, float] = {}     # tid -> engine min_on_duration
        self._eng_t: dict[int, float] = {}       # tid -> last t fed to engine
        self._stats: dict[int, dict] = {}        # tid -> cumulative diagnostics
        self._finished: list[EventSegment] = []  # segments flushed on prune
        self.event_info: list[dict] = []         # DEBUG: per flushed segment

    # ------------------------------------------------------------------ API
    def update(self, tracks, motion_states, geometry=None, t_sec: float = 0.0) -> dict:
        """Evaluate stopped_vehicle evidence at time t_sec (strictly causal).

        Args:
            tracks:        dict {track_id -> TrackTrajectory} (position + label).
            motion_states: dict {track_id -> MotionState} (smoothed motion).
            geometry:      Geometry (road/lane polygons; None -> 0 events).
            t_sec:         current time.
        Returns a per-frame report:
            {"t_sec", "evidence", "active_tracks", "tracks": {tid: record},
             "track_ids", "rejected": {tid: reason}, "stationary_candidates",
             "queue_suppressed", "congestion_context", "signal": None}
        and feeds ("stopped_vehicle", evidence) into each vehicle's engine.
        """
        records: dict[int, dict] = {}
        for tid in sorted(tracks.keys()):
            tr = tracks[tid]
            if tr.last is None:
                continue
            records[tid] = self._evaluate(tid, tr, motion_states.get(tid),
                                          geometry, t_sec)

        # ---- queue / congestion context for stationary candidates ----
        if geometry is not None:
            candid = {tid: r for tid, r in records.items() if r["_suppressible"]}
            if candid:
                info = self._queue_suppressions(candid, tracks, motion_states,
                                                geometry)
                for tid, s in info.items():
                    r = records[tid]
                    r["nearby_vehicle_count"] = s["near_count"]
                    r["nearby_stationary_count"] = s["near_stationary"]
                    r["nearby_stationary_ratio"] = s["ratio"]
                    r["group_total"] = s["group_total"]
                    r["group_stationary"] = s["group_stationary"]
                    r["cluster_extent"] = s["extent"]
                    if s["congestion"]:
                        r["congestion_context"] = True
                        r["reason"] = "congestion_context"
                        r["evidence"] = False
                    elif s["queue"]:
                        r["queue_suppressed"] = True
                        r["queue_context"] = True
                        r["reason"] = "queue_context"
                        r["evidence"] = False

        active = [tid for tid, rec in records.items() if rec["evidence"]]
        active.sort()

        # ---- per-vehicle engines ----
        fed: set = set()
        for tid, rec in records.items():
            engine = self._engine_for(tid, self._engine_min_on(tid))
            if self._eng_t.get(tid) is not None and \
                    t_sec < self._eng_t[tid] - 1e-9:
                engine.reset()
            engine.update(LABEL, t_sec, evidence=rec["evidence"])
            self._eng_t[tid] = t_sec
            fed.add(tid)
            self._accumulate(tid, rec, t_sec)
        for tid in list(self._engines):
            if tid not in fed:
                eng = self._engines[tid]
                if self._eng_t.get(tid) is not None and \
                        t_sec < self._eng_t[tid] - 1e-9:
                    eng.reset()
                eng.update(LABEL, t_sec, evidence=False)
                self._eng_t[tid] = max(self._eng_t.get(tid, 0.0), t_sec)

        self._prune(t_sec)

        rejected = {tid: rec["reason"] for tid, rec in records.items()
                    if not rec["evidence"] and rec["reason"] is not None}
        expected = set(self._state.keys()) | set(records.keys())
        return {"t_sec": t_sec, "evidence": bool(active),
                "active_tracks": active, "tracks": records,
                "track_ids": sorted(expected), "rejected": rejected,
                "stationary_candidates": sum(1 for r in records.values()
                                             if r["stationary"]),
                "queue_suppressed": sum(1 for r in records.values()
                                        if r["queue_suppressed"]),
                "congestion_context": sum(1 for r in records.values()
                                          if r["congestion_context"]),
                "signal": None}

    def finalize(self) -> list[EventSegment]:
        """Close every still-active engine and return confirmed segments."""
        for tid in list(self._engines):
            self._flush_engine(tid)
        return sorted(self._finished, key=lambda s: (s.label, s.start, s.end))

    def reset(self) -> None:
        for tid in list(self._engines):
            self._engines[tid].reset()
        self._state.clear()
        self._engines.clear()
        self._eng_min.clear()
        self._eng_t.clear()
        self._stats.clear()
        self._finished.clear()
        self.event_info.clear()

    # ------------------------------------------------------------- evaluate
    def _new_state(self, t_sec: float) -> dict:
        return {"last_pos": None, "last_t": t_sec, "last_seen": t_sec,
                "seen_moving": False, "first_t": t_sec,
                "stationary_duration": 0.0, "episode_start": None,
                "last_stationary_t": None, "anchor": None, "born": False,
                "engine_min_on": None, "label": None, "quality": None,
                "lane_id": None, "road_state": "unknown"}

    def _evaluate(self, tid, tr, st, geometry, t_sec: float) -> dict:
        pos = (float(tr.last.x), float(tr.last.bottom_y))
        cls = tr.label
        base = self._state.get(tid)
        gap_absent = 0.0 if base is None else t_sec - base["last_seen"]
        was_gap = base is not None and gap_absent > self.max_track_gap + 1e-9
        if base is None or was_gap:
            # A recycled / never-seen id: continuity is NOT proven, so no
            # stationary state may be inherited from a previous object.
            if was_gap:
                self._flush_engine(tid)     # commit/drop the old episode
            base = self._new_state(t_sec)
            self._state[tid] = base
        first_t = base["first_t"]
        base["last_seen"] = t_sec

        speed = st.speed if st is not None else 0.0
        quality = st.quality if st is not None else None
        is_stationary = speed < self.stationary_speed
        in_episode = base["episode_start"] is not None

        rec = {
            "track_id": tid, "class": cls,
            "first_t": first_t, "last_t": t_sec,
            "speed": speed, "heading_deg": st.heading_deg if st else None,
            "accel": st.accel if st else None,
            "stationary": bool(is_stationary and st is not None),
            "motion_stationary": bool(st.stationary) if st else None,
            "quality": quality, "signal": None,
            "road_state": "unknown", "lane_id": None,
            "stop_line_distance": None, "in_crosswalk": False,
            "episode_start": base["episode_start"],
            "born_stationary": base["born"],
            "stationary_duration": base["stationary_duration"] if in_episode
            else 0.0,
            "in_episode": in_episode, "noise_gap": False,
            "class_changed": bool(base["label"] is not None
                                  and base["label"] != cls),
            "track_absent_gap": round(gap_absent, 3),
            "nearby_vehicle_count": 0, "nearby_stationary_count": 0,
            "nearby_stationary_ratio": 0.0, "group_total": 0,
            "group_stationary": 0, "cluster_extent": 0.0,
            "queue_suppressed": False, "queue_context": False,
            "congestion_context": False,
            "_suppressible": False, "_qpos": None, "evidence": False,
            "reason": None,
        }
        base["label"] = cls
        base["quality"] = quality

        # ---- continuity gates --------------------------------------------
        if was_gap:
            # the id was gone for longer than max_track_gap_sec
            self._remember(base, pos, t_sec)
            rec["reason"] = "track_gap"
            return rec
        regr = t_sec < base["last_t"] - 1e-9
        if regr:
            self._close_episode(base)
            base["first_t"] = t_sec
            base["seen_moving"] = False
            rec["first_t"] = t_sec
            rec["born_stationary"] = False
            self._remember(base, pos, t_sec)
            rec["reason"] = "identity_reset"
            return rec
        if geometry is None:
            self._close_episode(base)
            self._remember(base, pos, t_sec)
            rec["reason"] = "no_valid_geometry"
            return rec

        # ---- road / lane / context features (never a traffic-light read) ----
        road_state = "unknown"
        if geometry.road_polygon:
            road_state = "on_road" if geometry.is_on_road(pos) else "off_road"
        rec["road_state"] = road_state
        rec["lane_id"] = (geometry.get_lane(pos) if geometry.lanes else None)
        rec["in_crosswalk"] = bool(geometry.crosswalks and
                                   geometry.is_in_crosswalk(pos))
        if geometry.stop_lines:
            ref = geometry.to_ref(pos)
            scale = (geometry.sx + geometry.sy) / 2.0 or 1.0
            best = None
            for line in geometry.stop_lines:
                d = _point_line_dist_px(ref, line, scale)
                if best is None or d < best:
                    best = d
            rec["stop_line_distance"] = round(best, 3)
        base["lane_id"] = rec["lane_id"]
        base["road_state"] = road_state

        if road_state == "off_road":
            self._close_episode(base)
            self._remember(base, pos, t_sec)
            rec["reason"] = "outside_road"
            return rec
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

        # ---- stationary episode state machine ----
        if is_stationary:
            if in_episode and \
                    t_sec - base["last_stationary_t"] > \
                    self.allowed_stationary_gap + 1e-9:
                self._close_episode(base)
                in_episode = False
            if not in_episode:
                self._open_episode(base, t_sec, pos)
                rec["episode_start"] = base["episode_start"]
                rec["born_stationary"] = base["born"]
                rec["in_episode"] = True
            base["last_stationary_t"] = t_sec
            if _dist(pos, base["anchor"]) > self.anchor_distance:
                self._close_episode(base)
                rec.update(in_episode=False, episode_start=None,
                           born_stationary=False, stationary_duration=0.0)
                self._remember(base, pos, t_sec)
                rec["reason"] = "unstable_position"
                return rec
            base["stationary_duration"] = t_sec - base["episode_start"]
            rec["stationary_duration"] = base["stationary_duration"]
            rec["_suppressible"] = True
            rec["_qpos"] = pos
            rec["evidence"] = True
        elif in_episode:
            gap_since = t_sec - base["last_stationary_t"]
            # A non-stationary frame is only "noise" when it stays near the
            # anchor (tracking flicker / tiny creep). A real drive-away moves
            # fast AND drifts, so the episode ends immediately and the reported
            # event end is not inflated by idle continuation.
            near_anchor = base["anchor"] is not None and \
                _dist(pos, base["anchor"]) <= self.anchor_distance
            if gap_since <= self.allowed_stationary_gap + 1e-9 and near_anchor:
                rec["noise_gap"] = True
                base["stationary_duration"] = t_sec - base["episode_start"]
                rec["stationary_duration"] = base["stationary_duration"]
                rec["_suppressible"] = True
                rec["_qpos"] = pos
                rec["evidence"] = True
            else:
                self._close_episode(base)
                base["seen_moving"] = True
                self._remember(base, pos, t_sec)
                rec["reason"] = ("slow_not_stationary"
                                 if speed < self.slow_speed else "moving")
                return rec
        else:
            base["seen_moving"] = True
            self._remember(base, pos, t_sec)
            rec["reason"] = ("slow_not_stationary"
                             if speed < self.slow_speed else "moving")
            return rec

        self._remember(base, pos, t_sec)
        return rec

    # ------------------------------------------------------------- episode
    def _open_episode(self, base: dict, t_sec: float, pos) -> None:
        base["episode_start"] = t_sec
        base["last_stationary_t"] = t_sec
        base["stationary_duration"] = 0.0
        base["anchor"] = pos
        base["born"] = not base["seen_moving"]
        base["engine_min_on"] = self._min_on_for(base["born"])

    def _close_episode(self, base: dict) -> None:
        base["episode_start"] = None
        base["last_stationary_t"] = None
        base["stationary_duration"] = 0.0
        base["anchor"] = None
        base["born"] = False
        base["engine_min_on"] = None

    def _min_on_for(self, born: bool) -> float:
        if self._min_on_override is not None:
            return self._min_on_override
        return self.min_stationary_duration + (
            self.born_stationary_grace if born else 0.0)

    # --------------------------------------------------------------- queue
    def _queue_suppressions(self, candid, tracks, motion_states,
                            geometry) -> dict:
        """Local, lane-partitioned, grid-binned queue / congestion context."""
        cell = max(self.queue_radius, 1e-9)
        road_on = bool(getattr(geometry, "road_polygon", None))
        lane_on = bool(getattr(geometry, "lanes", None))
        groups: dict[str, dict] = {}

        # ONE pass: build the valid-neighbour pool, partition by lane key and
        # grid-bin it once per group (not per candidate).
        for tid in sorted(tracks.keys()):
            tr = tracks[tid]
            if tr.last is None or tr.label not in self.vehicle_labels:
                continue
            st = motion_states.get(tid)
            if st is None or st.quality is None or st.quality < self.min_quality:
                continue
            pos = (tr.last.x, tr.last.bottom_y)
            if road_on and not geometry.is_on_road(pos):
                continue
            key = (geometry.get_lane(pos) or "<none>") if lane_on else "<none>"
            g = groups.get(key)
            if g is None:
                g = groups[key] = {"items": [], "bins": {}}
            gx, gy = _cell(pos, cell)
            g["bins"].setdefault((gx, gy), []).append(len(g["items"]))
            g["items"].append({"tid": tid, "pos": pos, "speed": st.speed})

        out: dict[int, dict] = {}
        for tid in sorted(candid):
            rec = candid[tid]
            pos = rec["_qpos"]
            key = rec["lane_id"] or "<none>"
            g = groups.get(key)
            near, ext_pts = ([], [pos]) if g is None else self._neighbours(
                tid, pos, g, cell)
            near_count = len(near)
            near_stationary = sum(1 for m in near
                                  if m["speed"] <= self.queue_max_speed)
            group_total = near_count + 1
            group_stationary = near_stationary + int(bool(rec["stationary"]))
            ratio = (group_stationary / group_total) if group_total else 0.0
            xs = [p[0] for p in ext_pts]
            ys = [p[1] for p in ext_pts]
            extent = (math.hypot(max(xs) - min(xs), max(ys) - min(ys))
                      if len(ext_pts) >= 2 else 0.0)
            out[tid] = {
                "near_count": near_count,
                "near_stationary": near_stationary,
                "group_total": group_total,
                "group_stationary": group_stationary,
                "ratio": round(ratio, 4),
                "extent": round(extent, 3),
                "queue": bool(group_stationary >= self.queue_min_count and
                              ratio >= self.queue_ratio - 1e-9),
                "congestion": bool(group_total >= self.cong_min_count and
                                   ratio >= self.cong_ratio - 1e-9),
            }
        return out

    def _neighbours(self, tid, pos, group, cell):
        """Neighbours of `pos` in the same lane group, within the radius."""
        cx, cy = _cell(pos, cell)
        items = group["items"]
        idxs: list[int] = []
        for dx, dy in _CELL_OFFSETS:
            for i in group["bins"].get((cx + dx, cy + dy), ()):
                if items[i]["tid"] == tid:
                    continue
                if _dist(pos, items[i]["pos"]) <= self.queue_radius:
                    idxs.append(i)
        idxs.sort()
        near = [items[i] for i in idxs]
        ext_pts = [pos] + [m["pos"] for m in near]
        return near, ext_pts

    # --------------------------------------------------------------- stats
    def _accumulate(self, tid, rec, t_sec: float) -> None:
        st = self._stats.get(tid)
        if st is None:
            st = {"vehicle_id": tid, "class": rec["class"],
                  "seen_frames": 0, "stationary_frames": 0,
                  "evidence_frames": 0, "queue_frames": 0,
                  "cong_frames": 0, "min_speed": rec["speed"],
                  "max_stationary_duration": rec["stationary_duration"],
                  "engine_max_stationary": rec["stationary_duration"],
                  "lane_id": rec["lane_id"], "near_sum": 0.0,
                  "first_seen": t_sec, "last_seen": t_sec}
            self._stats[tid] = st
        st["seen_frames"] += 1
        st["last_seen"] = t_sec
        st["class"] = rec["class"]
        if rec["lane_id"] is not None:
            st["lane_id"] = rec["lane_id"]
        if rec["speed"] is not None:
            st["min_speed"] = min(st["min_speed"], rec["speed"])
        if rec["stationary"]:
            st["stationary_frames"] += 1
        st["max_stationary_duration"] = max(st["max_stationary_duration"],
                                            rec["stationary_duration"])
        # per-ENGINE-lifetime maximum: one engine can span several episodes
        # (a brief creep above slow_speed closes the episode, the engine then
        # merges the stops back), so the track-lifetime max would belong to a
        # different event than the one being reported.
        st["engine_max_stationary"] = max(st["engine_max_stationary"],
                                          rec["stationary_duration"])
        if rec["evidence"]:
            st["evidence_frames"] += 1
            st["near_sum"] += rec["nearby_vehicle_count"]
        if rec["queue_suppressed"] and not rec["congestion_context"]:
            st["queue_frames"] += 1
        if rec["congestion_context"]:
            st["cong_frames"] += 1

    # ------------------------------------------------------------- engine
    def _engine_min_on(self, tid: int) -> float:
        """min_on_duration of the engine for `tid`.

        While an episode is open the value is pinned at episode start (so a
        transient rejected frame can never swap the engine mid-episode and
        throw away the accumulated on-time). With no episode the current
        engine setting is kept.
        """
        base = self._state.get(tid) or {}
        min_on = base.get("engine_min_on")
        if min_on is None:
            min_on = self._eng_min.get(tid, self.min_stationary_duration)
        return min_on

    def _engine_for(self, tid: int, min_on: float) -> TemporalEventEngine:
        eng = self._engines.get(tid)
        cur = self._eng_min.get(tid)
        if eng is None or cur is None or abs(cur - min_on) > 1e-9:
            if eng is not None:
                self._flush_engine(tid)
            eng = TemporalEventEngine(min_on_duration=min_on, **self._temp)
            self._engines[tid] = eng
            self._eng_min[tid] = min_on
        return eng

    def _flush_engine(self, tid: int) -> None:
        eng = self._engines.pop(tid, None)
        self._eng_min.pop(tid, None)
        self._eng_t.pop(tid, None)
        if eng is None:
            return
        segs = [s for s in eng.finalize() if s.end > s.start]
        self._finished.extend(segs)
        if segs:
            st = self._stats.get(tid) or {}
            frames = st.get("evidence_frames", 0) or 1
            self.event_info.append({
                "vehicle_id": tid,
                "class": st.get("class"),
                "segments": [(s.start, s.end) for s in segs],
                "min_speed": st.get("min_speed"),
                "max_stationary_duration": st.get("engine_max_stationary", 0.0),
                "track_max_stationary_duration":
                    st.get("max_stationary_duration"),
                "lane_id": st.get("lane_id"),
                "mean_near_count": round(st.get("near_sum", 0.0) / frames, 2),
                "queue_frames": st.get("queue_frames", 0),
                "cong_frames": st.get("cong_frames", 0),
            })
            # the next engine for this track starts a fresh measurement
            st["engine_max_stationary"] = 0.0

    # ------------------------------------------------------------ pruning
    def _prune(self, t_sec: float) -> None:
        """Drop state + flush engines of tracks gone for > max_track_gap."""
        stale = [tid for tid, base in self._state.items()
                 if t_sec - base["last_seen"] > self.max_track_gap]
        for tid in sorted(stale):
            self._state.pop(tid, None)
            self._flush_engine(tid)

    # ----------------------------------------------------------------- util
    def _remember(self, base: dict, pos, t_sec: float) -> None:
        base["last_pos"] = pos
        base["last_t"] = t_sec


def segments_to_events(segments: list[EventSegment]) -> list[list]:
    """EventSegments -> harness events [[start, end, label]]. (Part A glue.)"""
    return [s.to_list() for s in segments]
