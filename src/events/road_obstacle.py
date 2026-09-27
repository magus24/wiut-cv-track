"""road_obstacle event detector (PHASE 21).

Definition (official, HACKATHON_CONTEXT.md:58):
    `road_obstacle` = DEBRIS / ANIMAL / FALLEN OBJECT on the carriageway.
It is explicitly NOT `stopped_vehicle` (a vehicle that has stopped) and NOT
`jaywalking` (a pedestrian on the carriageway outside a crossing).

WHY THE DEFAULT CLASS SET IS EMPTY — read this before "fixing" it
-----------------------------------------------------------------
The label universe of this pipeline is decided by the class filter in
`src/detection/detector.py`:

    COCO_VEHICLE_CLASSES = {2: car, 3: motorcycle, 5: bus, 6: truck, 7: truck}
    COCO_PERSON = 0
    if cls not in COCO_VEHICLE_CLASSES and cls != COCO_PERSON: continue

so the ONLY labels that can ever reach a detector are
    {car, motorcycle, bus, truck, person}
(there is no "obstacle" class; COCO_BICYCLE=1 is defined but dropped by that
filter). NOT ONE of those five is debris / animal / fallen object.

Mapping them onto `road_obstacle` would be a class-confusion double penalty
(PLAN.md:18): a standing vehicle is `stopped_vehicle`, a pedestrian on the
carriageway is `jaywalking`, both already implemented and already in CLASSES.
Therefore `DEFAULT_OBSTACLE_LABELS` is EMPTY and this detector is a provable
NO-OP on the current pipeline: it can emit zero events, by construction,
regardless of traffic. That is the evidence-supported conservative result for
this phase (no samples exist - `video/` is empty; the only hand-made draft,
`labels_draft/C3905.draft.json`, contains no road_obstacle and no obstacle
class).

The detector is nevertheless complete and correct, not a stub: if a model that
emits a genuine obstruction class is ever plugged in, pass its label(s) via
`obstacle_labels` (or TCV_ROAD_OBSTACLE_LABELS) and the full evidence chain
below activates with no other change.

PRINCIPLE
    YOLO -> ByteTrack -> MotionEngine -> road/exclusion geometry
         -> persistent stationary episode -> per-obstacle TemporalEventEngine
         -> "road_obstacle"

CLASS GATE (the primary, and deliberately strict, filter)
    A track is only ever a candidate if `tr.label in obstacle_labels`.
    `vehicle_labels` is checked FIRST and hard-excluded, so a vehicle can never
    become a road_obstacle by accident — a standing car is `stopped_vehicle`.
    `person` is excluded by default for the same reason (it is `jaywalking` /
    `failure_to_yield` territory); it can be opted into explicitly, at the cost
    of a known class-confusion penalty.

OBSTRUCTION EVIDENCE (all must hold, per frame)
    * on the drivable roadway          -> `outside_road` otherwise
    * not in an exclusion zone         -> `in_exclusion` otherwise
    * judged motion (MotionState)      -> `no_motion_state` otherwise
    * quality >= min_quality           -> `low_quality` otherwise
    * speed < stationary_speed_px_s    -> `moving` / `slow_not_stationary`
    * bottom-center within
      stationary_anchor_distance_px of the episode anchor
                                       -> `unstable_position` otherwise
  Crosswalk membership and lane id are recorded as CONTEXT only and never gate
  the event (debris lying in a crossing is still an obstruction).
  The traffic-light state is never read: `signal` is always None.

EPISODE (bounded causal state, identical shape to stopped_vehicle)
  An episode starts at the first stationary frame and survives short
  non-stationary frames while the time since the last stationary frame stays
  within `allowed_stationary_gap_sec` and the box stayed near the anchor
  (one jitter frame, a tiny creep). It ENDS when the object really moves away,
  leaves the road, enters an exclusion zone, drifts past the anchor, or its id
  was absent for more than `max_track_gap_sec` (`track_gap` -> the engine is
  flushed, so a recycled id can never inherit state). Episode duration =
  t - first_stationary_time; the event START is that real first stationary
  frame, never the confirmation moment.

TEMPORAL
  One TemporalEventEngine per obstacle track, so two obstacles give two
  independent events and one obstacle never gives overlapping ones. The
  engine's min_on_duration is pinned at episode start to
  `min_obstacle_duration_sec` (+ `born_obstacle_grace_sec` when the track was
  never seen moving, which protects against a fragmentary tracker blob born
  already-stationary). Simultaneous same-class segments are unioned by the
  shared `clean_events` post-processing step, not by a private merge here.

REPORT
  `update()` returns one record per track. A track that passed the class gate
  gets the full candidate record (motion, geometry, episode fields). A track
  rejected by the class gate — which is every track in the default
  configuration — gets a compact record carrying the identity fields plus
  `reason`; its obstacle-only fields would always be None/False, and allocating
  them for the whole scene was measurably the dominant per-frame cost.

Deterministic (sorted iteration, no randomness), strictly causal, no new
tracker, no new model, no coupling to any other detector (in particular it
never consults the congestion or stopped_vehicle detectors).
"""

from __future__ import annotations

import math

from .temporal import EventSegment, TemporalEventEngine

LABEL = "road_obstacle"

# The complete label universe this pipeline can emit (see module docstring).
DETECTABLE_LABELS = ("car", "truck", "bus", "motorcycle", "person")
VEHICLE_LABELS = ("car", "truck", "bus", "motorcycle")

# Official class = debris / animal / fallen object. None of DETECTABLE_LABELS
# is such an object -> empty by default -> provable no-op. See module docstring.
DEFAULT_OBSTACLE_LABELS: tuple = ()

# canonical ordered rejection reasons (CSV / debug report order-friendly)
REASONS = ("not_obstacle_class", "vehicle_class_excluded", "no_valid_geometry",
           "outside_road", "in_exclusion", "no_motion_state", "low_quality",
           "track_gap", "identity_reset", "moving", "slow_not_stationary",
           "unstable_position")


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _bbox_height(point) -> float | None:
    """Height of a TrajectoryPoint's box in pixels (None if it has no box)."""
    box = getattr(point, "xyxy", None)
    if not box or len(box) < 4:
        return None
    return float(box[3]) - float(box[1])


class RoadObstacleDetector:
    """Per-frame road_obstacle evidence for obstacle-class tracks.

    Signature mirrors the other PHASE detectors:
        `update(tracks, motion_states, geometry, t_sec)` -> per-frame report
        `finalize()` -> confirmed "road_obstacle" segments
        `reset()`
    `signal` is always None (the traffic-light state is never consulted).
    """

    def __init__(
        self,
        obstacle_labels: tuple = DEFAULT_OBSTACLE_LABELS,
        vehicle_labels: tuple = VEHICLE_LABELS,
        stationary_speed_px_s: float = 6.0,
        slow_speed_px_s: float = 20.0,
        min_obstacle_duration_sec: float = 3.0,
        born_obstacle_grace_sec: float = 2.0,
        allowed_stationary_gap_sec: float = 0.8,
        max_track_gap_sec: float = 2.0,
        stationary_anchor_distance_px: float = 30.0,
        min_quality: float = 0.3,
        min_on_duration: float | None = None,
        allowed_gap: float = 0.8,
        merge_gap: float = 1.0,
        min_duration: float = 0.2,
    ) -> None:
        assert stationary_speed_px_s > 0.0
        assert slow_speed_px_s >= stationary_speed_px_s
        assert min_obstacle_duration_sec > 0.0
        assert born_obstacle_grace_sec >= 0.0
        assert allowed_stationary_gap_sec >= 0.0
        assert max_track_gap_sec > 0.0
        assert stationary_anchor_distance_px >= 0.0
        assert min_quality > 0.0
        self.obstacle_labels = frozenset(obstacle_labels)
        # a vehicle is NEVER a road_obstacle, whatever the operator configures
        self.vehicle_labels = frozenset(vehicle_labels)
        self.candidate_labels = self.obstacle_labels
        self.stationary_speed = float(stationary_speed_px_s)
        self.slow_speed = float(slow_speed_px_s)
        self.min_obstacle_duration = float(min_obstacle_duration_sec)
        self.born_obstacle_grace = float(born_obstacle_grace_sec)
        self.allowed_stationary_gap = float(allowed_stationary_gap_sec)
        self.max_track_gap = float(max_track_gap_sec)
        self.anchor_distance = float(stationary_anchor_distance_px)
        self.min_quality = float(min_quality)
        self._min_on_override = (float(min_on_duration)
                                 if min_on_duration is not None else None)
        self._temp = dict(allowed_gap=float(allowed_gap),
                          merge_gap=float(merge_gap),
                          min_duration=float(min_duration))
        # bounded causal state
        self._state: dict[int, dict] = {}        # tid -> episode/continuity
        self._engines: dict[int, TemporalEventEngine] = {}
        self._eng_min: dict[int, float] = {}
        self._eng_t: dict[int, float] = {}
        self._stats: dict[int, dict] = {}
        self._finished: list[EventSegment] = []
        self.event_info: list[dict] = []         # DEBUG: per flushed segment

    # ------------------------------------------------------------------ API
    def update(self, tracks, motion_states, geometry=None, t_sec: float = 0.0) -> dict:
        """Evaluate road_obstacle evidence at time t_sec (strictly causal).

        Args:
            tracks:        dict {track_id -> TrackTrajectory} (position + label).
            motion_states: dict {track_id -> MotionState} (smoothed motion).
            geometry:      Geometry (road / exclusion polygons; None -> 0 events).
            t_sec:         current time.
        Returns a per-frame report:
            {"t_sec", "evidence", "active_tracks", "tracks": {tid: record},
             "track_ids", "rejected": {tid: reason}, "obstacle_candidates",
             "not_applicable", "signal": None}
        and feeds ("road_obstacle", evidence) into each obstacle's engine.
        """
        records: dict[int, dict] = {}
        for tid in sorted(tracks.keys()):
            tr = tracks[tid]
            if tr.last is None:
                continue
            records[tid] = self._evaluate(tid, tr, motion_states.get(tid),
                                          geometry, t_sec)

        active = [tid for tid, rec in records.items() if rec["evidence"]]
        active.sort()

        # ---- per-obstacle engines -----------------------------------------
        # Only an *applicable* (obstacle-class) track is ever fed. With the
        # default (empty) obstacle-label set NO track is applicable, so the
        # detector allocates no engine at all and its per-frame cost is two
        # frozenset lookups per track. A track that stops being an obstruction
        # (or disappears) is flushed by _gate_record / _prune, and the second
        # loop below feeds the `evidence=False` frame that lets an open run
        # close inside its allowed_gap.
        fed: set = set()
        for tid, rec in records.items():
            if not rec["_applicable"]:
                continue
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
                "obstacle_candidates": sum(1 for r in records.values()
                                           if r["_applicable"]),
                "not_applicable": sum(1 for r in records.values()
                                      if not r["_applicable"]),
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
                "episode_start": None, "last_stationary_t": None,
                "stationary_duration": 0.0,
                "anchor": None, "born": False, "engine_min_on": None,
                "label": None, "quality": None, "lane_id": None,
                "road_state": "unknown"}

    def _evaluate(self, tid, tr, st, geometry, t_sec: float) -> dict:
        cls = tr.label
        base = self._state.get(tid)
        gap_absent = 0.0 if base is None else t_sec - base["last_seen"]
        was_gap = base is not None and gap_absent > self.max_track_gap + 1e-9
        if base is None or was_gap:
            # a recycled / never-seen id proves no continuity, so no stationary
            # state may be inherited from a previous object
            if was_gap:
                self._flush_engine(tid)      # commit/drop the old episode
            base = self._new_state(t_sec)
            self._state[tid] = base
        first_t = base["first_t"]
        base["last_seen"] = t_sec

        # ---- CLASS GATE (cheapest AND strictest filter, run before any
        #      record is allocated, so a non-obstacle track costs two frozenset
        #      lookups). A track that used to be an obstruction and is now a
        #      car / person has lost its episode, hence _close_episode. ------
        if cls in self.vehicle_labels:
            return self._gate_record(base, tid, cls, t_sec, first_t,
                                     gap_absent, "vehicle_class_excluded")
        if cls not in self.obstacle_labels:
            return self._gate_record(base, tid, cls, t_sec, first_t,
                                     gap_absent, "not_obstacle_class")

        pos = (float(tr.last.x), float(tr.last.bottom_y))
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
            "bbox_height": _bbox_height(tr.last),
            "road_state": "unknown", "lane_id": None,
            "in_exclusion": False, "in_crosswalk": False,
            "episode_start": base["episode_start"],
            "born_obstacle": base["born"],
            "stationary_duration": base["stationary_duration"] if in_episode
            else 0.0,
            "in_episode": in_episode, "noise_gap": False,
            "track_absent_gap": round(gap_absent, 3),
            "_applicable": True, "evidence": False, "reason": None,
        }

        # ---- continuity gates ---------------------------------------------
        if was_gap:
            return self._reject(base, rec, pos, t_sec, "track_gap")
        if t_sec < base["last_t"] - 1e-9:
            self._close_episode(base)
            base["first_t"] = t_sec
            base["seen_moving"] = False
            rec["first_t"] = t_sec
            return self._reject(base, rec, pos, t_sec, "identity_reset")
        if geometry is None or not getattr(geometry, "road_polygon", None):
            # no road polygon -> nothing can be judged to be on the roadway
            return self._reject(base, rec, pos, t_sec, "no_valid_geometry")

        # ---- road / exclusion context (never a traffic-light read) --------
        road_state = ("on_road" if geometry.is_on_road(pos) else "off_road")
        rec["road_state"] = road_state
        if geometry.lanes:
            rec["lane_id"] = geometry.get_lane(pos)
        if geometry.crosswalks and geometry.is_in_crosswalk(pos):
            rec["in_crosswalk"] = True
        if geometry.exclusion_regions and geometry.is_in_exclusion(pos):
            rec["in_exclusion"] = True
        base["lane_id"] = rec["lane_id"]
        base["road_state"] = road_state

        if road_state == "off_road":
            return self._reject(base, rec, pos, t_sec, "outside_road")
        if rec["in_exclusion"]:
            return self._reject(base, rec, pos, t_sec, "in_exclusion")
        if st is None:
            return self._reject(base, rec, pos, t_sec, "no_motion_state")
        if quality is None or quality < self.min_quality:
            return self._reject(base, rec, pos, t_sec, "low_quality")

        # ---- persistent stationary episode --------------------------------
        if is_stationary:
            if in_episode and \
                    t_sec - base["last_stationary_t"] > \
                    self.allowed_stationary_gap + 1e-9:
                self._close_episode(base)
                in_episode = False
            if not in_episode:
                self._open_episode(base, t_sec, pos)
            base["last_stationary_t"] = t_sec
            if _dist(pos, base["anchor"]) > self.anchor_distance:
                return self._reject(base, rec, pos, t_sec, "unstable_position")
            self._sync(rec, base)
            rec["evidence"] = True
        elif in_episode:
            gap_since = t_sec - base["last_stationary_t"]
            # a non-stationary frame is "noise" only while it stays near the
            # anchor; a real drive-away both moves and drifts
            near_anchor = base["anchor"] is not None and \
                _dist(pos, base["anchor"]) <= self.anchor_distance
            if gap_since <= self.allowed_stationary_gap + 1e-9 and near_anchor:
                rec["noise_gap"] = True
                self._sync(rec, base)
                rec["evidence"] = True
            else:
                base["seen_moving"] = True
                return self._reject(base, rec, pos, t_sec, self._motion_reason(
                    speed))
        else:
            base["seen_moving"] = True
            return self._reject(base, rec, pos, t_sec, self._motion_reason(speed))

        self._remember(base, pos, t_sec)
        return rec

    # ------------------------------------------------------------- internals
    def _gate_record(self, base: dict, tid: int, cls: str, t_sec: float,
                     first_t: float, gap_absent: float,
                     reason: str) -> dict:
        """Compact report record for a track the class gate rejected.

        A non-obstacle track can never contribute evidence, so it gets only
        the identity + gate fields instead of the full 22-field candidate
        record. Consumers need `reason`; the obstacle-only keys would always be
        None/False here, so omitting them carries no information.

        An engine still owned by this track is flushed: the track stopped
        being an obstruction at this frame, so its episode ends here (and any
        already-confirmed segment is committed, not lost).
        """
        self._close_episode(base)
        base["last_t"] = t_sec
        if tid in self._engines:
            self._flush_engine(tid)
        return {"track_id": tid, "class": cls, "first_t": first_t,
                "last_t": t_sec, "reason": reason, "evidence": False,
                "_applicable": False, "signal": None,
                "episode_start": None, "in_episode": False,
                "stationary_duration": 0.0, "born_obstacle": False,
                "road_state": "unknown", "lane_id": None,
                "in_exclusion": False, "in_crosswalk": False,
                "noise_gap": False, "track_absent_gap": round(gap_absent, 3)}

    def _motion_reason(self, speed: float) -> str:
        return "slow_not_stationary" if speed < self.slow_speed else "moving"

    def _sync(self, rec: dict, base: dict) -> None:
        """Copy the live episode state of `base` into the report record `rec`."""
        rec["episode_start"] = base["episode_start"]
        rec["born_obstacle"] = base["born"]
        rec["in_episode"] = base["episode_start"] is not None
        rec["stationary_duration"] = base["stationary_duration"]

    def _reject(self, base: dict, rec: dict, pos, t_sec: float,
                reason: str) -> dict:
        """Close any open episode, sync the record and attach the reason.

        Every rejection that ends an obstruction goes through here, so
        `episode_start` / `in_episode` / `stationary_duration` in the report can
        never contradict the rejection reason.
        """
        self._close_episode(base)
        self._sync(rec, base)
        self._remember(base, pos, t_sec)
        rec["reason"] = reason
        return rec

    # -------------------------------------------------------------- episode
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
        return self.min_obstacle_duration + (
            self.born_obstacle_grace if born else 0.0)

    # --------------------------------------------------------------- stats
    def _accumulate(self, tid, rec, t_sec: float) -> None:
        """Debug-only per-track tallies (never feeds a decision)."""
        speed = rec.get("speed")
        lane = rec.get("lane_id")
        dur = rec.get("stationary_duration", 0.0)
        st = self._stats.get(tid)
        if st is None:
            st = {"vehicle_id": tid, "class": rec["class"],
                  "seen_frames": 0, "stationary_frames": 0,
                  "evidence_frames": 0, "min_speed": speed,
                  "max_stationary_duration": dur,
                  "engine_max_stationary": dur,
                  "lane_id": lane, "first_seen": t_sec,
                  "last_seen": t_sec}
            self._stats[tid] = st
        st["seen_frames"] += 1
        st["last_seen"] = t_sec
        st["class"] = rec["class"]
        if lane is not None:
            st["lane_id"] = lane
        if speed is not None:
            st["min_speed"] = min(st["min_speed"], speed)
        st["max_stationary_duration"] = max(st["max_stationary_duration"], dur)
        st["engine_max_stationary"] = max(st["engine_max_stationary"], dur)
        if rec.get("stationary"):
            st["stationary_frames"] += 1
        if rec.get("evidence"):
            st["evidence_frames"] += 1

    # -------------------------------------------------------------- engine
    def _engine_min_on(self, tid: int) -> float:
        """min_on_duration of the engine for `tid`, pinned at episode start so
        a transient rejected frame can never swap the engine mid-episode."""
        base = self._state.get(tid) or {}
        min_on = base.get("engine_min_on")
        if min_on is None:
            min_on = self._eng_min.get(tid, self.min_obstacle_duration)
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
            self.event_info.append({
                "object_id": tid,
                "class": st.get("class"),
                "segments": [(s.start, s.end) for s in segs],
                "min_speed": st.get("min_speed"),
                "max_stationary_duration": st.get("engine_max_stationary", 0.0),
                "track_max_stationary_duration":
                    st.get("max_stationary_duration"),
                "lane_id": st.get("lane_id"),
            })
            st["engine_max_stationary"] = 0.0

    # ------------------------------------------------------------- pruning
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
