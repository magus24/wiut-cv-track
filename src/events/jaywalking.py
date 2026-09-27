"""jaywalking event detector (PHASE 15).

Definition: a PEDESTRIAN travels along the ROADWAY outside every configured
crosswalk without a legal zone. Evidence requires, per frame and per person
track (bottom-center in scene pixels, the ONLY spatial convention used):

  * on-road      : Geometry.is_on_road(pos)           (road polygon, ref-scaled)
  * off-crosswalk: Geometry.is_in_crosswalk(pos) is False. `crosswalk_margin_px`
                   ONLY widens the LEGAL zone around the SAME configured
                   crosswalk polygons (scene_config.json is never touched).
  * moving       : MotionEngine speed >= min_pedestrian_speed_px_s, or within a
                   bounded stationary_grace of a real prior movement
                   (`stationary_grace_sec`); a pedestrian that never moved on
                   the road is NOT jaywalking (people standing next to the
                   curb are normal).
  * trajectory   : MotionState quality >= min_quality and the track holds at
                   least `min_trajectory_points`.
  * persistence  : a track BORN mid-road has no observed entry, so it needs
                   `min_road_presence_sec` of continuous on-road presence
                   before it can produce evidence (single-frame road blips and
                   newly-spawned mid-road tracks must not fire).

The primary signal is the trajectory transition out of a NON-violating zone
(sidewalk / outside the road / inside a crosswalk) INTO the road outside a
crosswalk. Each track keeps a bounded, causal per-track state (first/last t,
previous road + previous crosswalk status, current violation-run start, whether
the entry was observed, last movement time, observed-entry count, last rejection
reason). The state is pruned after `max_track_gap_sec`; a track id that
reappears after that gap starts a FRESH state (no state bleed, id-reuse safe).

Evidence is the per-frame boolean input to the shared TemporalEventEngine
(label "jaywalking", same temporal defaults as the rest of the project), so
events start at the real first violation frame and are only CONFIRMED after
min_on_duration of active time; the engine merges/splits with allowed_gap /
merge_gap. Direction is NOT required: heading is only reported. Vehicles are
irrelevant: this detector is pedestrian-trajectory + geometry only.

Deterministic and strictly causal (only t <= current_time).
"""

from __future__ import annotations

from .failure_to_yield import crosswalk_distance_px
from .temporal import EventSegment, TemporalEventEngine

LABEL = "jaywalking"
DEFAULT_PEDESTRIAN_LABEL = "person"

# canonical ordered rejection reasons (CSV / debug reports order-friendly)
REASONS = ("no_road_geometry", "off_road", "in_crosswalk", "stationary",
           "appeared_in_road", "insufficient_trajectory", "low_quality")


class JaywalkingDetector:
    """Per-frame jaywalking evidence for pedestrian tracks.

    Signature mirrors the other detectors: `update(tracks, motion_states,
    geometry, t_sec)` -> report; `finalize()` -> confirmed segments; `reset()`.
    Geometry is REQUIRED (road + crosswalk queries); if the road polygon is
    missing the detector degrades gracefully to no evidence.
    """

    def __init__(
        self,
        min_pedestrian_speed_px_s: float = 20.0,
        stationary_grace_sec: float = 1.5,
        min_road_presence_sec: float = 1.2,
        min_trajectory_points: int = 3,
        max_track_gap_sec: float = 2.0,
        crosswalk_margin_px: float = 0.0,
        min_quality: float = 0.2,
        pedestrian_label: str = DEFAULT_PEDESTRIAN_LABEL,
        temporal: TemporalEventEngine | None = None,
        min_on_duration: float = 0.6,
        allowed_gap: float = 0.6,
        merge_gap: float = 1.2,
        min_duration: float = 0.5,
    ) -> None:
        assert min_pedestrian_speed_px_s > 0.0
        assert stationary_grace_sec >= 0.0
        assert min_road_presence_sec >= 0.0
        assert min_trajectory_points >= 1
        assert max_track_gap_sec > 0.0
        assert crosswalk_margin_px >= 0.0
        self.min_ped_speed = float(min_pedestrian_speed_px_s)
        self.stationary_grace = float(stationary_grace_sec)
        self.min_road_presence = float(min_road_presence_sec)
        self.min_traj_points = int(min_trajectory_points)
        self.max_track_gap = float(max_track_gap_sec)
        self.cw_margin = float(crosswalk_margin_px)
        self.min_quality = float(min_quality)
        self.ped_label = pedestrian_label
        self.candidate_labels = frozenset({pedestrian_label})
        self.temporal = temporal if temporal is not None else TemporalEventEngine(
            min_on_duration=min_on_duration, allowed_gap=allowed_gap,
            merge_gap=merge_gap, min_duration=min_duration)
        # bounded per-track causal state, pruned by max_track_gap_sec
        self._state: dict[int, dict] = {}

    # ------------------------------------------------------------------ API
    def update(self, tracks, motion_states, geometry=None, t_sec: float = 0.0) -> dict:
        """Evaluate jaywalking evidence for every person track at time t_sec.

        Args:
            tracks:        dict {track_id -> TrackTrajectory} (position + label).
            motion_states: dict {track_id -> MotionState} (smoothed motion).
            geometry:      Geometry with a road polygon + crosswalks (required).
            t_sec:         current time (causal).
        Returns per-frame report:
            {"t_sec", "evidence", "active_tracks", "tracks": {tid: record},
             "rejected": {tid: reason}}
        and feeds ("jaywalking", evidence) into the TemporalEventEngine.
        """
        if geometry is None or not geometry.road_polygon:
            records = {}
            for tid, tr in tracks.items():
                if tr.last is None or tr.label != self.ped_label:
                    continue
                records[tid] = self._rec_base(tr, motion_states.get(tid), t_sec,
                                              reason="no_road_geometry")
            self.temporal.update(LABEL, t_sec, evidence=False)
            return {"t_sec": t_sec, "evidence": False, "active_tracks": [],
                    "tracks": records,
                    "rejected": {tid: "no_road_geometry" for tid in records}}

        records: dict[int, dict] = {}
        active: list[int] = []
        for tid, tr in tracks.items():
            if tr.last is None or tr.label != self.ped_label:
                continue
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
                "tracks": records, "rejected": rejected}

    def finalize(self) -> list[EventSegment]:
        return self.temporal.finalize()

    def reset(self) -> None:
        self.temporal.reset()
        self._state.clear()

    # ------------------------------------------------------------- evaluate
    def _evaluate(self, tr, st, geometry, t_sec: float) -> dict:
        tid = tr.track_id
        pos = (tr.last.x, tr.last.bottom_y)
        on_road = geometry.is_on_road(pos)
        cw_only = geometry.is_in_crosswalk(pos)
        if self.cw_margin > 0.0:
            d_cw = crosswalk_distance_px(geometry, pos)
            in_cw = cw_only or (d_cw is not None and d_cw <= self.cw_margin)
        else:
            in_cw = cw_only
        moving = st is not None and st.speed >= self.min_ped_speed

        s = self._state.get(tid)
        fresh = s is None
        if fresh:
            s = {"first_t": t_sec, "last_t": t_sec, "last_pos": pos,
                 "last_road": on_road, "last_cw": in_cw,
                 "run_start": None, "entry_observed": False,
                 "last_moved": (t_sec if moving else None),
                 "transitions": 0, "reason": None}
            self._state[tid] = s
        else:
            s["last_t"] = t_sec
            s["last_pos"] = pos

        if moving:
            s["last_moved"] = t_sec

        raw_violating = on_road and not in_cw
        if raw_violating:
            if s["run_start"] is None:
                # entering (or born in) the violation zone
                s["run_start"] = t_sec
                s["entry_observed"] = not fresh
                if not fresh:
                    s["transitions"] += 1
        else:
            # outside the violation zone: the running episode is over
            s["run_start"] = None

        if not on_road:
            reason = "off_road"
            evidence = False
        elif in_cw:
            reason = "in_crosswalk"
            evidence = False
        elif not moving and s["last_moved"] is None:
            reason = "stationary"
            evidence = False
        elif not moving and t_sec - s["last_moved"] > self.stationary_grace:
            reason = "stationary"
            evidence = False
        elif len(tr.points) < self.min_traj_points:
            reason = "insufficient_trajectory"
            evidence = False
        elif st is None or st.quality < self.min_quality:
            reason = "low_quality"
            evidence = False
        elif not s["entry_observed"] and s["run_start"] is not None and \
                t_sec - s["run_start"] < self.min_road_presence:
            reason = "appeared_in_road"
            evidence = False
        else:
            reason = None
            evidence = True

        if reason is not None:
            s["reason"] = reason

        rec = {
            "track_id": tid, "class": tr.label,
            "first_t": s["first_t"], "last_t": t_sec,
            "on_road": bool(on_road), "in_crosswalk": bool(in_cw),
            "prev_road": bool(s["last_road"]), "prev_in_crosswalk": bool(s["last_cw"]),
            "violating": bool(raw_violating),
            "entry_time": s["run_start"], "entry_observed": bool(s["entry_observed"]),
            "road_run_duration": (t_sec - s["run_start"]) if raw_violating and
                                 s["run_start"] is not None else None,
            "transitions": s["transitions"],
            "speed": st.speed if st is not None else None,
            "heading_deg": st.heading_deg if st is not None else None,
            "stationary": st.stationary if st is not None else None,
            "quality": st.quality if st is not None else None,
            "evidence": bool(evidence), "reason": reason,
        }
        # persist current status for the NEXT frame (previous-status reporting)
        s["last_road"] = on_road
        s["last_cw"] = in_cw
        return rec

    @staticmethod
    def _rec_base(tr, st, t_sec, reason=None) -> dict:
        return {
            "track_id": tr.track_id, "class": tr.label,
            "first_t": t_sec, "last_t": t_sec,
            "on_road": False, "in_crosswalk": False,
            "prev_road": False, "prev_in_crosswalk": False,
            "violating": False, "entry_time": None, "entry_observed": False,
            "road_run_duration": None, "transitions": 0,
            "speed": st.speed if st is not None else None,
            "heading_deg": st.heading_deg if st is not None else None,
            "stationary": st.stationary if st is not None else None,
            "quality": st.quality if st is not None else None,
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


def segments_to_events(segments: list[EventSegment]) -> list[list]:
    """EventSegments -> harness events [[start, end, label]]. (Part A glue.)"""
    return [s.to_list() for s in segments]