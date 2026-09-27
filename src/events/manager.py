"""EventManager - the single Part-A detector pool (per video).

Owns the shared per-frame state (scene, flag-engine tracks, trajectory, motion,
geometry, pairwise interaction) and feeds every registered detector the SAME
prepared data, exactly as the target architecture requires. One decode, one YOLO
pass, one tracker, one MotionEngine and ONE PairwiseInteractionEngine serve the
whole pool - no detector re-opens or re-processes the video (PHASE 24).

Exactly ONE provider serves each label (see `PROVIDERS`), so a phenomenon can
never be reported by two code paths and same-class segments cannot overlap by
construction. Cross-class overlaps stay untouched and are expected. There are
three kinds of provider:

  * ``"legacy"``   the flag engine (src/events/rules.py). Default for the five
    labels it actually implements (`LEGACY_LABELS`); its emitted segments are
    BYTE-IDENTICAL to the pre-refactor pipeline.
  * ``"detector"`` one of the class detectors in this package, each owning its
    own TemporalEventEngine, which turns per-frame evidence into confirmed,
    non-overlapping segments. All of them run over the trajectory / motion /
    interaction state built from the very same detections.
  * ``"none"``     no detector is allocated and the label is never predicted:
    `road_obstacle` (PHASE 21 - no detectable COCO class is debris/animal/
    fallen object), `fire_smoke` (PHASE 22 - no pixel ever reaches the event
    layer), and `stop_line` (gated - it decides on geometry alone and would
    report a signal-governed event while the signal state is UNKNOWN; see
    SIGNAL_GATED_LABELS). Being *integrated* is not the same as being
    *predicted*: a predicted class that is absent from the GT scores F1=0 AND
    joins the macro mean, which is the dominant Score_A risk.

The default production table is 5 legacy + 6 detector + 3 none = 14, and it was
chosen from measurement, not from the existence of a detector file. An A/B of
pool-ON vs pool-OFF over the 127.5 s C3905 render (Temp/opencode/p24/ab_pool.py,
14 -> 40 segments) showed the legacy five are bit-identical either way and that
the pool's entire real-footage contribution is `near_miss`; `stop_line`'s +15
segments are what the gate above removes. `near_miss` stays enabled on the
evidence available (it is one of the two classes the architecture treats as
live, alongside `accident`) but its +11 segments / 70.9 s on that clip are an
OPEN QUESTION that needs ground truth, not a settled result.

Four of the enabled detectors are additionally self-gating on the current dev
scene config, because lanes / stop lines / solid lines / u-turn zones are all
OFF until they are calibrated visually. They are wired and they run, and each one
reports an explicit rejection reason instead of a positive - e.g.
``no_stop_line_geometry``, ``turn_rule_unknown``, ``outside_u_turn_zone``,
``unknown_lane``. No state is ever faked: `get_traffic_light_state` returns
"UNKNOWN" and `red_light` requires an authoritative RED source on top of the
stop-line geometry, so it cannot invent a red light from a missing signal.

`_phase` stays the pre-existing extension point for the PHASE detectors
(`TCV_ENABLE_PHASE_DETECTORS=1`); it is empty by default and its labels
override the pool and the legacy engine, so enabling it can never double-report.

Usage (strictly causal, per video, in order):
    mgr = EventManager(scene, settings, width=W, height=H)
    mgr.step(dets, t_sec)            # every sampled frame
    events = mgr.finalize(duration)  # [[start, end, label], ...]
"""

from __future__ import annotations

import sys

from ..config.settings import Settings, settings as _default_settings
from ..postprocessing import clean_events, events_from_flags
from ..scene import Scene
from ..scene.geometry import Geometry
from ..tracking.interaction import PairwiseInteractionEngine
from ..tracking.motion import MotionEngine
from ..tracking.trajectory import Detection, TrajectoryEngine
from . import rules
from .accident import AccidentDetector
from .congestion import CongestionDetector
from .failure_to_yield import FailureToYieldDetector
from .illegal_turn import IllegalTurnDetector
from .illegal_u_turn import IllegalUTurnDetector
from .jaywalking import JaywalkingDetector
from .near_miss import NearMissDetector
from .red_light import RedLightDetector
from .road_obstacle import RoadObstacleDetector
from .solid_line_crossing import SolidLineCrossingDetector
from .stop_line import StopLineDetector
from .stopped_vehicle import StoppedVehicleDetector
from .wrong_way import WrongWayDetector

_ALL_LABELS = (
    "wrong_way", "stopped_vehicle", "congestion", "jaywalking",
    "failure_to_yield", "solid_line_crossing", "illegal_turn",
    "illegal_u_turn", "red_light", "stop_line", "accident",
    "near_miss", "road_obstacle", "fire_smoke")

# The five labels the legacy flag engine can actually raise. In
# src/events/rules.py `frame_flags` the other nine are hardcoded False and never
# written, so they are exactly the gap PHASE 24 closes.
LEGACY_LABELS = ("wrong_way", "stopped_vehicle", "congestion", "jaywalking",
                 "failure_to_yield")

# Every class detector in this package, keyed by the label it owns. Five of them
# (the ones the legacy engine also implements) are drop-in replacements that are
# only allocated when their provider is switched; the rest are the production
# path.
DETECTOR_CLASSES: dict[str, type] = {
    "accident": AccidentDetector,
    "near_miss": NearMissDetector,
    "illegal_turn": IllegalTurnDetector,
    "illegal_u_turn": IllegalUTurnDetector,
    "red_light": RedLightDetector,
    "solid_line_crossing": SolidLineCrossingDetector,
    "congestion": CongestionDetector,
    "failure_to_yield": FailureToYieldDetector,
    "jaywalking": JaywalkingDetector,
    "stopped_vehicle": StoppedVehicleDetector,
    "wrong_way": WrongWayDetector,
}

# `stop_line` is a fully-built class detector and stays REGISTERED and tested,
# but it is not allocated in production. `StopLineDetector` decides on geometry
# alone - its report hardcodes "signal": None and never consults the signal
# state - so on a busy intersection it fires on ordinary vehicles and measured
# 15 segments / 23.7 s on the 127.5 s C3905 render, all of them attributable to
# PHASE 24 (pool ON vs OFF A/B: stop_line 0 -> 15). Enabling it while
# `Geometry.get_traffic_light_state` returns "UNKNOWN" for every ROI would be
# reporting a signal-governed event with no signal behind it, which the brief
# forbids outright. It is therefore gated, not deleted: the moment a real signal
# classifier exists, moving the key back to "detector" is the whole change.
#
# This is a GATE, not an incapacity - unlike `road_obstacle`/`fire_smoke` below,
# which no input can ever satisfy. Hence it is documented separately rather than
# being quietly lumped in with them.
SIGNAL_GATED_LABELS = ("stop_line",)

# Wired, tested, reachable through the pool - but never allocated, so the label
# is never predicted. `fire_smoke` has no class detector at all (PHASE 22: no
# pixel reaches the event layer); `road_obstacle` is PHASE 21; `stop_line` is
# the signal gate described above.
NO_PROVIDER_LABELS = ("road_obstacle", "fire_smoke") + SIGNAL_GATED_LABELS
NO_PROVIDER_CLASSES: dict[str, type] = {
    "road_obstacle": RoadObstacleDetector,
    "stop_line": StopLineDetector,
}
assert set(NO_PROVIDER_CLASSES) <= set(NO_PROVIDER_LABELS)

# The detectors that consume the shared TTC engine.
_PAIRWISE_CONSUMERS = frozenset({"accident", "failure_to_yield", "near_miss"})

# Labels the legacy TCV_ENABLE_PHASE_DETECTORS flag promotes to their detector.
_PHASE_DETECTOR_LABELS = ("wrong_way", "near_miss", "illegal_turn",
                          "illegal_u_turn")

DEFAULT_PROVIDERS: dict[str, str] = {
    # legacy flag engine - byte-identical to the pre-refactor pipeline
    "wrong_way": "legacy",
    "stopped_vehicle": "legacy",
    "congestion": "legacy",
    "jaywalking": "legacy",
    "failure_to_yield": "legacy",
    # class detectors over trajectory / motion / interaction state
    "accident": "detector",
    "near_miss": "detector",
    "illegal_turn": "detector",
    "illegal_u_turn": "detector",
    "red_light": "detector",
    "stop_line": "detector",
    "solid_line_crossing": "detector",
    # provably cannot fire (PHASE 21 / PHASE 22), or gated off because the
    # signal state is UNKNOWN (stop_line) -> nothing allocated
    "road_obstacle": "none",
    "fire_smoke": "none",
    "stop_line": "none",
}

# The provider table must stay a total function over the official label set -
# that is what makes "exactly one provider per label" checkable.
assert set(DEFAULT_PROVIDERS) == set(_ALL_LABELS)
assert set(LEGACY_LABELS) == {k for k, v in DEFAULT_PROVIDERS.items()
                              if v == "legacy"}
assert set(NO_PROVIDER_LABELS) == {k for k, v in DEFAULT_PROVIDERS.items()
                                  if v == "none"}
assert set(DETECTOR_CLASSES) == {k for k, v in DEFAULT_PROVIDERS.items()
                                 if v in ("detector", "legacy")}


def _build_label_map() -> dict[type, str]:
    """Reverse map detector-class -> label, validating the registry keys.

    It is needed because `LABEL` is a module-level constant in every detector
    module, and a module-level name is NOT reachable as an attribute of an
    instance (`det.LABEL` raises AttributeError). Keying on `type(det)` also
    covers detectors appended to `_phase` from outside.

    The registry key and each module's own `LABEL` must agree, otherwise the
    provider table would silently serve the wrong label — so assert it here, at
    import time, where a mismatch is impossible to miss.
    """
    mapping: dict[type, str] = {}
    for label, cls in list(DETECTOR_CLASSES.items()) + \
            list(NO_PROVIDER_CLASSES.items()):
        module = sys.modules.get(cls.__module__)
        declared = getattr(module, "LABEL", None)
        assert declared == label, (
            f"{cls.__name__} lives in {cls.__module__} whose LABEL is "
            f"{declared!r}, but the registry says {label!r}")
        mapping[cls] = label
    return mapping


LABEL_BY_CLASS: dict[type, str] = _build_label_map()


def _label_of(detector) -> str | None:
    """The label a detector instance owns (module-level `LABEL` constant)."""
    return LABEL_BY_CLASS.get(type(detector))


def _build_detector(label: str, pairwise: PairwiseInteractionEngine):
    """Instantiate the class detector owning `label`.

    The three pairwise consumers receive the SHARED engine. That is
    behaviour-identical to a private one because `PairwiseInteractionEngine`
    holds no per-video state (src/tracking/interaction.py: `compute`/`pairs` are
    pure), and it keeps a single TTC engine for the whole pipeline.
    """
    cls = DETECTOR_CLASSES[label]
    if label in _PAIRWISE_CONSUMERS:
        return cls(pairwise=pairwise)
    return cls()


class EventManager:
    def __init__(self, scene: Scene, settings: Settings | None = None,
                 width: int | None = None, height: int | None = None):
        self.settings = settings if settings is not None else _default_settings
        self.scene = scene
        # per-video reset of the legacy flag engine (same as pre-refactor)
        rules._congestion_hold["on"] = False
        rules._congestion_hold["at"] = 0.0
        self.tracks: dict[int, rules.TrackState] = {}
        self.timestamps: list[float] = []
        self.flags_map: dict[str, list[bool]] = {k: [] for k in _ALL_LABELS}

        # ---- shared low-level state (one pass, no duplication) --------------
        self._geometry = Geometry.from_json(
            self.settings.scene_config_path,
            frame_w=width, frame_h=height)
        self._trajectory = TrajectoryEngine()
        self._motion = MotionEngine()
        self._pairwise = PairwiseInteractionEngine()

        # ---- provider resolution -------------------------------------------
        self.providers = dict(DEFAULT_PROVIDERS)
        if self.settings.enable_phase_detectors:
            # the PHASE implementations replace the legacy rule for their label
            # (near_miss / illegal_turn / illegal_u_turn are already the
            # detector provider; this additionally moves wrong_way off the
            # heading-vs-flow heuristic and onto the lane-based detector)
            for label in _PHASE_DETECTOR_LABELS:
                self.providers[label] = "detector"

        # ---- detector pool --------------------------------------------------
        # `_phase` is the pre-existing extension point; it stays empty unless
        # TCV_ENABLE_PHASE_DETECTORS=1 and its labels shadow the pool, so the
        # flag can never double-report a label.
        self._phase: list = []
        if self.settings.enable_phase_detectors:
            self._phase = [
                WrongWayDetector(),
                NearMissDetector(),
                IllegalTurnDetector(),
                IllegalUTurnDetector(),
            ]
        shadowed = {lbl for lbl in map(_label_of, self._phase) if lbl}
        pool_labels = {k for k, v in self.providers.items() if v == "detector"}
        self._pool: dict[str, object] = {
            label: _build_detector(label, self._pairwise)
            for label in sorted(pool_labels - shadowed)
        }

    # ------------------------------------------------------------------ pool
    def _active_detectors(self) -> list:
        """Every detector to run this frame: the pool, then `_phase`.

        Deterministic order (pool is built in sorted-label order, `_phase` in
        append order) so two runs on the same input see the same call order.
        """
        return [*self._pool.values(), *self._phase]

    def _detector_labels(self, dets: list | None = None) -> set[str]:
        """Labels owned by an active detector (never by the legacy engine)."""
        if dets is None:
            dets = self._active_detectors()
        return {lbl for lbl in map(_label_of, dets) if lbl}

    def audit(self) -> dict[str, str]:
        """label -> provider actually in force for this video (PHASE 24 §4)."""
        claimed = self._detector_labels()
        out = {}
        for label in _ALL_LABELS:
            if label in claimed:
                out[label] = "detector"
            else:
                out[label] = self.providers[label]
        return out

    # ------------------------------------------------------------------ step
    def step(self, detections: list[dict], t_sec: float) -> None:
        """Ingest one sampled frame (detections in FULL-RES pixels)."""
        self.timestamps.append(t_sec)

        rules.update(self.tracks, detections, t_sec, self.scene)
        flags = rules.frame_flags(self.tracks, self.scene, t_sec)
        dets = self._active_detectors()
        claimed = self._detector_labels(dets)
        for k, v in flags.items():
            # A label owned by a detector is served by that detector alone, so
            # its legacy flag column stays an all-False list of the right
            # length (keeps `zip(timestamps, flags)` aligned and leaves the
            # pre-refactor behaviour untouched for unclaimed labels).
            self.flags_map[k].append(False if k in claimed else bool(v))

        if not dets:
            return
        trajs = self._trajectory.update(
            [Detection.from_dict(d) for d in detections], t_sec)
        motion_states: dict[int, object] = {}
        for tr in trajs:
            st = self._motion.update(tr, t_sec) if tr.last is not None else None
            if st is not None:
                motion_states[tr.track_id] = st
        interactions = self._shared_interactions(dets, motion_states, t_sec)
        for det in dets:
            if getattr(det, "SHARES_PAIRWISE", False):
                # report=False: the manager discards what update() returns, so
                # the per-pair diagnostics would be built only to be dropped
                det.update(self._trajectory.tracks, motion_states,
                           self._geometry, t_sec,
                           interactions=interactions, record=False)
            else:
                det.update(self._trajectory.tracks, motion_states,
                           self._geometry, t_sec)

    def _shared_interactions(self, dets: list, motion_states: dict,
                             t_sec: float):
        """ONE PairInteraction list per frame for every pairwise consumer.

        PHASE 26. `accident` and `near_miss` both build the same (id, label,
        position, state) items from the same tracks and ran the O(n^2) pair
        sweep TWICE per frame on identical input, producing two independent but
        identical lists, and then threw both away (the manager uses only the
        evidence flag). Sharing one list is behaviour-identical because
        `PairwiseInteractionEngine` holds no per-video state and
        `compute`/`pairs` are pure.

        The class gate is the OR of the sharers' own `_relevant` predicates, so
        every pair a sharer could ever act on is still present, in the same
        order. Any disagreement about the candidate label set (a detector built
        with custom labels) disables sharing rather than risk dropping a pair.
        """
        sharers = [d for d in dets if getattr(d, "SHARES_PAIRWISE", False)]
        if len(sharers) < 2:
            return None
        if len({id(d.pairwise) for d in sharers}) != 1:
            return None
        label_sets = {frozenset(d.candidate_labels) for d in sharers}
        if len(label_sets) != 1:
            return None
        labels = label_sets.pop()
        relevants = [d._relevant for d in sharers]

        def relevant(label_a, label_b):
            return any(f(label_a, label_b) for f in relevants)

        items = []
        for tid, tr in self._trajectory.tracks.items():
            if tr.last is None or tr.label not in labels:
                continue
            items.append((tid, tr.label, (tr.last.x, tr.last.bottom_y),
                          motion_states.get(tid)))
        if len(items) < 2:
            return []
        return self._pairwise.pairs_filtered(items, t_sec, relevant)

    # -------------------------------------------------------------- finalize
    def finalize(self, duration: float) -> list[list]:
        """All providers -> segments, through the single post-processing path.

        Legacy flags become segments (`events_from_flags`: merge fragments, drop
        sub-second blips) and every detector returns its own confirmed segments
        from its TemporalEventEngine; both then pass through `clean_events`,
        which unions same-class overlaps, clamps `end <= duration` and sorts
        deterministically. Cross-class overlaps are preserved on purpose.
        """
        claimed = self._detector_labels()
        legacy_map = {k: v for k, v in self.flags_map.items() if k not in claimed}
        events = events_from_flags(
            self.timestamps, legacy_map,
            min_dur=self.settings.min_duration,
            gap_max=self.settings.gap_max)
        extra: list[list] = []
        for det in self._active_detectors():
            extra.extend(seg.to_list() for seg in det.finalize())
        return clean_events(events + extra, duration)
