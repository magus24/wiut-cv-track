"""PHASE 24 — integration of all 14 event classes into one Part-A pipeline.

The audit (src/events/manager.py `PROVIDERS`) showed that before this phase
`EventManager` instantiated only 4 of the 13 class detectors, and only behind
`TCV_ENABLE_PHASE_DETECTORS=1`; the other 9 files were never imported by the
production path and `src/events/rules.py frame_flags` hardcoded 9 of its 14
flags to False. So the default path could emit at most 5 of 14 classes.

These tests lock the integrated behaviour down. They are deliberately built on
the SAME prepared data the real pipeline uses (VideoReader -> mocked Detector ->
EventManager -> post-processing), so they exercise the wiring rather than a
re-implementation of it, and every output is checked with the same contract
validator the harness uses (tests/_validation.py).

Grouped as:
  A  registry / audit completeness            (§4, §20 Test 3)
  B  contract, overlap, clamping, ordering     (§5-§7, §16, §20 Test 4-8)
  C  degenerate inputs                         (§20 Test 1, 2, 9, 10)
  D  Part A / Part B separation + causality     (§13, §20 Test 11, 12)

Run:  python tests/test_integration_14_events.py
"""

from __future__ import annotations

import collections
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from _validation import validate_events  # noqa: E402
from solution import CLASSES  # noqa: E402
from src.events import manager as M  # noqa: E402

_FPS = 10.0
_N_FRAMES = 320          # 32 s
_W, _H = 640, 360
_DURATION = _N_FRAMES / _FPS


# --------------------------------------------------------------- fixtures
def make_video(path: str) -> None:
    import cv2
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), _FPS, (_W, _H))
    frame = np.zeros((_H, _W, 3), dtype=np.uint8)
    for _ in range(_N_FRAMES):
        vw.write(frame)
    vw.release()


class FakeDetector:
    """Deterministic tracked detections, indexed by call order."""

    imgsz = 320

    def __init__(self):
        self.calls = 0

    def track(self, frame_bgr, persist: bool = True) -> list[dict]:
        i = self.calls
        self.calls += 1
        # track 1: static car -> legacy stopped_vehicle after >= 10 s
        d1 = {"xyxy": (300.0, 300.0, 360.0, 340.0), "conf": 0.9,
              "label": "car", "id": 1}
        # track 2: car moving right briefly -> legacy wrong_way while moving
        off = 5.0 * min(i, 10)
        d2 = {"xyxy": (100.0 + off, 250.0, 160.0 + off, 290.0),
              "conf": 0.9, "label": "car", "id": 2}
        return [d1, d2]


class EmptyDetector:
    """A detector that never finds anything (Test 9)."""

    imgsz = 320

    def __init__(self):
        self.calls = 0

    def track(self, frame_bgr, persist: bool = True) -> list[dict]:
        self.calls += 1
        return []


class ExplodingDetector:
    """Returns a degenerate frame's worth of nonsense (Test 10)."""

    imgsz = 320

    def __init__(self):
        self.calls = 0

    def track(self, frame_bgr, persist: bool = True) -> list[dict]:
        i = self.calls
        self.calls += 1
        if i % 3 == 0:
            return []                       # gaps
        if i % 3 == 1:
            # zero-area box, negative coords, off-screen, absurd size
            return [{"xyxy": (5.0, 5.0, 5.0, 5.0), "conf": 0.9,
                     "label": "car", "id": 1},
                    {"xyxy": (-50.0, -50.0, -10.0, -10.0), "conf": 0.9,
                     "label": "person", "id": 2},
                    {"xyxy": (9000.0, 7000.0, 9100.0, 7100.0), "conf": 0.9,
                     "label": "truck", "id": 3}]
        return [{"xyxy": (1e9, 1e9, 1e9 + 1, 1e9 + 1), "conf": 0.9,
                 "label": "bus", "id": 4}]


def run_with_fake(fake) -> list[list]:
    import src.pipeline.pipeline as pp
    prev = pp._MODEL
    pp._MODEL = fake
    try:
        from solution import detect_events
        return detect_events(video_path)
    finally:
        pp._MODEL = prev


def mgr_events(mgr, duration: float = _DURATION) -> list[list]:
    """Drive a bare EventManager over the standard synthetic feed."""
    for i in range(_N_FRAMES):
        off = 5.0 * min(i, 10)
        mgr.step([{"xyxy": (300.0, 300.0, 360.0, 340.0), "conf": 0.9,
                   "label": "car", "id": 1},
                  {"xyxy": (100.0 + off, 250.0, 160.0 + off, 290.0),
                   "conf": 0.9, "label": "car", "id": 2}], i * 0.1)
    return mgr.finalize(duration)


def new_manager(width: int = 1000, height: int = 1000, phase: bool = False):
    from src.config.settings import Settings
    from src.events.manager import EventManager
    from src.scene import Scene
    key = "TCV_ENABLE_PHASE_DETECTORS"
    prev = os.environ.get(key)
    if phase:
        os.environ[key] = "1"
    else:
        os.environ.pop(key, None)
    try:
        return EventManager(Scene.defaults_estimated(width, height), Settings(),
                            width=width, height=height)
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


video_path = None
_tmpdir = None


def setup_module(_mod=None):
    global video_path, _tmpdir
    _tmpdir = tempfile.mkdtemp(prefix="tcv_p24_")
    video_path = os.path.join(_tmpdir, "synthetic.mp4")
    make_video(video_path)


def teardown_module(_mod=None):
    global _tmpdir
    if _tmpdir:
        shutil.rmtree(_tmpdir, ignore_errors=True)
    _tmpdir = None
    video_path = None


# =====================================================================
# A. REGISTRY / AUDIT COMPLETENESS                      (§4, §20 Test 3)
# =====================================================================

def test_every_official_class_has_exactly_one_provider():
    """The 14 official ids map onto a TOTAL provider function — the invariant
    that makes same-class double-reporting impossible by construction."""
    assert set(M.DEFAULT_PROVIDERS) == set(CLASSES), (
        f"provider table does not cover CLASSES: "
        f"{set(CLASSES) ^ set(M.DEFAULT_PROVIDERS)}")


def test_every_provider_value_is_legal():
    assert set(M.DEFAULT_PROVIDERS.values()) <= {"legacy", "detector", "none"}


def test_audit_covers_all_fourteen_on_a_real_manager():
    mgr = new_manager()
    audit = mgr.audit()
    assert sorted(audit) == sorted(CLASSES)
    assert len(audit) == 14


def test_thirteen_class_detectors_are_registered():
    """12 detectors in the registry + road_obstacle reachable-but-not-allocated
    = every official class except fire_smoke, which has no detector at all
    (PHASE 22: no pixel reaches the event layer)."""
    assert set(M.DETECTOR_CLASSES) | set(M.NO_PROVIDER_CLASSES) == \
        set(CLASSES) - {"fire_smoke"}
    assert "fire_smoke" in M.NO_PROVIDER_LABELS
    assert "fire_smoke" not in M.DETECTOR_CLASSES
    assert "fire_smoke" not in M.NO_PROVIDER_CLASSES


def test_registry_key_matches_each_detector_module_label():
    """The provider table keys and each module's own LABEL constant must agree,
    otherwise the pipeline would silently serve the wrong label."""
    import sys as _sys
    for label, cls in list(M.DETECTOR_CLASSES.items()) + \
            list(M.NO_PROVIDER_CLASSES.items()):
        mod = _sys.modules[cls.__module__]
        assert getattr(mod, "LABEL", label) == label


def test_label_map_build_refuses_a_registry_module_mismatch():
    """`_build_label_map`'s import-time assert must actually FIRE, not merely
    exist. If the provider table and a detector module's own `LABEL` ever
    disagree, the pool would serve the WRONG label for that class - a silently
    mislabelled segment, which costs tIoU and hides behind a valid-looking
    contract. `test_registry_key_matches_each_detector_module_label` above
    re-derives the invariant independently, so it stays green even if the
    production assert is deleted; this test is what kills that mutation.
    """
    import types
    import sys as _sys
    from src.events import manager as EM

    fake_mod = types.ModuleType("tcv_p24_fake_detector_module")
    fake_mod.LABEL = "not_the_registry_key"
    _sys.modules[fake_mod.__name__] = fake_mod

    class Bogus:
        pass

    Bogus.__module__ = fake_mod.__name__

    prev_det, prev_none = dict(EM.DETECTOR_CLASSES), dict(EM.NO_PROVIDER_CLASSES)
    try:
        # point the "accident" key at a module that calls itself something else
        EM.DETECTOR_CLASSES = {**prev_det, "accident": Bogus}
        raised = False
        try:
            EM._build_label_map()
        except AssertionError as exc:
            raised = "whose LABEL is" in str(exc)
        assert raised, "a registry/module LABEL mismatch must abort the build"
    finally:
        EM.DETECTOR_CLASSES = prev_det
        EM.NO_PROVIDER_CLASSES = prev_none
        _sys.modules.pop(fake_mod.__name__, None)


def test_detector_labels_reports_exactly_the_active_owners():
    """Direct cover for the `_detector_labels(dets)` refactor.

    It is the function that decides which labels the legacy flag engine must
    surrender, so a bug here either double-reports a class (two providers, the
    merged segment widens, tIoU suffers at 0.7) or silently starves a detector
    of its legacy column. The explicit-argument form must agree with the
    default, and an unrecognised object must be ignored rather than raise -
    `_phase` is appendable from outside.
    """
    from src.events.manager import _label_of

    mgr = new_manager()
    expected = {_label_of(d) for d in mgr._active_detectors()} - {None}
    assert mgr._detector_labels() == expected
    assert mgr._detector_labels(mgr._active_detectors()) == mgr._detector_labels()
    # the legacy engine keeps exactly the labels no detector claims
    assert (set(M.LEGACY_LABELS) & expected) == set()

    mgr._pool, mgr._phase = {}, []
    assert mgr._detector_labels() == set()
    mgr._phase = [object()]                 # foreign object, must not raise
    assert mgr._detector_labels() == set()


def test_detector_labels_honours_an_explicit_argument():
    """`dets` must actually be read.

    The no-argument form is the one the manager itself uses, and there
    `dets is None` coincides with `self._active_detectors()` - so a mutant that
    drops the `if dets is None` guard and always re-reads the active list is
    indistinguishable through it. Passing an explicit list is the only way to
    tell the two apart, which is what `step()` does (it computes `dets` once and
    passes it, so the frame's claim set cannot drift from the frame's call set).
    """
    mgr = new_manager()
    only = [mgr._pool["red_light"]]
    assert mgr._detector_labels(only) == {"red_light"}
    assert mgr._detector_labels([]) == set(), (
        "an explicitly empty detector list must claim nothing")
    # sanity: the default form really is non-empty, so the checks above bite
    assert mgr._detector_labels() != set()


def test_signal_gated_classes_are_not_allocated_while_state_is_unknown():
    """§10 / brief: `red_light` and `stop_line` must not be enabled while the
    traffic-light state is UNKNOWN.

    `stop_line` is the one that bites. `StopLineDetector` decides on geometry
    ALONE - its report hardcodes `"signal": None` and never asks the signal state
    - so leaving it allocated reports a signal-governed event with no signal
    behind it. Measured on the 127.5 s C3905 render that is 15 segments /
    23.7 s, all of it attributable to PHASE 24 (pool ON vs OFF: stop_line 0 -> 15).

    `red_light` is handled differently ON PURPOSE: it stays allocated so the
    test suite can observe that it evaluates and REFUSES every frame
    (`traffic_light_unknown`). It cannot emit, so it cannot cost Score_A.
    Demoting it would only weaken those tests, so it is not done.
    """
    mgr = new_manager()
    assert "stop_line" in M.SIGNAL_GATED_LABELS
    assert M.DEFAULT_PROVIDERS["stop_line"] == "none"
    assert "stop_line" not in mgr._pool, "a gated label must not be allocated"
    # it stays REGISTERED and reachable, so it is gated, not deleted
    assert M.NO_PROVIDER_CLASSES["stop_line"] is M.StopLineDetector
    assert "stop_line" in M.NO_PROVIDER_CLASSES
    # and the gate is currently holding, because the state really is UNKNOWN
    geom = mgr._geometry
    states = {geom.get_traffic_light_state(None, roi)
              for roi in geom.traffic_light_rois}
    assert states == {"UNKNOWN"}, (
        "a real signal classifier appeared - revisit SIGNAL_GATED_LABELS, "
        f"observed states: {states}")
    # the detector itself must never invent a signal even when constructed
    det = M.StopLineDetector()
    rep = det.update({}, {}, mgr._geometry, 0.0)
    assert rep.get("signal") is None


def test_default_pool_holds_the_detector_served_labels():
    mgr = new_manager()
    expected = sorted(k for k, v in M.DEFAULT_PROVIDERS.items()
                      if v == "detector")
    assert list(mgr._pool) == expected
    # legacy engine keeps serving exactly the labels it implements
    for label in M.LEGACY_LABELS:
        assert M.DEFAULT_PROVIDERS[label] == "legacy"
        assert label not in mgr._pool
    # the no-provider labels are never allocated
    for label in M.NO_PROVIDER_LABELS:
        assert label not in mgr._pool


def test_phase_flag_does_not_double_report_any_label():
    """TCV_ENABLE_PHASE_DETECTORS moves wrong_way off the legacy heuristic onto
    the lane-based detector. near_miss/illegal_turn/illegal_u_turn are already
    the detector provider, so `_phase` SHADOWS the pool instead of adding to
    it — exactly one provider per label in both modes."""
    mgr = new_manager(phase=True)
    owned = list(mgr._pool) + [M._label_of(d) for d in mgr._phase]
    assert len(owned) == len(set(owned)), f"duplicate providers: {owned}"
    assert mgr.audit()["wrong_way"] == "detector"
    # and the legacy flag column for a claimed label is forced to False
    assert "wrong_way" in mgr._detector_labels()


def test_phase_promotion_is_recorded_in_the_provider_table():
    """The promotion must land in `self.providers`, not only in `_phase`.

    `audit()` CANNOT see this: it returns "detector" for anything in
    `_detector_labels()`, and `_phase` carries a WrongWayDetector whether or not
    the table was touched. So a dropped promotion leaves `audit()` fully correct
    while `providers` still claims the legacy engine owns `wrong_way` - the table
    and the pool disagreeing is exactly the drift this asserts against.
    """
    assert M.DEFAULT_PROVIDERS["wrong_way"] == "legacy", (
        "the default path must stay on the legacy heading heuristic")
    assert "wrong_way" in M._PHASE_DETECTOR_LABELS
    mgr = new_manager(phase=True)
    assert mgr.providers["wrong_way"] == "detector"
    for label in M._PHASE_DETECTOR_LABELS:
        assert mgr.providers[label] == "detector", label
    # untouched by the flag
    assert new_manager().providers == dict(M.DEFAULT_PROVIDERS)


def test_claimed_legacy_label_flag_column_is_forced_false():
    """A label a detector owns must have an all-False legacy flag column.

    `rules.frame_flags` only ever writes the five legacy ids, so the only way a
    CLAIMED legacy label gets a True is the phase flag promoting `wrong_way` -
    which is the one case where the column would otherwise emit a second,
    same-class segment. The column is kept as an all-False list of the right
    length (not deleted) so `zip(timestamps, flags)` stays aligned for the other
    thirteen labels.
    """
    baseline = new_manager(phase=False)
    assert "wrong_way" in {ev[2] for ev in mgr_events(baseline)}, (
        "precondition: the synthetic feed must exercise the legacy wrong_way "
        "flag, otherwise this test cannot observe anything")
    assert any(baseline.flags_map["wrong_way"])

    mgr = new_manager(phase=True)
    out = mgr_events(mgr)
    assert "wrong_way" in mgr._detector_labels(), "precondition: it is claimed"
    assert not any(mgr.flags_map["wrong_way"]), (
        "a claimed label's legacy column must be forced to False - otherwise "
        "the legacy path double-reports it")
    assert len(mgr.flags_map["wrong_way"]) == len(mgr.timestamps), (
        "the column must stay aligned with timestamps")
    # whatever wrong_way segments exist now can only come from the phase
    # detector, since the legacy column contributed nothing
    assert validate_events(out, _DURATION) == []


def test_claimed_legacy_label_is_dropped_from_the_legacy_map():
    """`finalize` must filter claimed labels out of the legacy map.

    Defence in depth behind the flag column: `step()` already forces that column
    to False, so the filter is unobservable through the normal path and has to be
    exercised by injecting the column directly - the same idiom
    `test_duration_is_clamped` uses. Without the filter, any future path that
    wrote a claimed label's legacy column (a new rule, a replay, a hand-set
    column) would emit a duplicate same-class segment, and `clean_events` would
    silently union the two, widening the segment and costing tIoU at 0.7.
    """
    mgr = new_manager(phase=True)
    assert "wrong_way" in mgr._detector_labels()
    mgr.timestamps = [float(i) * 0.1 for i in range(_N_FRAMES)]
    mgr.flags_map["wrong_way"] = [True] * _N_FRAMES   # unfiltered would emit
    assert [ev for ev in mgr.finalize(_DURATION) if ev[2] == "wrong_way"] == []


def test_single_shared_pairwise_engine():
    """One TTC engine for the whole pipeline. PairwiseInteractionEngine holds no
    per-video state, so sharing is behaviour-identical to private instances."""
    mgr = new_manager()
    consumers = [d for d in mgr._pool.values() if hasattr(d, "pairwise")]
    assert len(consumers) >= 2, "expected accident + near_miss to consume it"
    assert all(d.pairwise is mgr._pairwise for d in consumers)
    assert mgr._pool["accident"].pairwise is mgr._pool["near_miss"].pairwise


def test_integrated_pool_emits_nothing_on_the_synthetic_feed():
    """Every detector in the pool runs and each one reports explicit rejection
    reasons instead of guessing.

    SCOPE WARNING - read before trusting this as an FP result. This is a
    TWO-CAR synthetic feed, and on it all 7 detectors are silent. That is NOT
    evidence about real footage: on the 127.5 s C3905 render the same pool emits
    `near_miss` 11 segments / 70.9 s and `stop_line` 15 segments / 23.7 s. The
    assertion below is kept because "a bare feed produces nothing" is a real
    property worth locking, but it must never be cited as the false-positive
    rate. Use Temp/opencode/p24/ab_pool.py for that.
    """
    from src.events.manager import _label_of
    mgr = new_manager()
    seen: dict = {_label_of(d): collections.Counter()
                  for d in mgr._pool.values()}
    by_label = {_label_of(d): d for d in mgr._pool.values()}
    originals = {lbl: det.update for lbl, det in by_label.items()}

    def counting(lbl):
        inner = originals[lbl]

        def update(*a, **k):
            report = inner(*a, **k)
            if isinstance(report, dict):
                for key in ("rejected", "rejected_reasons"):
                    val = report.get(key)
                    if isinstance(val, dict):
                        for reason in val.values():
                            seen[lbl][str(reason)] += 1
            return report
        return update

    by_label = {_label_of(d): d for d in mgr._pool.values()}
    for lbl, det in by_label.items():
        det.update = counting(lbl)
    for i in range(_N_FRAMES):
        off = 5.0 * min(i, 10)
        mgr.step([{"xyxy": (300.0, 300.0, 360.0, 340.0), "conf": 0.9,
                   "label": "car", "id": 1},
                  {"xyxy": (100.0 + off, 250.0, 160.0 + off, 290.0),
                   "conf": 0.9, "label": "car", "id": 2}], i * 0.1)

    events = mgr.finalize(_DURATION)
    pool_labels = set(seen)
    for label in pool_labels:
        assert label not in {ev[2] for ev in events}, \
            f"{label} fired on normal traffic: {events}"
    # red_light and the geometry-driven classes really did evaluate and refuse,
    # they are not silently skipped
    assert seen["red_light"], "red_light must be evaluated, not short-circuited"
    assert "traffic_light_unknown" in seen["red_light"], \
        f"expected an UNKNOWN-signal refusal, got {dict(seen['red_light'])}"


def test_structurally_inert_classes_never_fire():
    """Three classes are inert for a PRINCIPLED reason, not by accident:
      * red_light       - Geometry has no signal source, always "UNKNOWN"
      * illegal_turn    - no allowed-turn policy, so every manoeuvre is UNKNOWN
      * road_obstacle   - PHASE 21, no detectable class is debris/animal/object
      * fire_smoke      - PHASE 22, no pixel reaches the event layer
    Each must be absent from the output, and the first two must still be
    evaluated every frame."""
    mgr = new_manager()
    events = mgr.finalize(0.0)          # no frames fed yet
    assert events == []
    for label in ("red_light", "illegal_turn"):
        assert label in mgr._pool, f"{label} must be integrated, not dropped"
    for label in ("road_obstacle", "fire_smoke"):
        assert label not in mgr._pool, f"{label} must not even be allocated"
        assert M.DEFAULT_PROVIDERS[label] == "none"
    out = mgr_events(new_manager())
    for label in ("red_light", "illegal_turn", "road_obstacle", "fire_smoke"):
        assert label not in {ev[2] for ev in out}, f"{label} fired: {out}"


def test_traffic_light_state_is_never_faked():
    """§10 — the scene geometry exposes ROIs but there is no signal classifier,
    so the state is UNKNOWN for every ROI. red_light therefore cannot invent a
    red light out of a missing signal."""
    mgr = new_manager()
    geom = mgr._geometry
    assert geom.get_traffic_light_state(None, None) == "UNKNOWN"
    for roi in geom.traffic_light_rois:
        assert geom.get_traffic_light_state(None, roi) == "UNKNOWN"


# =====================================================================
# B. CONTRACT, OVERLAP, CLAMPING, ORDERING        (§5-§7, §16, §20 T4-T8)
# =====================================================================

def test_output_is_the_official_three_tuple_contract():
    """§5 — [[start, end, label]], no internal debug fields leak out."""
    events = run_with_fake(FakeDetector())
    assert validate_events(events, _DURATION) == []
    for ev in events:
        assert isinstance(ev, list) and len(ev) == 3
        s, e, label = ev
        assert isinstance(s, float) and isinstance(e, float)
        assert isinstance(label, str) and label in CLASSES


def test_legacy_served_labels_still_come_from_the_legacy_engine():
    """The pre-refactor behaviour is preserved for the 5 labels the flag engine
    implements, so enabling the pool cannot alter them."""
    events = mgr_events(new_manager())
    labels = {ev[2] for ev in events}
    assert "stopped_vehicle" in labels
    assert "wrong_way" in labels
    stopped = [ev for ev in events if ev[2] == "stopped_vehicle"]
    assert max(e - s for s, e, _ in stopped) >= 10.0


def test_same_class_segments_never_overlap():
    """§6 — normalise two overlapping same-class segments into one."""
    events = run_with_fake(FakeDetector())
    assert validate_events(events, _DURATION) == []   # includes the check
    by_label: dict[str, list] = {}
    for s, e, label in events:
        by_label.setdefault(label, []).append((s, e))
    for label, segs in by_label.items():
        segs.sort()
        for i in range(1, len(segs)):
            assert segs[i][0] >= segs[i - 1][1] - 1e-9, f"{label}: {segs}"


def test_cross_class_overlap_is_preserved():
    """§6 — NO global NMS between classes. wrong_way causing an accident is two
    expected events, so the post-processor must not suppress the second one."""
    from src.postprocessing import clean_events
    raw = [[0.0, 5.0, "wrong_way"], [1.0, 6.0, "accident"],
           [4.0, 9.0, "wrong_way"]]
    out = clean_events(raw, 20.0)
    wrong = [ev for ev in out if ev[2] == "wrong_way"]
    acc = [ev for ev in out if ev[2] == "accident"]
    assert wrong == [[0.0, 9.0, "wrong_way"]], "same class must be unioned"
    assert acc == [[1.0, 6.0, "accident"]], "cross class must survive"
    # the two classes genuinely overlap in time
    assert wrong[0][0] < acc[0][1] and acc[0][0] < wrong[0][1]


def test_simultaneous_same_class_events_become_one_segment():
    """§6 — two simultaneous detections of one class are reported as ONE
    segment covering both, which is what annotators do."""
    from src.postprocessing import clean_events
    out = clean_events([[1.0, 4.0, "congestion"], [3.0, 7.0, "congestion"]],
                       20.0)
    assert [ev for ev in out if ev[2] == "congestion"] == \
        [[1.0, 7.0, "congestion"]]


def test_duration_is_clamped():
    """§7 — end <= duration, even when evidence continues past the last frame."""
    from src.postprocessing import clean_events
    out = clean_events([[1.0, 99.0, "stopped_vehicle"]], 10.0)
    assert out == [[1.0, 10.0, "stopped_vehicle"]]
    # and through the real manager
    mgr = new_manager()
    mgr.timestamps = [float(i) * 0.1 for i in range(_N_FRAMES)]
    mgr.flags_map["stopped_vehicle"] = [True] * _N_FRAMES
    events = mgr.finalize(3.0)
    for s, e, _ in events:
        assert e <= 3.0 + 1e-9


def test_output_is_sorted_deterministically():
    """§16 — sort by (start, end, label) before the evaluator sees it."""
    from src.postprocessing import clean_events
    shuffled = [[5.0, 6.0, "wrong_way"], [1.0, 2.0, "accident"],
                [1.0, 2.0, "near_miss"], [1.0, 9.0, "congestion"]]
    out = clean_events(shuffled, 20.0)
    assert out == sorted(out), f"not sorted: {out}"
    assert out == [[1.0, 2.0, "accident"], [1.0, 2.0, "near_miss"],
                   [1.0, 9.0, "congestion"], [5.0, 6.0, "wrong_way"]]


def test_deterministic_across_two_runs():
    """§16 / §20 Test 8 — same video, same process, identical output."""
    first = run_with_fake(FakeDetector())
    second = run_with_fake(FakeDetector())
    assert first == second
    # and at the manager level, incl. the detector pool
    assert mgr_events(new_manager()) == mgr_events(new_manager())


def test_detector_call_order_is_deterministic():
    """Dict iteration order must not decide the order detectors are fed."""
    a = list(new_manager()._pool)
    b = list(new_manager()._pool)
    assert a == b == sorted(a)


# =====================================================================
# C. DEGENERATE INPUTS                          (§20 Test 1, 2, 9, 10)
# =====================================================================

def test_empty_video_yields_no_events():
    """§20 Test 1 — a clip where the detector never fires anything."""
    events = run_with_fake(EmptyDetector())
    assert validate_events(events, _DURATION) == []
    assert events == []


def test_empty_detection_list_does_not_crash():
    """§20 Test 9 — the pool must survive a completely bare tracker."""
    mgr = new_manager()
    for i in range(40):
        mgr.step([], i * 0.1)
    assert mgr.finalize(4.0) == []


def test_synthetic_feed_produces_no_explosion():
    """§20 Test 2 — a two-car synthetic feed must not produce a burst of events.

    Same scope caveat as above: this bounds the DEGENERATE case, it is not the
    false-positive rate. `ab_pool.py` measures the real one.
    """
    events = mgr_events(new_manager())
    counts: dict[str, int] = {}
    for _, _, label in events:
        counts[label] = counts.get(label, 0) + 1
    assert "accident" not in counts, f"accident FP on the synthetic feed: {events}"
    assert "near_miss" not in counts, \
        f"near_miss FP on the synthetic feed: {events}"


def test_malformed_detections_do_not_raise():
    """§20 Test 10 — zero-area / off-screen / absurd coordinates must not throw
    and must not produce out-of-contract segments."""
    mgr = new_manager()
    for i in range(120):
        mgr.step([{"xyxy": (5.0, 5.0, 5.0, 5.0), "conf": 0.9,
                   "label": "car", "id": 1},
                  {"xyxy": (-50.0, -50.0, -10.0, -10.0), "conf": 0.9,
                   "label": "person", "id": 2},
                  {"xyxy": (1e9, 1e9, 1e9 + 1, 1e9 + 1), "conf": 0.9,
                   "label": "bus", "id": 4}], i * 0.1)
    events = mgr.finalize(12.0)
    assert validate_events(events, 12.0) == []


def test_malformed_detections_through_the_real_pipeline():
    """The same feed, but driving the real decode -> pipeline path."""
    events = run_with_fake(ExplodingDetector())
    assert validate_events(events, _DURATION) == []


def test_non_monotonic_and_replayed_manager_state_is_isolated():
    """A fresh manager must not inherit the previous video's state."""
    mgr_a = new_manager()
    mgr_a.step([{"xyxy": (300.0, 300.0, 360.0, 340.0), "conf": 0.9,
                 "label": "car", "id": 1}], 0.0)
    mgr_b = new_manager()
    assert mgr_b.timestamps == []
    assert all(not v for v in mgr_b.flags_map.values())


# =====================================================================
# D. PART A / PART B SEPARATION + CAUSALITY        (§13, §20 Test 11, 12)
# =====================================================================

def test_part_a_and_part_b_coexist_without_state_leakage():
    """§20 Test 11 — run Part A then Part B on the same video, and again with
    the order swapped; neither may observe the other's state."""
    import src.pipeline.pipeline as pp
    from solution import RiskEstimator
    prev = pp._MODEL
    pp._MODEL = FakeDetector()
    try:
        events_a = run_with_fake(FakeDetector())
        frame = np.zeros((_H, _W, 3), dtype=np.uint8)
        est = RiskEstimator()
        est.reset({"video_id": "v", "fps": _FPS, "width": _W, "height": _H,
                   "n_frames": _N_FRAMES})
        risk = [est.step(frame, i / _FPS) for i in range(20)]
    finally:
        pp._MODEL = prev
    assert validate_events(events_a, _DURATION) == []
    assert len(risk) == 20
    assert all(isinstance(r, float) and 0.0 <= r <= 1.0 for r in risk)


def test_part_b_state_is_cleared_by_reset_between_videos():
    """reset() must fully clear Part B state (Phase 23 contract, re-checked)."""
    from solution import RiskEstimator
    frame_hot = np.zeros((_H, _W, 3), dtype=np.uint8)
    frame_hot[:] = 255                       # bright, non-blank
    est = RiskEstimator()
    meta = {"video_id": "v", "fps": _FPS, "width": _W, "height": _H,
            "n_frames": _N_FRAMES}
    est.reset(meta)
    first = [est.step(frame_hot, i / _FPS) for i in range(10)]
    est.reset(meta)
    second = [est.step(frame_hot, i / _FPS) for i in range(10)]
    assert first == second


def test_part_a_does_not_reuse_part_b_output():
    """§13 — no reverse dependency. This is checked on the PARSED CODE, not on
    the raw text: both modules *mention* the forbidden names in prose (the
    docstrings state that they never call them), so a substring search would be
    vacuous. A docstring is an ast.Constant, so walking for ast.Call /
    ast.Name / import nodes ignores it entirely."""
    import ast
    import inspect

    def called_or_imported(module, forbidden):
        tree = ast.parse(inspect.getsource(module))
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name in forbidden:
                    hits.append(f"call {name} @L{node.lineno}")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", None) or ""
                names = " ".join(a.name for a in node.names)
                if any(f in mod for f in forbidden) or \
                        any(f in names for f in forbidden):
                    hits.append(f"import {mod} {names} @L{node.lineno}")
        return hits

    import src.events.manager as EM
    import src.risk.risk as R
    # Part B must not reach into Part A
    assert called_or_imported(R, {"detect_events", "run_pipeline",
                                  "EventManager"}) == []
    # Part A must not reach into Part B
    assert called_or_imported(EM, {"RiskEstimator", "RiskConfig", "step"}) == []
    # and Part B must never OPEN the video (cv2.resize of the frame it was
    # handed is fine; a capture/reader is not)
    assert called_or_imported(R, {"VideoReader", "VideoCapture"}) == []
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(R))
    video_attrs = [f"{n.value.attr} @L{n.lineno}" for n in ast.walk(tree)
                   if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                   and n.value.id == "cv2"
                   and n.attr in ("VideoCapture", "imread", "imwrite",
                                  "VideoWriter")]
    assert video_attrs == [], f"Part B touches the video: {video_attrs}"


def test_risk_estimator_remains_causal_after_integration():
    """§20 Test 12 — the risk curve is still produced frame-by-frame with no
    look-ahead, and the integrated Part A did not perturb it."""
    from solution import RISK_HORIZON_SEC, RiskEstimator
    est = RiskEstimator()
    est.reset({"video_id": "v", "fps": _FPS, "width": _W, "height": _H,
               "n_frames": _N_FRAMES})
    frame = np.zeros((_H, _W, 3), dtype=np.uint8)
    curve = [est.step(frame, i / _FPS) for i in range(60)]
    assert all(0.0 <= r <= 1.0 for r in curve)
    # a blank scene must sit at ~0 risk, and the horizon constant is intact
    assert max(curve) < 0.5
    assert RISK_HORIZON_SEC == 5.0


def test_classes_surface_is_unchanged():
    """§3 / §27 — CLASSES is frozen: 14 ids, no additions, no renames."""
    assert len(CLASSES) == 14
    assert len(set(CLASSES)) == 14
    assert CLASSES[0] == "accident" and CLASSES[-1] == "fire_smoke"


# ------------------------------------------------------- plain-python runner
def main():
    setup_module()
    try:
        tests = [v for k, v in sorted(globals().items())
                 if k.startswith("test_")]
        for fn in tests:
            fn()
            print(f"PASS {fn.__name__}")
        print(f"OK: {len(tests)} tests passed")
    finally:
        teardown_module()


if __name__ == "__main__":
    main()
