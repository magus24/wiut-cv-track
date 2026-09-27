"""PHASE 26 - the pairwise-event optimisation must be UNOBSERVABLE.

The manager used to build the identical (id, label, position, state) item list
twice per frame and run the O(n^2) pair sweep twice, once for `accident` and
once for `near_miss`, then threw both result lists away. It now computes ONE
list, gates the pairs on the class labels alone before allocating a
PairInteraction, and skips the per-pair diagnostic records (the manager
discards them). Segments, evidence and per-pair state must be untouched.

The A/B switch is `SHARES_PAIRWISE` on the two detectors:
  True  -> the manager shares one list and passes record=False  (production)
  False -> the manager calls update() with no keyword arguments at all, which is
           the pre-PHASE-26 call signature and the unfiltered `pairs()` sweep
           (the pre-PHASE-26 behaviour).
So every test below compares production against the old code path on the same
input, and a wrong gate, a mis-ordered list or a dropped pair shows up as a
difference.

Run:  python -m pytest tests/test_pairwise_sharing.py -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

from src.events.accident import AccidentDetector  # noqa: E402
from src.events.manager import EventManager  # noqa: E402
from src.events.near_miss import NearMissDetector  # noqa: E402
from src.scene import Scene  # noqa: E402
from src.config.settings import Settings  # noqa: E402
import src.events.accident as accident_mod  # noqa: E402
import src.events.near_miss as near_miss_mod  # noqa: E402
from src.tracking.interaction import (  # noqa: E402
    PairwiseInteractionEngine, heading_difference_deg,
)

N_FRAMES = 60
DURATION = 6.0


# --------------------------------------------------------------------- feeds
def car(x, y=300.0, w=60.0, h=40.0):
    return {"xyxy": (x, y, x + w, y + h), "conf": 0.9, "label": "car", "id": 0}


def truck(x, y=300.0):
    return {"xyxy": (x, y, x + 90.0, y + 60.0), "conf": 0.9, "label": "truck",
            "id": 0}


def person(x, y=400.0):
    return {"xyxy": (x, y, x + 30.0, y + 70.0), "conf": 0.9, "label": "person",
            "id": 0}


def frames(kind="mixed"):
    """Synthetic streams. `mixed` deliberately contains person-person and
    person-bicycle pairs (irrelevant to BOTH detectors) next to pairs that can
    fire, so an over-eager gate would show up as a lost event."""
    out = []
    for i in range(N_FRAMES):
        t = i * 0.1
        if kind == "irrelevant_only":
            # a crowd crossing: 6 pedestrians, no vehicle at all
            dets = [person(100.0 + 40.0 * k, 400.0 + 10.0 * (i % 3))
                    for k in range(6)]
        elif kind == "approaching":
            # two cars closing on each other across the frame
            dets = [car(100.0 + 6.0 * i), car(900.0 - 6.0 * i)]
            dets[0]["id"] = 1
            dets[1]["id"] = 2
        elif kind == "collision":
            # head-on contact followed by both cars stopping: distance shrinks
            # 400 -> 0 (contact + pre-separation), then speed_drop / stop /
            # collocation -> a CONFIRMED accident segment
            f = min(i, 12)
            dets = [car(300.0 + 40.0 * f, 1000.0), car(700.0 - 40.0 * f, 1000.0)]
            dets[0]["id"] = 1
            dets[1]["id"] = 2
        elif kind == "near_miss":
            # head-on but 40 px apart laterally: dangerous, never touching
            f = 20.0 * min(i, 14)
            dets = [car(200.0 + f, 1000.0), car(800.0 - f, 1040.0)]
            dets[0]["id"] = 1
            dets[1]["id"] = 2
        elif kind == "second_pair":
            # the FIRING pair is (2, 3), not the first pair in id order: a
            # pedestrian holds the lowest id, so a sweep that kept only the
            # first pair would lose the accident
            f = min(i, 12)
            dets = [person(100.0, 700.0),
                    car(300.0 + 40.0 * f, 1000.0),
                    car(700.0 - 40.0 * f, 1000.0)]
            dets[0]["id"] = 1
            dets[1]["id"] = 2
            dets[2]["id"] = 3
        else:
            dets = [car(100.0 + 6.0 * i), truck(900.0 - 6.0 * i),
                    person(400.0 + 2.0 * i), person(430.0 + 2.0 * i),
                    person(460.0 + 2.0 * i)]
            dets[0]["id"] = 1
            dets[1]["id"] = 2
            dets[2]["id"] = 3
            dets[3]["id"] = 4
            dets[4]["id"] = 5
        out.append((dets, t))
    return out


ALL_KINDS = ("mixed", "approaching", "irrelevant_only", "collision", "near_miss",
             "second_pair")
FIRING = {"collision": "accident", "near_miss": "near_miss",
          "second_pair": "accident"}


def run_manager(kind, sharing):
    """Full EventManager over a stream, with the A/B switch forced."""
    mgr = EventManager(Scene.defaults_estimated(1000, 1000), Settings(),
                       width=1000, height=1000)
    old = {cls: getattr(cls, "SHARES_PAIRWISE", None)
           for cls in (AccidentDetector, NearMissDetector)}
    for cls in (AccidentDetector, NearMissDetector):
        setattr(cls, "SHARES_PAIRWISE", sharing)
    try:
        for dets, t in frames(kind):
            mgr.step(dets, t)
        return mgr.finalize(DURATION)
    finally:
        for cls, val in old.items():
            setattr(cls, "SHARES_PAIRWISE", val)


# ------------------------------------------------------------------- engine
def test_pairs_filtered_is_pairs_restricted_to_the_gate():
    """The filtered sweep returns exactly the objects `pairs` would return for
    the accepted pairs, in the same order and with identical field values."""
    eng = PairwiseInteractionEngine()
    labels = ["car", "person", "truck", "bicycle", "motorcycle", "car"]
    items = [(i + 1, labels[i], (100.0 * (i + 1), 200.0 + 10.0 * i), None)
             for i in range(6)]
    full = eng.pairs(items, 3.0)
    keep = lambda a, b: a in ("car", "truck", "motorcycle") or \
        b in ("car", "truck", "motorcycle")
    got = eng.pairs_filtered(items, 3.0, keep)
    want = [p for p in full
            if labels[p.track_id_a - 1] in ("car", "truck", "motorcycle")
            or labels[p.track_id_b - 1] in ("car", "truck", "motorcycle")]
    assert len(got) == len(want) > 0
    for a, b in zip(got, want):
        assert a == b                      # dataclass __eq__ over every slot
    # and the rejected ones really are rejected
    assert len(got) < len(full)


def test_pairs_filtered_keeps_the_full_pair_order():
    eng = PairwiseInteractionEngine()
    items = [(i + 1, "car", (50.0 * i, 0.0), None) for i in range(8)]
    assert eng.pairs_filtered(items, 1.0, lambda a, b: True) == eng.pairs(items, 1.0)


def test_pairs_filtered_rejects_everything_when_the_gate_is_false():
    eng = PairwiseInteractionEngine()
    items = [(i + 1, "car", (50.0 * i, 0.0), None) for i in range(5)]
    assert eng.pairs_filtered(items, 1.0, lambda a, b: False) == []


# ----------------------------------------------------------------- detectors
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_manager_events_identical_with_and_without_sharing(kind):
    """The headline guarantee: production output == pre-PHASE-26 output."""
    assert run_manager(kind, True) == run_manager(kind, False)


@pytest.mark.parametrize("kind,label", sorted(FIRING.items()))
def test_the_streams_actually_fire(kind, label):
    """Non-vacuity for the A/B above: an equality test over two EMPTY event
    lists proves nothing, so the pairwise detectors must really emit here."""
    events = run_manager(kind, False)
    assert label in [e[2] for e in events], \
        f"{kind} stream did not produce a {label} event: {events}"


def test_sharing_actually_changes_how_many_pairs_are_computed():
    """Non-vacuity: the A/B must reach DIFFERENT internals, otherwise the
    equality above proves nothing."""
    def n_pairs(sharing):
        mgr = EventManager(Scene.defaults_estimated(1000, 1000), Settings(),
                           width=1000, height=1000)
        old = {c: getattr(c, "SHARES_PAIRWISE", None)
               for c in (AccidentDetector, NearMissDetector)}
        for c in (AccidentDetector, NearMissDetector):
            setattr(c, "SHARES_PAIRWISE", sharing)
        built = []
        real = PairwiseInteractionEngine.pairs

        def spy(self, items, t_sec):
            out = real(self, items, t_sec)
            built.append(len(out))
            return out

        realf = PairwiseInteractionEngine.pairs_filtered

        def spyf(self, items, t_sec, relevant):
            out = realf(self, items, t_sec, relevant)
            built.append(len(out))
            return out

        PairwiseInteractionEngine.pairs = spy
        PairwiseInteractionEngine.pairs_filtered = spyf
        try:
            for dets, t in frames("mixed"):
                mgr.step(dets, t)
        finally:
            PairwiseInteractionEngine.pairs = real
            PairwiseInteractionEngine.pairs_filtered = realf
            for c, v in old.items():
                setattr(c, "SHARES_PAIRWISE", v)
        return sum(built), len(built)

    old_total, old_calls = n_pairs(False)
    new_total, new_calls = n_pairs(True)
    assert old_calls == 2 * N_FRAMES, "one sweep per pairwise detector per frame"
    assert new_calls <= N_FRAMES, "one shared sweep per frame"
    assert new_total < old_total, "the gate must actually remove pairs"


def test_record_false_keeps_evidence_and_segments():
    """`record=False` is a report switch, not a decision switch."""
    def run(record):
        det = AccidentDetector()
        tracks, states = _tracks()
        for i in range(40):
            t = i * 0.1
            kwargs = {} if record is True else {"record": False}
            r = det.update(tracks, states, None, t, **kwargs)
            assert isinstance(r["pairs"], dict)
        return ([s.to_list() for s in det.finalize()],
                det.impact_pairs_count, sorted(det.confirmed_keys))

    assert run(True) == run(False)


def test_record_false_still_reports_the_same_shape():
    det = AccidentDetector()
    tracks, states = _tracks()
    r_full = det.update(tracks, states, None, 0.0)
    det2 = AccidentDetector()
    r_lite = det2.update(tracks, states, None, 0.0, record=False)
    assert r_full.keys() == r_lite.keys()
    assert r_lite["pairs"] == {}
    assert (r_full["evidence"], r_full["impact_pairs"], r_full["confirmed_pairs"]) \
        == (r_lite["evidence"], r_lite["impact_pairs"], r_lite["confirmed_pairs"])


def test_injected_interactions_match_the_detectors_own_sweep():
    """An injected list must be interchangeable with the one the detector builds:
    same segments, same evidence, same impact bookkeeping."""
    def run(inject):
        eng = PairwiseInteractionEngine()
        det = AccidentDetector(pairwise=eng)
        tracks, states = _tracks()
        for i in range(40):
            t = i * 0.1
            if inject:
                items = det._build_items(tracks, states)
                inter = eng.pairs_filtered(items, t, det._relevant) \
                    if len(items) >= 2 else []
                det.update(tracks, states, None, t, interactions=inter,
                           record=False)
            else:
                det.update(tracks, states, None, t)
        return ([s.to_list() for s in det.finalize()], det.impact_pairs_count,
                sorted(det.confirmed_keys), sorted(det.rejected_reasons.items()))

    assert run(True) == run(False)


def test_part_b_pair_gate_yields_the_same_channels():
    """Part B (src/risk/risk.py `_observe`) now filters before building a
    PairInteraction instead of after. Reconstruct both loops over the same
    items and require bit-identical risk channels."""
    from src.risk.risk import (RiskEstimator, RiskConfig, _clamp01,
                               _pair_relevant)
    from src.tracking.motion import MotionState

    cfg = RiskConfig()
    pair_channels = RiskEstimator._pair_channels     # staticmethod by design
    eng = PairwiseInteractionEngine()
    labels = ["car", "person", "truck", "bicycle", "person", "motorcycle"]
    items = []
    for i, lbl in enumerate(labels):
        items.append((i + 1, lbl, (120.0 * (i + 1), 500.0 + 20.0 * i),
                      MotionState(0.0, 40.0 * (1 if i % 2 else -1), 5.0,
                                  40.0, 0.0, 0.0 if i % 2 else 180.0,
                                  False, 0.9)))
    widths = {i + 1: 60.0 for i in range(len(labels))}

    # old loop: build everything, then skip
    old = {"ttc": 0.0, "proximity": 0.0}
    for pr in eng.pairs(items, 2.0):
        if not _pair_relevant(getattr(pr, "class_a", None),
                              getattr(pr, "class_b", None)):
            continue
        scale = max(widths.get(pr.track_id_a, 0.0),
                    widths.get(pr.track_id_b, 0.0), 1.0)
        ttc_ch, prox_ch = pair_channels(pr, scale, cfg)
        old["ttc"] = max(old["ttc"], ttc_ch)
        old["proximity"] = max(old["proximity"], prox_ch)

    # new loop: gate first
    new = {"ttc": 0.0, "proximity": 0.0}
    for pr in eng.pairs_filtered(items, 2.0, _pair_relevant):
        scale = max(widths.get(pr.track_id_a, 0.0),
                    widths.get(pr.track_id_b, 0.0), 1.0)
        ttc_ch, prox_ch = pair_channels(pr, scale, cfg)
        new["ttc"] = max(new["ttc"], ttc_ch)
        new["proximity"] = max(new["proximity"], prox_ch)

    assert new == old
    assert _clamp01(new["ttc"]) == _clamp01(old["ttc"])
    # and the gate is not vacuous on this mix
    assert len(eng.pairs_filtered(items, 2.0, _pair_relevant)) \
        < len(eng.pairs(items, 2.0))


def test_heading_difference_helper_untouched():
    """Guard against an accidental edit to the shared geometry helper."""
    assert heading_difference_deg(359.0, 1.0) == 2.0
    assert heading_difference_deg(None, 1.0) is None


# ------------------------------------------------- production wiring (A/B is not enough)
def test_shares_pairwise_is_a_class_attribute_not_a_module_global():
    """Regression, and the reason the A/B tests above were not sufficient.

    PHASE 26 declared `SHARES_PAIRWISE = True` at MODULE scope in both
    detector modules. The manager reads it off the INSTANCE
    (`getattr(det, "SHARES_PAIRWISE", False)`, manager.py:330/357), and a
    module-level name is not reachable through an instance - so `getattr`
    returned False, the shared branch was dead code in production, and the
    unshared O(n^2) sweep still ran twice per frame. The A/B tests stayed
    green because `run_manager` setattr'd the attribute onto the class itself,
    manufacturing a state production never had. Both spellings are asserted so
    the module-scope mistake cannot come back unnoticed.
    """
    for mod, cls in ((accident_mod, AccidentDetector),
                     (near_miss_mod, NearMissDetector)):
        assert "SHARES_PAIRWISE" not in vars(mod), (
            f"{mod.__name__} declares SHARES_PAIRWISE at module scope; the "
            "manager reads it off the instance, so it would be dead code again"
        )
        assert cls.SHARES_PAIRWISE is True, f"{cls.__name__} lost the flag"
        # exactly what manager.py:330 evaluates
        assert getattr(cls(), "SHARES_PAIRWISE", False) is True


def test_production_manager_actually_takes_the_shared_branch():
    """Non-vacuity for the wiring above, with NO attribute patching at all.

    A default `EventManager` must run exactly one pair sweep per frame through
    `pairs_filtered` and ZERO through the unshared `pairs()`. If the flag ever
    stops being visible to the manager this fails, whereas the equality A/B
    would still pass (both arms would be the unshared path).
    """
    mgr = EventManager(Scene.defaults_estimated(1000, 1000), Settings(),
                       width=1000, height=1000)
    shared, unshared = [], []
    real_pairs = PairwiseInteractionEngine.pairs
    real_filtered = PairwiseInteractionEngine.pairs_filtered

    def spy_pairs(self, items, t_sec):
        unshared.append(len(items))
        return real_pairs(self, items, t_sec)

    def spy_filtered(self, items, t_sec, relevant):
        shared.append(len(items))
        return real_filtered(self, items, t_sec, relevant)

    PairwiseInteractionEngine.pairs = spy_pairs
    PairwiseInteractionEngine.pairs_filtered = spy_filtered
    try:
        for dets, t in frames("mixed"):
            mgr.step(dets, t)
    finally:
        PairwiseInteractionEngine.pairs = real_pairs
        PairwiseInteractionEngine.pairs_filtered = real_filtered

    assert len(shared) == N_FRAMES, (
        f"expected one shared sweep per frame, got {len(shared)}")
    assert unshared == [], (
        f"the unshared sweep ran {len(unshared)} times; the manager is not "
        "seeing SHARES_PAIRWISE")


def test_sharers_agree_on_candidate_labels():
    """The other silent bail in `_shared_interactions`: if the two sharers
    disagree about the candidate label set it returns None and sharing quietly
    switches off. Their label sets are the documented reason this cannot
    drift, so pin them."""
    eng = PairwiseInteractionEngine()
    a = AccidentDetector(pairwise=eng)
    n = NearMissDetector(pairwise=eng)
    assert frozenset(a.candidate_labels) == frozenset(n.candidate_labels)
    assert a.pairwise is n.pairwise is eng


def _tracks():
    """One car pair closing fast, with motion states, as the engines hand over."""
    from src.tracking.motion import MotionState

    class _Tr:
        def __init__(self, tid, label, x, y):
            self.track_id = tid
            self.label = label
            self.last = type("P", (), {"x": x, "bottom_y": y})()

    tracks = {1: _Tr(1, "car", 300.0, 300.0), 2: _Tr(2, "truck", 700.0, 300.0),
              3: _Tr(3, "person", 500.0, 420.0)}
    states = {1: MotionState(0.0, 100.0, 0.0, 100.0, 0.0, 90.0, False, 0.9),
              2: MotionState(0.0, -100.0, 0.0, 100.0, 0.0, 270.0, False, 0.9),
              3: MotionState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0.9)}
    return tracks, states
