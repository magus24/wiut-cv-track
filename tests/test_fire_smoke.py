"""fire_smoke decision tests (PHASE 22).

Official definition (HACKATHON_CONTEXT.md:59): `fire_smoke` = "visible fire or
smoke". It is an APPEARANCE class: no trajectory, motion, lane, or geometry
predicate can distinguish a burning vehicle from a normal one.

THE DECISION OF THIS PHASE IS "IMPLEMENTATION JUSTIFIED = NO", and these tests
are the machine-checked form of that decision. They assert the three facts that
make the class unreachable, so that anyone who later changes one of them is
told immediately instead of silently shipping a broken or noisy class:

  1. LABEL UNIVERSE.  `src/detection/detector.py` drops every COCO id except
     {car, motorcycle, bus, truck, person} and the wrapper reads only
     `res.boxes` - no masks, no segmentation, no second model. Nothing the
     detector can emit means fire or smoke.
  2. NO PIXEL CHANNEL.  `EventManager.step(detections, t_sec)` receives geometry
     and labels only. The frame is consumed by the detector inside
     `run_pipeline` and is never forwarded, so no event detector - of any kind -
     can look at a pixel. Proven at runtime by spying on the real call.
  3. NO PRODUCER.  The literal "fire_smoke" exists in `src/` only as a
     declared-but-never-raised flag key, never as a segment producer.

WHAT THESE TESTS ARE NOT
    They are not a fire/smoke detector and they contain no synthetic "fire".
    Every scenario below is built ONLY from labels the real pipeline can emit,
    because that is the whole point: the reachable input space is finite and it
    was enumerated exhaustively. Claiming otherwise would manufacture confidence
    the real class does not have (§17 of the phase brief).

FALSE-POSITIVE EVIDENCE (real footage, not synthetic)
    The only real traffic footage in the repo is a 127.6 s / 3840x2160 debug
    render of organizer sample C3905. A strict HSV survey of all 1275 frames
    found no fire-like region at all (max 64 px = 0.012% of frame, longest run
    above p99 = 4 frames = 0.4 s). Its one large "smoke-like" region - 9.2% of
    frame, persisting 43.6 s - was identified by running the shipped YOLO11x
    weights on those exact frames: it is a `truck` at conf 0.97 covering 99.5%
    of the blob. So the strongest grey-blob candidate in the real sample is an
    ordinary truck, which is precisely the false positive any colour/texture
    heuristic would have shipped. `test_stationary_truck_never_becomes_fire_smoke`
    pins that.

No production module is added, changed or registered by this phase.

Run:  python tests/test_fire_smoke.py
"""

from __future__ import annotations

import ast
import inspect
import itertools
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _validation import validate_events  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
LABEL = "fire_smoke"

# Files allowed to mention the literal: the flag declaration (always False), the
# label registry, and the offline annotation helper (metadata only). A
# PRODUCER would have to live somewhere else, which is what the test enforces.
ALLOWED_LITERAL_FILES = {
    os.path.join("src", "events", "rules.py"),
    os.path.join("src", "events", "manager.py"),
    os.path.join("src", "label_assist.py"),
}

# The exhaustive enumeration axes (see `test_fire_smoke_never_flagged_over_...`).
_REACHABLE_LABELS = ("car", "truck", "bus", "motorcycle", "person")
_SPEEDS = (0.0, 1.0, 1.9, 2.0, 2.5, 30.0)      # spans rules.STILL_PX_S = 2.0
_HEADINGS = (195.0, 15.0, 90.0, 270.0)        # spans rules.WRONG_WAY_DEV = 80
_STATIONARY_AT = (None, 0.0)                   # None / already stopped at t=0
_TIMES = (0.0, 120.0)                          # 0 s / past every 10 s threshold


def _scene():
    from src.scene import Scene
    return Scene.defaults_estimated(3840, 2160)


# 4 bottom-centre positions inside the estimated scene:
# on-road outside the crossing, on-road inside the crossing, off-road, far side.
_POSITIONS = {
    "on_road": (500.0, 1600.0),
    "in_crosswalk": (2000.0, 1000.0),
    "off_road": (500.0, 2050.0),
    "far_side": (3700.0, 400.0),
}


def _track_state(label, x, y, speed, heading, stationary_at):
    from src.events.rules import TrackState
    st = TrackState(label=label, last_t=0.0, x=x, y=y, speed=speed,
                    heading=heading, stationary_at=stationary_at)
    return st


def _enumerate_single_track_scenarios():
    """Every (label, speed, heading, stationary_at, position, time) combination.

    This is the COMPLETE input space of the legacy flag engine for a single
    road user: `rules.frame_flags` reads only label/x/y/speed/heading/
    stationary_at, and each of those is swept across every value that can flip
    a rule, including the exact thresholds.
    """
    for (label, speed, heading, stationary_at, pos_name, t) in itertools.product(
            _REACHABLE_LABELS, _SPEEDS, _HEADINGS, _STATIONARY_AT,
            sorted(_POSITIONS), _TIMES):
        x, y = _POSITIONS[pos_name]
        yield {"label": label, "pos_name": pos_name, "t": t,
               "track": _track_state(label, x, y, speed, heading, stationary_at)}


def _flags_for(tracks, scene, t_sec):
    """frame_flags on a clean congestion hold (mirrors EventManager.__init__)."""
    from src.events import rules
    rules._congestion_hold["on"] = False
    rules._congestion_hold["at"] = 0.0
    return rules.frame_flags(tracks, scene, t_sec)


# =========================================================== 1. label universe
class _FakeBox:
    """One ultralytics box, with the exact duck type detector.py consumes."""

    def __init__(self, xyxy, cls, conf):
        self.xyxy = xyxy          # [1, 4] -> b.xyxy[0].tolist()
        self.cls = cls            # [1]    -> b.cls[0].item()
        self.conf = conf          # [1]    -> float(b.conf[0])


class _FakeBoxes:
    def __init__(self, cls_ids):
        import torch
        n = len(cls_ids)
        self.cls = torch.tensor([float(c) for c in cls_ids])
        self.conf = torch.ones(n)
        # (n, 4): `track` reads boxes.xyxy.cpu().numpy()[i]; `detect` receives a
        # per-box (1, 4) slice via __iter__ - exactly the ultralytics layout.
        self.xyxy = torch.tensor([[10.0, 10.0, 20.0, 20.0]
                                  for _ in cls_ids])
        self.id = torch.arange(1, n + 1, dtype=torch.float32)
        # a segmentation result WOULD expose this; the wrapper must ignore it
        self.masks = torch.ones((n, 1, 20, 20))

    def __iter__(self):
        for i in range(len(self.cls)):
            # b.xyxy must be (1, 4) and b.cls (1,), exactly like ultralytics
            yield _FakeBox(self.xyxy[i].reshape(1, 4), self.cls[i:i + 1],
                           self.conf[i:i + 1])


class _FakeResult:
    def __init__(self, cls_ids):
        self.boxes = _FakeBoxes(cls_ids)


class _FakeModel:
    """Stands in for ultralytics.YOLO; emits one box per requested COCO id."""

    def __init__(self, cls_ids):
        self._ids = list(cls_ids)

    def predict(self, *a, **k):
        return [_FakeResult(self._ids)]

    def track(self, *a, **k):
        return [_FakeResult(self._ids)]


def _detector_over(cls_ids):
    """Run the REAL production filter over `cls_ids` -> the surviving labels."""
    import numpy as np

    from src.detection.detector import Detector
    det = Detector(model_path="unused", device="cpu", imgsz=64)
    det._model = _FakeModel(cls_ids)          # never touches the weights file
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    return det.detect(frame), det.track(frame)


ALL_COCO_IDS = list(range(80))               # COCO detection has 80 classes

# The only COCO ids this project is allowed to keep, and the name each one is
# given. Widening this table changes the input of ALL twelve event detectors, so
# it is pinned here rather than left implicit.
EXPECTED_SURVIVORS = {0: "person", 2: "car", 3: "motorcycle", 5: "bus",
                      6: "truck", 7: "truck"}


def _surviving_ids():
    """{coco_id: label} for every id the REAL filter lets through."""
    survivors = {}
    for cid in ALL_COCO_IDS:
        dets, tracks = _detector_over([cid])
        assert not dets and not tracks or dets and tracks, \
            "detect() and track() must agree on the class filter"
        if dets:
            survivors[cid] = dets[0]["label"]
    return survivors


def test_detector_reachable_label_universe():
    """Execute the real filter over all 80 COCO ids -> the reachable labels."""
    dets, tracks = _detector_over(ALL_COCO_IDS)
    assert dets, "the fake model produced nothing - the test proves nothing"
    for out in (dets, tracks):
        assert {d["label"] for d in out} == {"car", "motorcycle", "bus",
                                             "truck", "person"}
        for d in out:
            assert not any(k in d["label"] for k in
                           ("fire", "smoke", "flame", "steam", "hydrant"))
            # no pixel/mask channel leaks out of the wrapper
            assert set(d) <= {"xyxy", "conf", "label", "id"}, d

    # A widened filter would relabel a new id as some existing road-user name
    # (COCO 10 "fire hydrant" becomes "person" via the .get default), so the
    # names alone are not enough: pin WHICH ids survive as well.
    assert _surviving_ids() == EXPECTED_SURVIVORS


def test_detector_class_filter_drops_every_other_coco_id():
    """6 of 80 ids survive: {person, car, motorcycle, bus, truck, truck}."""
    dets, tracks = _detector_over(ALL_COCO_IDS)
    for out in (dets, tracks):
        assert len(out) == 6, sorted(d["label"] for d in out)
    # COCO 10 = fire hydrant, 15 = stop sign, 1 = bicycle are all dropped
    hydrant_only, _ = _detector_over([10])
    bicycle_only, _ = _detector_over([1])
    assert hydrant_only == [] and bicycle_only == []
    from src.detection.detector import COCO_BICYCLE
    assert COCO_BICYCLE == 1                  # defined, deliberately dropped


def test_detector_never_touches_masks_or_segmentation():
    """`res.boxes` only: no masks, no retina_masks, no segmentation call."""
    path = os.path.join(SRC, "detection", "detector.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    attrs, kws, strings, model_calls = set(), set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attrs.add(node.attr.lower())
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value.lower())
        if isinstance(node, ast.Call):
            for k in node.keywords:
                if k.arg:
                    kws.add(k.arg.lower())
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in (
                    "predict", "track", "segment"):
                model_calls.add(f.attr)
    assert "mask" not in attrs and "masks" not in attrs, \
        "detector must not consume segmentation masks"
    assert not any("mask" in k for k in kws), kws
    assert not any("mask" in s for s in strings), strings
    assert model_calls <= {"predict", "track"}, model_calls


# ============================================================ 2. no pixel path
def test_event_manager_step_signature_has_no_image_channel():
    from src.events.manager import EventManager
    params = list(inspect.signature(EventManager.step).parameters)
    assert params == ["self", "detections", "t_sec"], params


def test_pipeline_hands_the_event_layer_geometry_only():
    """Runtime proof: the real pipeline never forwards a frame to the events.

    The spy below has the same two-parameter signature as the real `step`, so if
    `run_pipeline` ever tried to pass a frame the call would raise TypeError
    before the assertion is ever reached.
    """
    import cv2
    import numpy as np

    import src.pipeline.pipeline as pp
    from src.events.manager import EventManager

    class FakeDetector:
        imgsz = 320

        def track(self, frame_bgr, persist: bool = True) -> list[dict]:
            return [{"xyxy": (10.0, 10.0, 60.0, 60.0), "conf": 0.9,
                     "label": "car", "id": 1}]

    tmp = tempfile.mkdtemp(prefix="tcv_p22_")
    path = os.path.join(tmp, "clip.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180))
    for _ in range(30):
        vw.write(np.zeros((180, 320, 3), dtype=np.uint8))
    vw.release()

    captured = []
    original = EventManager.step
    prev_model = pp._MODEL

    def spy(self, detections, t_sec):
        captured.append((detections, t_sec))
        return original(self, detections, t_sec)

    try:
        EventManager.step = spy
        pp._MODEL = FakeDetector()
        events = pp.run_pipeline(path)
    finally:
        EventManager.step = original
        pp._MODEL = prev_model
        shutil.rmtree(tmp, ignore_errors=True)

    assert captured, "the spy never saw a call - the test proves nothing"
    assert events == [] or all(isinstance(e, list) and len(e) == 3
                               for e in events)
    for dets, t_sec in captured:
        assert isinstance(t_sec, float)
        for d in dets:
            assert set(d) <= {"xyxy", "conf", "label", "id"}, d
            for v in d.values():
                assert not isinstance(v, np.ndarray), \
                    "a pixel/mask array reached the event layer"


def test_no_event_detector_accepts_an_image_parameter():
    """No `update`/`step` under src/events may take a frame/array argument."""
    banned = ("frame", "image", "img", "bgr", "pixel", "ndarray", "mask",
              "pixels")
    offenders = []
    events_dir = os.path.join(SRC, "events")
    for name in sorted(os.listdir(events_dir)):
        if not name.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(events_dir, name),
                              encoding="utf-8").read())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in ("update", "step"):
                continue
            a = node.args
            for arg in list(a.args) + list(a.posonlyargs) + list(a.kwonlyargs):
                if arg.arg == "self":
                    continue
                if any(b in arg.arg.lower() for b in banned):
                    offenders.append((name, node.name, arg.arg))
    assert offenders == [], offenders


# ============================================ 3. no producer / exhaustive falsif
def test_fire_smoke_never_flagged_over_the_whole_reachable_input_space():
    """Exhaustive: no reachable single-road-user state can raise the flag."""
    scene = _scene()
    checked = 0
    for sc in _enumerate_single_track_scenarios():
        flags = _flags_for({1: sc["track"]}, scene, sc["t"])
        assert LABEL in flags, "the flag column disappeared from the engine"
        assert flags[LABEL] is False, (
            "fire_smoke raised for label=%(label)s speed=%(speed)s "
            "heading=%(heading)s pos=%(pos_name)s" % sc)
        checked += 1
    assert checked == (len(_REACHABLE_LABELS) * len(_SPEEDS) * len(_HEADINGS)
                       * len(_STATIONARY_AT) * len(_POSITIONS) * len(_TIMES))
    assert checked == 5 * 6 * 4 * 2 * 4 * 2 == 1920


def test_exhaustive_enumeration_is_sensitive_to_the_other_rules():
    """Control: the sweep above is not vacuous - it does fire other classes."""
    scene = _scene()
    fired = set()
    for sc in _enumerate_single_track_scenarios():
        flags = _flags_for({1: sc["track"]}, scene, sc["t"])
        fired |= {k for k, v in flags.items() if v}
    assert "stopped_vehicle" in fired, fired
    assert "wrong_way" in fired, fired
    assert "jaywalking" in fired, fired
    # congestion needs >= 3 vehicles, so it is exercised separately
    veh = {i: _track_state("car", 500.0 + 40 * i, 1600.0, 0.0, 195.0, 0.0)
           for i in range(3)}
    assert _flags_for(veh, scene, 30.0)["congestion"] is True


def test_fire_smoke_absent_for_every_lane_flow_direction():
    """The sweep above pins one scene; `dominant_flow_deg` is the only scene
    parameter the vehicle rules read, so sweep it too. The value the pipeline
    actually ships is asserted first, to document the scope of the sweep."""
    scene = _scene()
    assert scene.dominant_flow_deg == [195.0], scene.dominant_flow_deg
    from src.events import rules
    flows = (0.0, 45.0, 90.0, 135.0, 180.0, 195.0, 225.0, 270.0, 315.0)
    checked = 0
    for flow, label, speed in itertools.product(flows, _REACHABLE_LABELS,
                                                _SPEEDS):
        scene.dominant_flow_deg = [flow]
        for pos in _POSITIONS.values():
            tracks = {1: _track_state(label, pos[0], pos[1], speed, 195.0, 0.0)}
            assert _flags_for(tracks, scene, 120.0)[LABEL] is False, \
                (flow, label, speed, pos)
            checked += 1
    assert checked == 9 * 5 * 6 * 4
    scene.dominant_flow_deg = [195.0]           # restore the shipped value


def test_all_label_pairs_never_raise_fire_smoke():
    """Multi-road-user scenes: every reachable label pair, worst-case config."""
    scene = _scene()
    checked = 0
    for a, b in itertools.product(_REACHABLE_LABELS, repeat=2):
        tracks = {
            1: _track_state(a, 2000.0, 1000.0, 0.0, 15.0, 0.0),   # stopped,
            2: _track_state(b, 500.0, 1600.0, 0.0, 195.0, 0.0),   # in a queue
        }
        assert _flags_for(tracks, scene, 120.0)[LABEL] is False, (a, b)
        checked += 1
    assert checked == 25


def test_stationary_truck_never_becomes_fire_smoke():
    """The real C3905 trap: a big grey truck parked on the road for 60 s.

    Identified empirically in the only real footage available (see the module
    docstring): a 43.6 s "smoke-like" blob that YOLO11x labels `truck` @0.97.
    It must stay silent, and the other classes it legitimately triggers must
    keep working.
    """
    from src.events import rules
    scene = _scene()
    tracks: dict = {}
    dets = []
    for i in range(601):                       # 0..120 s at 0.2 s
        t = i * 0.2
        rules._congestion_hold["on"] = False
        rules._congestion_hold["at"] = 0.0
        rules.update(tracks, [{"xyxy": (3092.0, 1100.0, 3836.0, 1576.0),
                               "conf": 0.97, "label": "truck", "id": 7}], t,
                     scene)
        flags = rules.frame_flags(tracks, scene, t)
        assert flags[LABEL] is False, t
    assert flags["stopped_vehicle"] is True, "the truck should still be a " \
        "stopped_vehicle - the test scene is otherwise inert"


def test_low_confidence_persistent_vehicle_never_becomes_fire_smoke():
    """A shaky, low-conf detection must not be read as fire evidence either."""
    scene = _scene()
    for conf in (0.05, 0.15, 0.25, 0.99):
        tracks = {1: _track_state("bus", 3700.0, 400.0, 0.0, 195.0, 0.0)}
        assert _flags_for(tracks, scene, 120.0)[LABEL] is False, conf


def test_no_fire_smoke_event_from_the_full_event_manager():
    """End of the default pool: the label never reaches `finalize` output."""
    from src.events.manager import EventManager
    scene = _scene()
    mgr = EventManager(scene, width=3840, height=2160)
    for i in range(120):                        # 0..23.8 s at 0.2 s
        t = i * 0.2
        mgr.step([{"xyxy": (2000.0, 1000.0, 2400.0, 1100.0), "conf": 0.9,
                   "label": "truck", "id": 3},
                  {"xyxy": (500.0, 1500.0, 560.0, 1600.0), "conf": 0.9,
                   "label": "person", "id": 4}], t)
    events = mgr.finalize(24.0)
    assert validate_events(events, 24.0) == []
    assert not [e for e in events if e[2] == LABEL], events


def test_fire_smoke_flag_column_stays_all_false_in_event_manager():
    """Mixes a MOVING car (wrong_way branch) and a STOPPED car (the branch a
    naive implementation would abuse). Both are covered, so the all-False
    column is a real statement and not an artefact of an inert scenario."""
    from src.events.manager import EventManager
    scene = _scene()
    mgr = EventManager(scene, width=3840, height=2160)
    for i in range(60):
        mgr.step([{"xyxy": (100.0 + 5 * i, 300.0, 160.0 + 5 * i, 340.0),
                   "conf": 0.9, "label": "car", "id": 1},          # moving
                  {"xyxy": (2000.0, 1600.0, 2100.0, 1650.0),
                   "conf": 0.9, "label": "car", "id": 2}],         # stopped
                 i * 0.2)
    assert LABEL in mgr.flags_map
    assert set(mgr.flags_map[LABEL]) == {False}
    assert len(mgr.flags_map[LABEL]) == 60
    # the stopped car really did trip its own class, so the loop is live
    assert any(mgr.flags_map["stopped_vehicle"]), "scenario is inert"
    assert any(mgr.flags_map["wrong_way"]), "scenario is inert"


# ==================================================== 4. output contract / CLASSES
def test_clean_events_unions_overlapping_fire_smoke_segments():
    from src.postprocessing import clean_events
    out = clean_events([[1.0, 3.0, LABEL], [2.0, 5.0, LABEL]], 10.0)
    assert out == [[1.0, 5.0, LABEL]]


def test_clean_events_keeps_cross_class_overlap_with_fire_smoke():
    from src.postprocessing import clean_events
    out = clean_events([[1.0, 5.0, LABEL], [2.0, 4.0, "accident"]], 10.0)
    assert sorted(out) == sorted([[1.0, 5.0, LABEL], [2.0, 4.0, "accident"]])


def test_fire_smoke_is_an_official_class_id():
    from solution import CLASSES
    assert LABEL in CLASSES
    assert CLASSES.count(LABEL) == 1
    assert isinstance(LABEL, str)
    # a well-formed segment for this label satisfies the harness contract
    assert validate_events([[12.0, 20.0, LABEL]], 60.0) == []
    # ... and the label is the official spelling, not a look-alike
    assert "fire_smoke" in " ".join(CLASSES)


# ============================================================== 5. no producer
def test_no_fire_smoke_detector_module_is_registered():
    assert not os.path.exists(os.path.join(SRC, "events", "fire_smoke.py"))
    events_pkg = os.path.join(SRC, "events")
    for name in sorted(os.listdir(events_pkg)):
        if not name.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(events_pkg, name),
                              encoding="utf-8").read())
        for node in tree.body:                  # module-level assignments
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "LABEL":
                    value = node.value
                    if isinstance(value, ast.Constant) and \
                            value.value == LABEL:
                        raise AssertionError(
                            f"{name} declares LABEL = {LABEL!r}: a producer "
                            "exists, re-run the PHASE 22 audit")


def test_fire_smoke_literal_exists_only_as_a_declaration():
    """The label may be *declared* (flags, registry, annotation tool) and
    nowhere else - in particular never as a `TemporalEventEngine` input."""
    found = {}
    for root, _dirs, files in os.walk(SRC):
        for name in files:
            if not name.endswith(".py"):
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, REPO)
            text = open(full, encoding="utf-8").read()
            if LABEL in text:
                found[rel] = found.get(rel, 0) + text.count(LABEL)
    assert set(found) <= ALLOWED_LITERAL_FILES, found
    for rel in found:
        text = open(os.path.join(REPO, rel), encoding="utf-8").read()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            # no call may pass the label into the temporal engine
            if isinstance(node, ast.Call) and \
                    isinstance(node.func, ast.Attribute) and \
                    node.func.attr == "update":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and arg.value == LABEL:
                        raise AssertionError(
                            f"{rel} feeds {LABEL!r} into a temporal engine")


# ================================================================ 6. determinism
def test_fire_smoke_absence_is_deterministic():
    """Three identical runs -> identical flags and identical final events."""
    from src.events.manager import EventManager

    def run_once():
        scene = _scene()
        mgr = EventManager(scene, width=3840, height=2160)
        for i in range(80):
            t = i * 0.2
            mgr.step([{"xyxy": (2000.0, 1000.0, 2400.0, 1100.0), "conf": 0.9,
                       "label": "truck", "id": 3},
                      {"xyxy": (500.0 + 3 * i, 1500.0, 560.0 + 3 * i, 1600.0),
                       "conf": 0.9, "label": "person", "id": 4}], t)
        return (list(mgr.flags_map[LABEL]), mgr.finalize(16.0))

    a, b, c = run_once(), run_once(), run_once()
    assert a == b == c
    assert set(a[0]) == {False}


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()
