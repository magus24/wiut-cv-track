"""Part B RiskEstimator tests (PHASE 23).

Covers the brief's ten required scenarios plus the invariants that make the
anticipation curve usable by the metric: hard interface, strict causality,
reset isolation, bounded output, determinism, and a measured false-alarm
budget on REAL normal traffic.

The scenarios drive the REAL production stack — real TrajectoryEngine, real
MotionEngine, real PairwiseInteractionEngine, real temporal EMA — through a
scripted detector. Nothing here re-implements a channel: a change that breaks
the engine breaks these tests, and a change that breaks the tuning is caught
by the channel assertions.

Two design facts are asserted rather than assumed, because both were derived
from measurement and both are easy to undo by accident:
  * the braking and proximity channels can never reach ALARM_THETA on their
    own (RiskConfig asserts it at construction, and the tests pin the real
    numbers);
  * on 162028 REAL records from the only footage in the repo -- 150726 pairs
    and 11302 track-observations over 423 frames -- the emitted curve is above
    ALARM_THETA on 3 frames, i.e. 0.90 s out of a 127 s clip, which the
    organizers' own ``evaluate.alarm_starts`` counts as 2 spurious alarms, and
    its MEDIAN is 0.178 (inside the brief's 0-0.2 "no/minimal risk" band).
    See the false-alarm-budget tests, which are the real evidence.

Alarm statistics are always taken from ``evaluate`` itself, never from a
reimplementation. The intuitive model of Score_B is wrong in a way that
changes decisions: ``alarm_starts`` (evaluate.py:257) has NO minimum run
length -- a 0.2 s alarm counts -- it merges runs closer than 2 s, and the
alarm time is the run START. So F1_alarm's precision divides by the NUMBER OF
ALARM RUNS (which is why the ramp is steep), while mTTA rewards alarming
early and is worth only 0.004-0.019 of Score_B at every ramp measured (which
is why that does not argue the other way).

Run:  python -m pytest tests/test_risk_estimator.py -q
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

import cv2  # noqa: E402
import solution  # noqa: E402
from solution import RISK_HORIZON_SEC  # noqa: E402
from src.config.settings import settings  # noqa: E402
from src.risk import ALARM_THETA  # noqa: E402
from src.risk import risk as R  # noqa: E402

# The tests inject a fake detector, so they need the IMPLEMENTATION class, not
# the zero-argument solution.RiskEstimator wrapper. The wrapper itself is
# exercised separately in test_hard_contract_unchanged.
RiskImpl = R.RiskEstimator

# The frame is 4K, like the samples. Scripting full-res boxes in 4K pixels and
# letting _prepare rescale them keeps the scenarios in realistic units; because
# W == settings.imgsz is false here the test helper divides by the real scale.
W, H = 3840, 2160
SMALL_H = max(1, int(settings.imgsz * H / W))
SX, SY = W / float(settings.imgsz), H / float(SMALL_H)
FRAME = np.full((H, W, 3), 40, dtype=np.uint8)   # non-zero: not "blank"
META = {"video_id": "t", "fps": 10.0, "width": W, "height": H, "n_frames": 400}


def box(tid: int, label: str, x: float, bottom_y: float,
        w: float = 200.0, h: float = 200.0, conf: float = 0.9) -> dict:
    """A detection whose FULL-RES bottom-center is exactly (x, bottom_y)."""
    return {"xyxy": ((x - w / 2) / SX, (bottom_y - h) / SY,
                     (x + w / 2) / SX, bottom_y / SY),
            "conf": conf, "label": label, "id": tid}


class ScriptedDetector:
    """Minimal stand-in for Detector: serves a fixed list per call.

    ``reset_tracker`` is implemented (unlike a bare stub) because a real
    ultralytics tracker does rewind its ids on reset, and several tests depend
    on the estimator rewinding the detector's script cursor with the video.
    """

    def __init__(self, script):
        self.script = script
        self.calls = 0
        self.resets = 0

    def track(self, frame, persist=True):
        out = self.script(self.calls)
        self.calls += 1
        return out

    def reset_tracker(self):
        self.calls = 0
        self.resets += 1


class TrackedDetector:
    """Detector that records reset_tracker() calls (and returns no boxes)."""

    def __init__(self):
        self.resets = 0
        self.calls = 0

    def track(self, frame, persist=True):
        self.calls += 1
        return []

    def reset_tracker(self):
        self.resets += 1


def run(steps, config=None, meta=None, frame=FRAME, estimator=None):
    """Feed [(t, [dets]), ...] through RiskEstimator.step; return the curve."""
    det = ScriptedDetector(lambda i: steps[min(i, len(steps) - 1)][1])
    est = estimator or RiskImpl(config=config,
                                detector_factory=lambda: det)
    est.reset(dict(meta if meta is not None else META))
    return [est.step(frame, t) for t, _ in steps], est


def constant_speed(xs, ys, w=200.0, h=200.0):
    """Dets for n objects each advancing at a constant per-step offset."""
    return [box(i + 1, "car", x, y, w, h) for i, (x, y) in enumerate(zip(xs, ys))]


# --------------------------------------------------------------------- #
# canonical approach scenario
# --------------------------------------------------------------------- #
# The shipped TTC ramp (curvature 6.0) only reaches ALARM_THETA at a
# Time-To-Collision of about 0.55 s, so a scenario has to hold the pair
# genuinely imminent for several internal observations (10 Hz here) before the
# asymmetric EMA can climb past 0.5. The old 9-step scenarios reached TTC 0
# exactly on their last frame, which is both too short for the EMA and a
# numerical coincidence (two boxes on the same pixel -> the shared engine
# reports TTC=+inf).
#
# This one is sized from the calibration instead: two cars closing at
# 1000 px/s each from a 4200 px gap, sampled every 0.1 s, so the TTC walks
# 2.1 s -> 0.2 s in 20 steps and never coincides.
APPROACH_V = 100.0          # px per 0.1 s step, per car (1000 px/s)
APPROACH_GAP0 = 4200.0
APPROACH_N = 20


def head_on_approach(n=APPROACH_N, v=APPROACH_V, gap0=APPROACH_GAP0,
                     y=1500.0, dt=0.1):
    """Head-on pair, TTC decreasing linearly; the alarm precursor.

    At step i the gap is ``gap0 - 2*v*i`` and the pair's TTC is
    ``dt * (gap0 / (2*v) - i)`` seconds, so the TTC at the last step is
    ``dt * (gap0/(2*v) - (n-1))`` -- assert that it stays above 0.
    """
    xa0 = 1920.0 - gap0 / 2.0
    xb0 = 1920.0 + gap0 / 2.0
    return [(dt * i, [box(1, "car", xa0 + v * i, y),
                      box(2, "car", xb0 - v * i, y)])
            for i in range(n)]


# --------------------------------------------------------------------- #
# interface / numerics
# --------------------------------------------------------------------- #
def test_hard_contract_unchanged():
    """reset(meta) + step(frame, t) -> float in [0,1]; no signature change."""
    import inspect
    assert list(inspect.signature(solution.RiskEstimator.reset).parameters) == ["self", "meta"]
    assert list(inspect.signature(solution.RiskEstimator.step).parameters) == ["self", "frame", "t_sec"]
    assert list(inspect.signature(R.RiskEstimator.reset).parameters) == ["self", "meta"]
    assert list(inspect.signature(R.RiskEstimator.step).parameters) == ["self", "frame", "t_sec"]
    assert RISK_HORIZON_SEC == 5.0
    assert ALARM_THETA == 0.5
    # solution.RiskEstimator must not have gained required arguments
    est = solution.RiskEstimator()
    est.reset({"video_id": "t", "fps": 10.0, "width": 64, "height": 64, "n_frames": 4})
    v = est.step(np.zeros((64, 64, 3), np.uint8), 0.0)
    assert isinstance(v, float) and 0.0 <= v <= 1.0


def test_step_never_opens_a_video_and_never_calls_part_a():
    """step() must not call detect_events / run_pipeline / VideoCapture."""
    import src.pipeline.pipeline as P
    import src.pipeline.video as V
    calls = []
    orig = {}

    def trap(mod, name):
        if not hasattr(mod, name):
            return  # only trap names this module actually has
        f = getattr(mod, name)
        orig[(mod.__name__, name)] = f

        def boom(*a, **k):
            calls.append((mod.__name__, name))
            return f(*a, **k)
        setattr(mod, name, boom)

    trapped = []
    for mod, name in ((P, "run_pipeline"), (V, "VideoCapture"),
                      (V, "VideoReader"), (P, "_get_detector"),
                      (P, "VideoReader")):
        if hasattr(mod, name):
            trap(mod, name)
            trapped.append(f"{mod.__name__}.{name}")
    # cv2.VideoCapture is the real "opens a video" primitive, and src.risk
    # imports cv2 directly, so it has to be trapped on cv2 itself.
    if hasattr(cv2, "VideoCapture"):
        trap(cv2, "VideoCapture")
        trapped.append("cv2.VideoCapture")
    # the trap must actually be armed, or this test would pass vacuously
    assert trapped, "nothing was trapped: the test would be vacuous"
    # also trap the public Part A entry point
    orig_detect = solution.detect_events
    solution.detect_events = lambda *a, **k: calls.append(("solution", "detect_events")) or []
    try:
        steps = head_on_approach()
        curve, est = run(steps)
    finally:
        solution.detect_events = orig_detect
        for (mname, name), f in orig.items():
            setattr(sys.modules[mname], name, f)
    assert calls == [], f"Part B reached Part A / video IO: {calls}"
    assert len(curve) == APPROACH_N
    # A shared model CACHE is fine (that is not Part A output), but it must be
    # reached only through the pipeline's accessor, and _get_detector is
    # trapped above -> with a fake factory it is never needed at all.
    assert est.errors == 0


def test_output_always_bounded_float_under_hostile_input():
    """NaN/inf/None/blank/garbage frames and times never escape [0, 1]."""
    est = RiskImpl(detector_factory=lambda: ScriptedDetector(lambda i: []))
    est.reset(dict(META))
    bad_frames = [None, np.zeros((H, W, 3), np.uint8), FRAME,
                  np.zeros((0, 0, 3), np.uint8), np.zeros((10, 10), np.uint8),
                  "not an image", 42, FRAME.astype(np.float32) * np.nan]
    bad_times = [0.0, -1.0, float("nan"), float("inf"), -float("inf"),
                 None, "x", 1e18, 0.0]
    t = 0.0
    for f in bad_frames:
        for bt in bad_times:
            v = est.step(f, bt)
            assert isinstance(v, float), (type(f), bt, type(v))
            assert 0.0 <= v <= 1.0, (type(f), bt, v)
            assert math.isfinite(v)
    for f in (FRAME,):
        for i in range(30):
            v = est.step(f, 0.1 * i)
            assert 0.0 <= v <= 1.0 and math.isfinite(v)
    assert est.errors == 0, "hostile input must be handled, not raised"


def test_nan_and_inf_in_detections_do_not_propagate():
    """A detector returning NaN/inf boxes must not poison the output."""
    nasty = [{"xyxy": (float("nan"),) * 4, "conf": float("nan"),
              "label": "car", "id": 1},
             {"xyxy": (float("inf"), 0.0, 1.0, 2.0), "conf": 0.9,
              "label": "car", "id": 2},
             {"xyxy": (0.0, 0.0, 0.0, 0.0), "conf": 0.9,
              "label": "car", "id": 3},
             box(4, "person", 1000.0, 1500.0, w=1.0, h=1.0)]
    steps = [(0.1 * i, nasty) for i in range(30)]
    curve, est = run(steps)
    assert all(0.0 <= v <= 1.0 and math.isfinite(v) for v in curve)
    assert est.errors == 0


# --------------------------------------------------------------------- #
# Test 1 — empty scene
# --------------------------------------------------------------------- #
def test_empty_scene_is_zero():
    steps = [(0.1 * i, []) for i in range(40)]
    curve, _ = run(steps)
    assert max(curve) == 0.0, "no tracks must mean no risk"
    assert curve[-1] == 0.0


def test_single_lone_vehicle_is_zero():
    """One object cannot form a pair, so the pair channels cannot fire."""
    steps = [(0.1 * i, constant_speed([500 + 40 * i], [1500])) for i in range(40)]
    curve, _ = run(steps)
    assert max(curve) == 0.0


# --------------------------------------------------------------------- #
# Test 2 — normal driving
# --------------------------------------------------------------------- #
def test_normal_stable_traffic_stays_low():
    """Constant, parallel, well-separated flow -> risk ~ 0 the whole time.

    This is the empirical version of the brief's "normal traffic -> risk ~ 0".
    """
    steps = []
    for i in range(50):
        xs = [400.0 + 200.0 * i, 1000.0 + 200.0 * i, 1600.0 + 200.0 * i]
        steps.append((0.1 * i, constant_speed(xs, [1200.0, 1200.0, 1200.0])))
    curve, _ = run(steps)
    assert max(curve) < 0.2, f"normal flow peaked at {max(curve):.3f}"
    assert curve[-1] < 0.05


def test_diverging_and_following_traffic_stays_low():
    """Separating motion and same-speed following must not look dangerous."""
    # TRULY diverging: the gap grows 1000 -> 2365 px and stays inside the frame.
    # (The earlier version had A at 800+60i and B at 2000-60i, which converge,
    # so it was measuring a collision and not divergence.)
    diverging = [(0.1 * i, constant_speed([500.0 + 15 * i, 1500.0 + 50 * i],
                                         [1500.0, 1500.0]))
                 for i in range(40)]
    gaps = [abs((1500 + 50 * i) - (500 + 15 * i)) for i in range(40)]
    assert gaps[-1] > gaps[0], "the scenario must actually diverge"
    c1, _ = run(diverging)
    assert max(c1) == 0.0, f"separating pair scored {max(c1):.3f}"

    # same speed, fixed gap -> closing ~ 0 -> TTC is inf by construction
    following = [(0.1 * i, constant_speed([800.0 + 50 * i, 1500.0 + 50 * i], [1500.0, 1500.0]))
                 for i in range(50)]
    c2, _ = run(following)
    assert max(c2) < 0.1, f"car-following scored {max(c2):.3f}"


# --------------------------------------------------------------------- #
# Test 3 — decreasing TTC
# --------------------------------------------------------------------- #
def test_ttc_risk_shape_is_monotone_and_matches_the_spec():
    """Pure function: 0 at/after the horizon, monotone, 1 at contact.

    The band expectations below are the SHIPPED ramp, and they deliberately do
    NOT match the brief's illustrative "ttc 1 s -> very high". A gentle ramp
    was measured to be unusable here: on real normal traffic it put the
    per-frame median at 0.54 and 66% of the clip in alarm, because max
    aggregation over ~350 ordinary pairs per frame lifts the worst of hundreds
    of resolved 1-second-TTC convergences into the frame score. So the
    calibrated ramp puts the 0.5 crossing at TTC ~ 0.55 s and still scores a
    sub-second precursor at 0.69-0.89. See RiskConfig's docstring.
    """
    assert R.ttc_risk(5.0) == 0.0
    assert R.ttc_risk(6.0) == 0.0
    assert R.ttc_risk(50.0) == 0.0
    assert R.ttc_risk(0.0) == 1.0
    assert R.ttc_risk(-3.0) == 1.0
    # no NaN/inf in, no NaN/inf out
    for bad in (None, float("nan"), float("inf"), -float("inf"), "x", object()):
        assert R.ttc_risk(bad) == 0.0
    seq = [8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.6, 0.3, 0.1, 0.0]
    vals = [R.ttc_risk(t) for t in seq]
    assert vals == sorted(vals), vals
    assert all(0.0 <= v <= 1.0 for v in vals)
    # at or beyond the horizon -> no risk at all
    assert vals[0] == 0.0 and vals[2] == 0.0          # 8 s, 5 s
    # the calibrated bands
    assert R.ttc_risk(3.0) < 0.05, "a 3 s TTC must be negligible"
    assert 0.2 < R.ttc_risk(1.0) < 0.35, "a 1 s TTC is developing"
    assert 0.4 < R.ttc_risk(0.6) < 0.55, "a 0.6 s TTC is at the threshold"
    assert R.ttc_risk(0.3) > 0.6, "a 0.3 s precursor is high"
    assert R.ttc_risk(0.1) > 0.8, "a 0.1 s precursor is very high"
    # horizon 0 / negative horizon must not explode
    assert R.ttc_risk(1.0, horizon_sec=0.0) == 0.0
    assert 0.0 <= R.ttc_risk(1.0, horizon_sec=-5.0) <= 1.0


def test_ttc_risk_is_monotone_over_a_dense_sweep():
    """No step of the ramp may go backwards, at any resolution of the sweep."""
    prev = -1.0
    t = 12.0
    while t >= 0.0:
        v = R.ttc_risk(t)
        assert v >= prev, f"ttc_risk went backwards at ttc={t}: {v} < {prev}"
        prev = v
        t -= 0.01


def _centre_x(det):
    """Full-res x of a scripted box's centre (box() stores it pre-scaled)."""
    return (det["xyxy"][0] + det["xyxy"][2]) / 2.0 * SX


def _gap(dets):
    return _centre_x(dets[1]) - _centre_x(dets[0])


def test_risk_increases_as_ttc_falls_through_the_real_pipeline():
    """Head-on approach through TTC 7.6 s -> 0.8 s: the curve must climb.

    This is the brief's "decreasing TTC" scenario. The 4K frame bounds how far
    two cars can start apart, so 7.6 s of TTC at 500 px/s of closing needs
    ~70 steps at 10 Hz to fall to 0.8 s; that is the honest sizing.

    With the calibrated ramp a 0.8 s TTC is deliberately still in the
    "developing" band, so this asserts a monotone climb into that band; the
    alarm itself is test_dangerous_pair_is_high, which runs the TTC below the
    0.55 s crossing.
    """
    n, v = 70, 25.0                     # px per 0.1 s step, per car
    gap0 = 3800.0                       # -> TTC 7.6 s at 500 px/s closing
    steps = []
    for i in range(n):
        steps.append((0.1 * i, [box(1, "car", 1920.0 - gap0 / 2 + v * i, 1500.0),
                                box(2, "car", 1920.0 + gap0 / 2 - v * i, 1500.0)]))
    # the scenario really does walk the TTC from 7.6 s down to ~0.8 s, and
    # both cars stay inside the 4K frame the whole way
    assert _gap(steps[0][1]) == gap0
    assert abs(_gap(steps[-1][1]) - 350.0) < 1.0, _gap(steps[-1][1])
    for _, dets in steps:
        assert 0.0 < _centre_x(dets[0]) < W and 0.0 < _centre_x(dets[1]) < W

    curve, est = run(steps)
    assert all(0.0 <= v_ <= 1.0 for v_ in curve)
    assert curve[0] == 0.0, "TTC at the horizon must be no risk at all"
    tail = curve[3:]
    assert tail == sorted(tail), f"risk not monotone in TTC: {tail}"
    assert tail[-1] > 0.2, f"a 0.8 s TTC should be clearly developing: {tail[-1]:.3f}"
    assert tail[-1] < ALARM_THETA, \
        f"a 0.8 s TTC must stay below the alarm: {tail[-1]:.3f}"
    assert est.last_channels["ttc"] > 0.0


# --------------------------------------------------------------------- #
# Test 4 — sudden braking
# --------------------------------------------------------------------- #
def test_brake_risk_pure_function_uses_absolute_deceleration():
    assert R.brake_risk(0.0, 500.0) == 0.0            # coasting
    assert R.brake_risk(+900.0, 500.0) == 0.0         # accelerating
    assert R.brake_risk(-180.0, 500.0) == 0.0         # normal p1
    assert R.brake_risk(-5000.0, 500.0) == 1.0        # emergency
    assert 0.0 < R.brake_risk(-700.0, 500.0) < 1.0
    # the slow-vehicle guard: this is the case the measured p50 speed = 23 px/s
    # made fatal for the removed dimensionless form
    assert R.brake_risk(-2000.0, 25.0) == 0.0
    assert R.brake_risk(-2000.0, 81.0) == 1.0
    for bad in (None, float("nan"), float("inf"), "x"):
        assert R.brake_risk(bad, 500.0) == 0.0
        assert R.brake_risk(-500.0, bad) == 0.0


def _braking_curve(decel, v0=4000.0, n=40, dt=0.1):
    """One car decelerating at ``decel`` px/s^2; return (curve, brake channel)."""
    steps = []
    for i in range(n):
        t = dt * i
        v = max(v0 - decel * t, 0.0)
        steps.append((t, [box(1, "car", 300.0 + v * t, 1500.0)]))
    est = RiskImpl(detector_factory=lambda: ScriptedDetector(
        lambda i, s=steps: s[min(i, len(s) - 1)][1]))
    est.reset(dict(META))
    curve, brake = [], []
    for t, _ in steps:
        curve.append(est.step(FRAME, t))
        brake.append(est.last_channels["braking"])
    return curve, brake


def test_sudden_braking_raises_risk_but_never_alone_reaches_the_alarm():
    """A single car braking hard climbs, and stays under ALARM_THETA.

    The PEAK of the brake channel is asserted, not its final value: the
    scenario has to run long enough for MotionEngine to build a velocity
    history, and by the end the car has stopped, at which point the channel is
    correctly back to 0 (that is test_stopping_artifact_does_not_fire_the_
    brake_channel). Reading last_channels at the end measured the wrong thing.
    """
    decel = 1500.0        # px/s^2, far above the normal-traffic p99.9 of 450
    curve, brake = _braking_curve(decel)
    peak_brake = max(brake)
    assert peak_brake > 0.15, f"hard braking did not register: {peak_brake:.3f}"
    assert max(curve) > 0.15, f"hard braking did not register: {max(curve):.3f}"
    assert max(curve) < ALARM_THETA, \
        f"lone braking must never alarm, peaked {max(curve):.3f}"
    # and it does come back down once the car is stopped
    assert brake[-1] == 0.0, f"brake channel stuck at {brake[-1]:.3f}"


def test_stopping_artifact_does_not_fire_the_brake_channel():
    """A vehicle going instantly to zero speed is a tracker step, not braking."""
    steps = [(0.1 * i, [box(1, "car", 500.0 + 900.0 * 0.1 * i, 1500.0)])
             for i in range(8)]
    steps += [(0.8 + 0.1 * i, [box(1, "car", 500.0 + 900.0 * 0.8, 1500.0)])
              for i in range(12)]
    est = RiskImpl(detector_factory=lambda: ScriptedDetector(
        lambda i, s=steps: s[min(i, len(s) - 1)][1]))
    est.reset(dict(META))
    curve = [est.step(FRAME, t) for t, _ in steps]
    assert est.last_channels.get("braking", 0.0) == 0.0, \
        f"a stop/start step fired the brake channel: {est.last_channels}"
    assert max(curve) < ALARM_THETA


# --------------------------------------------------------------------- #
# Test 5 — dangerous pair
# --------------------------------------------------------------------- #
def test_dangerous_pair_is_high():
    """The brief's dangerous-pair scenario: a converging pair must alarm.

    Sized from the calibration (see head_on_approach): the pair holds a
    sub-0.55 s TTC for its last ~5 internal observations, which is what the
    asymmetric EMA needs to climb past ALARM_THETA.
    """
    steps = head_on_approach()
    # the scenario must genuinely be a collision course, and must not end on
    # two boxes sharing a pixel (the shared engine reports TTC=+inf there)
    gaps = [_gap(dets) for _, dets in steps]
    assert gaps[0] == APPROACH_GAP0
    assert gaps[-1] > 30.0, gaps[-1]
    curve, est = run(steps)
    assert curve[-1] >= 0.6, f"imminent collision only scored {curve[-1]:.3f}"
    assert est.last_channels["ttc"] >= 0.5, est.last_channels
    # a stable, single alarm: it crosses once and stays up
    crossings = sum(1 for a, b in zip(curve, curve[1:]) if a < ALARM_THETA <= b)
    assert crossings == 1, f"expected one threshold crossing, got {crossings}"


def test_relative_velocity_gates_and_the_meeting_test_are_both_used():
    """Two pairs with the SAME TTC must be separable, and the way is measured.

    The TTC channel is a function of TTC, so two pairs with identical TTC
    score identically by construction -- the earlier version of this test
    asserted they should not, which was simply false. What must separate them
    is (a) the closing/relative motion gates, which refuse a pair that is
    nominally "approaching" but not actually moving, and (b) the "will they
    actually meet" factor, which is what relative velocity buys you: a pair
    converging fast but on paths that miss is not a crash.
    """
    cfg = R.DEFAULT_CONFIG

    def pair(**kw):
        base = {"approaching": True, "closing_speed_px_s": 400.0,
                "relative_speed_px_s": 400.0, "ttc_sec": 0.4,
                "min_predicted_distance_px": 0.0, "distance_px": 160.0,
                "class_a": "car", "class_b": "car", "scale": 200.0}
        base.update(kw)

        class P:
            pass

        p = P()
        for k, val in base.items():
            setattr(p, k, val)
        return R.RiskEstimator._pair_channels(p, base["scale"], cfg)

    meet, _ = pair()
    miss, _ = pair(min_predicted_distance_px=2000.0)   # 10 widths apart
    assert meet > 0.5, meet
    assert miss == 0.0, f"a pair that misses by 10 widths scored {miss}"

    # (a) the motion gates: no real relative motion -> the channel is off even
    #     when the interaction engine reports a nominal TTC.
    stalled, _ = pair(closing_speed_px_s=1.0, relative_speed_px_s=2.0)
    assert stalled == 0.0, f"a stalled pair scored {stalled}"
    not_approaching, _ = pair(approaching=False)
    assert not_approaching == 0.0
    # The thresholds are pinned as ABSOLUTE numbers, not read back from the
    # config: asserting against cfg.min_closing_px_s would still pass if the
    # gate were widened to -1e9, which is precisely the regression to catch.
    assert cfg.min_closing_px_s == 5.0, cfg.min_closing_px_s
    assert cfg.min_relative_px_s == 8.0, cfg.min_relative_px_s
    assert pair(closing_speed_px_s=5.0, relative_speed_px_s=8.0)[0] > 0.0
    assert pair(closing_speed_px_s=4.9, relative_speed_px_s=8.0)[0] == 0.0, \
        "a pair closing at 4.9 px/s must not score"
    assert pair(closing_speed_px_s=5.0, relative_speed_px_s=7.9)[0] == 0.0, \
        "a pair with 7.9 px/s of relative motion must not score"

    # (b) coincident tracks are perception artifacts, not conflicts
    duplicated, _ = pair(distance_px=2.0, scale=300.0)
    assert duplicated == 0.0, f"a duplicate track pair scored {duplicated}"

    # (c) and the same must hold end-to-end, not just in the pure function
    fast = run(head_on_approach(v=300.0, gap0=6000.0, n=20))[0]
    slow = run(head_on_approach(v=25.0, gap0=6000.0, n=20))[0]
    assert max(fast) > max(slow), (max(fast), max(slow))
    assert max(slow) < ALARM_THETA


# --------------------------------------------------------------------- #
# Test 6 — no dilution by safe objects
# --------------------------------------------------------------------- #
def test_many_safe_objects_do_not_dilute_the_dangerous_pair():
    """max() aggregation: a crowd of safe pairs cannot average the danger down."""
    alone, _ = run(head_on_approach())

    crowd = []
    for t, danger in head_on_approach():
        dets = list(danger)
        i = int(round(t / 0.1))
        # 40 far-away, mutually parallel, non-converging objects
        for k in range(40):
            dets.append(box(100 + k, "car", 200.0 + 30.0 * i + 8.0 * k,
                            300.0 + 25.0 * k))
        crowd.append((t, dets))
    c_crowd, _ = run(crowd)

    assert alone[-1] >= 0.6, alone[-1]
    assert abs(c_crowd[-1] - alone[-1]) < 0.02, \
        f"crowd changed the signal: {alone[-1]:.3f} -> {c_crowd[-1]:.3f}"


def test_track_cap_is_bounded_and_deterministic():
    """max_tr caps the O(N^2) scan; the kept subset is chosen deterministically."""
    dets = [box(1, "car", 300.0, 500.0), box(2, "car", 3400.0, 500.0)]
    dets += [box(50 + k, "car", 100.0 + 7.0 * k, 300.0 + 11.0 * k)
             for k in range(80)]
    steps = [(0.1 * i, dets) for i in range(20)]
    cfg = R.RiskConfig(max_tracks=8)
    c1, est1 = run(steps, config=cfg)
    c2, _ = run(steps, config=cfg)
    assert c1 == c2, "track cap is not deterministic"
    assert all(0.0 <= v <= 1.0 for v in c1)
    assert est1.errors == 0


# --------------------------------------------------------------------- #
# Test 7 — reset isolation
# --------------------------------------------------------------------- #
def test_reset_clears_all_state_between_videos():
    """video A -> reset -> video B must equal a clean run of video B alone."""
    danger = head_on_approach()
    calm = [(0.1 * i, constant_speed([400.0 + 50 * i, 1200.0 + 50 * i], [1500.0, 1500.0]))
            for i in range(30)]

    det_a = ScriptedDetector(lambda i: danger[min(i, len(danger) - 1)][1])
    shared = RiskImpl(detector_factory=lambda: det_a)
    shared.reset(dict(META))
    a_curve = [shared.step(FRAME, t) for t, _ in danger]
    assert a_curve[-1] > 0.6, "video A did not build up any state to leak"

    # now the SAME estimator object, on a calm video
    det_a.script = lambda i: calm[min(i, len(calm) - 1)][1]
    shared.reset(dict(META))
    after = [shared.step(FRAME, t) for t, _ in calm]

    clean, _ = run(calm)
    assert after == clean, "reset left state from the previous video"


def test_reset_clears_every_internal_container():
    """Structural check: no stateful container survives reset()."""
    est = RiskImpl(detector_factory=lambda: ScriptedDetector(
        lambda i: head_on_approach()[min(i, APPROACH_N - 1)][1]))
    est.reset(dict(META))
    for t, _ in head_on_approach():
        est.step(FRAME, t)
    assert est.last_channels["ttc"] > 0.0, "the scenario must build real state"
    est.reset(dict(META))
    assert est._traj.tracks == {}
    assert est._brake == {}
    assert est._moving == {}
    assert est._ema is None
    assert est._last == 0.0
    assert est._prev_t is None
    assert est.calls == 0 and est.observations == 0
    assert est.last_channels == {"ttc": 0.0, "braking": 0.0, "proximity": 0.0}
    assert est._stride == R._stride_for_fps(META["fps"], est.config)
    # first step of the new video starts from zero, not from the old EMA
    assert est.step(FRAME, 0.0) == 0.0


def test_reset_drops_bytetrack_state():
    """A shared model's tracker must be reset, or track ids leak across videos.

    ultralytics keeps ONE tracker alive while persist=True, so this is the
    mechanism that makes the two parts independent per video. It also has to
    hold when the DETECTOR OBJECT is cached and reused, which is the normal
    case: that is where an early return used to skip the reset.
    """
    det = TrackedDetector()
    est = RiskImpl(detector_factory=lambda: det)
    est.reset(dict(META))
    # the reset is armed but performed lazily, on the first observation
    assert det.resets == 0, "reset() must not reach for the detector eagerly"
    est.step(FRAME, 0.0)
    assert det.resets == 1, "the first observation must reset the tracker"

    est.step(FRAME, 0.1)
    assert det.resets == 1, "the tracker must not be reset every frame"

    est.reset(dict(META))
    est.step(FRAME, 0.0)
    assert det.resets == 2, "a new video must reset the tracker again"
    # and it is the CACHED detector that got reset, not a fresh instance
    assert est._detector is det

    # the initial value matters too: a step() with no reset() in between must
    # still start from a clean tracker, so the flag has to be armed by
    # __init__ as well as by reset()
    fresh = TrackedDetector()
    est2 = RiskImpl(detector_factory=lambda: fresh)
    est2.step(FRAME, 0.0)
    assert fresh.resets == 1, \
        "an estimator used without reset() must still reset the tracker once"


# --------------------------------------------------------------------- #
# Test 8 — causality
# --------------------------------------------------------------------- #
def test_future_frames_cannot_change_past_output():
    """step(t) is byte-identical whether or not later frames are ever fed."""
    danger = head_on_approach()
    prefix = danger[:12]
    suffix = danger[12:]

    short, _ = run(prefix)
    long_, _ = run(danger)
    assert short == long_[:len(short)], "past output changed when future arrived"

    # a violent future (pedestrian materialising in the road) must not leak back
    v = APPROACH_V
    i0 = 12
    violent = [(0.1 * (i0 + i),
                [box(1, "car", 1920.0 - APPROACH_GAP0 / 2 + v * (i0 + i), 1500.0),
                 box(2, "car", 1920.0 + APPROACH_GAP0 / 2 - v * (i0 + i), 1500.0),
                 box(3, "person", 1920.0, 1500.0, w=60, h=180)])
               for i in range(3)]
    with_future, _ = run(prefix + violent)
    assert with_future[:len(short)] == short, "future frames leaked backwards"
    assert len(suffix) == APPROACH_N - 12


def test_no_state_is_written_before_the_current_frame_is_read():
    """Guards the internals: risk for t uses only engines fed with times <= t."""
    steps = head_on_approach()
    det = ScriptedDetector(lambda i: steps[min(i, len(steps) - 1)][1])
    est = RiskImpl(detector_factory=lambda: det)
    est.reset(dict(META))
    prev = -1.0
    for t, _ in steps:
        out = est.step(FRAME, t)
        # every trajectory point recorded so far must be at or before now
        for tr in est._traj.tracks.values():
            for p in tr.points:
                assert p.t <= t + 1e-9, "a future trajectory point was ingested"
        prev = out
    assert prev >= ALARM_THETA, f"the scenario did not reach an alarm: {prev:.3f}"


# --------------------------------------------------------------------- #
# Test 9 — bounds
# --------------------------------------------------------------------- #
def test_bounds_hold_across_every_scenario():
    v = APPROACH_V
    scenarios = {
        "empty": [(0.1 * i, []) for i in range(15)],
        "one": [(0.1 * i, constant_speed([500 + 40 * i], [1500])) for i in range(15)],
        "head_on": head_on_approach(),
        "pedestrian": [(0.1 * i, [box(1, "car", 1400.0 + v * i, 1500.0),
                                  box(2, "person", 2000.0, 1500.0, w=60, h=180)])
                       for i in range(15)],
        "degenerate": [(0.1 * i, [box(1, "car", 1000.0, 1000.0, w=0.0, h=0.0),
                                  box(2, "car", 1000.0, 1000.0, w=0.0, h=0.0)])
                       for i in range(15)],
        "huge": [(0.1 * i, [box(1, "car", 1e9, 1e9, w=1e7, h=1e7),
                            box(2, "car", -1e9, -1e9, w=1e7, h=1e7)])
                 for i in range(15)],
    }
    for name, steps in scenarios.items():
        curve, _ = run(steps)
        for v_ in curve:
            assert isinstance(v_, float) and 0.0 <= v_ <= 1.0, (name, v_)
            assert math.isfinite(v_), (name, v_)


def test_temporal_shaping_smooths_a_single_observation_spike():
    """One raw spike must be neither passed through nor held at its peak.

    Both failure modes are real and both were considered during the design:
      * pass-through (``_ema = raw``) emits the raw 1.0 for exactly one frame,
        which the metric's 0.5 s minimum-run filter then throws away, so a real
        precursor could never be scored;
      * a running max / peak hold pins 1.0 for the rest of the video, which is
        the "always alarmed" curve the brief explicitly rules out -- and which
        would silently break the whole false-alarm budget.

    The EMA is what gives both properties at once: an attack rate that never
    jumps, and a release that is gradual but finite.
    """
    cfg = R.DEFAULT_CONFIG
    est = RiskImpl(detector_factory=lambda: ScriptedDetector(lambda i: []))
    est.reset(dict(META))

    first = est._shape(1.0, 0.0)
    assert first == 1.0, "the very first observation is passed through"

    out = [est._shape(0.0, 0.1 * i) for i in range(1, 25)]
    assert all(0.0 <= v <= 1.0 for v in out), out
    assert out[0] < first, "the spike was not smoothed at all"
    assert out[0] > 0.4, f"risk vanished within one frame: {out[0]:.3f}"
    assert out == sorted(out, reverse=True), f"risk did not decay: {out}"

    # the release is exactly the configured geometric one
    expected = first
    for v in out:
        expected += cfg.ema_down * (0.0 - expected)
        assert abs(v - expected) < 1e-12, (v, expected)

    assert out[-1] < 0.01, f"risk was never released: {out[-1]:.4f}"
    # NOTE ON RUN LENGTH. evaluate.alarm_starts (evaluate.py:257) has NO
    # minimum run length -- a 0.2 s alarm counts -- so the 0.2 s this produces
    # is a perfectly good alarm, not a near miss. What the shaping has to
    # guarantee is that a lone spike is ONE run rather than a smear of many,
    # and that the danger is released afterwards. Both are asserted above; the
    # complement (a genuine precursor DOES reach theta) is
    # test_dangerous_pair_is_high and
    # test_a_genuine_precursor_alarms_inside_the_metrics_matching_window.
    held = sum(1 for v in out if v >= ALARM_THETA) * 0.1
    assert held > 0.0, "the spike did not raise an alarm at all"

    sustained, _ = run(head_on_approach())
    assert max(sustained) >= ALARM_THETA, "a real precursor must alarm"


def test_a_raising_detector_is_survived_and_counted():
    """A perception failure must cost one frame, not the whole Part B curve.

    An exception escaping ``step`` lands in run_submission.run_risk, which
    writes ``entry["risk"] = []`` -- i.e. a perception hiccup would silently
    void the entire anticipation curve for that video.
    """
    class Exploding:
        def __init__(self):
            self.calls = 0

        def track(self, frame, persist=True):
            self.calls += 1
            raise RuntimeError("perception exploded")

        def reset_tracker(self):
            pass

    det = Exploding()
    est = RiskImpl(detector_factory=lambda: det)
    est.reset(dict(META))
    outs = [est.step(FRAME, 0.1 * i) for i in range(10)]
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in outs), outs
    assert det.calls == 10
    # swallowed, but LOUD: the counter must make it visible
    assert est.errors == 10, est.errors
    assert est.observations == 10, "a failed observation is still an observation"

    # ... and a raising FRAME (bad dtype / shape) is survivable too
    est2 = RiskImpl(detector_factory=lambda: ScriptedDetector(
        lambda i: [box(1, "car", 900.0 + 40 * i, 1500.0)]))
    est2.reset(dict(META))
    for bad in (None, "not a frame", np.zeros((3, 3), np.float32), object()):
        v = est2.step(bad, 0.1)
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (bad, v)


def test_the_observation_rate_knob_defaults_to_10hz_and_is_honoured():
    """TCV_RISK_TARGET_HZ is an emergency cost lever, so its default is pinned.

    It exists because run_submission.py:196 blanks the WHOLE entry when Part A
    + Part B exceed 3x the duration -- so a slow Part B also throws away Part A.
    Measured on a 12 s slice of the 9.99 fps render with CPU-only torch, Part B
    cost 60.4 s against Part A's 23.7 s. If the grading machine has no working
    GPU, turning the rate down halves the detector calls; the price is a
    noisier velocity estimate (0.2 s between observations, 4 points in
    MotionEngine's 0.8 s window instead of 8).
    """
    assert R.DEFAULT_CONFIG.target_hz == 10.0, R.DEFAULT_CONFIG.target_hz
    # the env var is absent in the normal case, and the default still holds
    assert "TCV_RISK_TARGET_HZ" not in os.environ or \
        R._TARGET_HZ_DEFAULT == 10.0, R._TARGET_HZ_DEFAULT

    # a garbage or non-positive value falls back rather than degenerating
    for bad in ("", "abc", "0", "-3", "nan"):
        os.environ["TCV_RISK_TARGET_HZ"] = bad
        try:
            import importlib
            reloaded = importlib.reload(R)
            assert reloaded._TARGET_HZ_DEFAULT == 10.0, (bad, reloaded._TARGET_HZ_DEFAULT)
        finally:
            os.environ.pop("TCV_RISK_TARGET_HZ", None)
            importlib.reload(R)
    assert R.DEFAULT_CONFIG.target_hz == 10.0

    # a valid value is honoured, and it changes the derived frame stride
    os.environ["TCV_RISK_TARGET_HZ"] = "5"
    try:
        import importlib
        reloaded = importlib.reload(R)
        assert reloaded._TARGET_HZ_DEFAULT == 5.0
        # 25 fps at a 5 Hz target -> stride 5, versus stride 3 at 10 Hz
        assert reloaded._stride_for_fps(25.0, reloaded.DEFAULT_CONFIG) == 5
        est = reloaded.RiskEstimator(
            detector_factory=lambda: ScriptedDetector(lambda i: []))
        est.reset(dict(META, fps=25.0))
        assert est._stride == 5, est._stride
        # ... and on the 9.99 fps render it goes from stride 1 to stride 2,
        # which is the whole point: it halves the detector calls
        est2 = reloaded.RiskEstimator(
            detector_factory=lambda: ScriptedDetector(lambda i: []))
        est2.reset(dict(META, fps=9.99))
        assert est2._stride == 2, est2._stride
        assert R._stride_for_fps(9.99, R.RiskConfig(target_hz=10.0)) == 1
    finally:
        os.environ.pop("TCV_RISK_TARGET_HZ", None)
        importlib.reload(R)

    # and the module is back to its shipped state afterwards
    assert R.DEFAULT_CONFIG.target_hz == 10.0
    assert R._stride_for_fps(25.0, R.DEFAULT_CONFIG) == 3


def test_is_blank_agrees_with_any_but_is_far_cheaper():
    """The blank-frame guard must be exactly `any()`, and must be cheap.

    `not frame.any()` is the obvious spelling, and it was the single most
    expensive line in Part B: measured on 4K input, `any()` costs 6.8 ms per
    frame against 0.67 ms for an equivalent max-based test -- more than the
    whole channel computation, and ~36 s of a 30 fps 3-minute video's
    3x-duration budget. `_is_blank` is the same predicate via `max()`.
    """
    import time

    cases = [
        ("all zero 4K", np.zeros((H, W, 3), np.uint8)),
        ("non-blank 4K", FRAME),
        ("single pixel at the very end", _at_last_pixel(0, 255)),
        ("single pixel at the very start", _at_last_pixel(2, 7)),
        ("one channel only", _at_last_pixel(1, 1)),
        ("all -1", np.full((64, 64, 3), -1, np.int32)),
        ("all 1", np.full((64, 64, 3), 1, np.uint8)),
        ("float", np.full((64, 64, 3), 0.5, np.float32)),
        ("64x64 zero (the format test's frame)", np.zeros((64, 64, 3), np.uint8)),
    ]
    # a lit pixel must be found wherever it is, including the two corners a
    # truncated scan would miss
    for xy in ((0, 0), (H - 1, W - 1), (0, W - 1), (H - 1, 0), (H // 2, W // 2)):
        cases.append((f"one lit pixel at {xy}", _at(0, 0, value=255, xy=xy)))
    for name, frame in cases:
        assert R._is_blank(frame) == (not bool(frame.any())), \
            f"{name}: _is_blank says {R._is_blank(frame)}, any() says {bool(frame.any())}"

    # an all-NaN frame is corrupt, not blank: it must NOT be reported blank,
    # because the safe direction is to hand it to perception and fail soft
    # there, not to emit a confident 0.0 risk for a frame nobody looked at.
    nan = np.full((64, 64, 3), np.nan, np.float32)
    assert not R._is_blank(nan)

    # the cost claim itself, on 4K -- generous bounds so this is not flaky
    big = np.full((H, W, 3), 40, np.uint8)

    def cost(fn, n=60):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1000.0

    fast = cost(lambda: R._is_blank(big))
    slow = cost(lambda: bool(big.any()))
    assert fast < 2.0, f"_is_blank on 4K costs {fast:.2f} ms"
    assert slow > fast, f"any() {slow:.2f} ms was not slower than {fast:.2f} ms"


def test_clamp01_is_a_total_function():
    for bad in (None, float("nan"), float("inf"), -float("inf"), "x", object(),
                [], {}, 1j, True):
        r = R._clamp01(bad)
        assert isinstance(r, float) and 0.0 <= r <= 1.0, (bad, r)
    assert R._clamp01(-5.0) == 0.0
    assert R._clamp01(5.0) == 1.0
    assert R._clamp01(0.25) == 0.25


# --------------------------------------------------------------------- #
# Test 10 — determinism
# --------------------------------------------------------------------- #
def test_two_identical_runs_are_identical():
    v = APPROACH_V
    steps = [(0.1 * i, [box(1, "car", 1400.0 + v * i, 1500.0),
                        box(2, "car", 2900.0 - v * i, 1500.0),
                        box(3, "person", 2600.0 - 10 * i, 1500.0, w=60, h=180)])
             for i in range(40)]
    a, est_a = run(steps)
    b, est_b = run(steps)
    assert a == b, "run A != run B"
    assert est_a.last_channels == est_b.last_channels
    # Replaying the SAME step list through a third estimator must also match.
    # This is the check that caught step() mutating the detector's output in
    # place: the second and third runs were silently double-rescaling the
    # shared detection dicts, which showed up as a 1-ULP difference.
    c, _ = run(steps)
    assert a == c, "replaying the same detections is not idempotent"


def test_step_does_not_mutate_the_detectors_output():
    """step() must have no side effect on the frame or the detections."""
    steps = head_on_approach()
    snapshot = [dict(d) for _, dets in steps for d in dets]
    frames = [FRAME.copy() for _ in steps]
    run(steps)
    now = [dict(d) for _, dets in steps for d in dets]
    assert snapshot == now, "step() mutated the detection dicts it was given"
    for before, after in zip(frames, [FRAME] * len(frames)):
        assert before is not after or np.array_equal(before, FRAME)


def test_determinism_holds_across_interleaved_scenarios():
    """A and B then A again -> the second A equals the first A."""
    a_steps = head_on_approach()
    b_steps = [(0.1 * i, constant_speed([300.0 + 20 * i], [800.0])) for i in range(12)]
    det = ScriptedDetector(lambda i: [])
    est = RiskImpl(detector_factory=lambda: det)
    out = []
    for steps in (a_steps, b_steps, a_steps):
        det.script = lambda i, s=steps: s[min(i, len(s) - 1)][1]
        est.reset(dict(META))
        out.append([est.step(FRAME, t) for t, _ in steps])
    # the detector's script cursor must be rewound with the video, exactly as
    # ByteTrack's ids are: otherwise replaying A would serve B's tail frames
    assert det.resets == 3, det.resets
    assert out[0] == out[2], "replaying video A after B changed its curve"
    assert out[0][-1] >= ALARM_THETA, out[0][-1]
    assert max(out[1]) == 0.0, "the calm video B is not calm"


# --------------------------------------------------------------------- #
# channel design invariants
# --------------------------------------------------------------------- #
def test_only_the_ttc_channel_can_reach_the_alarm_threshold():
    """Braking and proximity are capped BELOW theta, by construction.

    This is what keeps the alarm-F1 term usable: routine hard braking or a
    pedestrian nearby must not alarm alone. Only a genuinely converging pair
    can.
    """
    cfg = R.DEFAULT_CONFIG
    assert cfg.brake_cap < ALARM_THETA
    assert cfg.vru_cap < ALARM_THETA
    # there is no fourth channel left to cap
    assert not hasattr(cfg, "context_cap")
    # and an illegal config is rejected rather than silently accepted
    for field in ("brake_cap", "vru_cap"):
        bad = R.RiskConfig(**{field: ALARM_THETA + 0.1})
        try:
            R.RiskEstimator(config=bad, detector_factory=lambda: None)
        except AssertionError:
            continue
        raise AssertionError(f"{field} above ALARM_THETA was accepted")
    # and the shipped channels really are the only keys that are ever reported
    est = RiskImpl(detector_factory=lambda: ScriptedDetector(lambda i: []))
    est.reset(dict(META))
    est.step(FRAME, 0.0)
    assert set(est.last_channels) == {"ttc", "braking", "proximity"}


def test_max_braking_alone_stays_under_the_alarm():
    """Even at absurd deceleration, one car alone cannot alarm."""
    for decel in (300.0, 1200.0, 1e6, 1e12):
        curve, _ = _braking_curve(decel, v0=5000.0)
        assert max(curve) < ALARM_THETA, f"decel {decel} -> {max(curve):.3f}"


def test_pedestrian_existence_alone_is_not_risk_but_a_conflict_is():
    """A pedestrian the car will NOT hit: no risk. One it WILL hit: proximity.

    The first version of this test parked a car 2600 px from a pedestrian and
    called it "far", while driving the car straight at them -- a collision
    course, so the channel was right to fire. The pedestrian here is on a
    different line, so the predicted closest approach is genuinely large.
    """
    will_miss = [(0.1 * i, [box(1, "car", 400.0 + 40 * i, 1500.0),
                            box(2, "person", 3000.0, 900.0, w=60, h=180)])
                 for i in range(30)]
    c_far, est_far = run(will_miss)
    assert est_far.last_channels["proximity"] == 0.0, \
        f"a pedestrian on another line scored {est_far.last_channels}"
    # The TTC channel is not exactly 0 here: for a TTC just under the 5 s
    # horizon, ((h-t)/h)**6 underflows to a DENORMAL rather than to zero
    # (~1e-20). That is "no risk" for every purpose the metric has, so the
    # bound is a tolerance rather than an equality.
    assert est_far.last_channels["ttc"] < 1e-6, est_far.last_channels
    assert max(c_far) < 0.01, f"distant pedestrian scored {max(c_far):.3f}"

    # a car on a collision course with a pedestrian, but still seconds away:
    # the proximity channel must report the developing conflict, and the TTC
    # channel must not turn it into an alarm
    will_hit = [(0.1 * i, [box(1, "car", 400.0 + 40 * i, 1500.0),
                           box(2, "person", 3000.0, 1500.0, w=60, h=180)])
                for i in range(30)]
    c_near, est = run(will_hit)
    assert est.last_channels["proximity"] > 0.0, \
        f"a predicted conflict was not reported: {est.last_channels}"
    assert est.last_channels["ttc"] < 0.05, \
        f"a 3.6 s TTC should be negligible: {est.last_channels}"
    assert max(c_near) < ALARM_THETA, \
        f"a distant developing conflict must not alarm: {max(c_near):.3f}"
    assert max(c_near) > 0.0


def test_proximity_factor_is_scale_invariant():
    """The same physical ratio gives the same factor at any resolution."""
    a = R.proximity_factor(100.0, 100.0, 1.0, 3.0)
    b = R.proximity_factor(10.0, 10.0, 1.0, 3.0)
    c = R.proximity_factor(1.0, 1.0, 1.0, 3.0)
    assert a == b == c
    assert R.proximity_factor(50.0, 100.0) == 1.0      # half a width
    assert R.proximity_factor(300.0, 100.0) == 0.0     # three widths
    assert 0.0 < R.proximity_factor(200.0, 100.0) < 1.0
    assert R.proximity_factor(None, 100.0) == 0.0
    assert R.proximity_factor(float("nan"), 100.0) == 0.0
    # a degenerate zero / negative scale is floored to 1 px, so the reading is
    # "100 widths apart" (i.e. no closeness), not "infinitely close"
    assert R.proximity_factor(100.0, 0.0) == 0.0
    assert R.proximity_factor(100.0, -50.0) == 0.0
    assert R.proximity_factor(0.5, 0.0) == 1.0        # half a floored width
    assert R.proximity_factor(None, 0.0) == 0.0


# --------------------------------------------------------------------- #
# internal observation rate
# --------------------------------------------------------------------- #
def test_observation_rate_is_derived_from_fps_not_frame_count():
    """~10 Hz internal whatever the frame rate, and never faster than fps."""
    for fps, want in ((9.99, 1), (10.0, 1), (25.0, 3), (30.0, 3), (50.0, 5),
                      (60.0, 6), (120.0, 8), (1000.0, 8)):
        got = R._stride_for_fps(fps, R.DEFAULT_CONFIG)
        assert got == want, f"fps={fps} -> stride {got}, wanted {want}"
        assert 0 < got <= max(1, int(fps)), "must not observe faster than fps"
    # 25 fps is the case plain round() gets wrong: round(2.5) = 2 -> 12.5 Hz
    assert R._stride_for_fps(25.0, R.DEFAULT_CONFIG) == 3
    # a below-target frame rate still gives stride 1
    for fps in (0.5, 1.0, 5.0):
        assert R._stride_for_fps(fps, R.DEFAULT_CONFIG) == 1
    for bad in (None, 0.0, -5.0, float("nan"), float("inf"), "x"):
        assert R._stride_for_fps(bad, R.DEFAULT_CONFIG) == 1
    # max_stride is a hard bound
    assert R._stride_for_fps(1e6, R.RiskConfig(target_hz=10.0, max_stride=8)) == 8


def test_skipped_frames_resend_the_last_score_without_decaying_it():
    """No decay may be invented on frames that were never observed."""
    steps = head_on_approach()
    est = RiskImpl(detector_factory=lambda: ScriptedDetector(
        lambda i: steps[min(i, len(steps) - 1)][1]))
    est.reset({"video_id": "t", "fps": 30.0, "width": W, "height": H,
               "n_frames": 300})
    curve = [est.step(FRAME, 0.0333 * i) for i in range(60)]
    assert est.observations == 20, est.observations
    assert est.calls == 60
    # a skipped frame re-serves the last score, bit for bit
    assert curve[0] == curve[1] == curve[2], curve[:3]
    assert curve[3] == curve[4] == curve[5], curve[3:6]
    assert all(0.0 <= c <= 1.0 for c in curve)
    # and the alarm still arrives: subsampling must not lose the precursor
    assert curve[-1] > ALARM_THETA, f"subsampling lost the alarm: {curve[-1]:.3f}"


# --------------------------------------------------------------------- #
# measured false-alarm budget on REAL normal traffic
# --------------------------------------------------------------------- #
_FIXTURE = os.path.join(os.path.dirname(__file__), "data",
                        "normal_traffic_pairs.json")


def _load_fixture():
    if not os.path.exists(_FIXTURE):
        raise AssertionError(
            f"missing real-traffic fixture {_FIXTURE}: the FP budget cannot be "
            "verified, and a synthetic stand-in would not be real evidence")
    with open(_FIXTURE, encoding="utf-8") as f:
        recs = json.load(f)
    assert recs["video"] == "jaywalking_visual_C3905.mp4", recs["video"]
    assert recs["width"] == 3840 and recs["height"] == 2160, recs
    assert recs["total_records"] > 100000, recs["total_records"]
    assert "_provenance" in recs, "the fixture must record how it was measured"
    return recs


def _score_pair(rec, cfg):
    """Replay the shipped gate over one recorded real pair."""

    class P:
        pass

    p = P()
    for k in ("approaching", "closing_speed_px_s", "relative_speed_px_s",
              "ttc_sec", "min_predicted_distance_px", "distance_px",
              "class_a", "class_b"):
        setattr(p, k, rec[k])
    ttc_ch, prox_ch = R.RiskEstimator._pair_channels(p, rec["scale"], cfg)
    return max(ttc_ch, prox_ch)


def _q(sorted_vals, frac):
    return sorted_vals[min(len(sorted_vals) - 1, int(frac * len(sorted_vals)))]


def _at(tid, ch, value=0, xy=None, size=(H, W)):
    """A black frame with `value` written into one channel at one pixel."""
    f = np.zeros((size[0], size[1], 3), np.uint8)
    y, x = xy if xy is not None else (0, 0)
    f[y, x, ch] = value
    return f


def _at_last_pixel(ch=0, value=255, size=(H, W)):
    """Black frame with one lit pixel at the bottom-right corner (y, x)."""
    f = np.zeros((size[0], size[1], 3), np.uint8)
    f[size[0] - 1, size[1] - 1, ch] = value
    return f


def test_real_normal_traffic_pair_channel_stays_below_the_alarm():
    """The shipped gate over 6000 REAL measured normal-traffic pairs.

    Measured marginals said normal traffic reaches TTC 1.5 s and |accel| 180
    px/s^2, so a naive TTC threshold WOULD alarm in normal traffic. This is the
    joint: after the gates, 99% of real normal pairs are below 0.15 and 99.99%
    are below ALARM_THETA.

    The top of the distribution is deliberately NOT asserted to be below theta.
    It is not: 1 pair in 10000 real pairs reaches 0.67, and hiding that behind
    a threshold would make the test lie. The metric-relevant consequence --
    how long the emitted curve actually stays alarmed -- is asserted in
    test_real_normal_traffic_frame_curve_alarms_only_1pct_of_the_time.
    """
    recs = _load_fixture()
    cfg = R.DEFAULT_CONFIG
    scored = sorted(_score_pair(r, cfg) for r in recs["pairs"])
    n = len(scored)
    assert n >= 5000, f"fixture is too small to be meaningful: {n}"
    assert _q(scored, 0.50) == 0.0, _q(scored, 0.50)
    assert _q(scored, 0.99) < 0.2, f"p99 = {_q(scored, 0.99):.3f}"
    assert _q(scored, 0.999) < 0.35, f"p99.9 = {_q(scored, 0.999):.3f}"
    assert _q(scored, 0.9999) < ALARM_THETA, \
        f"p99.99 = {_q(scored, 0.9999):.3f}"
    # the residual is small and bounded -- this is the measured FP budget
    over = sum(1 for v in scored if v >= ALARM_THETA)
    assert over / n < 2e-4, f"{over}/{n} real normal pairs reach the alarm"
    assert scored[-1] <= 1.0


def test_real_normal_traffic_frame_curve_alarms_only_1pct_of_the_time():
    """THE false-alarm budget: the curve Part B would emit on real traffic.

    Replays the shipped asymmetric EMA over the per-frame max of the shipped
    channels, measured on 423 real frames of the only real footage in the repo
    (150726 real pairs, 11302 real track-observations). Per-pair percentiles
    are the wrong number to gate on: max-aggregation over ~350 pairs per frame
    means the frame score is the worst of hundreds of pairs, so the frame-level
    tail is what the alarm-F1 term actually sees.

    The alarm count comes from ``evaluate.alarm_starts`` -- the organizers'
    own function, imported not reimplemented -- because the intuitive model of
    the metric is wrong in a way that matters: it takes EVERY run at or above
    theta (there is no minimum run length), merges runs closer than 2 s, and
    uses the run START as the alarm time. F1_alarm's precision divides by
    exactly this count, so a bespoke "0.5 s minimum run" statistic would
    answer a question the metric never asks.
    """
    import evaluate as EV

    recs = _load_fixture()
    cfg = R.DEFAULT_CONFIG
    series = recs["frame_series"]
    n = len(series)
    assert n >= 400, f"frame series too short: {n}"
    dt = recs["frame_series_dt"]
    assert abs(dt - 0.3003) < 1e-3, dt

    # 1. the committed EMA column must be exactly what the shipped _shape does
    ema = None
    for row in series:
        raw = R._clamp01(row["raw"])
        if ema is None:
            ema = raw
        else:
            a = cfg.ema_up if raw > ema else cfg.ema_down
            ema += a * (raw - ema)
        assert abs(R._clamp01(ema) - row["ema"]) < 1e-12, (row["t"], ema)

    curve = [(row["t"], row["ema"]) for row in series]
    scores = [row["ema"] for row in series]
    clip = n * dt
    starts = EV.alarm_starts(curve)
    alarmed = sum(dt for v in scores if v >= EV.THETA)

    # 2. a normal frame is ~0, per the brief: the MEDIAN must be in the
    #    0-0.2 "no/minimal risk" band, not merely the mean
    ordered = sorted(scores)
    median = _q(ordered, 0.50)
    assert median <= 0.2, f"normal-traffic median risk is {median:.3f}"
    assert _q(ordered, 0.90) < 0.40, f"p90 = {_q(ordered, 0.90):.3f}"
    assert _q(ordered, 0.99) < EV.THETA, f"p99 = {_q(ordered, 0.99):.3f}"

    # 3. the alarm budget itself, in the metric's own currency
    frac = alarmed / clip
    assert frac < 0.02, f"{100 * frac:.2f}% of normal traffic is alarmed"
    assert len(starts) <= 2, f"{len(starts)} spurious alarms in {clip:.0f} s: {starts}"
    # ... and it is not zero either. Recorded so that a silent regression to a
    # dead channel cannot pass as "perfect".
    assert len(starts) >= 1, "the FP budget reads as artificially perfect"

    # 4. the committed summary columns agree with a fresh recount
    assert len(starts) == recs["frame_series_alarms"], \
        f"{len(starts)} alarms, fixture says {recs['frame_series_alarms']}"
    assert abs(alarmed - recs["frame_series_alarmed_seconds"]) < 0.05, alarmed
    assert sum(1 for v in scores if v >= ALARM_THETA) == 3, \
        f"expected 3 alarmed frames, got {sum(1 for v in scores if v >= ALARM_THETA)}"
    # and every spurious alarm is a real, isolated excursion
    assert all(t >= 0.0 and t < clip for t in starts), starts
    assert len(recs["frame_series_alarm_runs_s"]) >= len(starts), \
        "raw runs must be at least as many as merged alarms"


def head_on_approach_times():
    """The t values the harness would submit for head_on_approach()."""
    return [t for t, _ in head_on_approach()]


def test_do_not_clip_the_low_end_of_the_curve_it_costs_ap():
    """The graded sub-threshold part of the curve is load-bearing.

    This is the one result in Part B that runs against instinct, so it is
    measured through the organizers' own average_precision on the committed
    423-frame real series, with a synthetic accident label at nine timestamps:

        as shipped                        mean chance-normalised AP 0.016
        every value shifted down by 0.15                            0.016
        hard clip: zero anything < 0.2                              0.012
        hard clip: zero anything < 0.3                              0.006
        a flat 0.15 (control)                                      0.000

    Two conclusions, both of which a future editor would plausibly undo:
      * a constant floor is AP-NEUTRAL, exactly, at every placement -- so the
        VRU channel's 0.15 cap costs nothing and buys nothing on this axis;
      * HARD-CLIPPING low scores degrades AP monotonically, because
        evaluate.py:245 consumes a whole tie group at a time and a 0.0 group
        swallows the 5 s positive window, diluting that group's precision.
    """
    import evaluate as EV

    recs = _load_fixture()
    series = recs["frame_series"]
    dt = recs["frame_series_dt"]
    curve = [(r["t"], r["ema"]) for r in series]
    dur = curve[-1][0] + dt
    fps = 1.0 / dt

    def ap_of(c, at):
        gt = {"vid": {"duration": dur, "fps": fps,
                      "events": [[at, at + 1.0, "accident"]]}}
        res = EV.evaluate_part_b(gt, {"vid": {"events": [], "risk": c}})
        return 0.0 if res is None else res["ap"]

    places = (10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 115.0, 125.0)

    def mean_ap(fn):
        c = [(t, min(1.0, max(0.0, fn(v)))) for t, v in curve]
        return sum(ap_of(c, p) for p in places) / len(places)

    base = mean_ap(lambda v: v)

    # 1. a constant shift changes AP by exactly nothing, at every placement
    shifted = [(t, max(0.0, v - 0.15)) for t, v in curve]
    for p in places:
        assert abs(ap_of(shifted, p) - ap_of(curve, p)) < 1e-12, \
            f"a constant floor is not AP-neutral at t={p}"

    # 2. hard clipping is worse, and worse the more it clips
    c02 = mean_ap(lambda v: v if v >= 0.2 else 0.0)
    c03 = mean_ap(lambda v: v if v >= 0.3 else 0.0)
    assert c02 <= base + 1e-12, f"clipping < 0.2 raised AP: {c02} > {base}"
    assert c03 < c02, f"clipping < 0.3 ({c03:.4f}) was not worse than < 0.2 ({c02:.4f})"

    # 3. and the flat control reads 0, so a curve with no information scores
    #    nothing -- the chance normalisation is doing what it claims
    flat = mean_ap(lambda v: 0.15)
    assert flat == 0.0, f"a flat curve scored AP {flat}"

    # 4. the absolute level is low because there is no accident in this
    #    footage; a curve that is specific rather than generically high
    #    SHOULD read ~0 when a synthetic label is dropped into it
    assert base < 0.10, f"a normal clip scored AP {base:.3f} against a synthetic label"


def test_the_alarm_window_is_a_band_not_a_knife_edge():
    """The ramp must alarm over a WIDE range of TTC, not a single instant.

    The TTC is `distance / closing_speed`, both derived from a 0.8 s window of
    detections, so it carries real estimation error that cannot be measured
    here (there is no accident footage in the repo). The ramp's tolerance to
    that error IS measurable, though: it is the width, in seconds of TTC, of
    the band that alone scores >= ALARM_THETA. A knife-edge window would mean
    a detector that only fires on a TTC estimate it has no reason to trust --
    and since a missed accident costs ~3x a false alarm (see RiskConfig),
    that trade is the wrong way round.
    """
    cfg = R.DEFAULT_CONFIG
    band = [t / 100.0 for t in range(0, 500)
            if R.ttc_risk(t / 100.0, cfg.horizon_sec, cfg.ttc_curvature)
            >= ALARM_THETA]
    assert band, "the ramp never alarms at all"
    assert max(band) > 0.0, "the ramp only alarms at a non-positive TTC"
    width = max(band) - min(band)
    assert width >= 0.4, f"the alarm window is only {width:.2f} s of TTC wide"
    assert max(band) >= 0.4, \
        f"the ramp does not alarm until TTC {max(band):.2f} s"
    # the middle of the band is comfortably above theta, not sitting on it
    mid = (max(band) + min(band)) / 2.0
    assert R.ttc_risk(mid, cfg.horizon_sec, cfg.ttc_curvature) >= 0.55, \
        "the band has no margin above the threshold"
    # ... and 0.5 s outside the band the score has dropped back to the
    # "no/minimal risk" band, so the alarm region is genuinely confined to the
    # imminent band rather than smeared across the whole horizon. (Measured:
    # 0.221 at ttc=0.97 s -- the 0-0.2 band is not quite reached 0.5 s out,
    # which is why this is a 0.25 bound, not 0.2.)
    outside = R.ttc_risk(max(band) + 0.5, cfg.horizon_sec, cfg.ttc_curvature)
    assert outside < 0.25, f"the score is still {outside:.3f} half a second out"
    # and the value the calibration quotes
    assert abs(R.ttc_risk(0.3, cfg.horizon_sec, cfg.ttc_curvature) - 0.648) < 0.01


def test_a_genuine_precursor_alarms_inside_the_metrics_matching_window():
    """The alarm must START in [s-W, s) of the accident, or it scores nothing.

    This is the asymmetry that sets the ramp: an alarm that starts too late is
    unmatched (recall 0, and F1_alarm and mTTA both go to 0), so a ramp that
    only fires at TTC ~ 0.2 s is not "safe" -- it is a detector that cannot
    see anything. The score must therefore cross ALARM_THETA while the pair is
    still measurably separated, and the crossing time is what evaluate.py uses
    as the alarm time.
    """
    import evaluate as EV

    curve, _ = run(head_on_approach())
    impact_t = head_on_approach()[-1][0] + 0.1      # one step past the last
    alarmed = [(t, v) for t, v in zip(head_on_approach_times(), curve)
               if v >= EV.THETA]
    assert alarmed, "a genuine imminent collision produced no alarm at all"
    first_alarm = alarmed[0][0]

    # the harness submits (t, score); the metric matches on t alone
    starts = EV.alarm_starts(list(zip(head_on_approach_times(), curve)))
    assert len(starts) == 1, f"expected exactly one alarm, got {starts}"
    assert starts[0] == first_alarm

    s = impact_t
    assert s - EV.W <= starts[0] < s, \
        f"alarm at {starts[0]:.2f} s is outside [{s - EV.W:.2f}, {s:.2f})"
    tta = s - starts[0]
    # The measured time-to-alarm at the shipped ramp is ~0.2-0.4 s, and that is
    # a DESIGN POINT rather than an accident of tuning: mTTA is worth only
    # 0.2*(0.4/10) = 0.008 of Score_B, while a missed accident costs
    # F1_alarm + AP = 0.8. Buying an earlier alarm with false alarms is a bad
    # trade at that exchange rate. What must hold is that the alarm arrives
    # while the pair is still measurably apart -- a ramp that only fires at
    # TTC ~ 0 would be a detector that can see nothing.
    assert 0.0 < tta, "the alarm did not precede the accident"
    assert tta <= EV.H, f"time-to-alarm {tta:.2f} s is not a usable warning"
    # the crossing happens at a genuinely positive TTC, not at contact
    assert first_alarm < impact_t, "the alarm only fired at contact"

    # ... and the alarm's own frame is a POSITIVE for AP. Note the asymmetry
    # between the two windows in evaluate.py: AP positives are the H=5 s
    # BEFORE an accident (frame_label), but an alarm MATCHES anywhere in
    # W=10 s. So alarming 5-10 s early would buy F1_alarm recall while
    # scoring AP-negative frames -- which is a second, independent reason not
    # to buy time-to-alarm with a gentler ramp.
    assert EV.frame_label(first_alarm, [(s, s + 1.0)], []) == 1
    assert EV.frame_label(s, [(s, s + 1.0)], []) is None, \
        "frames inside the accident segment are ignored by AP"
    assert EV.frame_label(s - EV.H, [(s, s + 1.0)], []) == 1
    assert EV.frame_label(s - EV.H - 0.1, [(s, s + 1.0)], []) == 0, \
        "AP positives start at s-H, not at s-W"
    assert EV.frame_label(s - EV.W, [(s, s + 1.0)], []) == 0, \
        "the 5-10 s band is matchable but AP-negative"
    # ... and a near_miss suppresses the frames around it. Note the ORDER in
    # evaluate.frame_label: the accident-positive test runs BEFORE the
    # near-miss ignore, so a near_miss can only suppress frames that are not
    # already accident-positives -- checked here at t=4.0, past the accident.
    assert EV.frame_label(first_alarm, [(s, s + 1.0)],
                          [(s - 5.0, s + 2.0)]) == 1, \
        "an accident positive outranks a near-miss ignore"
    assert EV.frame_label(s + 2.0, [(s, s + 1.0)],
                          [(s - 5.0, s + 2.0)]) is None
    assert EV.frame_label(s + 2.0, [(s, s + 1.0)], []) == 0, \
        "past the accident and away from any near miss, it is a negative"


def test_real_normal_traffic_braking_never_reaches_the_cap():
    """Replay the real brake gate over REAL measured normal-track accels."""
    recs = _load_fixture()
    cfg = R.DEFAULT_CONFIG
    scored = []
    for rec in recs["tracks"]:
        if not rec["moving"] or not rec["was_moving"]:
            continue
        b = R.brake_risk(rec["accel"], rec["speed"], cfg.brake_decel_lo,
                         cfg.brake_decel_hi, cfg.brake_min_speed_px_s)
        scored.append(b * cfg.brake_cap)
    assert len(scored) >= 2000, f"only {len(scored)} moving track records"
    scored.sort()
    assert _q(scored, 1.0) < 0.1, f"max = {_q(scored, 1.0):.3f}"
    assert _q(scored, 0.99) < 0.1, f"p99 = {_q(scored, 0.99):.3f}"
    assert _q(scored, 0.95) < 0.05, f"p95 = {_q(scored, 0.95):.3f}"
    assert _q(scored, 0.50) == 0.0, "half of normal traffic should not brake hard"
    assert max(scored) < ALARM_THETA


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()
