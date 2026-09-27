"""Part B — causal accident anticipation (PHASE 23).

``step`` returns P(an ``accident`` event starts within the next
``RISK_HORIZON_SEC`` seconds) using ONLY frames already delivered to it. It
never opens the video, never calls ``detect_events()`` / ``run_pipeline()`,
never reads a future frame, and never reuses a Part A event segment.

The perception stack is the one Part A already uses — there is no second TTC
engine, no second tracker and no second model instance:

    frame -> Detector.track -> TrajectoryEngine -> MotionEngine
                                      |                |
                                      +-> PairwiseInteractionEngine -> PairInteraction

Part B owns its OWN trajectory / motion / pair engines, so nothing it computes
depends on what Part A did. The only thing shared with Part A is the loaded
model object itself (a model CACHE, so weights load once per process instead
of once per video); a test spies on both Part A entry points to prove neither
is ever reached from ``step``.

Three causal channels are combined with ``max`` (never a mean, so one dangerous
pair cannot be diluted by a crowd of safe objects -- brief §9):

  1. ``ttc``       per converging pair. A monotone ramp of the pair's
                   Time-To-Collision across the 5 s horizon, gated on real
                   relative motion and on the pair not being two ByteTrack ids
                   on one vehicle, then scaled by how likely the pair is to
                   actually MEET (a pair with TTC 1 s that will miss by ten car
                   widths is not a crash).
  2. ``braking``   per track. Absolute deceleration in px/s^2 at full
                   resolution, calibrated on the measured normal-traffic
                   acceleration distribution. Sampled only between two MOVING
                   observations, because MotionEngine's raw speed difference
                   has a large step artifact at a stop/start transition, then
                   causally EMA-smoothed per track so one noisy frame cannot
                   fire it.
  3. ``proximity`` vehicle<->vulnerable-road-user pairs whose PREDICTED closest
                   approach is small (both currently slow, say). A pedestrian
                   EXISTING is not risk; a car on a trajectory that meets one
                   is. Deliberately NOT gated on the closing-speed test, which
                   is the whole reason this channel exists.

There is deliberately NO fourth "wrong-way candidate" channel. It was built and
then removed on measurement: the only heading reference available is
``Scene.defaults_estimated(...).dominant_flow_deg``, an UNCALIBRATED dev default
(AGENTS.md: only road + crosswalk are enabled; lanes are off), and on this
footage it fires at its full cap on plain single-direction traffic. Its
marginal value is also near zero, because a wrong-way vehicle meeting
oncoming traffic produces exactly the low-TTC converging pair channel 1
already scores. Shipping a channel that fires on ordinary traffic to catch a
signal it does not need was not a trade worth making.

Channels 2-3 are capped BELOW ``ALARM_THETA`` (0.5) on purpose. Hard braking and
a pedestrian nearby are both routine in a city intersection; on their own they
are "developing risk" (< 0.5, the 0.2-0.5 band of the brief's §12 calibration).
Only a genuinely converging pair can raise an alarm. That is what keeps the
alarm-F1 term usable — one stable alarm before the accident instead of a
permanently high score.

Temporal shaping is a single asymmetric EMA (fast attack, slow release), which
looks strictly backwards. There is deliberately NO max-hold / peak-extension
stage: stretching a one-frame spike above 0.5 for long enough to clear the
metric's 0.5 s minimum-run filter would manufacture alarm runs out of
detection glitches, costing exactly the precision the metric rewards. A genuine
precursor persists for many observations at 10 Hz, so it clears that filter on
its own.

Observations are subsampled to ``target_hz`` (default 10 Hz) *derived from*
``meta['fps']``, not from a frame count: the samples in this repo run at
9.99 fps while a real sample is likely 25-30 fps, so one fixed frame stride
would give 3.3 Hz on the first and 10 Hz on the second, and MotionEngine's
0.8 s window would hold 3 points instead of 8. On frames between observations
the last score is re-served unchanged (no decay is invented on frames we did
not observe), which is what the original module docstring sanctioned.

Conventions, identical to the rest of the repo:
  * positions are bottom-centers in FULL-resolution pixels;
  * speeds / accelerations are px/s and px/s^2 at full resolution;
  * heading is the MotionEngine convention (0 = +x, 90 = +y up, [0,360));
  * the output is a plain float in [0, 1], never NaN / inf / None.

Determinism: no randomness, no clock, no I/O beyond the shared model, and every
aggregation iterates a deterministically ordered sequence.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from ..config import budget
from ..config.settings import settings
from ..tracking import (Detection, MotionEngine, PairwiseInteractionEngine,
                        TrajectoryEngine)

# Anticipation horizon used by the metric (seconds) and the alarm threshold.
RISK_HORIZON_SEC = 5.0
ALARM_THETA = 0.5

# Pair eligibility, matching the composition near_miss uses. Only
# {car, truck, bus, motorcycle, person} is actually reachable: the detector
# class filter drops COCO bicycle (id 1) upstream, so "bicycle" here is inert.
VEHICLE_LABELS = frozenset({"car", "truck", "bus", "motorcycle"})
VULNERABLE_LABELS = frozenset({"person", "bicycle"})
CANDIDATE_LABELS = VEHICLE_LABELS | VULNERABLE_LABELS


def _read_target_hz() -> float:
    """Part B's internal observation rate, from TCV_RISK_TARGET_HZ or 10 Hz.

    Read once, here, so that a RiskConfig built later cannot see a different
    rate from one built earlier -- the value has to be a module constant
    because a frozen dataclass evaluates its defaults at class-creation time.
    An unparseable or non-positive value falls back to 10 Hz rather than
    degenerating the derived frame stride on a pathological video.
    """
    raw = os.environ.get("TCV_RISK_TARGET_HZ", "")
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 10.0
    return v if v > 0.0 else 10.0


_TARGET_HZ_DEFAULT = _read_target_hz()


@dataclass(frozen=True)
class RiskConfig:
    """Every Part B knob in one immutable, injectable place.

    The defaults are set so that NORMAL traffic measured on the only real
    footage in this repo (the 9.99 fps / 3840x2160 debug render of sample
    C3905) yields risk ~ 0. Measured over 423 sampled frames / 136059 relevant
    pairs / 11302 track-observations, the shipped configuration produces a
    per-frame curve (the ``max`` over pairs, i.e. what Part B aggregates) of

        raw  p50 0.150  p90 0.317  p99 0.547  max 0.629
        ema  p50 0.178  p90 0.345  p99 0.470  max 0.554
        3 of 423 frames at or above ALARM_THETA -> 0.90 s of a 127 s clip,
        which evaluate.alarm_starts counts as 2 spurious alarms (t=23.4 s
        and t=117.4 s, far enough apart not to merge)
        i.e. 0.7% of a normal clip alarmed, and the curve's MEDIAN -- which is
        what the brief's "a normal frame is ~0" actually means -- is 0.178,
        inside the 0-0.2 no/minimal-risk band.

    All alarm statistics here are counted with the organizer's OWN
    ``evaluate.alarm_starts``, not with an approximation. That matters, because
    the intuitive model of the metric is wrong in one specific way:
    ``alarm_starts`` (evaluate.py:257) takes EVERY maximal run at or above
    theta -- there is no minimum run length, so a 0.3 s alarm counts -- merges
    runs whose gap is under 2 s, and uses the run START as the alarm time.
    So what F1_alarm's precision divides by is the NUMBER OF RUNS, and mTTA
    rewards alarming early.

    The same measurement is what rejected two earlier designs, both recorded
    here because "why is this number what it is" is the whole point of a
    calibration:

    * ``ttc_curvature=1.6`` (a gentle ramp): median 0.54, 66% of the clip in
      alarm, 19 runs in the 127 s clip. Root cause is not the ramp being wrong
      but the scene: normal urban driving contains many 1-second-TTC
      convergences that the drivers resolve, and brief §9 mandates
      max-aggregation, so with ~350 pairs per frame the worst of hundreds of
      ordinary pairs becomes the frame score. mTTA cannot pay for fixing it:
      the alarm start is under 1 s out at every curvature measured.
    * A "already in contact -> 1.0" shortcut gated on
      ``distance_px <= 0.5 * larger box width``: it fired on 1436 of 150726
      real pairs (~1%) and pushed p99 of the channel to a flat 1.0. A person
      is a tall narrow box, so two people a stride apart legitimately put
      their bottom-centres inside half a width.

    What remains above theta are ~25 real pairs with genuine separation
    (0.7-1.9 widths) and genuine closing speed (247-560 px/s) at TTC
    0.6-1.2 s. Whether those are near-misses or perception artifacts cannot be
    settled here: there is no footage of an accident in this repo and images
    cannot be inspected. They are the residual false-alarm budget, and they are
    the reason the ramp is steep.

    Do NOT "tidy" the curve by zeroing small scores. Measured through
    evaluate.average_precision on the committed 423-frame series, with a
    synthetic accident label at nine different timestamps (mean chance-
    normalised AP):

        as shipped                       0.016
        every value shifted down by 0.15  0.016   (identical at every placement)
        hard clip: zero anything < 0.2    0.012
        hard clip: zero anything < 0.3    0.006
        hard clip: zero anything < 0.05   0.016   (below the minimum, no-op)
        a flat 0.15 (control)             0.000

    So the low, graded end of the curve is load-bearing: clipping it costs AP
    monotonically, because evaluate consumes a whole tie group at a time
    (evaluate.py:245, the sklearn definition) and a 0.0 tie group swallows the
    5 s positive window and dilutes that group's precision. Adding or removing
    a constant floor, by contrast, changes AP by exactly nothing. And the flat
    control reading 0.000 is the documented behaviour of the chance
    normalisation, confirmed rather than assumed.

    The absolute values are small (0.016) because this is footage with no
    accident in it and a synthetic label dropped into it: a curve that is
    specific rather than generically high SHOULD score ~0 there. That is the
    point of the number, not a defect in it.
    """

    # ---- sampling -------------------------------------------------------
    # Internal observation rate, derived from meta['fps'] rather than fixed as a
    # frame count (see _stride_for_fps). This is also the single largest cost
    # lever in Part B, and the one that matters most, because the harness
    # BLANKS THE WHOLE ENTRY when Part A + Part B exceed 3x the video duration
    # (run_submission.py:196) -- so Part B being slow does not just degrade
    # Part B, it throws away Part A too.
    #
    # Measured, 12 s / 120-frame slice of the 9.99 fps debug render, CPU-only
    # torch (593 ms per YOLO call): Part A (stride 3) 23.7 s, Part B (stride
    # 1) 60.4 s. Part B is 2.5x Part A precisely because Part A subsamples at
    # 3 and this rate resolves to 1 at 9.99 fps. On a 25-30 fps sample the
    # derived stride is 3 and the two parts cost the same. With a working CUDA
    # torch (45 ms/call, AGENTS.md) a 3-minute video is ~20-25% of budget
    # either way; on CPU-only it is ~230%, which is why this knob exists.
    #
    # TCV_RISK_TARGET_HZ is an EMERGENCY lever, not a tuning knob: set it to 5
    # to halve Part B's detector calls if the grading machine turns out to have
    # no working GPU. The cost is real and measured -- at 5 Hz there are 0.2 s
    # between observations, and MotionEngine's 0.8 s window then holds 4
    # points instead of 8, so velocity estimates get noisier while the ramp's
    # 0.47 s alarm window absorbs ~0.15 s of TTC error. Prefer fixing the
    # device over turning this down.
    target_hz: float = _TARGET_HZ_DEFAULT
    max_stride: int = 8            # bound on the derived frame stride
    max_tracks: int = 40           # bounds the O(N^2) pair scan

    # ---- channel 1: TTC -------------------------------------------------
    horizon_sec: float = RISK_HORIZON_SEC
    # Ramp exponent -- the single most consequential number in Part B, and set
    # from measurement rather than taste. Measured on real normal traffic, with
    # the spurious-alarm count taken from the organizer's OWN
    # evaluate.alarm_starts (see the table in RiskConfig's docstring):
    #   curv   FP alarms/127 s   TTA     alarm window   ttc_risk(0.3 s)
    #   3.0         11          0.95 s      1.03 s          0.831
    #   5.0          5          0.60 s      0.64 s          0.734
    #   6.0          3          0.50 s      0.54 s          0.690
    #   7.0          2          0.40 s      0.47 s          0.648   <-- shipped
    #   8.0          1          0.35 s      0.41 s          0.610
    #  10.0          0          0.25 s      0.33 s          0.539
    # 13.0+         0          <0.2 s      <0.25 s         <0.47
    # "alarm window" is the width, in seconds of TTC, of the band that alone
    # scores >= ALARM_THETA: it is the ramp's tolerance to error in the
    # velocity-derived TTC.
    #
    # Why not steeper, given that FP alarms are the dominant term: the loss is
    # ASYMMETRIC and the asymmetry is measured. Two extra spurious alarms cost
    # 0.4 * (2/3) = 0.27 of F1_alarm. Missing the accident outright costs
    # F1_alarm -> 0 AND AP -> 0, i.e. 0.4 + 0.4 = 0.80 of Score_B. So a false
    # alarm is roughly three times cheaper than a miss, and the ramp must not
    # be steepened past the point where the TTC estimate can miss the window.
    # 7.0 keeps 0.47 s of window and a 30% score margin at TTC 0.3 s, and takes
    # most of the available precision gain.
    #
    # Why not gentler: the brief's "gentle" ramp (1.6) puts the per-frame
    # median on real normal traffic at 0.54 and 66% of the clip in alarm,
    # because max-aggregation over ~350 ordinary pairs per frame lifts the
    # worst of hundreds of resolved 1-second-TTC convergences into the frame
    # score. mTTA cannot pay for that: the alarm start is under 1 s out at
    # EVERY curvature measured, so mTTA is worth 0.004-0.019 of Score_B across
    # the entire range and cannot buy anything.
    ttc_curvature: float = 7.0
    # "is there any real relative motion at all" noise guards (px/s)
    min_closing_px_s: float = 5.0
    min_relative_px_s: float = 8.0
    # A pair whose bottom-centres are closer than this fraction of the larger
    # box width is a DUPLICATE/coincident track, not a traffic conflict: two
    # ByteTrack ids on one vehicle, two boxes on one occluded object. Measured
    # on real normal traffic, the highest-scoring pairs are exactly this shape
    # (truck/truck at dist 1-2 px with scale 290 px). Rejecting them costs
    # nothing: a genuine conflict always has real separation first.
    min_sep_widths: float = 0.15
    # "will they actually meet", in units of the LARGER object's box width
    prox_full_widths: float = 1.0  # predicted closest approach <= 1 width -> 1.0
    prox_zero_widths: float = 3.0  # >= 3 widths -> 0.0

    # ---- channel 2: braking ---------------------------------------------
    # px/s^2, calibrated on the measured normal-traffic acceleration
    # distribution (see brake_risk). Normal traffic: p1 = -180, p99.9 = -450.
    brake_decel_lo: float = 300.0
    brake_decel_hi: float = 1200.0
    brake_alpha: float = 0.5       # per-track causal EMA on the brake signal
    brake_cap: float = 0.45        # < ALARM_THETA on purpose
    brake_min_speed_px_s: float = 80.0  # above normal speed p50 (23 px/s)

    # ---- channel 3: vulnerable-road-user proximity ----------------------
    # Measured on the VRU PREDICTED closest approach, not the raw separation:
    # "a car is currently beside a pedestrian" fires for any pedestrian on the
    # pavement, while "this car is on a trajectory that meets the pedestrian" is
    # about conflict. On a hand-checked pair set the predicted form separates
    # the two cases completely (0.350 in-path vs 0.000 on the pavement) where
    # raw separation leaked 0.084 into the safe case.
    vru_full_widths: float = 0.6
    vru_zero_widths: float = 1.6
    # Capped low on purpose, and it is the measured reason the real
    # normal-traffic median is 0.15 rather than 0.35: with a pedestrian simply
    # present in a busy 4K intersection the old cap of 0.35 pinned the whole
    # curve in the "developing risk" band. 0.15 keeps a normal frame inside the
    # brief's 0-0.2 "no/minimal risk" band, and it can never raise an alarm.
    vru_cap: float = 0.15          # < ALARM_THETA on purpose

    # ---- temporal shaping ----------------------------------------------
    ema_up: float = 0.55           # attack: reach real danger quickly
    ema_down: float = 0.25         # release: decay slower than it attacks


DEFAULT_CONFIG = RiskConfig()


def _clamp01(value: object) -> float:
    """Total order on the output: a plain float in [0, 1].

    Equivalent to ``float(np.clip(v, 0.0, 1.0))`` but rejects NaN / inf /
    non-numerics explicitly instead of propagating them (``np.clip`` passes
    NaN straight through). Pure ``math`` because this runs once per frame.
    """
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    if f <= 0.0:
        return 0.0
    if f >= 1.0:
        return 1.0
    return f


def _finite(value: object) -> float | None:
    """``float(value)`` when it is a real finite number, else ``None``."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def ttc_risk(ttc_sec: object, horizon_sec: float = RISK_HORIZON_SEC,
             curvature: float = 7.0) -> float:
    """Map a Time-To-Collision to [0, 1] over the anticipation horizon.

    ``None`` / NaN / +inf (separating, parallel, or zero relative velocity) and
    anything at or beyond the horizon give exactly 0.0; TTC <= 0 (already
    collocated) gives 1.0. In between it rises monotonically. At the shipped
    ``curvature=7.0`` the curve is deliberately concentrated in the genuinely
    imminent sub-second band:

        ttc=5.0 -> 0.00   ttc=3.0 -> 0.00   ttc=2.0 -> 0.03
        ttc=1.0 -> 0.21   ttc=0.6 -> 0.41   ttc=0.3 -> 0.65
        ttc=0.1 -> 0.87   ttc=0.0 -> 1.00

    The 0.5 crossing sits at TTC ~ 0.45 s, so a pair alone alarms over a
    0.47 s-wide window of TTC values -- that width is the ramp's tolerance to
    error in the velocity-derived TTC, and it is the reason the exponent is
    7.0 and not 10.0 (see RiskConfig).
    """
    t = _finite(ttc_sec)
    if t is None:
        return 0.0
    h = _finite(horizon_sec)
    if h is None or h <= 0.0:
        return 0.0
    if t >= h:
        return 0.0
    if t <= 0.0:
        return 1.0
    c = _finite(curvature)
    return _clamp01(((h - t) / h) ** (c if c is not None else 1.0))


def proximity_factor(distance_px: object, scale_px: object,
                     full_widths: float = 1.0,
                     zero_widths: float = 3.0) -> float:
    """Scale-aware closeness in [0, 1].

    1.0 at/below ``full_widths`` object widths, 0.0 at/above ``zero_widths``
    widths, linear in between. Object widths rather than raw pixels, so the
    same thresholds mean the same thing for a near car and a distant truck,
    and for a 640-wide and a 3840-wide frame.
    """
    d = _finite(distance_px)
    if d is None:
        return 0.0
    d = max(d, 0.0)
    s = _finite(scale_px)
    s = max(s if s is not None else 0.0, 1.0)
    full, zero = _finite(full_widths), _finite(zero_widths)
    full = 1.0 if full is None else full
    zero = 3.0 if zero is None else zero
    if zero <= full:
        return 1.0 if d <= s * full else 0.0
    if d <= s * full:
        return 1.0
    if d >= s * zero:
        return 0.0
    return (zero - d / s) / (zero - full)


def brake_risk(accel_px_s2: object, speed_px_s: object,
               decel_lo: float = 300.0, decel_hi: float = 1200.0,
               min_speed_px_s: float = 80.0) -> float:
    """Braking severity in [0, 1] from absolute deceleration (px/s^2).

    Accelerating or coasting gives exactly 0.0. A near-zero speed is refused:
    below ``min_speed_px_s`` the track is stopped or queued, and its
    deceleration is measurement noise rather than braking.

    Thresholds are calibrated on real normal traffic (the C3905 render), not
    guessed. Measured signed acceleration there: p1 = -180, p5 = -77,
    p50 = -0.3, p99 = +204 px/s^2, over 6317 track-observations. So
    ``decel_lo = 300`` sits above the whole p1..p99.9 body of normal traffic
    (only its extreme tail exceeds it) and ``decel_hi = 1200`` is well beyond
    anything normal traffic produced (|accel| p99.9 = 450).

    An earlier dimensionless form, ``-accel / speed`` (1/s), was measured to be
    unusable on this footage and was removed: normal traffic here has speed
    p50 = 23 px/s (half the tracks are stopped or queued), so the ratio blows
    up as speed approaches the guard and a crawling car produced a full-scale
    braking alarm. Absolute px/s^2 is also the scale the rest of the repo uses
    (see near_miss's px/s gates).
    """
    a = _finite(accel_px_s2)
    sp = _finite(speed_px_s)
    if a is None or sp is None or a >= 0.0:
        return 0.0
    if sp <= _as_float(min_speed_px_s, 0.0):
        return 0.0
    lo = _as_float(decel_lo, 0.0)
    hi = _as_float(decel_hi, lo + 1.0)
    decel = -a
    if decel <= lo:
        return 0.0
    if decel >= hi:
        return 1.0
    return _clamp01((decel - lo) / (hi - lo))


def _as_float(value: object, default: float) -> float:
    f = _finite(value)
    return default if f is None else f


def _pair_relevant(label_a: object, label_b: object) -> bool:
    """vehicle<->vehicle or vehicle<->vulnerable, mirroring near_miss."""
    a = label_a in VEHICLE_LABELS
    b = label_b in VEHICLE_LABELS
    if a and b:
        return True
    return bool((a and label_b in VULNERABLE_LABELS)
                or (b and label_a in VULNERABLE_LABELS))


def _has_vulnerable(label_a: object, label_b: object) -> bool:
    a = label_a in VEHICLE_LABELS
    b = label_b in VEHICLE_LABELS
    return bool((a and label_b in VULNERABLE_LABELS)
                or (b and label_a in VULNERABLE_LABELS))


def _stride_for_fps(fps: object, cfg: RiskConfig,
                    target_hz: float | None = None) -> int:
    """Frame stride yielding ~``target_hz`` internal observations.

    Frame-count strides are wrong here: the samples in this repo run at
    9.99 fps while a real sample is likely 25-30 fps, so one fixed stride would
    give 3.3 Hz on the first and 10 Hz on the second, and MotionEngine's 0.8 s
    window would hold 3 points instead of 8.

    When ``fps / target_hz`` is not a whole number, ``round`` is the wrong
    tool: at 25 fps it returns 2, i.e. 12.5 Hz -- 25% over target, and enough
    to change the shape of MotionEngine's velocity fit. Both neighbouring
    strides are compared and the one whose resulting rate is CLOSEST to the
    target wins, so 25 fps gives 3 (8.3 Hz), not 2 (12.5 Hz).
    """
    f = _finite(fps)
    # `target_hz` is an override so the PHASE 25 budget guard can lower the
    # rate for one video without mutating the shared RiskConfig.
    target = cfg.target_hz if target_hz is None else float(target_hz)
    if f is None or f <= 0.0 or target <= 0.0:
        return 1
    lo = max(1, int(math.floor(f / target)))
    hi = max(1, int(math.ceil(f / target)))
    best = min((lo, hi), key=lambda s: (abs(f / s - target), s))
    return max(1, min(int(cfg.max_stride), best))


# Part B's internal observation rate. Read once, at import, so a single
# RiskEstimator instance cannot see two different rates (which would make its
# own output depend on when it was constructed). Kept as a named constant so
# the mutation harness can flip it and so the emergency lever is greppable.
def _budget_guard_enabled() -> bool:
    """TCV_BUDGET_GUARD, read lazily so tests can toggle it per process.

    Lazy for the same reason `Settings.enable_phase_detectors` is: the
    production value is fixed at launch, so reading it late cannot make one
    run observe two different rates.
    """
    return os.environ.get("TCV_BUDGET_GUARD", "").strip().lower() in {
        "1", "true", "yes", "on"}


def _sec_per_obs() -> float:
    """Declared seconds per observation, from the active Settings."""
    return float(settings.sec_per_obs)


def _is_blank(frame) -> bool:
    """True iff every pixel of ``frame`` is exactly 0 -- i.e. no evidence at all.

    ``not frame.any()`` is the obvious spelling and it is 10x too expensive on
    4K input: measured on this machine, ``frame.any()`` on a 3840x2160x3 uint8
    frame costs 6.8 ms while ``int(frame.max()) != 0`` costs 0.67 ms for the
    same answer. At 9.99 fps that is 68 ms/s, and on a 30 fps 3-minute video
    ~36 s -- a tenth of the whole 3x-duration budget -- spent deciding whether
    a frame is black.

    The two agree on every real input because "non-blank" means "not all zero",
    which for any numeric dtype is exactly "the maximum is not 0":
      * a non-negative frame (what a decoder produces) with a single non-zero
        pixel has max > 0 -- the test the format suite relies on, and the
        all-zero 64x64 case still short-circuits;
      * a frame of all -1 is not blank and max(-1) != 0 says so;
      * a frame of all 0.5 is not blank either, which is why the comparison is
        ``max == 0`` and NOT ``int(max) == 0``: int() truncates 0.5 to 0 and
        would call a uniformly grey frame blank;
      * an all-NaN float frame yields NaN, and ``NaN == 0`` is False, so it is
        treated as NON-blank and the detector runs. That is the conservative
        direction: a corrupt frame is handed to perception (which fail-softs)
        rather than silently producing a confident 0.0 risk score.
    """
    try:
        return bool(np.asarray(frame).max() == 0)
    except Exception:
        # An exotic dtype (e.g. object) has no usable max. Do not treat that as
        # "blank": letting it through is safe, calling it blank is not.
        return False


# The value ``reset()``/``__init__`` store in ``_tracker_reset`` to ARM the
# per-video ByteTrack reset. Named, so that flipping it is a one-token change
# that the mutation harness can make and that is greppable rather than a bare
# boolean buried in two places.
_NEEDS_TRACKER_RESET = False


class RiskEstimator:
    """Matches the harness interface: ``reset(meta)`` + ``step(frame, t)``."""

    def __init__(self, config: RiskConfig | None = None,
                 detector_factory: Callable[[], object] | None = None):
        self.config = config if config is not None else DEFAULT_CONFIG
        # Injection seam: tests pass a fake detector so the suite never needs
        # weights, a GPU or the network. None -> the shared Part A model.
        self._detector_factory = detector_factory
        self.meta: dict = {}
        # Observability only; never affects the returned value.
        self.errors = 0
        self.observations = 0
        self.calls = 0
        self.last_channels: dict[str, float] = {}
        self._detector: object | None = None
        self._tracker_reset = _NEEDS_TRACKER_RESET
        self._assert_config()
        self._reset_runtime()

    # ------------------------------------------------------------------ #
    # harness interface
    # ------------------------------------------------------------------ #
    def reset(self, meta: dict) -> None:
        """Clear ALL state of the previous video. Called once per video.

        Every stateful object is re-created rather than cleared in place, so no
        previous-video trajectory, speed, acceleration, TTC, brake history, EMA
        or alarm value can survive — provable by construction, and checked by a
        regression test that replays video B after video A.
        """
        self.meta = dict(meta) if isinstance(meta, dict) else {}
        self._reset_runtime()
        # Arm the ByteTrack reset for this video. It is performed lazily on the
        # first observation (see _detector_or_none) rather than here, because
        # doing it eagerly would have to reach for the shared model -- and so
        # load the weights, or ignore an injected detector. Lazy is also
        # sufficient: nothing is tracked before the first observation, and the
        # first observation of every video resets the tracker, including when
        # the cached detector object is reused.
        self._tracker_reset = _NEEDS_TRACKER_RESET

    def step(self, frame, t_sec: float) -> float:
        """Causal P(accident within the next RISK_HORIZON_SEC s) in [0, 1]."""
        self.calls += 1
        t = _finite(t_sec)
        if t is None:
            t = self._prev_t if self._prev_t is not None else 0.0
        if not self._should_observe():
            # Not an observation frame: re-serve the last score unchanged. No
            # decay is invented here, because nothing was measured.
            return _clamp01(self._last)
        self.observations += 1
        try:
            raw = self._observe(frame, t)
        except Exception:
            # Fail soft, and LOUD: an exception escaping into
            # run_submission.run_risk would void the whole risk curve for this
            # video. The counter keeps it observable instead of silent.
            self.errors += 1
            return _clamp01(self._last)
        return self._shape(raw, t)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _assert_config(self) -> None:
        cfg = self.config
        assert cfg.max_tracks >= 2, "need at least 2 tracks to form a pair"
        assert 0.0 < cfg.ema_up <= 1.0, "ema_up must be in (0, 1]"
        assert 0.0 < cfg.ema_down <= 1.0, "ema_down must be in (0, 1]"
        assert 0.0 <= cfg.brake_cap <= ALARM_THETA, \
            "braking alone must never raise an alarm"
        assert 0.0 <= cfg.vru_cap <= ALARM_THETA, \
            "pedestrian proximity alone must never raise an alarm"
        assert 0.0 < cfg.brake_decel_hi > cfg.brake_decel_lo >= 0.0
        assert cfg.prox_zero_widths > cfg.prox_full_widths

    def _reset_runtime(self) -> None:
        # Re-created, never cleared in place -> no state can leak by omission.
        self._traj = TrajectoryEngine()
        self._motion = MotionEngine()
        self._pairs = PairwiseInteractionEngine()
        self._brake: dict[int, float] = {}
        self._moving: dict[int, bool] = {}
        self._ema: float | None = None
        self._last = 0.0
        self._prev_t: float | None = None
        # PHASE 25 budget guard: may only LOWER the observation rate, and only
        # when the projection says the configured one would not fit. Off by
        # default, so this is `self.config.target_hz` unchanged in production.
        self._target_hz = self._budgeted_target_hz()
        self._stride = _stride_for_fps(self.meta.get("fps"), self.config,
                                       target_hz=self._target_hz)
        self._scale = (1.0, 1.0)
        self.errors = 0
        self.observations = 0
        self.calls = 0
        self.last_channels = {"ttc": 0.0, "braking": 0.0, "proximity": 0.0}

    def _budgeted_target_hz(self) -> float:
        """Effective Part B observation rate, after the PHASE 25 guard.

        Deliberately a method on the estimator rather than an import-time
        constant: the guard needs `meta` (fps, n_frames) and `reset()` is the
        first point where those exist. Returns the configured rate untouched
        when the guard is disabled, which is the default.
        """
        configured = float(self.config.target_hz)
        if not _budget_guard_enabled():
            return configured
        n_frames = _finite(self.meta.get("n_frames"))
        fps = _finite(self.meta.get("fps"))
        if n_frames is None or fps is None or n_frames <= 0.0 or fps <= 0.0:
            return configured
        return budget.target_hz_for_budget(n_frames / fps, configured,
                                           _sec_per_obs())

    def _should_observe(self) -> bool:
        """Internal subsample: always frame 0, then every ``_stride`` frames."""
        return self._stride <= 1 or (self.calls - 1) % self._stride == 0

    def _prepare(self, frame):
        """Resize a full-res frame for inference, or None if unusable.

        Also derives the full-res<->inference scale from the frame's OWN shape,
        so the geometry never depends on ``meta`` being complete.
        """
        if frame is None:
            return None
        try:
            if getattr(frame, "ndim", 0) != 3 or frame.size == 0:
                return None
            h, w = int(frame.shape[0]), int(frame.shape[1])
        except Exception:
            return None
        if w <= 0 or h <= 0:
            return None
        # A completely blank frame carries no evidence of anything, and running
        # a detector over it is pure waste. This also keeps the official format
        # tests weight-free: they step a 64x64 all-zero frame.
        if _is_blank(frame):
            return None
        imgsz = int(settings.imgsz)
        if imgsz <= 0:
            return None
        small_h = max(1, int(imgsz * h / w))
        self._scale = (w / imgsz, h / small_h)
        try:
            return cv2.resize(frame, (imgsz, small_h))
        except Exception:
            return None

    def _detector_or_none(self):
        """The SHARED Part A Detector (one weight load per process), or None.

        Sharing the model OBJECT is a model cache, not Part A output: nothing
        Part A computed is read, and Part B keeps its own trajectory / motion /
        pair engines. Falls back to a private Detector if the shared one is
        unreachable, so ``src.risk`` is never hard-coupled to it.
        """
        if self._detector is None:
            det = None
            if self._detector_factory is not None:
                try:
                    det = self._detector_factory()
                except Exception:
                    self.errors += 1
                    return None
            else:
                det = self._shared_detector()
            if det is None:
                return None
            self._detector = det
        # AFTER the cache check, and on every path: the tracker must be reset
        # on the first observation of each video, including the video where the
        # cached detector object is reused. Checking before the early return
        # let video B inherit video A's ByteTrack ids.
        if not self._tracker_reset:
            self._reset_tracker_on(self._detector)
            self._tracker_reset = True
        return self._detector

    def _shared_detector(self):
        try:
            from ..pipeline.pipeline import _get_detector
            return _get_detector()
        except Exception:
            pass
        try:
            from ..detection import Detector
            return Detector(model_path=settings.weights_path,
                            conf=settings.conf, iou=settings.iou,
                            device=settings.device, imgsz=settings.imgsz)
        except Exception:
            self.errors += 1
            return None

    @staticmethod
    def _reset_tracker_on(det) -> None:
        """Reset ByteTrack on ``det`` if it exposes the hook.

        ultralytics attaches ONE tracker to the model and, with
        ``persist=True``, reuses it forever (``on_predict_start`` returns early
        when persisting), so track ids and Kalman filters otherwise leak from
        the previous video -- or from Part A, which tracked the same video just
        before this one -- into the next. Duck-typed because the test
        detectors deliberately do not implement it.
        """
        reset = getattr(det, "reset_tracker", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass

    # ---- channels ------------------------------------------------------ #
    @staticmethod
    def _pair_channels(pr, scale_px: float, cfg: RiskConfig) -> tuple[float, float]:
        """(ttc, proximity) danger for one pair.

        Static (not a method) on purpose: it is the measured, config-parameterised
        gate, and the false-alarm budget replays it over real recorded pairs
        without constructing an estimator.

        There is deliberately NO "already in contact -> risk 1.0" shortcut. One
        was built and removed on measurement: gating on
        ``distance_px <= 0.5 * max(box widths)`` fired on 1436 of 150726 REAL
        normal-traffic pairs (~1%, mostly motorcycle/person), which pushed p99 of
        this channel to 1.0 -- a full alarm on ordinary traffic. A person is a
        tall narrow box, so two people standing a stride apart legitimately put
        their bottom-centres inside half a width. The only genuinely degenerate
        case the shortcut would have covered is
        ``distance_px < TINY_DISTANCE_PX`` (1e-3 px), where the shared engine
        reports ``TTC = +inf``; that is a numerical coincidence at 4K, not a
        collision, and the alarm has already been raised for several seconds
        earlier on the approach.
        """
        ttc_ch = 0.0
        distance = _finite(getattr(pr, "distance_px", None))
        scale = max(_as_float(scale_px, 1.0), 1.0)
        # Separation floor: coincident tracks are perception artifacts.
        if distance is None or distance < cfg.min_sep_widths * scale:
            pass
        else:
            closing = _finite(getattr(pr, "closing_speed_px_s", None))
            relative = _finite(getattr(pr, "relative_speed_px_s", None))
            if (getattr(pr, "approaching", False) and closing is not None
                    and relative is not None
                    and closing >= cfg.min_closing_px_s
                    and relative >= cfg.min_relative_px_s):
                ttc = ttc_risk(getattr(pr, "ttc_sec", None), cfg.horizon_sec,
                               cfg.ttc_curvature)
                if ttc > 0.0:
                    meet = proximity_factor(
                        getattr(pr, "min_predicted_distance_px", None), scale,
                        cfg.prox_full_widths, cfg.prox_zero_widths)
                    ttc_ch = _clamp01(ttc * meet)
        prox_ch = 0.0
        if _has_vulnerable(getattr(pr, "class_a", None),
                           getattr(pr, "class_b", None)):
            # NOT gated on closing speed on purpose: this channel exists to
            # cover a pair that is about to conflict but not yet converging.
            close = proximity_factor(
                getattr(pr, "min_predicted_distance_px", None), scale,
                cfg.vru_full_widths, cfg.vru_zero_widths)
            prox_ch = _clamp01(close * cfg.vru_cap)
        return ttc_ch, prox_ch

    def _brake_channel(self, tid: int, st) -> float:
        """Braking danger for one track, updating its causal brake memory."""
        cfg = self.config
        if bool(getattr(st, "stationary", False)):
            # A stop/start transition puts a large step into MotionEngine's
            # speed difference; drop it and clear the brake memory.
            self._brake[tid] = 0.0
            self._moving[tid] = False
            return 0.0
        was_moving = self._moving.get(tid, False)
        self._moving[tid] = True
        if not was_moving:
            # No previous moving sample: accel here is a start-up artifact.
            self._brake[tid] = 0.0
            return 0.0
        b = brake_risk(getattr(st, "accel", None), getattr(st, "speed", None),
                       cfg.brake_decel_lo, cfg.brake_decel_hi,
                       cfg.brake_min_speed_px_s)
        prev = self._brake.get(tid)
        eased = b if prev is None else prev + cfg.brake_alpha * (b - prev)
        self._brake[tid] = eased
        return _clamp01(eased * cfg.brake_cap)

    # ---- per-observation compute --------------------------------------- #
    def _observe(self, frame, t: float) -> float:
        ch = {"ttc": 0.0, "braking": 0.0, "proximity": 0.0}
        img = self._prepare(frame)
        det = self._detector_or_none() if img is not None else None
        if img is None or det is None:
            self.last_channels = ch
            return 0.0
        sx, sy = self._scale

        # Rescale into NEW dicts. Mutating the detector's output in place works
        # in production (Detector.track builds fresh dicts every call) but it
        # makes ``step`` have a side effect on its input, and it silently
        # double-scales anything that replays the same detection list twice.
        dets = []
        for d in det.track(img, persist=True):
            x1, y1, x2, y2 = d["xyxy"]
            dets.append({**d, "xyxy": (x1 * sx, y1 * sy, x2 * sx, y2 * sy)})

        trajs = self._traj.update([Detection.from_dict(d) for d in dets], t)
        if len(trajs) > self.config.max_tracks:
            # Most recently observed first, tie-broken by id: deterministic.
            trajs = sorted(
                trajs,
                key=lambda tr: (-(tr.last.t if tr.last is not None else -1.0),
                                tr.track_id))[:self.config.max_tracks]

        items: list[tuple] = []
        widths: dict[int, float] = {}
        for tr in trajs:
            if tr.last is None or tr.label not in CANDIDATE_LABELS:
                continue
            st = self._motion.update(tr, t)
            if st is None:
                continue
            widths[tr.track_id] = max(tr.last.xyxy[2] - tr.last.xyxy[0], 1.0)
            items.append((tr.track_id, tr.label,
                          (tr.last.x, tr.last.bottom_y), st))
            ch["braking"] = max(ch["braking"],
                                self._brake_channel(tr.track_id, st))

        if len(items) >= 2:
            # PHASE 26: the relevance test is a function of the two class labels
            # alone, so run it BEFORE building a PairInteraction instead of on
            # the ~600 objects the sweep allocated per frame. Same predicate,
            # same surviving pairs, same order.
            for pr in self._pairs.pairs_filtered(items, t, _pair_relevant):
                scale = max(widths.get(pr.track_id_a, 0.0),
                            widths.get(pr.track_id_b, 0.0), 1.0)
                ttc_ch, prox_ch = self._pair_channels(pr, scale, self.config)
                ch["ttc"] = max(ch["ttc"], ttc_ch)
                ch["proximity"] = max(ch["proximity"], prox_ch)

        self.last_channels = {k: _clamp01(v) for k, v in ch.items()}
        return max(self.last_channels.values())

    # ---- causal temporal shaping --------------------------------------- #
    def _shape(self, raw: float, t: float) -> float:
        """Asymmetric causal EMA. Reads only ``raw`` and past EMA values."""
        cfg = self.config
        raw = _clamp01(raw)
        self._prev_t = t
        if self._ema is None:
            self._ema = raw
        else:
            alpha = cfg.ema_up if raw > self._ema else cfg.ema_down
            self._ema += alpha * (raw - self._ema)
        self._last = _clamp01(self._ema)
        return self._last
