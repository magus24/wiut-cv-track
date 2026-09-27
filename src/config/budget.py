"""Budget-aware sampling (PHASE 25).

The harness allows ``budget_factor x duration`` seconds for Part A + Part B
together and BLANKS THE WHOLE ENTRY when it is exceeded, so an over-budget
video scores exactly 0 for both parts. This module answers one question:

    given a video's metadata and a cost-per-observation figure, what is the
    coarsest sampling rate that still fits the budget?

Two properties are deliberate and load-bearing.

**1. It is a pure function.** It takes ``(n_frames, fps, sec_per_obs)`` and
returns an int. It never reads a clock. Wall-clock feedback would make the
emitted events depend on machine load, which would break the determinism hard
rule ("two runs on the same machine must produce the same output"). The cost
figure is therefore a *declared constant* per device, not a measurement.

**2. It can only ever INCREASE a stride, never decrease one.** The caller
passes the stride it would otherwise use and the guard returns
``max(current, required)``. Consequences:

  * On hardware with headroom the guard is a provable no-op. At the documented
    T4 figure (45 ms/observation) a 29.97 fps video needs stride 1 to fit, so
    the default stride 3 passes through untouched and Part A output on the
    grading machine is byte-identical to shipping without this module.
  * Quality is never silently upgraded by the guard, so it cannot change
    results on a fast device.

Why a guard at all, if the T4 has ~3.3x headroom: the frame rate of the test
videos is not under our control, and cost per second of video is
``observations_per_second x sec_per_obs`` while the budget is denominated in
seconds. A fixed *frame* stride therefore makes the budget outcome depend on
the test set's frame rate - measured on this repo's own material, Part A costs
4.71 s per second of video at imgsz 800 on CPU, and 0.45 s/s at the documented
T4 figure, a 10x spread on identical code. The guard converts that hidden
coupling into an explicit, declared cost assumption.

Disabled by default (``TCV_BUDGET_GUARD=1`` enables it). It is OFF by default
because the grading device does not need it, and a default-ON path that can
only degrade quality buys nothing there while adding a way to lose points. It
is a lever for slower hardware, not an optimisation.
"""

from __future__ import annotations

import math
import os

# Declared cost of ONE detector observation (one YOLO call incl. tracking) in
# seconds. Measured on this repo's 4K material on CPU at imgsz 800: 0.4717 s
# median (cost_model.py); the documented GPU figure is 0.045 s
# (AGENTS.md: "GPU 45 ms/frame"). Deliberately rounded UP for the CPU case: an
# over-estimate costs a little resolution, an under-estimate costs the entire
# video's score.
SEC_PER_OBS: dict[str, float] = {
    "cuda": 0.05,
    "cpu": 0.60,
}
DEFAULT_SEC_PER_OBS = 0.60          # unknown / unlisted device: assume slow

# Fraction of the total budget each part may plan to use. The two parts make a
# comparable number of observations, so an even split is the neutral choice; it
# only has to leave room for the other part, not to be exact.
DEFAULT_SHARE = 0.5

# Ceiling on the coarsening. Separate from Part B's own `max_stride` (8), which
# bounds how far the RISK CURVE's temporal resolution is reduced for signal
# quality reasons; this one only has to keep the video from being voided, and
# needs to reach far higher to do that on CPU.
DEFAULT_MAX_STRIDE = 60

# Floor for Part B's observation rate. Must be strictly positive because the
# consumer (`_stride_for_fps`) reads a non-positive target as "stride 1", i.e.
# observe EVERY frame - the most expensive setting there is. See
# `target_hz_for_budget`.
TARGET_HZ_FLOOR = 1e-3


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(int(value), int(high)))


def _is_fast_device(d: str) -> bool:
    """True only for a real accelerator device string.

    Deliberately NOT a bare ``startswith``: that would also accept
    ``cuda_if_available`` (an env-var *setting* name, not a device) and hand it
    the FAST cost figure, i.e. the optimistic misclassification this module
    exists to avoid. Accepted forms are exactly what torch accepts for an
    accelerator here: ``cuda``/``cuda:<digits>``, ``mps``, ``gpu``, ``gpu:<n>``.
    """
    if d in ("mps", "gpu"):
        return True
    for name in ("cuda", "gpu"):
        if d == name:
            return True
        if d.startswith(name + ":") and d[len(name) + 1:].isdigit():
            return True
    return False


def sec_per_obs_for(device: str | None) -> float:
    """Declared seconds per observation for a torch device string.

    ``None``/empty/anything unrecognised falls back to the SLOW figure, because
    the failure mode of guessing wrong in the optimistic direction is a voided
    video (score 0) while guessing slow only costs resolution.
    """
    d = (device or "").strip().lower()
    if _is_fast_device(d):
        return SEC_PER_OBS["cuda"]
    if d == "cpu" or d.startswith("cpu:"):
        return SEC_PER_OBS["cpu"]
    return DEFAULT_SEC_PER_OBS


def read_sec_per_obs(device: str | None = None,
                     env: str = "TCV_SEC_PER_OBS") -> float:
    """`sec_per_obs_for(device)` with an explicit override for experiments.

    An override that is not a finite positive number is IGNORED rather than
    used: ``TCV_SEC_PER_OBS=inf`` would otherwise drive affordable
    observations to 0, and ``nan`` would poison every comparison downstream.
    """
    raw = os.environ.get(env)
    if raw is None:
        return sec_per_obs_for(device)
    val = _finite(raw)
    if val is None or val <= 0.0:
        return sec_per_obs_for(device)
    return val


def affordable_observations(duration_sec: float, sec_per_obs: float,
                            budget_factor: float = 3.0,
                            share: float = DEFAULT_SHARE) -> float:
    """How many detector observations fit in this part's share of the budget.

    Returns 0.0 for a non-positive duration or a non-positive cost, which the
    callers read as "cannot afford anything" and degrade to the coarsest
    sampling rather than dividing by zero.
    """
    d = _finite(duration_sec)
    c = _finite(sec_per_obs)
    if d is None or c is None or d <= 0.0 or c <= 0.0:
        return 0.0
    return max(0.0, float(budget_factor) * d * float(share)) / c


def _finite(value: object) -> float | None:
    try:
        out = float(value)          # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def stride_for_budget(n_frames: object, fps: object, current_stride: int,
                      sec_per_obs: float, budget_factor: float = 3.0,
                      share: float = DEFAULT_SHARE,
                      max_stride: int = DEFAULT_MAX_STRIDE) -> int:
    """Part A frame stride that fits the budget, never finer than `current`.

    ``current_stride`` is what the caller would use without the guard. The
    result is ``max(current_stride, required)`` clamped to
    ``[1, max_stride]``, so a video that already fits keeps its sampling rate
    exactly and a video that cannot fit is coarsened as far as `max_stride`
    allows.
    """
    n = _finite(n_frames)
    f = _finite(fps)
    cur = _clamp(current_stride, 1, max_stride)
    if n is None or f is None or n <= 0.0 or f <= 0.0:
        return cur                       # unusable metadata -> change nothing
    duration = n / f
    obs = affordable_observations(duration, sec_per_obs, budget_factor, share)
    if obs <= 0.0:
        return _clamp(max_stride, 1, max_stride)
    affordable = int(math.floor(obs))
    if affordable < 1:
        return _clamp(max_stride, 1, max_stride)
    required = int(math.ceil(n / affordable))
    return _clamp(max(cur, required), 1, max_stride)


def target_hz_for_budget(duration_sec: object, current_hz: object,
                         sec_per_obs: float, budget_factor: float = 3.0,
                         share: float = DEFAULT_SHARE) -> float:
    """Part B observation rate that fits the budget, never above `current_hz`.

    The mirror of :func:`stride_for_budget` for Part B, which thinks in Hz
    rather than frame stride. Part B re-serves the last score on frames it does
    not observe (``_should_observe``), so lowering the rate shortens the curve's
    temporal resolution without changing its shape or its causal ordering.

    Unusable input (``duration`` or ``current_hz`` missing, non-finite or
    non-positive) is returned UNCHANGED - the same "do not touch what you
    cannot reason about" contract the stride function follows.

    The "nothing is affordable" answer is :data:`TARGET_HZ_FLOOR`, never
    ``0.0``. That is not cosmetic: ``_stride_for_fps`` maps a non-positive
    target to stride 1, which is the FINEST sampling and so the most expensive
    possible outcome - a literal zero would invert the direction of the whole
    guard exactly when it is most needed.
    """
    cur = _finite(current_hz)
    d = _finite(duration_sec)
    if cur is None or cur <= 0.0 or d is None or d <= 0.0:
        return current_hz                       # type: ignore[return-value]
    obs = affordable_observations(d, sec_per_obs, budget_factor, share)
    if obs <= 0.0:
        return TARGET_HZ_FLOOR
    return min(cur, max(TARGET_HZ_FLOOR, obs / d))
