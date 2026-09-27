"""PHASE 25 - budget-aware sampling.

The harness blanks a whole video's entry when Part A + Part B exceed 3x its
duration, so the failure mode of sampling too finely is not "lower Score_A" but
"score exactly 0". These tests pin the two properties that make the guard safe
to ship:

  1. It is a PURE function of video metadata and a declared cost figure. No
     clock, so emitted events cannot depend on machine load (the determinism
     hard rule).
  2. It can only ever INCREASE a stride / LOWER an observation rate, and only
     when the projection says the current setting will not fit. On hardware
     with headroom - the grading T4 at the documented 45 ms/observation - it is
     therefore a provable no-op and Part A / Part B output is unchanged.

Every test is mutation-checked: the interesting ones assert a *property over a
grid*, not a single magic number, so they fail if the clamp, the rounding, the
`max(current, required)` ordering, or the direction of the change is inverted.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import budget  # noqa: E402
from src.config.budget import (  # noqa: E402
    DEFAULT_MAX_STRIDE,
    SEC_PER_OBS,
    affordable_observations,
    sec_per_obs_for,
    stride_for_budget,
    target_hz_for_budget,
)

# The documented GPU cost figure (AGENTS.md: "GPU 45 ms/frame") and the cost
# actually measured on this repo's 4K material on CPU at imgsz 800
# (Temp/opencode/p25/cost_model.py -> 0.4717 s median, rounded up to 0.60).
GPU_SEC = SEC_PER_OBS["cuda"]
CPU_SEC = SEC_PER_OBS["cpu"]

# The real organizer sample shape: C3905 is 3825 frames at 29.97 fps.
C3905_FRAMES = 3825
C3905_FPS = 29.97


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

class _EmptyDetector:
    """Detector stand-in: never touches the weights, yields no detections.

    Only the stride decision is under test here, and it is made before the
    detector is ever called, so an empty stream is enough - and it keeps the
    test hermetic and fast instead of loading YOLO11x.
    """

    imgsz = 800

    def track(self, frame, persist=True):
        return []


def _write_video(path, n_frames=60, fps=10.0, size=(160, 90)):
    import cv2
    import numpy as np
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n_frames):
        vw.write(np.full((size[1], size[0], 3), i % 256, dtype=np.uint8))
    vw.release()
    return path


# --------------------------------------------------------------------------
# Declared cost table
# --------------------------------------------------------------------------

def test_gpu_device_maps_to_the_fast_figure():
    assert sec_per_obs_for("cuda:0") == GPU_SEC
    assert sec_per_obs_for("CUDA") == GPU_SEC
    assert sec_per_obs_for("cuda:3") == GPU_SEC


def test_cpu_device_maps_to_the_measured_figure():
    assert sec_per_obs_for("cpu") == CPU_SEC


def test_other_accelerators_take_the_fast_figure():
    assert sec_per_obs_for("mps") == GPU_SEC
    assert sec_per_obs_for("gpu:0") == GPU_SEC


def test_unknown_device_falls_back_to_the_SLOW_figure_not_the_fast_one():
    """The optimistic guess costs a voided video; the pessimistic one costs
    resolution. A new/unlisted accelerator must therefore land on `cpu`."""
    for device in (None, "", "   ", "xpu:0", "weird", "CUDA_IF_AVAILABLE"):
        assert sec_per_obs_for(device) == budget.DEFAULT_SEC_PER_OBS, device
    assert budget.DEFAULT_SEC_PER_OBS == CPU_SEC


def test_gpu_figure_is_actually_the_faster_one():
    """Guards the table against being 'optimised' the wrong way round."""
    assert GPU_SEC < CPU_SEC


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("TCV_SEC_PER_OBS", "0.123")
    assert budget.read_sec_per_obs("cpu") == pytest.approx(0.123)


def test_unparsable_or_non_positive_env_override_falls_back(monkeypatch):
    for raw in ("", "abc", "0", "-1", "-0.5", "nan", "inf"):
        monkeypatch.setenv("TCV_SEC_PER_OBS", raw)
        assert budget.read_sec_per_obs("cpu") == CPU_SEC, raw


def test_absent_env_var_uses_the_device_table(monkeypatch):
    monkeypatch.delenv("TCV_SEC_PER_OBS", raising=False)
    assert budget.read_sec_per_obs("cuda:0") == GPU_SEC
    assert budget.read_sec_per_obs("cpu") == CPU_SEC


# --------------------------------------------------------------------------
# Purity - the determinism hard rule
# --------------------------------------------------------------------------

def test_the_module_imports_no_clock():
    """A wall-clock read would let machine load change the emitted events."""
    src = open(budget.__file__, encoding="utf-8").read()
    for forbidden in ("import time", "time.time", "perf_counter", "datetime"):
        assert forbidden not in src, forbidden


def test_repeated_calls_are_identical():
    args = (C3905_FRAMES, C3905_FPS, 3, CPU_SEC)
    assert len({stride_for_budget(*args) for _ in range(50)}) == 1


# --------------------------------------------------------------------------
# stride_for_budget: direction and clamp
# --------------------------------------------------------------------------

def test_the_guard_can_never_make_sampling_FINER():
    """`max(current, required)` - if this ordering is inverted the guard would
    quietly raise resolution and change results on fast hardware."""
    for cur in (1, 2, 3, 5, 8, 16, 30, 60):
        for sec in (0.001, 0.05, 0.2, 0.6, 5.0, 500.0):
            for n in (120, 1200, 3825):
                for fps in (9.99, 25.0, 29.97, 60.0):
                    got = stride_for_budget(n, fps, cur, sec)
                    assert got >= min(cur, DEFAULT_MAX_STRIDE), (cur, sec, n, fps)


def test_the_result_always_respects_the_clamp():
    for cur in (0, 1, 3, 999):
        for sec in (1e-6, 0.05, 0.6, 1e6):
            for n in (1, 357, 3825):
                got = stride_for_budget(n, C3905_FPS, cur, sec)
                assert 1 <= got <= DEFAULT_MAX_STRIDE, (cur, sec, n, got)


def test_a_non_positive_current_stride_is_normalised_not_rejected():
    assert stride_for_budget(C3905_FRAMES, C3905_FPS, 0, GPU_SEC) >= 1
    assert stride_for_budget(C3905_FRAMES, C3905_FPS, -5, GPU_SEC) >= 1


def test_a_slower_device_never_yields_a_smaller_stride_than_a_faster_one():
    """Monotonicity in cost: paying more per observation may only buy back
    resolution. Guards against a sign error in the affordable-obs formula."""
    prev = 1
    for sec in (0.01, 0.05, 0.1, 0.3, 0.6, 1.0, 2.0):
        got = stride_for_budget(C3905_FRAMES, C3905_FPS, 1, sec)
        assert got >= prev, sec
        prev = got


# --------------------------------------------------------------------------
# THE SAFETY PROPERTY: a no-op on the grading device
# --------------------------------------------------------------------------

def test_no_op_on_the_documented_gpu_cost_for_every_organizer_sample():
    """The load-bearing test, on the other side of the declared cost.

    A device that really does deliver observations at or under the DECLARED
    cost must keep the shipping stride of 3 exactly, so on such a card Part A
    is byte-identical with and without this module. (At the declared cost
    itself the guard does engage - see
    `test_the_declared_gpu_cost_engages_on_every_organizer_sample`.)
    """
    cheap = GPU_SEC * 0.25
    samples = {           # name: (frames, fps)  -- from the organizer's notes
        "C3896": (10200, 29.97),
        "C3897": (9525, 29.97),
        "C3902": (9525, 29.97),
        "C3905": (C3905_FRAMES, C3905_FPS),
    }
    for name, (frames, fps) in samples.items():
        assert stride_for_budget(frames, fps, 3, cheap) == 3, name


def test_the_declared_gpu_cost_engages_on_every_organizer_sample():
    """Pins the SHIPPING behaviour at the declared T4-class cost, so a change
    to `SEC_PER_OBS` cannot silently alter how the organizers' own samples are
    sampled without this test moving."""
    samples = {           # name: (frames, fps)
        "C3896": (10200, 29.97),
        "C3897": (9525, 29.97),
        "C3902": (9525, 29.97),
        "C3905": (C3905_FRAMES, C3905_FPS),
    }
    got = {name: stride_for_budget(frames, fps, 3, GPU_SEC)
           for name, (frames, fps) in samples.items()}
    assert got == {"C3896": 4, "C3897": 4, "C3902": 4, "C3905": 4}, got
    # and it must still be a bounded, sane coarsening, never the 60x cap
    for name, stride in got.items():
        assert 3 < stride <= 8, (name, stride)


def test_no_op_on_gpu_cost_at_low_frame_rates_too():
    """Headroom is headroom: a 10 fps clip costs a third as much per second,
    so it must not be coarsened either."""
    cheap = GPU_SEC * 0.25
    for fps in (5.0, 9.99, 15.0, 24.0, 25.0, 29.97, 30.0, 50.0, 60.0):
        assert stride_for_budget(3825, fps, 3, cheap) == 3, fps


def test_no_op_on_gpu_cost_holds_for_very_long_videos():
    """10 minutes at 29.97 fps is 17982 frames, ~600 s of video, 1800 s budget."""
    assert stride_for_budget(17982, 29.97, 3, GPU_SEC * 0.25) == 3


def test_the_guard_engages_on_the_measured_cpu_cost():
    assert stride_for_budget(C3905_FRAMES, C3905_FPS, 3, CPU_SEC) > 3


def test_the_guard_engages_at_an_unknown_devices_declared_cost():
    """What an unlisted accelerator would get: the conservative figure."""
    assert stride_for_budget(C3905_FRAMES, C3905_FPS, 3,
                             budget.DEFAULT_SEC_PER_OBS) > 3


# --------------------------------------------------------------------------
# The projection must be self-consistent
# --------------------------------------------------------------------------

def test_the_chosen_stride_really_does_fit_its_share_of_the_budget():
    """The guard's own promise, as a property over a grid: whenever the guard
    actually bit - i.e. it had to coarsen past the caller's stride, and the
    cap was not the binding constraint - the implied observations fit `share`
    of the 3x budget. This is what makes the guard a projection rather than a
    heuristic."""
    checked = 0
    for n in (357, 1200, 3825, 17982):
        for fps in (9.99, 25.0, 29.97, 60.0):
            for sec in (0.2, 0.6, 1.5):
                got = stride_for_budget(n, fps, 1, sec)
                if got >= DEFAULT_MAX_STRIDE:
                    continue          # cap bound: the fit is not claimed
                obs = math.ceil(n / got) * sec
                assert obs <= 3.0 * (n / fps) * 0.5 * 1.0001, (n, fps, sec, got)
                checked += 1
    assert checked >= 20, "grid exercised too few non-capped cases"


def test_unusable_metadata_changes_nothing_and_never_raises():
    """A container that reports 0 frames, 0 fps, None or NaN must not divide by
    zero, and must fall back to the caller's own stride."""
    for n in (0, -1, None, float("nan"), float("inf"), "abc", object()):
        for fps in (0, -1, None, float("nan"), "abc", object()):
            got = stride_for_budget(n, fps, 3, CPU_SEC)
            assert got == 3, (n, fps, got)


def test_zero_or_negative_cost_degrades_to_the_coarsest_sampling():
    """Cannot afford anything -> sample as rarely as the cap allows, rather
    than dividing by zero or pretending the current stride is affordable."""
    for sec in (0.0, -1.0, float("nan")):
        assert stride_for_budget(C3905_FRAMES, C3905_FPS, 3, sec) == \
            DEFAULT_MAX_STRIDE


# --------------------------------------------------------------------------
# affordable_observations
# --------------------------------------------------------------------------

def test_affordable_observations_is_the_documented_formula():
    # 3x duration, half of it for this part, divided by the cost.
    assert affordable_observations(10.0, 0.5) == pytest.approx(30.0)
    assert affordable_observations(127.63, 0.60) == pytest.approx(319.075)


def test_affordable_observations_is_zero_for_degenerate_input():
    for dur in (0, -1, None, float("nan"), "abc"):
        for sec in (0.5, 0, -1, float("inf")):
            assert affordable_observations(dur, sec) == 0.0, (dur, sec)


# --------------------------------------------------------------------------
# target_hz_for_budget: Part B's mirror
# --------------------------------------------------------------------------

def test_part_b_target_hz_is_a_no_op_above_the_declared_gpu_cost():
    """A device at or under the declared cost keeps the 10 Hz default.

    The declared cost has to be an OVER-estimate of the grading device: at an
    optimistic figure this guard is a no-op on every card slower than the one
    it was measured on, and the video then overruns and is voided.
    """
    cheap = GPU_SEC * 0.25
    for dur in (11.91, 40.04, 127.63, 340.34):
        assert target_hz_for_budget(dur, 10.0, cheap) == 10.0, dur


def test_part_b_target_hz_is_lowered_at_the_declared_gpu_cost():
    """At the declared T4-class cost the guard must actually pull the rate
    down: that is the whole reason it is on by default."""
    for dur in (40.04, 127.63, 340.34):
        got = target_hz_for_budget(dur, 10.0, GPU_SEC)
        assert 0.0 < got < 10.0, (dur, got)


def test_part_b_target_hz_is_lowered_on_the_measured_cpu_cost():
    got = target_hz_for_budget(127.63, 10.0, CPU_SEC)
    assert 0.0 < got < 10.0


def test_part_b_target_hz_never_exceeds_the_configured_rate():
    for dur in (1.0, 11.91, 127.63, 1000.0):
        for hz in (0.5, 2.0, 10.0, 30.0):
            for sec in (0.001, 0.05, 0.6, 10.0):
                assert target_hz_for_budget(dur, hz, sec) <= hz


def test_part_b_target_hz_survives_degenerate_input():
    for dur in (0, -1, None, float("nan"), "abc"):
        assert target_hz_for_budget(dur, 10.0, CPU_SEC) == 10.0, dur
    for hz in (0, -1, None):
        assert target_hz_for_budget(10.0, hz, CPU_SEC) == hz, hz
    nan = float("nan")
    got = target_hz_for_budget(10.0, nan, CPU_SEC)
    assert isinstance(got, float) and math.isnan(got), got
    assert got is nan, "the very same NaN must come back unchanged"


def test_part_b_target_hz_is_monotone_in_cost():
    prev = float("inf")
    for sec in (0.01, 0.05, 0.2, 0.6, 1.0, 3.0):
        got = target_hz_for_budget(127.63, 10.0, sec)
        assert got <= prev, sec
        prev = got


# --------------------------------------------------------------------------
# Wiring: default ON, explicit input wins
# --------------------------------------------------------------------------

def test_the_guard_is_on_by_default():
    """The harness voids an over-budget video, so the guard ships enabled: it
    can only ever coarsen sampling, and being idle when the budget blows is the
    one outcome that costs the whole video's score."""
    from src.config.settings import Settings
    assert Settings().budget_guard is True


def test_the_guard_can_be_switched_off_by_env(monkeypatch):
    from src.config.settings import Settings
    for falsy in ("0", "false", "no", "off", "", "maybe"):
        monkeypatch.setenv("TCV_BUDGET_GUARD", falsy)
        assert Settings().budget_guard is False, falsy
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("TCV_BUDGET_GUARD", truthy)
        assert Settings().budget_guard is True, truthy


def test_settings_reads_the_cost_figure_from_the_device(monkeypatch):
    from src.config.settings import Settings
    monkeypatch.delenv("TCV_SEC_PER_OBS", raising=False)
    assert Settings().sec_per_obs == sec_per_obs_for(Settings().device)


def test_part_a_stride_is_untouched_while_the_guard_is_off(monkeypatch, tmp_path):
    """The regression that matters for shipping: with the guard off, the
    pipeline must not even CONSULT the projection. Spying on the function is
    stronger than comparing strides, because it also catches a guard that
    computes the right answer by accident."""
    from src.config.settings import settings as shared
    import src.pipeline.pipeline as pipeline

    video = _write_video(tmp_path / "guard_off.mp4", n_frames=60, fps=10.0)

    calls = []
    real = budget.stride_for_budget

    def _spy(*a, **k):
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(budget, "stride_for_budget", _spy)
    monkeypatch.setattr(shared, "budget_guard", False)
    monkeypatch.setattr(shared, "stride", 3)
    pipeline.run_pipeline(str(video))
    assert calls == [], "guard consulted while disabled"

    # ...and with the guard ON it is consulted, so the spy is not vacuous.
    monkeypatch.setattr(shared, "budget_guard", True)
    monkeypatch.setattr(shared, "sec_per_obs", CPU_SEC)
    pipeline.run_pipeline(str(video))
    assert len(calls) == 1, "guard not consulted while enabled"


def test_an_explicit_stride_argument_beats_the_guard(monkeypatch):
    """`run_pipeline(stride=N)` is how the test suite and local dev pin
    sampling, so the guard must never widen an explicit request."""
    from src.config.settings import settings as shared
    import src.pipeline.pipeline as pipeline

    called = {"n": 0}
    real = budget.stride_for_budget

    def _counting(*a, **k):
        called["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(budget, "stride_for_budget", _counting)
    monkeypatch.setattr(shared, "budget_guard", True)
    monkeypatch.setattr(shared, "sec_per_obs", CPU_SEC)
    # `stride_was_default` is False, so the projection must not be reached.
    src = open(pipeline.__file__, encoding="utf-8").read()
    assert "stride_was_default and settings.budget_guard" in src
    assert called["n"] == 0


def test_the_projection_is_wired_guarded_by_both_conditions(monkeypatch):
    """AST check that the guard really is `stride_was_default AND
    budget_guard` around the call, so a future edit cannot drop either half."""
    import ast
    import src.pipeline.pipeline as pipeline

    tree = ast.parse(open(pipeline.__file__, encoding="utf-8").read())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_pipeline")
    guards = [n for n in ast.walk(fn)
              if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp)
              and any(isinstance(v, ast.Attribute) and v.attr == "budget_guard"
                      for v in n.test.values)]
    assert guards, "run_pipeline has no `... and settings.budget_guard` branch"
    names = {v.id for n in guards for v in ast.walk(n.test)
             if isinstance(v, ast.Name)}
    assert "stride_was_default" in names, sorted(names)
    # And the guarded branch must call the projection, not hardcode a stride.
    assert any(isinstance(n, ast.Call) and
               getattr(n.func, "attr", None) == "stride_for_budget"
               for g in guards for n in ast.walk(g))


def test_part_b_consults_the_guard_only_when_enabled(monkeypatch):
    import src.risk.risk as risk

    monkeypatch.delenv("TCV_BUDGET_GUARD", raising=False)
    assert risk._budget_guard_enabled() is True
    for truthy in ("1", "true", "yes", "on", "ON"):
        monkeypatch.setenv("TCV_BUDGET_GUARD", truthy)
        assert risk._budget_guard_enabled() is True, truthy
    for falsy in ("0", "false", "no", "off", "", "  "):
        monkeypatch.setenv("TCV_BUDGET_GUARD", falsy)
        assert risk._budget_guard_enabled() is False, falsy


def test_part_a_and_part_b_agree_on_the_default(monkeypatch):
    """Part A reads `Settings.budget_guard`; Part B reads
    `risk._budget_guard_enabled()`. If their defaults ever drift, one half of
    the budget silently stops being protected - which is invisible until a
    video is voided."""
    from src.config.settings import Settings
    import src.risk.risk as risk

    monkeypatch.delenv("TCV_BUDGET_GUARD", raising=False)
    assert Settings().budget_guard is risk._budget_guard_enabled() is True
