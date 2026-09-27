"""The detector must not be able to score zero SILENTLY.

Failure taxonomy, which the old code did not distinguish:

  * one corrupt FRAME      -> fail-soft, keep going, is correct
  * weights that never load -> a configuration error whose symptom is identical
    (no detections anywhere) and whose cost is the whole submission: every video
    gets an empty event list, Score_A = Score_B = 0, and `evaluate.py
    --validate-only` still prints VALID because an empty event list is legal.

Before this module existed, both were the same two lines
(`try: self._ensure_model() except Exception: return []`), so a missing
yolo11x.pt produced a perfectly valid, perfectly empty predictions file and no
indication of why.

Run:  python -m pytest tests/test_detector_preflight.py -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.detection.detector import Detector  # noqa: E402

FRAME = np.zeros((64, 64, 3), dtype=np.uint8)


# --------------------------------------------------------------- preflight
def test_preflight_accepts_the_shipped_weights():
    """The real file that ships must pass. If this fails, the submission is
    already scoring zero and this is the first place it shows up."""
    from src.config.settings import settings
    det = Detector(settings.weights_path, device="cpu")
    if not os.path.isfile(settings.weights_path):
        pytest.skip("weights not present on this box")
    assert det.preflight() is None, det.preflight()


def test_preflight_names_a_missing_file():
    det = Detector("weights/definitely_not_here.pt", device="cpu")
    problem = det.preflight()
    assert problem is not None
    assert "not found" in problem
    assert "definitely_not_here.pt" in problem
    # the message must tell the operator what to do about it
    assert "download.sh" in problem


def test_preflight_names_an_empty_file(tmp_path):
    empty = tmp_path / "yolo11x.pt"
    empty.write_bytes(b"")
    det = Detector(str(empty), device="cpu")
    problem = det.preflight()
    assert problem is not None and "empty" in problem


def test_preflight_names_a_missing_directory():
    det = Detector("no/such/dir/yolo11x.pt", device="cpu")
    problem = det.preflight()
    assert problem is not None and "not found" in problem


def test_preflight_names_an_unset_path():
    det = Detector("", device="cpu")
    problem = det.preflight()
    assert problem is not None and "no weights path" in problem


# ------------------------------------------------- the loud failure banner
def test_missing_weights_warns_once_and_keeps_returning_empty(capsys):
    """The whole point: empty results, but NO SILENCE."""
    det = Detector("weights/definitely_not_here.pt", device="cpu")
    for _ in range(5):
        assert det.detect(FRAME) == []
        assert det.track(FRAME) == []
    err = capsys.readouterr().err
    assert "FATAL" in err
    assert "ZERO detections" in err
    assert "Score_A = Score_B = 0" in err
    # once, not five times: five videos is a log, thirty is spam
    assert err.count("FATAL") == 1, err


def test_load_error_is_recorded_for_inspection():
    det = Detector("weights/definitely_not_here.pt", device="cpu")
    det.detect(FRAME)
    assert det.load_error is not None
    assert "not found" in det.load_error


def test_two_detectors_each_report_their_own_failure(tmp_path):
    """The one-shot warning is per instance, not a module global: Part A and
    Part B build separate Detectors and both must be able to complain."""
    a = Detector(str(tmp_path / "nope_a.pt"), device="cpu")
    b = Detector(str(tmp_path / "nope_b.pt"), device="cpu")
    a.detect(FRAME)
    b.detect(FRAME)
    assert a.load_error != b.load_error
    assert "nope_a.pt" in a.load_error and "nope_b.pt" in b.load_error


# ------------------------------------------------------- fail-soft preserved
def test_a_failing_load_returns_empty_and_never_raises(capsys):
    """The per-frame contract is unchanged: a load failure yields [] and never
    propagates, because one bad frame must not void a whole video."""
    det = Detector("weights/definitely_not_here.pt", device="cpu")
    det._load_warned = True              # silence the banner; assert the contract
    assert det._model_or_none() is None
    assert det.detect(FRAME) == []
    assert det.track(FRAME) == []
    assert det.load_error is not None
    # "no banner again" - NOT "stderr is empty": importing ultralytics installs
    # a logging handler bound to the real stderr, which pytest has closed, so an
    # unrelated "Logging error: I/O operation on closed file" can appear here.
    # Assert on our own output, not on global silence.
    assert "FATAL" not in capsys.readouterr().err


def test_the_banner_reappears_on_a_fresh_instance(capsys):
    """One-shot is per instance, so a SECOND video (a new Detector) that also
    fails is not silent either."""
    first = Detector("weights/nope_one.pt", device="cpu")
    first.detect(FRAME)
    capsys.readouterr()
    second = Detector("weights/nope_two.pt", device="cpu")
    second.detect(FRAME)
    err = capsys.readouterr().err
    assert err.count("FATAL") == 1
    assert "nope_two.pt" in err


def test_inference_errors_are_not_swallowed():
    """Documents the pre-existing boundary, deliberately NOT changed here.

    Only the LOAD is wrapped. An exception raised by `model.predict()` on an
    already-loaded model still propagates to the caller, exactly as before this
    module existed. Widening that is a separate robustness decision (it would
    turn a mid-video inference crash into silently-empty detections, the very
    failure mode this file exists to make visible) and is not taken silently.
    """
    class _Boom:
        def predict(self, *a, **k):
            raise RuntimeError("CUDA out of memory")

        def track(self, *a, **k):
            raise RuntimeError("CUDA out of memory")

    det = Detector("weights/yolo11x.pt", device="cpu")
    det._model = _Boom()
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        det.detect(FRAME)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        det.track(FRAME)
