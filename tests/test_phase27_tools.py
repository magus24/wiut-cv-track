"""PHASE 27 dev-set tooling: the numbers must be the official numbers.

The dev set is the project's only measuring instrument, so the tools that produce
its numbers are worth locking down. Three failure modes are covered:

  1. `error_report` re-implements evaluate.match_segments in order to attribute
     a match to a segment. If that ever drifts from the official rule it would
     quietly report a different metric, so it is fuzzed against the real one.
  2. `build_dev_gt` must never let an `uncertain` or `rejected` entry become
     ground truth, and must refuse a sheet the official validator would reject.
  3. evaluate.py iterates EVERY top level key of --gt and requires it to be a
     predicted video, so a provenance block written inline into the GT file
     silently turns the whole submission INVALID. That is checked here.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import evaluate as E  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-arg]
    return mod


er = _load("p27_error_report", ROOT / "tools" / "error_report.py")
bg = _load("p27_build_dev_gt", ROOT / "tools" / "build_dev_gt.py")


# --------------------------------------------------------------------------
# 1. the pairing is the official pairing
# --------------------------------------------------------------------------

def test_pairing_matches_evaluate_on_a_hand_computed_case():
    gt = [(0.0, 10.0), (20.0, 30.0)]
    pred = [(0.0, 9.0), (1.0, 10.5), (21.0, 29.0)]
    for thr in E.TIOU_THRESHOLDS:
        matched, used_p, _ = er.official_pairs(gt, pred, thr)
        assert (len(matched), len(pred) - len(used_p), len(gt) - len(matched)) \
            == E.match_segments(gt, pred, thr)


def test_pairing_matches_evaluate_under_fuzz():
    rng = random.Random(20260926)
    for _ in range(300):
        gt = sorted((round(rng.uniform(0, 60), 2), round(rng.uniform(0, 60), 2))
                    for _ in range(rng.randint(0, 6)))
        gt = [(s, e) for s, e in gt if e > s]
        pred = sorted((round(rng.uniform(0, 60), 2), round(rng.uniform(0, 60), 2))
                      for _ in range(rng.randint(0, 6)))
        pred = [(s, e) for s, e in pred if e > s]
        for thr in E.TIOU_THRESHOLDS:
            er.check_against_official(gt, pred)


def test_a_duplicate_prediction_is_a_duplicate_not_a_second_tp():
    gt = {"v.mp4": {"duration": 60.0, "fps": 25.0,
                    "events": [[10.0, 20.0, "congestion"]]}}
    pred = {"team": "t", "videos": {"v.mp4": {"events": [
        [10.0, 20.0, "congestion"], [10.5, 20.0, "congestion"]], "risk": []}}}
    res = er.analyse(pred, gt, 0.5)
    rec = [e for e in res["per_event"] if e["side"] == "pred"]
    assert sorted(e["verdict"] for e in rec) == ["TP", "duplicate"]
    row = res["summary"][0]
    assert (row["gt"], row["pred"], row["tp"], row["fp"], row["fn"]) == (1, 2, 1, 1, 0)
    # the report rounds for display; the value must still be the official one
    assert row["f1"]["0.5"] == pytest.approx(E.evaluate_part_a(
        gt, pred["videos"])["per_class"]["congestion"]["0.5"]["f1"], abs=1e-4)


def test_boundary_error_is_measured_not_just_scored():
    gt = {"v.mp4": {"duration": 60.0, "fps": 25.0,
                    "events": [[10.0, 20.0, "stopped_vehicle"]]}}
    pred = {"team": "t", "videos": {"v.mp4": {"events": [
        [8.0, 18.0, "stopped_vehicle"]], "risk": []}}}
    res = er.analyse(pred, gt, 0.5)
    tp = [e for e in res["per_event"] if e["side"] == "pred"][0]
    assert tp["verdict"] == "TP"
    assert tp["start_err"] == pytest.approx(-2.0)
    assert tp["end_err"] == pytest.approx(-2.0)
    assert res["boundary"]["direction"] == {"start_too_early": 1, "start_too_late": 0,
                                            "end_too_early": 1, "end_too_late": 0}


def test_wrong_class_and_missed_are_distinguished():
    gt = {"v.mp4": {"duration": 60.0, "fps": 25.0,
                    "events": [[10.0, 20.0, "jaywalking"]]}}
    pred = {"team": "t", "videos": {"v.mp4": {"events": [
        [10.0, 20.0, "failure_to_yield"]], "risk": []}}}
    res = er.analyse(pred, gt, 0.5)
    fn = [e for e in res["per_event"] if e["side"] == "gt"][0]
    assert fn["verdict"] == "FN: wrong class"
    assert any(e.get("kind") == "cross_class" for e in res["per_event"])


def test_a_segment_found_but_misplaced_is_a_boundary_error_not_a_plain_fp():
    gt = {"v.mp4": {"duration": 60.0, "fps": 25.0,
                    "events": [[10.0, 40.0, "congestion"]]}}
    pred = {"team": "t", "videos": {"v.mp4": {"events": [
        [10.0, 22.0, "congestion"]], "risk": []}}}
    res = er.analyse(pred, gt, 0.5)
    iou = E.tiou((10.0, 40.0), (10.0, 22.0))
    assert iou == pytest.approx(0.4), "fixture must sit in the boundary band"
    assert res["summary"][0]["tp"] == 0 and res["summary"][0]["fp"] == 1
    rec = [e for e in res["per_event"] if e["side"] == "pred"][0]
    assert rec["verdict"] == "wrong temporal boundary"
    assert res["bands"]["0.3<=IoU<0.5"] == 1


# --------------------------------------------------------------------------
# 2. build_dev_gt keeps uncertainty out of the ground truth
# --------------------------------------------------------------------------

SHEET = {
    "videos": {
        "v.mp4": {
            "duration": 60.0, "fps": 25.0,
            "events": [
                {"class": "congestion", "start_sec": 1.0, "end_sec": 5.0,
                 "status": "confirmed", "note": "both lanes full"},
                {"class": "accident", "start_sec": 20.0, "end_sec": 22.0,
                 "status": "uncertain", "note": "cannot tell contact from swerve"},
                {"class": "fire_smoke", "start_sec": 30.0, "end_sec": 31.0,
                 "status": "rejected", "note": "it was a truck"},
            ],
        }
    }
}


def _build(tmp_path, sheet, capsys):
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(sheet), encoding="utf-8")
    out = tmp_path / "gt.json"
    rc = bg.main.__wrapped__() if hasattr(bg.main, "__wrapped__") else None
    argv = sys.argv
    sys.argv = ["build_dev_gt.py", "--annotations", str(ann), "--out", str(out),
                "--durations", str(tmp_path / "none.json")]
    try:
        rc = bg.main()
    finally:
        sys.argv = argv
    return rc, out, capsys.readouterr().out


def test_only_confirmed_events_become_ground_truth(tmp_path, capsys):
    rc, out, printed = _build(tmp_path, SHEET, capsys)
    assert rc == 0
    gt = json.loads(out.read_text(encoding="utf-8"))
    assert gt["v.mp4"]["events"] == [[1.0, 5.0, "congestion"]]
    assert "uncertain: kept out of ground truth" in printed
    assert "fire_smoke" in printed


def test_gt_file_contains_video_keys_only(tmp_path, capsys):
    """A provenance block inside the GT makes evaluate.py reject the submission.

    evaluate.validate iterates every top level key of --gt (evaluate.py:127) and
    requires it to be a predicted video, so an inline `_meta` is not ignored
    metadata - it is an unpredicted "video".
    """
    rc, out, _ = _build(tmp_path, SHEET, capsys)
    gt = json.loads(out.read_text(encoding="utf-8"))
    pred = {"team": "t", "videos": {k: {"events": [], "risk": []} for k in gt}}
    errors, _w = E.validate(pred, gt)
    assert errors == [], errors
    bad = dict(gt, _meta={"note": "inline provenance"})
    errors, _w = E.validate(pred, bad)
    assert errors, "an inline _meta key must be rejected by the official validator"
    assert "_meta" in errors[0]


def test_include_uncertain_is_opt_in_and_loud(tmp_path, capsys):
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(SHEET), encoding="utf-8")
    out = tmp_path / "gt2.json"
    argv = sys.argv
    sys.argv = ["build_dev_gt.py", "--annotations", str(ann), "--out", str(out),
                "--durations", str(tmp_path / "none.json"), "--include-uncertain"]
    try:
        assert bg.main() == 0
    finally:
        sys.argv = argv
    printed = capsys.readouterr().out
    gt = json.loads(out.read_text(encoding="utf-8"))
    assert sorted(lab for _s, _e, lab in gt["v.mp4"]["events"]) == ["accident", "congestion"]
    assert "UNCERTAIN counted as GT" in printed


@pytest.mark.parametrize("event,needle", [
    ({"class": "nope", "start_sec": 1, "end_sec": 2}, "not an official class"),
    ({"class": "congestion", "start_sec": 5, "end_sec": 5}, "0 <= start < end"),
    ({"class": "congestion", "start_sec": 1, "end_sec": 900}, "duration"),
    ({"class": "congestion", "start_sec": 1, "end_sec": 30}, "overlaps an earlier"),
    ({"class": "congestion", "start_sec": 1, "end_sec": 2, "status": "maybe"}, "status"),
])
def test_invalid_sheets_are_refused(tmp_path, capsys, event, needle):
    sheet = {"videos": {"v.mp4": {"duration": 60.0, "fps": 25.0,
                                  "events": [{"class": "congestion", "start_sec": 10.0,
                                              "end_sec": 20.0, "status": "confirmed"},
                                             event]}}}
    rc, out, printed = _build(tmp_path, sheet, capsys)
    assert rc == 1, printed
    assert needle in printed
    assert not out.exists(), "nothing may be written when the sheet is invalid"


def test_statuses_are_a_closed_set():
    assert set(bg.STATUSES) == {"confirmed", "uncertain", "rejected"}
