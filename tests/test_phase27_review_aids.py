"""The three PHASE 27 review aids must work, and must stay class-agnostic.

`error_report` and `build_dev_gt` are covered in test_phase27_tools.py. This file
covers the other three tools that the annotation guide tells a human to run:
`scan_motion`, `make_previews` and `prediction_stats`.

Two claims in the guide are machine-checked rather than asserted in prose:

  * "It has no idea what an accident is ... does not import anything from src/".
    That is what makes the shortlist admissible as a review aid. A single stray
    `from src.events...` import would let detector opinions leak into the review,
    so the import graph is walked instead of grepped for a comment.
  * "A window is not an event." The scan's output carries no class field, and the
    guide must not tell a reviewer to write one.

Everything here runs on a synthetic 64x64 clip, so the suite stays fast and does
not touch the 4K sample footage (whose decode alone is ~8 CPU cores).
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import evaluate as E  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-arg]
    return mod


sm = _load("p27_scan_motion", ROOT / "tools" / "scan_motion.py")
mp = _load("p27_make_previews", ROOT / "tools" / "make_previews.py")
ps = _load("p27_prediction_stats", ROOT / "tools" / "prediction_stats.py")

W, H, FPS = 64, 64, 10.0
STILL_A, STILL_B = 20, 40


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    """30 s synthetic clip: a box crosses the frame between 2 s and 4 s."""
    path = tmp_path_factory.mktemp("clips") / "synthetic.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert vw.isOpened(), "synthetic clip could not be encoded"
    rng = np.random.default_rng(7)
    for i in range(300):
        f = np.full((H, W, 3), 40, np.uint8)
        if STILL_A <= i < STILL_B:
            x = int((i - STILL_A) / max(1, STILL_B - STILL_A - 1) * (W - 8))
            cv2.rectangle(f, (x, H // 2 - 4), (x + 8, H // 2 + 4), (220, 220, 220), -1)
        else:
            f = np.clip(f.astype(np.int16) + rng.integers(-1, 2, f.shape),
                        0, 255).astype(np.uint8)
        vw.write(f)
    vw.release()
    return path


def _run(mod, argv: list[str]) -> int:
    old = sys.argv
    sys.argv = [mod.__name__] + argv
    try:
        return mod.main()
    finally:
        sys.argv = old


# --------------------------------------------------------------------------
# scan_motion: finds the moving part, and cannot know what it is
# --------------------------------------------------------------------------

def test_scan_finds_the_moving_window_and_ignores_the_still_parts(clip, tmp_path):
    out = tmp_path / "scan"
    assert _run(sm, ["--video", str(clip), "--out-dir", str(out),
                     "--stride", "1", "--pct", "95"]) == 0
    rows = list(csv.DictReader((out / "synthetic.windows.csv").open(encoding="utf-8")))
    assert rows, "a box crossing the frame must produce at least one window"
    hits = [r for r in rows
            if float(r["t_start"]) <= 2.5 <= float(r["t_end"])
            or float(r["t_start"]) <= 3.5 <= float(r["t_end"])]
    assert hits, f"the 2-4 s motion was missed; windows were {rows}"
    for r in rows:
        assert float(r["t_end"]) > float(r["t_start"])
        assert float(r["length_sec"]) > 0


def test_scan_output_carries_no_class(clip, tmp_path):
    """The guide says 'a window is not an event'; the schema must enforce it."""
    out = tmp_path / "scan"
    _run(sm, ["--video", str(clip), "--out-dir", str(out), "--stride", "1"])
    for name in ("synthetic.windows.csv", "synthetic.buckets.csv"):
        text = (out / name).read_text(encoding="utf-8")
        for banned in E.OFFICIAL_CLASSES:
            assert banned not in text, f"{name} leaked the class {banned!r}"
    blob = (out / "synthetic.scan.json").read_text(encoding="utf-8")
    for banned in E.OFFICIAL_CLASSES:
        assert banned not in blob


def test_review_aids_cannot_import_the_system_they_are_measuring():
    """Admissibility of the shortlist rests on this, so walk the AST.

    The AST walk is the whole check. A substring search for "from src" would be
    vacuous: these docstrings say in prose that they do not import src, so the
    string is present by construction.
    """
    for tool in ("scan_motion.py", "make_previews.py", "prediction_stats.py"):
        source = (ROOT / "tools" / tool).read_text(encoding="utf-8")
        found = []
        for node in ast.walk(ast.parse(source)):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            found += [m for m in mods
                      if m == "src" or m.startswith("src.") or m.startswith("solution")]
        assert not found, f"{tool} imports {found}; a review aid must stay independent"


# --------------------------------------------------------------------------
# make_previews: the timestamp is the whole point, so it must be on the tile
# --------------------------------------------------------------------------

def test_preview_sheet_names_its_time_range_and_exists(clip, tmp_path):
    out = tmp_path / "prev"
    assert _run(mp, ["--video", str(clip), "--out-dir", str(out),
                     "--start", "2", "--end", "4", "--every", "0.5",
                     "--cols", "2", "--tile-w", "128"]) == 0
    pages = sorted(out.glob("*.jpg"))
    assert pages, "no contact sheet was written"
    assert any("0002.0-0004.0" in p.name for p in pages), \
        f"the page name must carry the time range, got {[p.name for p in pages]}"
    img = cv2.imread(str(pages[0]))
    assert img is not None and img.shape[0] > 128, "sheet looks empty"


def test_preview_native_keeps_the_source_resolution(clip, tmp_path):
    out = tmp_path / "nat"
    _run(mp, ["--video", str(clip), "--out-dir", str(out), "--times", "3", "--native"])
    pages = list(out.glob("*.jpg"))
    assert pages
    img = cv2.imread(str(pages[0]))
    assert img.shape[1] >= W, "native mode must not downscale"


def test_parse_times_accepts_the_forms_the_guide_uses():
    assert mp.parse_times("30,60,90") == [30.0, 60.0, 90.0]
    assert mp.parse_times(" 1.5 , 2 ") == [1.5, 2.0]
    with pytest.raises(Exception):
        mp.parse_times("not,a,time")


# --------------------------------------------------------------------------
# prediction_stats: counts events and flags the budget failure mode
# --------------------------------------------------------------------------

def _pred():
    return {
        "team": "t",
        "videos": {
            "a.mp4": {"events": [[1.0, 2.0, "congestion"], [3.0, 9.0, "jaywalking"]],
                      "risk": [[0.0, 0.0], [1.0, 0.9], [2.0, 0.1], [3.0, 0.8]]},
            "b.mp4": {"events": [], "risk": []},
        },
        "log": {
            "a.mp4": {"duration": 10.0, "budget_sec": 30.0, "part_a_sec": 10.0,
                      "part_b_sec": 10.0, "total_sec": 20.0, "errors": []},
            "b.mp4": {"duration": 10.0, "budget_sec": 30.0, "part_a_sec": 40.0,
                      "part_b_sec": 40.0, "total_sec": 80.0,
                      "errors": ["over time budget (80.0s > 30s): scored as empty"]},
        },
    }


def test_prediction_stats_counts_events_and_risk(tmp_path):
    p = tmp_path / "pred.json"
    p.write_text(json.dumps(_pred()), encoding="utf-8")
    assert _run(ps, ["--pred", str(p),
                     "--json-out", str(tmp_path / "s.json")]) == 0
    out = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert out["format_valid"] is True
    assert out["events_total"] == 2
    assert out["events_by_class"] == {"congestion": 1, "jaywalking": 1}
    a = next(r for r in out["per_video"] if r["video"] == "a.mp4")
    assert a["risk_samples"] == 4
    assert a["risk_max"] == 0.9
    assert a["risk_median"] == 0.45
    assert a["alarm_runs"] == 2, "0.9 and 0.8 are separate runs across a 0.1 dip"
    assert a["budget_used_pct"] == pytest.approx(66.7, abs=0.1)


def test_prediction_stats_flags_an_over_budget_video(tmp_path):
    p = tmp_path / "pred.json"
    p.write_text(json.dumps(_pred()), encoding="utf-8")
    _run(ps, ["--pred", str(p), "--json-out", str(tmp_path / "s.json")])
    out = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert out["totals"]["videos_over_budget"] == ["b.mp4"]
    b = next(r for r in out["per_video"] if r["video"] == "b.mp4")
    assert b["budget_used_pct"] == pytest.approx(266.7, abs=0.1)


def test_prediction_stats_lists_the_classes_we_never_predict(tmp_path):
    """Classes with zero events still have to be visible: they enter the mean."""
    p = tmp_path / "pred.json"
    p.write_text(json.dumps(_pred()), encoding="utf-8")
    _run(ps, ["--pred", str(p), "--json-out", str(tmp_path / "s.json")])
    out = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    zero = set(out["classes_with_zero_predictions"])
    assert zero == set(E.OFFICIAL_CLASSES) - {"congestion", "jaywalking"}
    assert "accident" in zero, "Part B's class must be explicitly absent, not hidden"


def test_prediction_stats_reports_an_invalid_file_rather_than_scoring_it(tmp_path):
    """Overlapping same-class segments are a hard format error; do not paper over."""
    bad = _pred()
    bad["videos"]["a.mp4"]["events"].append([3.0, 9.0, "jaywalking"])
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    _run(ps, ["--pred", str(p), "--json-out", str(tmp_path / "s.json")])
    out = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert out["format_valid"] is False
    assert any("overlapping" in e for e in out["format_errors"])
