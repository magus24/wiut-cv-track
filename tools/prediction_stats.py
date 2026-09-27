"""Summarise a predictions.json: what came out, and what it cost.

Ground-truth-free by design. Everything here is measurable without labels, which
matters because it is the part of the baseline that is available on day one: how
many events of each class the system emits, how long each part took against the
3x budget, and what the Part B curve looks like. Per-class accuracy needs a dev
set; these numbers do not.

    python tools/prediction_stats.py --pred predictions.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evaluate as E  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--json-out", default="reports/phase27_prediction_stats.json")
    ap.add_argument("--theta", type=float, default=E.THETA)
    args = ap.parse_args()

    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    log = pred.get("log", {})
    by_class: Counter = Counter()
    per_video = []
    risk_rows = []

    for vid, entry in pred.get("videos", {}).items():
        evs = entry.get("events", [])
        counts = Counter(e[2] for e in evs)
        by_class.update(counts)
        curve = entry.get("risk", []) or []
        scores = [float(s) for _t, s in curve]
        starts = E.alarm_starts(curve, theta=args.theta) if curve else []
        lg = log.get(vid, {})
        dur = lg.get("duration")
        budget = lg.get("budget_sec")
        total = lg.get("total_sec")
        row = {
            "video": vid,
            "duration": dur,
            "budget_sec": budget,
            "part_a_sec": lg.get("part_a_sec"),
            "part_b_sec": lg.get("part_b_sec"),
            "total_sec": total,
            "budget_used_pct": round(100.0 * total / budget, 1) if total and budget else None,
            "events": len(evs),
            "events_by_class": dict(sorted(counts.items())),
            "risk_samples": len(curve),
            "risk_median": round(statistics.median(scores), 4) if scores else None,
            "risk_max": round(max(scores), 4) if scores else None,
            "risk_mean": round(statistics.fmean(scores), 4) if scores else None,
            "frac_ge_theta": round(sum(1 for s in scores if s >= args.theta) / len(scores), 4)
            if scores else None,
            "alarm_runs": len(starts),
            "alarm_starts": [round(a, 2) for a in starts[:20]],
            "harness_problems": lg.get("errors", []),
        }
        per_video.append(row)
        if scores:
            risk_rows.append({
                "video": vid, "n": len(scores),
                "p50": row["risk_median"], "p90": round(
                    statistics.quantiles(scores, n=10)[8], 4) if len(scores) > 9 else None,
                "max": row["risk_max"],
                "n_ge_theta": sum(1 for s in scores if s >= args.theta),
            })

    errors, warnings = E.validate(pred, None)
    out = {
        "pred": args.pred,
        "team": pred.get("team"),
        "format_valid": not errors,
        "format_errors": errors,
        "format_warnings": warnings,
        "videos": len(pred.get("videos", {})),
        "events_total": sum(r["events"] for r in per_video),
        "events_by_class": dict(sorted(by_class.items())),
        "classes_with_zero_predictions": [c for c in E.OFFICIAL_CLASSES
                                          if by_class.get(c, 0) == 0],
        "per_video": per_video,
        "risk_summary": risk_rows,
        "totals": {
            "part_a_sec": round(sum(r["part_a_sec"] or 0 for r in per_video), 1),
            "part_b_sec": round(sum(r["part_b_sec"] or 0 for r in per_video), 1),
            "total_sec": round(sum(r["total_sec"] or 0 for r in per_video), 1),
            "budget_sec": round(sum(r["budget_sec"] or 0 for r in per_video), 1),
            "videos_over_budget": [r["video"] for r in per_video
                                   if r["harness_problems"]],
        },
    }
    print(f"predictions : {args.pred}")
    print(f"format      : {'VALID' if not errors else 'INVALID'} "
          f"({len(errors)} error(s), {len(warnings)} warning(s))")
    for e in errors:
        print("  ERROR:", e)
    print(f"videos      : {out['videos']}   events: {out['events_total']}")
    print()
    print(f"{'class':<22}{'predicted':>10}")
    for c in E.OFFICIAL_CLASSES:
        print(f"{c:<22}{by_class.get(c, 0):>10}")
    print()
    print(f"{'video':<14}{'dur':>8}{'budget':>8}{'A s':>7}{'B s':>7}{'tot s':>7}"
          f"{'used%':>7}{'events':>8}{'risk n':>8}{'p50':>7}{'max':>7}{'alarms':>7}")
    for r in per_video:
        print(f"{r['video']:<14}{(r['duration'] or 0):>8.1f}{(r['budget_sec'] or 0):>8.0f}"
              f"{(r['part_a_sec'] or 0):>7.1f}{(r['part_b_sec'] or 0):>7.1f}"
              f"{(r['total_sec'] or 0):>7.1f}{(r['budget_used_pct'] or 0):>7.1f}"
              f"{r['events']:>8}{r['risk_samples']:>8}{(r['risk_median'] or 0):>7.3f}"
              f"{(r['risk_max'] or 0):>7.3f}{r['alarm_runs']:>7}")
        for p in r["harness_problems"]:
            print(f"    ! {p.splitlines()[0]}")
    t = out["totals"]
    print(f"\ntotals: Part A {t['part_a_sec']}s + Part B {t['part_b_sec']}s = "
          f"{t['total_sec']}s against a combined budget of {t['budget_sec']}s")
    if t["videos_over_budget"]:
        print(f"videos with harness problems: {', '.join(t['videos_over_budget'])}")
    print(f"classes with zero predictions: {', '.join(out['classes_with_zero_predictions'])}")

    p = Path(args.json_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
