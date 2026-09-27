"""Per-event error analysis for the dev set, using the OFFICIAL matching rule.

evaluate.py is imported, never modified, and its own `tiou` / `match_segments` /
`evaluate_part_a` / `evaluate_part_b` are the source of truth. The greedy pairing
here is a deliberate re-implementation of evaluate.match_segments -- it has to
exist, because that function returns three integers and an error report needs to
know *which* segment went with which -- and every run asserts that the
re-implementation agrees with the official one on TP/FP/FN at all three
thresholds, per class and overall. If they ever disagree, this tool fails instead
of reporting a different metric.

    python tools/error_report.py --pred predictions.json --gt my_labels_dev.json

Outputs
    reports/phase27_class_metrics.json   per class TP/FP/FN/F1/P/R at each tIoU
    reports/phase27_errors.json          per-event verdicts, boundary errors,
                                         IoU band tallies, Part B block
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evaluate as E  # noqa: E402

THRS = E.TIOU_THRESHOLDS


def official_pairs(gt: list[tuple[float, float]],
                   pred: list[tuple[float, float]], thr: float):
    """Byte-for-byte the pairing rule of evaluate.match_segments, plus the pairs."""
    pairs = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            iou = E.tiou(g, p)
            if iou >= thr:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_g, used_p, matched = set(), set(), {}
    for iou, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        matched[i] = (j, iou)
    return matched, used_p, pairs


def check_against_official(gt, pred) -> None:
    for thr in THRS:
        want = E.match_segments(gt, pred, thr)
        matched, used_p, _ = official_pairs(gt, pred, thr)
        got = (len(matched), len(pred) - len(used_p), len(gt) - len(matched))
        assert want == got, f"pairing disagrees with evaluate.py at {thr}: {want} vs {got}"


def ev_key(vid, ev):
    return {"video": vid, "class": ev[2], "start": round(float(ev[0]), 3),
            "end": round(float(ev[1]), 3)}


def classify_pred(vid, pred_events, pred, j, matched_pairs, thr, gt_all):
    """Verdict for one predicted segment, using only official matching facts."""
    p = pred[j]
    best = None
    for gi, (pj, iou) in matched_pairs.items():
        if pj == j:
            best = (gi, iou)
    if best is not None:
        gi, iou = best
        return {"verdict": "TP", "gt_index": gi, "iou": round(iou, 4)}
    # unmatched. same-class GT with IoU in [0.3, thr) -> boundary problem
    same = [(gi, E.tiou(gt_all[gi], p)) for gi in range(len(gt_all))]
    same = [(gi, iou) for gi, iou in same if iou >= 0.3]
    if same:
        gi, iou = max(same, key=lambda x: x[1])
        if iou < thr:
            return {"verdict": "wrong temporal boundary", "gt_index": gi,
                    "iou": round(iou, 4)}
        # IoU >= thr but not matched => its GT was already taken: duplicate
        return {"verdict": "duplicate", "gt_index": gi, "iou": round(iou, 4)}
    return {"verdict": "FP", "gt_index": None,
            "iou": round(max([i for _, i in same], default=0.0), 4)}


def analyse(pred: dict, gt: dict, ref_thr: float) -> dict:
    part_a = E.evaluate_part_a(gt, pred.get("videos", {}), per_video=True)
    classes = part_a["classes"]

    per_event = []
    for vid, g in gt.items():
        p_events = pred.get("videos", {}).get(vid, {}).get("events", [])
        for c in classes:
            gsel = [(float(s), float(e)) for s, e, lab in g["events"] if lab == c]
            psel = [(float(s), float(e)) for s, e, lab in p_events if lab == c]
            check_against_official(gsel, psel)
            matched, used_p, _ = official_pairs(gsel, psel, ref_thr)
            inv = {pj: (gi, iou) for gi, (pj, iou) in matched.items()}
            for j, (ps, pe) in enumerate(psel):
                v = classify_pred(vid, p_events, psel, j, inv, ref_thr, gsel)
                rec = {**ev_key(vid, [ps, pe, c]), "side": "pred", "kind": "primary",
                       **v}
                if v["verdict"] == "TP":
                    gs, ge = gsel[v["gt_index"]]
                    rec["gt_start"], rec["gt_end"] = round(gs, 3), round(ge, 3)
                    rec["start_err"] = round(ps - gs, 3)
                    rec["end_err"] = round(pe - ge, 3)
                    rec["gt_duration"] = round(ge - gs, 3)
                    rec["pred_duration"] = round(pe - ps, 3)
                per_event.append(rec)
            for gi, (gs, ge) in enumerate(gsel):
                ious = [(j, E.tiou((gs, ge), psel[j])) for j in range(len(psel))]
                best_j, best_iou = (max(ious, key=lambda x: x[1]) if ious else (None, 0.0))
                if gi in matched:
                    pj, iou = matched[gi]
                    per_event.append({**ev_key(vid, [gs, ge, c]), "side": "gt",
                                      "kind": "primary", "verdict": "TP",
                                      "pred_index": pj, "iou": round(iou, 4),
                                      "pred_start": round(psel[pj][0], 3),
                                      "pred_end": round(psel[pj][1], 3)})
                    continue
                if best_iou >= 0.3:
                    why = "duplicate" if best_iou >= ref_thr else "wrong temporal boundary"
                else:
                    other = [(lab, E.tiou((gs, ge), (float(s), float(e))))
                             for s, e, lab in p_events if lab != c]
                    oc = max(other, key=lambda x: x[1]) if other else (None, 0.0)
                    why = ("wrong class" if oc[1] >= 0.3 else "missed (no overlap)")
                per_event.append({**ev_key(vid, [gs, ge, c]), "side": "gt",
                                  "kind": "primary", "verdict": f"FN: {why}",
                                  "best_same_class_iou": round(best_iou, 4),
                                  "best_pred_start": (round(psel[best_j][0], 3)
                                                      if best_j is not None else None),
                                  "best_pred_end": (round(psel[best_j][1], 3)
                                                    if best_j is not None else None)})

    # wrong-class cross links (a prediction sitting on another class's GT)
    for vid, g in gt.items():
        p_events = pred.get("videos", {}).get(vid, {}).get("events", [])
        for c in classes:
            gsel = [(float(s), float(e)) for s, e, lab in g["events"] if lab == c]
            psel = [(float(s), float(e), lab) for s, e, lab in p_events if lab != c]
            for ps, pe, lab in psel:
                iou = max((E.tiou(x, (ps, pe)) for x in gsel), default=0.0)
                if iou >= 0.3:
                    per_event.append({**ev_key(vid, [ps, pe, lab]), "side": "pred",
                                      "kind": "cross_class",
                                      "verdict": f"wrong class (sits on a {c} GT)",
                                      "iou": round(iou, 4)})

    summary = []
    for c in classes:
        pc = part_a["per_class"][c]
        gsel = [e for e in per_event
                if e["side"] == "gt" and e["kind"] == "primary" and e["class"] == c]
        psel = [e for e in per_event
                if e["side"] == "pred" and e["kind"] == "primary" and e["class"] == c]
        verdicts = defaultdict(int)
        for e in psel:
            verdicts[e["verdict"]] += 1
        for e in gsel:
            verdicts[e["verdict"]] += 1
        non_tp = {k: v for k, v in verdicts.items() if k != "TP"}
        main = max(non_tp, key=lambda k: non_tp[k]) if non_tp else "-"
        summary.append({
            "class": c,
            "gt": len(gsel), "pred": len(psel),
            "tp": pc[str(ref_thr)]["tp"], "fp": pc[str(ref_thr)]["fp"],
            "fn": pc[str(ref_thr)]["fn"],
            "f1": {str(t): round(pc[str(t)]["f1"], 4) for t in THRS},
            "precision": round(pc[str(ref_thr)]["precision"], 4),
            "recall": round(pc[str(ref_thr)]["recall"], 4),
            "verdicts": dict(sorted(verdicts.items())),
            "main_error_type": main,
        })

    bands = {"0.3<=IoU<0.5": 0, "0.5<=IoU<0.7": 0, "IoU>=0.7": 0, "IoU<0.3": 0}
    for e in per_event:
        if e["side"] != "pred" or e.get("kind") != "primary" or "iou" not in e:
            continue
        i = e["iou"]
        if i >= 0.7:
            bands["IoU>=0.7"] += 1
        elif i >= 0.5:
            bands["0.5<=IoU<0.7"] += 1
        elif i >= 0.3:
            bands["0.3<=IoU<0.5"] += 1
        else:
            bands["IoU<0.3"] += 1

    tps = [e for e in per_event
           if e["side"] == "pred" and e.get("kind") == "primary" and e["verdict"] == "TP"]
    def _stat(key):
        vals = [e[key] for e in tps if key in e]
        if not vals:
            return None
        return {"n": len(vals), "mean": round(statistics.fmean(vals), 3),
                "median": round(statistics.median(vals), 3),
                "min": round(min(vals), 3), "max": round(max(vals), 3),
                "early": sum(1 for v in vals if v < -0.05),
                "late": sum(1 for v in vals if v > 0.05)}
    boundary = {
        "ref_threshold": ref_thr, "n_matched": len(tps),
        "start_err_pred_minus_gt": _stat("start_err"),
        "end_err_pred_minus_gt": _stat("end_err"),
    }
    if tps:
        se = [e["start_err"] for e in tps]
        ee = [e["end_err"] for e in tps]
        d = []
        for e in tps:
            if "gt_duration" in e and e["gt_duration"] > 0:
                d.append(round((e["pred_duration"] - e["gt_duration"]) / e["gt_duration"], 3))
        boundary["relative_duration_err"] = {
            "n": len(d), "mean": round(statistics.fmean(d), 3),
            "median": round(statistics.median(d), 3)}
        boundary["direction"] = {
            "start_too_early": sum(1 for v in se if v < -0.05),
            "start_too_late": sum(1 for v in se if v > 0.05),
            "end_too_early": sum(1 for v in ee if v < -0.05),
            "end_too_late": sum(1 for v in ee if v > 0.05),
        }
    part_b = E.evaluate_part_b(gt, pred.get("videos", {}))
    return {"ref_threshold": ref_thr, "score_a": part_a["score_a"],
            "classes": classes, "summary": summary, "bands": bands,
            "boundary": boundary, "per_event": per_event,
            "micro": part_a["micro"], "class_agnostic": part_a["class_agnostic"],
            "per_video": part_a["per_video"], "part_b": part_b}


def render(res: dict) -> str:
    L = []
    thr = res["ref_threshold"]
    L.append(f"reference tIoU = {thr}   Score A = {res['score_a']:.4f}")
    L.append("")
    L.append(f"{'class':<22}{'GT':>4}{'pred':>6}{'TP':>4}{'FP':>4}{'FN':>4}"
             f"{'F1@0.3':>8}{'F1@0.5':>8}{'F1@0.7':>8}  main error")
    L.append("-" * 100)
    for s in res["summary"]:
        L.append(f"{s['class']:<22}{s['gt']:>4}{s['pred']:>6}{s['tp']:>4}{s['fp']:>4}"
                 f"{s['fn']:>4}{s['f1']['0.3']:>8.3f}{s['f1']['0.5']:>8.3f}"
                 f"{s['f1']['0.7']:>8.3f}  {s['main_error_type']}")
    L.append("")
    L.append("IoU bands over predicted segments with an overlapping GT: " +
             ", ".join(f"{k}={v}" for k, v in res["bands"].items()))
    b = res["boundary"]
    L.append(f"\nboundary errors on {b['n_matched']} matched pair(s) @ tIoU {thr}:")
    for key in ("start_err_pred_minus_gt", "end_err_pred_minus_gt"):
        st = b.get(key)
        if st:
            L.append(f"  {key:<28} mean={st['mean']:>8.3f}s median={st['median']:>8.3f}s "
                     f"min={st['min']:>8.3f} max={st['max']:>8.3f} "
                     f"early={st['early']} late={st['late']}")
    if "direction" in b:
        L.append(f"  direction: {b['direction']}")
    pb = res["part_b"]
    L.append("")
    if pb is None:
        L.append("Part B: not scored (no `accident` in the ground truth)")
    else:
        L.append(f"Part B: Score B = {pb['score_b']:.4f}  AP = {pb['ap']:.3f} "
                 f"(raw {pb['ap_raw']:.3f}, chance {pb['positive_rate']:.3f})  "
                 f"alarm F1 = {pb['f1_alarm']:.3f}  mTTA = {pb['mtta_sec']:.2f}s")
        L.append(f"         accidents={pb['n_accidents']} alarms={pb['n_alarms']} "
                 f"matched={pb['n_matched']} frames={pb['n_frames_scored']} "
                 f"positive={pb['n_frames_positive']}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ref-thr", type=float, default=0.5)
    ap.add_argument("--json-dir", default="reports")
    args = ap.parse_args()

    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    errors, warnings = E.validate(pred, gt)
    print(f"validate: {len(errors)} error(s), {len(warnings)} warning(s)")
    for e in errors:
        print("  ERROR:", e)
    for w in warnings:
        print("  warning:", w)
    if errors:
        print("refusing to analyse an invalid prediction file")
        return 1

    res = analyse(pred, gt, args.ref_thr)
    print()
    print(render(res))
    out = Path(args.json_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase27_class_metrics.json").write_text(json.dumps({
        "score_a": res["score_a"], "ref_threshold": res["ref_threshold"],
        "classes": res["classes"], "summary": res["summary"],
        "micro": res["micro"], "class_agnostic": res["class_agnostic"],
        "per_video": res["per_video"], "part_b": res["part_b"],
    }, indent=1), encoding="utf-8")
    (out / "phase27_errors.json").write_text(json.dumps({
        "bands": res["bands"], "boundary": res["boundary"],
        "per_event": res["per_event"],
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out / 'phase27_class_metrics.json'}")
    print(f"wrote {out / 'phase27_errors.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
