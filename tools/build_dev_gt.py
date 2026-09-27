"""Turn reviewed annotations into an evaluate.py-compatible dev ground truth.

evaluate.py is never modified or re-implemented here: the class list and the
segment rules are imported from it, so the file this tool writes is validated by
exactly the same code that will score the submission.

Input  reports/phase27_annotations.json   (the human review sheet; see --help)
Output my_labels_dev.json                 {<video>: {duration, fps, events}}

Only entries a human marked ``"status": "confirmed"`` become ground truth.
``uncertain`` and ``rejected`` never do -- they are counted and listed, because
silently promoting a maybe into a label is how a dev set quietly becomes fiction.
Pass --include-uncertain only to see the sensitivity of the score, and it says so
loudly in the output.

The ground-truth file contains video keys ONLY. evaluate.py iterates every top
level key of --gt and requires it to be a predicted video, so provenance and
build warnings go to a sidecar ``<out>_build_meta.json`` instead of inline.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import OFFICIAL_CLASSES  # noqa: E402

STATUSES = ("confirmed", "uncertain", "rejected")


def load_duration(video: str, table: dict, override: float | None) -> float | None:
    if override:
        return override
    if video in table:
        return float(table[video]["duration"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default="reports/phase27_annotations.json")
    ap.add_argument("--out", default="my_labels_dev.json")
    ap.add_argument("--durations", default="my_labels.json",
                    help="JSON with authoritative per-video duration/fps")
    ap.add_argument("--include-uncertain", action="store_true",
                    help="sensitivity run only: also count 'uncertain' as GT")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ann = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    dur_path = Path(args.durations)
    table = json.loads(dur_path.read_text(encoding="utf-8")) if dur_path.exists() else {}

    videos = ann.get("videos", {})
    problems: list[str] = []
    warnings: list[str] = []
    counts: dict[str, Counter] = {c: Counter() for c in OFFICIAL_CLASSES}

    out: dict[str, dict] = {}
    excluded: list[dict] = []
    for vid, block in videos.items():
        duration = load_duration(vid, table, block.get("duration"))
        fps = block.get("fps") or (table.get(vid, {}) or {}).get("fps")
        if duration is None:
            problems.append(f"{vid}: no duration (give it in the annotation sheet "
                            f"or in {dur_path})")
            continue
        kept: list[list] = []
        per_class: dict[str, list[tuple[float, float]]] = {}
        for i, ev in enumerate(block.get("events", [])):
            label = ev.get("class") or ev.get("label")
            status = ev.get("status", "confirmed")
            note = ev.get("note", "")
            if label not in OFFICIAL_CLASSES:
                problems.append(f"{vid}[{i}]: {label!r} is not an official class")
                continue
            if status not in STATUSES:
                problems.append(f"{vid}[{i}]: status {status!r} not in {STATUSES}")
                continue
            counts[label][status] += 1
            if status == "rejected":
                excluded.append({"video": vid, "class": label, "status": status,
                                 "note": note, "reason": "rejected by reviewer"})
                continue
            if status == "uncertain" and not args.include_uncertain:
                excluded.append({"video": vid, "class": label, "status": status,
                                 "note": note,
                                 "reason": "uncertain: kept out of ground truth"})
                continue
            if status == "uncertain":
                warnings.append(f"{vid}[{i}] {label}: UNCERTAIN counted as GT "
                                f"(--include-uncertain) note={note!r}")
            try:
                s, e = float(ev["start_sec"]), float(ev["end_sec"])
            except (KeyError, TypeError, ValueError):
                problems.append(f"{vid}[{i}]: start_sec/end_sec must be numbers")
                continue
            if not (0.0 <= s < e):
                problems.append(f"{vid}[{i}]: need 0 <= start < end, got [{s}, {e}]")
                continue
            if e > duration + 0.5:
                problems.append(f"{vid}[{i}]: end {e:.3f} > duration {duration:.3f} + 0.5")
                continue
            spans = per_class.setdefault(label, [])
            if any(s < pe and ps < e for ps, pe in spans):
                problems.append(f"{vid}[{i}]: {label} [{s:.2f},{e:.2f}] overlaps an "
                                f"earlier {label} segment; merge it by hand "
                                f"(annotators report one segment for both)")
                continue
            spans.append((s, e))
            kept.append([round(s, 3), round(e, 3), label])
        kept.sort()
        out[vid] = {"duration": round(float(duration), 3),
                    "fps": round(float(fps), 4) if fps else 0.0,
                    "events": kept}

    print(f"annotations : {args.annotations}")
    print(f"durations   : {dur_path} ({len(table)} video(s))")
    print(f"output      : {args.out}{' (dry run, not written)' if args.dry_run else ''}")
    print(f"mode        : confirmed only"
          f"{' + UNCERTAIN' if args.include_uncertain else ''}")
    print()
    print(f"{'class':<22}{'confirmed':>10}{'uncertain':>10}{'rejected':>10}{'in GT':>8}")
    tot = Counter()
    for c in OFFICIAL_CLASSES:
        k = counts[c]
        tot.update(k)
        in_gt = sum(1 for v in out.values() for _, _, lab in v["events"] if lab == c)
        print(f"{c:<22}{k['confirmed']:>10}{k['uncertain']:>10}{k['rejected']:>10}{in_gt:>8}")
    print(f"{'TOTAL':<22}{tot['confirmed']:>10}{tot['uncertain']:>10}{tot['rejected']:>10}"
          f"{sum(len(v['events']) for v in out.values()):>8}")
    print()
    for vid, v in out.items():
        print(f"  {vid:<14} duration={v['duration']:>8.2f}s  events={len(v['events'])}")
    zero = [c for c in OFFICIAL_CLASSES
            if not any(lab == c for v in out.values() for _, _, lab in v["events"])]
    print(f"\nclasses with zero confirmed examples ({len(zero)}): {', '.join(zero)}")
    if excluded:
        print(f"\nEXCLUDED from ground truth ({len(excluded)}):")
        for x in excluded:
            print(f"  {x['video']:<12} {x['class']:<20} {x['status']:<10} {x['reason']}")
    for wmsg in warnings:
        print(f"WARNING: {wmsg}")
    for p in problems:
        print(f"ERROR: {p}")

    if problems:
        print(f"\n{len(problems)} problem(s): fix the annotation sheet, nothing written.")
        return 1
    payload = dict(out)
    meta = {"source": args.annotations, "mode": "confirmed_only",
            "counts": {c: dict(counts[c]) for c in OFFICIAL_CLASSES},
            "excluded": excluded, "warnings": warnings}
    if not args.dry_run:
        Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        side = Path(args.out).with_name(Path(args.out).stem + "_build_meta.json")
        side.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out} and {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
