# PHASE 27 — dev-set baseline: status report

**Date:** 2026-09-26
**Verdict:** the workflow is complete and the measuring instrument is proven
correct, but **no score is reported**. Two independent blockers, both recorded
rather than worked around:

1. **No human ground truth.** A dev set cannot be produced by the assistant: the
   footage cannot be viewed here, and inventing labels would make every number
   meaningless. The annotation tooling is built and waiting for a human.
2. **The laptop cannot produce a valid submission.** Part A + Part B measure
   ~6.4x realtime against a 3.0x budget, and the CPU cost is 4K decode that no
   available knob throttles. The run was stopped rather than left to overheat the
   machine; the harness would have blanked all four videos anyway. Evidence and
   per-stage numbers: [`PHASE_27_RUNTIME.md`](PHASE_27_RUNTIME.md).

No score is invented, approximated, or copied from a fixture. Where a number does
not exist, this report says so.

## 1. What this phase established

### The official contract, confirmed from source

Read from `evaluate.py`, not from the docs:

- Predictions: `{"videos": {name: {"events": [[start, end, label], ...], "risk": [[t, score], ...]}}}`.
- Ground truth: a **top-level video mapping**, not a `videos` wrapper:
  `{"name": {"duration": ..., "fps": ..., "events": [[start, end, label], ...]}}`.
- `evaluate.validate` iterates **every** top-level key of `--gt` and requires it to
  exist in `pred["videos"]` (`evaluate.py:127`). A provenance block written inline
  into a GT file is therefore not ignored metadata, it is an unpredicted "video",
  and the whole submission is INVALID. `build_dev_gt.py` writes its provenance to
  a **sidecar** for exactly this reason; the trap is pinned by a test.
- Score A: per class, greedy one-to-one matching by descending temporal IoU at
  0.3 / 0.5 / 0.7, pooled over videos, macro F1 over `classes(GT) | classes(pred)`.
  A class that is only predicted joins the mean at zero.
- Score B: accident only, positive window `H=5 s` before the event, alarm match
  window `W=10 s`, `theta=0.5`. `M = 0.7*Score_A + 0.3*Score_B`.
- `--gt` accepts a custom file, so a self-labelled dev set is a legal measurement
  instrument; `--validate-only` checks format only and never needs GT.

### The sample inventory

| video | frames | fps | duration | resolution |
|---|---|---|---|---|
| `C3896.MP4` | 10,200 | 29.97 | 340.34 s | 3840x2160 |
| `C3897.MP4` | 9,525 | 29.97 | 317.82 s | 3840x2160 |
| `C3902.MP4` | 9,525 | 29.97 | 317.82 s | 3840x2160 |
| `C3905.MP4` | 3,825 | 29.97 | 127.63 s | 3840x2160 |

Total 1103.6 s. `AGENTS.md` still says `video/` is empty and that no `samples/`
directory exists; that is now stale — the four files are present, in
`C:\Users\user\Documents\Traffic Computer Vision\video`.

### Existing artefacts that are NOT ground truth

- `my_labels.json` — four videos, correct duration/fps, `events: []`. A scaffold.
- `labels_draft/C3905.draft.json` — produced by the detector under test, every
  entry `status: "unconfirmed"`. Using it as GT would score the detector against
  itself and inflate everything. Not used.
- `predictions_samples.json` — from unrelated video names, all entries empty.
- `examples/ground_truth.json` / `examples/predictions.json` — organizers'
  self-test fixture, used below **only** to prove a tool reproduces the official
  evaluator.

## 2. The measuring instrument is proven

`tools/error_report.py` re-implements the matching rule so it can attribute a
match to a specific segment and report *why* a segment failed. A report that
disagreed with `evaluate.py` would be worse than no report, so equivalence is
asserted at runtime (`check_against_official` on every call) and fuzzed in tests.

Self-test on the organizer fixture — `python evaluate.py --pred
examples/predictions.json --gt examples/ground_truth.json --per-video`:

```
Score_A = 0.9074    Score_B = 0.7600    MODEL SCORE = 0.8632
```

`python tools/error_report.py --pred examples/predictions.json --gt
examples/ground_truth.json --json-dir reports/tooltest` reproduces every figure
exactly, and additionally attributes the 9 segments:

```
accident          gt 2  pred 2  tp 2  fp 0  fn 0
jaywalking        gt 1  pred 1  tp 1  fp 0  fn 0
near_miss         gt 1  pred 1  tp 1  fp 0  fn 0
red_light         gt 2  pred 2  tp 2  fp 0  fn 0
stopped_vehicle   gt 1  pred 2  tp 1  fp 1  fn 0
wrong_way         gt 1  pred 1  tp 1  fp 0  fn 0
IoU bands  >=0.7: 7   0.5-0.7: 1   0.3-0.5: 0   <0.3: 1
boundary errors on 8 matched pairs: 1 end_too_early, 0 start errors
```

The FP is a second `stopped_vehicle` segment; the boundary error is a segment
ending 5 s early. That is exactly the kind of finding the phase is for, and it is
on a fixture, not on real footage.

Test status: `15 passed` in `tests/test_phase27_tools.py`, `10 passed` in
`tests/test_phase27_review_aids.py`, full suite `714 passed, 0 failed` in 33.6 s.
The tests cover matching equivalence under a seeded 300-case fuzz against
`evaluate.match_segments`, duplicate-vs-second-TP disambiguation, wrong-class vs
missed, misplaced-segment classification, GT shape traps, refusal of invalid
annotation sheets, the review aids on a synthetic clip, and two guide claims that
were previously only prose:

- `scan_motion`, `make_previews` and `prediction_stats` are walked with `ast` to
  prove they import nothing from `src/` or `solution`. That is what makes the
  shortlist admissible as a review aid rather than a detector in disguise.
  A substring search would be vacuous here: their own docstrings say in prose
  that they do not import `src`, so the string is present by construction.
- `scan_motion`'s CSV and JSON outputs are asserted to contain none of the 14
  class names, which is the machine-checked form of "a window is not an event".

Both were caught the honest way: four of the ten review-aid tests failed on the
first run. Three were wrong fixtures (CSV columns are `t_start`/`t_end`, page
names are zero-padded, and a duplicated segment is correctly rejected by the
official validator). The fourth was a genuine defect in the *test* — a
`"from src" not in source` assertion that the docstring made vacuous, exactly
the failure mode `AGENTS.md` warns about for `test_integration_14_events.py`.

## 3. The annotation workflow (ready, waiting for a human)

Working file: `reports/phase27_annotations.json` — four videos, all 14 official
classes with their definitions from `HACKATHON_CONTEXT.md:45-59`, and an `events`
array per video. Instructions: `reports/ANNOTATION_GUIDE.md`.

Three statuses per event: `confirmed` (becomes ground truth), `uncertain` and
`rejected` (excluded, but the note is kept for analysis). `build_dev_gt.py` emits
**confirmed only** by default and refuses the build on an unknown class, an empty
segment, a segment past the duration, or a same-class overlap.

Two candidate-finding aids, both deliberately class-agnostic and both
**inadmissible as labels**:

- `tools/make_previews.py` — timestamped contact sheets (full frame or crop) with
  the exact time range printed, so a segment boundary can be read off the image.
  Smoke-tested; the 4K frames are too large to view here, which is the blocker.
- `tools/scan_motion.py` — raw frame-difference shortlist, 1 s buckets, no
  `src` imports and no class semantics. Its output is a "look here first" list,
  never a label. **Run on `C3905.MP4` only** (the other three are deferred: at
  ~30 ms/frame of 4K decode a full dev-set sweep is another heating episode for
  no new information).

### Delivered for C3905 (127.63 s) — what a human should look at

`reports/scan/C3905.windows.csv`, `C3905.buckets.csv`, `C3905.scan.json`; motion
p50 = 3.72, p90 = 7.43, p99 = 8.16. Four candidate windows, deterministic across
two runs at `--stride 6 --pct 90`:

| window | length | peak | where | previews |
|---|---|---|---|---|
| 39.0 – 41.0 s | 2.0 s | 8.356 | 40.0 s | `previews/w39/` (9 tiles, 38–42 s @ 0.5 s) |
| 53.0 – 60.0 s | 7.0 s | 8.243 | 54.0 s | `previews/w53/` (19 tiles, 52–61 s @ 0.5 s, 2 pages) |
| 63.0 – 65.0 s | 2.0 s | 7.695 | 63.0 s | `previews/w63/` (9 tiles, 62–66 s @ 0.5 s) |
| 43.0 – 45.0 s | 2.0 s | 7.592 | 43.0 s | `previews/w43/` (9 tiles, 42–46 s @ 0.5 s) |

Plus a uniform overview at a 10 s stride for the long-lived classes:
`previews/C3905_overview/` (13 tiles, 2 pages). Every tile is stamped
`t=<seconds>s f=<frame>`, so `start_sec` can be read straight off the image.

Caveat to keep in mind while annotating: at a 10 s stride the overview is 29.97
frames apart, so it is blind to anything shorter than ~20 s, and the motion scan
is blind to a *stationary* event appearing (a newly parked vehicle, a stopped
lane) because raw pixel difference cannot see "nothing changed". The two together
are a sweep, not a proof of absence.

Intended order once a human is available:

```text
python tools/scan_motion.py --video "<abs path>\C3905.MP4" --out-dir reports\scan
python tools/make_previews.py --video "<abs path>\C3905.MP4" --out-dir previews --start 38 --end 42 --every 0.5
#   human fills reports/phase27_annotations.json
python tools/build_dev_gt.py --annotations reports/phase27_annotations.json --out my_labels_dev.json
python run_submission.py --videos "<abs path>\video" --out predictions.json --team wiut-cv-track
python evaluate.py --pred predictions.json --validate-only
python evaluate.py --pred predictions.json --gt my_labels_dev.json
python tools/error_report.py --pred predictions.json --gt my_labels_dev.json --json-dir reports
python tools/prediction_stats.py --pred predictions.json
```

`build_dev_gt.py` takes each video's duration from the sheet itself; `--durations`
is only an override and is not needed.

The per-video ground-truth segment count caps the value of the dev set. **Part B is
additionally gated on confirmed `accident` events**: with none, `Score_B`, alarm
F1 and mTTA must be reported as insufficient, not as zero, and AP measured on
non-accident footage must not be presented as a Part B result.

## 4. Timing baseline: measured, and it fails on this machine

Full detail in `PHASE_27_RUNTIME.md`. Summary:

| stage | wall | CPU | cores |
|---|---|---|---|
| YOLO11x @800 stride 3 | 4.50 s | 8.16 s | 1.81 |
| 4K decode to numpy | 1.42 s | 11.42 s | 8.06 |

`cv2.setNumThreads(1)`, `OPENCV_FFMPEG_CAPTURE_OPTIONS=threads;1` and
`TCV_OMP_THREADS=1` all failed to reduce decode below ~5.4 cores. D3D11VA hardware
decode works, frames verified identical, and buys 19% less CPU — not enough.
End to end that is **~6.4x realtime against a 3.0x budget**, so all four videos
would be blanked by the harness after ~118 min of wall time and a CPU package
observed at 99 C. The run was stopped at 13 min on `C3896.MP4`; no
`predictions.json` was written.

The GPU was in use throughout: `device=cuda:0`, `torch 2.14.0+cu126`,
`cuda=True`, `nvidia-smi` confirms the RTX 3070 Ti Laptop. Inference is not the
problem and is not what the user should try to switch off.

## 5. Part A and Part B status

| quantity | status | reason |
|---|---|---|
| Score_A | **not measured** | no confirmed GT; and no valid local submission |
| per-class F1 @0.3/0.5/0.7 | **not measured** | same |
| temporal boundary errors | **not measured** | same |
| Score_B / AP / alarm F1 / mTTA | **not measured** | same, and no confirmed accident |
| runtime vs 3x budget | **measured: fails locally (~6.4x)** | reported above |
| event/risk emission counts | **not measured** | requires the submission run |
| tool equivalence to `evaluate.py` | **measured: exact** | fixture self-test |
| C3905 review shortlist + contact sheets | **delivered** | 4 windows, ~55 tiles on disk |
| C3896 / C3897 / C3902 shortlist | **not run** | 4K decode is ~8 CPU cores; deferred |

`reports/phase27_class_metrics.json` and `reports/phase27_errors.json` are
deliberately **absent** for real footage; the only copies on disk are under
`reports/tooltest/`, named for what they are.

## 6. Files created or modified in PHASE 27

New, in the package `C:\Users\user\Documents\Traffic Computer Vision\Project\Project`:

- `tools/make_previews.py`, `tools/scan_motion.py`, `tools/build_dev_gt.py`,
  `tools/error_report.py`, `tools/prediction_stats.py`
- `tests/test_phase27_tools.py`, `tests/test_phase27_review_aids.py`
- `reports/phase27_annotations.json`, `reports/ANNOTATION_GUIDE.md`,
  `reports/PHASE_27_RUNTIME.md`, `reports/PHASE_27_BASELINE.md`
- `reports/tooltest/phase27_class_metrics.json`,
  `reports/tooltest/phase27_errors.json`
- `reports/scan/C3905.buckets.csv`, `C3905.windows.csv`, `C3905.scan.json`
- `previews/C3905_overview/`, `previews/w39/`, `previews/w43/`, `previews/w53/`,
  `previews/w63/` — contact sheets, ~55 JPEGs total

Throwaway measurement scripts live outside the repo in
`C:\Users\user\AppData\Local\Temp\opencode\p27\`.

**Protected files unchanged:** `evaluate.py`, `run_submission.py`, `solution.py`,
`src/detection/`, `src/events/`, `src/tracking/`, `src/scene/`, `src/risk/`,
`src/postprocessing/`, `src/config/`, `src/pipeline/`, the 14 ids in `CLASSES`,
and every threshold. No detector logic, geometry, temporal rule, interaction/TTC
parameter or Part B parameter was touched, so this phase cannot have improved the
model. Nothing was committed; the `Project/` tree is untracked, so provenance
here is the file list above rather than a git diff.

## 7. Recommended next step

Run the baseline on the grading-class hardware (a T4; `AGENTS.md` records
~45 ms/frame there, comfortably inside 3x) with a human-annotated
`my_labels_dev.json`, using the command block in section 3. If it must run on the
laptop, `--time-factor 10` on `C3905.MP4` alone yields a real Part A number in
~14 min at the cost of comparable timings and one more heating episode — a
judgement call, and the reason it was not made unilaterally here.
