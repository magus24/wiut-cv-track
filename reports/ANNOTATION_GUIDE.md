# Dev-set annotation guide (PHASE 27)

The dev set is the only way to tell whether a change helped or hurt. It is also
the easiest thing in this project to ruin: a ground truth built from the
detector's own output is not a measurement, it is a mirror.

## The one rule

**Ground truth comes from the footage, or it does not come at all.**

Not from `predictions.json`, not from `debug/*_candidates_*.csv`, not from
`labels_draft/`. Those are the system's opinions. If they are copied into the GT,
Score_A measures self-consistency and every later decision is made on noise.

## Files

| file | role |
|---|---|
| `reports/phase27_annotations.json` | **the only file you edit** |
| `tools/make_previews.py` | contact sheets so you can actually watch a 4K clip |
| `tools/scan_motion.py` | shortlists *when* to look, using raw pixel change only |
| `tools/build_dev_gt.py` | turns the sheet into the official GT schema |
| `my_labels_dev.json` | generated — do not hand-edit |
| `my_labels_dev_build_meta.json` | generated — what was included, excluded and why |
| `tools/error_report.py` | per-event verdicts, boundary errors, Part B |
| `tools/prediction_stats.py` | what came out and what it cost — needs no GT |
| `reports/scan/`, `previews/` | generated: motion shortlist and contact sheets |

## Workflow

Run everything from the package directory (`...\Traffic Computer Vision\Project\Project`).
The sample videos are **two levels up**, so set this once:

```powershell
$VIDEO = "C:\Users\user\Documents\Traffic Computer Vision\video"
```

### 1. Overview sweep (catches everything long-lived)

```powershell
python tools/make_previews.py --video "$VIDEO\C3905.MP4" --out-dir previews\C3905_overview --every 10 --cols 4 --rows 3 --tile-w 640
```

Each tile is stamped `t=12.40s f=372`, so you can write `start_sec` straight off
the picture. Long events (`congestion`, `wrong_way`, `stopped_vehicle`) are
visible at a 10 s stride; short ones are not, which is step 2.

### 2. Motion shortlist (catches the short events)

```powershell
python tools/scan_motion.py --video "$VIDEO\C3905.MP4" --out-dir reports\scan --pct 90
```

Reads consecutive-frame pixel difference and buckets it per second; the windows
are where the picture changes fastest. It has no idea what an accident is — it
only says "look here". **A window is not an event.** Open the candidates with
`--start/--end/--every 1`:

```powershell
python tools/make_previews.py --video "$VIDEO\C3905.MP4" --out-dir previews\C3905_w1 --start 96 --end 108 --every 0.5
```

For anything you must read literally (a signal head, a pedestrian's feet relative
to a crosswalk) use `--native` or `--crop`, so you are not judging a 640 px
downscale.

#### Already done for C3905 — start here

`C3905.MP4` has been swept and the contact sheets are on disk. Motion p50 = 3.72,
p90 = 7.43, p99 = 8.16; four candidate windows, stable across two runs:

| window | len | peak | previews on disk |
|---|---|---|---|
| **39.0 – 41.0 s** | 2.0 s | 8.356 @ 40.0 s | `previews/w39/` (9 tiles, 38–42 s @ 0.5 s) |
| **43.0 – 45.0 s** | 2.0 s | 7.592 @ 43.0 s | `previews/w43/` (9 tiles, 42–46 s @ 0.5 s) |
| **53.0 – 60.0 s** | 7.0 s | 8.243 @ 54.0 s | `previews/w53/` (19 tiles, 52–61 s @ 0.5 s, 2 pages) |
| **63.0 – 65.0 s** | 2.0 s | 7.695 @ 63.0 s | `previews/w63/` (9 tiles, 62–66 s @ 0.5 s) |

Raw numbers: `reports/scan/C3905.windows.csv`, `C3905.buckets.csv`,
`C3905.scan.json`. Uniform 10 s overview for the long-lived classes:
`previews/C3905_overview/` (13 tiles, 2 pages).

The other three videos are **not** swept yet — 4K decode is ~8 CPU cores on this
laptop, so the sweep of the full 1104 s dev set was deliberately not run. Sweep
them one at a time when the machine is cool:

```powershell
python tools\scan_motion.py --video "$VIDEO\C3896.MP4" --out-dir reports\scan --stride 6 --pct 90
```

Two things the sweep cannot tell you, so do not treat an empty window as an
absence:

- **A stationary event is invisible to it.** Raw frame difference cannot see "a
  car parked and stayed parked" or "a lane stayed blocked". `stopped_vehicle`,
  `congestion` and `road_obstacle` mostly do not move; they are the classes the
  10 s overview exists for.
- **A 10 s stride is blind to anything under ~20 s.** The overview finds
  long-lived states, the motion scan finds brief motion, and the two together
  still are not proof that a class is absent.

### 3. Write the labels

Edit `reports/phase27_annotations.json`, one entry per event:

```json
{"class": "failure_to_yield", "start_sec": 41.2, "end_sec": 44.8,
 "status": "confirmed", "note": "car 2 enters the crosswalk while ped 12 is on it"}
```

Boundaries: `start_sec` is the first moment the event is *clearly* active,
`end_sec` the last. Use seconds. If two same-class events run together, emit one
segment covering both — annotators do the same, and the metric cannot match two
overlapping predictions to two overlapping labels anyway.

Statuses:

* `confirmed` — you can point at the pixels.
* `uncertain` — you cannot. It is **excluded** from the GT and listed in the build
  report. Say what made it unclear. This is a normal, useful outcome.
* `rejected` — you looked, it is not that class. Say what it was instead.

Leave a class out entirely if it does not occur. An empty class is information;
an invented one is poison, because `evaluate.py` scores over `GT ∪ predictions`
and a predicted class that never occurs adds a 0 to the macro mean.

### 4. Build and check

```powershell
python tools/build_dev_gt.py --annotations reports/phase27_annotations.json --out my_labels_dev.json
python evaluate.py --pred predictions.json --validate-only
python tools/error_report.py --pred predictions.json --gt my_labels_dev.json
```

`build_dev_gt.py` imports `OFFICIAL_CLASSES` from `evaluate.py` and re-applies
its segment rules, so it will refuse an invalid sheet instead of writing a GT
file that the official harness rejects. It exits non-zero on any problem.

To see how much the uncertain entries would have mattered:

```powershell
python tools/build_dev_gt.py --annotations reports/phase27_annotations.json --out my_labels_dev_unc.json --include-uncertain
```

## What the numbers mean

* `Score_A` is a macro average over `GT ∪ predicted` classes. Adding a class to
  either side changes the denominator, so **the class list is part of the
  metric** — a run-to-run comparison is only valid if the class set is unchanged.
* `tIoU >= 0.7` is the strict threshold. An event that is right but a second or
  two long fails it while passing 0.3, which is why `error_report.py` reports the
  boundary errors rather than just the F1.
* Part B needs at least one `accident` in the GT. Without one the official
  evaluator reports `Part B not scored` and `M = Score_A`; that is a real result,
  not a bug, and it must not be presented as a Part B estimate.
