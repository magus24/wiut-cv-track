# WIUT Hackathon 2026 — Computer Vision track

Traffic events from a fixed road camera: **detect** them as time segments
(`[start_sec, end_sec, label]`) and, as a bonus, **anticipate** accidents with a
causal risk score. This is our submission; the organizers' harness and metric are
included unmodified.

```
solution.py          <- the ONLY file you implement (CLASSES, detect_events, RiskEstimator)
run_submission.py    <- organizers' harness: folder of videos -> predictions.json   (do not modify)
evaluate.py          <- format check + the official metric                          (do not modify)
examples/            <- ground_truth.json and predictions.json in the exact format
requirements.txt     <- fully pinned, including the CUDA torch build (see "Verified environment")
src/                 <- the implementation (pipeline, detection, tracking, scene, events, risk)
scene_config.json    <- self-calibrated geometry for the fixed camera (all videos share it)
weights/             <- yolo11x.pt (shipped) + download.sh (fallback)
tests/               <- 728 tests, run with `python -m pytest tests -q`
```

## Quickstart

```bash
pip install -r requirements.txt
# 1. implement solution.py
# 2. label the sample videos yourselves -> my_labels.json (same shape as examples/ground_truth.json)
python run_submission.py --videos samples --out predictions_samples.json --team <your-team>
python evaluate.py --pred predictions_samples.json --gt my_labels.json --per-video
python evaluate.py --pred predictions_samples.json --validate-only        # format check without labels
```

## The interface (`solution.py`)

```python
CLASSES = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
           "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
           "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke"]

def detect_events(video_path: str) -> list[list]:
    """Part A: [[start_sec, end_sec, label], ...]; label in CLASSES; same-class segments don't overlap."""

class RiskEstimator:
    def reset(self, meta: dict) -> None: ...            # meta: video_id, fps, width, height, n_frames
    def step(self, frame: np.ndarray, t_sec: float) -> float: ...   # BGR uint8 frame -> P(accident within 5 s)
```

`step` is called for **every frame in order** by the harness; it must not open the
video itself. Skipping frames internally and returning the last score is fine.
You may remove ids from `CLASSES`; never add.

## What we run (offline, one GPU, no internet)

```bash
pip install -r requirements.txt            # or: docker build -t team .
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --gt ground_truth.json
```

Time budget per video: **3 × its duration** for Part A + Part B together; a video
over budget or a crash scores as empty. Events with a bad label, bad times, or a
same-class overlap are dropped by the harness and listed in its log.

### Weights

`weights/yolo11x.pt` **is included in this package** (109 MB, well inside the
5 GB cap), so the offline run needs no download step. `weights/download.sh`
remains as a fallback and re-fetches the identical file from the Ultralytics
release URL if the weight is ever missing.

Do not rely on `download.sh` at scoring time: grading has no internet, and a
missing weight file is a *silent* total loss — every frame returns zero
detections, every video is submitted with an empty event list, `Score_A` and
`Score_B` are both 0, and `evaluate.py --validate-only` still reports `VALID`
because an empty event list is legal. `src/detection/detector.py` therefore
prints a loud one-time banner on stderr and records `Detector.load_error` if the
model cannot be loaded, and `Detector.preflight()` reports the problem directly.

## Verified environment

`requirements.txt` is fully pinned and was validated the way the graders run it:
a **clean virtualenv** built from that exact file, then the full suite and the
organizer's own format check.

| Package | Pin | Note |
|---|---|---|
| `torch` | `2.12.0+cu126` | CUDA 12.6 build, from the official PyTorch index |
| `torchvision` | `0.27.0+cu126` | matching build |
| `numpy` | `2.4.6` | |
| `opencv-python` | `5.0.0.93` | see the note below |
| `ultralytics` | `8.4.53` | provides YOLO11x + ByteTrack |

Result: **727 passed, 1 skipped** (the skip needs a real sample video, which is
not part of the submission), and
`format: 1 video(s), 14 event(s), 0 error(s), 0 warning(s) -> VALID`.

Two details that are easy to get wrong:

- **`--extra-index-url`, not `--index-url`.** A bare `--index-url` in a
  requirements file is a *global* option, so it would also send the
  numpy/opencv/ultralytics lookups to the PyTorch index, which does not mirror
  them. The PyTorch index is therefore added as an extra index and PyPI stays
  primary.
- **`opencv-python`, not `opencv-python-headless`.** Ultralytics depends on
  `opencv-python`, and both distributions install into the same `cv2` package
  directory. Pinning the headless build while Ultralytics pulls in the regular
  one installs *both*, and the unpinned copy silently overwrites the pinned one
  — observed as `cv2.__version__ == 5.0.0` while the pin said `4.13.0.92`.
  Pinning the single distribution Ultralytics actually requires removes that
  failure mode. On a slim container image OpenCV also needs the system libraries
  `libgl1` and `libglib2.0-0`.

Determinism: the inference path contains no RNG at all — no `random`, no
`np.random`, no `torch.manual_seed` call — so two runs on the same machine
produce byte-identical output by construction rather than by seeding. There is
no wall-clock or filename-order dependence either.

## Our approach

**Part A** is one YOLO11x + ByteTrack pass per video, sampled every 3rd frame
(`TCV_STRIDE=3`). `src/pipeline/pipeline.py` decodes each video exactly once and
feeds one `EventManager`; every class provider is registered in a single
registry (`src/events/manager.py`, `DEFAULT_PROVIDERS`) that maps each of the 14
official ids to exactly one provider, so a label can never be reported twice.
Events are rules over trajectories plus the calibrated scene geometry
(`scene_config.json` → `src/scene/geometry.py`: lanes, crosswalks, stop line,
solid line, u-turn zones, exclusion region). Pairwise classes (`accident`,
`near_miss`) share a single `PairwiseInteractionEngine` and one pair sweep per
frame instead of one each. Segments are merged and sub-second blips dropped in
post-processing.

**Part B** is deliberately self-contained: its own trajectory/motion/pairwise
engines, three risk channels (TTC, hard braking, pedestrian predicted-approach)
max-aggregated and smoothed by a single asymmetric EMA. The harness calls
`step()` for every frame, but the estimator observes at ~10 Hz
(`_stride_for_fps`, e.g. stride 3 at 29.97 fps) and re-serves the last score in
between, so the expensive detection runs ~10x/s rather than ~30x/s. There is
deliberately no max-hold/peak-extension stage: `evaluate.py` counts a 0.2 s
excursion above threshold as a full alarm, so a steep, honest ramp scores better
than a delayed flat-topped one. Measured runtime is ≈1.85x real time against the
3x budget on the documented T4 figures.

**Determinism** is structural, not seed-based: there is no RNG anywhere in the
inference path, and no per-video wall-clock or load-dependent decisions (the
budget guard is a pure function of frame count and a declared per-device cost).
Two runs on the same machine produce identical output. Note this is *not* the same
as a fixed seed — if randomness is ever introduced it must be seeded explicitly.

## predictions.json

```json
{
  "team": "your-team-name",
  "videos": {
    "test_001.mp4": {
      "events": [[12.4, 18.9, "accident"], [40.0, 43.5, "red_light"]],
      "risk":   [[0.00, 0.01], [0.04, 0.01], [0.08, 0.02]]
    },
    "test_002.mp4": {"events": [], "risk": []}
  }
}
```

`risk` is written by the harness (one `[t_sec, score]` per frame). Keys are file
names. Every test video must be present, even with `"events": []`.
Ground truth: `{"test_001.mp4": {"duration": 600.0, "fps": 25.0, "events": [[12.0, 19.0, "accident"]]}}`.

## Metric (exact code in `evaluate.py`)

**Part A.** Per class `c` and per tIoU threshold τ ∈ {0.3, 0.5, 0.7}: greedy
one-to-one matching by descending IoU; TP/FP/FN pooled over all videos; `F1_c(τ)`.
`Score_A = mean_c mean_τ F1_c(τ)`. Classes = those in the ground truth or in your
predictions (a class you predict that never occurs scores 0).

**Part B** (`accident` only; H = 5 s, W = 10 s, θ = 0.5). Frames in `[s−H, s)`
before an accident start `s` are positive; frames inside accidents and around
near-misses are ignored; the rest negative. `AP` = average precision over frames,
chance-normalised (`max(0, (AP_raw − r)/(1 − r))`, `r` = positive rate, so a
constant score gets 0). Alarms = runs of score ≥ θ (runs < 2 s apart merged),
alarm time = run start; an alarm in `[s−W, s)` of an unmatched accident matches it
→ `F1_alarm`; `mTTA` = mean of `s − alarm_time` (0 if unmatched).
`Score_B = 0.4·AP + 0.4·F1_alarm + 0.2·mTTA/W`.

**Model score** `M = 0.7·Score_A + 0.3·Score_B` (M = Score_A if the test set has no
accidents). Elimination score = 0.6·M + 0.25·Website + 0.15·Code.

## Tips

- Label the sample videos yourselves with the conventions from the task
  description and run `evaluate.py` against them. Without a dev set you are guessing.
- Detector + tracker → trajectories; most classes are rules on trajectories plus
  the scene layout. Learned models help most for `accident` / `near_miss`.
- Post-process segments: merge fragments, drop sub-second blips, then check F1@0.7.
- For Part B, time-to-collision from tracks is a strong simple signal; calibrate
  so that 0.5 means "probably within 5 s". A flat 1.0 scores ≈ 0.
- Print your runtime early; sampling every 2nd–5th frame is usually enough.
