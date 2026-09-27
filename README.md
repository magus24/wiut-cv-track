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
requirements.txt     <- pinned deps EXCEPT torch (safe to install offline)
requirements-torch.txt <- the pinned CUDA torch/torchvision, for a clean build
Dockerfile           <- offline-safe image (build BEFORE the offline evaluation)
src/                 <- the implementation (pipeline, detection, tracking, scene, events, risk)
scene_config.json    <- self-calibrated geometry for the fixed camera (all videos share it)
weights/             <- yolo11x.pt (shipped) + download.sh (fallback)
tests/               <- 732 tests, run with `python -m pytest tests -q`
```

## Quickstart

```bash
pip install -r requirements-torch.txt -r requirements.txt   # or: docker build -t team .
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
# ONLINE, before the offline evaluation - the sanctioned Docker step:
docker build -t wiut-cv-track .
# then, on the grading machine (no internet needed from here on):
docker run --rm --gpus all -v /data/test:/data/test wiut-cv-track
```

```bash
# ...or, without Docker, on an image that already has a CUDA torch:
pip install --no-deps -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --gt ground_truth.json
```

Time budget per video: **3 × its duration** for Part A + Part B together; a video
over budget or a crash scores as empty. Events with a bad label, bad times, or a
same-class overlap are dropped by the harness and listed in its log.

### Weights

`weights/yolo11x.pt` **is included in this package** (114,636,239 bytes =
109.3 MiB, well inside the 5 GB cap), so the offline run needs no download
step — **but read the Git LFS note below first, because a plain clone without
git-lfs silently produces a broken checkout.**

#### Git LFS — the one thing that will bite you

GitHub refuses any file over 100 MiB (104,857,600 bytes) in a normal blob, and
this weight is 109.3 MiB. It is therefore stored in **Git LFS**. The file
content in the repository is a ~130-byte *pointer*, not the model:

```
version https://git-lfs.github.com/spec/v1
oid sha256:7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24
size 114636239
```

A pointer is not a model. `torch.load` fails on it, and the consequence is the
worst kind: **zero detections, empty event lists, `Score_A` = `Score_B` = 0, and
`evaluate.py --validate-only` still says `VALID`**, because an empty event list
is legal. Nothing crashes and nothing warns you.

Two independent ways out, either is enough:

```bash
git lfs install          # BEFORE cloning, or: git lfs pull after cloning
git clone <repo>
ls -l weights/yolo11x.pt   # must be ~110 MB, not ~130 bytes
```

```bash
bash weights/download.sh   # detects a pointer, re-downloads, verifies SHA-256
```

`weights/download.sh` treats a pointer exactly like a missing file, re-fetches
from the Ultralytics release URL and verifies
`sha256 7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24` — the
same digest as the shipped copy — so a corrupted download cannot pass silently
either. Run it once, with internet, before the offline evaluation. The
`Dockerfile` builds a normal (non-LFS) checkout, so it never depends on any of
this.

Do not rely on either path at scoring time: grading has no internet, and a
missing weight file is a *silent* total loss — every frame returns zero
detections, every video is submitted with an empty event list, `Score_A` and
`Score_B` are both 0, and `evaluate.py --validate-only` still reports `VALID`
because an empty event list is legal. `src/detection/detector.py` therefore
prints a loud one-time banner on stderr and records `Detector.load_error` if the
model cannot be loaded, and `Detector.preflight()` reports the problem directly.

## Verified environment

Pinned to the versions this solution was developed and verified against. Every
pin below was re-read from the live environment with
`importlib.metadata.version(...)` and matches exactly, so the pins describe
what actually ran rather than what was intended.

| Package | Pin | Where | Note |
|---|---|---|---|
| `torch` | `2.14.0+cu126` | `requirements-torch.txt` | CUDA 12.6 build, official PyTorch index |
| `torchvision` | `0.29.0+cu126` | `requirements-torch.txt` | matching build |
| `numpy` | `2.5.3` | `requirements.txt` | |
| `opencv-python` | `5.0.0.93` | `requirements.txt` | see the note below |
| `ultralytics` | `8.4.160` | `requirements.txt` | provides YOLO11x + ByteTrack |

Python 3.12, on an RTX 3070 Ti (8 GB) — the only GPU this was actually run on.
The budget arithmetic in `src/config/budget.py` is written against a slower
T4-class 16 GB card, so the defaults are conservative there; that is an
assumption about the grading hardware, not a measurement of it.

Verified on the environment above: **731 passed, 1 skipped** (the skip needs a
real sample video, which is not part of the submission), and
`format: 1 video(s), 14 event(s), 0 error(s), 0 warning(s) -> VALID`.

**What is *not* verified:** a from-scratch `pip install` of the two requirements
files into an empty environment, and the `Dockerfile`. Both need the 2.5 GB CUDA
torch download, and on this machine the default pip target path exceeds the
Windows `MAX_PATH` limit (which is why torch is installed to `C:\pylibs` via
`--target`). The split and the image are provided for the pre-evaluation online
step; treat them as untested plumbing rather than as measured results. The one
dependency fact that *is* mechanical: `weights/download.sh` verifies the weights
against a pinned SHA-256, so a failed download cannot pass silently.


### Why torch is in a separate file

The evaluation machine has no internet, but `pip install -r requirements.txt` is
still one of the two commands the organizers run. A pinned torch wheel is a
2.5 GB download, so the single most likely way to lose the whole run is pip
deciding it needs a torch it cannot fetch:

- if the grading image already ships a working CUDA torch and our pin differs by
  one patch number, `pip install` fails offline instead of using what is there;
- if it ships none, nothing can install 2.5 GB offline anyway.

So the torch pin lives in `requirements-torch.txt` (used by the Dockerfile, at
build time, where the network is available) and `requirements.txt` holds only the
packages that are small and already cached in any sane image. On a grading image
that already has torch:

```bash
pip install --no-deps -r requirements.txt      # seconds, no network
```

Three details that are easy to get wrong:

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
  `libgl1` and `libglib2.0-0` (the Dockerfile installs them).
- **The Dockerfile is built while the network is still available.** It is the
  step the brief sanctions for exactly this problem, and after `docker build`
  the run needs no network at all: the weight, the scene config and the code are
  all baked in.

Determinism: the inference path contains no RNG at all — no `random`, no
`np.random`, no `torch.manual_seed` call — so two runs on the same machine
produce byte-identical output by construction rather than by seeding. There is
no wall-clock or filename-order dependence either.

## Our approach

**Part A** is one YOLO11x + ByteTrack pass per video, sampled every 3rd frame
(`TCV_STRIDE=3`, or coarser if the budget guard below says so).
`src/pipeline/pipeline.py` decodes each video exactly once and feeds one
`EventManager`; every class provider is registered in a single
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
than a delayed flat-topped one.

### The 3x budget, measured

Measured end to end through the organizers' own `run_submission.py` on the 4K
sample `C3905` (127.6 s, 29.97 fps, 3840x2160) at `imgsz=800` on an RTX 3070 Ti:

| | wall time | per observation |
|---|---|---|
| Part A (1275 observations) | 143.2 s | 0.112 s |
| Part B (1276 observations) | 147.7 s | 0.116 s |
| **total** | **290.8 s = 2.28x** real time | 0.114 s |

Part B costs as much as Part A because both halves run their own detector at
roughly the same rate, and a 4K decode (~50 ms/frame) is paid on top either way.
2.28x against a 3x cap is 24% headroom — enough on this machine, not something to
leave to a different card. So the budget guard is **on by default**:
`src/config/budget.py` declares a per-device cost per observation
(`SEC_PER_OBS`, `0.20` s for CUDA — deliberately ~1.7x the figure measured on a
3070 Ti, because the graders' card is a different, slower class of GPU) and
returns the coarsest stride that still fits 3x the duration. On the organizers'
four samples at 29.97 fps it plans stride 4 for Part A and 7.5 Hz for Part B
instead of stride 3 / 10 Hz, roughly a third less work in each half.

Two properties keep that safe. The guard can only ever **increase** a stride or
**lower** a rate, so it cannot silently upgrade quality; and it is a **pure
function** of `(n_frames, fps, declared cost)` — it never reads a clock, so it
cannot make the output depend on machine load. Set `TCV_BUDGET_GUARD=0` to
freeze the sampling rate, or `TCV_SEC_PER_OBS` to re-declare the cost for a
different device.


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

## Website

The team site (team, approach, EDA, results, event catalogue, report) lives in
this repository's parent, under `website/`, with the FastAPI inference backend
under `backend/`:

```
website/   Next.js 16 + React 19 + Tailwind 4   -> the public site
backend/   FastAPI /api/analyze                 -> the live demo's real inference
```

`backend/` calls **this** package: `solution.detect_events` for Part A and
`solution.RiskEstimator` for Part B, resolved through the `SOLUTION_ROOT`
environment variable. It is not a re-implementation and not a replay of
`predictions_samples.json`, so the demo exercises the submitted code. Set
`SOLUTION_ROOT` to this directory when the backend is deployed elsewhere.

## Datasets, licences and attribution

**We trained no models.** Every parameter in this solution is either a
hand-set threshold or comes from an off-the-shelf pretrained checkpoint, so
there is no training set to license and no dataset-derived artifact in the
repository. The development set consisted of the four 4K sample clips
(`C3896`, `C3897`, `C3902`, `C3905`) supplied with the task, plus small
self-generated synthetic fixtures for the unit tests.

| Component | Origin | Licence |
| --- | --- | --- |
| `weights/yolo11x.pt` (YOLO11x) | Ultralytics, pretrained | AGPL-3.0 |
| `ultralytics` (detector + tracker) | Ultralytics | AGPL-3.0 |
| `torch` / `torchvision` | PyTorch | BSD-3-Clause |
| `numpy` | NumPy | BSD-3-Clause |
| `opencv-python` | OpenCV | Apache-2.0 |
| Sample clips `C389*`/`C390*` | task organisers | provided with the task |
| Everything under `src/`, `solution.py`, `website/`, `backend/` | this team | — |

Ultralytics is distributed under **AGPL-3.0**, which is why the weight and the
library are shipped rather than vendored, and why the pretrained COCO weights
are redistributed unmodified with their licence intact. We are not affiliated
with or endorsed by Ultralytics; the "YOLO" name and logo are theirs.

COCO appears only indirectly, as the training data behind the pretrained
Ultralytics checkpoint; its annotations are (c) Microsoft, licensed under
CC BY 4.0. No COCO data is redistributed here.

The organizers' `run_submission.py` and `evaluate.py` are included unmodified
and belong to the task organizers.
