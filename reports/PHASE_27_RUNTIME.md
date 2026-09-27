# PHASE 27 — local runtime characterisation (why the baseline cannot be measured here)

All numbers below were measured on 2026-09-26 on the team laptop, read-only, on
`C3905.MP4` (3840x2160, 29.97 fps). Nothing in this file is a guess: every value is
a printed measurement from a throwaway script in `%TEMP%\opencode\p27\`.

The purpose is to record **why** the PHASE 27 submission run was abandoned rather
than reported, because the reason is a property of the machine, not of the code.

## 1. The GPU was already doing the inference

```
torch 2.14.0+cu126  cuda=True  device=cuda:0
cv2 5.0.0
detector device=cuda:0  imgsz=800
```

`nvidia-smi` confirms the GPU is the active device (RTX 3070 Ti Laptop, 8192 MiB).
The Phase 27 submission was launched with `TCV_DEVICE=cuda:0`. There is no
CPU-fallback path that was silently taken.

## 2. The heat is 4K decode, not inference

| stage | wall | CPU time | cores | per unit |
|---|---|---|---|---|
| YOLO11x inference @800, stride 3 | 4.50 s | 8.16 s | **1.81** | 225 ms/obs |
| 4K H.264 decode to a numpy array | 1.42 s | 11.42 s | **8.06** | 23.6 ms/frame wall, 190 ms/frame CPU |

Inference costs about a fifth of what decode costs. `run_submission.py` reads
every frame (Part A) and then re-opens the file and reads it again (Part B), so
the expensive half of the pipeline is paid twice per video.

## 3. Decode cannot be throttled with the available knobs

| attempt | result | cores |
|---|---|---|
| `cv2.setNumThreads(1)` (`TCV_CV_THREADS=1`) | no effect | 5.41 |
| `OPENCV_FFMPEG_CAPTURE_OPTIONS=threads;1` | no effect | 5.75 |
| hardware decode, `CAP_PROP_HW_ACCELERATION=VIDEO_ACCELERATION_D3D11` | works, `hw_prop=2`, sample frame means identical (41.64 vs 41.64) | 6.32 |
| hardware decode, `VIDEO_ACCELERATION_ANY` | works, `hw_prop=2` | 6.57 |

D3D11VA hardware decode is the only thing that helped, and it bought **19% less
CPU** (11.42 s -> 9.30 s for 60 frames), not an order of magnitude. The residual
is OpenCV's D3D11 surface -> numpy BGR conversion, which is single-threaded CPU
work by construction.

`TCV_OMP_THREADS=1` does not reach torch either: `torch.get_num_threads()`
reported `8` after `cap_cpu_threads` ran, because torch reads `OMP_NUM_THREADS`
at import time and the torch import happens first. That is a measurement note,
not a defect worth changing in a measurement-only phase.

## 4. Consequence: the official command cannot meet the 3x budget here

Per second of video at 29.97 fps, stride 3:

- decode: `29.97 x 30.8 ms` = 0.92 s
- inference: `9.99 x 225 ms` = 2.25 s
- Part A subtotal: ~3.2 s of wall per second of video = **3.2x realtime**
- Part B re-reads and re-infers: another ~3.2 s -> **~6.4x realtime overall**

The harness budget is `time_factor 3.0`, i.e. 3.0x realtime. At 6.4x every video
is blanked by `run_submission.py`, which is exactly what the pre-existing
`predictions_samples.json` shows (all entries empty, over-budget).

Extrapolated to the 1104 s dev set: **~118 min of wall time** for output that the
harness would discard. On a laptop whose CPU package was observed at 99 C, that
is a bad trade, so the run was stopped rather than completed.

## 5. Two caveats on these numbers

- Inference was measured twice. A cold run that interleaved decode with tracking
  reported 638 ms/obs; a warmed run reported 225 ms/obs. The gap is CUDA and
  ByteTrack warmup. **225 ms is the floor, 638 ms the first-pass cost**; the 6.4x
  figure uses the floor, so the real overrun is worse, not better.
- A bare `model(1x3x800x800)` forward on a synthetic tensor was attempted to
  separate preprocessing from the GPU, but the detector builds its YOLO model
  lazily so the handle was still `None` at the point of measurement and the
  number was not obtained. It is not quoted here. This is the one open thread
  worth closing if anyone wants to know whether 225 ms is GPU-bound.

## 6. What this means for PHASE 27

- The dev-set **annotation workflow is built and ready** (see `ANNOTATION_GUIDE.md`);
  it needs a human with eyes on the footage, not compute.
- Score_A, Score_B and the timing baseline are **unavailable on this machine** and
  are reported as insufficient, not as zeros and not as a guess.
- The real baseline belongs on the grading hardware (a T4, per the project notes),
  where AGENTS.md records ~45 ms/frame, i.e. comfortably inside 3x.
- Nothing in `evaluate.py`, `run_submission.py`, `solution.py`, the detectors, the
  event logic, the thresholds or Part B was modified to obtain any of this.
