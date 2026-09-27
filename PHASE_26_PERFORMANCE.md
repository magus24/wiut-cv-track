# PHASE 26 — Performance optimization

Goal: make the graded run fit the harness time budget (**3× the video duration
for Part A + Part B together**) without changing a single event, score, segment
boundary or public interface. Determinism preserved. No threshold touched, no
detector redesigned, no test removed or weakened.

Canonical tree: `Project/Project/` (the live one; `wiut_cv_scripts/` is the
stale pre-PHASE-24 copy).

---

## 1. Verdict

| | pre-PHASE 26 | optimised | delta |
|---|---|---|---|
| Part A, whole loop, 300 source frames (best of 3, interleaved) | 19.530 s | 13.558 s | **−30.6%** |
| `EventManager.step`, 100 observations, real C3905 detections | 19.68 ms/obs | 9.38 ms/obs | **−52.4%** |
| 4K decode | 20.2 ms/frame | 11.5 ms/frame | **−43%** |
| Part B risk series | — | — | bit-identical, time within noise |

**Events are identical in every A/B**, on synthetic streams and on the real
C3905 sample, and `detect_events` is byte-reproducible across two runs.

Files changed (7): `src/pipeline/video.py`, `src/detection/detector.py`,
`src/tracking/interaction.py`, `src/events/manager.py`, `src/events/accident.py`,
`src/events/near_miss.py`, `src/risk/risk.py`.
Files added (2): `tests/test_video_reader_stride.py`, `tests/test_pairwise_sharing.py`.
`run_submission.py`, `evaluate.py`, `solution.py` — untouched.

---

## 2. What was measured before touching anything

Component split of Part A on the 300-frame C3905 slice (stride 3, imgsz 800):

| stage | share |
|---|---|
| 4K decode (`cap.read`) | 39.5% |
| YOLO forward | 26.4% |
| ByteTrack | 4.1% |
| NMS + postprocess | 2.2% |
| ultralytics preprocess (letterbox) | 1.3% |
| `Detector.track` wrapper (our code) | 3.1% |
| `EventManager.step` | 13.9% |
| pairwise sweep (inside the above) | 2.4% |
| pipeline resize | 1.3% |

That set the order of work: the two biggest blocks (4K decode, event layer) are
OURS to fix; the YOLO forward is not.

---

## 3. Change 1 — `VideoReader`: `grab()`/`retrieve()` instead of `read()`

`read()` does two things: decode the H.264 frame **and** convert it to BGR.
With `stride=3` two frames out of three are thrown away immediately, but their
BGR conversion was being paid for anyway.

Measured on the 4K sample, per source frame:

| | cost |
|---|---|
| `grab()` (decode only) | 3.7–4.4 ms |
| `retrieve()` (BGR convert + copy) | 22 ms |
| `read()` | 20–21 ms |
| **saving from `grab`/`retrieve` at stride 3** | **8.4–9.0 ms/frame** |

Identical across five windows of the file (2%, 25%, 50%, 75%, 95%), so this is
not a property of the clip's start. The pixels are **bit-identical** to
`read()` — asserted against the real file in
`tests/test_video_reader_stride.py::test_real_file_grab_matches_read_bitwise`.

The one behavioural subtlety is buffer ownership: the yielded array is a view
into OpenCV's decoder ring and is valid only until the next iteration. The
module docstring now states this, and `run_pipeline`'s first statement on the
frame is `cv2.resize`, which allocates a new array.

`max_frames` keeps its exact old meaning (stop the loop, don't decode past the
window) — the first draft of this change accidentally kept decoding to EOF and
`test_max_frames_window` caught it.

## 4. Change 2 — one pairwise sweep for the whole detector pool

`accident` and `near_miss` each built the *same* `(id, label, position, state)`
item list from the same tracks and each ran the full O(n²) sweep on it. The
engine is stateless and `compute` is pure, so the two lists were identical by
construction — and the manager then threw both away, using only the boolean
evidence flag that comes back.

At ~37 active tracks that is ~680 `PairInteraction` objects per detector per
frame, allocated, read once, discarded.

`EventManager._shared_interactions()` now computes the list once and injects it
(`interactions=`), and passes `record=False`, which skips the per-pair
diagnostic dictionaries (14 keys each, plus a copy of the pair's life record)
that the manager also discarded.

Two safety properties make this output-preserving rather than merely plausible:

* the class gate moved from *after* `compute` to *before* it. `_relevant` is a
  function of the two class labels alone, and a track's label is immutable, so
  a pair it rejects could never have produced evidence. `pairs_filtered`
  returns the surviving objects in the same order with identical field values
  (`test_pairs_filtered_is_pairs_restricted_to_the_gate`).
* the shared gate is the **OR** of the sharers' own `_relevant` predicates, so
  it can only ever be more permissive than either detector. Any disagreement
  about candidate labels, or a private pairwise engine, disables sharing rather
  than risk dropping a pair.

`_impact_now()` was also being evaluated twice per pair per frame (once for the
report, once for the state machine) on a pure function of `(inter, st_a, st_b)`;
it is now evaluated once and the result reused. The report's state snapshot is
still taken *after* the state machine runs, as before.

Effect on the real sample: `EventManager.step` 19.68 → 9.38 ms/obs.

## 5. Change 3 — `Detector.track`: one transfer per tensor

`float(boxes.conf[i])` inside the detection loop was a **synchronising
device→host scalar read per detection** (~27 per frame on this clip). The four
tensors are now fetched once each. Same tensors, same dtype, same order, same
dicts; the conversion A/B measured −2.0 ms per call with identical output.

## 6. Change 4 — Part B: gate before building

`_observe` built every pair and then skipped the irrelevant ones, with the same
`_pair_relevant` predicate. Now it uses `pairs_filtered`. The risk series is
**bit-identical** (mean 0.2770 on the 150-frame slice, all 50 values equal).

Honest note: on C3905 the measured Part B effect is **within noise** (−3.7%,
i.e. the run-to-run spread), because Part B is dominated by the detector at
~10 Hz and the footage is mostly vehicles, so few pairs get filtered. It is
kept because it is the same already-required primitive, strictly less work, and
provably identical — a pedestrian-heavy scene is where it pays.

---

## 7. What was measured and REJECTED

| idea | measurement | verdict |
|---|---|---|
| FP16 / `half=True` | 68.5 ms vs 69.6 ms fp32; boxes shift up to 0.63 px, conf 0.00116 | rejected — no gain, changes numbers |
| `classes=[...]` pre-NMS filter | +0.10 ms; output verified identical | rejected — no gain |
| `cudnn.benchmark=True` | 66.7 vs 69.3 ms, drift check ±0.5 ms | rejected — inside the noise, global side effect |
| `channels_last` / pure-forward variants | script crashed inside ultralytics' AutoBackend (module-internal channel split) | not pursued |
| batching frames (2/4) | 56.5 → 38.2 → 36.0 ms/frame | rejected — needs a restructure, can change ByteTrack ids, unusable for causal Part B |
| `imgsz` 640/800/960 | 62.3 / 61.5 / 55.8 ms, non-monotonic | rejected — would change every detection |
| OpenCV D3D11 HW decode | never activates | rejected |
| **cv2 thread cap (`TCV_CV_THREADS`)** | 1/2/4/6/8/16 threads → 20.8/21.3/21.4/21.2/21.2/21.3 ms | **not a factor** — hypothesis tested and dropped |
| parallelising the event layer | — | rejected on the operator's instruction (laptop overheats) |

## 8. The measurement trap on this laptop

Long runs on this machine are **thermologically distorted**, and it is worth
recording because it invalidates the obvious approach:

| run | wall | realtime factor |
|---|---|---|
| full C3905, 3825 frames, one process | 802 s | 6.29 |
| first 1200 frames, fresh process | 64 s | 1.61 |
| decode-only, 1200 frames | — | 20 ms/frame, flat |

The full-file run's decode cost 100 ms/frame against 20 ms/frame in a short
process, and a decode-only run does not degrade at all — so the long figure is
sustained-load throttling, not algorithmic cost. Consequences adopted for the
rest of this phase:

* every decision comes from **paired/interleaved A/B in one process**, never
  from comparing against a number measured earlier;
* all A/B scripts alternate the arm order between rounds;
* long runs are not used as evidence.

The 802 s figure should therefore not be read as "the pipeline needs 6.3× the
budget on the grading machine" — but it *is* a real warning: on a machine that
throttles, a 3× budget is a thin margin, and every reduction in work
transfers 1:1 to the grader.

## 9. Tests

* `tests/test_video_reader_stride.py` — 10 tests: sampling window, `t_sec`,
  `max_frames`, grab/retrieve counts, retrieve failure, unopened capture, and
  a bit-exact comparison against the real 4K file.
* `tests/test_pairwise_sharing.py` — 18 tests: the A/B described below, the
  Part B channel reconstruction, the report-switch equivalence, and the
  `pairs_filtered` contract.
* Full suite: **687 passed, 2 failed**.

The 2 failures are **pre-existing PHASE 25 defects** in
`tests/test_budget_guard.py`, in `src/config/budget.py`, which this phase does
not touch:

1. `test_the_chosen_stride_really_does_fit_its_share_of_the_budget` — at
   (357 frames, 9.99 fps, 0.6 s/obs) the chosen stride needs 54.0 s against a
   53.6 s allowance. Arithmetic in the guard, not in anything changed here.
2. `test_part_b_target_hz_survives_degenerate_input` — asserts
   `target_hz_for_budget(10.0, nan, ...) == nan`, and `nan == nan` is `False`
   in Python. The assertion is unsatisfiable as written.

Both were left alone deliberately: fixing them means editing PHASE 25's
arithmetic or its test, which is outside this phase's mandate and must not be
papered over by weakening the test. They need a decision from the team.

One further flake, unrelated to this phase: `test_risk_estimator.py::
test_is_blank_agrees_with_any_but_is_far_cheaper` asserts `_is_blank` on a 4K
frame costs < 2.0 ms; it measured 2.12 ms once under load and passed in the
full run. It is a wall-clock assertion on a loaded machine.

### Non-vacuity: the A/B tests have teeth

An equality test over two *empty* event lists proves nothing, so the harness
A/B forces `SHARES_PAIRWISE` off on `AccidentDetector`/`NearMissDetector`,
which makes the manager call `update()` with no keyword arguments — the exact
pre-PHASE-26 signature and the unfiltered `pairs()` sweep. Both arms are then
compared over six synthetic streams, two of which are asserted to actually
emit (`accident`, `near_miss`) and one of which puts the firing pair *second*
in id order. Mutation check:

| mutation of the optimisation | caught |
|---|---|
| gate inverted (keep only pairs no sharer wants) | **yes** — 4 of 6 streams |
| only the first pair survives | **yes** — the `second_pair` stream |
| gate always true (more work, same output) | no — correctly, it is equivalent |

## 10. Real-footage equivalence

`EventManager` over 100 real C3905 observations (2724 detections), optimised
vs pre-PHASE-26 path:

```
events (optimised)   : 5
events (pre-PHASE 26): 5
IDENTICAL: True
```

and through the public interface, two consecutive `detect_events` runs on the
same clip return identical output:

```json
[[0.0, 5.906, "congestion"], [0.0, 9.91, "failure_to_yield"],
 [0.0, 9.91, "jaywalking"], [0.1, 9.91, "wrong_way"], [0.2, 9.209, "near_miss"]]
```

## 11. Where the time goes now, and what is left

After these changes, on a 100-observation slice, the remaining Part A budget is
roughly: YOLO forward ~45 ms/obs (irreducible without changing the model or
imgsz, both off limits), ByteTrack ~7 ms, the letterbox+resize of a 4K frame
~16 ms, and the event layer ~9 ms.

Not attempted, and why:

* **the 4K→800 resize** (16 ms/obs) — any faster path changes the pixels
  `INTER_AREA`/`NEAREST`, hence every detection.
* **the letterbox** — inside ultralytics.
* **ByteTrack / NMS** — inside ultralytics.
* **frame batching** — would change track ids and cannot serve causal Part B.
* **the model itself** — `imgsz`, architecture and weights are fixed by the
  plan and by the 5 GB packaging limit.

The honest summary: on this footage the pipeline was not model-bound, it was
I/O- and allocation-bound, and roughly a third of the wall clock has been taken
out without touching a single decision the detectors make.

## 12. Reproducing

Scripts live outside the package, in
`%LOCALAPPDATA%\Temp\opencode\p26\`:

| script | what it answers |
|---|---|
| `bench.py` | component benchmark (`--part a/b`, `--frames N`) |
| `decode_windows.py`, `decode_long.py`, `threads_decode.py` | where 4K decode time goes |
| `infer_profile.py`, `loop.py`, `loop_seg.py` | per-stage breakdown, real frames |
| `cfg_ab.py`, `fp16_ab.py`, `gpu_exp.py` | FP16 / `classes=` / cudnn / channels_last |
| `track_ab.py` | detector result-conversion A/B |
| `real_equivalence.py` | event layer: optimised vs pre-PHASE-26, real detections |
| `e2e_ab.py` | whole Part A loop, interleaved, both arms in one process |
| `partb_ab.py` | Part B risk series identity + timing |
| `mutation_check.py` | proves the A/B tests fail when the optimisation is broken |
