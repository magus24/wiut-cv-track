# Offline evaluation image for the CV track.
#
# WHY THIS EXISTS: the organizers' brief states the evaluation machine has NO
# internet access, but the two commands they run are still
#     pip install -r requirements.txt
#     python run_submission.py --videos <dir> --out predictions.json --team <t>
# `pip install` of a pinned torch wheel is a 2.5 GB download and fails offline.
# The brief also sanctions a Docker step, so this image is the sanctioned way
# to make that install happen BEFORE the offline run:
#
#     docker build -t wiut-cv-track .                                  # network OK here
#     docker run --rm --gpus all -v /path/to/test:/data/test wiut-cv-track
#
# The wheels come from the PyTorch CUDA 12.6 index (see
# requirements-torch.txt). Everything else - the weights, the scene config and
# the code - is copied in, so the run itself needs no network at all.
FROM python:3.12-slim-bookworm

# libgl1/libglib2.0-0: required by opencv-python. ffmpeg: probe/rotation
# metadata, which ultralytics touches on some code paths.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-torch.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-torch.txt \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# The weight is in Git LFS (109.3 MiB > GitHub's 100 MiB blob limit), so a
# checkout made WITHOUT git-lfs holds a ~130-byte pointer in weights/yolo11x.pt.
# COPY would faithfully copy that pointer into the image, the detector would
# then fail to load, and every frame would come back empty - a valid-looking
# submission that scores zero. This step is at build time, where the network is
# available, so it can repair the damage instead of shipping it.
#
#   1. a real weight passes untouched;
#   2. a pointer is re-downloaded and checksum-verified by download.sh;
#   3. anything else that is not a real weight fails the build loudly, because
#      a silently broken image is strictly worse than no image.
RUN set -eu; \
    w=weights/yolo11x.pt; \
    if [ -f "$w" ] && [ "$(wc -c < "$w")" -lt 1048576 ] \
       && head -c 40 "$w" | grep -q "git-lfs.github.com/spec"; then \
      echo "GIT LFS POINTER detected in $w - fetching the real weights"; \
      bash weights/download.sh; \
    fi; \
    sz=$(wc -c < "$w"); \
    if [ "$sz" -ne 114636239 ]; then \
      echo "FATAL: $w is $sz bytes, expected 114636239 (LFS pointer? partial clone? real download needed)" >&2; \
      exit 1; \
    fi; \
    echo "weights/yolo11x.pt verified: $sz bytes"

# Inference on the GPU. The 3070 Ti used for development is 8 GB, so nothing
# here assumes more memory than that; imgsz stays at 800 for the same reason.
# TCV_STRIDE is left at 3: src/config/budget.py plans stride 4 for these
# lengths anyway, and it owns that decision so local and graded runs agree.
ENV TCV_DEVICE=cuda:0 \
    TCV_IMGSZ=800 \
    TCV_STRIDE=3 \
    PYTHONUNBUFFERED=1

# The organizers' own runner, unchanged. Mount the test videos at /data/test.
CMD ["python", "run_submission.py", "--videos", "/data/test", \
     "--out", "/app/predictions.json", "--team", "wiut-cv-track"]
