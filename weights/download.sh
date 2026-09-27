#!/usr/bin/env bash
# Re-fetch the detector weights.
#
# The repository stores weights/yolo11x.pt in Git LFS: the file is 114,636,239
# bytes and GitHub refuses any file over 100 MiB in a normal blob. A clone on a
# machine WITHOUT git-lfs therefore leaves a ~130-byte POINTER FILE where the
# weights should be, and a pointer is not a model - `torch.load` fails, the
# detector prints its banner, and every frame comes back empty, which scores
# zero on both parts. So this script treats a pointer exactly like a missing
# file and re-downloads the real thing.
#
# Run it once, with internet, before the offline evaluation:
#     bash weights/download.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .

url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt"
want="7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24"
out="yolo11x.pt"

is_lfs_pointer() {
  # LFS pointers are tiny and start with this header.
  [ ! -s "$1" ] && return 0
  [ "$(wc -c < "$1")" -lt 1048576 ] || return 1
  head -c 40 "$1" | grep -q "git-lfs.github.com/spec" || return 1
  return 0
}

have_real_weights() {
  [ -s "$1" ] || return 1
  is_lfs_pointer "$1" && return 1
  return 0
}

if have_real_weights "$out"; then
  got=$(sha256sum "$out" | cut -d' ' -f1)
  if [ "$got" = "$want" ]; then
    echo "yolo11x.pt already present and verified"
    exit 0
  fi
  echo "yolo11x.pt has the wrong checksum ($got), re-fetching"
fi

if is_lfs_pointer "$out"; then
  echo "$out is a Git LFS pointer, not the model - fetching the real weights"
fi

echo "downloading $url"
rm -f "$out"
curl -fL -o "$out" "$url"

got=$(sha256sum "$out" | cut -d' ' -f1)
if [ "$got" != "$want" ]; then
  echo "FATAL: checksum mismatch after download" >&2
  echo "  expected $want" >&2
  echo "  got      $got" >&2
  exit 1
fi
echo "yolo11x.pt downloaded and verified"
