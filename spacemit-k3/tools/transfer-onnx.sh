#!/usr/bin/env bash
# Push an exported ONNX bundle (graph + sidecar .data + configs) to the device.
#
#   ./transfer-onnx.sh [local_dir] [remote_dir]
set -euo pipefail

LOCAL="${1:-export/build/sd-turbo-onnx}"
REMOTE="${2:-/home/bianbu/image-generation/models/onnx/sd-turbo}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$LOCAL" ]; then
  echo "no such dir: $LOCAL" >&2
  exit 1
fi

cd "$LOCAL"
# recurse over whatever the exporter produced, keeping the flat layout the runtime expects
FILES=$(find . -type f \( -name '*.onnx' -o -name '*.onnx.data' -o -name '*.json' \
        -o -name '*.npy' -o -name '*.txt' \) | sed 's|^\./||' | sort)

TOTAL=$(echo "$FILES" | wc -l)
i=0
for f in $FILES; do
  i=$((i + 1))
  echo "[$i/$TOTAL] $f"
  MSYS_NO_PATHCONV=1 python "$HERE/rcmd.py" -u "$LOCAL/$f" "$REMOTE/$f"
done

echo
echo "done. remote listing:"
MSYS_NO_PATHCONV=1 python "$HERE/rcmd.py" "ls -la $REMOTE"
