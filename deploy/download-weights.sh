#!/usr/bin/env bash
# Download the three Z-Image-Turbo weights and verify them by SHA-256.
#
#   ./download-weights.sh [target_dir]        # default: models/zimage
#
# Everything here goes through ModelScope. Hugging Face is NOT reachable from
# the networks this was built on (huggingface.co and hf-mirror.com both fail to
# complete a TLS handshake), and ModelScope carries all three files with
# byte-identical hashes, so it is the reliable path.
#
# Re-running is cheap: files that already match their hash are skipped.
set -euo pipefail

DEST="${1:-models/zimage}"
BASE_MODEL="https://www.modelscope.cn/models/jayn7/Z-Image-Turbo-GGUF/resolve/master"
BASE_QWEN="https://www.modelscope.cn/models/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/master"
BASE_VAE="https://www.modelscope.cn/models/black-forest-labs/FLUX.1-schnell/resolve/master"

# name|url|sha256|size
FILES=(
  "z_image_turbo-Q4_K_M.gguf|$BASE_MODEL/z_image_turbo-Q4_K_M.gguf|745ec270db042409fde084d6b5cfccabf214a7fe5a494edf994a391125656afd|4981532736"
  "Qwen3-4B-Instruct-2507-Q4_K_M.gguf|$BASE_QWEN/Qwen3-4B-Instruct-2507-Q4_K_M.gguf|3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597|2497281120"
  "ae.safetensors|$BASE_VAE/ae.safetensors|afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38|335304388"
)

mkdir -p "$DEST"

for entry in "${FILES[@]}"; do
  IFS='|' read -r name url want size <<<"$entry"
  out="$DEST/$name"

  if [ -f "$out" ] && [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$want" ]; then
    echo "[skip] $name already present and correct"
    continue
  fi

  echo "[get ] $name  ($((size / 1048576)) MiB)"
  # -C - resumes a partial download instead of starting over
  curl -fL --retry 5 --retry-delay 3 -C - -o "$out" "$url"

  got="$(sha256sum "$out" | cut -d' ' -f1)"
  if [ "$got" != "$want" ]; then
    echo "[FAIL] $name hash mismatch" >&2
    echo "       want $want" >&2
    echo "       got  $got" >&2
    exit 1
  fi
  echo "[ ok ] $name"
done

echo
echo "all three weights verified under $DEST"
du -sh "$DEST"
