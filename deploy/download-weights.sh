#!/usr/bin/env bash
# Download the three Z-Image-Turbo weights and verify them by SHA-256.
#
#   ./download-weights.sh [target_dir]        # default: models/zimage
#
# Everything here goes through ModelScope, for two separate reasons:
#
#   * huggingface.co is unreachable from the networks this was built on. On the
#     Jetson it fails outright (curl gets no connection in 0.02 s); on the PC the
#     name resolves but the TLS handshake never completes.
#   * hf-mirror.com IS reachable from the PC (5/5 requests returned 200) but is
#     both slower (2.7 MiB/s vs 6.0 MiB/s measured on the same file) and
#     incomplete: FLUX.1-schnell is a gated repo there, so ae.safetensors comes
#     back as a 183-byte error page instead of the real 335 MB file.
#
# ModelScope serves all three with byte-identical hashes, so it is the one
# channel that covers the whole set. A mirror fallback for the first two files
# is documented in MANIFEST.md.
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

check() { sha256sum "$1" | cut -d' ' -f1; }

for entry in "${FILES[@]}"; do
  IFS='|' read -r name url want size <<<"$entry"
  out="$DEST/$name"

  if [ -f "$out" ] && [ "$(check "$out")" = "$want" ]; then
    echo "[skip] $name already present and correct"
    continue
  fi

  echo "[get ] $name  ($((size / 1048576)) MiB)"
  # -C - resumes a partial download instead of starting over
  curl -fL --retry 5 --retry-delay 3 -C - -o "$out" "$url"

  if [ "$(check "$out")" != "$want" ]; then
    # Resume cannot repair a file that is already the right length but wrong
    # content: curl would see nothing left to fetch and return the same bad
    # file forever. Start over once from an empty file before giving up.
    echo "[warn] $name did not match; discarding and re-downloading from scratch" >&2
    rm -f "$out"
    curl -fL --retry 5 --retry-delay 3 -o "$out" "$url"
  fi

  got="$(check "$out")"
  if [ "$got" != "$want" ]; then
    echo "[FAIL] $name hash mismatch after a clean re-download" >&2
    echo "       want $want" >&2
    echo "       got  $got" >&2
    echo "       The upstream file may have been re-uploaded; check the" >&2
    echo "       release page before trusting this copy." >&2
    exit 1
  fi
  echo "[ ok ] $name"
done

echo
echo "all ${#FILES[@]} weights verified under $DEST"
du -sh "$DEST"
