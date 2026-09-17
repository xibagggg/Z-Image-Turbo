#!/bin/bash
# Convenience entry for the Z-Image-Turbo deployment on Jetson AGX Orin.
set -euo pipefail
cd "$(dirname "$0")"

MODEL=models/zimage/z_image_turbo-Q4_K_M.gguf
VAE=models/zimage/ae.safetensors
LLM=models/zimage/Qwen3-4B-Instruct-2507-Q4_K_M.gguf

STEPS=${STEPS:-8}
WIDTH=${WIDTH:-1024}
HEIGHT=${HEIGHT:-1024}
SEED=${SEED:-42}

common_args=(
  --diffusion-model "$MODEL"
  --vae "$VAE"
  --llm "$LLM"
  --cfg-scale 1.0
  --diffusion-fa
)

case "${1:-generate}" in
  generate)
    prompt=${2:-a red fox sitting in a snowy forest, soft morning light, highly detailed, cinematic}
    out=${3:-output/$(date +%Y%m%d_%H%M%S).png}
    mkdir -p "$(dirname "$out")"
    exec ./bin/sd "${common_args[@]}" \
      -p "$prompt" --steps "$STEPS" -W "$WIDTH" -H "$HEIGHT" --seed "$SEED" -o "$out"
    ;;
  serve)
    port=${PORT:-8080}
    exec ./bin/sd-server "${common_args[@]}" \
      --steps "$STEPS" -W "$WIDTH" -H "$HEIGHT" \
      --listen-ip 0.0.0.0 --listen-port "$port"
    ;;
  bench)
    for s in 4 8; do
      echo "--- ${WIDTH}x${HEIGHT} steps=$s ---"
      ./bin/sd "${common_args[@]}" -p 'a red fox sitting in a snowy forest' \
        --steps "$s" -W "$WIDTH" -H "$HEIGHT" --seed "$SEED" \
        -o "output/bench_${s}.png" 2>&1 | grep -aE 's/it|generate_image completed'
    done
    ;;
  devices)
    exec ./bin/sd --list-devices
    ;;
  *)
    echo "usage: $0 {generate [prompt] [out.png] | serve | bench | devices}" >&2
    exit 1
    ;;
esac
