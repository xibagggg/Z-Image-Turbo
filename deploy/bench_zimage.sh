set -e
cd /home/bianbu/image-generation
echo "=== A: 512x512 8 steps ==="
/usr/bin/time -f "WALL %e s" ./bin/sd --diffusion-model models/zimage/z_image_turbo-Q4_K_M.gguf \
  --vae models/zimage/ae.safetensors --llm models/zimage/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -p 'a red fox sitting in a snowy forest, soft morning light, highly detailed, cinematic' \
  --cfg-scale 1.0 --steps 8 -W 512 -H 512 --seed 42 --diffusion-fa \
  -o output/zimage_512.png 2>&1 | grep -E "s/it|completed, taking|generate_image completed|WALL|compute buffer|params backend|available_memory" | tail -20
echo "=== B: 1024x1024 4 steps ==="
/usr/bin/time -f "WALL %e s" ./bin/sd --diffusion-model models/zimage/z_image_turbo-Q4_K_M.gguf \
  --vae models/zimage/ae.safetensors --llm models/zimage/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -p 'a red fox sitting in a snowy forest, soft morning light, highly detailed, cinematic' \
  --cfg-scale 1.0 --steps 4 -W 1024 -H 1024 --seed 42 --diffusion-fa \
  -o output/zimage_1024_s4.png 2>&1 | grep -E "s/it|completed, taking|generate_image completed|WALL|compute buffer|params backend|available_memory" | tail -20
echo "=== files ==="
ls -la output/
