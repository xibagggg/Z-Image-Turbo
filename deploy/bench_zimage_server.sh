#!/bin/bash
# Measure steady-state (model resident) latency of sd-server on the Jetson,
# plus peak unified-memory use via tegrastats.
set -u
cd /home/bianbu/image-generation
cp -f src/stable-diffusion.cpp/build/bin/sd-server bin/ 2>/dev/null

echo "=== sd-server help (endpoint names) ==="
./bin/sd-server --help 2>&1 | head -40

echo "=== starting tegrastats ==="
tegrastats --interval 500 > /home/bianbu/tegrastats.log 2>&1 &
TEGRA=$!
sleep 1

echo "=== starting sd-server on :8080 ==="
./bin/sd-server --diffusion-model models/zimage/z_image_turbo-Q4_K_M.gguf \
  --vae models/zimage/ae.safetensors \
  --llm models/zimage/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --cfg-scale 1.0 --steps 8 -W 1024 -H 1024 --diffusion-fa \
  --listen-ip 0.0.0.0 --listen-port 8080 > /home/bianbu/sdserver.log 2>&1 &
SRV=$!

for i in $(seq 1 90); do
  if curl -s -m 3 http://127.0.0.1:8080/ >/dev/null 2>&1; then break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "server died"; break; fi
  sleep 2
done
sleep 3
echo "=== server log head ==="
head -20 /home/bianbu/sdserver.log

PROMPT='a red fox sitting in a snowy forest, soft morning light, highly detailed, cinematic'
for n in 1 2 3; do
  S=$(date +%s.%N)
  curl -s -m 600 -X POST http://127.0.0.1:8080/v1/images/generations \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"$PROMPT\",\"steps\":8,\"width\":1024,\"height\":1024,\"seed\":$n,\"response_format\":\"b64_json\"}" \
    -o /home/bianbu/srv_resp_$n.json
  RC=$?
  E=$(date +%s.%N)
  SZ=$(stat -c %s /home/bianbu/srv_resp_$n.json 2>/dev/null || echo 0)
  echo "req $n: rc=$RC http_bytes=$SZ wall=$(echo "$E-$S" | bc)s"
done

echo "=== stopping ==="
kill $SRV 2>/dev/null; sleep 3; kill -9 $SRV 2>/dev/null
kill $TEGRA 2>/dev/null

echo "=== peak from tegrastats ==="
grep -o 'RAM [0-9]*/[0-9]*MB' /home/bianbu/tegrastats.log | awk -F'[ /]' '{print $2}' | sort -n | tail -1 | sed 's/^/peak RAM used MB: /'
grep -o 'GR3D_FREQ [0-9]*%' /home/bianbu/tegrastats.log | awk '{print $2}' | tr -d '%' | sort -n | tail -1 | sed 's/^/peak GR3D %: /'
echo "=== server log tail ==="
tail -25 /home/bianbu/sdserver.log
