#!/bin/bash
# Sample tegrastats across one full generation so the GPU/CPU/RAM trace lines
# up in time with the sampling loop.
#
# 1024x1024/8 steps runs ~62s: long enough to see the phases (text encode ->
# sampling -> VAE decode) separate out in the samples.

OUT=/tmp/gpu_watch
mkdir -p "$OUT"
rm -f "$OUT"/tegra.log "$OUT"/gen.json "$OUT"/gen.txt

curl -s -m 600 -X POST http://127.0.0.1:8081/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a red fox sitting in a snowy forest, soft morning light","width":1024,"height":1024,"steps":8,"seed":42}' \
  -o "$OUT/gen.json" -w 'gen http=%{http_code} total=%{time_total}s\n' > "$OUT/gen.txt" 2>&1 &
GENPID=$!

sleep 1                      # let the request land before the baseline samples
timeout 72 tegrastats --interval 1000 > "$OUT/tegra.log" 2>&1
wait $GENPID

cat "$OUT/gen.txt"
echo "samples: $(wc -l < "$OUT/tegra.log")"
