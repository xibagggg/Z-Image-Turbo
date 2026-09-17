#!/bin/bash
# Generate the two comparison series:
#   A) same seed, steps 1/2/4/8   -> what steps buy you
#   B) same steps, seeds 42/43/44 -> what the seed changes
# 512x512 keeps each run short so the whole set finishes in about a minute.

OUT=/tmp/sd_demo
mkdir -p "$OUT"
rm -f "$OUT"/*.json

gen() {  # gen <tag> <steps> <seed>
  local tag=$1 steps=$2 seed=$3
  local t0=$SECONDS
  curl -s -m 300 -X POST http://127.0.0.1:8081/api/generate \
    -H "Content-Type: application/json" \
    -d "{\"prompt\":\"一只红色的狐狸坐在雪地里，柔和的晨光，细节丰富，电影感\",\"width\":512,\"height\":512,\"steps\":$steps,\"seed\":$seed}" \
    -o "$OUT/$tag.json"
  python3 - "$OUT/$tag.json" "$tag" "$((SECONDS - t0))" <<'PY'
import json, sys
p, tag, wall = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(p))
print("%-10s steps=%-2s seed=%-4s elapsed=%6.2fs  wall=%ss  %s"
      % (tag, d["steps"], d["seed"], d["elapsed"], wall, d["id"][:28]))
PY
}

echo "--- A) 同 seed=42，不同步数 ---"
for s in 1 2 4 8; do gen "steps$s" "$s" 42; done

echo
echo "--- B) 同步数=8，不同 seed ---"
for sd in 42 43 44; do gen "seed$sd" 8 "$sd"; done

echo
echo "--- 结果文件 ---"
ls "$OUT"/*.png 2>/dev/null | sed 's|.*/||'
