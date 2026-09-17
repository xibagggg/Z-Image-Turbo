#!/usr/bin/env python3
"""Unified text-to-image CLI for the SpacemiT K3.

Two engines share one interface:

  onnx  ONNX Runtime on the X100 cores (RVV 1.0). Default and recommended.
        Models: SD-Turbo exported to ONNX by export/export_sd_onnx.py.
  gguf  stable-diffusion.cpp on the X100 cores.
        Models: any SD/SDXL/Flux GGUF.

  --device npu routes the ONNX engine through the SpacemiT execution provider, but that
  EP currently segfaults at session creation on any graph with self-attention.

Examples:
  python3 generate.py -p "a red fox in a snowy forest" -o output/fox.png
  python3 generate.py -e gguf -m models/gguf/sd15-q8.gguf -p "..." -s 6
  python3 generate.py --bench
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DEFAULT_ONNX = os.path.join(ROOT, "models", "onnx", "sd-turbo")
DEFAULT_GGUF = os.path.join(ROOT, "models", "gguf", "sd15-q8.gguf")
SD_CLI = os.path.join(ROOT, "bin", "sd")


# --------------------------------------------------------------------- npu
def run_npu(args):
    from imagegen.pipeline import TextToImage

    pipe = TextToImage(args.model_dir, device=args.device, threads=args.threads, verbose=args.verbose)

    def progress(i, n):
        print("  step %d/%d" % (i, n), flush=True)

    img, info = pipe.generate(args.prompt, steps=args.steps, seed=args.seed)
    out = args.output or os.path.join(ROOT, "output", "npu.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    img.save(out)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print("saved -> %s" % out)


# --------------------------------------------------------------------- cpu
def run_cpu(args):
    import subprocess

    model = args.model or DEFAULT_GGUF
    out = args.output or os.path.join(ROOT, "output", "cpu.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    cmd = [
        SD_CLI,
        "-m", model,
        "-p", args.prompt,
        "-o", out,
        "--steps", str(args.steps),
        "-W", str(args.width),
        "-H", str(args.height),
        "-s", str(args.seed),
        "-t", str(args.threads),
        "-v",
    ]
    if args.cfg_scale is not None:
        cmd += ["--cfg-scale", str(args.cfg_scale)]
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)
    print("total %.1fs -> %s" % (time.time() - t0, out))


# ------------------------------------------------------------------- bench
def run_bench(args):
    import numpy as np

    from imagegen.pipeline import TextToImage

    print("=== ONNX Runtime on X100 CPU, per-component (SD-Turbo) ===")
    results = {}
    for device in ("cpu",):
        pipe = TextToImage(args.model_dir, device=device, threads=args.threads)
        pipe.generate("warm up", steps=1, seed=0)  # warm caches / compile EP subgraphs
        _, info = pipe.generate(args.prompt, steps=args.steps, seed=args.seed)
        results[device] = info
        print("\n[%s]" % device)
        for k, v in info["timings"].items():
            print("  %-14s %8.2f s" % (k, v))
        print("  %-14s %8.2f s" % ("TOTAL", info["total_s"]))

    print("\n=== summary (SD-Turbo @%d steps) ===" % args.steps)
    for k in results["cpu"]["timings"]:
        c = results["cpu"]["timings"].get(k, 0)
        n = results["npu"]["timings"].get(k, 0)
        if c and n:
            print("  %-14s cpu %7.2fs  npu %7.2fs  speedup %.2fx" % (k, c, n, c / n))
    c, n = results["cpu"]["total_s"], results["npu"]["total_s"]
    print("  %-14s cpu %7.2fs  npu %7.2fs  speedup %.2fx" % ("TOTAL", c, n, c / n))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-e", "--engine", choices=["onnx", "gguf"], default="onnx",
                    help="onnx = ONNX Runtime on the X100 cores; gguf = stable-diffusion.cpp")
    ap.add_argument("--device", choices=["cpu", "npu"], default="cpu",
                    help="ONNX Runtime provider; 'npu' currently crashes the SpacemiT EP")
    ap.add_argument("-p", "--prompt", default="a red fox sitting in a snowy forest, soft morning light")
    ap.add_argument("-o", "--output")
    ap.add_argument("-m", "--model", help="GGUF path for the gguf engine")
    ap.add_argument("--model-dir", default=DEFAULT_ONNX, help="ONNX dir for the onnx engine")
    ap.add_argument("-s", "--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-t", "--threads", type=int, default=4)
    ap.add_argument("-W", "--width", type=int, default=512)
    ap.add_argument("-H", "--height", type=int, default=512)
    ap.add_argument("--cfg-scale", type=float, default=None)
    ap.add_argument("--bench", action="store_true", help="compare NPU vs CPU per component")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    if a.bench:
        run_bench(a)
    elif a.engine == "onnx":
        run_npu(a)
    else:
        run_cpu(a)


if __name__ == "__main__":
    main()
