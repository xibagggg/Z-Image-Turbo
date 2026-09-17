#!/usr/bin/env python3
"""Numerically validate every ONNX component against the PyTorch reference tensors.

The exporter saves ref_text_emb.npy / ref_unet_out.npy / ref_vae_out.npy produced by the
original torch modules with fixed zero inputs. Re-running the ONNX graphs with the same
inputs must reproduce them, on both the NPU provider and the CPU provider.

Usage:
  python3 validate.py                          # npu then cpu, all components
  python3 validate.py --device cpu             # only the CPU provider
  python3 validate.py --only unet --device npu
  python3 validate.py --npu-opt SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=Erf;Sin;Cos
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import spacemit_ort  # noqa: F401
import onnxruntime as ort

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(ROOT, "models", "onnx", "sd-turbo")


def rel_err(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = max(float(np.abs(b).max()), 1e-8)
    return float(np.abs(a - b).max() / denom)


def make_session(path, device, threads, extra_opts):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = threads
    if device == "npu":
        opts = {"SPACEMIT_EP_INTRA_THREAD_NUM": str(threads)}
        opts.update(extra_opts)
        return ort.InferenceSession(
            path, so, providers=["SpaceMITExecutionProvider"], provider_options=[opts]
        )
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_DIR)
    ap.add_argument("--device", default="both", choices=["npu", "cpu", "both"])
    ap.add_argument("--only", default="all",
                    choices=["all", "text_encoder", "unet", "vae_decoder"])
    ap.add_argument("-t", "--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--npu-opt", action="append", default=[],
                    help="extra SpaceMIT EP provider option NAME=VALUE")
    a = ap.parse_args()

    extra_opts = {}
    for kv in a.npu_opt:
        k, _, v = kv.partition("=")
        extra_opts[k] = v

    with open(os.path.join(a.model_dir, "meta.json")) as fh:
        meta = json.load(fh)
    lat, emb_dim = meta["latent_size"], meta.get("cross_dim", 1024)

    cases = {}
    cases["text_encoder"] = (
        {"input_ids": np.zeros((1, 77), dtype=np.int64)},
        np.load(os.path.join(a.model_dir, "ref_text_emb.npy")),
    )
    cases["unet"] = (
        {
            "sample": np.zeros((1, 4, lat, lat), dtype=np.float16),
            "timestep": np.asarray([1.0], dtype=np.float16),
            "encoder_hidden_states": np.zeros((1, 77, emb_dim), dtype=np.float16),
        },
        np.load(os.path.join(a.model_dir, "ref_unet_out.npy")),
    )
    vae_dtype = np.float16 if meta.get("vae_dtype") == "float16" else np.float32
    cases["vae_decoder"] = (
        {"latent": np.zeros((1, 4, lat, lat), dtype=vae_dtype)},
        np.load(os.path.join(a.model_dir, "ref_vae_out.npy")),
    )

    names = list(cases) if a.only == "all" else [a.only]
    devices = ["npu", "cpu"] if a.device == "both" else [a.device]

    print("model dir : %s" % a.model_dir)
    print("latent=%d emb_dim=%d vae_dtype=%s threads=%d" % (lat, emb_dim, meta.get("vae_dtype"), a.threads))
    if extra_opts:
        print("ep opts   : %s" % extra_opts)
    print()

    results = {}
    for device in devices:
        print("=== %s ===" % device, flush=True)
        for name in names:
            path = os.path.join(a.model_dir, name + ".onnx")
            if not os.path.isfile(path):
                print("  %-14s MISSING %s" % (name, path), flush=True)
                continue
            try:
                t0 = time.time()
                sess = make_session(path, device, a.threads, extra_opts)
                init = time.time() - t0
                feed = cases[name][0]
                sess.run(None, feed)  # warm-up (also triggers EP subgraph compile)
                ts = []
                for _ in range(a.runs):
                    t0 = time.time()
                    out = sess.run(None, feed)[0]
                    ts.append(time.time() - t0)
                err = rel_err(out, cases[name][1])
                tag = "OK " if err < 5e-2 else "BAD"
                lat_ms = min(ts) * 1e3
                print("  %-14s %s init %7.2fs  run %9.2f ms  rel_err %.4g"
                      % (name, tag, init, lat_ms, err), flush=True)
                results[(device, name)] = lat_ms
            except Exception as e:
                print("  %-14s FAILED: %s: %s" % (name, type(e).__name__, str(e)[:300]), flush=True)
        print(flush=True)

    if results:
        print("=== latency summary (ms) ===")
        for name in names:
            n = results.get(("npu", name))
            c = results.get(("cpu", name))
            line = "  %-14s" % name
            line += "  npu %10.1f" % n if n else "  npu %10s" % "-"
            line += "  cpu %10.1f" % c if c else "  cpu %10s" % "-"
            if n and c:
                line += "   speedup %6.2fx" % (c / n)
            print(line)


if __name__ == "__main__":
    main()
