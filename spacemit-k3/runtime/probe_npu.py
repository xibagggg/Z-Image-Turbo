#!/usr/bin/env python3
"""Pinpoint where the SpacemiT EP dies on a given ONNX model.

Prints a marker before/after session creation and before/after the first run, with ORT
verbose logging on, so a segfault can be attributed to a specific phase.

Usage: python3 probe_npu.py <model.onnx> [--disable OPS] [--no-ep]
"""
import sys
import time

import numpy as np
import spacemit_ort  # noqa: F401
import onnxruntime as ort

path = sys.argv[1]
argv = sys.argv[2:]
opts = {"SPACEMIT_EP_INTRA_THREAD_NUM": "4"}
disable = None
no_ep = "--no-ep" in argv
if "--disable" in argv:
    disable = argv[argv.index("--disable") + 1]
for a in argv:
    if a.startswith("--opt="):
        k, _, v = a[6:].partition("=")
        opts[k] = v

so = ort.SessionOptions()
so.log_severity_level = 0 if "--verbose" in argv else 2
so.intra_op_num_threads = 4
if "--no-opt" in argv:
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    print("[!] graph optimizations disabled", flush=True)

print("[1] model   : %s" % path, flush=True)
print("[2] providers available: %s" % ort.get_available_providers(), flush=True)

opts = {"SPACEMIT_EP_INTRA_THREAD_NUM": "4"}
if disable:
    opts["SPACEMIT_EP_DISABLE_OP_TYPE_FILTER"] = disable
if "--verbose" in argv:
    opts["SPACEMIT_EP_DEBUG_PROFILE"] = "/tmp/ep_profile"

if no_ep:
    providers, popts = ["CPUExecutionProvider"], [{}]
else:
    providers, popts = ["SpaceMITExecutionProvider"], [opts]
print("[3] providers requested: %s opts=%s" % (providers, popts), flush=True)

print("[4] creating session ...", flush=True)
t0 = time.time()
sess = ort.InferenceSession(path, so, providers=providers, provider_options=popts)
print("[5] session created in %.2fs, active: %s" % (time.time() - t0, sess.get_providers()), flush=True)

feed = {}
for i in sess.get_inputs():
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
    if "int" in i.type or "int64" in i.type:
        feed[i.name] = np.zeros(shape, dtype=np.int64)
    else:
        feed[i.name] = np.zeros(shape, dtype=np.float32)
    print("[6] input %s %s %s" % (i.name, i.shape, i.type), flush=True)

print("[7] running ...", flush=True)
t0 = time.time()
out = sess.run(None, feed)
print("[8] ran in %.2fs, outputs: %s" % (time.time() - t0, [o.shape for o in out]), flush=True)
print("RESULT_OK", flush=True)
