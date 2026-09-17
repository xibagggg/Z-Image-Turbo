#!/usr/bin/env python3
"""Smoke-test the SpacemiT NPU (A100 AI core) through the shared ORT execution provider."""
import sys
import time

import numpy as np
import onnxruntime as ort
import spacemit_ort  # noqa: F401  (patches ORT so 'SpaceMITExecutionProvider' resolves)

model = sys.argv[1] if len(sys.argv) > 1 else "/home/bianbu/image-generation/models/mobilenet_v3_small.fp16.onnx"
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5

print("available providers:", ort.get_available_providers())

so = ort.SessionOptions()
so.log_severity_level = 3

t0 = time.time()
sess = ort.InferenceSession(model, so, providers=["SpaceMITExecutionProvider"])
print("session init: %.2fs" % (time.time() - t0))
print("active providers:", sess.get_providers())

feed = {}
for i in sess.get_inputs():
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
    print("  input:", i.name, i.shape, i.type)
    feed[i.name] = np.random.rand(*shape).astype(np.float32)

sess.run(None, feed)  # warm-up

ts = []
for _ in range(runs):
    t0 = time.time()
    out = sess.run(None, feed)
    ts.append(time.time() - t0)

print("outputs:", [o.shape for o in out])
print("latency ms: min=%.2f avg=%.2f" % (min(ts) * 1e3, sum(ts) / len(ts) * 1e3))
print("RESULT_OK")
