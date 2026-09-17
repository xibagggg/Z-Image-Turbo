#!/usr/bin/env python3
"""A/B the SpacemiT NPU against plain CPU on the same ONNX graph.

Usage: ab_test.py <model.onnx> [runs]
"""
import sys
import time

import numpy as np
import onnxruntime as ort
import spacemit_ort  # noqa: F401

model = sys.argv[1]
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10

so = ort.SessionOptions()
so.log_severity_level = 3


def bench(providers, label):
    t0 = time.time()
    sess = ort.InferenceSession(model, so, providers=providers)
    init = time.time() - t0
    feed = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        feed[i.name] = np.random.rand(*shape).astype(np.float32)
    sess.run(None, feed)
    ts = []
    for _ in range(runs):
        t0 = time.time()
        sess.run(None, feed)
        ts.append(time.time() - t0)
    print("%-28s init=%6.2fs  avg=%8.2f ms  min=%8.2f ms" % (label, init, sum(ts) / len(ts) * 1e3, min(ts) * 1e3))
    return min(ts)


npu = bench(["SpaceMITExecutionProvider", "CPUExecutionProvider"], "NPU(SpaceMIT EP)")
cpu = bench(["CPUExecutionProvider"], "CPU only")
print("NPU speedup: %.2fx" % (cpu / npu))
print("RESULT_OK")
