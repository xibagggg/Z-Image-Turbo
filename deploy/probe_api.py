#!/usr/bin/env python3
"""Probe what /v1/images/generations actually honours.

routes_openai.cpp only reads prompt/n/size/output_format from the JSON body and
takes everything else from the server's CLI defaults. The one escape hatch is a
<sd_cpp_extra_args>{...}</sd_cpp_extra_args> block embedded in the prompt, which
is fed to SDGenerationParams::from_json_str. This script tries several payload
shapes at a small size and reads back the PNG's embedded metadata to see what
the server really used. Stdlib only.
"""

import base64
import json
import re
import struct
import time
import urllib.request

API = "http://127.0.0.1:8080/v1/images/generations"


def post(body, timeout=600):
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return time.time() - t0, base64.b64decode(data["data"][0]["b64_json"])


def png_meta(blob):
    """Pull tEXt/iTXt chunks out of a PNG without any third-party library."""
    out, i = {}, 8
    while i + 8 <= len(blob):
        (length,) = struct.unpack(">I", blob[i:i + 4])
        ctype = blob[i + 4:i + 8]
        payload = blob[i + 8:i + 8 + length]
        if ctype == b"tEXt":
            k, _, v = payload.partition(b"\x00")
            out[k.decode("latin-1")] = v.decode("latin-1")
        elif ctype == b"iTXt":
            parts = payload.split(b"\x00", 5)
            if len(parts) == 6:
                out[parts[0].decode("latin-1")] = parts[5].decode("utf-8", "replace")
        if ctype == b"IEND":
            break
        i += 12 + length
    return out


def dims(blob):
    return struct.unpack(">II", blob[16:24])


def interesting(meta):
    """Keep the handful of fields we care about, plus raw keys for discovery."""
    keys = ("steps", "seed", "width", "height", "sample_steps", "cfg_scale", "scheduler")
    picked = {k: meta[k] for k in keys if k in meta}
    return picked or {"_keys": sorted(meta)[:14]}


PROMPT = "a red fox sitting in a snowy forest"

CASES = [
    ("baseline, no extras", PROMPT),
    ("extra_args: seed", PROMPT + ' <sd_cpp_extra_args>{"seed": 999}</sd_cpp_extra_args>'),
    ("extra_args: width/height", PROMPT + ' <sd_cpp_extra_args>{"width": 192, "height": 320}</sd_cpp_extra_args>'),
    ("extra_args: sample_params.sample_steps",
     PROMPT + ' <sd_cpp_extra_args>{"sample_params": {"sample_steps": 2}}</sd_cpp_extra_args>'),
    ("extra_args: extra_sample_args=--steps 2",
     PROMPT + ' <sd_cpp_extra_args>{"extra_sample_args": "--steps 2"}</sd_cpp_extra_args>'),
    ("extra_args: combined seed+steps",
     PROMPT + ' <sd_cpp_extra_args>{"seed": 999, "sample_params": {"sample_steps": 2}}</sd_cpp_extra_args>'),
]

print("probing /v1/images/generations ...\n")
for label, prompt in CASES:
    body = {"prompt": prompt, "size": "256x256", "response_format": "b64_json"}
    try:
        elapsed, blob = post(body)
    except Exception as exc:  # noqa: BLE001
        print(f"{label:44s} ERROR {exc}")
        continue
    meta = png_meta(blob)
    print(f"{label:44s} {elapsed:6.2f}s  png={dims(blob)}  meta={interesting(meta)}")
