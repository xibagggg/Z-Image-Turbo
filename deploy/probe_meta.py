#!/usr/bin/env python3
"""Dump the PNG 'parameters' chunk to confirm exactly what the server used."""

import base64
import json
import struct
import time
import urllib.request

API = "http://127.0.0.1:8080/v1/images/generations"
PROMPT = "a red fox sitting in a snowy forest"


def gen(prompt, size):
    body = {"prompt": prompt, "size": size, "response_format": "b64_json"}
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        blob = base64.b64decode(json.loads(resp.read().decode())["data"][0]["b64_json"])
    return time.time() - t0, blob


def chunk(blob, want):
    i = 8
    while i + 8 <= len(blob):
        (length,) = struct.unpack(">I", blob[i:i + 4])
        ctype = blob[i + 4:i + 8]
        payload = blob[i + 8:i + 8 + length]
        if ctype == want:
            return payload
        if ctype == b"IEND":
            break
        i += 12 + length
    return None


cases = [
    ("baseline (server defaults)", PROMPT, "256x256"),
    ("seed + 2 steps", PROMPT + ' <sd_cpp_extra_args>{"seed": 999, "sample_params": {"sample_steps": 2}}</sd_cpp_extra_args>', "256x256"),
    ("seed + 3 steps + cfg", PROMPT + ' <sd_cpp_extra_args>{"seed": 12345, "sample_params": {"sample_steps": 3, "guidance": 1.0, "cfg_scale": 1.0}}</sd_cpp_extra_args>', "320x256"),
]

for label, prompt, size in cases:
    elapsed, blob = gen(prompt, size)
    raw = chunk(blob, b"tEXt") or chunk(blob, b"iTXt")
    text = raw.decode("utf-8", "replace") if raw else "<none>"
    print(f"--- {label} ---")
    print(f"elapsed={elapsed:.2f}s  size={size}  png={struct.unpack('>II', blob[16:24])}")
    print(text.replace("\x00", ": ", 1))
    print()
