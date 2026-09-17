#!/usr/bin/env python3
"""Studio UI + proxy for the Z-Image-Turbo sd-server.

Serves one single-page UI and proxies generation to sd-server on :8080.
Results are written to output/ui/ with a metadata index, so the gallery lives
on the device (shared by every browser) instead of in one browser's
localStorage, which would blow its ~5 MB quota after two images.

Parameter plumbing - the OpenAI-compatible route in stable-diffusion.cpp
(examples/server/routes_openai.cpp) only reads prompt / n / size /
output_format from the JSON body; every other knob comes from the server's own
CLI defaults. The one escape hatch is a
    <sd_cpp_extra_args>{"seed":N,"sample_params":{"sample_steps":S}}</sd_cpp_extra_args>
block embedded in the prompt, which is stripped from the text and parsed into
SDGenerationParams. Verified against this build: the resulting PNG reports the
requested Steps and Seed. Dimensions go through "size": "WxH".

Stdlib only: the device has no pip packages we want to depend on.
"""

import base64
import json
import os
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "web"
OUT_DIR = HERE / "output" / "ui"
INDEX = OUT_DIR / "index.json"

SD_API = os.environ.get("SD_API", "http://127.0.0.1:8080")
PORT = int(os.environ.get("UI_PORT", "8081"))
# A 4-image batch at 1024x1024 is ~4 min; leave plenty of headroom.
GEN_TIMEOUT = float(os.environ.get("GEN_TIMEOUT", "3600"))

EXTRA_RE = re.compile(r"<sd_cpp_extra_args>(.*?)</sd_cpp_extra_args>", re.S)

_lock = threading.Lock()
# sd-server serialises jobs internally anyway; serialise here too so the
# progress the user sees per image matches what the GPU is actually doing.
_gen_lock = threading.Lock()


# --------------------------------------------------------------- indexing
def load_index():
    if not INDEX.exists():
        return []
    try:
        with INDEX.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (ValueError, OSError):
        return []


def save_index(items):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=1)
    tmp.replace(INDEX)


def slugify(text, limit=48):
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", text).strip("-")
    return (text[:limit] or "image").lower()


# ------------------------------------------------------------ png metadata
def png_text_chunks(blob):
    """Read tEXt/iTXt chunks. sd.cpp embeds the full parameter set, which is the
    only trustworthy record of what the server actually ran with."""
    out, i = {}, 8
    while i + 8 <= len(blob):
        (length,) = struct.unpack(">I", blob[i:i + 4])
        ctype = blob[i + 4:i + 8]
        payload = blob[i + 8:i + 8 + length]
        if ctype == b"tEXt":
            key, _, val = payload.partition(b"\x00")
            out[key.decode("latin-1")] = val.decode("latin-1")
        elif ctype == b"iTXt":
            parts = payload.split(b"\x00", 5)
            if len(parts) == 6:
                out[parts[0].decode("latin-1")] = parts[5].decode("utf-8", "replace")
        elif ctype == b"IEND":
            break
        i += 12 + length
    return out


def parse_applied(meta):
    """Pull the sdcpp.image.params/v1 JSON blob out of the 'parameters' chunk."""
    raw = meta.get("parameters")
    if not raw or "SDCPP: " not in raw:
        return None
    try:
        return json.loads(raw.split("SDCPP: ", 1)[1])
    except ValueError:
        return None


def summarize(raw_meta, applied):
    """A short human-readable line for the UI, built from the server's own report."""
    if not applied:
        return None
    sampling = applied.get("sampling") or {}
    guidance = sampling.get("guidance") or {}
    return {
        "steps": sampling.get("steps"),
        "seed": applied.get("seed"),
        "width": applied.get("width"),
        "height": applied.get("height"),
        "rng": applied.get("rng"),
        "txt_cfg": guidance.get("txt_cfg"),
        "distilled_guidance": guidance.get("distilled_guidance"),
        "header": (raw_meta.get("parameters") or "").split("\n")[1] if "\n" in (raw_meta.get("parameters") or "") else None,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ZImageStudio/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    # ----------------------------------------------------------------- utils
    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    # ------------------------------------------------------------------- GET
    def _route(self):
        """Decoded path without the query string.

        self.path arrives percent-encoded, and our filenames keep the prompt's
        CJK characters, so a Chinese prompt would 404 on /images/... if we used
        the raw value. Decoding happens before the traversal check in the
        callers, so %2e%2e%2f cannot sneak past it.
        """
        return unquote(self.path.split("?", 1)[0])

    def do_GET(self):
        path = self._route()
        if path in ("/", "/index.html"):
            return self._serve_index()
        if path == "/api/health":
            return self._health()
        if path == "/api/gallery":
            with _lock:
                items = load_index()
            return self._json(200, {"items": items})
        if path.startswith("/images/"):
            return self._serve_image(path[len("/images/"):])
        self._json(404, {"error": "not found"})

    def do_HEAD(self):
        self.do_GET()

    def _serve_index(self):
        page = WEB_DIR / "index.html"
        if not page.exists():
            return self._send(500, "web/index.html missing", "text/plain; charset=utf-8")
        self._send(200, page.read_bytes(), "text/html; charset=utf-8")

    def _serve_image(self, name):
        # Only ever serve files that sit directly in OUT_DIR.
        if "/" in name or "\\" in name or name.startswith("."):
            return self._json(400, {"error": "bad name"})
        target = OUT_DIR / name
        if not target.is_file():
            return self._json(404, {"error": "no such image"})
        ctype = "image/png" if name.endswith(".png") else "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.end_headers()
        self.wfile.write(data)

    def _health(self):
        try:
            with urllib.request.urlopen(SD_API + "/v1/models", timeout=5) as resp:
                models = json.loads(resp.read().decode("utf-8"))
            mid = (models.get("data") or [{}])[0].get("id", "unknown")
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            return self._json(503, {"ok": False, "error": str(exc)})
        with _lock:
            count = len(load_index())
        return self._json(200, {"ok": True, "model": mid, "images": count})

    # ------------------------------------------------------------------ POST
    def do_POST(self):
        path = self._route()
        if path == "/api/generate":
            return self._generate()
        self._json(404, {"error": "not found"})

    def _generate(self):
        req = self._read_json()
        if req is None:
            return self._json(400, {"error": "invalid JSON body"})

        # A user could paste an extra-args block themselves; drop it so they
        # cannot smuggle in parameters the UI claims to control.
        prompt = EXTRA_RE.sub("", (req.get("prompt") or "")).strip()
        if not prompt:
            return self._json(400, {"error": "prompt is required"})

        def clamp(value, low, high, default):
            try:
                return max(low, min(high, int(value)))
            except (TypeError, ValueError):
                return default

        steps = clamp(req.get("steps"), 1, 50, 8)
        width = clamp(req.get("width"), 256, 2048, 1024)
        height = clamp(req.get("height"), 256, 2048, 1024)
        seed = clamp(req.get("seed"), 0, 2**31 - 1, 42)

        # CFG stays at 1.0: Z-Image-Turbo is guidance-distilled, so raising it
        # only costs time and quality. It is not exposed in the UI.
        extra = json.dumps({"seed": seed, "sample_params": {"sample_steps": steps}}, separators=(",", ":"))
        wire_prompt = "%s <sd_cpp_extra_args>%s</sd_cpp_extra_args>" % (prompt, extra)

        payload = {
            "prompt": wire_prompt,
            "size": "%dx%d" % (width, height),
            "response_format": "b64_json",
        }

        started = time.time()
        try:
            with _gen_lock:
                body = json.dumps(payload).encode("utf-8")
                api = urllib.request.Request(
                    SD_API + "/v1/images/generations",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(api, timeout=GEN_TIMEOUT) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            return self._json(502, {"error": "sd-server %s: %s" % (exc.code, detail)})
        except Exception as exc:  # noqa: BLE001
            return self._json(502, {"error": "sd-server unreachable: %s" % exc})

        elapsed = round(time.time() - started, 2)
        try:
            raw = base64.b64decode(result["data"][0]["b64_json"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return self._json(502, {"error": "unexpected sd-server response: %s" % exc})

        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = "%s-%s-%d.png" % (stamp, slugify(prompt), seed)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / name).write_bytes(raw)

        # Trust the server's own record over what we asked for; they can differ
        # if a parameter name is wrong, which is exactly what we want to catch.
        meta = png_text_chunks(raw)
        applied = parse_applied(meta)
        if applied:
            seed = applied.get("seed", seed)
            width = applied.get("width", width)
            height = applied.get("height", height)
            steps = ((applied.get("sampling") or {}).get("steps")) or steps

        item = {
            "id": name,
            "url": "/images/" + name,
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "width": width,
            "height": height,
            "elapsed": elapsed,
            "bytes": len(raw),
            "created": int(time.time()),
            "applied": summarize(meta, applied),
        }
        with _lock:
            items = load_index()
            items.insert(0, item)
            del items[500:]  # keep the index bounded
            save_index(items)

        self._json(200, item)

    # ---------------------------------------------------------------- DELETE
    def do_DELETE(self):
        path = self._route()
        if not path.startswith("/api/gallery/"):
            return self._json(404, {"error": "not found"})
        name = path[len("/api/gallery/"):]
        if "/" in name or "\\" in name or name.startswith("."):
            return self._json(400, {"error": "bad name"})
        with _lock:
            items = load_index()
            kept = [it for it in items if it.get("id") != name]
            if len(kept) == len(items):
                return self._json(404, {"error": "no such image"})
            save_index(kept)
        try:
            (OUT_DIR / name).unlink()
        except OSError:
            pass
        self._json(200, {"ok": True})


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    print("Z-Image Studio on http://0.0.0.0:%d  (proxy -> %s)" % (PORT, SD_API), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
