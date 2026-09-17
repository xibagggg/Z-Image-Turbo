#!/usr/bin/env python3
"""HTTP API + minimal web UI for the on-device text-to-image service.

Endpoints
  GET  /                     web UI (prompt box, live preview)
  POST /api/generate         {"prompt": str, "steps": int, "seed": int, "engine": "onnx"|"gguf"}
                             -> {"ok": true, "image": "data:image/png;base64,...", "info": {...}}
  GET  /health               {"ok": true, "engines": [...]}
  GET  /images/<file>        generated PNGs from output/

Run:  python3 server.py --host 0.0.0.0 --port 8080
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DEFAULT_ONNX = os.path.join(ROOT, "models", "onnx", "sd-turbo")
DEFAULT_GGUF = os.path.join(ROOT, "models", "gguf", "sd15-q8.gguf")
OUT_DIR = os.path.join(ROOT, "output")

_lock = threading.Lock()
_pipes = {}


def get_pipe(engine, threads):
    """Cache one pipeline per engine; the NPU session compile is expensive."""
    key = (engine, threads)
    if key not in _pipes:
        with _lock:
            if key not in _pipes:
                if engine == "onnx":
                    from imagegen.pipeline import TextToImage

                    _pipes[key] = TextToImage(DEFAULT_ONNX, device="cpu", threads=threads)
                else:
                    _pipes[key] = "cpu"
    return _pipes[key]


def generate(prompt, steps, seed, engine, threads, width, height):
    if engine == "onnx":
        pipe = get_pipe("onnx", threads)
        img, info = pipe.generate(prompt, steps=steps, seed=seed)
    else:
        import subprocess
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=OUT_DIR)
        tmp.close()
        cmd = [
            os.path.join(ROOT, "bin", "sd"),
            "-m", DEFAULT_GGUF, "-p", prompt, "-o", tmp.name,
            "--steps", str(steps), "-W", str(width), "-H", str(height),
            "-s", str(seed), "-t", str(threads),
        ]
        t0 = time.time()
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from PIL import Image

        img = Image.open(tmp.name)
        info = {"total_s": time.time() - t0, "engine": "gguf", "steps": steps}

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    name = "img_%d.png" % int(time.time() * 1000)
    with open(os.path.join(OUT_DIR, name), "wb") as fh:
        fh.write(buf.getvalue())
    info["file"] = name
    return base64.b64encode(buf.getvalue()).decode("ascii"), info


PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>端侧文生图 · SpacemiT K3</title>
<style>
 body{font-family:system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;margin:0;background:#111;color:#eee}
 .wrap{max-width:920px;margin:0 auto;padding:24px}
 h1{font-size:20px;font-weight:600;margin:0 0 4px}
 .sub{color:#888;font-size:13px;margin-bottom:20px}
 textarea{width:100%;box-sizing:border-box;background:#1c1c1c;color:#eee;border:1px solid #333;
   border-radius:8px;padding:12px;font-size:15px;font-family:inherit;resize:vertical}
 .row{display:flex;gap:12px;align-items:center;margin-top:12px;flex-wrap:wrap}
 label{font-size:13px;color:#aaa}
 input[type=number]{width:70px;background:#1c1c1c;color:#eee;border:1px solid #333;border-radius:6px;padding:6px}
 button{background:#3b82f6;color:#fff;border:0;border-radius:8px;padding:10px 22px;font-size:15px;cursor:pointer}
 button:disabled{background:#333;cursor:default}
 #out{margin-top:20px}
 img{max-width:100%;border-radius:10px;display:block}
 .meta{color:#888;font-size:13px;margin-top:8px;white-space:pre-wrap}
</style></head><body><div class="wrap">
<h1>端侧文生图 · SpacemiT K3</h1>
<div class="sub">X100 应用核 · ONNX Runtime / stable-diffusion.cpp</div>
<textarea id="p" rows="3">a red fox sitting in a snowy forest, soft morning light, highly detailed</textarea>
<div class="row">
  <label>引擎
    <select id="e"><option value="onnx">X100 · ONNX</option><option value="gguf">X100 · GGUF</option></select>
  </label>
  <label>步数 <input type="number" id="s" value="4" min="1" max="20"></label>
  <label>种子 <input type="number" id="seed" value="42"></label>
  <button id="go" onclick="gen()">生成</button>
</div>
<div id="out"></div>
</div>
<script>
async function gen(){
  const b=document.getElementById('go'); b.disabled=true; b.textContent='生成中…';
  const out=document.getElementById('out'); out.innerHTML='<div class="meta">正在推理，请稍候…</div>';
  const t0=Date.now();
  try{
    const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt:document.getElementById('p').value,
        steps:+document.getElementById('s').value, seed:+document.getElementById('seed').value,
        engine:document.getElementById('e').value})});
    const d=await r.json();
    if(!d.ok){ out.innerHTML='<div class="meta">错误：'+d.error+'</div>'; return; }
    const ms=((Date.now()-t0)/1000).toFixed(1);
    out.innerHTML='<img src="data:image/png;base64,'+d.image+'"><div class="meta">'+
      '引擎 '+d.info.engine+' · '+d.info.steps+' 步 · 端到端 '+ms+'s\\n'+
      JSON.stringify(d.info.timings||{},null,1)+'</div>';
  }catch(err){ out.innerHTML='<div class="meta">请求失败：'+err+'</div>'; }
  finally{ b.disabled=false; b.textContent='生成'; }
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % a))

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html")
        if u.path == "/health":
            return self._send(200, json.dumps({"ok": True, "engines": _engines()}))
        if u.path.startswith("/images/"):
            name = os.path.basename(u.path)
            path = os.path.join(OUT_DIR, name)
            if os.path.isfile(path):
                with open(path, "rb") as fh:
                    return self._send(200, fh.read(), "image/png")
            return self._send(404, json.dumps({"ok": False, "error": "not found"}))
        return self._send(404, json.dumps({"ok": False, "error": "not found"}))

    def do_POST(self):
        if urlparse(self.path).path != "/api/generate":
            return self._send(404, json.dumps({"ok": False, "error": "not found"}))
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, json.dumps({"ok": False, "error": "bad json: %s" % e}))

        global THREADS
        try:
            t0 = time.time()
            img, info = generate(
                req.get("prompt", ""),
                int(req.get("steps", 4)),
                int(req.get("seed", 42)),
                req.get("engine", "onnx"),
                THREADS,
                int(req.get("width", 512)),
                int(req.get("height", 512)),
            )
            info["engine"] = req.get("engine", "onnx")
            info["server_s"] = round(time.time() - t0, 2)
            return self._send(200, json.dumps({"ok": True, "image": img, "info": info}))
        except Exception as e:
            traceback.print_exc()
            return self._send(500, json.dumps({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}))


THREADS = 4


def _engines():
    out = []
    if os.path.isfile(os.path.join(ROOT, "bin", "sd")):
        out.append("gguf")
    if os.path.isdir(DEFAULT_ONNX):
        out.append("onnx")
    return out


def main():
    global THREADS
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("-t", "--threads", type=int, default=4)
    a = ap.parse_args()
    THREADS = a.threads
    os.makedirs(OUT_DIR, exist_ok=True)
    print("serving on http://%s:%d  engines=%s" % (a.host, a.port, _engines()))
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
