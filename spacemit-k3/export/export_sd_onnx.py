#!/usr/bin/env python3
"""Export SD-Turbo (or any Turbo SD1.5-family checkpoint) to device-ready ONNX.

Produces, under out_dir/:
  text_encoder.onnx      input_ids[1,77] int64 -> last_hidden_state[1,77,768] fp16
  unet.onnx              sample[1,4,64,64] fp16 + timestep[1] fp16 + emb[1,77,768] fp16 -> [1,4,64,64] fp16
  vae_decoder.onnx       latent[1,4,64,64] -> image[1,3,512,512] fp32
  tokenizer/vocab.json, tokenizer/merges.txt
  meta.json              scaling factor + precomputed Euler-ancestral sigma tables
  ref_*.npy              reference tensors for pipeline parity checks

Fixed batch-1 / 512x512 shapes throughout: the SpacemiT NPU provider compiles
against static shapes.
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

STEPS_TO_DUMP = list(range(1, 13))


class VaeDecode(torch.nn.Module):
    """vae.decode(z) without the BaseOutput wrapper, so it exports cleanly."""

    def __init__(self, vae):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, z):
        return self.decoder(self.post_quant_conv(z))


def export(mod, args, path, input_names, output_names, exporter="dynamo"):
    """Export one component.

    exporter="dynamo"  -- fast (~30-90s even for the 1.7GB UNet), but emits opset 20 and
                          puts weights in a sidecar `<path>.data` file.
    exporter="legacy"  -- opset 17 in a single self-contained file, matching what the
                          SpacemiT model zoo ships; the TorchScript exporter needs >20
                          minutes on the SD2.1 UNet, so it is opt-in only.

    Note: the SpacemiT NPU provider segfaults on both flavours anyway (see README), so
    the default is the fast one.
    """
    kw = dict(input_names=input_names, output_names=output_names, do_constant_folding=False)
    if exporter == "legacy":
        kw.update(opset_version=17, dynamo=False)
    else:
        kw.update(dynamo=True)
    torch.onnx.export(mod, args, path, **kw)
    data = path + ".data"
    size = os.path.getsize(path) + (os.path.getsize(data) if os.path.exists(data) else 0)
    print("wrote %s (%.1f MB)" % (path, size / 1e6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="build/sd-turbo", help="local diffusers folder")
    ap.add_argument("--out", default="build/sd-turbo-onnx")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--vae-fp16", action="store_true")
    ap.add_argument("--exporter", default="dynamo", choices=["dynamo", "legacy"])
    ap.add_argument("--only", default="all",
                    choices=["all", "text_encoder", "unet", "vae_decoder", "meta"])
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    lat = a.res // 8
    # Three heavyweight models in one process slows the dynamo exporter down badly, so
    # `--only` lets each component export in a fresh process.
    want = lambda n: a.only in ("all", n)

    with open(os.path.join(a.model, "scheduler", "scheduler_config.json")) as fh:
        sched_cfg = json.load(fh)
    sched_name = sched_cfg.get("_class_name", "EulerAncestralDiscreteScheduler")
    print("scheduler:", sched_name)

    if want("text_encoder"):
        tokenizer = CLIPTokenizer.from_pretrained(a.model, subfolder="tokenizer")
        te = CLIPTextModel.from_pretrained(
            a.model, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float16
        ).eval()
        ids = torch.zeros((1, 77), dtype=torch.int64)
        with torch.no_grad():
            ref_emb = te(ids)[0].float().numpy()
        export(te, (ids,), os.path.join(a.out, "text_encoder.onnx"), ["input_ids"], ["last_hidden_state"], a.exporter)
        np.save(os.path.join(a.out, "ref_text_emb.npy"), ref_emb)
        _write_meta_extra(a.out, {"cross_dim": int(te.config.hidden_size)})
        del te

    if want("unet"):
        cross_dim = _read_meta_extra(a.out).get("cross_dim", 1024)
        unet = UNet2DConditionModel.from_pretrained(
            a.model, subfolder="unet", variant="fp16", torch_dtype=torch.float16
        ).eval()
        args = (
            torch.zeros((1, 4, lat, lat), dtype=torch.float16),
            torch.tensor([1.0], dtype=torch.float16),
            torch.zeros((1, 77, cross_dim), dtype=torch.float16),
        )
        # PyTorch's CPU fp16 convolution path is pathologically slow (~10 min for one
        # forward here), so compute the reference in fp32 and put the model back to
        # fp16 before tracing.
        with torch.no_grad():
            ref_unet = unet.float()(*[t.float() for t in args]).sample.float().numpy()
        unet.half()
        export(unet, args, os.path.join(a.out, "unet.onnx"),
               ["sample", "timestep", "encoder_hidden_states"], ["out_sample"], a.exporter)
        np.save(os.path.join(a.out, "ref_unet_out.npy"), ref_unet)
        print("cross-attention dim:", cross_dim)
        del unet

    if want("vae_decoder"):
        vae = AutoencoderKL.from_pretrained(
            a.model, subfolder="vae", variant="fp16", torch_dtype=torch.float32
        ).eval()
        dec = VaeDecode(vae)
        dtype = torch.float16 if a.vae_fp16 else torch.float32
        dec = dec.to(dtype)
        z = torch.zeros((1, 4, lat, lat), dtype=dtype)
        with torch.no_grad():
            ref_vae = dec(z).float().numpy()
        export(dec, (z,), os.path.join(a.out, "vae_decoder.onnx"), ["latent"], ["image"], a.exporter)
        np.save(os.path.join(a.out, "ref_vae_out.npy"), ref_vae)
        _write_meta_extra(a.out, {
            "scaling_factor": float(vae.config.scaling_factor),
            "shift_factor": float(getattr(vae.config, "shift_factor", 0.0) or 0.0),
        })
        print("scaling_factor=%.6f latent=%d" % (vae.config.scaling_factor, lat))

    if want("meta"):
        tdir = os.path.join(a.out, "tokenizer")
        os.makedirs(tdir, exist_ok=True)
        for f in ("vocab.json", "merges.txt"):
            shutil.copyfile(os.path.join(a.model, "tokenizer", f), os.path.join(tdir, f))

        # The device sampler consumes these tables verbatim so it can never drift
        # from the reference scheduler.
        sched_cls = {"EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
                     "EulerDiscreteScheduler": EulerDiscreteScheduler}.get(sched_name)
        if sched_cls is None:
            raise SystemExit("unsupported scheduler %s" % sched_name)
        schedules = {}
        for n in STEPS_TO_DUMP:
            s = sched_cls.from_pretrained(a.model, subfolder="scheduler")
            s.set_timesteps(n)
            schedules[str(n)] = {
                "timesteps": [float(t) for t in s.timesteps.cpu().numpy()],
                "sigmas": [float(x) for x in s.sigmas.cpu().numpy()],
            }
        extra = _read_meta_extra(a.out)
        meta = {
            "model": a.model,
            "resolution": a.res,
            "latent_size": lat,
            "cross_dim": extra.get("cross_dim", 1024),
            "vae_dtype": "float16" if a.vae_fp16 else "float32",
            "scaling_factor": extra.get("scaling_factor", 0.18215),
            "shift_factor": extra.get("shift_factor", 0.0),
            "scheduler": {"name": sched_name},
            "schedules": schedules,
            "default_steps": 4,
        }
        with open(os.path.join(a.out, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        print("dumped schedules for steps:", STEPS_TO_DUMP)


def _meta_extra_path(out):
    return os.path.join(out, "_meta_extra.json")


def _write_meta_extra(out, fields):
    cur = _read_meta_extra(out)
    cur.update(fields)
    with open(_meta_extra_path(out), "w") as fh:
        json.dump(cur, fh, indent=2)


def _read_meta_extra(out):
    p = _meta_extra_path(out)
    if os.path.isfile(p):
        with open(p) as fh:
            return json.load(fh)
    return {}


if __name__ == "__main__":
    main()
