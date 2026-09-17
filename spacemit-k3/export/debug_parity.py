#!/usr/bin/env python3
"""Step-by-step parity check: my ONNX/numpy pipeline vs the diffusers reference.

Both run on the PC (CPU) with the same prompt, seed and step count. The reference is
spied on every UNet call so we can compare stage by stage:

  token ids -> prompt embeddings -> per-step UNet input (after scale_model_input)
           -> per-step UNet output -> sampled latents -> decoded image

Usage: python3 debug_parity.py [--steps 4] [--prompt ...] [--seed 42]
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "build", "sd-turbo")
ONNX = os.path.join(HERE, "build", "sd-turbo-onnx")
sys.path.insert(0, os.path.join(HERE, "..", "runtime"))


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-8))


def show(label, a, b):
    r = rel(a, b)
    print("  %-22s rel=%-12.4g %s" % (label, r, "OK" if r < 5e-2 else "<-- DIVERGES"))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a red fox sitting in a snowy forest, soft morning light")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

    print("=== diffusers reference (fp32) ===")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL, variant="fp16", torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    hist = {"in": [], "t": [], "out": []}
    orig = pipe.unet.forward

    def spy(sample, timestep, encoder_hidden_states, **kw):
        out = orig(sample, timestep, encoder_hidden_states, **kw)
        tensor = out.sample if hasattr(out, "sample") else out[0]
        hist["in"].append(sample.detach().float().numpy())
        hist["t"].append(float(np.ravel(timestep.detach().numpy())[0]))
        hist["out"].append(tensor.detach().float().numpy())
        return out

    pipe.unet.forward = spy
    g = torch.Generator().manual_seed(a.seed)
    img = pipe(a.prompt, num_inference_steps=a.steps, guidance_scale=0.0, generator=g).images[0]
    img.save(os.path.join(HERE, "build", "ref_diffusers.png"))

    ids = pipe.tokenizer(a.prompt, padding="max_length", max_length=77, truncation=True,
                         return_tensors="pt").input_ids
    with torch.no_grad():
        ref_emb = pipe.text_encoder(ids)[0].float().numpy()
    print("  steps=%d  unet calls=%d  t=%s" % (a.steps, len(hist["in"]), hist["t"]))
    print("  ref unet in[0] absmax=%.4f  out[0] absmax=%.4f"
          % (np.abs(hist["in"][0]).max(), np.abs(hist["out"][0]).max()))
    print()

    print("=== my pipeline (ONNX, CPU) ===")
    import onnxruntime as ort
    import imagegen.tokenizer as T
    import imagegen.scheduler as S

    tok = T.load(os.path.join(ONNX, "tokenizer"))
    my_ids = np.asarray([tok.encode(a.prompt)], dtype=np.int64)
    show("token ids", my_ids, ids.numpy())

    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    so.log_severity_level = 3
    sess = {n: ort.InferenceSession(os.path.join(ONNX, n + ".onnx"), so,
                                    providers=["CPUExecutionProvider"])
            for n in ("text_encoder", "unet", "vae_decoder")}

    my_emb = sess["text_encoder"].run(None, {"input_ids": my_ids})[0]
    show("prompt_embeds", my_emb, ref_emb)

    meta = json.load(open(os.path.join(ONNX, "meta.json")))
    sch = S.Euler(meta["schedules"][str(a.steps)]["timesteps"],
                  meta["schedules"][str(a.steps)]["sigmas"])

    # Start from the reference's own noise so the trajectories are comparable; numpy and
    # torch draw different values for the same seed.
    sig0 = float(sch.sigmas[0])
    lat = hist["in"][0].astype(np.float32) * np.float32((sig0 ** 2 + 1.0) ** 0.5)
    rng = np.random.default_rng(a.seed)

    for i, t in enumerate(sch.timesteps):
        sample = sch.scale_model_input(lat, i).astype(np.float16)
        if i == 0:
            show("unet in[0]", sample, hist["in"][0])
        out = sess["unet"].run(None, {
            "sample": sample,
            "timestep": np.asarray([t], dtype=np.float16),
            "encoder_hidden_states": my_emb.astype(np.float16),
        })[0]
        show("unet out[%d]" % i, out, hist["out"][i])
        lat = sch.step(out.astype(np.float32), i, lat, rng)

    latent = (lat / meta["scaling_factor"]).astype(np.float32)
    my_img = sess["vae_decoder"].run(None, {"latent": latent})[0]
    my_img = np.clip(my_img[0].transpose(1, 2, 0) / 2 + 0.5, 0, 1)
    Image.fromarray((my_img * 255).round().astype(np.uint8)).save(
        os.path.join(HERE, "build", "mine_onnx.png"))

    show("final image", my_img, np.asarray(img).astype(np.float64) / 255.0)
    print("\nwrote build/ref_diffusers.png and build/mine_onnx.png")


if __name__ == "__main__":
    main()
