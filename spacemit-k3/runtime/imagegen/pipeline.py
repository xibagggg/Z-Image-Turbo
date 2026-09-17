"""Text-to-image inference on the SpacemiT K3 (X100 CPU + A100 NPU).

Runs an ONNX Stable-Diffusion-Turbo pipeline through ONNX Runtime. On the NPU path the
SpacemiT execution provider compiles subgraphs for the A100 AI core and silently leaves
unsupported ops on the CPU provider, so a single session already spans both.

Pure numpy + onnxruntime: no torch, no diffusers, no transformers on the device.
"""
import json
import os
import time

import numpy as np

import spacemit_ort  # noqa: F401  -- must precede onnxruntime use (patches the EP in)
import onnxruntime as ort

from . import scheduler as sched_mod
from .tokenizer import load as load_tokenizer

NPU = "SpaceMITExecutionProvider"
CPU = "CPUExecutionProvider"


class TextToImage:
    def __init__(
        self,
        model_dir,
        device="cpu",
        threads=4,
        ep_options=None,
        verbose=False,
    ):
        self.dir = model_dir
        with open(os.path.join(model_dir, "meta.json")) as fh:
            self.meta = json.load(fh)
        self.device = device
        self.verbose = verbose
        self.timings = {}

        self.tokenizer = load_tokenizer(os.path.join(model_dir, "tokenizer"))
        self.latent = self.meta["latent_size"]
        self.scale = self.meta["scaling_factor"]
        self.shift = self.meta.get("shift_factor", 0.0) or 0.0

        opts = {"SPACEMIT_EP_INTRA_THREAD_NUM": str(threads)}
        if ep_options:
            opts.update(ep_options)
        self.ep_options = opts

        self._sessions = {}
        self._load("text_encoder")
        self._load("unet")
        self._load("vae_decoder")

    # ------------------------------------------------------------------ setup
    def _providers(self):
        # The NPU provider is opt-in: the SpacemiT EP segfaults at session creation on
        # any graph with self-attention, which every component here has.
        if self.device == "npu":
            return [NPU], [dict(self.ep_options)]
        return [CPU], [{}]

    def _load(self, name):
        path = os.path.join(self.dir, name + ".onnx")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = int(self.ep_options["SPACEMIT_EP_INTRA_THREAD_NUM"])
        so.add_session_config_entry("session.intra_op.allow_spinning", "1")
        if name == "vae_decoder":
            # The decoder allocates ~130MB tensors at 512x512 and ORT's CPU arena never
            # gives that back, which pushed the service to 9GB RSS after a single image.
            # It runs once per image, so trading a little speed for bounded memory is
            # the right call.
            so.enable_cpu_mem_arena = False
        providers, popts = self._providers()
        t0 = time.time()
        self._sessions[name] = ort.InferenceSession(
            path, so, providers=providers, provider_options=popts
        )
        if self.verbose:
            print(
                "[load] %-13s %6.2fs  providers=%s"
                % (name, time.time() - t0, self._sessions[name].get_providers())
            )

    def _run(self, name, feed):
        t0 = time.time()
        out = self._sessions[name].run(None, feed)[0]
        self.timings[name] = self.timings.get(name, 0.0) + time.time() - t0
        return out

    # -------------------------------------------------------------- inference
    def encode_prompt(self, prompt):
        ids = np.asarray([self.tokenizer.encode(prompt)], dtype=np.int64)
        emb = self._run("text_encoder", {"input_ids": ids})
        return emb.astype(np.float16)

    def _scheduler(self, steps):
        key = str(steps)
        table = self.meta["schedules"][key]
        cls = sched_mod.REGISTRY[self.meta["scheduler"]["name"]]
        return cls(table["timesteps"], table["sigmas"])

    def generate(self, prompt, steps=4, seed=42, negative_prompt=None, callback=None):
        """Return (PIL.Image, info-dict). SD-Turbo runs without classifier-free guidance."""
        self.timings = {}
        t_start = time.time()

        emb = self.encode_prompt(prompt)
        if negative_prompt:
            # SD-Turbo is CFG-free by design; a negative prompt would change the sampler
            # contract, so we only warn rather than silently doing nothing.
            raise ValueError("SD-Turbo is a guidance-free model: negative_prompt is not supported")

        sch = self._scheduler(steps)
        rng = np.random.default_rng(seed)

        latents = rng.standard_normal((1, 4, self.latent, self.latent)).astype(np.float32)
        latents = latents * sch.init_noise_sigma

        emb16 = emb.astype(np.float16)
        for i, t in enumerate(sch.timesteps):
            sample = sch.scale_model_input(latents, i).astype(np.float16)
            ts = np.asarray([t], dtype=np.float16)
            noise = self._run(
                "unet", {"sample": sample, "timestep": ts, "encoder_hidden_states": emb16}
            )
            latents = sch.step(noise.astype(np.float32), i, latents, rng)
            if callback:
                callback(i + 1, len(sch.timesteps))

        latent = latents / self.scale
        if self.shift:
            latent = latent - self.shift
        vae_dtype = np.float16 if self.meta.get("vae_dtype") == "float16" else np.float32
        image = self._run("vae_decoder", {"latent": latent.astype(vae_dtype)})

        image = np.clip(image[0].transpose(1, 2, 0) / 2.0 + 0.5, 0.0, 1.0)
        image = (image * 255.0).round().astype(np.uint8)
        info = {
            "total_s": time.time() - t_start,
            "steps": steps,
            "device": self.device,
            "timings": dict(self.timings),
            "resolution": self.meta["resolution"],
        }
        return _to_pil(image), info


def _to_pil(arr):
    from PIL import Image

    return Image.fromarray(arr)


def available_devices():
    return [NPU] if NPU in ort.get_available_providers() else []
