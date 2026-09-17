"""Euler ancestral sampler, matching diffusers' EulerAncestralDiscreteScheduler.

The sigma/timestep tables are precomputed on the host by export_sd_onnx.py and shipped
in meta.json, so the device never has to re-derive the schedule (and can't drift from
the reference implementation).
"""
import numpy as np


class EulerAncestral:
    def __init__(self, timesteps, sigmas):
        # sigmas has len(timesteps) + 1 entries, the last being 0.0
        self.timesteps = np.asarray(timesteps, dtype=np.float32)
        self.sigmas = np.asarray(sigmas, dtype=np.float32)

    @property
    def init_noise_sigma(self):
        return float(self.sigmas.max())

    def scale_model_input(self, sample, step):
        return sample

    def step(self, model_output, step, sample, rng):
        sigma = float(self.sigmas[step])
        sigma_next = float(self.sigmas[step + 1])

        # epsilon prediction
        pred_original = sample - sigma * model_output

        # ancestral noise injection
        sigma_up = (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / (sigma ** 2)) ** 0.5
        sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5

        derivative = (sample - pred_original) / sigma
        prev = sample + derivative * (sigma_down - sigma)
        if sigma_up > 0:
            prev = prev + rng.standard_normal(sample.shape).astype(np.float32) * sigma_up
        return prev.astype(np.float32)


class Euler:
    """Plain (deterministic) Euler, matching diffusers' EulerDiscreteScheduler.

    Unlike the ancestral variant, this one rescales the model input by
    1/sqrt(sigma^2 + 1) before every UNet call. Skipping that puts the UNet on a
    ~15x wrong input scale at step 1 and the result is pure noise.
    """

    def __init__(self, timesteps, sigmas):
        self.timesteps = np.asarray(timesteps, dtype=np.float32)
        self.sigmas = np.asarray(sigmas, dtype=np.float32)

    @property
    def init_noise_sigma(self):
        return float(self.sigmas.max())

    def scale_model_input(self, sample, step):
        sigma = float(self.sigmas[step])
        return sample / np.float32((sigma ** 2 + 1.0) ** 0.5)

    def step(self, model_output, step, sample, rng):
        sigma = float(self.sigmas[step])
        sigma_next = float(self.sigmas[step + 1])
        pred_original = sample - sigma * model_output
        derivative = (sample - pred_original) / sigma
        return (sample + derivative * (sigma_next - sigma)).astype(np.float32)


REGISTRY = {"EulerAncestralDiscreteScheduler": EulerAncestral, "EulerDiscreteScheduler": Euler}
