"""Runtime setup that must run before importing JAX."""

from __future__ import annotations

import os
import sys


RTX_5050_DEFAULTS = {
    "device": "gpu",
    "num_envs": 256,
    "horizon": 256,
    "minibatches": 32,
    "gpu_memory_fraction": 0.60,
}


RTX_5050_LONG_CREDIT_DEFAULTS = {
    **RTX_5050_DEFAULTS,
    "num_envs": 64,
    "horizon": 1024,
    "strict_course_training": True,
    "gamma": 0.99979992,
    "gae_lambda": 0.99799195,
    "potential_gamma": 0.99979992,
    # Keep exploration elevated even when resuming a consumed schedule.
    "exploration_std_start": 0.35,
    "exploration_std_end": 0.35,
    "exploration_std_floor": 0.25,
    "entropy_coef": 0.001,
    "entropy_coef_end": 0.001,
}

TRAINING_PROFILES = {
    "rtx-5050": RTX_5050_DEFAULTS,
    "rtx-5050-long-credit": RTX_5050_LONG_CREDIT_DEFAULTS,
    "rtx-5050-corner": {
        **RTX_5050_LONG_CREDIT_DEFAULTS,
        "strict_course_training": False,
        "corner_reset_bank": "artifacts/motor_diagnostics/g3_approach_bank.npz",
    },
}


def configure_runtime(device: str, gpu_memory_fraction: float | None = None) -> None:
    if gpu_memory_fraction is not None and not 0 < gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be greater than 0 and at most 1")
    if device == "gpu":
        if sys.platform == "win32":
            raise ValueError(
                "JAX CUDA training requires Linux or WSL2. Native Windows Python "
                "cannot use this GPU backend. See the README GPU setup section."
            )
        # Require CUDA instead of silently falling back to CPU. Explicit CLI device
        # selection also keeps model allocations on the simulator's backend.
        os.environ["JAX_PLATFORMS"] = "cuda"
        if gpu_memory_fraction is not None:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(gpu_memory_fraction)
    elif device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
