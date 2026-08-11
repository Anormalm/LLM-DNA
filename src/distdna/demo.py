"""Deterministic synthetic data for exercising the complete pilot pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .data import EmbeddingDataset


def _make_values(
    rng: np.random.Generator,
    model_centers: np.ndarray,
    setting_noise: Tuple[float, ...],
    n_prompts: int,
    n_generations: int,
) -> np.ndarray:
    n_models, feature_dim = model_centers.shape
    prompt_effects = rng.normal(0.0, 0.35, size=(n_prompts, feature_dim))
    values = np.empty(
        (n_models, len(setting_noise), n_prompts, n_generations, feature_dim),
        dtype=np.float32,
    )
    for model_index in range(n_models):
        for setting_index, noise in enumerate(setting_noise):
            values[model_index, setting_index] = (
                model_centers[model_index, None, None, :]
                + prompt_effects[:, None, :]
                + rng.normal(
                    0.0, noise, size=(n_prompts, n_generations, feature_dim)
                )
            )
    return values


def _balanced_block_signs(rng: np.random.Generator, count: int) -> np.ndarray:
    if count % 4 != 0:
        raise ValueError("diagnostic generation pools must be divisible by four")
    signs = np.empty(count, dtype=np.float64)
    for start in range(0, count, 4):
        block = np.asarray([-1.0, -1.0, 1.0, 1.0])
        rng.shuffle(block)
        signs[start : start + 4] = block
    return signs


def _zero_mean_block_noise(
    rng: np.random.Generator, count: int, feature_dim: int, scale: float
) -> np.ndarray:
    noise = rng.normal(0.0, scale, size=(count, feature_dim))
    for start in range(0, count, 4):
        noise[start : start + 4] -= np.mean(noise[start : start + 4], axis=0)
    return noise


def _make_distributional_values(
    rng: np.random.Generator,
    directions: np.ndarray,
    setting_parameters: Tuple[Tuple[float, float], ...],
    n_prompts: int,
    pool_size: int,
) -> np.ndarray:
    """Create equal-mean, model-specific symmetric response distributions."""

    n_models, feature_dim = directions.shape
    prompt_centers = rng.normal(0.0, 0.4, size=(n_prompts, feature_dim))
    values = np.empty(
        (n_models, len(setting_parameters), n_prompts, pool_size * 2, feature_dim),
        dtype=np.float32,
    )
    for model_index in range(n_models):
        for setting_index, (amplitude, noise_scale) in enumerate(setting_parameters):
            for prompt_index in range(n_prompts):
                for pool_index in range(2):
                    signs = _balanced_block_signs(rng, pool_size)
                    noise = _zero_mean_block_noise(
                        rng, pool_size, feature_dim, noise_scale
                    )
                    samples = (
                        prompt_centers[prompt_index]
                        + amplitude * signs[:, None] * directions[model_index]
                        + noise
                    )
                    start = pool_index * pool_size
                    values[
                        model_index,
                        setting_index,
                        prompt_index,
                        start : start + pool_size,
                    ] = samples
    return values


def create_demo(
    output_dir: str | Path, seed: int = 2027, profile: str = "smoke"
) -> Path:
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"demo directory already exists: {target}")
    target.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    if profile == "smoke":
        model_ids = tuple(f"model_{index}" for index in range(5))
        setting_ids = ("deterministic", "temp_0_7", "temp_1_0")
        model_centers = rng.normal(0.0, 1.0, size=(len(model_ids), 16))
        evaluation = EmbeddingDataset(
            _make_values(rng, model_centers, (0.02, 0.45, 0.8), 12, 32),
            model_ids,
            setting_ids,
            tuple(f"eval_prompt_{index}" for index in range(12)),
        )
        calibration = EmbeddingDataset(
            _make_values(rng, model_centers, (0.02, 0.45, 0.8), 6, 32),
            model_ids,
            setting_ids,
            tuple(f"cal_prompt_{index}" for index in range(6)),
        )
        generation_counts = [1, 4, 16]
        rff_dimensions = [64, 256]
        normalization = "l2"
    elif profile == "distributional":
        model_ids = tuple(f"shape_model_{index}" for index in range(6))
        setting_ids = ("stochastic_low", "stochastic_high")
        raw_directions = rng.normal(size=(8, len(model_ids)))
        directions = np.linalg.qr(raw_directions)[0].T
        setting_parameters = ((0.8, 0.2), (1.2, 0.35))
        evaluation = EmbeddingDataset(
            _make_distributional_values(
                rng, directions, setting_parameters, n_prompts=12, pool_size=32
            ),
            model_ids,
            setting_ids,
            tuple(f"eval_prompt_{index}" for index in range(12)),
        )
        calibration = EmbeddingDataset(
            _make_distributional_values(
                rng, directions, setting_parameters, n_prompts=6, pool_size=32
            ),
            model_ids,
            setting_ids,
            tuple(f"cal_prompt_{index}" for index in range(6)),
        )
        generation_counts = [1, 4, 16, 32]
        rff_dimensions = [64, 256, 1024]
        normalization = "none"
    else:
        raise ValueError(f"unknown demo profile: {profile}")

    evaluation.save(target / "evaluation.npz")
    calibration.save(target / "calibration.npz")
    config: Dict[str, Any] = {
        "data": {
            "evaluation": "evaluation.npz",
            "calibration": "calibration.npz",
        },
        "experiment": {
            "generation_counts": generation_counts,
            "rff_dimensions": rff_dimensions,
            "methods": [
                "single_sample_cosine",
                "mean_dna_cosine",
                "exact_mmd",
                "rfftrace",
            ],
            "normalization": normalization,
            "seed": seed,
            "top_ks": [1, 3, 5],
            "projection_dimension": None,
            "comparisons": [],
            "bandwidth": {
                "strategy": "median",
                "multiplier": 1.0,
                "max_pairs": 100000,
            },
        },
        "output": {"directory": "results", "save_distances": True},
    }
    config_path = target / "pilot.json"
    with config_path.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    return config_path
