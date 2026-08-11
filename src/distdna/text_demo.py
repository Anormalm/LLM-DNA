"""Controlled text-response diagnostic for the complete collection pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import (
    CollectionManifest,
    DecodingSetting,
    HashingResponseEncoder,
    Prompt,
    ResponseGenerator,
    build_embedding_datasets,
    collect_responses,
    save_embedding_datasets,
)


class CategoricalTextGenerator(ResponseGenerator):
    """Generate model-specific categorical response distributions.

    This generator exists only to verify the response-to-retrieval pipeline. No model
    identifier is written into a response; identity is present only in the distribution
    over shared response styles.
    """

    _STYLES = (
        "analytical",
        "narrative",
        "skeptical",
        "pragmatic",
        "formal",
        "playful",
    )
    _OPENERS = (
        "A concise answer",
        "One useful framing",
        "The central observation",
        "A practical response",
    )

    def __init__(self, manifest: CollectionManifest) -> None:
        self.manifest = manifest

    def generate(
        self,
        model_id: str,
        prompt: Prompt,
        setting: DecodingSetting,
        seed: int,
    ) -> str:
        model_index = self.manifest.model_ids.index(model_id)
        if setting.setting_id == "stochastic_low":
            base = np.asarray([0.58, 0.18, 0.10, 0.06, 0.05, 0.03])
        elif setting.setting_id == "stochastic_high":
            base = np.asarray([0.40, 0.22, 0.14, 0.10, 0.08, 0.06])
        else:
            raise KeyError(f"unsupported diagnostic setting: {setting.setting_id}")
        probabilities = np.roll(base, model_index)
        rng = np.random.default_rng(seed)
        style = self._STYLES[int(rng.choice(len(self._STYLES), p=probabilities))]
        opener = self._OPENERS[int(rng.integers(0, len(self._OPENERS)))]
        return (
            f"{opener} takes a {style} approach. "
            f"For the task '{prompt.text}', it emphasizes a {style} explanation."
        )


def create_text_pipeline_demo(output_dir: str | Path, seed: int = 2027) -> Path:
    """Collect, encode, and configure a controlled end-to-end text pilot."""

    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"text-pipeline demo directory already exists: {target}")
    target.mkdir(parents=True)

    calibration_texts = (
        "Explain why repeated measurements can be useful.",
        "Compare two ways to summarize uncertain observations.",
        "Describe a careful approach to a small empirical study.",
    )
    evaluation_texts = (
        "Explain how to evaluate a noisy system.",
        "Discuss the value of controlled comparisons.",
        "Describe what makes an experiment reproducible.",
        "Compare a point estimate with a distribution.",
        "Explain how randomness can reveal behavior.",
        "Outline a robust model-identification test.",
    )
    prompts = tuple(
        Prompt(f"cal_prompt_{index}", text, "calibration")
        for index, text in enumerate(calibration_texts)
    ) + tuple(
        Prompt(f"eval_prompt_{index}", text, "evaluation")
        for index, text in enumerate(evaluation_texts)
    )
    manifest = CollectionManifest(
        dataset_id="controlled-text-distribution-v1",
        model_ids=tuple(f"latent_model_{index}" for index in range(6)),
        settings=(
            DecodingSetting("stochastic_low", temperature=0.7, top_p=0.9),
            DecodingSetting("stochastic_high", temperature=1.0, top_p=1.0),
        ),
        prompts=prompts,
        generations=32,
        random_seed=seed,
        metadata={
            "purpose": "end-to-end controlled pipeline diagnostic",
            "evidence_status": "synthetic; not a paper result or real-model benchmark",
            "identity_signal": "distribution over shared response styles only",
        },
    )
    manifest.save(target / "collection.json")
    responses = collect_responses(
        manifest,
        CategoricalTextGenerator(manifest),
        target / "responses",
    )
    encoder = HashingResponseEncoder(dimension=128)
    datasets = build_embedding_datasets(manifest, responses, encoder)
    embeddings_dir = save_embedding_datasets(
        datasets, manifest, encoder, target / "embeddings"
    )
    config = {
        "data": {
            "evaluation": str(embeddings_dir / "evaluation.npz"),
            "calibration": str(embeddings_dir / "calibration.npz"),
        },
        "experiment": {
            "generation_counts": [1, 4, 16],
            "rff_dimensions": [64, 256],
            "methods": [
                "single_sample_cosine",
                "mean_dna_cosine",
                "exact_mmd",
                "rfftrace",
            ],
            "normalization": "l2",
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
        "output": {"directory": str(target / "results"), "save_distances": True},
    }
    config_path = target / "pilot.json"
    with config_path.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return config_path

