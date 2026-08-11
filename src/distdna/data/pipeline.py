"""Convert complete text-response caches into RFFTrace embedding tensors."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np

from .dataset import EmbeddingDataset
from .encoders import ResponseEncoder
from .manifest import CollectionManifest
from .responses import ResponseDataset


def build_embedding_datasets(
    manifest: CollectionManifest,
    responses: ResponseDataset,
    encoder: ResponseEncoder,
) -> Dict[str, EmbeddingDataset]:
    if responses.manifest.fingerprint != manifest.fingerprint:
        raise ValueError("response dataset and collection manifest do not match")
    responses.require_complete()
    records = responses.by_key
    result: Dict[str, EmbeddingDataset] = {}
    feature_dimension = None
    for split in ("calibration", "evaluation"):
        split_prompts = manifest.prompts_for_split(split)
        ordered_texts = [
            records[(model_id, setting_id, prompt.prompt_id, generation_index)].response
            for model_id in manifest.model_ids
            for setting_id in manifest.setting_ids
            for prompt in split_prompts
            for generation_index in range(manifest.generations)
        ]
        encoded = np.asarray(encoder.encode(ordered_texts), dtype=np.float32)
        if encoded.ndim != 2 or encoded.shape[0] != len(ordered_texts):
            raise ValueError("response encoder returned an incompatible tensor")
        if not np.isfinite(encoded).all():
            raise ValueError("response encoder returned non-finite embeddings")
        if feature_dimension is None:
            feature_dimension = encoded.shape[1]
        elif feature_dimension != encoded.shape[1]:
            raise ValueError("response encoder dimension changed between prompt splits")
        values = encoded.reshape(
            len(manifest.model_ids),
            len(manifest.settings),
            len(split_prompts),
            manifest.generations,
            encoded.shape[1],
        )
        result[split] = EmbeddingDataset(
            values,
            manifest.model_ids,
            manifest.setting_ids,
            tuple(prompt.prompt_id for prompt in split_prompts),
        )
    return result


def save_embedding_datasets(
    datasets: Dict[str, EmbeddingDataset],
    manifest: CollectionManifest,
    encoder: ResponseEncoder,
    output_dir: str | Path,
) -> Path:
    required = {"calibration", "evaluation"}
    if set(datasets) != required:
        raise ValueError(f"embedding datasets must contain exactly: {sorted(required)}")
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"embedding output directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        datasets["calibration"].save(staging / "calibration.npz")
        datasets["evaluation"].save(staging / "evaluation.npz")
        with (staging / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "format_version": 1,
                    "manifest_fingerprint": manifest.fingerprint,
                    "encoder_id": encoder.encoder_id,
                    "calibration_shape": list(datasets["calibration"].shape),
                    "evaluation_shape": list(datasets["evaluation"].shape),
                    "model_ids": list(manifest.model_ids),
                    "setting_ids": list(manifest.setting_ids),
                    "calibration_prompt_ids": [
                        prompt.prompt_id
                        for prompt in manifest.prompts_for_split("calibration")
                    ],
                    "evaluation_prompt_ids": [
                        prompt.prompt_id
                        for prompt in manifest.prompts_for_split("evaluation")
                    ],
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target

