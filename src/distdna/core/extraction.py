"""Shared orchestration for public RFFTrace extraction APIs."""

from __future__ import annotations

from typing import Sequence, Tuple

from ..data import EmbeddingDataset, normalize_embeddings, require_disjoint_prompts
from ..dna import DNACollection, RFFTraceExtractor
from ..kernels import median_heuristic


def extract_rfftrace_signatures(
    evaluation: EmbeddingDataset,
    calibration: EmbeddingDataset,
    settings: Sequence[str],
    generations: int,
    rff_dimension: int,
    normalization: str,
    random_seed: int,
    dna_dimension: int | None = None,
    sigma: float | None = None,
    bandwidth_max_pairs: int = 100_000,
) -> Tuple[DNACollection, RFFTraceExtractor, float]:
    require_disjoint_prompts(calibration.prompt_ids, evaluation.prompt_ids)
    if evaluation.feature_dim != calibration.feature_dim:
        raise ValueError("evaluation and calibration embedding dimensions must match")
    if generations <= 0 or generations > evaluation.generation_count:
        raise ValueError(
            f"generations must be in [1, {evaluation.generation_count}], received {generations}"
        )
    unknown = sorted(set(settings).difference(evaluation.setting_ids))
    if unknown:
        raise ValueError(f"unknown decoding settings: {unknown}")
    if not settings:
        raise ValueError("at least one decoding setting is required")
    if len(set(settings)) != len(settings):
        raise ValueError("decoding settings must be unique")

    selected_sigma = sigma
    if selected_sigma is None:
        calibration_values = normalize_embeddings(calibration.embeddings, normalization)
        selected_sigma = median_heuristic(
            calibration_values, max_pairs=bandwidth_max_pairs, seed=random_seed
        )

    extractor = RFFTraceExtractor(
        embedding_dim=evaluation.feature_dim,
        prompt_count=len(evaluation.prompt_ids),
        rff_dimension=rff_dimension,
        sigma=selected_sigma,
        random_seed=random_seed,
        dna_dimension=dna_dimension,
        normalization=normalization,
    )
    signatures = []
    for setting_id in settings:
        signatures.extend(
            extractor.extract_dna(
                evaluation.setting(setting_id, generations),
                evaluation.model_ids,
                setting_id,
            )
        )
    return DNACollection(signatures), extractor, selected_sigma
