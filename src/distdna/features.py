"""Shared Random Fourier Features and optional compact DNA projection."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class RFFTrace:
    """A fixed, shareable RFFTrace feature map.

    Independent RNG streams make the first ``D`` frequencies and phases stable in
    a dimension sweep. The map is fixed at construction and reused across models,
    prompts, decoding settings, and query/reference roles.
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int,
        sigma: float,
        n_prompts: int,
        seed: int,
        projection_dim: int | None = None,
        sample_chunk_size: int = 16_384,
    ) -> None:
        if input_dim <= 0 or n_features <= 0 or n_prompts <= 0:
            raise ValueError("input_dim, n_features, and n_prompts must be positive")
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be a positive finite number")
        if projection_dim is not None and projection_dim <= 0:
            raise ValueError("projection_dim must be positive when provided")
        if sample_chunk_size <= 0:
            raise ValueError("sample_chunk_size must be positive")

        self.input_dim = int(input_dim)
        self.n_features = int(n_features)
        self.sigma = float(sigma)
        self.n_prompts = int(n_prompts)
        self.seed = int(seed)
        self.projection_dim = projection_dim
        self.sample_chunk_size = int(sample_chunk_size)

        frequency_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        phase_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        # Store features first so a D sweep uses nested prefixes.
        self.frequencies = frequency_rng.normal(
            loc=0.0, scale=1.0 / sigma, size=(n_features, input_dim)
        )
        self.phases = phase_rng.uniform(0.0, 2.0 * np.pi, size=n_features)

        self.projection: NDArray[np.float64] | None = None
        if projection_dim is not None:
            projection_rng = np.random.default_rng(
                np.random.SeedSequence([seed, 2, n_features, n_prompts])
            )
            self.projection = projection_rng.normal(
                loc=0.0,
                scale=1.0 / math.sqrt(projection_dim),
                size=(projection_dim, n_prompts * n_features),
            )

    def _mean_features(self, samples: NDArray[np.floating]) -> NDArray[np.float64]:
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 4:
            raise ValueError("samples must have shape [models, prompts, generations, features]")
        if values.shape[1] != self.n_prompts or values.shape[-1] != self.input_dim:
            raise ValueError(
                "sample shape is incompatible with the fixed RFF map: "
                f"expected prompts={self.n_prompts}, features={self.input_dim}"
            )
        if values.shape[2] == 0:
            raise ValueError("at least one generation is required")

        groups = values.reshape(-1, values.shape[2], self.input_dim)
        output = np.empty((groups.shape[0], self.n_features), dtype=np.float64)
        groups_per_chunk = max(1, self.sample_chunk_size // values.shape[2])
        scale = math.sqrt(2.0 / self.n_features)
        for start in range(0, groups.shape[0], groups_per_chunk):
            stop = min(start + groups_per_chunk, groups.shape[0])
            block = groups[start:stop]
            angles = np.einsum(
                "...d,kd->...k", block, self.frequencies, optimize=False
            )
            mapped = np.cos(angles + self.phases) * scale
            output[start:stop] = np.mean(mapped, axis=1)
        return output.reshape(values.shape[0], self.n_prompts, self.n_features)

    def transform(self, samples: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return one explicit vector per model."""

        means = self._mean_features(samples)
        vectors = means.reshape(means.shape[0], -1) / math.sqrt(self.n_prompts)
        if self.projection is not None:
            vectors = np.einsum(
                "ij,kj->ik", vectors, self.projection, optimize=False
            )
        return vectors

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        projection = (
            self.projection
            if self.projection is not None
            else np.empty((0, self.n_prompts * self.n_features), dtype=np.float64)
        )
        np.savez_compressed(
            target,
            frequencies=self.frequencies,
            phases=self.phases,
            projection=projection,
            input_dim=np.asarray(self.input_dim),
            n_features=np.asarray(self.n_features),
            sigma=np.asarray(self.sigma),
            n_prompts=np.asarray(self.n_prompts),
            seed=np.asarray(self.seed),
            projection_dim=np.asarray(-1 if self.projection_dim is None else self.projection_dim),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RFFTrace":
        source = Path(path)
        with np.load(source, allow_pickle=False) as payload:
            projection_dim_value = int(payload["projection_dim"])
            instance = cls(
                input_dim=int(payload["input_dim"]),
                n_features=int(payload["n_features"]),
                sigma=float(payload["sigma"]),
                n_prompts=int(payload["n_prompts"]),
                seed=int(payload["seed"]),
                projection_dim=None if projection_dim_value < 0 else projection_dim_value,
            )
            frequencies = np.asarray(payload["frequencies"], dtype=np.float64)
            phases = np.asarray(payload["phases"], dtype=np.float64)
            projection = np.asarray(payload["projection"], dtype=np.float64)
        if frequencies.shape != (instance.n_features, instance.input_dim):
            raise ValueError("saved frequencies have an invalid shape")
        if phases.shape != (instance.n_features,):
            raise ValueError("saved phases have an invalid shape")
        instance.frequencies = frequencies
        instance.phases = phases
        if instance.projection_dim is None:
            if projection.shape[0] != 0:
                raise ValueError("unexpected projection in unprojected RFF map")
            instance.projection = None
        else:
            expected = (instance.projection_dim, instance.n_prompts * instance.n_features)
            if projection.shape != expected:
                raise ValueError("saved projection has an invalid shape")
            instance.projection = projection
        return instance


def pairwise_squared_euclidean(
    query: NDArray[np.floating], reference: NDArray[np.floating]
) -> NDArray[np.float64]:
    query_values = np.asarray(query, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    if query_values.ndim != 2 or reference_values.ndim != 2:
        raise ValueError("query and reference vectors must be two-dimensional")
    if query_values.shape[1] != reference_values.shape[1]:
        raise ValueError("query and reference vectors must share a feature dimension")
    distances = (
        np.sum(query_values * query_values, axis=1)[:, None]
        + np.sum(reference_values * reference_values, axis=1)[None, :]
        - 2.0
        * np.einsum("id,jd->ij", query_values, reference_values, optimize=False)
    )
    return np.maximum(distances, 0.0)
