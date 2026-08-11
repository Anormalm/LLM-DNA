"""Strict JSON configuration for pilot experiments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise ValueError(f"unknown {context} keys: {unknown}")


def _positive_int_tuple(value: Any, name: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty JSON array")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must contain JSON integers")
    values = tuple(value)
    if any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} cannot contain duplicates")
    return values


@dataclass(frozen=True)
class ComparisonConfig:
    query: str
    reference: str


@dataclass(frozen=True)
class BandwidthConfig:
    strategy: str = "median"
    value: float | None = None
    multiplier: float = 1.0
    max_pairs: int = 100_000


@dataclass(frozen=True)
class ExperimentConfig:
    evaluation_path: Path
    calibration_path: Path
    output_dir: Path
    generation_counts: Tuple[int, ...]
    rff_dimensions: Tuple[int, ...]
    methods: Tuple[str, ...]
    normalization: str
    seed: int
    top_ks: Tuple[int, ...]
    projection_dimensions: Tuple[int | None, ...]
    comparisons: Tuple[ComparisonConfig, ...]
    bandwidth: BandwidthConfig
    save_distances: bool

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        source = Path(path).resolve()
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("configuration root must be a JSON object")
        return cls.from_dict(payload, base_dir=source.parent)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], base_dir: str | Path = "."
    ) -> "ExperimentConfig":
        _reject_unknown(payload, {"data", "experiment", "output"}, "top-level")
        data = payload.get("data")
        experiment = payload.get("experiment")
        output = payload.get("output")
        if not isinstance(data, dict) or not isinstance(experiment, dict) or not isinstance(output, dict):
            raise ValueError("data, experiment, and output must be JSON objects")
        _reject_unknown(data, {"evaluation", "calibration"}, "data")
        _reject_unknown(
            experiment,
            {
                "generation_counts",
                "rff_dimensions",
                "methods",
                "normalization",
                "seed",
                "top_ks",
                "projection_dimension",
                "projection_dimensions",
                "comparisons",
                "bandwidth",
            },
            "experiment",
        )
        _reject_unknown(output, {"directory", "save_distances"}, "output")

        base = Path(base_dir).resolve()

        def resolve_path(raw: Any, name: str) -> Path:
            if not isinstance(raw, str) or not raw:
                raise ValueError(f"{name} must be a non-empty path string")
            candidate = Path(raw)
            return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()

        generation_counts = _positive_int_tuple(
            experiment.get("generation_counts", [1, 4, 16]), "generation_counts"
        )
        rff_dimensions = _positive_int_tuple(
            experiment.get("rff_dimensions", [256, 1024]), "rff_dimensions"
        )
        top_ks = _positive_int_tuple(experiment.get("top_ks", [1, 3, 5]), "top_ks")
        if 1 not in top_ks:
            raise ValueError("top_ks must include 1 for exact-model identification")

        supported_methods = {
            "single_sample_cosine",
            "mean_dna_cosine",
            "exact_mmd",
            "rfftrace",
        }
        raw_methods = experiment.get("methods", sorted(supported_methods))
        if not isinstance(raw_methods, list) or not raw_methods:
            raise ValueError("methods must be a non-empty JSON array")
        methods = tuple(str(item) for item in raw_methods)
        unsupported = sorted(set(methods).difference(supported_methods))
        if unsupported:
            raise ValueError(f"unsupported methods: {unsupported}")
        if len(set(methods)) != len(methods):
            raise ValueError("methods cannot contain duplicates")

        normalization = str(experiment.get("normalization", "l2"))
        if normalization not in {"none", "l2"}:
            raise ValueError("normalization must be 'none' or 'l2'")
        seed_raw = experiment.get("seed", 2027)
        if not isinstance(seed_raw, int) or isinstance(seed_raw, bool):
            raise ValueError("seed must be a JSON integer")
        seed = seed_raw

        if "projection_dimension" in experiment and "projection_dimensions" in experiment:
            raise ValueError(
                "use either projection_dimension or projection_dimensions, not both"
            )
        if "projection_dimensions" in experiment:
            projection_raw = experiment["projection_dimensions"]
            if not isinstance(projection_raw, list) or not projection_raw:
                raise ValueError(
                    "projection_dimensions must be a non-empty JSON array"
                )
            projection_dimensions = tuple(projection_raw)
        else:
            projection_dimensions = (experiment.get("projection_dimension"),)
        if any(
            value is not None
            and (not isinstance(value, int) or isinstance(value, bool) or value <= 0)
            for value in projection_dimensions
        ):
            raise ValueError(
                "projection dimensions must contain positive JSON integers or null"
            )
        if len(set(projection_dimensions)) != len(projection_dimensions):
            raise ValueError("projection dimensions cannot contain duplicates")

        comparisons_raw = experiment.get("comparisons", [])
        if not isinstance(comparisons_raw, list):
            raise ValueError("comparisons must be a JSON array")
        comparisons = []
        for item in comparisons_raw:
            if not isinstance(item, dict):
                raise ValueError("every comparison must be a JSON object")
            _reject_unknown(item, {"query", "reference"}, "comparison")
            query = item.get("query")
            reference = item.get("reference")
            if not isinstance(query, str) or not isinstance(reference, str):
                raise ValueError("comparison query and reference must be strings")
            comparisons.append(ComparisonConfig(query=query, reference=reference))
        if len(set(comparisons)) != len(comparisons):
            raise ValueError("comparisons cannot contain duplicates")

        bandwidth_raw = experiment.get("bandwidth", {"strategy": "median"})
        if not isinstance(bandwidth_raw, dict):
            raise ValueError("bandwidth must be a JSON object")
        _reject_unknown(
            bandwidth_raw, {"strategy", "value", "multiplier", "max_pairs"}, "bandwidth"
        )
        strategy = str(bandwidth_raw.get("strategy", "median"))
        if strategy not in {"median", "fixed"}:
            raise ValueError("bandwidth strategy must be 'median' or 'fixed'")
        value_raw = bandwidth_raw.get("value")
        value = None if value_raw is None else float(value_raw)
        multiplier = float(bandwidth_raw.get("multiplier", 1.0))
        max_pairs_raw = bandwidth_raw.get("max_pairs", 100_000)
        if not isinstance(max_pairs_raw, int) or isinstance(max_pairs_raw, bool):
            raise ValueError("bandwidth max_pairs must be a JSON integer")
        max_pairs = max_pairs_raw
        if strategy == "fixed" and value is None:
            raise ValueError("fixed bandwidth requires a value")
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("bandwidth value must be positive and finite")
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("bandwidth multiplier must be positive and finite")
        if max_pairs <= 0:
            raise ValueError("bandwidth max_pairs must be positive")
        bandwidth = BandwidthConfig(strategy, value, multiplier, max_pairs)

        save_distances = output.get("save_distances", True)
        if not isinstance(save_distances, bool):
            raise ValueError("output.save_distances must be a JSON boolean")

        return cls(
            evaluation_path=resolve_path(data.get("evaluation"), "data.evaluation"),
            calibration_path=resolve_path(data.get("calibration"), "data.calibration"),
            output_dir=resolve_path(output.get("directory"), "output.directory"),
            generation_counts=generation_counts,
            rff_dimensions=rff_dimensions,
            methods=methods,
            normalization=normalization,
            seed=seed,
            top_ks=top_ks,
            projection_dimensions=projection_dimensions,
            comparisons=tuple(comparisons),
            bandwidth=bandwidth,
            save_distances=save_distances,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "data": {
                "evaluation": str(self.evaluation_path),
                "calibration": str(self.calibration_path),
            },
            "experiment": {
                "generation_counts": list(self.generation_counts),
                "rff_dimensions": list(self.rff_dimensions),
                "methods": list(self.methods),
                "normalization": self.normalization,
                "seed": self.seed,
                "top_ks": list(self.top_ks),
                **(
                    {"projection_dimension": self.projection_dimensions[0]}
                    if len(self.projection_dimensions) == 1
                    else {"projection_dimensions": list(self.projection_dimensions)}
                ),
                "comparisons": [
                    {"query": item.query, "reference": item.reference}
                    for item in self.comparisons
                ],
                "bandwidth": {
                    "strategy": self.bandwidth.strategy,
                    "value": self.bandwidth.value,
                    "multiplier": self.bandwidth.multiplier,
                    "max_pairs": self.bandwidth.max_pairs,
                },
            },
            "output": {
                "directory": str(self.output_dir),
                "save_distances": self.save_distances,
            },
        }

    @property
    def projection_dimension(self) -> int | None:
        """Backward-compatible singular value for non-sweep configurations."""

        return (
            self.projection_dimensions[0]
            if len(self.projection_dimensions) == 1
            else None
        )
