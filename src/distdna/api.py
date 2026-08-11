"""Public API for programmatic RFFTrace DNA extraction."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .core import extract_rfftrace_signatures
from .data import DatasetLoader
from .dna import DNACollection, DNASignature


@dataclass
class RFFTraceExtractionConfig:
    """Configuration for extracting a shared family of RFFTrace DNA vectors."""

    evaluation_path: Path
    calibration_path: Path
    settings: Optional[Tuple[str, ...]] = None
    generations: int = 16
    rff_dimension: int = 1024
    dna_dimension: Optional[int] = 128
    sigma: Optional[float] = None
    normalization: str = "l2"
    random_seed: int = 42
    bandwidth_max_pairs: int = 100_000
    output_dir: Path = Path("./out/rfftrace")
    save: bool = True


@dataclass(frozen=True)
class RFFTraceExtractionResult:
    """Result payload for a multi-model, multi-setting extraction run."""

    signatures: DNACollection
    output_dir: Optional[Path]
    collection_path: Optional[Path]
    summary_path: Optional[Path]
    parameters_path: Optional[Path]
    sigma: float
    elapsed_seconds: float

    @property
    def vectors(self) -> NDArray[np.float32]:
        return self.signatures.vectors

    def get_signature(self, model_name: str, setting_id: str) -> DNASignature:
        matches = [
            signature
            for signature in self.signatures
            if signature.model_name == model_name and signature.setting_id == setting_id
        ]
        if len(matches) != 1:
            raise KeyError(
                f"expected one DNA signature for model={model_name!r}, setting={setting_id!r}"
            )
        return matches[0]


def _slug(value: str) -> str:
    readable = "".join(character if character.isalnum() else "_" for character in value)[:64]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{readable}_{digest}"


def _config_dict(config: RFFTraceExtractionConfig) -> dict:
    payload = asdict(config)
    payload["evaluation_path"] = str(Path(config.evaluation_path).resolve())
    payload["calibration_path"] = str(Path(config.calibration_path).resolve())
    payload["output_dir"] = str(Path(config.output_dir).resolve())
    payload["settings"] = None if config.settings is None else list(config.settings)
    return payload


def calc_rfftrace(config: RFFTraceExtractionConfig) -> RFFTraceExtractionResult:
    """Compute RFFTrace DNA vectors with one map shared across the whole run."""

    start = time.perf_counter()
    loader = DatasetLoader()
    evaluation = loader.load(config.evaluation_path)
    calibration = loader.load(config.calibration_path)
    settings = evaluation.setting_ids if config.settings is None else tuple(config.settings)
    collection, extractor, sigma = extract_rfftrace_signatures(
        evaluation=evaluation,
        calibration=calibration,
        settings=settings,
        generations=config.generations,
        rff_dimension=config.rff_dimension,
        normalization=config.normalization,
        random_seed=config.random_seed,
        dna_dimension=config.dna_dimension,
        sigma=config.sigma,
        bandwidth_max_pairs=config.bandwidth_max_pairs,
    )

    output_dir: Optional[Path] = None
    collection_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    parameters_path: Optional[Path] = None
    if config.save:
        output_dir = Path(config.output_dir).resolve()
        if output_dir.exists():
            raise FileExistsError(
                f"output directory already exists; choose a fresh path: {output_dir}"
            )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
        )
        try:
            signature_dir = staging / "signatures"
            signature_dir.mkdir()
            for signature in collection:
                signature.save(
                    signature_dir
                    / f"{_slug(signature.model_name)}__{_slug(signature.setting_id)}.npz"
                )
            collection_path = staging / "rfftrace_signatures.npz"
            collection.save(collection_path)
            parameters_path = staging / (
                f"rff_parameters_D{config.rff_dimension}_seed{config.random_seed}.npz"
            )
            extractor.save_parameters(parameters_path)
            summary_path = staging / "summary.json"
            with summary_path.open("w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "format_version": 1,
                        "method": "rfftrace",
                        "shared_feature_map": True,
                        "sigma": sigma,
                        "signature_count": len(collection),
                        "vector_shape": list(collection.vectors.shape),
                        "model_names": list(evaluation.model_ids),
                        "setting_ids": list(settings),
                        "prompt_ids": list(evaluation.prompt_ids),
                        "config": _config_dict(config),
                    },
                    stream,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
            os.replace(staging, output_dir)
            collection_path = output_dir / collection_path.name
            parameters_path = output_dir / parameters_path.name
            summary_path = output_dir / summary_path.name
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    return RFFTraceExtractionResult(
        signatures=collection,
        output_dir=output_dir,
        collection_path=collection_path,
        summary_path=summary_path,
        parameters_path=parameters_path,
        sigma=sigma,
        elapsed_seconds=time.perf_counter() - start,
    )

