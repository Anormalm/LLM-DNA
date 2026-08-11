"""Materialize immutable factorial experiment configuration suites."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Sequence

from .config import ExperimentConfig


def _float_token(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace(".", "p")


def write_ablation_suite(
    base_config_path: str | Path,
    config_dir: str | Path,
    result_root: str | Path,
    normalizations: Sequence[str],
    bandwidth_multipliers: Sequence[float],
) -> Path:
    """Write one fresh config per normalization/bandwidth combination."""

    if not normalizations or any(item not in {"none", "l2"} for item in normalizations):
        raise ValueError("normalizations must contain one or more of: none, l2")
    if len(set(normalizations)) != len(normalizations):
        raise ValueError("normalizations cannot contain duplicates")
    multipliers = tuple(float(item) for item in bandwidth_multipliers)
    if not multipliers or any(
        not math.isfinite(item) or item <= 0 for item in multipliers
    ):
        raise ValueError("bandwidth multipliers must be positive and finite")
    if len(set(multipliers)) != len(multipliers):
        raise ValueError("bandwidth multipliers cannot contain duplicates")

    base_source = Path(base_config_path).resolve()
    base = ExperimentConfig.load(base_source)
    target = Path(config_dir).resolve()
    results = Path(result_root).resolve()
    if target.exists():
        raise FileExistsError(f"suite config directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    members = []
    try:
        for normalization in normalizations:
            for multiplier in multipliers:
                condition = (
                    f"norm-{normalization}__bw-{_float_token(multiplier)}"
                    f"__seed-{base.seed}"
                )
                output_dir = results / condition
                config = replace(
                    base,
                    normalization=normalization,
                    bandwidth=replace(base.bandwidth, multiplier=multiplier),
                    output_dir=output_dir,
                )
                config_path = staging / f"{condition}.json"
                with config_path.open("w", encoding="utf-8") as stream:
                    json.dump(config.as_dict(), stream, indent=2, sort_keys=True)
                    stream.write("\n")
                members.append(
                    {
                        "normalization": normalization,
                        "bandwidth_multiplier": multiplier,
                        "config": str(target / config_path.name),
                        "output_dir": str(output_dir),
                    }
                )
        with (staging / "suite.json").open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "format_version": 1,
                    "base_config": str(base_source),
                    "seed": base.seed,
                    "normalizations": list(normalizations),
                    "bandwidth_multipliers": list(multipliers),
                    "members": members,
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
    return target / "suite.json"


def load_suite(path: str | Path) -> Dict[str, Any]:
    with Path(path).resolve().open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("members"), list):
        raise ValueError("suite must be a JSON object containing a members array")
    return payload
