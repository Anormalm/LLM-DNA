"""Reproducible publication figure rendering from immutable experiment reports."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


_COLORS = {
    "single_sample_cosine": "#7A7A7A",
    "mean_dna_cosine": "#009E73",
    "exact_mmd": "#0072B2",
    "rfftrace": "#D55E00",
}
_LABELS = {
    "single_sample_cosine": "Single sample",
    "mean_dna_cosine": "Mean DNA",
    "exact_mmd": "Exact MMD",
    "rfftrace": "RFFTrace",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: str | Path) -> tuple[Path, Dict[str, Any]]:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"figure source must be a JSON object: {source}")
    return source, payload


def _protocol_row(row: Mapping[str, Any]) -> bool:
    return (
        row.get("normalization", "l2") == "l2"
        and np.isclose(float(row.get("bandwidth_multiplier", 1.0)), 1.0)
    )


def _configure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "figure rendering requires the optional dependency: pip install 'distdna[figures]'"
        ) from exc
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def _save(fig, output_dir: Path, stem: str, formats: Sequence[str]) -> List[Path]:
    written = []
    for format_name in formats:
        path = output_dir / f"{stem}.{format_name}"
        fig.savefig(
            path,
            format=format_name,
            bbox_inches="tight",
            metadata={"Creator": "DistDNA reproducible figure pipeline"},
        )
        written.append(path)
    return written


def _retrieval_scaling(plt, aggregate: Mapping[str, Any]):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, constrained_layout=True)
    for axis, comparison in zip(axes, ("same_setting", "cross_setting")):
        for method in _LABELS:
            rows = [
                row
                for row in aggregate["retrieval"]
                if _protocol_row(row)
                and row["comparison_type"] == comparison
                and row["method"] == method
                and (
                    method != "rfftrace"
                    or (
                        row.get("rff_dimension") == 512
                        and row.get("projection_dimension") is None
                    )
                )
            ]
            if not rows and method == "rfftrace":
                candidates = [
                    row
                    for row in aggregate["retrieval"]
                    if _protocol_row(row)
                    and row["comparison_type"] == comparison
                    and row["method"] == method
                    and row.get("projection_dimension") is None
                ]
                if candidates:
                    dimension = max(int(row["rff_dimension"]) for row in candidates)
                    rows = [row for row in candidates if row["rff_dimension"] == dimension]
            rows.sort(key=lambda row: int(row["generations"]))
            if not rows:
                continue
            x = [row["generations"] for row in rows]
            y = [row["top_1_mean"] for row in rows]
            error = [row["top_1_std"] for row in rows]
            axis.errorbar(
                x,
                y,
                yerr=error,
                marker="o",
                linewidth=1.6,
                capsize=2,
                color=_COLORS[method],
                label=_LABELS[method],
            )
        axis.set_title("Same setting" if comparison == "same_setting" else "Cross setting")
        axis.set_xlabel("Generations per prompt (R)")
        axis.set_ylim(0.0, 1.03)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Top-1 identity retrieval")
    axes[1].legend(frameon=False, loc="lower right")
    return fig


def _approximation_scaling(plt, aggregate: Mapping[str, Any]):
    fig, axis = plt.subplots(figsize=(4.3, 3.1), constrained_layout=True)
    for comparison, color, marker in (
        ("same_setting", "#0072B2", "o"),
        ("cross_setting", "#D55E00", "s"),
    ):
        rows = [
            row
            for row in aggregate["rff_approximation"]
            if _protocol_row(row)
            and row["comparison_type"] == comparison
            and row["generations"] == 4
            and row.get("projection_dimension") is None
        ]
        rows.sort(key=lambda row: int(row["rff_dimension"]))
        axis.errorbar(
            [row["rff_dimension"] for row in rows],
            [row["distance_correlation_mean"] for row in rows],
            yerr=[row["distance_correlation_std"] for row in rows],
            marker=marker,
            linewidth=1.6,
            capsize=2,
            color=color,
            label="Same setting" if comparison == "same_setting" else "Cross setting",
        )
    axis.set_xscale("log", base=2)
    axis.set_ylim(0.75, 1.01)
    axis.set_xlabel("Random Fourier features (D)")
    axis.set_ylabel("Correlation with exact MMD")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    return fig


def _projection_scaling(plt, aggregate: Mapping[str, Any]):
    fig, axis = plt.subplots(figsize=(4.3, 3.1), constrained_layout=True)
    for comparison, color, marker in (
        ("same_setting", "#0072B2", "o"),
        ("cross_setting", "#D55E00", "s"),
    ):
        rows = [
            row
            for row in aggregate["projection_approximation"]
            if _protocol_row(row)
            and row["comparison_type"] == comparison
            and row["generations"] == 4
            and row["rff_dimension"] == 512
        ]
        rows.sort(key=lambda row: int(row["projection_dimension"]))
        axis.errorbar(
            [row["projection_dimension"] for row in rows],
            [row["distance_correlation_mean"] for row in rows],
            yerr=[row["distance_correlation_std"] for row in rows],
            marker=marker,
            linewidth=1.6,
            capsize=2,
            color=color,
            label="Same setting" if comparison == "same_setting" else "Cross setting",
        )
    axis.set_xscale("log", base=2)
    axis.set_ylim(0.5, 1.01)
    axis.set_xlabel("Projected DNA dimension (L)")
    axis.set_ylabel("Correlation with unprojected RFFTrace")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    return fig


def _bandwidth_ablation(plt, aggregate: Mapping[str, Any]):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, constrained_layout=True)
    width = 0.34
    multipliers = (0.5, 1.0, 2.0)
    for axis, comparison in zip(axes, ("same_setting", "cross_setting")):
        for offset, method in ((-width / 2, "exact_mmd"), (width / 2, "rfftrace")):
            rows = []
            for multiplier in multipliers:
                matches = [
                    row
                    for row in aggregate["retrieval"]
                    if row["normalization"] == "l2"
                    and np.isclose(row["bandwidth_multiplier"], multiplier)
                    and row["comparison_type"] == comparison
                    and row["generations"] == 4
                    and row["method"] == method
                    and (
                        method != "rfftrace"
                        or (
                            row["rff_dimension"] == 512
                            and row["projection_dimension"] is None
                        )
                    )
                ]
                if len(matches) != 1:
                    raise ValueError("bandwidth figure requires exactly one row per protocol cell")
                rows.append(matches[0])
            positions = np.arange(len(multipliers)) + offset
            axis.bar(
                positions,
                [row["top_1_mean"] for row in rows],
                width,
                yerr=[row["top_1_std"] for row in rows],
                capsize=2,
                color=_COLORS[method],
                label=_LABELS[method],
            )
        axis.set_xticks(np.arange(len(multipliers)), [str(value) for value in multipliers])
        axis.set_title("Same setting" if comparison == "same_setting" else "Cross setting")
        axis.set_xlabel("Median-bandwidth multiplier")
        axis.set_ylim(0.0, 1.03)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Top-1 identity retrieval")
    axes[1].legend(frameon=False, loc="lower left")
    return fig


def _decoding_heatmaps(plt, report: Mapping[str, Any]):
    temperatures = sorted(
        {float(row["temperature"]) for row in report["cells"] if row["temperature"] > 0}
    )
    top_ps = sorted(
        {float(row["top_p"]) for row in report["cells"] if row["temperature"] > 0}
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=True)
    image = None
    for row_index, method in enumerate(("exact_mmd", "rfftrace")):
        for column_index, evaluation in enumerate(("same_setting", "cross_to_deterministic")):
            axis = axes[row_index, column_index]
            grid = np.full((len(temperatures), len(top_ps)), np.nan)
            for cell in report["cells"]:
                if (
                    cell["method"] == method
                    and cell["evaluation"] == evaluation
                    and cell["temperature"] > 0
                ):
                    i = temperatures.index(float(cell["temperature"]))
                    j = top_ps.index(float(cell["top_p"]))
                    grid[i, j] = float(cell["top_1_mean"])
            if np.any(~np.isfinite(grid)):
                raise ValueError("decoding heatmap source has an incomplete factorial grid")
            image = axis.imshow(grid, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
            for i in range(len(temperatures)):
                for j in range(len(top_ps)):
                    axis.text(
                        j,
                        i,
                        f"{grid[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if grid[i, j] < 0.65 else "black",
                        fontsize=8,
                    )
            axis.set_xticks(range(len(top_ps)), [str(value) for value in top_ps])
            axis.set_yticks(range(len(temperatures)), [str(value) for value in temperatures])
            axis.set_xlabel("Top-p")
            axis.set_ylabel("Temperature")
            relation = "same" if evaluation == "same_setting" else "cross"
            axis.set_title(f"{_LABELS[method]} — {relation}")
    assert image is not None
    fig.colorbar(image, ax=axes, label="Top-1 identity retrieval", shrink=0.84)
    return fig


def _relationship_summary(plt, report: Mapping[str, Any]):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    methods = list(_LABELS)
    x = np.arange(len(methods))
    width = 0.34
    for axis, metric, label in zip(
        axes,
        ("roc_auc", "nearest_group_accuracy"),
        ("Family-pair AUROC", "Nearest-family accuracy"),
    ):
        for offset, evaluation, color in (
            (-width / 2, "same_setting", "#0072B2"),
            (width / 2, "cross_to_deterministic", "#D55E00"),
        ):
            rows = []
            for method in methods:
                matches = [
                    row
                    for row in report["summary"]
                    if row["method"] == method and row["evaluation"] == evaluation
                ]
                if len(matches) != 1:
                    raise ValueError("relationship figure requires one summary row per cell")
                rows.append(matches[0])
            axis.bar(
                x + offset,
                [row[f"{metric}_mean"] for row in rows],
                width,
                yerr=[row[f"{metric}_std"] for row in rows],
                capsize=2,
                color=color,
                label="Same setting" if evaluation == "same_setting" else "Cross setting",
            )
        axis.set_xticks(x, [_LABELS[item] for item in methods], rotation=20, ha="right")
        axis.set_ylim(0.0, 1.03)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
        axis.axhline(
            float(report["chance_baselines"][metric]),
            color="#333333",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )
    axes[1].legend(frameon=False, loc="lower right")
    return fig


def render_figures(
    output_dir: str | Path,
    *,
    retrieval_aggregate_path: str | Path,
    feature_aggregate_path: str | Path,
    projection_aggregate_path: str | Path,
    factorial_aggregate_path: str | Path,
    decoding_report_path: str | Path,
    relationship_report_path: str | Path,
    formats: Sequence[str] = ("svg", "pdf", "png"),
) -> Path:
    allowed_formats = {"svg", "pdf", "png"}
    formats = tuple(dict.fromkeys(formats))
    if not formats or set(formats).difference(allowed_formats):
        raise ValueError("figure formats must be a non-empty subset of svg, pdf, and png")
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"figure output already exists; choose a fresh path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    sources = {}
    try:
        retrieval_path, retrieval = _load(retrieval_aggregate_path)
        feature_path, feature = _load(feature_aggregate_path)
        projection_path, projection = _load(projection_aggregate_path)
        factorial_path, factorial = _load(factorial_aggregate_path)
        decoding_path, decoding = _load(decoding_report_path)
        relationship_path, relationship = _load(relationship_report_path)
        sources = {
            "retrieval_aggregate": retrieval_path,
            "feature_aggregate": feature_path,
            "projection_aggregate": projection_path,
            "factorial_aggregate": factorial_path,
            "decoding_report": decoding_path,
            "relationship_report": relationship_path,
        }
        plt = _configure_matplotlib()
        figures = (
            ("identity-retrieval-scaling", _retrieval_scaling(plt, retrieval)),
            ("rff-approximation-scaling", _approximation_scaling(plt, feature)),
            ("projection-approximation-scaling", _projection_scaling(plt, projection)),
            ("bandwidth-ablation", _bandwidth_ablation(plt, factorial)),
            ("decoding-factorial", _decoding_heatmaps(plt, decoding)),
            ("relationship-recovery", _relationship_summary(plt, relationship)),
        )
        artifacts = []
        for stem, figure in figures:
            try:
                artifacts.extend(_save(figure, staging, stem, formats))
            finally:
                plt.close(figure)
        manifest = {
            "format_version": 1,
            "sources": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in sources.items()
            },
            "artifacts": [
                {
                    "filename": path.name,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(artifacts)
            ],
        }
        with (staging / "figure-manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target
