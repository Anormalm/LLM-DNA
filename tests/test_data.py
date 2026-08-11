from pathlib import Path

import numpy as np
import pytest

from distdna.data import (
    EmbeddingDataset,
    normalize_embeddings,
    require_disjoint_prompts,
)
from distdna.demo import create_demo


def dataset() -> EmbeddingDataset:
    values = np.arange(2 * 2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 2, 3, 4, 5)
    return EmbeddingDataset(
        values,
        ("m0", "m1"),
        ("s0", "s1"),
        ("p0", "p1", "p2"),
    )


def test_round_trip_and_setting_slice(tmp_path: Path) -> None:
    original = dataset()
    path = tmp_path / "dataset.npz"
    original.save(path)
    restored = EmbeddingDataset.load(path)
    np.testing.assert_array_equal(restored.embeddings, original.embeddings)
    assert restored.model_ids == original.model_ids
    assert restored.setting("s1", generations=2).shape == (2, 3, 2, 5)


def test_rejects_non_finite_or_wrong_identifier_lengths() -> None:
    values = np.zeros((1, 1, 1, 1, 2), dtype=np.float32)
    values[..., 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        EmbeddingDataset(values, ("m",), ("s",), ("p",))
    with pytest.raises(ValueError, match="identifier lengths"):
        EmbeddingDataset(np.zeros((1, 1, 1, 1, 2)), ("m", "extra"), ("s",), ("p",))


def test_normalization_and_split_guard() -> None:
    values = np.asarray([[3.0, 4.0]])
    normalized = normalize_embeddings(values, "l2")
    np.testing.assert_allclose(normalized, [[0.6, 0.8]])
    with pytest.raises(ValueError, match="disjoint"):
        require_disjoint_prompts(("p0", "p1"), ("p1", "p2"))


def test_distributional_demo_has_equal_block_means(tmp_path: Path) -> None:
    config_path = create_demo(tmp_path / "diagnostic", profile="distributional")
    generated = EmbeddingDataset.load(config_path.parent / "evaluation.npz")
    assert generated.shape == (6, 2, 12, 64, 8)
    # Every four-sample block has the same prompt center for every model.
    block_means = np.mean(generated.embeddings[:, :, :, :4, :], axis=3)
    np.testing.assert_allclose(
        block_means,
        np.broadcast_to(block_means[:1], block_means.shape),
        atol=1e-6,
    )
