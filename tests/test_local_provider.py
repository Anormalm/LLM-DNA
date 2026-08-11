from dataclasses import replace
from types import SimpleNamespace

import pytest

from distdna.data import CollectionManifest, DecodingSetting, Prompt
from distdna.providers import LocalTransformersGenerator, resolve_model_revisions
from distdna.providers.transformers_local import _clear_inherited_max_length


def manifest() -> CollectionManifest:
    return CollectionManifest(
        dataset_id="provider-test",
        model_ids=("m0", "m1"),
        settings=(DecodingSetting("sample", 0.7, 0.9),),
        prompts=(
            Prompt("c0", "calibration", "calibration"),
            Prompt("e0", "evaluation", "evaluation"),
        ),
        generations=2,
    )


class FakeInfo:
    def __init__(self, sha: str) -> None:
        self.sha = sha


class FakeHubApi:
    def __init__(self) -> None:
        self.calls = []

    def model_info(self, repo_id: str, revision: str) -> FakeInfo:
        self.calls.append((repo_id, revision))
        digit = str(len(self.calls))
        return FakeInfo(digit * 40)


def test_model_resolution_pins_every_model_without_loading_runtime() -> None:
    source = manifest()
    api = FakeHubApi()
    resolved = resolve_model_revisions(source, api=api)
    assert api.calls == [("m0", "main"), ("m1", "main")]
    assert resolved.metadata["model_revisions"] == {
        "m0": "1" * 40,
        "m1": "2" * 40,
    }
    assert resolved.fingerprint != source.fingerprint
    generator = LocalTransformersGenerator(resolved, device="auto")
    assert generator.requested_device == "auto"


def test_local_generator_rejects_unpinned_or_partial_revisions() -> None:
    source = manifest()
    with pytest.raises(ValueError, match="resolve-models"):
        LocalTransformersGenerator(source)
    partial_metadata = dict(source.metadata)
    partial_metadata["model_revisions"] = {"m0": "a" * 40}
    partial = replace(source, metadata=partial_metadata)
    with pytest.raises(ValueError, match="exactly every"):
        LocalTransformersGenerator(partial)


def test_model_resolution_rejects_unknown_revision_entries() -> None:
    source = manifest()
    metadata = dict(source.metadata)
    metadata["model_revisions"] = {"not-a-model": "main"}
    changed: CollectionManifest = replace(source, metadata=metadata)
    with pytest.raises(ValueError, match="unknown model IDs"):
        resolve_model_revisions(changed, api=FakeHubApi())


def test_local_generator_clears_inherited_max_length() -> None:
    model = SimpleNamespace(
        generation_config=SimpleNamespace(max_length=2048, max_new_tokens=None)
    )

    _clear_inherited_max_length(model)

    assert model.generation_config.max_length is None
    assert model.generation_config.max_new_tokens is None
