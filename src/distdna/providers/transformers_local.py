"""Sequential local Hugging Face generation for reproducible pilot collections."""

from __future__ import annotations

import gc
import re
import time
from dataclasses import replace
from typing import Any, Callable, Dict, Mapping

from ..data import (
    CollectionManifest,
    DecodingSetting,
    GeneratedResponse,
    Prompt,
)

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _stop_reason(
    generated_tokens: int, max_new_tokens: int, last_token: int | None, eos_token_id: Any
) -> str:
    if eos_token_id is not None and last_token is not None:
        eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)
        if last_token in eos_ids:
            return "eos_token"
    return "max_new_tokens" if generated_tokens >= max_new_tokens else "other"


def _clear_inherited_max_length(model: Any) -> None:
    """Let the manifest's ``max_new_tokens`` be the sole length constraint.

    Some chat checkpoints persist ``max_length`` in their generation config.
    Transformers warns on every call when that inherited value is combined with
    the explicit ``max_new_tokens`` supplied by this provider, even though the
    latter takes precedence. Clearing the unused inherited value keeps the
    generation contract unambiguous and collection logs quiet.
    """

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.max_length = None


def resolve_model_revisions(
    manifest: CollectionManifest, api: Any | None = None
) -> CollectionManifest:
    """Resolve every Hub model revision to an immutable commit SHA."""

    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is required; install distdna[local]"
            ) from exc
        api = HfApi()
    configured = manifest.metadata.get("model_revisions", {})
    if not isinstance(configured, dict):
        raise ValueError("manifest metadata.model_revisions must be a JSON object")
    unknown = sorted(set(configured).difference(manifest.model_ids))
    if unknown:
        raise ValueError(f"model_revisions contains unknown model IDs: {unknown}")

    resolved: Dict[str, str] = {}
    for model_id in manifest.model_ids:
        requested = configured.get(model_id, "main")
        if not isinstance(requested, str) or not requested:
            raise ValueError(f"invalid requested revision for model: {model_id}")
        try:
            info = api.model_info(repo_id=model_id, revision=requested)
        except Exception as exc:
            raise RuntimeError(
                f"failed to resolve Hugging Face revision for {model_id}: {exc}"
            ) from exc
        sha = getattr(info, "sha", None)
        if not isinstance(sha, str) or not _COMMIT_PATTERN.fullmatch(sha):
            raise ValueError(f"Hub returned an invalid commit SHA for model: {model_id}")
        resolved[model_id] = sha

    metadata = dict(manifest.metadata)
    metadata["model_revisions"] = resolved
    metadata["revision_policy"] = "immutable Hugging Face commit SHA"
    return replace(manifest, metadata=metadata)


def inherit_model_revisions(
    manifest: CollectionManifest, source: CollectionManifest
) -> CollectionManifest:
    """Reuse immutable model commits from a compatible resolved manifest."""

    revisions = source.metadata.get("model_revisions")
    if not isinstance(revisions, dict):
        raise ValueError("source manifest has no pinned model_revisions")
    if set(revisions) != set(source.model_ids):
        raise ValueError("source model_revisions must contain exactly its model IDs")
    if set(manifest.model_ids) != set(source.model_ids):
        raise ValueError("target and source manifests must contain the same model IDs")
    invalid = sorted(
        model_id
        for model_id in manifest.model_ids
        if not isinstance(revisions.get(model_id), str)
        or not _COMMIT_PATTERN.fullmatch(revisions[model_id])
    )
    if invalid:
        raise ValueError(f"source manifest contains invalid commit SHAs: {invalid}")
    metadata = dict(manifest.metadata)
    metadata["model_revisions"] = {
        model_id: revisions[model_id] for model_id in manifest.model_ids
    }
    metadata["revision_policy"] = "immutable Hugging Face commit SHA"
    metadata["revision_source_manifest_fingerprint"] = source.fingerprint
    return replace(manifest, metadata=metadata)


class LocalTransformersGenerator:
    """Load one model at a time and generate through its native chat template."""

    def __init__(
        self,
        manifest: CollectionManifest,
        device: str = "auto",
        dtype: str = "float16",
        local_files_only: bool = False,
        status: Callable[[str], None] | None = None,
    ) -> None:
        revisions = manifest.metadata.get("model_revisions")
        if not isinstance(revisions, dict):
            raise ValueError(
                "manifest must contain metadata.model_revisions; run resolve-models first"
            )
        if set(revisions) != set(manifest.model_ids):
            raise ValueError("model_revisions must contain exactly every manifest model ID")
        invalid = sorted(
            model_id
            for model_id, revision in revisions.items()
            if not isinstance(revision, str)
            or not _COMMIT_PATTERN.fullmatch(revision)
        )
        if invalid:
            raise ValueError(f"model revisions must be immutable commit SHAs: {invalid}")
        if dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty string")

        self.manifest = manifest
        self.revisions: Mapping[str, str] = dict(revisions)
        self.requested_device = device
        self.dtype_name = dtype
        self.local_files_only = local_files_only
        self.status = status
        self.system_prompt = manifest.metadata.get(
            "system_prompt", "You are a helpful assistant."
        )
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("metadata.system_prompt must be a non-empty string")
        self._torch = None
        self._transformers = None
        self._device = None
        self._model_id = None
        self._model = None
        self._tokenizer = None
        self._resolved_commit = None

    def _load_runtime(self) -> None:
        if self._torch is not None:
            return
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required; install distdna[local]"
            ) from exc
        if self.requested_device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        else:
            device = self.requested_device
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this PyTorch build")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch build")
        self._torch = torch
        self._transformers = transformers
        self._device = device

    @property
    def device(self) -> str:
        self._load_runtime()
        assert self._device is not None
        return self._device

    def _release_model(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_id = None
        self._resolved_commit = None
        gc.collect()
        if self._torch is not None:
            if self._device == "mps" and hasattr(self._torch, "mps"):
                self._torch.mps.empty_cache()
            elif self._device == "cuda":
                self._torch.cuda.empty_cache()

    def close(self) -> None:
        self._release_model()

    def _ensure_model(self, model_id: str) -> None:
        if self._model_id == model_id:
            return
        if model_id not in self.revisions:
            raise KeyError(f"model is absent from pinned revisions: {model_id}")
        self._load_runtime()
        self._release_model()
        assert self._torch is not None and self._transformers is not None
        revision = self.revisions[model_id]
        if self.status is not None:
            self.status(f"loading {model_id}@{revision} on {self.device}")
        try:
            tokenizer = self._transformers.AutoTokenizer.from_pretrained(
                model_id,
                revision=revision,
                local_files_only=self.local_files_only,
                trust_remote_code=False,
            )
            if not getattr(tokenizer, "chat_template", None):
                raise ValueError(f"model tokenizer has no chat template: {model_id}")
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            dtype = getattr(self._torch, self.dtype_name)
            model = self._transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                dtype=dtype,
                local_files_only=self.local_files_only,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
            model.to(self.device)
            model.eval()
            _clear_inherited_max_length(model)
        except Exception as exc:
            self._release_model()
            raise RuntimeError(f"failed to load local model {model_id}: {exc}") from exc
        resolved_commit = getattr(model.config, "_commit_hash", None)
        if resolved_commit is not None and resolved_commit != revision:
            raise RuntimeError(
                f"loaded model revision differs from manifest for {model_id}: "
                f"{resolved_commit} != {revision}"
            )
        self._tokenizer = tokenizer
        self._model = model
        self._model_id = model_id
        self._resolved_commit = revision
        if self.status is not None:
            self.status(f"ready {model_id}@{revision}")

    def _seed_runtime(self, seed: int) -> None:
        assert self._torch is not None
        self._torch.manual_seed(seed)
        if self._device == "mps" and hasattr(self._torch, "mps"):
            self._torch.mps.manual_seed(seed)
        elif self._device == "cuda":
            self._torch.cuda.manual_seed_all(seed)

    def generate(
        self,
        model_id: str,
        prompt: Prompt,
        setting: DecodingSetting,
        seed: int,
    ) -> GeneratedResponse:
        self._ensure_model(model_id)
        assert self._torch is not None
        assert self._transformers is not None
        assert self._model is not None and self._tokenizer is not None
        self._seed_runtime(seed)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt.text},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": setting.max_new_tokens,
            "do_sample": setting.temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "use_cache": True,
        }
        if setting.temperature > 0:
            generation_kwargs.update(
                {"temperature": setting.temperature, "top_p": setting.top_p}
            )
        started = time.perf_counter()
        with self._torch.inference_mode():
            output = self._model.generate(**inputs, **generation_kwargs)
        elapsed_seconds = time.perf_counter() - started
        generated_tokens = int(output.shape[-1] - prompt_tokens)
        last_token = int(output[0, -1].item()) if generated_tokens else None
        text = self._tokenizer.decode(
            output[0, prompt_tokens:], skip_special_tokens=True
        ).strip()
        if not text:
            raise ValueError(f"model produced an empty response: {model_id}")
        return GeneratedResponse(
            text=text,
            metadata={
                "provider": "transformers-local",
                "model_revision": self._resolved_commit,
                "device": self.device,
                "dtype": self.dtype_name,
                "torch_version": self._torch.__version__,
                "transformers_version": self._transformers.__version__,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "elapsed_seconds": elapsed_seconds,
                "stop_reason": _stop_reason(
                    generated_tokens,
                    setting.max_new_tokens,
                    last_token,
                    self._tokenizer.eos_token_id,
                ),
            },
        )
