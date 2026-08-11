"""Optional response-provider adapters."""

from .transformers_local import LocalTransformersGenerator, resolve_model_revisions

__all__ = ["LocalTransformersGenerator", "resolve_model_revisions"]

