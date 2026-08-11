"""Optional response-provider adapters."""

from .transformers_local import (
    LocalTransformersGenerator,
    inherit_model_revisions,
    resolve_model_revisions,
)

__all__ = [
    "LocalTransformersGenerator",
    "inherit_model_revisions",
    "resolve_model_revisions",
]
