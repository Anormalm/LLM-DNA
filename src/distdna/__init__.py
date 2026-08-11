"""Distributional representations for stochastic language-model behavior."""

from .api import RFFTraceExtractionConfig, RFFTraceExtractionResult, calc_rfftrace
from .data import (
    CollectionManifest,
    DecodingSetting,
    EmbeddingDataset,
    GeneratedResponse,
    HashingResponseEncoder,
    Prompt,
    ResponseCache,
    ResponseRecord,
    collect_responses,
)
from .dna import DNACollection, DNAMetadata, DNASignature, RFFTraceExtractor
from .features import RFFTrace
from .kernels import exact_mmd2, exact_mmd_distance_matrix, median_heuristic
from .metrics import RetrievalResult, identity_retrieval

__all__ = [
    "EmbeddingDataset",
    "CollectionManifest",
    "DecodingSetting",
    "HashingResponseEncoder",
    "GeneratedResponse",
    "Prompt",
    "ResponseCache",
    "ResponseRecord",
    "DNACollection",
    "DNAMetadata",
    "DNASignature",
    "RFFTrace",
    "RFFTraceExtractionConfig",
    "RFFTraceExtractionResult",
    "RFFTraceExtractor",
    "RetrievalResult",
    "exact_mmd2",
    "exact_mmd_distance_matrix",
    "identity_retrieval",
    "median_heuristic",
    "calc_rfftrace",
    "collect_responses",
]

__version__ = "0.1.0"
