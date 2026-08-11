# Relationship to LLM-DNA

DistDNA is a distributional continuation of the public
[Xtra-Computing/LLM-DNA](https://github.com/Xtra-Computing/LLM-DNA) workflow. The repository is
organized so an LLM-DNA user encounters familiar concepts while the stochastic-decoding method
remains faithful to RFFTrace.

## Structural mapping

| LLM-DNA | DistDNA | Role |
| --- | --- | --- |
| `DNAExtractionConfig` | `RFFTraceExtractionConfig` | Typed extraction settings |
| `DNAExtractionResult` | `RFFTraceExtractionResult` | Vectors and saved artifact paths |
| `calc_dna(config)` | `calc_rfftrace(config)` | Primary Python API |
| `calc-dna` | `calc-rfftrace` | Dedicated extraction CLI |
| `dna/DNASignature.py` | `dna/signature.py` | Vector, metadata, persistence, distance |
| `dna/EmbeddingDNAExtractor.py` | `dna/extractor.py` | Response-embedding to DNA transformation |
| `core/extraction.py` | `core/extraction.py` | Pipeline orchestration |
| `data/DatasetLoader.py` | `data/loader.py` | Stable data-loading boundary |
| model-list response generation | `data/manifest.py`, `data/responses.py` | Repeated, resumable stochastic collection |
| response embedding stage | `data/encoders.py`, `data/pipeline.py` | Shared batch encoding into canonical tensors |
| `scripts/calc_dna.py` | `scripts/calc_rfftrace.py` | Direct script wrapper |
| `out/` | `out/` | User-facing extraction artifacts |

## Deliberate method-level differences

The following differences are required by the RFFTrace manuscript and are not architectural drift:

1. LLM-DNA can extract one model independently. RFFTrace extraction operates on the complete
   evaluation tensor so the RFF frequencies, phases, and optional random projection are sampled
   exactly once and shared across models, prompts, and decoding settings.
2. Each model-prompt-setting cell contains repeated generations rather than one response.
3. Bandwidth selection uses a separate calibration prompt split. Held-out evaluation prompts cannot
   overlap the calibration split.
4. The exact RBF-MMD implementation is a diagnostic/reference computation. Saved RFFTrace DNA
   vectors remain the scalable representation.
5. Same-setting retrieval uses disjoint query and reference generation pools. It never retrieves a
   vector against the identical response samples.

## Artifact continuity

`calc-rfftrace` writes:

- one compressed `DNASignature` per model and setting under `signatures/`;
- `rfftrace_signatures.npz`, the complete aligned collection;
- `rff_parameters_D*_seed*.npz`, the shared map and optional projection;
- `summary.json`, containing the resolved extraction config and all model, setting, and prompt IDs.

The response-generation and sentence-encoding layer remains behind `data/` rather than being mixed
with the statistical representation code. It now retains LLM-DNA's useful operational patterns:
visible response caches, resumable generation, a provider/model wrapper boundary, shared batch text
encoding, and model-list execution. Each prompt is expanded to `R` stochastic generations per
decoding setting, and the manifest pins the complete collection design.

`distdna pipeline-demo --run` verifies this entire path with controlled text distributions.
Real-provider code implements the small `ResponseGenerator` protocol, so API retries and credentials
remain outside the kernel/RFF implementation and cannot silently alter stored experiment cells.

The student experiment branch's `*_tXX_pXX_rN/responses.json` layout is supported through the
strict `audit-legacy` and `import-legacy` commands. They reuse only raw response text with known
seeds and immutable revisions. They do not import legacy character-hashing features, point-DNA
vectors, incomplete grids, or the historical unknown-provenance gallery. See
[`LEGACY_LLM_DNA_IMPORT.md`](LEGACY_LLM_DNA_IMPORT.md).
