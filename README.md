# DistDNA

DistDNA is a reproducible experiment package for the distributional comparison stage of the
RFFTrace manuscript. It can start from stochastic text responses or validated response embeddings
and implements the paper's controlled pilot exactly:

- single-sample cosine distance;
- mean-DNA cosine distance;
- exact biased RBF-MMD (the V-statistic in equations 6-7);
- RFFTrace squared Euclidean distance, with an optional shared random projection;
- same-decoding-setting and cross-decoding-setting identity retrieval;
- Top-1, Top-3, Top-5, mean reciprocal rank, and per-query ranks.

The collection layer uses an explicit prompt/model/decoding manifest and an append-only JSONL
response cache. Collection can resume after interruption, every generation has a stable seed, and
the cache is pinned to the manifest fingerprint. A shared batch encoder then publishes separate
calibration and evaluation embedding tensors for the statistical pipeline.

The package follows the public shape of the preceding LLM-DNA toolkit: a small Python API, a
dedicated extraction CLI, `core/` orchestration, `data/` loaders, `dna/` extractors/signatures,
scripts, configs, tests, and self-describing saved vectors. Unlike pointwise LLM-DNA extraction,
RFFTrace extraction is deliberately dataset-level so its random features and optional projection
are sampled once and shared across all models and decoding settings.

## Quick start

Python 3.9+ and NumPy are required. From the repository root:

```bash
python3 -m pip install -e '.[dev]'
distdna demo --output-dir demo --run
```

Without installing the package, the same smoke run is:

```bash
PYTHONPATH=src python3 -m distdna demo --output-dir demo --run
```

The command creates deterministic synthetic evaluation and calibration datasets, runs all four
methods over three generation counts and every same/cross-setting pair, then writes the artifact
bundle to `demo/results/`.

To exercise the complete text-response path before connecting a model provider:

```bash
PYTHONPATH=src python3 -m distdna pipeline-demo \
  --output-dir results/text-response-pilot \
  --run
```

This writes 3,456 visible responses, encodes them, and runs all four methods. The generator and
hashing encoder are controlled diagnostics, clearly labeled synthetic in the manifest; they are
not research evidence or a substitute for real model generations.

For a deliberately harder method diagnostic, use equal-mean response distributions whose shape is
model-specific. Point and mean representations should be weak, while exact MMD and RFFTrace should
improve with more generations:

```bash
PYTHONPATH=src python3 -m distdna demo \
  --profile distributional \
  --output-dir results/distributional-diagnostic \
  --run
```

This profile is a controlled implementation check, not paper evidence or a substitute for real LLM
response embeddings.

## Extract RFFTrace DNA vectors

The dedicated CLI mirrors LLM-DNA's `calc-dna` workflow:

```bash
calc-rfftrace \
  --evaluation data/evaluation.npz \
  --calibration data/calibration.npz \
  --settings deterministic temp_0_7 temp_1_0 \
  --generations 16 \
  --rff-dim 1024 \
  --dna-dim 128 \
  --output-dir out/rfftrace-run
```

The equivalent package command is `distdna extract ...`.

The Python API is intentionally close to LLM-DNA's `DNAExtractionConfig` plus `calc_dna` pattern:

```python
from pathlib import Path

from distdna import RFFTraceExtractionConfig, calc_rfftrace

config = RFFTraceExtractionConfig(
    evaluation_path=Path("data/evaluation.npz"),
    calibration_path=Path("data/calibration.npz"),
    generations=16,
    rff_dimension=1024,
    dna_dimension=128,
    random_seed=2027,
    output_dir=Path("out/rfftrace-run"),
)
result = calc_rfftrace(config)
print(result.vectors.shape)
print(result.summary_path)
```

Extraction writes individual `DNASignature` files, one combined signature collection, the exact
shared RFF/projection parameters, and a JSON summary. `DNASignature.distance_to()` supports
Euclidean, squared-Euclidean, and cosine distances; RFFTrace comparisons should normally use
squared Euclidean distance.

## Package layout

```text
configs/                 experiment configurations
scripts/                 direct command wrappers
src/distdna/
  api.py                  public config/result objects and calc_rfftrace
  cli.py                  distdna and calc-rfftrace commands
  core/                   extraction orchestration
  data/                   manifests, response caches, encoders, datasets, and loaders
  dna/                    RFFTrace extractor and DNA signature objects
  experiment.py           retrieval pilot runner
  features.py             shared RFF and projection implementation
  kernels.py              exact MMD and bandwidth calibration
  text_demo.py            controlled response-to-retrieval diagnostic
tests/                    unit and end-to-end coverage
```

See [the LLM-DNA compatibility note](docs/LLM_DNA_COMPATIBILITY.md) for the exact module, API, and
artifact mapping, plus the method-level differences that must remain.

While the final manuscript roster is pending, follow the
[temporary local-model pilot guide](docs/LOCAL_PILOT_SETUP.md). It provides a six-model,
Apple-Silicon-friendly collection whose Hub revisions are resolved to immutable commit SHAs before
generation. Its outputs are explicitly diagnostic and must not be reported as manuscript evidence.

Existing student-branch response folders must pass the
[legacy LLM-DNA audit/import workflow](docs/LEGACY_LLM_DNA_IMPORT.md) before reuse. The importer
requires complete cells, actual generation seeds, source-file hashes, and exact model revisions;
unknown-provenance responses are reported but never admitted to the paper pipeline.

Summarize retrieval performance and the RFF approximation to exact MMD:

```bash
distdna summarize demo/results --output demo/results/summary.json
```

The summary also emits `pilot_checks`. It flags a ceiling-effect pilot, missing method/comparison
coverage, a missing generation-count sweep, or degenerate RFF-to-exact diagnostics. These checks
control whether the experiment is ready for repeated seeds; they do not declare a method superior.

Aggregate repeated runs and report mean plus sample standard deviation:

```bash
distdna aggregate results/seed*/results --output results/aggregate.json
```

The aggregate emits `scale_readiness` and does not approve cohort expansion until at least two
distinct seeds pass every pilot check. Passing this operational gate is not a paper claim.

## Collect and encode real responses

Start from [`configs/collection.example.json`](configs/collection.example.json). The public
`ResponseGenerator` contract keeps provider-specific SDK code outside the experiment machinery:

```python
from distdna import CollectionManifest, collect_responses

manifest = CollectionManifest.load("configs/my-collection.json")
responses = collect_responses(
    manifest,
    generator=my_provider_adapter,  # implements generate(model_id, prompt, setting, seed)
    cache_dir="data/my-run/responses",
)
```

Each cache contains a copy of `manifest.json` plus append-only `responses.jsonl`. Calling
`collect_responses` again skips every valid existing cell and generates only missing cells. A
manifest change is rejected rather than silently mixing experimental conditions.

Encode a complete cache with a sentence-transformer:

```bash
python3 -m pip install -e '.[embedding]'
distdna encode-responses \
  --manifest configs/my-collection.json \
  --cache-dir data/my-run/responses \
  --encoder sentence-transformer \
  --encoder-model sentence-transformers/all-mpnet-base-v2 \
  --output-dir data/my-run/embeddings
```

The dependency-free `--encoder hashing` option exists only for tests and controlled diagnostics.
`encode-responses` refuses incomplete caches and publishes the two NPZ files atomically.

## Input contract

Each input is a compressed NumPy `.npz` file containing exactly these required arrays:

| Array | Shape | Meaning |
| --- | --- | --- |
| `embeddings` | `[M, S, T, R, d]` | model, decoding setting, prompt, generation, response feature |
| `model_ids` | `[M]` | unique string model identifiers |
| `setting_ids` | `[S]` | unique string decoding-setting identifiers |
| `prompt_ids` | `[T]` | unique string prompt identifiers |

`embeddings` must be a dense floating-point array containing only finite values. Object arrays,
empty axes, duplicate IDs, missing generations, and NaN/Inf values fail validation. This strictness
prevents missing measurements from silently entering a table as NA.

Evaluation and bandwidth-calibration data are separate files. Their `prompt_ids` must be disjoint,
and their final embedding dimensions must match. The model and setting axes may differ because
calibration is used only to select the RBF bandwidth.

Validate files before a long run:

```bash
distdna validate-data data/evaluation.npz
distdna validate-config configs/pilot.json
```

## Configuration

Copy [`configs/pilot.example.json`](configs/pilot.example.json) and adjust paths and sweeps. Paths
are resolved relative to the config file, not the current shell directory.

An empty `comparisons` list evaluates the Cartesian product of decoding settings, covering both
same-setting and cross-setting retrieval. To run only selected comparisons:

```json
"comparisons": [
  {"query": "temp_1_0", "reference": "deterministic"},
  {"query": "temp_1_0", "reference": "temp_1_0"}
]
```

`normalization: "l2"` normalizes each individual response embedding before bandwidth calibration
and every comparison. `normalization: "none"` preserves encoder magnitudes. The median heuristic
uses only calibration embeddings; a fixed positive `bandwidth.value` is also supported. The
configured multiplier is applied after either selection strategy.

`generation_counts` always uses the first `R` generations. Therefore the source dataset should
store generations in a deterministic order produced by the collection manifest. For same-setting
retrieval, the runner prevents self-comparison leakage by using two disjoint, nested pools: query
samples use indices `[0, R)`, and reference samples use
`[max(generation_counts), max(generation_counts) + R)`. Any run containing a same-setting
comparison therefore requires at least `2 * max(generation_counts)` stored generations. Cross-
setting comparisons use the primary pool from each distinct setting. RFF dimensions are controlled
nested-prefix ablations: the first `D` frequencies and phases are identical across dimension
settings for a fixed seed.

## Output bundle

Every run requires a new output directory and publishes it only after all computations succeed:

- `metrics.csv`: one aggregate retrieval row per method/ablation/setting comparison;
- `ranks.csv`: one auditable rank per query model and aggregate row;
- `distances.npz`: the exact distance matrix behind each row, when enabled;
- `parameters/rff_D*_seed*.npz`: the exact shared frequencies, phases, and optional projection;
- `metadata.json`: resolved config, selected sigma, input SHA-256 hashes, tensor shapes, IDs,
  artifact counts, and the NA policy.
- `summary.json`, when requested: retrieval averages split by same/cross-setting evaluation and
  RFF-to-exact-MMD distance correlation, MAE, and RMSE.

NA has only two structural meanings: a non-RFF method has no RFF dimension, and an unprojected
method has no projection dimension. Missing experimental measurements cause a validation error.

## Reproducibility invariants

- RFF frequencies and phases are sampled once per `(seed, D)` and reused across all models,
  prompts, decoding settings, generation-count ablations, and query/reference roles.
- The optional Gaussian projection is also sampled once and saved with the feature parameters.
- Calibration and held-out evaluation prompts cannot overlap.
- Same-setting query and reference representations use disjoint generation samples.
- Exact MMD uses the manuscript's biased estimator, including kernel diagonals.
- RFFTrace concatenates prompt mean features with the required `1/sqrt(T)` scaling, so squared
  Euclidean distance estimates prompt-averaged squared MMD.
- Cosine baselines and distributional methods receive the same normalized response embeddings.
- Existing output directories are never overwritten.
- Collection order is fixed by the manifest, and every generation seed is derived from the full
  `(base seed, model, setting, prompt, generation index)` key.
- A response cache cannot be opened under a different manifest fingerprint.
- Externally collected records require an explicit manifest opt-in, preserve their real seed under
  `seed_scheme: external`, and must match the manifest's immutable model revision.

## Development checks

```bash
python3 -m pytest
python3 -m compileall -q src tests
```
