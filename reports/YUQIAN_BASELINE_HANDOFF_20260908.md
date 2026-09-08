# Yuqian: baseline handoff for the current main experiment

Please evaluate LLM-DNA and PhyloLM against the following shared experimental
protocol. Baseline implementation and validation are your task; our earlier
baseline numbers are provisional and should not be treated as target results.

This handoff publishes the protocol and manifests, not the response cache or
embeddings. Data and implementation paths below refer to the existing DistDNA
workspace; those artifacts must be shared separately if you do not have access.

## 1. Task and models

- Closed-set identification at **model granularity**: fingerprint a query and
  rank all **30 enrolled models**. A different model in the same family is wrong.
- No query-family information, family filtering, or family prediction stage.
- Use exactly the model roster and immutable revisions in
  `configs/exact30-fresh-panel-20260906.json` (`model_ids` and
  `metadata.model_revisions`). Preserve its gallery ordering for tie-breaking.
- Human-readable roster: `reports/CURRENT_EXPERIMENT_SETTINGS_20260908.md`.
  There are 4 BLOOM/BLOOMZ, 3 Falcon-H1, 2 OPT, 3 Pythia-410m seed variants,
  8 Qwen2.5, 6 SmolLM2, 2 StableLM2 and 2 TinyLlama models.
- The 73-model expansion is not part of the current reported comparison.

## 2. Exact decoding grid

| Condition | Temperature | Top-p | Manifest setting ID |
| --- | ---: | ---: | --- |
| Baseline | 0.7 | 1.0 | `temp_0_7_top_p_1_0` |
| Low temperature | 0.5 | 1.0 | `temp_0_5_top_p_1_0` |
| High temperature | 0.9 | 1.0 | `temp_0_9_top_p_1_0` |
| Top-p shift | 0.7 | 0.9 | `temp_0_7_top_p_0_9` |
| Combined shift | 0.9 | 0.9 | `temp_0_9_top_p_0_9` |

All 30 models are evaluated under all five cells. This is not a full Cartesian
grid, and the planned extra temperature/top-p cells are not included.

## 3. Identical prompts and response budget

The collection has **20 calibration prompts + 80 evaluation prompts**, with
**8 generations per model/prompt/setting**: 120,000 responses total.

The main comparison uses only the **fixed 20 evaluation prompts listed below**,
with **4 responses per prompt per arm**: 80 responses per model per arm.
The remaining evaluation prompts support ablations, not the main 20x4 score.

Prompt text and split labels are in the manifest; do not resample or rewrite
prompts. The main subset is the 20 lowest SHA-256 hashes of
`hard13-prompt-ablation` + NUL + prompt ID among evaluation prompts, restored
to manifest order. Implementation: `nested_prompt_indices` in
`scripts/run_hard13_exact_model_replication.py`.

```text
exact30_fresh_brainstorming_06
exact30_fresh_brainstorming_11
exact30_fresh_brainstorming_15
exact30_fresh_brainstorming_18
exact30_fresh_classification_06
exact30_fresh_classification_09
exact30_fresh_classification_17
exact30_fresh_creative_writing_06
exact30_fresh_creative_writing_08
exact30_fresh_creative_writing_12
exact30_fresh_creative_writing_13
exact30_fresh_creative_writing_19
exact30_fresh_general_qa_06
exact30_fresh_general_qa_09
exact30_fresh_general_qa_14
exact30_fresh_general_qa_15
exact30_fresh_general_qa_18
exact30_fresh_open_qa_08
exact30_fresh_open_qa_10
exact30_fresh_open_qa_17
```

Use generation indices **A={0,1,2,3}** and **B={4,5,6,7}**. For each shifted
condition, evaluate all four assignments:

| Reference | Query |
| --- | --- |
| Baseline, bank A | Shifted, bank B |
| Baseline, bank B | Shifted, bank A |
| Shifted, bank A | Baseline, bank B |
| Shifted, bank B | Baseline, bank A |

Headline results average the four shifts and these four assignments. That is
480 query predictions per encoder. Also retain the first two assignments as
the forward-only result. Baseline-vs-baseline with independent banks is a
separate same-setting robustness result, not part of the headline average.

## 4. Shared cached data and generation controls

Prefer reusing the existing responses so differences are due to methods rather
than a new response draw:

`data/exact30-fresh-panel-20260906/responses/responses.jsonl`

Each record contains `model_id`, `prompt_id`, `setting_id`, `generation_index`,
response text, generation seed and provenance metadata. The original collection
uses plain-text prompts (no chat template), a 128-new-token ceiling,
`min_new_tokens=24`, and a common response boundary of four sentences,
eight nonempty lines, or 72 whitespace-delimited words. The cache strips
boundary whitespace and skips special tokens. Empty-decode retry cap is 16;
records retain the actual generation seed and attempt metadata.

If a native baseline requires different inputs or unstripped token-level
outputs, report that as a separate native-protocol run rather than silently
mixing it into the response-matched table. Record additional calls and tokens.
The feature seed below is not a generation seed.

## 5. Encoders and our fixed method

For embedding-based matched comparisons, run both:

- `BAAI/bge-base-en-v1.5`, revision
  `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` (BGE **base**, not large).
- `sentence-transformers/all-mpnet-base-v2`, revision
  `e8c3b32edf5434bc2275fc9bab85f82640a19130`.

Normalized response embeddings are available in
`data/exact30-fresh-panel-20260906/embeddings-bge/` and
`data/exact30-fresh-panel-20260906/embeddings-mpnet/`.
Each directory contains `evaluation.npz`, `calibration.npz`, and `summary.json`
with array axes and provenance. Use the recorded model/prompt/setting axes.

Our fixed DistDNA method uses L2-normalized embeddings, a shared RBF random
Fourier feature map with D=512 and seed 20260822, prompt-wise feature means,
concatenation divided by sqrt(20), and squared-Euclidean scoring. Bandwidth is
the median calibration-embedding pair distance, at most 100,000 pairs,
seed 20260822. Evaluation prompts must not tune the representation.

These are DistDNA parameters, not required baseline hyperparameters. PhyloLM
does not need an encoder and should not be counted twice as independent
BGE/MPNet evidence.

## 6. Baseline choices versus the shared protocol

Keep each baseline's scoring rule explicit. Do not force cosine on all methods.
The shared outcomes are ranks and identification accuracy, not raw distances.

For LLM-DNA, please distinguish the published one-response construction from
any multi-response adaptation. Specify whether responses are aggregated before
or after distance calculation, and record projection width, map seed and
normalization. Our earlier 128D, average-cross-draw-distance implementation is
**not a fixed requirement**: the audit showed large aggregation/compression
effects. Native settings and explicit sensitivity controls are needed; do not
select the best variant using the evaluation identities and report only it.

For PhyloLM, distinguish cross-tokenizer four-character prefixes from a
same-tokenizer token-distribution experiment. If using our text cache, label
the whitespace-stripped prefix variant as adapted. Its original completion
contexts and larger sampling budgets belong in a separately labeled native
comparison. In either main-table adaptation, all 30 models remain candidates.

## 7. Metrics and requested outputs

- Primary: model-level Top-1. Secondary: Top-3, Top-5 and MRR.
- Smaller dissimilarity ranks first; exact ties use fixed manifest gallery order.
  Record tie rates, particularly for sparse prefix estimates.
- Headline aggregation: average bank assignments, setting directions, shifts,
  and (for embedding methods) BGE/MPNet within each model; then average models
  within each of the eight benchmark groups; then average groups equally.
  Report ordinary model-micro accuracy separately. Groups enter only here.
- Preserve per-query ranks and preferably each 30x30 query/reference score
  matrix, with method, encoder if applicable, setting pair, bank assignment,
  ordered model IDs, and all method hyperparameters.
- Do not count response draws, role swaps, shifts, or encoders as independent
  model populations. Paired uncertainty can be computed centrally from these
  outputs with the existing 10,000-resample model/family bootstrap.
- Retain per-shift, forward-only and same-setting results separately. Do not
  substitute family accuracy, pairwise relatedness AUC, or SVM relationship
  accuracy for the model-retrieval endpoint.

Analysis protocol: `configs/exact30-fresh-panel-analysis-20260906.json`.
This handoff does not alter that frozen collection or its original contrasts.
