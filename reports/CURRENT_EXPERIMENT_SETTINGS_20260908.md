# Current main experiment: settings and model roster

The reported results use **30 models**, evaluated together in one global gallery.
Eight benchmark groupings are used for balanced reporting, not supplied to the
predictor. Repository identifiers below document the individual models; the
prediction task is model-granularity identification, not repository classification.
Immutable revisions are pinned in `configs/exact30-fresh-panel-20260906.json`.

## Decoding combinations

| Role | Temperature | Top-p |
| --- | ---: | ---: |
| Baseline | 0.7 | 1.0 |
| Lower temperature | 0.5 | 1.0 |
| Higher temperature | 0.9 | 1.0 |
| Nucleus truncation | 0.7 | 0.9 |
| Combined shift | 0.9 | 0.9 |

All 30 models have responses under all five combinations. Each primary contrast
compares the baseline with one shifted condition; headline results average both
setting directions and the two independent generation-bank assignments.
Same-setting baseline comparisons use independent banks and are secondary.
This is five selected cells, not a full temperature-by-top-p Cartesian grid.

## Sampling and fingerprints

- Collection: 20 calibration prompts + 80 evaluation prompts; eight generations
  per prompt and setting, giving 30 x 100 x 5 x 8 = **120,000 responses**.
- Main fingerprint: a fixed subset of 20 evaluation prompts x four generations
  = **80 responses per model per arm**. Query/reference banks are 0:3 and 4:7.
- Response ceiling: 128 new tokens; common boundary of four sentences, eight
  nonempty lines, or 72 whitespace-delimited words.
- Main encoders: `BAAI/bge-base-en-v1.5` and
  `sentence-transformers/all-mpnet-base-v2`; pinned revisions in the artifacts.
- DistDNA: shared RBF random Fourier features, D=512, seed 20260822;
  bandwidth estimated from calibration prompts only; squared-Euclidean scoring.
- Main endpoint: exact model Top-1. Top-3, Top-5 and MRR are secondary ranking
  metrics. All candidates compete without family information at prediction time.

## All 30 models

### BLOOM / BLOOMZ (4)

1. `bigscience/bloom-560m`
2. `bigscience/bloomz-560m`
3. `bigscience/bloom-1b1`
4. `bigscience/bloomz-1b1`

### Falcon-H1 (3)

5. `tiiuae/Falcon-H1-0.5B-Base`
6. `tiiuae/Falcon-H1-1.5B-Base`
7. `tiiuae/Falcon-H1-3B-Base`

### OPT (2)

8. `facebook/opt-1.3b`
9. `facebook/opt-iml-1.3b`

### Pythia-410m training seeds (3)

10. `EleutherAI/pythia-410m-seed1`
11. `EleutherAI/pythia-410m-seed2`
12. `EleutherAI/pythia-410m-seed3`

### Qwen2.5 (8)

13. `Qwen/Qwen2.5-0.5B`
14. `Qwen/Qwen2.5-0.5B-Instruct`
15. `Qwen/Qwen2.5-1.5B`
16. `Qwen/Qwen2.5-1.5B-Instruct`
17. `Qwen/Qwen2.5-Coder-0.5B`
18. `Qwen/Qwen2.5-Coder-0.5B-Instruct`
19. `Qwen/Qwen2.5-Coder-1.5B`
20. `Qwen/Qwen2.5-Coder-1.5B-Instruct`

### SmolLM2 (6)

21. `HuggingFaceTB/SmolLM2-135M`
22. `HuggingFaceTB/SmolLM2-135M-Instruct`
23. `HuggingFaceTB/SmolLM2-360M`
24. `HuggingFaceTB/SmolLM2-360M-Instruct`
25. `HuggingFaceTB/SmolLM2-1.7B`
26. `HuggingFaceTB/SmolLM2-1.7B-Instruct`

### StableLM2 (2)

27. `stabilityai/stablelm-2-1_6b`
28. `stabilityai/stablelm-2-zephyr-1_6b`

### TinyLlama (2)

29. `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
30. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## Expansion and follow-up are separate

As checked on 2026-09-08, the model expansion is still collecting. All 43 added
models passed collection preflight; eight finished full response collection.
The target is 73 models, subject to quality gates, encoding and completed global
gallery analysis. These added models are **not yet included in the main scores**.
Live status: `data/model-expansion-20260908/state.json`.

The motivated follow-up plans four additional cells: (0.8,1.0), (1.0,1.0),
(0.7,0.8), (0.7,0.7). These are **planned, not completed main settings**.
See `reports/MOTIVATED_DECODING_FOLLOWUP_PLAN_20260908.md`.
