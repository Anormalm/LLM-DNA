# Expanded public-model preflight

Status: end-to-end retrieval plumbing and checkpoint compatibility are verified. The original v1
response protocol fails the strengthened per-model truncation gate and is frozen. The v2 protocol
preserves the model roster, semantic tasks, decoding factorial, seeds, and analysis plan while
making every prompt's existing concise-answer intent measurable. Its all-roster sentinel must pass
before any full v2 seed is collected.

All results here are engineering diagnostics from a temporary public roster, not manuscript
evidence.

## Manuscript alignment

The design follows the manuscript's experimental object: repeated generations are samples from a
prompt-conditioned response distribution. It retains shared prompt IDs, disjoint calibration and
evaluation splits, deterministic and stochastic decoding, exact RBF-MMD as a controlled diagnostic,
one shared RFF map, optional fixed projection, same-setting and cross-setting retrieval, and the
planned generation-count, RFF-dimension, projection, bandwidth, normalization, and seed analyses.

The manuscript does not prescribe a response-generation token ceiling. The temporary local
collection keeps `max_new_tokens=128`, logs the stop reason for every response, and treats the ceiling
as a censoring boundary: both the overall cohort and every individual model must have at most 25%
hard-limit stops. Changing prompt text creates a new manifest fingerprint and cache namespace.

## Expanded v2 protocol

- Roster: 12 pinned public instruction checkpoints in four declared groups: SmolLM2, Qwen 2.5,
  TinyLlama Chat, and Granite 2B releases.
- Prompts: 8 calibration and 16 evaluation prompts with disjoint IDs and texts.
- Prompt revision: all 24 v1 semantic tasks are retained; only answer length and format are made
  explicit through tracked, ID-keyed revisions.
- Decoding: deterministic plus the full `3 x 3` temperature/top-p factorial.
- Collection: 32 generations per model/setting/prompt cell and three independent seeds.
- Response limit: 128 new tokens, with explicit prompt contracts and stop-reason logging.
- Encoder/reference: MPNet 768, exact biased RBF-MMD, and unprojected RFFTrace `D=512`.
- Compact evaluation: retain `L=1024` and `L=2048` as explicit comparisons.
- Quality gate: complete cache, every cell present, stable deterministic cells, diverse stochastic
  cells, adequate median length, overall truncation at most 25%, and every-model truncation at most
  25%.

The v1 manifest fingerprint is
`d1c20caf208cd29f5c9c1a10e3d3c2b7182cbb9a43bbf0e3be908e70f31941c5`. The derived v2
manifest fingerprint is
`51b6c05ebd9bd27ffd17e3a3717599ba90df8690089024ba8e10f9f89b51b532`. The v2 loader
re-derives that manifest from the tracked v1 manifest and prompt-revision file on every run and
fails closed on any mismatch. Every model remains pinned to the same 40-character Hub commit.

## What the v1 preflights established

The bounded preflight used four cached models, two calibration prompts, two evaluation prompts,
deterministic and high-temperature decoding, and four generations per cell: 128 responses in 32
complete cells. Tightening three open-ended instructions reduced overall truncation from 43.0% to
17.2%, and the retrieval pipeline completed with non-ceiling Top-1 results. At `R=2`, same-setting
Top-1 was 0.875 for mean DNA and 0.750 for exact MMD and RFFTrace; RFFTrace-exact distance
correlation ranged from 0.9773 to 0.9952. These figures validate execution only.

The original overall-only quality rule incorrectly labeled that cache ready. The per-model re-audit
finds truncation of 12.5% for SmolLM2-135M, 3.1% for SmolLM2-1.7B, 6.2% for Qwen2.5-0.5B,
and 46.9% for TinyLlama v1.0. Therefore the cache is not acceptable for scale inference.

The six-checkpoint compatibility benchmark completed all 24 calls through pinned revisions, native
templates, float16 weights, and Apple MPS. Its overall truncation is exactly 25%, but the per-model
re-audit finds 75% for TinyLlama v0.6 and 50% for Granite 3.2. It proves compatibility and provides
runtime observations; it does not pass the strengthened response-quality gate.

## Why v1 scale collection stopped

The one-generation v1 all-cell sentinel was intentionally stopped at 997 of 2,880 durable records
once failure was established. Its observed overall truncation was 44.5%. All four completed model
blocks independently failed:

| Model | Records | Truncation |
| --- | ---: | ---: |
| SmolLM2-135M-Instruct | 240 | 54.6% |
| SmolLM2-360M-Instruct | 240 | 37.9% |
| SmolLM2-1.7B-Instruct | 240 | 39.6% |
| Qwen2.5-0.5B-Instruct | 240 | 47.5% |

The next model had 37 observed records at 35.1% truncation. Failures were distributed across
decoding settings and concentrated in open-ended prompts, so this was not a temperature-specific
effect. Continuing the v1 sentinel would spend compute without changing the completed-model
decision.

The partially collected v1 seed 2027 cache remains frozen at 3,999 durable records. All v1 caches
are retained for audit but are never reused by v2 because prompt definitions changed.

## V2 execution and gates

The exact full design still requires 92,160 responses per seed and 276,480 across three seeds. The
earlier mixed-model benchmark projected roughly 164 central sequential MPS hours and 258 hours
under the conservative per-call projection. Those are planning estimates, not completion claims;
the v2 sentinel supplies a new checkpoint-specific runtime basis.

The runner performs the following fail-closed sequence:

1. Collect the complete 2,880-cell v2 sentinel.
2. Require both overall and every-model truncation to be at most 25%.
3. Collect seed 2027 and require the full response-quality gate.
4. Encode and run R/D, projection, bandwidth, and normalization analyses.
5. Repeat independently for seeds 2028 and 2029 only after each preceding gate passes.
6. Aggregate seeds, build decoding and relationship reports, and render a provenance-hashed figure
   bundle.

The complete resumable command is:

```bash
caffeinate -dimsu .venv/bin/python scripts/run_scale_program.py
```

The v2 manifest can be independently regenerated at a fresh path with:

```bash
distdna revise-manifest \
  --manifest configs/scale-expanded.collection.json \
  --prompt-revisions configs/scale-expanded.prompt-revisions.json \
  --dataset-id temporary-public-scale-v2 \
  --require-all-prompts \
  --output /tmp/scale-expanded-v2.collection.json
```

The sentinel estimates truncation and checkpoint-specific runtime. With one response per cell, it
cannot test within-cell stochastic diversity and is never a substitute for the complete seed gate.

## Local evidence

- V1 partial sentinel report: `results/scale-expanded-v1-sentinel-partial.json`
- V1 partial sentinel report SHA-256:
  `95d30eabf6c77bde0f6255c54557a55dc940713cc4fa93512b02ad9db9f7a290`
- Per-model bounded-preflight re-audit:
  `results/scale-preflight-final-quality-per-model.json`
- Per-model bounded-preflight re-audit SHA-256:
  `55424a28692f4d27ed77d3ece8eff0a7b18b16c8e082febd478af91db26d7062`
- Per-model new-checkpoint re-audit:
  `results/scale-new-models-benchmark-quality-per-model.json`
- Per-model new-checkpoint re-audit SHA-256:
  `950a2f6e75068f464ea53e96e3c607a4c8d48c4c117b1ae6d33a674499ba80e5`
- V2 prompt-revision file SHA-256:
  `56cc96d38aa7699667ba43c2fb417d2c3d66aca0ede8544ad07c728ec6cf05f6`
- V2 manifest file SHA-256:
  `22203fc5f211620308c6b2a23c640f160632103d96a9dddef306d87577d45b01`
- Final v1 cost estimate: `results/scale-expanded-final-v2-estimate.json`
- Final v1 cost-estimate SHA-256:
  `2b1774845b613f5dac8457af4114c1c0dd037a371261a78a2c8052af4e07b3d9`

Response caches, embeddings, and generated reports are intentionally excluded from version control.
The tracked manifests, revision map, commands, tests, and this record make every decision
reproducible.
