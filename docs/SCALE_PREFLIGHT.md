# Expanded public-model preflight

Status: the bounded preflight and six-checkpoint compatibility benchmark pass. The full temporary
collection is a resumable multi-day local run. All results here are engineering diagnostics, not
manuscript evidence.

## Expanded protocol

- Roster: 12 pinned public instruction checkpoints in four declared groups: SmolLM2, Qwen 2.5,
  TinyLlama Chat, and Granite 2B releases.
- Prompts: 8 calibration and 16 evaluation prompts with disjoint IDs and texts.
- Decoding: deterministic plus the full `3 × 3` temperature/top-p factorial.
- Collection: 32 generations per model/setting/prompt cell and three independent seeds.
- Response limit: 128 new tokens, with explicit prompt length contracts and stop-reason logging.
- Encoder/reference: MPNet 768, exact biased RBF-MMD, and unprojected RFFTrace `D=512`.
- Compact evaluation: retain `L=1024` and `L=2048` as explicit comparisons.

The collection manifest fingerprint is
`d1c20caf208cd29f5c9c1a10e3d3c2b7182cbb9a43bbf0e3be908e70f31941c5`.
Each proposed model is pinned to a 40-character Hub commit. Public metadata checks confirm every
repository is ungated, non-private, safe-tensor backed, and has a native chat template.

## Bounded preflight

The final preflight used four already cached models spanning the existing families, two new
calibration prompts, two new evaluation prompts, deterministic and high-temperature decoding, and
four generations per cell: 128 responses in 32 complete cells.

The initial 64-token protocol failed because 84.4% of responses reached the hard limit. Raising the
limit to 128 without changing prompt contracts still truncated 43.0%. Tightening the three most
open-ended instructions and explicitly asking models to stop when complete reduced final truncation
to 17.2%, which passes the declared 25% gate.

Final response-quality checks:

| Check | Result |
| --- | ---: |
| Complete records | 128 / 128 |
| Complete cells | 32 / 32 |
| Deterministic cell unique ratio | 0.25 (one response among four) |
| Stochastic cell unique ratio | 1.00 |
| Median response length | 38 words |
| Deterministic truncation | 12.5% |
| High-temperature truncation | 21.9% |
| Overall truncation | 17.2% |
| Measured generation throughput | 69.3 tokens/s |

Manual inspection found some weak or instruction-inaccurate answers from the smallest checkpoints.
That behavior is model signal rather than a collection failure; no answer is filtered or rewritten.
The system records visible text, stop reason, prompt/generated token counts, elapsed inference time,
device, library versions, and immutable model revision for every response.

## End-to-end retrieval check

The 128 responses were encoded to calibration and evaluation tensors of shape
`[4, 2, 2, 4, 768]`. All four methods ran over `R={1,2}` and both same/cross-setting comparisons.
The pilot gates pass, with non-ceiling Top-1 values from `0.625` to `0.875`. At `R=2`, same-setting
Top-1 is `0.875` for mean DNA and `0.750` for exact MMD and RFFTrace. The cohort has only four
models and eight aggregate queries per comparison, so these numbers validate execution only.
RFFTrace–exact distance correlation ranges from `0.9773` to `0.9952`.

## Scale estimate and decision

The exact design requires 92,160 responses per seed and 276,480 across three seeds, with an upper
bound of 35.4 million requested generation tokens. The preflight-based estimate is approximately
74 sequential MPS hours centrally and 131 hours under the deliberately conservative p90-per-call
projection. New Granite checkpoints may be slower, so their one-response compatibility benchmark
must precede full collection. Responses plus uncompressed embeddings are estimated near 1.01 GiB,
excluding downloaded weights and experiment outputs.

The six newly added checkpoints completed 24/24 bounded calls through their pinned revisions,
native templates, float16 weights, and Apple MPS. Their response-quality gate passes at exactly
25% truncation. This slower model mix revises the three-seed sequential estimate to approximately
164 central hours and 258 hours under the p90-per-call projection.

The old 64-token caches are not reused: `max_new_tokens` is part of the decoding setting, and
changing it to 128 changes the collection contract even when an individual old answer stopped
early. The appropriate next action is therefore a staged scale run:

1. Collect one complete 12-model seed and run its response-quality gate.
2. Encode and run the full R/D, projection, and bandwidth/normalization analyses for that seed.
3. Proceed to seeds 2028/2029 only when the preceding seed passes completeness, truncation, and
   diversity gates.
4. Aggregate all independent seeds, build decoding and relationship reports, and render a fresh
   provenance-hashed figure bundle.

The complete resumable command is:

```bash
caffeinate -dimsu .venv/bin/python scripts/run_scale_program.py
```

## Local evidence

- Quality report: `results/scale-preflight-final-quality.json`
- Quality-report SHA-256: `2ba6691b219bfe630de008ce5639f58a7ad5cfbd4e1283a740606916c3494982`
- Preflight retrieval: `results/scale-preflight-seed2027/summary.json`
- Retrieval-summary SHA-256: `3d519fe5d3b460d9fafb9963cd4eba86cccc6fc89b39dbabb3e1ad554f844dcc`
- Final cost estimate: `results/scale-expanded-final-v2-estimate.json`
- Cost-estimate SHA-256: `2b1774845b613f5dac8457af4114c1c0dd037a371261a78a2c8052af4e07b3d9`
- Response cache and embeddings: `data/scale-preflight-128-revised-seed2027/`

These artifacts are intentionally excluded from version control. The tracked manifests, commands,
tests, and this record make the decision and protocol reproducible.
