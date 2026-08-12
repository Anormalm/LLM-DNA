# Temporary local experiment matrix

This matrix tracks the self-directed public-model program while the final collaborator-provided
roster and prompts are unavailable. Every value is an engineering diagnostic, not manuscript
evidence. The authoritative protocol remains `docs/PAPER_PROTOCOL.md`.

| Requirement | Design | Independent repeats | Status | Evidence |
| --- | --- | ---: | --- | --- |
| Four identity methods | Single sample, mean DNA, exact biased RBF-MMD, RFFTrace | 3 | Complete | `results/local-factorial-grid-aggregate.json` |
| Same/cross retrieval | Top-1/3/5, MRR, saved ranks and matrices | 3 | Complete | `results/local-factorial-grid-aggregate.json` |
| Generation count R | 1, 2, 3, 4 | 3 | Complete | `results/local-smoke-ablation-aggregate.json` |
| RFF dimension D | 16, 32, 64, 128, 256, 512, 1024 | 3 | Complete | `results/local-smoke-ablation-aggregate.json` |
| Projection dimension L | none, 32, 64, 128, 256, 512, 1024, 2048 | 3 | Complete | `results/local-smoke-projection-full-aggregate.json` |
| RBF bandwidth | median multiplier 0.5, 1.0, 2.0 | 3 per cell | Complete | `results/local-smoke-factorial-aggregate.json` |
| Feature normalization | per-response L2 and none | 3 per cell | Complete | `results/local-smoke-factorial-aggregate.json` |
| Temperature × top-p | 3 × 3 factorial plus deterministic reference | 3 | Complete | `results/local-factorial-grid-decoding-report.json` |
| Random seeds | independent collection and experiment seeds 2027–2029 | 3 | Complete | `results/local-factorial-grid-aggregate.json` |
| Relationship recovery | family-pair AUROC/AP and nearest-family accuracy | 3 | Complete | `results/local-factorial-grid-relationship-report.json` |
| Final regenerated figures | six figures × PDF/SVG/PNG plus hash manifest | NA | Complete | `results/local-factorial-final-figures-v2/figure-manifest.json` |

## Split and roster policy

- Calibration prompts: `cal_000`–`cal_002`; evaluation prompts: `eval_000`–`eval_005`.
- Prompt texts and IDs are disjoint across splits.
- Identity retrieval has no fitted model and therefore no train/test model split. Query and reference
  roles use the same six-model roster, with disjoint generation pools for same-setting evaluation.
- Relationship evaluation is likewise training-free and excludes identity self-pairs. Its family
  labels are declared before analysis and cover the full roster.
- The fixed encoder is `sentence-transformers/all-mpnet-base-v2` with 768-dimensional outputs.
- The six checkpoints are pinned to immutable Hub commit SHAs in each resolved manifest.

## Completion rule

The temporary program is complete only when all three decoding caches and embedding tensors are
complete, all three grid runs pass, the aggregate observes three distinct input hash pairs, decoding
and relationship reports pass their completeness gates, all final figures render, and the full test,
package, provenance, and artifact audit succeeds.

Completion record, 2026-08-11: all three caches contain 4,320 unique records; all embedding tensors
have their declared shapes; the three grid runs contain balanced coverage and distinct input hashes;
the decoding and relationship reports each contain 76 expected analysis cells; and the final figure
manifest covers six sources and 18 rendered artifacts. Detailed values and hashes are recorded in
`docs/REAL_PILOT_RESULTS.md`.
