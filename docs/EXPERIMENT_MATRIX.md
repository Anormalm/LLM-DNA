# Temporary local experiment matrix

This matrix tracks the self-directed public-model program while the final collaborator-provided
roster and prompts are unavailable. Every value is an engineering diagnostic, not manuscript
evidence. The authoritative protocol remains `docs/PAPER_PROTOCOL.md`.

| Requirement | Design | Independent repeats | Status | Evidence |
| --- | --- | ---: | --- | --- |
| Four identity methods | Single sample, mean DNA, exact biased RBF-MMD, RFFTrace | 3 | Complete | `results/local-smoke-aggregate.json` |
| Same/cross retrieval | Top-1/3/5, MRR, saved ranks and matrices | 3 | Complete | `results/local-smoke-aggregate.json` |
| Generation count R | 1, 2, 3, 4 | 3 | Complete | `results/local-smoke-ablation-aggregate.json` |
| RFF dimension D | 16, 32, 64, 128, 256, 512, 1024 | 3 | Complete | `results/local-smoke-ablation-aggregate.json` |
| Projection dimension L | none, 32, 64, 128, 256, 512, 1024, 2048 | 3 | Complete | `results/local-smoke-projection-full-aggregate.json` |
| RBF bandwidth | median multiplier 0.5, 1.0, 2.0 | 3 per cell | Complete | `results/local-smoke-factorial-aggregate.json` |
| Feature normalization | per-response L2 and none | 3 per cell | Complete | `results/local-smoke-factorial-aggregate.json` |
| Temperature × top-p | 3 × 3 factorial plus deterministic reference | 3 | Running | `configs/local-factorial.collection.json` |
| Random seeds | independent collection and experiment seeds 2027–2029 | 3 | Running | resolved manifests and output metadata |
| Relationship recovery | family-pair AUROC/AP and nearest-family accuracy | 3 | Waiting on decoding grid | `configs/local-model-relationships.json` |
| Final regenerated figures | six figures × PDF/SVG/PNG plus hash manifest | NA | Waiting on final reports | `distdna render-figures` |

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
