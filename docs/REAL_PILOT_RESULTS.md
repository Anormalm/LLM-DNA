# Three-seed local-model pilot

Status: engineering diagnostic only. Do not cite these values as manuscript evidence.

## Protocol

- Date: 2026-08-11
- Independent collection seeds: 2027, 2028, 2029
- Models: six pinned public instruction checkpoints from the local-pilot manifest
- Decoding settings: two temperature/top-p combinations
- Prompts: three calibration and six evaluation prompts
- Responses: eight generations per model/setting/prompt, 864 per seed and 2,592 total
- Encoder: `sentence-transformers/all-mpnet-base-v2`, 768 dimensions
- Retrieval methods: single-sample cosine, mean-DNA cosine, exact biased RBF-MMD, RFFTrace
- RFF dimensions: 64 and 256
- Same-setting policy: disjoint query and reference generation pools

Every seed passed the pilot checks. The aggregate observed three unique experiment seeds and three
distinct calibration/evaluation hash pairs, so `scale_readiness.ready_for_scale` is true.

## Four-generation Top-1 retrieval

Values are mean ± sample standard deviation over three independent response datasets. Each seed has
12 queries per comparison type, so the uncertainty remains material.

| Method | RFF D | Cross-setting | Same-setting |
| --- | ---: | ---: | ---: |
| Single-sample cosine | NA | 0.417 ± 0.144 | 0.639 ± 0.048 |
| Mean-DNA cosine | NA | 0.778 ± 0.048 | 0.861 ± 0.096 |
| Exact MMD | NA | 0.833 ± 0.000 | 0.861 ± 0.048 |
| RFFTrace | 64 | 0.806 ± 0.127 | 0.889 ± 0.048 |
| RFFTrace | 256 | 0.806 ± 0.127 | 0.917 ± 0.000 |

The distributional methods improve substantially over the single-response baseline at four
generations in this small cohort. The differences among mean-DNA, exact MMD, and RFFTrace are too
small and the query count too limited to support a superiority claim.

## RFF approximation to exact MMD at four generations

| RFF D | Cross-setting correlation | Same-setting correlation |
| ---: | ---: | ---: |
| 64 | 0.967 ± 0.007 | 0.961 ± 0.023 |
| 256 | 0.988 ± 0.002 | 0.989 ± 0.004 |

The 256-feature map is a close approximation to the exact distance matrices in this pilot, with
low variation across independent response datasets.

## Reproducibility record

- Aggregate: `results/local-smoke-aggregate.json`
- Aggregate SHA-256: `cbcca80c23d9a769e3feb43728b85f368de5173823a19075b0fe8e29e68bf4d8`
- Runtime: macOS 26.4.1 arm64, PyTorch 2.13.0, Transformers 5.14.1,
  Sentence Transformers 5.6.1

The response caches, embedding tensors, distance matrices, ranks, and summaries are intentionally
kept outside version control. Their hashes and immutable model revisions are recorded in the local
artifact bundles.

## Decision

The implementation is ready for a larger protocol, but this temporary cohort should not be expanded
blindly. The next run should use the final model roster, more evaluation prompts, 32 generations per
cell, and at least three independent collection seeds. That supports disjoint same-setting
evaluation through `R=16` and gives a more credible uncertainty estimate.
