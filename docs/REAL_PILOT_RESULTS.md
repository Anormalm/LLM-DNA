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

## Feature-dimension ablation

The three datasets were rerun over `R = {1, 2, 3, 4}` and
`D = {16, 32, 64, 128, 256, 512, 1024}`. The sweep follows the manuscript's finite-sample analysis,
which predicts approximation error decreasing with both the generation count and feature dimension.

At four generations, distance correlation with exact MMD was:

| RFF D | Cross-setting | Same-setting |
| ---: | ---: | ---: |
| 16 | 0.821 ± 0.028 | 0.854 ± 0.013 |
| 32 | 0.922 ± 0.032 | 0.918 ± 0.004 |
| 64 | 0.967 ± 0.007 | 0.961 ± 0.023 |
| 128 | 0.980 ± 0.005 | 0.980 ± 0.009 |
| 256 | 0.988 ± 0.002 | 0.989 ± 0.004 |
| 512 | 0.996 ± 0.001 | 0.995 ± 0.001 |
| 1024 | 0.998 ± 0.000 | 0.998 ± 0.000 |

Approximation quality improves consistently with `D`. Top-1 retrieval is not monotonic because the
cohort contains only six reference models: small distance perturbations can flip a rank even when
the full matrices are nearly identical. Therefore retrieval peaks must not be used to claim that a
smaller randomized map outperforms exact MMD.

For the next protocol, `D=512` is the smallest tested dimension that exceeds approximately 0.99
correlation in both comparison types. `D=256` remains a reasonable lower-cost pilot setting at
approximately 0.988 correlation. Compact-projection dimensions must be evaluated separately and
must remain explicit in every summary and aggregate key.

## Compact-projection ablation

With `D=512`, the unprojected vector has `T × D = 6 × 512 = 3,072` coordinates. A second
three-seed sweep evaluated fixed Gaussian projections at
`L = {32, 64, 128, 256, 512, 1024, 2048}`. Each projection matrix was sampled once per seed and
shared across all models, prompts, decoding settings, generation counts, and comparison roles, as
required by the manuscript.

At four generations, correlation of projected distances with the corresponding unprojected
RFFTrace distances was:

| Projection L | Cross-setting | Same-setting |
| ---: | ---: | ---: |
| 32 | 0.714 ± 0.148 | 0.711 ± 0.085 |
| 64 | 0.825 ± 0.092 | 0.819 ± 0.052 |
| 128 | 0.926 ± 0.010 | 0.915 ± 0.007 |
| 256 | 0.949 ± 0.008 | 0.953 ± 0.010 |
| 512 | 0.967 ± 0.009 | 0.972 ± 0.008 |
| 1024 | 0.988 ± 0.004 | 0.990 ± 0.001 |
| 2048 | 0.995 ± 0.001 | 0.995 ± 0.001 |

`L=2048` is the smallest tested setting above 0.99 in both comparison types. It reduces the pilot
vector from 3,072 to 2,048 coordinates. `L=1024` gives threefold compression with a modest but
measurable loss, so it remains a cost–fidelity ablation rather than an equivalent replacement. The
end-to-end `L=2048` distances correlate with exact MMD at 0.991 ± 0.002 cross-setting and
0.989 ± 0.002 same-setting. Retrieval remains too coarse in this six-model cohort to select `L`.

## Reproducibility record

- Aggregate: `results/local-smoke-aggregate.json`
- Aggregate SHA-256: `cce4adf34b08fb8251f7be69cdfcda88a76c4f53ef2c1aa77f9986d957783ce4`
- Feature ablation: `results/local-smoke-ablation-aggregate.json`
- Feature-ablation SHA-256: `02cb989badc7b31a58fa69040e95f4227ffc895bc6fd2963aa2de95ae8a0f5eb`
- Projection ablation: `results/local-smoke-projection-full-aggregate.json`
- Projection-ablation SHA-256: `ae797c90f458a41562e2a3b2d5bdd2be76c84590ee9244ddc9dcf2f11191114d`
- Runtime: macOS 26.4.1 arm64, PyTorch 2.13.0, Transformers 5.14.1,
  Sentence Transformers 5.6.1

The response caches, embedding tensors, distance matrices, ranks, and summaries are intentionally
kept outside version control. Their hashes and immutable model revisions are recorded in the local
artifact bundles.

## Decision

The implementation is ready for a larger protocol, but this temporary cohort should not be expanded
blindly. The next run should use the final model roster, more evaluation prompts, 32 generations per
cell, and at least three independent collection seeds. It should retain unprojected `D=512` as the
reference and compare `L=1024` and `L=2048` explicitly. That supports disjoint same-setting
evaluation through `R=16` and gives a more credible uncertainty estimate.
