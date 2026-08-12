# Temporary public-model experiment results

Status: engineering diagnostic only. Do not cite these values as manuscript evidence.

## Protocol

- Date: 2026-08-11
- Independent collection seeds: 2027, 2028, 2029
- Models: six pinned public instruction checkpoints from the local-pilot manifest
- Prompts: three calibration and six evaluation prompts
- Initial ablation dataset: two decoding settings, eight generations per cell, 864 responses per
  seed and 2,592 total
- Expanded decoding dataset: deterministic decoding plus a `3 × 3` temperature/top-p factorial,
  eight generations per cell, 4,320 responses per seed and 12,960 total
- Encoder: `sentence-transformers/all-mpnet-base-v2`, 768 dimensions
- Retrieval methods: single-sample cosine, mean-DNA cosine, exact biased RBF-MMD, RFFTrace
- Expanded-grid reference: median bandwidth, per-response L2 normalization, `D=512`, no compact
  projection, and `R = {1, 2, 3, 4}`
- Same-setting policy: disjoint query and reference generation pools

Every seed passed the protocol checks. Both the initial and expanded aggregates observed three
unique experiment seeds and three distinct calibration/evaluation hash pairs, so their operational
scale-readiness gates pass. This gate checks reproducibility and coverage; it does not establish a
performance claim.

## Initial two-setting Top-1 retrieval

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

## Bandwidth and feature-normalization ablation

The three independent datasets were evaluated under the Cartesian product of response-feature
normalization `{l2, none}` and median-bandwidth multiplier `{0.5, 1.0, 2.0}`. All 18 runs passed
the protocol gates, and the aggregate confirmed balanced three-seed coverage for all six cells.

At `R=4`, the L2-normalized Top-1 results were:

| Bandwidth multiplier | Exact MMD cross | Exact MMD same | RFFTrace D=512 cross | RFFTrace D=512 same |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.833 ± 0.083 | 0.889 ± 0.096 | 0.861 ± 0.096 | 0.861 ± 0.048 |
| 1.0 | 0.833 ± 0.000 | 0.861 ± 0.048 | 0.750 ± 0.083 | 0.861 ± 0.048 |
| 2.0 | 0.833 ± 0.000 | 0.833 ± 0.000 | 0.750 ± 0.083 | 0.861 ± 0.048 |

The corresponding selected sigmas were `0.6702`, `1.3405`, and `2.6810`. The half-median cell has
the strongest observed RFFTrace cross-setting retrieval, but each seed contributes only 12 queries
per comparison type; this difference is not sufficient for bandwidth selection. Exact MMD is more
stable across the three tested bandwidths.

The normalized and unnormalized cells are numerically identical at reported precision. This is an
expected property of this encoder output, not evidence that normalization is generally irrelevant:
across all 2,592 response embeddings, the pre-ablation L2 norms range only from `0.9999999` to
`1.0000001` with standard deviation `3.49e-8`.

## Expanded ten-setting retrieval grid

The expanded dataset evaluates the deterministic reference and all nine stochastic
temperature/top-p cells. At `R=4`, averages over the complete setting grid are:

| Method | Cross Top-1 | Same Top-1 | Cross MRR | Same MRR |
| --- | ---: | ---: | ---: | ---: |
| Single-sample cosine | 0.631 ± 0.024 | 0.667 ± 0.044 | 0.768 ± 0.008 | 0.800 ± 0.024 |
| Mean-DNA cosine | 0.880 ± 0.024 | 0.939 ± 0.035 | 0.936 ± 0.014 | 0.969 ± 0.019 |
| Exact MMD | 0.897 ± 0.023 | 0.956 ± 0.010 | 0.945 ± 0.013 | 0.978 ± 0.005 |
| RFFTrace, `D=512` | 0.892 ± 0.020 | 0.950 ± 0.000 | 0.942 ± 0.012 | 0.975 ± 0.000 |

The distributional methods remain substantially above the single-sample baseline. Exact MMD has
the highest observed aggregate retrieval, but the gaps to RFFTrace and mean DNA are small in a
six-model cohort. RFFTrace at `D=512` correlates with exact MMD distances at
`0.9953 ± 0.0004` cross-setting and `0.9960 ± 0.0008` same-setting.

## Temperature and top-p factorial

The decoding report contains 76 complete method/evaluation cells: the deterministic reference and
nine stochastic cells for same-setting evaluation, plus all nine stochastic-query-to-deterministic
reference cells, across four methods. Deterministic same-setting Top-1 is `1.0` for every method.

Marginal Top-1 means at `R=4` show the strongest degradation along temperature:

| Method | Evaluation | T=0.3 | T=0.7 | T=1.0 |
| --- | --- | ---: | ---: | ---: |
| Single sample | Same | 0.741 | 0.611 | 0.537 |
| Single sample | Cross to deterministic | 0.926 | 0.778 | 0.667 |
| Mean DNA | Same | 0.981 | 0.907 | 0.907 |
| Mean DNA | Cross to deterministic | 0.981 | 0.870 | 0.741 |
| Exact MMD | Same | 1.000 | 0.907 | 0.944 |
| Exact MMD | Cross to deterministic | 1.000 | 0.889 | 0.778 |
| RFFTrace | Same | 0.981 | 0.907 | 0.944 |
| RFFTrace | Cross to deterministic | 0.963 | 0.852 | 0.815 |

Across methods and evaluation types, the temperature Top-1 range is `0.074–0.259`, while the
top-p range is `0.037–0.111`. Top-p effects are smaller and non-monotonic here. These are marginal
descriptions of a small factorial, not causal estimates or recommended decoding defaults.

## Coarse relationship recovery

The declared public-model family labels provide a deliberately coarse relationship proxy. The
analysis excludes identity self-pairs; nearest-family accuracy also excludes singleton groups.
Chance baselines are AUROC `0.5` and nearest-family accuracy `0.32`.

| Method | Same AUROC | Cross AUROC | Same nearest-family | Cross nearest-family |
| --- | ---: | ---: | ---: | ---: |
| Single-sample cosine | 0.691 ± 0.034 | 0.692 ± 0.021 | 0.640 ± 0.092 | 0.852 ± 0.071 |
| Mean-DNA cosine | 0.874 ± 0.010 | 0.793 ± 0.009 | 0.933 ± 0.012 | 0.941 ± 0.013 |
| Exact MMD | 0.886 ± 0.009 | 0.797 ± 0.009 | 0.933 ± 0.012 | 0.941 ± 0.034 |
| RFFTrace, `D=512` | 0.881 ± 0.018 | 0.784 ± 0.010 | 0.907 ± 0.042 | 0.933 ± 0.022 |

All distributional methods recover the declared family structure well above chance. This supports
the relationship-analysis pipeline, but family membership is not a substitute for the manuscript's
intended behavioral and provenance relationships.

## Collection and artifact completeness

- All three expanded response caches contain exactly 4,320 records, 4,320 unique cache keys, and
  540 complete model/setting/prompt cells.
- Deterministic cells contain one unique response among eight generations, as expected. Mean
  stochastic unique-response ratios are `0.9977`, `0.9961`, and `0.9979` for seeds 2027–2029.
- Each encoded seed has calibration shape `[6, 10, 3, 8, 768]` and evaluation shape
  `[6, 10, 6, 8, 768]`.
- Each grid run contains 1,600 metric rows and 9,600 auditable rank rows.
- The final bundle contains six visually inspected figures in PDF, SVG, and PNG, with source and
  artifact hashes recorded in `figure-manifest.json`.

## Reproducibility record

- Aggregate: `results/local-smoke-aggregate.json`
- Aggregate SHA-256: `5cecae2bd3b08af5abe1b088cb11b27fd15643817159665de2bd8595de9f2874`
- Feature ablation: `results/local-smoke-ablation-aggregate.json`
- Feature-ablation SHA-256: `bf3bd63fd848a8cde3b5c541c85ece5966e6b916ef03516dddcf696718135c0d`
- Projection ablation: `results/local-smoke-projection-full-aggregate.json`
- Projection-ablation SHA-256: `72b2b956e2834df8541d50488acb4b3fa1138dd05bc3f1c92cac2274359e9710`
- Bandwidth/normalization factorial: `results/local-smoke-factorial-aggregate.json`
- Bandwidth/normalization SHA-256: `d785751becf8b2019c23c34a03addd6edc6cdb65925d5134d52dd8e801bf4dc2`
- Expanded retrieval aggregate: `results/local-factorial-grid-aggregate.json`
- Expanded retrieval SHA-256: `2a380385aa74660c2f5996663aa4a3d2b9649b8b72991af3e5d6248cfcbd7c0c`
- Decoding report: `results/local-factorial-grid-decoding-report.json`
- Decoding-report SHA-256: `69ee8d6b46c90b64e6beca922a6c564ae9c53b6a2ddf836296f88ad33f11cd33`
- Relationship report: `results/local-factorial-grid-relationship-report.json`
- Relationship-report SHA-256: `47783cfaa94b0660d8b09f6096df20790ca828df327f33d8585a4e6d16b6739b`
- Final figures: `results/local-factorial-final-figures-v2/`
- Runtime: macOS 26.4.1 arm64, PyTorch 2.13.0, Transformers 5.14.1,
  Sentence Transformers 5.6.1

The response caches, embedding tensors, distance matrices, ranks, and summaries are intentionally
kept outside version control. Their hashes and immutable model revisions are recorded in the local
artifact bundles.

## Decision

The temporary self-directed program is complete and the implementation is operationally ready for
the manuscript-scale run. The current results support three engineering choices: retain exact MMD
as the reference, use unprojected `D=512` as the primary RFFTrace approximation, and treat
temperature as a required evaluation axis rather than nuisance metadata.

No paper-level empirical claim should be made from this cohort. The next run should replace the
temporary roster and prompts with the final collaborator-approved protocol, increase the model and
prompt counts, collect 32 generations per cell, and preserve at least three independent collection
seeds. It should compare `L=1024` and `L=2048` explicitly against unprojected `D=512`, supporting
disjoint same-setting evaluation through `R=16` and materially better uncertainty estimates.
