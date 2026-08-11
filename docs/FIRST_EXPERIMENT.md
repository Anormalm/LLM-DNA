# First end-to-end experiment

Run date: 2026-08-06

## Purpose

Verify the complete paper-prescribed data path before spending model/API budget:

`stochastic text responses -> response embeddings -> calibration-only bandwidth -> four retrieval methods`

This is a controlled implementation diagnostic, not empirical evidence about language models.

## Design

- Seeds: 2027, 2028, 2029.
- Six latent identities and two stochastic decoding settings.
- Three calibration prompts and six disjoint evaluation prompts.
- 32 responses per model/setting/prompt, providing disjoint same-setting pools up to `R=16`.
- Identity appears only in a distribution over shared response styles; generated text never contains
  the model identifier.
- Dependency-free 128-dimensional hashing encoder (diagnostic only).
- Generation sweep: `R in {1, 4, 16}`.
- RFF sweep: `D in {64, 256}`.
- Methods: single-sample cosine, mean-DNA cosine, exact biased RBF-MMD, and RFFTrace.
- All four same/cross-setting comparisons, with Top-1/3/5 and MRR.

## Aggregate Top-1 retrieval

Values are means over three seeds and both comparisons of each type.

| Method | R | Same setting | Cross setting |
| --- | ---: | ---: | ---: |
| Single-sample cosine | 1 | 0.417 | 0.444 |
| Mean-DNA cosine | 4 | 0.722 | 0.806 |
| Exact RBF-MMD | 4 | 0.750 | 0.778 |
| RFFTrace, D=64 | 4 | 0.750 | 0.806 |
| RFFTrace, D=256 | 4 | 0.750 | 0.806 |
| Mean-DNA cosine | 16 | 1.000 | 1.000 |
| Exact RBF-MMD | 16 | 1.000 | 1.000 |
| RFFTrace, D=64 | 16 | 1.000 | 1.000 |
| RFFTrace, D=256 | 16 | 1.000 | 1.000 |

The single-sample method always uses the first response, so its reported value is invariant to the
configured `R`. At `R=16`, the 256-feature RFF distance has mean correlation 0.975 with exact MMD
for same-setting comparisons and 0.966 for cross-setting comparisons. The corresponding values for
64 features are 0.953 and 0.948.

## Interpretation

The controlled signal is recovered as repeated generations accumulate, the same-setting leakage
guard works, cross-setting evaluation works, and the saved shared RFF maps approximate exact MMD
closely. Mean-DNA also succeeds because this diagnostic's categorical distributions have different
means; it is therefore a pipeline check, not a test of higher-order distribution shape.

The next empirical step is to replace only the generator and encoder with the manuscript's real
model roster and fixed response encoder while retaining the same manifests, cache format, split
policy, comparisons, and artifact bundle.
