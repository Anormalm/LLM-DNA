# Paper-faithful experiment protocol

This checklist is the executable interpretation of the RFFTrace method used by DistDNA.

1. Define model IDs, at least two decoding settings, and disjoint calibration/evaluation prompts
   in one immutable collection manifest.
2. Draw `R` independent stochastic responses for every model/setting/prompt cell. Store the full
   generation key, derived seed, and visible text in the resumable response cache.
3. Encode all responses with one pinned response encoder, preserving the canonical tensor order
   `[model, setting, prompt, generation, feature]`.
4. Normalize each individual response embedding consistently when configured.
5. Select the RBF bandwidth from calibration prompts only.
6. Compare single-sample cosine, mean-DNA cosine, exact biased RBF-MMD, and RFFTrace over the same
   generation-count sweep.
7. Sample each RFF map once and reuse it for every model, prompt, setting, and query/reference role.
   Concatenate prompt-wise mean features with the `1/sqrt(T)` scaling. If enabled, sample one shared
   fixed projection and save it with the RFF parameters.
8. Evaluate every requested same-setting and cross-setting retrieval comparison. Same-setting
   query/reference vectors must use disjoint response pools.
9. Report Top-k, MRR, per-query ranks, exact distance matrices, RFF approximation diagnostics,
   seeds, shapes, hashes, normalization, bandwidth, and the structural NA policy.
10. Repeat over seeds. Treat controlled diagnostics as implementation checks, not empirical claims
    about real language models.
11. Audit any externally collected response text before encoding. External records must preserve
    their actual generation seeds, source-file hashes, and immutable model revisions; incomplete or
    unknown-provenance cells remain quarantined rather than being silently backfilled or relabeled.

Scale is gated: first close a non-ceiling small-cohort comparison across point, mean, exact-MMD, and
RFFTrace methods; then repeat seeds; only then expand to the large model cohort.

The runner enforces these invariants at validation time and refuses to overwrite existing result
directories.
