# Temporary local-model pilot setup

This is a real-generation engineering pilot to use while the final manuscript roster and prompts
are pending. Its results must remain labeled **temporary** and must not be presented as manuscript
evidence. The design leaves the RFFTrace method unchanged.

The prepared roster contains six ungated Apache-2.0 instruction checkpoints from SmolLM2, Qwen2.5,
and TinyLlama. Models are loaded sequentially, so only one checkpoint occupies memory at a time.
The smoke design stores eight generations per cell: two disjoint pools support evaluation through
`R=4`, while six identities make Top-5 non-trivial.

## 1. Create an isolated Python 3.12 environment

From Terminal:

```bash
cd /Users/bytedance/Documents/ChatGPT/DistDNA
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,local]'
```

The local extra is deliberately pinned for this pilot: PyTorch 2.13.0, Transformers 5.14.1,
Sentence Transformers 5.6.1, Hugging Face Hub 1.24.0, and Accelerate 1.14.0.

## 2. Verify Apple GPU support

```bash
python -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_available())'
```

The final line must be `True`. If it is not, stop before collection; do not silently switch the
experiment to CPU.

## 3. Pin every model to an immutable Hub commit

```bash
mkdir -p data/local-smoke
distdna resolve-models \
  --manifest configs/local-smoke.collection.json \
  --output data/local-smoke/collection.resolved.json
```

Use the resolved manifest for every later command. Keep it with the experiment artifacts. Do not
rerun resolution midway through a collection.

## 4. Collect stochastic responses locally

```bash
export HF_HOME="$PWD/data/hf-cache"
export PYTORCH_ENABLE_MPS_FALLBACK=1
distdna collect-local \
  --manifest data/local-smoke/collection.resolved.json \
  --cache-dir data/local-smoke/responses \
  --device mps \
  --dtype float16
```

The first pass downloads the six checkpoints. Collection writes and syncs each response before
continuing. If Terminal closes or you press Control-C, run the identical command again; completed
cells are skipped. Never edit `responses.jsonl` manually.

The expected record count is:

```bash
wc -l data/local-smoke/responses/responses.jsonl
```

It should report `864`.

## 5. Encode all responses with the fixed encoder

```bash
distdna encode-responses \
  --manifest data/local-smoke/collection.resolved.json \
  --cache-dir data/local-smoke/responses \
  --encoder sentence-transformer \
  --encoder-model sentence-transformers/all-mpnet-base-v2 \
  --device mps \
  --batch-size 32 \
  --output-dir data/local-smoke/embeddings
```

This command refuses an incomplete response cache and atomically publishes separate calibration
and evaluation tensors.

## 6. Validate and run the four-method comparison

```bash
distdna validate-config configs/local-smoke.pilot.json
distdna run configs/local-smoke.pilot.json
distdna summarize results/local-smoke-seed2027 \
  --output results/local-smoke-seed2027/summary.json
```

The run covers single-sample cosine, mean-DNA cosine, exact biased RBF-MMD, and RFFTrace; `R` is
`1, 2, 4`, `D` is `64, 256`, and all same/cross-setting comparisons are included. Same-setting
retrieval automatically uses disjoint response pools.

## 7. Freeze the execution environment

```bash
python -m pip freeze > data/local-smoke/environment.txt
python -c 'import platform, torch; print(platform.platform()); print(torch.__version__)' \
  > data/local-smoke/runtime.txt
```

Preserve these files together:

- `collection.resolved.json`
- `responses/manifest.json` and `responses/responses.jsonl`
- `embeddings/`
- `environment.txt` and `runtime.txt`
- `results/local-smoke-seed2027/`

## Scaling after the smoke run

Do not reuse the smoke cache under a changed manifest. For the next run, copy the collection
manifest to a new filename, set `generations` to `32`, expand the final prompt set if available,
resolve it to a new immutable manifest, and use a new cache/output directory. This supports
disjoint same-setting evaluation through `R=16`.
