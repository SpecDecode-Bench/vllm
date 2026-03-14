# Speculative Decoding Profiling Suite

This directory contains scripts and tooling for benchmarking speculative decoding in vLLM across different models, methods, and workloads. The experiments are organized into **branches**, each representing a distinct profiling configuration (vLLM version, benchmark type, or hardware setup).

---

## Branch Overview

Each branch is a self-contained experiment. The scripts in each branch are tailored to a specific purpose:

| Branch | Type | Description |
|---|---|---|
| `perf/e2e-v0.10.1.1` | End-to-end | l3-8b, l3-70b (NR); qwen3-8b-thinking (R, thinking=True) on vLLM v0.10.1.1 |
| `perf/e2e-v0.11.1rc1` | End-to-end | qwen3-8b (4 NR datasets); MTP on gpqa-main and aime22-24 |
| `perf/e2e-dm` | End-to-end | l3-70b and qwen3-8b with draft model speculative decoding |
| `perf/tb-v0.10.1.1` | Time-breakdown | l3-8b, l3-70b, q3-8b (thinking disabled) on vLLM v0.10.1.1 |
| `perf/tb-dm` | Time-breakdown | l3-70b, q3-8b with draft model |
| `perf/acc-v0.10.1.1` | Acceptance rate | l3-8b, l3-70b, q3-8b-T (thinking enabled) on vLLM v0.10.1.1 |
| `perf/acc-dm` | Acceptance rate | l3-70b, q3-8b-T with draft model |

**NR** = non-reasoning workloads (instructcoder, cnndailymail, sharegpt, gsm8k)
**R** = reasoning workloads (aime, gpqa_main)

---

## Prerequisites

- `conda` (Miniconda or Anaconda)
- CUDA 12.8 at `/usr/local/cuda-12.8`
- HuggingFace model access for Llama-3 models
- ShareGPT dataset (auto-downloaded by `run-*.sh`; see below for manual setup)

---

## Setup: Building the Environment

Each branch should be installed into its own conda environment to isolate dependencies (different vLLM versions, patches, etc.).
This may take up to 30-50 minutes for the first time.

```bash
# Checkout the branch you want to run
git checkout perf/e2e-v0.10.1.1

# Set the path for this branch's environment and build
ENV_DIR=/path/to/.envs/e2e-v0.10.1.1 bash scripts/rebuild_env.sh
```

`rebuild_env.sh` will:
1. Create a conda environment at `ENV_DIR` with Python 3.10 (if it doesn't exist)
2. Install the current branch's vLLM in editable mode via `python -m pip install --editable .`

To rebuild (e.g. after pulling changes):
```bash
ENV_DIR=/path/to/.envs/e2e-v0.10.1.1 bash scripts/rebuild_env.sh
```

The environment is reused if it already exists, only the package reinstall runs.

---

## Running: End-to-End Benchmarks (perf/e2e-v0.10.1.1)

### Quick Experiment for AE Reviewers (recommended)

This script runs a quick experiment for AE reviewers. It will only run the first 100 requests from the gsm8k dataset, and only for the Llama-3.1-8B model. At the end of the experiment, it will generate a speedup figure, which is within 5% difference from Figure 1a. in the paper.

```bash
conda activate /path/to/.envs/e2e-v0.10.1.1
cd scripts/

bash run-l3-8b-quick.sh    # Llama-3.1-8B, only first 100 requests from gsm8k, 1 GPU, ~1 hour
```

### Quick start — run scripts

The `run-*.sh` scripts handle everything end-to-end: ShareGPT download,
warmup, all datasets × methods, and speedup figure generation.

```bash
conda activate /path/to/.envs/e2e-v0.10.1.1
cd scripts/

bash run-l3-8b.sh    # Llama-3.1-8B,  1 GPU, ~1.5 days
bash run-l3-70b.sh   # Llama-3-70B,   4 GPUs, ~1.5 days
```

Set `CUDA_VISIBLE_DEVICES` at the top of each script before running.
For a quick smoke test, set `num_reqs=5` and `batch_sizes="1 16"`.

ShareGPT is downloaded automatically on first run via `huggingface-cli`.
To use a pre-existing copy instead:
```bash
export SHAREGPT_PATH=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json
bash run-l3-8b.sh
```

Each run script:
- Downloads ShareGPT if not already present
- Creates a timestamped `results/run_<timestamp>/` output directory
- Copies itself there for reproducibility
- Runs a warmup pass, then iterates over datasets × methods × spec-token counts
- Logs per-run stdout to `<dataset>_<method>_<k>.log`
- Appends SUCCESS/FAILURE + elapsed time to `overview.log`
- Calls `vis_speedup.py` to generate PDF speedup figures

### Run scripts reference (perf/e2e-v0.10.1.1)

| Script | Model | GPU(s) | Methods |
|---|---|---|---|
| `run-l3-8b.sh` | Llama-3.1-8B-Instruct | 1 (GPU 0) | none, ngram(3,5†), eagle(3), eagle3(3) |
| `run-l3-70b.sh` | Meta-Llama-3-70B-Instruct | 4 (GPUs 0-3) | none, ngram(3,5†), eagle(3) |

† ngram k=5 is run on instructcoder only.

### Low-level profile scripts

For finer-grained control (individual dataset/method runs), the underlying
`profile-*.sh` scripts are also available:

| Script | Model | GPU(s) | Datasets | Methods |
|---|---|---|---|---|
| `profile-l3-8b.sh` | Llama-3.1-8B-Instruct | GPU 2 | instructcoder, cnndailymail, sharegpt, gsm8k | none, ngram(3), eagle3(3) |
| `profile-l3-70b.sh` | Meta-Llama-3-70B-Instruct | GPUs 4-7 | instructcoder, cnndailymail, sharegpt, gsm8k | none, ngram(3), eagle3(3) |
| `profile-q3-8b-thinking.sh` | Qwen/Qwen3-8B | GPU 3 | instructcoder, cnndailymail, sharegpt, gsm8k | none, ngram(3), eagle3(3) |
| `profile-q3-8b-thinking-aime-gpqa.sh` | Qwen/Qwen3-8B | GPU 3 | aime, gpqa_main | none, ngram(3), eagle3(3) |
| `profile-l3-8b-instructcoder-ngram5.sh` | Llama-3.1-8B-Instruct | GPU 3 | instructcoder | ngram(5) |
| `profile-l3-70b-instructcoder-ngram5.sh` | Meta-Llama-3-70B-Instruct | GPUs 4-7 | instructcoder | ngram(5) |

---

## Running: Acceptance Rate Benchmarks (perf/acc-v0.10.1.1)

The acceptance rate branches use `benchmarks/benchmark_throughput.py` directly. Scripts live in `benchmarks/`:

```bash
git checkout perf/acc-v0.10.1.1
conda activate /path/to/.envs/acc-v0.10.1.1
bash benchmarks/run-llama3.1-8b.sh
bash benchmarks/run-llama3-70b.sh
bash benchmarks/run-qwen3-8b.sh
```

These scripts measure throughput (tokens/sec) with `--enforce-eager` across datasets and speculative configs, logging to `results/run_<timestamp>/`.

---

## Core Scripts

### `bench_latency.py`

The main benchmarking driver (in `scripts/`). Called by all profile scripts.

```
python bench_latency.py \
  --model <hf-model-id> \
  --method <none|ngram|eagle|eagle3|draft_model> \
  --dataset <instructcoder|sharegpt|cnndailymail|gsm8k|aime|gpqa_main|gpqa_extended|gpqa_diamond> \
  --num_spec_tokens <N>     # -1 for no speculative decoding
  --num_reqs <N>            # number of requests (default: 500)
  --max_tokens <N>          # max output length in units of 1024 tokens (default: 32)
  --batch_sizes <B1 B2 ...> # batch sizes to sweep (default: 1 8 16 32 64 128)
  --results_dir <dir>
  [--is_warmup]             # warmup-only run, no results saved
```

Tensor parallelism is inferred automatically: 4 GPUs for 70B+ models, 1 GPU for smaller models.

### `common.py`

Shared argument parsing, dataset loading, and EAGLE model name resolution. Supported datasets:

| Key | Source |
|---|---|
| `instructcoder` | HuggingFace: `likaixin/InstructCoder` |
| `sharegpt` | Local path via `$SHAREGPT_PATH` env var (auto-downloaded by `run-*.sh`) |
| `cnndailymail` | HuggingFace: `abisee/cnn_dailymail` |
| `gsm8k` | HuggingFace: `openai/gsm8k` |
| `aime` | HuggingFace: `AI-MO/aimo-validation-aime` |
| `gpqa_main` | HuggingFace: `Idavidrein/gpqa` (gpqa_main subset) |

---

## Output Structure

```
scripts/results/
└── run_<timestamp>/
    ├── <profile-script>.sh        # copy of the script for reproducibility
    ├── warmup.log
    ├── <dataset>_<method>_<spec_tokens>.log
    └── overview.log               # SUCCESS/FAILURE + timing for each run
```

---

## Supported Speculative Decoding Methods

| Method | Description | EAGLE model resolved automatically? |
|---|---|---|
| `none` | Autoregressive baseline | N/A |
| `ngram` | N-gram prompt lookup | N/A |
| `eagle` | EAGLE draft model | Yes (per model) |
| `eagle3` | EAGLE3 draft model | Yes (per model) |
| `draft_model` | Generic draft model SD | Specify via `--draft_model` |
| `mtp` | MTP | N/A |

---

## Adding a New Branch

1. Create a new branch from `perf/main` (or the relevant vLLM version branch)
2. Copy `scripts/rebuild_env.sh` from `perf/e2e-v0.10.1.1`
3. Add `profile-<name>.sh` scripts for your target models/datasets/methods
4. Build the env: `ENV_DIR=... bash scripts/rebuild_env.sh`
5. Run the profile script: `bash profile-<name>.sh`