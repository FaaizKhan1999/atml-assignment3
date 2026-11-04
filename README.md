# ATML PA3 — Fine-tuning, VLM grounding, and test-time compute

Code for EE-5102/CS-6304 Programming Assignment 3. Every step is a `python -m ...`
command driven by a YAML config, so it runs identically on Kaggle and Colab
after cloning.

## Where things live

| Artifact | Location |
|---|---|
| Code, configs, split / example IDs | this repo |
| Logs, metrics, per-example outputs (small) | `results/` in this repo, pushed by the jobs |
| Model / adapter weights | private Hugging Face repo `project.hf_repo_id` |
| Resumable recovery checkpoints | private Hugging Face repo `project.hf_ckpt_repo_id` (latest per run only) |
| Datasets, caches, temporary files | platform scratch dir (`/tmp/atml` on Kaggle, `/content/atml_scratch` on Colab), never committed |

## Repository layout

```
common/        shared infrastructure (config, paths, secrets, seeding, logging,
               checkpointing, sync, losses, training loop, model utilities)
task1/         LLM fine-tuning: data.py (selection), adapters.py (LoRA / OPLoRA),
               prepare_data.py, train.py, launch.py
splits/        committed example IDs (training selection, probe sets)
scripts/       setup, environment and sync checks, smoke tests
configs/       project.yaml (edit once) and per-experiment configs
results/       committed, machine-readable results
```

## One-time setup

1. Edit `configs/project.yaml` (Hugging Face repo IDs, git name, GitHub no-reply email) and commit it.
2. Add two secrets on both platforms (Kaggle: *Add-ons → Secrets*; Colab: key icon):
   `HF_TOKEN` (write access to the two HF repos) and `GH_TOKEN` (fine-grained,
   *Contents: read & write* on this repo only).
3. Kaggle: enable *Internet* and a GPU accelerator (T4 or T4×2) in the notebook settings.

## Notebook preamble

**Colab**
```python
import os
from google.colab import userdata
for k in ("HF_TOKEN", "GH_TOKEN"):
    os.environ[k] = userdata.get(k)
!git clone https://github.com/<you>/<repo>.git
%cd <repo>
!bash scripts/setup.sh
```

**Kaggle**
```python
import os
from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
for k in ("HF_TOKEN", "GH_TOKEN"):
    os.environ[k] = s.get_secret(k)
!git clone https://github.com/<you>/<repo>.git
%cd <repo>
!bash scripts/setup.sh
```

Tokens are only ever read from the environment; they are never written to disk
or to `.git/config`.

## Phase 0 checks

```bash
python -m scripts.check_sync               # HF upload/delete + GitHub push; records results/env/
python -m scripts.smoke_task1 --push       # Task 1 throughput, memory and correctness checks
# separate, fresh session (vLLM may replace torch):
pip install -q vllm && python -m scripts.check_vllm
```

## Task 1: LLM fine-tuning

Configs: `configs/task1/{full_ft,l2sp,lora4,lora16,oplora16}.yaml` (shared recipe in `base.yaml`).

1. **Select training data once** (no GPU needed):
   `python -m task1.prepare_data --push` writes `splits/task1/metamath_selected.json`,
   `splits/task1/probe_set.json` and `results/task1/data/selection_stats.json`.
   Every training run re-derives the selection and fails if it differs from the committed hashes.
2. **Pipeline test** (T4x2, ~15-20 min): `python -m task1.launch --queue l2sp --queue oplora16 -- --dry-run`.
   Each dry run trains 6 updates, stops at update 3, deletes its local checkpoint, resumes from
   Hugging Face, finishes, runs the merge check, then deletes its Hugging Face folders.
3. **Training** (Kaggle T4x2, run in the background with *Save Version → Save & Run All*):
   `python -m task1.launch --queue full_ft l2sp lora16 --queue oplora16 lora4`.
   Rerunning the same command after an interruption resumes unfinished runs and skips finished ones.

Per run, `results/task1/<run>/` holds `config.yaml`, `train_log.jsonl` (loss, learning rate,
gradient norm, loss scale, skipped updates, timing; plus Omega for L2-SP), `stages.json`
(exported weights, displacement, OPLoRA projection checks) and `summary.json` (parameters,
time, memory, checkpoint sizes, method-specific overhead, merge check, concurrency).
Weights are in the HF weights repo under `task1/<run>/{midpoint,final}/`.

Implementation notes:
- Loss: next-token CE on response tokens and `<|im_end|>` only, summed and divided by the number of
  supervised tokens in the 32-example effective batch.
- L2-SP: lambda*(theta - theta0) is added to the unscaled gradients before clipping (equivalent to
  adding lambda*Omega to the loss); theta0 is an FP32 copy in pinned CPU memory because a GPU copy
  does not fit next to full fine-tuning on a 16 GB T4.
- OPLoRA: bases from FP32 `gesvd` SVD of each starting weight, re-orthonormalized with QR; the
  projections are applied to the factors in FP32 inside every forward pass, and the projected
  factors are what is exported and merged.
- Responses that start with newlines are left-stripped, exactly as Qwen3's chat template does;
  the count is recorded in `selection_stats.json`.

## Reproducibility

Seed 6304 everywhere. Every run saves its resolved config next to its results.
`results/env/` records package versions and GPU per platform.

## Attribution

Will list any materially reused external code here.
