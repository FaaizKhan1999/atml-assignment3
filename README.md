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
task1/         LLM fine-tuning (adapters.py: LoRA / OPLoRA)
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

## Reproducibility

Seed 6304 everywhere. Every run saves its resolved config next to its results.
`results/env/` records package versions and GPU per platform.

## Attribution

Will list any materially reused external code here.
