# ATML PA3: Fine-Tuning, Visual Grounding and Test-Time Compute

Code, configurations, split IDs and machine-readable results for EE-5102/CS-6304 Programming
Assignment 3 (Fall 2026). Every experiment is a `python -m ...` command driven by a YAML config. Every
number in the report traces to a file under `results/` produced by one of the commands below, and
every figure and table is regenerated from those files by a `make_report_assets` script.

- **Task 1:** supervised fine-tuning of Qwen3-0.6B on MetaMathQA with Full FT, Full FT + L2-SP, LoRA-4,
  LoRA-16 and OPLoRA-16; benchmark retention, weight-space and activation diagnostics, and qualitative behaviour.
- **Task 2:** a SigLIP → MLP connector → Qwen3-0.6B vision-language model trained through a three-stage curriculum,
  evaluated for visual grounding with complementary pairs and image controls.
- **Task 3:** repeated sampling, confidence-based selection, answer consensus and adaptive stopping on the
  supplied Qwen2.5 / Qwen2.5-VL rollouts, plus a live comparison of parallel sampling, a longer response and
  sequential refinement.

All runs used free-tier Kaggle and Colab NVIDIA T4 GPUs (16 GB) in FP16.

---

## Repository layout

```
common/      shared infrastructure: config loading, paths, secrets, seeding, logging, resumable training loop
             (FP16 autocast, dynamic loss scaling, L2-SP), response-only losses, checkpointing, GitHub / Hugging
             Face sync, report table/figure helpers
task1/       data selection, LoRA/OPLoRA adapters, training and launcher, lm-eval evaluation, qualitative
             generation, diagnostics, report assets
task2/       COCO / VQAv2 data construction, SigLIP feature caching, VLM model, three-stage training,
             development and final evaluation, report assets
task3/       rollout loading and answer grading, offline selection / signal / adaptive engine, live
             generation, report assets
configs/     project.yaml (Hugging Face repo IDs, git identity) and one YAML per experiment
scripts/     environment setup, token / sync check, Task 1 smoke test, vLLM check
assets/      task1/first_plot_questions.yaml (qualitative prompts)
splits/      committed IDs: Task 1 training selection and probe set, Task 2 data selections, Task 3 splits and
             rollout orders
results/     committed logs, metrics and per-example outputs for every task
report/figs/ generated figures (PDF/PNG) and tables (LaTeX/CSV/Markdown) for the report
```

**Where artifacts live.** Code, configs, split IDs and results are in this repository. Model and adapter
weights, cached SigLIP features and stage exports are in a private Hugging Face model repo
(`project.hf_repo_id`). Resumable recovery checkpoints were kept in a second private repo
(`project.hf_ckpt_repo_id`, latest checkpoint per run only). Raw datasets and caches live in a per-platform
scratch directory (`/tmp/atml` on Kaggle, `/content/atml_scratch` on Colab) and are never committed.

---

## Setup

1. Set the Hugging Face repo IDs and git identity in `configs/project.yaml`.
2. Add two secrets on Kaggle (*Add-ons → Secrets*) or Colab (key icon):
   - `HF_TOKEN`: write access to the two Hugging Face repos;
   - `GH_TOKEN`: fine-grained, *Contents: read & write* on this repository only.
3. Notebook preamble (Kaggle shown; on Colab read the secrets with `google.colab.userdata.get`):
   ```python
   import os
   from kaggle_secrets import UserSecretsClient
   s = UserSecretsClient()
   for k in ("HF_TOKEN", "GH_TOKEN"):
       os.environ[k] = s.get_secret(k)
   !git clone https://github.com/FaaizKhan1999/atml-assignment3.git
   %cd atml-assignment3
   !bash scripts/setup.sh
   ```
   `setup.sh` installs `requirements.txt` (PyTorch is the platform's preinstalled CUDA build) and writes
   `results/env/<platform>-pip-freeze.txt`. Task-specific extras:
   `pip install -q -r requirements.txt -r requirements-eval.txt` (Task 1 evaluation) and
   `pip install -q -r requirements.txt -r requirements-task3.txt` (Task 3).
4. Optional checks: `python -m scripts.check_sync` (Hugging Face and GitHub access; records the
   environment) and `python -m scripts.smoke_task1` (throughput, memory and correctness checks on a T4).

Tokens are read from the environment only and never written to disk or `.git/config`. Jobs push results to
GitHub from a private per-process clone, so parallel jobs never conflict. A failed push only logs a warning;
the files remain on disk and go out with the next push.

**Environment.** The platforms' Python 3 images with their preinstalled PyTorch CUDA build, `transformers<5`, `lm_eval` (HF backend),
`math-verify`, `pyarrow`. Exact versions per platform are in `results/env/`, and every result file records the
package versions and GPU used.

---

## Task 1: LLM fine-tuning

| Step | Command | Hardware |
|---|---|---|
| Select the 40,000 MetaMathQA examples and the probe set | `python -m task1.prepare_data --push` | CPU |
| Pipeline test (6 updates, crash-and-resume from Hugging Face) | `python -m task1.launch --queue l2sp --queue oplora16 -- --dry-run` | T4×2 |
| Train all five runs (one queue per GPU) | `python -m task1.launch --queue full_ft l2sp lora16 --queue oplora16 lora4` | T4×2, ~8 h |
| Benchmarks and qualitative generations | `python -m task1.evaluate` (or `--parallel` on T4×2) | T4 |
| Weight-space and activation diagnostics | `python -m task1.diagnostics --push` | CPU or GPU |
| Benchmark table and annotation sheet | `python -m task1.summarize_eval --push` and `python -m task1.build_annotation --push` | CPU |
| Report figures and tables | `python -m task1.make_report_assets --excerpt QUESTION:FORMAT --push` | CPU |

**Configs:** `configs/task1/{base,full_ft,l2sp,lora4,lora16,oplora16,eval}.yaml`.

**Outputs:** `results/task1/<run>/` holds `config.yaml`, `train_log.jsonl` (loss, learning rate, gradient norm,
loss scale, skipped updates and timing per update; Ω and λΩ for L2-SP), `stages.json` and `summary.json`.
Evaluation outputs are `results/task1/eval/<condition>/` (aggregate and per-example outputs per benchmark),
`results/task1/qualitative/` and `results/task1/diagnostics/`. Weights are in the Hugging Face repo under
`task1/<run>/{midpoint,final}/`.

**Implementation notes:**
- Loss is next-token cross-entropy on response tokens and `<|im_end|>` only, summed and divided by the number
  of supervised tokens in the 32-example effective batch.
- The output projection is applied only at supervised positions, in checkpointed chunks. This gives the same
  loss as projecting every position but fits full fine-tuning on a 16 GB T4.
- Training prompts are tokenized exactly as at inference: the Qwen3 chat template with thinking disabled,
  including the empty `<think></think>` block. Leading newlines in 667 responses are stripped, exactly as
  Qwen3's chat template does; the template check matches all 40,000 examples.
- GSM8K-test decontamination uses NFKC normalization, lowercasing and collapsing non-alphanumeric runs, applied to
  `query` and `original_question`. It removed 0 of the first 40,002 rows; 2 rows were dropped for length.
- **L2-SP:** λ(θ − θ₀) is added to the unscaled gradients before clipping, which is equivalent to adding λΩ to the
  loss. θ₀ is an FP32 copy in pinned CPU memory, because a GPU copy does not fit next to full fine-tuning on a T4.
- **OPLoRA:** the bases come from an FP32 `gesvd` SVD of each starting weight, re-orthonormalized with QR. The
  projections act on the factors inside every forward pass, and the projected factors are exported and merged.
  LoRA-16 and OPLoRA-16 share identical initial factors.
- Adapters are merged into fresh FP16 backbones before evaluation. Merged-vs-unmerged logits are checked after
  training; see the `merge_check` entry in each adapter run's `summary.json`.
- **Evaluation** uses lm-evaluation-harness (HF backend, FP16, batch size 32; 8 for BoolQ, whose long passages
  exceed T4 memory at 32).
  - A small subclass passes `enable_thinking=False` to the chat template, and every logged GSM8K prompt is checked
    for the empty think block.
  - The tokenizer is always loaded from the original Hub repo.
  - vLLM was tested (`results/env/vllm_colab_tesla_t4.json`) but not used: it conflicts with the pinned
    `transformers`/`datasets` versions, and its FP16 greedy outputs differ slightly from the HF backend.
- Qualitative responses use one fixed greedy generation config, verified deterministic by generating twice. The
  manual annotations are in `results/task1/qualitative/annotation_sheet_annotated.csv`.
- **Timing:** two runs shared each T4×2 machine. `summary.json` records which runs ran concurrently.

---

## Task 2: Vision-language model and visual grounding

| Step | Command | Hardware |
|---|---|---|
| Build all data selections (COCO 2014 + VQAv2 annotations) | `python -m task2.prepare_data --push` | CPU |
| Download images, cache SigLIP features, upload to Hugging Face | `python -m task2.cache_features --push` | T4 |
| Smoke test (checks + a few updates per stage + timing projection) | `python -m task2.train --smoke` | T4 |
| Three-stage training (4,500 updates, resumable) | `python -m task2.train` | T4, ~1.5 h |
| Final evaluation (4 image conditions, bootstrap) | `python -m task2.final_eval --push` | T4 |
| Report figures and tables | `python -m task2.make_report_assets --push` | CPU |

**Configs:** `configs/task2/base.yaml` (data and features) and `configs/task2/train.yaml` (model, curriculum,
evaluation).

**Outputs:** selections in `splits/task2/`, data statistics in `results/task2/data/`, caching cost in
`results/task2/feature_cache.json`, training logs in `results/task2/main/` (per-stage training logs, `dev_log.jsonl`,
`stage{k}_summary.json`, `calibration.json`), and final evaluation in `results/task2/eval/main_stage3/`
(`predictions.jsonl`, `metrics.json`, `mismatch_map.json`).

**Implementation notes:**
- **Data and features:**
  - Training uses train2014. The five evaluation sets come from val2014 and are allocated so every image is used
    once; disjointness is checked.
  - Grounding pairs follow fixed label-quality rules, set in `configs/task2/base.yaml`:
    - *presence positives:* at least one non-crowd instance covering ≥1% of the image;
    - *presence negatives:* no annotation of the category, and no mention of it or its synonyms in the five captions;
    - *counting:* 1–4 non-crowd instances, each ≥0.5% of the image, with no crowd annotation; broccoli and carrot
      are excluded;
    - *split:* evaluation sets are half presence and half counting;
    - *training pools:* reuse the caption and VQA training images.
  - Items over the token caps are left out rather than truncated.
  - **VQAv2:** both images need ≥8/10 agreement on their own consensus answer, the consensus answers must differ,
    and both questions must have the same answer type. The training target is the most frequent original string
    in the consensus group.
  - SigLIP's final layer norm is verified to be applied exactly once (the error is 0). Features are cached in FP16.
- **Model and training:**
  - Visual tokens sit in the user turn before the question, rendered from the Qwen3 chat template. Without an image
    they are omitted and the text is unchanged.
  - **Connector calibration:** the output is scaled once at initialization so the mean visual-token RMS equals the
    mean RMS of Qwen input embeddings over training text; the rule and values are in `calibration.json`.
  - Caption examples use the instruction "Briefly describe this image."
  - **Optimizer and schedule:** each stage gets a new AdamW (the trainable set and learning rates change), with 5%
    warmup and cosine decay to 10% per stage. FP16 dynamic loss scaling starts at 1,024 (the default 65,536
    overflowed at every stage start in the smoke test) and is carried across stages.
  - **Sampling:** stages 2–3 draw each update's data from the seed and update index, so resumption is exact.
- **Evaluation:**
  - VQA answers are normalized with a port of the official VQA evaluation normalization, applied to every answer.
  - **Supplementary metric:** a clearly labelled "lead answer" metric scores only the leading yes/no word or first
    number. It separates answer format from content; it is not a protocol metric.

---

## Task 3: Test-time compute

The supplied rollouts (`qwen25-math/`, `qwen25vl-mathvista/`) are read from a private Kaggle dataset, found by
folder name under `/kaggle/input`, Google Drive or `$ATML_PA3_DATA`. They are not committed.

| Step | Command | Hardware |
|---|---|---|
| Offline experiments: Steps 1, 2, 3, 5, both datasets | `python -m task3.offline --push` | CPU, ~15 min |
| MathVista scorer sensitivity check | `python -m task3.offline --datasets mathvista --mathvista-scorer lenient --push` | CPU |
| Live generation (Step 4; `--limit 2` = smoke test) | `python -m task3.live --push` | T4, ~2 h |
| Report figures and tables | `python -m task3.make_report_assets --push` | CPU |

**Config:** `configs/task3/base.yaml`.

**Outputs:** `splits/task3/` (development / held-out / live IDs and the ten rollout orders per question), and
`results/task3/{math,mathvista,mathvista_lenient}/`:
- `answer_groups.csv.gz`: every rollout's answer group and grade, i.e. the fixed scorer;
- `summary.json`: curves, failure decomposition and bootstrap intervals;
- `signals.json`, `adaptive.json`, `selections.csv.gz`, `per_question.csv`, `early_stop_errors.csv`.

Live generation results are in `results/task3/live/`, with every call's prompt, seed, tokens, truncation flag,
extracted answer and time.

**Implementation notes:**
- **Loading:** per-token lists are read as flat Arrow buffers. Every list is checked against
  `n_generated_text_tokens`, and the attention counter is checked to be `[0, …, N−1]`.
- **Answer groups** are formed without gold answers, and the same rule grades each group.
  - *MATH:* Math-Verify equivalence, with whitespace-insensitive string matching first; groups are built greedily
    in rollout-ID order. 80 comparisons reached Math-Verify's time limit and count as not equivalent; they are
    listed in `summary.json`.
  - *MathVista:* MathVista's answer-type, choice and precision rules with the released `testmini.json` metadata.
    Multiple-choice answers are option letters, with MathVista's edit-distance fallback. A non-empty extraction the
    rules cannot normalize counts as `NO_ANSWER`, as MathVista's normalizer returns None (1,889 rollouts; mostly
    ranges and symbolic expressions). The lenient sensitivity scorer strips units, spacing commands and similar
    symbols before those rules; it is reported separately and never replaces the official results.
- **Selection rules:**
  - Score ties go to the earliest rollout in the current order. Vote ties go to the answer whose earliest
    supporter comes first. RWC uses average ranks for tied scores.
  - VisConf calibration uses only the current prefix's tokens.
  - For vote-based rules, the recorded candidate is the earliest rollout supporting the winning answer.
- **Adaptive stopping:** thresholds are chosen on the 200 development questions (cheapest within 1 pp of fixed-32
  self-consistency, with the always-32 fallback included), then frozen for the 800 held-out questions.
- **Live generation:**
  - Qwen2.5-3B-Instruct at revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, FP16, batch size 1, on the first
    50 held-out MATH questions.
  - Each call is seeded from (6304, question, strategy, call).
  - The shared initial response is charged to both parallel and refinement.

---

## Reproducibility

- **Seeds:** seed 6304 everywhere the assignment specifies one, and for data selection, sampling, bootstrap
  resampling and seed derivation. Task 3's rollout orders and splits are committed, and reruns stop if they differ.
- **Configs:** each run saves its resolved config next to its results.
- **Data checks:** Task 1's training selection is verified against committed index and token hashes before every
  run. Task 2's selections are committed with SHA-256 hashes in `results/task2/data/prepare_stats.json`.
- **Evaluation hygiene:** no checkpoint, threshold or hyperparameter was chosen using final-test or held-out
  results. Task 3's selection and stopping rules never use gold answers or unrevealed rollouts.

---

## Attribution and external code

**Libraries used as-is:**
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (EleutherAI): benchmark tasks,
  prompts, answer filters and scoring for Task 1. `task1/lmeval_runner.py` only wraps it, adding the non-thinking
  chat template and parsing its logged samples.
- [Math-Verify](https://github.com/huggingface/Math-Verify) (Hugging Face): mathematical answer equivalence for
  Task 3.
- Hugging Face `transformers`, `datasets`, `huggingface_hub` and `safetensors`; PyTorch; NumPy, SciPy, pandas,
  PyArrow and Matplotlib.

**Code ported or adapted from external sources:**
- `task2/vqa_norm.py`: answer normalization (punctuation handling, number words, articles and the contraction
  table) ported from the official VQA evaluation code (VQA API / `vqaEval.py`, visualqa.org). The consensus
  accuracy follows the formula in the assignment.
- `task3/answers.py`: MathVista answer normalization (multiple-choice letter mapping with an edit-distance fallback,
  integer/float/precision rules) adapted from the evaluation code of
  [MathVista](https://github.com/lupantech/MathVista). The `testmini.json` metadata is downloaded from that
  repository, and its SHA-256 is recorded.
- `task3/live.py`: the boxed-answer extraction follows the balanced-brace rule described in the supplied data
  README.

**Methods implemented from their papers (no code reused):** LoRA (Hu et al.), OPLoRA (Xiong & Xie), L2-SP
(Li et al.), self-certainty (Kang et al.), VisConf and rank-weighted consensus (Usman et al.), self-consistency
(Wang et al.).

**Models and data:**
- Models: Qwen3-0.6B, SigLIP base (`google/siglip-base-patch16-224`), Qwen2.5-3B-Instruct.
- Datasets: MetaMathQA, GSM8K, HellaSwag, BoolQ, ARC-Easy (via lm-evaluation-harness), COCO 2014 captions and
  instances, VQAv2 with complementary pairs, MathVista testmini metadata, and the course-supplied ATML-PA3-Data
  rollouts.
- `assets/task1/first_plot_questions.yaml`: the evaluation questions from the emergent-misalignment study
  (Betley et al., 2025). Only the 24 prompts are used; its judge prompts and sampling settings are ignored.

**Use of AI assistance:** the code in this repository was written with an LLM coding assistant (Claude, Anthropic),
as the assignment permits. All code was reviewed, run and debugged by the author, who is responsible for every
line. No generative AI was used to write the report's text, interpretation or analysis. The Task 1 qualitative
annotations are the author's own.
