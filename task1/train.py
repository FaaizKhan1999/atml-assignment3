"""Task 1 training: one run per config (full_ft, l2sp, lora4, lora16, oplora16).

    python -m task1.train --config configs/task1/full_ft.yaml
    python -m task1.train --config configs/task1/oplora16.yaml --dry-run   # 6-update pipeline test

At every checkpoint update (312, 625, 937, 1250) the run
  1. exports stage weights if the update is an evaluation stage
     (full FT: midpoint + final; others: final) to the HF weights repo:
       full methods -> FP32 weights (needed for weight-space diagnostics)
       adapters     -> projected effective FP32 factors (+ OPLoRA bases)
  2. pushes results/task1/<run>/ (log, stages, config) to GitHub
  3. uploads the resumable recovery checkpoint to the HF checkpoint repo
in that order, so whatever a crashed session left behind is consistent.
After the final update, adapter runs merge their factors into a fresh FP16
backbone and compare logits with the unmerged model and the base model.

If a session dies, rerunning the same command resumes from the newest
recovery checkpoint (local, otherwise downloaded from Hugging Face).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from functools import partial

import torch

from common.checkpoint import save_tensors, trainable_state
from common.config import REPO_ROOT, add_config_args, load_config, save_config
from common.data_utils import SequentialWindows, collate_causal
from common.logging_utils import JsonlLogger, get_logger, read_json, run_metadata, write_json
from common.losses import TokenNormalizedObjective
from common.model_utils import (build_param_groups, count_parameters, enable_gradient_checkpointing,
                                load_causal_lm, load_tokenizer, logit_agreement, masked_logits)
from common.paths import get_paths
from common.profiling import dir_size_bytes
from common.seed import set_seed
from common.sync import (fetch_latest_recovery_checkpoint, hf_delete_dir, hf_repo, hf_upload_dir,
                         push_recovery_checkpoint, push_results)
from common.train_loop import L2SPRegularizer, ParamReference, Trainer
from task1.adapters import export_adapters, export_bases, inject_adapters, merge_exported_into, projection_residuals
from task1.data import select_examples, verify_selection

log = get_logger()


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def build_model(cfg):
    method = cfg.run.method
    dtype = torch.float32 if method == "full" else torch.float16
    t0 = time.perf_counter()
    model = load_causal_lm(cfg.model.name, revision=cfg.model.get("revision"), dtype=dtype)
    info = {"load_seconds": time.perf_counter() - t0, "dtype": str(dtype),
            "model_commit": getattr(model.config, "_commit_hash", None)}
    if cfg.training.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    modules = None
    if method == "lora":
        a = cfg.adapter
        modules, ainfo = inject_adapters(model, targets=a.targets, r=a.r, alpha=a.alpha, dropout=a.dropout,
                                         seed=cfg.project.seed, protect_rank=a.get("protect_rank"))
        info["adapter"] = ainfo
    return model, modules, info


class Run:
    """Holds everything for one training run; ``execute`` can be called twice to test resumption."""

    def __init__(self, cfg, args, examples):
        self.cfg, self.args, self.examples = cfg, args, examples
        self.name = cfg.run.name
        self.key = f"task1/{self.name}"
        self.paths = get_paths()
        self.res_dir = REPO_ROOT / "results" / "task1" / self.name
        self.res_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.paths.ckpt / "task1" / self.name
        self.stages = {int(k): v for k, v in cfg.run.stages.items()}
        self.window = cfg.training.microbatch * cfg.training.accum
        self.push_github = cfg.sync.push_github and not args.no_push and not args.dry_run
        self.push_hf = not args.no_push
        self.weights_repo = hf_repo("weights") if self.push_hf else None
        stages_path = self.res_dir / "stages.json"
        self.stage_records = read_json(stages_path) if stages_path.exists() else {}

    # ------------------------------------------------------------------ stage export
    def export_stage(self, update: int, stage: str) -> None:
        t0 = time.perf_counter()
        out = self.paths.artifacts / "task1" / self.name / stage
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        meta = {"run": self.name, "stage": stage, "update": update, "examples_seen": update * self.window,
                "method": self.cfg.run.method, "model": self.cfg.model.name,
                "model_commit": self.model_info.get("model_commit")}
        if self.cfg.run.method == "full":
            save_tensors(trainable_state(self.model), out / "weights.safetensors")
            meta["format"] = "full FP32 weights (tied embedding/lm_head stored once)"
            meta["displacement_l2"] = self.reference.displacement()
        else:
            factors, ameta = export_adapters(self.modules)
            save_tensors(factors, out / "adapter.safetensors", ameta)
            meta.update({"format": "effective FP32 adapter factors (projected for OPLoRA)", **ameta,
                         "targets": list(self.cfg.adapter.targets)})
            if ameta["projected"]:
                save_tensors(export_bases(self.modules), out / "bases.safetensors")
                meta["projection_check"] = projection_residuals(self.modules)
        meta["export_seconds"] = time.perf_counter() - t0
        meta["bytes"] = {p.name: dir_size_bytes(p) for p in out.iterdir()}
        write_json(out / "meta.json", meta)
        if self.push_hf:
            t1 = time.perf_counter()
            hf_upload_dir(out, f"{self.key}/{stage}", repo_id=self.weights_repo, message=f"{self.key}: {stage}")
            meta["upload_seconds"] = time.perf_counter() - t1
        self.stage_records[stage] = meta
        write_json(self.res_dir / "stages.json", self.stage_records)
        log.info("exported %s (update %d) in %.1fs", stage, update, meta["export_seconds"])

    def on_checkpoint(self, update: int, path, sizes: dict) -> None:
        if update in self.stages:
            self.export_stage(update, self.stages[update])
        if self.push_github:
            push_results([self.res_dir], f"{self.key}: update {update}")
        if self.push_hf and self.cfg.sync.push_recovery:
            push_recovery_checkpoint(path, self.key)

    # ------------------------------------------------------------------ merge check
    def merge_check(self) -> dict:
        probe = collate_causal(self.examples[:4], self.tok.pad_token_id)
        factors, meta = export_adapters(self.modules)
        unmerged = masked_logits(self.model, probe)
        self.trainer.optimizer.state.clear()
        _free()
        fresh = load_causal_lm(self.cfg.model.name, revision=self.cfg.model.get("revision"), dtype=torch.float16)
        base = masked_logits(fresh, probe)
        n = merge_exported_into(fresh, factors, meta["scaling"])
        merged = masked_logits(fresh, probe)
        effect = logit_agreement(unmerged, base)
        merge = logit_agreement(unmerged, merged)
        del fresh, base, merged, unmerged
        _free()
        return {"n_modules_merged": n, "merged_vs_unmerged": merge, "unmerged_vs_base": effect,
                "relative_merge_error": merge["mean_abs_diff"] / max(effect["mean_abs_diff"], 1e-12)}

    # ------------------------------------------------------------------ main
    def execute(self, stop_after: int | None = None) -> dict:
        cfg, tr = self.cfg, self.cfg.training
        set_seed(cfg.project.seed)
        self.tok = load_tokenizer(cfg.model.name, revision=cfg.model.get("revision"))
        self.model, self.modules, self.model_info = build_model(cfg)
        # theta0 must be captured BEFORE resuming loads trained weights
        self.reference = None
        if cfg.run.method == "full":
            lam = cfg.run.get("l2sp_lambda") or 0.0
            ref_device = cfg.run.get("ref_device", "cpu")
            self.reference = (L2SPRegularizer(self.model, lam=lam, ref_device=ref_device) if lam
                              else ParamReference(self.model, ref_device=ref_device))
        groups = build_param_groups([(n, p) for n, p in self.model.named_parameters() if p.requires_grad],
                                    lr=cfg.run.lr, weight_decay=tr.weight_decay)
        data = SequentialWindows(self.examples, microbatch=tr.microbatch, accum=tr.accum,
                                 collate_fn=partial(collate_causal, pad_id=self.tok.pad_token_id))
        self.trainer = Trainer(
            model=self.model, param_groups=groups, objective=TokenNormalizedObjective(), data=data,
            total_updates=tr.total_updates, warmup_updates=tr.warmup_updates, min_lr_ratio=tr.min_lr_ratio,
            grad_clip=tr.grad_clip, betas=tr.betas, eps=tr.eps, fused_adam=tr.fused_adam,
            l2sp=self.reference if isinstance(self.reference, L2SPRegularizer) else None,
            ckpt_dir=self.ckpt_dir, log_path=self.res_dir / "train_log.jsonl",
            checkpoint_at=tr.checkpoint_at, keep_last=1, on_checkpoint=self.on_checkpoint)
        fetch = (lambda: fetch_latest_recovery_checkpoint(self.key, self.paths.downloads)) if self.push_hf else None
        resumed = self.trainer.resume(fallback_fetch=fetch)
        resumed_from = self.trainer.update if resumed else 0
        self.trainer.train(stop_after=stop_after)
        if self.trainer.update < tr.total_updates:
            return {"stopped_at": self.trainer.update, "resumed_from": resumed_from}
        return self.finish(resumed_from)

    def finish(self, resumed_from: int) -> dict:
        records = JsonlLogger(self.res_dir / "train_log.jsonl").read()
        updates = [r for r in records if r.get("event") == "update"]
        ckpts = [r for r in records if r.get("event") == "checkpoint"]
        summary = {
            "run": self.name, "method": self.cfg.run.method, "lr": self.cfg.run.lr,
            "updates": self.trainer.update, "examples_seen": self.trainer.update * self.window,
            "resumed_from_update": resumed_from,
            "params": count_parameters(self.model), "model": self.model_info,
            "training": self.trainer.summary(),
            "skipped_updates": [r["update"] for r in updates if r.get("skipped")],
            "final_loss": updates[-1]["loss"] if updates else None,
            "checkpoints": {r["update"]: {k: r.get(k) for k in ("weights_bytes", "state_bytes", "total_bytes",
                                                                "save_seconds")} for r in ckpts},
            "stages": self.stage_records,
            "launch": json.loads(os.environ.get("ATML_LAUNCH_INFO", "{}")),
            "environment": run_metadata(),
        }
        if isinstance(self.reference, L2SPRegularizer):
            summary["l2sp"] = {"lambda": self.reference.lam, "ref_device": self.reference.ref_device,
                               "ref_bytes": self.reference.ref_bytes, "setup_seconds": self.reference.setup_seconds,
                               "total_penalty_seconds": sum(r.get("l2sp_seconds", 0.0) for r in updates),
                               "final_omega": updates[-1].get("omega") if updates else None}
        elif self.reference is not None:
            summary["reference"] = {"ref_device": self.reference.ref_device, "ref_bytes": self.reference.ref_bytes,
                                    "note": "theta0 copy used only to measure displacement"}
        if self.modules is not None:
            log.info("merge check")
            summary["merge_check"] = self.merge_check()
        write_json(self.res_dir / "summary.json", summary)
        if self.push_github:
            push_results([self.res_dir], f"{self.key}: finished")
        if self.push_hf and not self.args.dry_run:
            # everything is on Hugging Face now; free scratch disk for the next run in the queue
            shutil.rmtree(self.ckpt_dir, ignore_errors=True)
            shutil.rmtree(self.paths.artifacts / "task1" / self.name, ignore_errors=True)
        log.info("run %s finished: %d updates, %.2f h, peak %.2f GB", self.name, self.trainer.update,
                 summary["training"]["train_seconds"] / 3600, summary["training"]["peak_allocated_gb"])
        return summary


def apply_dry_run(cfg) -> None:
    cfg.run.name = f"{cfg.run.name}_dryrun"
    cfg.training.total_updates = 6
    cfg.training.warmup_updates = 2
    cfg.training.checkpoint_at = [3, 6]
    cfg.run.stages = {3: "midpoint", 6: "final"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--dry-run", action="store_true",
                        help="6 updates; stops at 3, deletes the local checkpoint, resumes from Hugging Face; "
                             "nothing is pushed to GitHub and the HF dry-run folders are deleted afterwards")
    parser.add_argument("--keep-dryrun", action="store_true", help="keep the dry-run HF folders")
    parser.add_argument("--no-push", action="store_true", help="no GitHub or Hugging Face uploads (local only)")
    parser.add_argument("--force", action="store_true", help="run even if a finished summary.json exists")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Training needs a GPU.")

    cfg = load_config(args.config, args.set)
    if args.dry_run:
        apply_dry_run(cfg)
    done = REPO_ROOT / "results" / "task1" / cfg.run.name / "summary.json"
    if not args.dry_run and not args.force and done.exists() \
            and read_json(done).get("updates") == cfg.training.total_updates:
        log.info("%s already finished (%s); skipping. Use --force to rerun.", cfg.run.name,
                 done.relative_to(REPO_ROOT))
        return
    set_seed(cfg.project.seed)
    tok = load_tokenizer(cfg.model.name, revision=cfg.model.get("revision"))
    examples, indices, stats = select_examples(cfg, tok)
    verify_selection(indices, stats)
    if len(examples) != cfg.training.total_updates * cfg.training.microbatch * cfg.training.accum and not args.dry_run:
        raise SystemExit("examples x updates mismatch")
    del tok

    run = Run(cfg, args, examples)
    save_config(cfg, run.res_dir / "config.yaml")
    if not args.dry_run:
        run.execute()
        return

    first = run.execute(stop_after=3)
    log.info("dry run: stopped at update %d; deleting local checkpoints to force a Hugging Face resume",
             first["stopped_at"])
    shutil.rmtree(run.ckpt_dir, ignore_errors=True)
    del run
    _free()
    run = Run(cfg, args, examples)
    summary = run.execute()
    ok = summary["resumed_from_update"] == 3 or args.no_push
    log.info("dry run %s: resumed from update %s, finished %d updates; stages exported: %s",
             "PASSED" if ok else "FAILED", summary["resumed_from_update"], summary["updates"],
             sorted(summary["stages"]))
    if summary.get("merge_check"):
        m = summary["merge_check"]
        log.info("merge check: argmax agreement %.4f, mean |dlogit| %.4f vs adapter effect %.4f (ratio %.3f)",
                 m["merged_vs_unmerged"]["argmax_agreement"], m["merged_vs_unmerged"]["mean_abs_diff"],
                 m["unmerged_vs_base"]["mean_abs_diff"], m["relative_merge_error"])
    if summary["stages"].get("final", {}).get("projection_check"):
        log.info("OPLoRA projection check: %s", summary["stages"]["final"]["projection_check"])
    if not args.keep_dryrun and not args.no_push:
        for repo in (hf_repo("weights"), hf_repo("ckpt")):
            hf_delete_dir(repo, run.key, message=f"remove {run.key}")
        log.info("deleted dry-run folders from Hugging Face")
    if not ok:
        raise SystemExit("dry run failed: did not resume from the Hugging Face checkpoint")


if __name__ == "__main__":
    main()
