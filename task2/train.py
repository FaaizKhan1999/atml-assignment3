"""Task 2: train the SigLIP-Qwen VLM through the three-stage curriculum.

    python -m task2.train --smoke          # checks + a few updates per stage, timing projection (no push)
    python -m task2.train                  # the full 4,500-update main run (resumable)

Stages (one continuous model; a new AdamW per stage because the trainable set and learning rates
change; per-stage 5% warmup then cosine decay to 10% of peak):
  1. caption alignment   2,250 updates  connector only (two passes over 18,000 captions)
  2. visual instruction  1,500 updates  connector + LoRA (r16, q/k/v/o); 50% presence / 25% counting / 25% caption
  3. natural VQA           750 updates  connector + LoRA; 70% VQA / 10% presence / 10% counting / 10% caption

Every 250 main-run updates: resumable checkpoint (local + HF checkpoint repo). Development
evaluation before training, every 500 updates and at stage ends (2,250 / 3,750 / 4,500). After each
stage the connector and adapter weights go to the HF weights repo (task2/<run>/stage<k>/).
Rerunning the same command resumes; finished stages are skipped.
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import statistics
import time

import torch

from common.checkpoint import load_tensors, save_tensors
from common.config import REPO_ROOT, add_config_args, load_config, save_config
from common.logging_utils import JsonlLogger, get_logger, read_json, run_metadata, write_json
from common.model_utils import (build_param_groups, count_parameters, enable_gradient_checkpointing, load_causal_lm,
                                load_tokenizer)
from common.paths import get_paths
from common.publish import try_push
from common.seed import set_seed
from common.sync import fetch_latest_recovery_checkpoint, hf_download_dir, hf_repo, hf_upload_dir, push_recovery_checkpoint
from common.train_loop import Trainer
from task2.datasets import EpochSampler, FeatureStore, MixtureSampler, PromptBuilder, collate, ensure_features, load_split_examples
from task2.evaluation import dev_eval
from task2.model import VLM, Connector, VLMObjective, calibrate

log = get_logger()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


class Run:
    def __init__(self, cfg, args):
        self.cfg, self.args = cfg, args
        self.smoke = bool(args.smoke)
        self.name = "smoke" if self.smoke else cfg.run.name
        self.paths = get_paths()
        base = self.paths.scratch / "task2_smoke" if self.smoke else REPO_ROOT / "results" / "task2"
        self.res = base / self.name
        self.res.mkdir(parents=True, exist_ok=True)
        self.ckpt = self.paths.ckpt / "task2" / self.name
        self.export = self.paths.artifacts / "task2" / self.name
        self.push = not (self.smoke or args.no_push)
        self.key = f"task2/{self.name}"
        self.seed = int(cfg.data.seed)
        tr = cfg.training
        self.stages = list(tr.stages)
        self.offsets = [0]
        for s in self.stages:
            self.offsets.append(self.offsets[-1] + int(s.updates))
        total = self.offsets[-1]
        self.eval_points = {0, *range(int(tr.dev_eval_every), total + 1, int(tr.dev_eval_every)), *self.offsets[1:]}
        self.dev_log = JsonlLogger(self.res / "dev_log.jsonl")

        self.tok = load_tokenizer(cfg.model.name, revision=cfg.model.get("revision"))
        self.pad_id = self.tok.pad_token_id
        self.pb = PromptBuilder(self.tok, cfg.prompt.caption_instruction)
        self.store = FeatureStore(ensure_features())
        self.data = load_split_examples(self.pb)
        self.vlm = None
        self.lora_injected = False

    # ------------------------------------------------------------------ model
    def build(self) -> None:
        cfg = self.cfg
        torch.manual_seed(int(cfg.connector.init_seed))  # connector init first, before anything consumes RNG
        connector = Connector(self.store.dim, int(cfg.connector.hidden), 1024).cuda()
        lm = load_causal_lm(cfg.model.name, revision=cfg.model.get("revision"), dtype=torch.float16)
        for p in lm.parameters():
            p.requires_grad_(False)
        if cfg.training.gradient_checkpointing:
            enable_gradient_checkpointing(lm)
        if lm.config.hidden_size != 1024:
            raise ValueError(f"unexpected LM hidden size {lm.config.hidden_size}")
        self.vlm = VLM(lm, connector)
        self.model_commit = getattr(lm.config, "_commit_hash", None)
        cal_file = self.res / "calibration.json"
        if cal_file.exists():
            cal = read_json(cal_file)
            connector.out_scale.fill_(float(cal["scale"]))
        else:
            cal = self._calibrate()
            write_json(cal_file, cal)
        self.calibration = cal
        log.info("connector output scale %.4f (text RMS %.4f, initial visual RMS %.4f)", cal["scale"],
                 cal["text_token_rms"], cal["visual_token_rms_before"])

    def _calibrate(self) -> dict:
        c = self.cfg.connector.calibration
        import numpy as np
        g = np.random.default_rng([self.seed, 17])
        tr = self.data["train"]
        texts = tr["caption"] + [e for pair in tr["vqa"] + tr["presence"] + tr["counting"] for e in pair]
        pick = g.choice(len(texts), size=min(int(c.n_texts), len(texts)), replace=False)
        texts_ids = [texts[i]["target_ids"] + self.pb.tok(texts[i]["text"], add_special_tokens=False)["input_ids"]
                     for i in pick]
        keys = sorted({e["image"] for e in tr["caption"]})
        img = [keys[i] for i in g.choice(len(keys), size=min(int(c.n_images), len(keys)), replace=False)]
        return calibrate(self.vlm, self.store, texts_ids, img)

    def setup_stage(self, k: int) -> list[dict]:
        """Trainable set for stage k (1-based) and its AdamW parameter groups."""
        s = self.stages[k - 1]
        if s.lora_lr is not None and not self.lora_injected:
            from task1.adapters import inject_adapters
            lc = self.cfg.lora
            _, info = inject_adapters(self.vlm.lm, targets=lc.targets, r=lc.r, alpha=lc.alpha, dropout=lc.dropout,
                                      seed=self.seed)
            self.lora_info = info
            self.lora_injected = True
        wd = float(self.cfg.training.weight_decay)
        groups = build_param_groups(self.vlm.connector.named_parameters(), lr=float(s.connector_lr), weight_decay=wd,
                                    name="connector")
        if s.lora_lr is not None:
            lora = [(n, p) for n, p in self.vlm.lm.named_parameters() if p.requires_grad]
            groups += build_param_groups(lora, lr=float(s.lora_lr), weight_decay=wd, name="lora")
        return groups

    def sampler(self, k: int):
        tr, t = self.data["train"], self.cfg.training
        kw = dict(store=self.store, pb=self.pb, pad_id=self.pad_id, microbatch=int(t.microbatch), accum=int(t.accum),
                  seed=self.seed, stage=k)
        mix = dict(self.stages[k - 1].mixture)
        if k == 1:
            return EpochSampler(tr["caption"], **kw)
        return MixtureSampler({n: tr[n] for n in mix}, mix, **kw)

    # ------------------------------------------------------------------ weights
    def trainable_tensors(self) -> dict:
        out = {n: p.detach() for n, p in self.vlm.named_parameters() if p.requires_grad}
        out["connector.out_scale"] = self.vlm.connector.out_scale.detach()
        return out

    def export_stage(self, k: int) -> dict:
        out = self.export / f"stage{k}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        save_tensors(self.trainable_tensors(), out / "weights.safetensors")
        meta = {"stage": k, "name": self.stages[k - 1].name, "global_update": self.offsets[k],
                "lora": dict(self.cfg.lora) if self.lora_injected else None, "calibration": self.calibration,
                "model_commit": self.model_commit, "vision": dict(self.cfg.vision)}
        write_json(out / "meta.json", meta)
        if self.push:
            hf_upload_dir(out, f"{self.key}/stage{k}", repo_id=hf_repo("weights"), message=f"{self.key}: stage {k}")
        return meta

    def load_stage_weights(self, k: int) -> None:
        folder = self.export / f"stage{k}"
        if not (folder / "weights.safetensors").exists():
            folder = hf_download_dir(hf_repo("weights"), f"{self.key}/stage{k}", self.paths.downloads)
        saved = load_tensors(folder / "weights.safetensors")
        named = dict(self.vlm.named_parameters())
        named["connector.out_scale"] = self.vlm.connector.out_scale
        missing = [n for n in saved if n not in named]
        if missing:
            raise KeyError(f"stage {k} weights have unknown tensors: {missing[:3]}")
        with torch.no_grad():
            for n, t in saved.items():
                named[n].copy_(t.to(named[n].device, named[n].dtype))
        log.info("loaded stage %d weights (%d tensors)", k, len(saved))

    # ------------------------------------------------------------------ dev evaluation
    def logged_evals(self) -> set:
        return {r["global_update"] for r in self.dev_log.read()}

    def maybe_dev_eval(self, g: int, k: int) -> None:
        if g not in self.eval_points or g in self.logged_evals():
            return
        res = dev_eval(self.vlm, self.data["dev"], self.store, self.pb, self.pad_id)
        self.dev_log.log({"global_update": g, "stage": k, **res})
        log.info("dev @%d: caption %.3f | grounding NLL %.3f, dNLL %+.3f | VQA NLL %.3f, dNLL %+.3f (%.0fs)", g,
                 res["caption_loss"], res["grounding"]["nll_correct"], res["grounding"]["delta_nll"],
                 res["vqa"]["nll_correct"], res["vqa"]["delta_nll"], res["seconds"])

    # ------------------------------------------------------------------ stages
    def stage_done(self, k: int) -> bool:
        return (self.res / f"stage{k}_summary.json").exists()

    def run_stage(self, k: int, stop_after: int | None = None) -> dict:
        s, tr = self.stages[k - 1], self.cfg.training
        n, off = int(s.updates), self.offsets[k - 1]
        groups = self.setup_stage(k)
        every = int(tr.checkpoint_every)

        def on_ckpt(u, path, sizes):
            g = off + u
            self.maybe_dev_eval(g, k)
            if self.push and (g in self.eval_points or u == n):
                try_push([self.res], f"{self.key}: update {g}")
            if self.push and self.cfg.sync.push_recovery:
                push_recovery_checkpoint(path, f"{self.key}/stage{k}")

        trainer = Trainer(model=self.vlm, param_groups=groups, objective=VLMObjective(), data=self.sampler(k),
                          total_updates=n, warmup_updates=max(1, round(float(tr.warmup_frac) * n)),
                          min_lr_ratio=float(tr.min_lr_ratio), grad_clip=float(tr.grad_clip), betas=tr.betas,
                          eps=float(tr.eps), fused_adam=tr.fused_adam, ckpt_dir=self.ckpt / f"stage{k}",
                          log_path=self.res / f"train_log_stage{k}.jsonl",
                          checkpoint_at=[] if self.smoke else list(range(every, n + 1, every)),
                          checkpoint_final=not self.smoke, keep_last=1, on_checkpoint=on_ckpt)
        fetch = (lambda: fetch_latest_recovery_checkpoint(f"{self.key}/stage{k}", self.paths.downloads)) \
            if self.push else None
        resumed = trainer.resume(fallback_fetch=fetch)
        if not resumed and k > 1 and not self.smoke:
            self.load_stage_weights(k - 1)  # no-op content-wise if this process just trained stage k-1
        if not self.smoke:
            self.maybe_dev_eval(off + trainer.update, k)  # before training, or a point missed by a crash
        t0 = time.perf_counter()
        summary = trainer.train(stop_after=stop_after)
        wall = time.perf_counter() - t0
        if self.smoke or trainer.update < n:
            return {"trainer": trainer, "summary": summary, "wall": wall}
        meta = self.export_stage(k)
        recs = [r for r in trainer.logger.read() if r.get("event") == "update"]
        mix = {}
        for r in recs:
            for kk, v in (r.get("mixture") or {}).items():
                mix[kk] = mix.get(kk, 0) + v
        stage_summary = {
            "stage": k, "name": s.name, "updates": n, "global_end": off + n,
            "trainable_params": count_parameters(self.vlm)["trainable"],
            "trainable_by_part": {"connector": sum(p.numel() for p in self.vlm.connector.parameters()),
                                  "lora": sum(p.numel() for n_, p in self.vlm.lm.named_parameters() if p.requires_grad)},
            "training": summary, "skipped_updates": [r["update"] for r in recs if r.get("skipped")],
            "examples_by_component": mix, "lr": {"connector": s.connector_lr, "lora": s.lora_lr},
            "warmup_updates": max(1, round(float(tr.warmup_frac) * n)),
            "dev_eval_seconds": sum(r.get("seconds", 0) for r in self.dev_log.read() if r["stage"] == k),
            "export": meta, "environment": run_metadata()}
        write_json(self.res / f"stage{k}_summary.json", stage_summary)
        if self.push:
            try_push([self.res], f"{self.key}: stage {k} finished")
        return stage_summary


# ============================================================================ checks (smoke)

def run_checks(run: Run) -> dict:
    vlm, store, pb, data = run.vlm, run.store, run.pb, run.data
    out = {}
    tr_keys = [e["image"] for e in data["train"]["caption"]]
    for name in ("presence", "counting", "vqa"):
        tr_keys += [e[k] for pair in data["train"][name] for e in pair for k in ("image", "partner")]
    ev_keys = {}
    for split in ("dev", "test"):
        for name, items in data[split].items():
            ev = items if name == "caption" else [e for pair in items for e in pair]
            ev_keys[f"{split}_{name}"] = {e["image"] for e in ev}
    all_ev = [k for v in ev_keys.values() for k in v]
    out["split_isolation"] = {
        "train_all_train2014": all(k.startswith("train2014/") for k in tr_keys),
        "eval_all_val2014": all(k.startswith("val2014/") for k in all_ev),
        "eval_suites_disjoint": len(all_ev) == len(set(all_ev)),
        "train_eval_overlap": len(set(tr_keys) & set(all_ev)),
        "all_keys_cached": all(k in store for k in set(tr_keys) | set(all_ev))}
    f = store.get([tr_keys[0]])
    out["feature_shape"] = list(f.shape)
    exs = data["train"]["caption"][:2]
    b = collate(exs, store, pb, run.pad_id)
    b0 = collate(exs, store, pb, run.pad_id, images=[None, None])
    b = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in b.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        emb = vlm.embed(b)
        vis = vlm.connector(b["features"])
    slot = emb[b["image_mask"]].view(2, store.n_tokens, -1)
    out["splice_order_max_err"] = float((slot.float() - vis.float()).abs().max())
    out["connector_output_shape"] = list(vis.shape)
    out["no_image_length_diff"] = int(b["attention_mask"].sum() - b0["attention_mask"].sum())
    trainable = [n for n, p in vlm.named_parameters() if p.requires_grad]
    out["stage1_trainable_only_connector"] = all(n.startswith("connector.") for n in trainable)
    vlm.train()
    with torch.autocast("cuda", dtype=torch.float16):
        from task2.model import per_example_losses
        loss = per_example_losses(vlm, b)[0].mean()
    loss.backward()
    gn = {n: float(p.grad.norm()) for n, p in vlm.connector.named_parameters() if p.grad is not None}
    out["connector_grad_norms"] = gn
    out["connector_receives_gradient"] = bool(gn) and all(v > 0 and v == v for v in gn.values())
    vlm.zero_grad(set_to_none=True)
    ok = all(v for k, v in out["split_isolation"].items() if k != "train_eval_overlap") \
        and out["split_isolation"]["train_eval_overlap"] == 0 and out["splice_order_max_err"] < 1e-2 \
        and out["no_image_length_diff"] == 2 * store.n_tokens and out["stage1_trainable_only_connector"] \
        and out["connector_receives_gradient"] and out["feature_shape"] == [1, store.n_tokens, store.dim]
    out["all_passed"] = bool(ok)
    return out


def lora_check(run: Run) -> dict:
    vlm = run.vlm
    names = [n for n, p in vlm.named_parameters() if p.requires_grad]
    b = collate([e for pair in run.data["train"]["vqa"][:2] for e in pair], run.store, run.pb, run.pad_id)
    b = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in b.items()}
    from task2.model import per_example_losses
    with torch.autocast("cuda", dtype=torch.float16):
        per_example_losses(vlm, b)[0].mean().backward()
    gB = [float(p.grad.norm()) for n, p in vlm.named_parameters() if n.endswith("lora_B") and p.grad is not None]
    vlm.zero_grad(set_to_none=True)
    return {"trainable_are_connector_or_lora": all(n.startswith("connector.") or "lora_" in n for n in names),
            "n_lora_tensors": sum("lora_" in n for n in names),
            "lora_B_receives_gradient": bool(gB) and max(gB) > 0}


# ============================================================================ main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, default="configs/task2/train.yaml")
    parser.add_argument("--smoke", action="store_true", help="checks + a few updates per stage; nothing pushed")
    parser.add_argument("--smoke-updates", type=int, default=6)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Task 2 training needs a GPU.")
    cfg = load_config(args.config, args.set)
    set_seed(int(cfg.data.seed))
    run = Run(cfg, args)
    save_config(cfg, run.res / "config.yaml")
    run.build()

    if args.smoke:
        report = {"calibration": run.calibration, "checks": run_checks(run), "stages": {}}
        for k in range(1, len(run.stages) + 1):
            out = run.run_stage(k, stop_after=args.smoke_updates)
            if k == 2:
                report["lora_check"] = lora_check(run)
            recs = [r for r in out["trainer"].logger.read() if r.get("event") == "update"][1:]
            per = statistics.median(r["update_seconds"] for r in recs)
            report["stages"][k] = {"seconds_per_update": per,
                                   "projected_hours": per * int(run.stages[k - 1].updates) / 3600,
                                   "peak_allocated_gb": out["summary"]["peak_allocated_gb"],
                                   "trainable_params": count_parameters(run.vlm)["trainable"],
                                   "losses": [round(r["loss"], 4) for r in recs],
                                   "mixture_last": recs[-1].get("mixture") if recs else None}
            del out
            _free()
        t0 = time.perf_counter()
        dev = dev_eval(run.vlm, run.data["dev"], run.store, run.pb, run.pad_id)
        report["dev_eval"] = {"seconds": time.perf_counter() - t0, "caption_loss": dev["caption_loss"],
                              "grounding_nll": dev["grounding"]["nll_correct"], "vqa_nll": dev["vqa"]["nll_correct"]}
        n_evals = len(run.eval_points)
        report["projected_total_hours"] = sum(v["projected_hours"] for v in report["stages"].values()) + \
            n_evals * report["dev_eval"]["seconds"] / 3600
        path = write_json(REPO_ROOT / "results" / "task2" / "smoke.json", report)
        c = report["checks"]
        print("\n" + "=" * 80)
        print(f"checks passed: {c['all_passed']}  | split isolation {c['split_isolation']}")
        print(f"splice order err {c['splice_order_max_err']:.1e}, no-image length diff {c['no_image_length_diff']}, "
              f"connector grad {c['connector_receives_gradient']}, LoRA: {report.get('lora_check')}")
        print(f"calibration: scale {run.calibration['scale']:.4f} (text RMS {run.calibration['text_token_rms']:.4f}, "
              f"visual RMS {run.calibration['visual_token_rms_before']:.4f} -> {run.calibration['visual_token_rms_after']:.4f})")
        for k, v in report["stages"].items():
            print(f"stage {k}: {v['seconds_per_update']:.2f} s/update -> {v['projected_hours']:.2f} h, "
                  f"peak {v['peak_allocated_gb']:.2f} GB, trainable {v['trainable_params']:,}, losses {v['losses']}")
        print(f"dev eval {report['dev_eval']['seconds']:.0f} s x {n_evals}; projected total "
              f"{report['projected_total_hours']:.2f} h")
        print("=" * 80)
        log.info("wrote %s (commit it if you want the smoke results on GitHub)", path)
        return

    for k in range(1, len(run.stages) + 1):
        if run.stage_done(k):
            log.info("stage %d already finished; skipping", k)
            continue
        run.run_stage(k)
        _free()
    log.info("all stages finished")


if __name__ == "__main__":
    main()
