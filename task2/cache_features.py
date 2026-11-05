"""Download the selected COCO images and cache SigLIP patch features (GPU recommended; CPU works, slower).

    python -m task2.cache_features --push

Features: google/siglip-base-patch16-224, deterministic SiglipImageProcessor
preprocessing (RGB, resize to 224x224, rescale, normalize), the 196 final patch
features in 14x14 row-major order from ``last_hidden_state``. In Hugging Face's
SiglipVisionTransformer that output already includes the final
``post_layernorm``; the script verifies this numerically (it equals
post_layernorm(last encoder hidden state)) so the layer norm is applied exactly
once, and never uses the attention-pooling head. Computed in FP32, stored FP16.

Output (scratch, then uploaded to the HF weights repo under task2/features/siglip196/):
  features.npy   [N, 196, 768] float16
  index.json     {"train2014/<id>" | "val2014/<id>": row}
  meta.json      model, preprocessing, verification, timing
Timing and memory (image download and feature caching reported separately from
training) go to results/task2/feature_cache.json.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common.config import REPO_ROOT, add_config_args, load_config
from common.logging_utils import get_logger, read_json, run_metadata, write_json
from common.paths import get_paths
from common.publish import try_push
from common.sync import hf_repo, hf_upload_dir
from task2.download import download_images, image_path

log = get_logger()
SPLITS = REPO_ROOT / "splits" / "task2"
HF_PATH = "task2/features/siglip196"


def feature_dir():
    return get_paths().data / "task2" / "features" / "siglip196"


def main() -> None:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, SiglipVisionModel

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, default="configs/task2/base.yaml")
    parser.add_argument("--no-upload", action="store_true", help="keep features local (no HF upload)")
    parser.add_argument("--push", action="store_true", help="commit timing stats to GitHub")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    v = cfg.vision
    paths = get_paths()
    img_root = paths.data / "task2" / "images"
    out = feature_dir()
    out.mkdir(parents=True, exist_ok=True)

    ims = read_json(SPLITS / "images.json")
    items = [("train2014", i) for i in ims["train2014"]] + [("val2014", i) for i in ims["val2014"]]
    dl = download_images(items, img_root, cfg.data.coco_image_url)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(v.name, revision=v.get("revision"))
    model = SiglipVisionModel.from_pretrained(v.name, revision=v.get("revision"), torch_dtype=torch.float32).to(device).eval()
    commit = getattr(model.config, "_commit_hash", None)

    # ---- verify that last_hidden_state already contains post_layernorm (applied exactly once)
    probe = proc(images=[Image.open(image_path(img_root, *items[0])).convert("RGB")], return_tensors="pt")
    with torch.no_grad():
        o = model(pixel_values=probe["pixel_values"].to(device), output_hidden_states=True)
        manual = model.vision_model.post_layernorm(o.hidden_states[-1])
    ln_err = float((manual - o.last_hidden_state).abs().max())
    if ln_err > 1e-4 or o.last_hidden_state.shape[1:] != (v.n_tokens, v.dim):
        raise RuntimeError(f"unexpected SigLIP output: shape {tuple(o.last_hidden_state.shape)}, "
                           f"post-LN check error {ln_err:.2e}")

    keys = [f"{s}/{i}" for s, i in items]
    feats = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=np.float16,
                                      shape=(len(items), v.n_tokens, v.dim))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    bs = int(v.cache_batch_size)
    with torch.no_grad():
        for s in range(0, len(items), bs):
            batch = [Image.open(image_path(img_root, *it)).convert("RGB") for it in items[s:s + bs]]
            px = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
            feats[s:s + len(batch)] = model(pixel_values=px).last_hidden_state.float().cpu().numpy().astype(np.float16)
            if (s // bs) % 20 == 0:
                log.info("  features %d / %d", s + len(batch), len(items))
    feats.flush()
    cache_seconds = time.perf_counter() - t0
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 3) if device == "cuda" else None
    finite = bool(np.isfinite(np.asarray(feats[: min(256, len(items))], dtype=np.float32)).all())
    if not finite:
        raise RuntimeError("non-finite values in cached features")

    write_json(out / "index.json", {k: i for i, k in enumerate(keys)})
    meta = {"model": v.name, "model_commit": commit, "n_images": len(items), "shape": [len(items), v.n_tokens, v.dim],
            "dtype": "float16", "compute_dtype": "float32", "device": device,
            "preprocessing": {k: getattr(proc, k, None) for k in
                              ("do_resize", "size", "resample", "do_rescale", "rescale_factor", "do_normalize",
                               "image_mean", "image_std")},
            "post_layernorm_check_max_abs_err": ln_err, "uses_pooling_head": False,
            "token_order": "14x14 patches, row-major", "cache_seconds": cache_seconds, "peak_gpu_gb": peak,
            "bytes": (out / "features.npy").stat().st_size}
    write_json(out / "meta.json", meta)
    log.info("cached %d images in %.1f min (%.2f GB)", len(items), cache_seconds / 60, meta["bytes"] / 2**30)

    upload_seconds = None
    if not args.no_upload:
        t1 = time.perf_counter()
        hf_upload_dir(out, HF_PATH, repo_id=hf_repo("weights"), message="task2: SigLIP 196-token features")
        upload_seconds = time.perf_counter() - t1
    stats = {"image_download": dl, "feature_cache": {k: meta[k] for k in
                                                     ("n_images", "cache_seconds", "peak_gpu_gb", "bytes", "device",
                                                      "post_layernorm_check_max_abs_err", "model_commit")},
             "upload_seconds": upload_seconds, "hf_path": None if args.no_upload else HF_PATH,
             "environment": run_metadata()}
    path = write_json(REPO_ROOT / "results" / "task2" / "feature_cache.json", stats)
    print(f"\n{len(items)} images, features {meta['bytes'] / 2**30:.2f} GB, caching {cache_seconds / 60:.1f} min "
          f"on {device} (peak {peak} GB), post-LN check error {ln_err:.1e}")
    if args.push:
        try_push([path], "task2: feature cache stats")


if __name__ == "__main__":
    main()
