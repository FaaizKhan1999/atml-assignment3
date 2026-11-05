"""Build all Task 2 data selections and commit them (CPU only; ~2 GB of annotation downloads).

    python -m task2.prepare_data --push
    python -m task2.prepare_data --local-dir /kaggle/input/vqa-v2 --push   # if the VQA site is unreachable

Writes splits/task2/ (committed; everything training and evaluation need besides image features):
  captions_train.json, captions_dev.json
  vqa_train.json, vqa_dev.json, vqa_test.json            pairs with all ten human answers
  grounding_train_pools.json, grounding_dev.json, grounding_test.json
  images.json                                            every image to download/cache, by split
and results/task2/data/prepare_stats.json (eligibility counts, exclusions, caps, disjointness check).
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from common.config import REPO_ROOT, add_config_args, load_config
from common.logging_utils import get_logger, package_versions, write_json
from common.model_utils import load_tokenizer
from common.paths import get_paths
from common.publish import try_push
from task2.data import (Caps, check_disjoint, eligible_vqa_pairs, grounding_index, load_coco, load_vqa, pair_images,
                        rng, sample_grounding, select_captions, select_vqa, sha)
from task2.download import get_json

log = get_logger()
SPLITS = REPO_ROOT / "splits" / "task2"
STATS = REPO_ROOT / "results" / "task2" / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, default="configs/task2/base.yaml")
    parser.add_argument("--local-dir", default=None, help="folder with pre-downloaded VQAv2/COCO JSONs or zips")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    d = cfg.data
    seed = int(d.seed)
    raw = get_paths().data / "task2" / "raw"
    local = Path(args.local_dir) if args.local_dir else None
    t0 = time.perf_counter()

    # --------------------------------------------------------------- downloads
    files = {}
    for split in ("train2014", "val2014"):
        files[f"captions_{split}"] = get_json(f"captions_{split}.json", d.coco_annotations_url, raw, local)
        files[f"instances_{split}"] = get_json(f"instances_{split}.json", d.coco_annotations_url, raw, local)
    s = {"train2014": "train", "val2014": "val"}
    for split, short in s.items():
        files[f"vqa_q_{split}"] = get_json(f"v2_OpenEnded_mscoco_{split}_questions.json", d.vqa_urls[f"questions_{short}"], raw, local)
        files[f"vqa_a_{split}"] = get_json(f"v2_mscoco_{split}_annotations.json", d.vqa_urls[f"annotations_{short}"], raw, local)
        files[f"vqa_p_{split}"] = get_json(f"v2_mscoco_{split}_complementary_pairs.json", d.vqa_urls[f"pairs_{short}"], raw, local)
    download_seconds = time.perf_counter() - t0
    log.info("annotations ready (%.0fs)", download_seconds)

    tok = load_tokenizer(cfg.model.name, revision=cfg.model.get("revision"))
    caps = Caps(tok, dict(d.caps))
    stats: dict = {"seed": seed, "caps": dict(d.caps), "download_seconds": download_seconds,
                   "packages": package_versions(["transformers", "tokenizers", "numpy"])}

    # --------------------------------------------------------------- training data (train2014)
    log.info("loading train2014 annotations")
    coco_tr = load_coco(files["captions_train2014"], files["instances_train2014"])
    vqa_tr = load_vqa(files["vqa_q_train2014"], files["vqa_a_train2014"], files["vqa_p_train2014"])
    c = Counter()
    elig_tr = eligible_vqa_pairs(vqa_tr, caps, d.vqa.min_agreement, c)
    stats["vqa_train_filter"] = dict(c, total_pairs=len(vqa_tr["pairs"]),
                                     eligible_by_type=dict(Counter(p["answer_type"] for p in elig_tr)))
    del vqa_tr
    vqa_train = select_vqa(elig_tr, dict(d.vqa.train), rng(seed, "vqa_train"), used=None)
    del elig_tr
    cap_train = select_captions(coco_tr, caps, d.captions.train_images, d.captions.captions_per_train_image,
                                rng(seed, "captions_train"), used=None)
    universe = sorted({x["image_id"] for x in cap_train} | set(pair_images(vqa_train)))
    gi = grounding_index(coco_tr, universe, d.grounding)
    pools = sample_grounding(gi, d.grounding.train_pool.presence, d.grounding.train_pool.counting,
                             rng(seed, "grounding_train"), used=None)
    stats["train"] = {"caption_images": len({x["image_id"] for x in cap_train}), "caption_examples": len(cap_train),
                      "vqa_pairs": len(vqa_train), "grounding_universe_images": len(universe),
                      "grounding_pool": dict(Counter(p["kind"] for p in pools)),
                      "grounding_categories": len({p["category"] for p in pools})}
    del coco_tr, gi

    # --------------------------------------------------------------- evaluation data (val2014, image-disjoint)
    log.info("loading val2014 annotations")
    coco_va = load_coco(files["captions_val2014"], files["instances_val2014"])
    vqa_va = load_vqa(files["vqa_q_val2014"], files["vqa_a_val2014"], files["vqa_p_val2014"])
    c = Counter()
    elig_va = eligible_vqa_pairs(vqa_va, caps, d.vqa.min_agreement, c)
    stats["vqa_val_filter"] = dict(c, total_pairs=len(vqa_va["pairs"]),
                                   eligible_by_type=dict(Counter(p["answer_type"] for p in elig_va)))
    del vqa_va
    used: set = set()
    vqa_test = select_vqa(elig_va, dict(d.vqa.test), rng(seed, "vqa_test"), used)
    vqa_dev = select_vqa(elig_va, dict(d.vqa.dev), rng(seed, "vqa_dev"), used)
    gi_va = grounding_index(coco_va, sorted(coco_va["captions"]), d.grounding)
    gr_test = sample_grounding(gi_va, d.grounding.test.presence, d.grounding.test.counting, rng(seed, "grounding_test"), used)
    gr_dev = sample_grounding(gi_va, d.grounding.dev.presence, d.grounding.dev.counting, rng(seed, "grounding_dev"), used)
    cap_dev = select_captions(coco_va, caps, d.captions.dev_images, 1, rng(seed, "captions_dev"), used)
    stats["disjointness"] = check_disjoint({
        "vqa_test": pair_images(vqa_test), "vqa_dev": pair_images(vqa_dev), "grounding_test": pair_images(gr_test),
        "grounding_dev": pair_images(gr_dev), "captions_dev": [x["image_id"] for x in cap_dev]})
    stats["eval"] = {"vqa_test_by_type": dict(Counter(p["answer_type"] for p in vqa_test)),
                     "vqa_dev_by_type": dict(Counter(p["answer_type"] for p in vqa_dev)),
                     "grounding_test": dict(Counter(p["kind"] for p in gr_test)),
                     "grounding_dev": dict(Counter(p["kind"] for p in gr_dev)),
                     "grounding_test_answers": dict(Counter(m["answer"] for p in gr_test for m in p["members"]))}

    # --------------------------------------------------------------- write
    train_ims = sorted({x["image_id"] for x in cap_train} | set(pair_images(vqa_train)) | set(pair_images(pools)))
    val_ims = sorted(used)
    out = {
        "captions_train.json": cap_train, "captions_dev.json": cap_dev,
        "vqa_train.json": vqa_train, "vqa_dev.json": vqa_dev, "vqa_test.json": vqa_test,
        "grounding_train_pools.json": pools, "grounding_dev.json": gr_dev, "grounding_test.json": gr_test,
        "images.json": {"train2014": train_ims, "val2014": val_ims},
    }
    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, obj in out.items():
        write_json(SPLITS / name, obj)
    stats["files_sha256"] = {name: sha(obj) for name, obj in out.items()}
    stats["n_images"] = {"train2014": len(train_ims), "val2014": len(val_ims)}
    stats["seconds_total"] = time.perf_counter() - t0
    write_json(STATS / "prepare_stats.json", stats)

    print(f"\ntrain: {len(cap_train)} caption examples ({stats['train']['caption_images']} images), "
          f"{len(vqa_train)} VQA pairs, grounding pool {stats['train']['grounding_pool']}")
    print(f"eval:  VQA test {stats['eval']['vqa_test_by_type']}, dev {stats['eval']['vqa_dev_by_type']}; "
          f"grounding test {stats['eval']['grounding_test']}, dev {stats['eval']['grounding_dev']}; "
          f"{len(cap_dev)} dev captions")
    print(f"images to cache: {len(train_ims)} train2014 + {len(val_ims)} val2014; suites image-disjoint: "
          f"{stats['disjointness']}")
    if args.push:
        try_push([SPLITS, STATS], "task2: data selections and splits")


if __name__ == "__main__":
    main()
