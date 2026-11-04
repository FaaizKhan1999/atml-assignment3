"""Select the 40,000 Task 1 training examples once, check formatting, and commit the IDs.

Writes
  splits/task1/metamath_selected.json   source indices + hashes (verified by every training run)
  splits/task1/probe_set.json           32 GSM8K + 32 HellaSwag doc IDs (seed 6304)
  results/task1/data/selection_stats.json  exclusion counts, lengths, template check on all examples

No GPU needed. Usage: python -m task1.prepare_data --push
"""
from __future__ import annotations

import argparse

from common.config import REPO_ROOT, add_config_args, load_config
from common.logging_utils import get_logger, package_versions, write_json
from common.model_utils import load_tokenizer
from common.chat_format import check_template
from task1.data import PROBE_FILE, SELECTION_FILE, make_probe_set, select_examples

log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, default="configs/task1/base.yaml")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)

    tok = load_tokenizer(cfg.model.name, revision=cfg.model.get("revision"))
    examples, indices, stats, raw = select_examples(cfg, tok, keep_raw=True)
    log.info("checking chat template on all %d selected examples", len(raw))
    template = check_template(tok, raw, strip_leading_newlines=cfg.data.strip_leading_newlines)
    log.info("render mismatches: %d, token mismatches: %d", template["n_render_mismatch"], template["n_token_mismatch"])

    write_json(SELECTION_FILE, {
        "dataset": cfg.data.dataset, "split": "train", "n": len(indices),
        "selection_rule": "source order; drop empty, GSM8K-test overlap (query/original_question), "
                          f"formatted length > {cfg.data.max_length}; keep first {cfg.data.n_examples}",
        "indices_sha256": stats["indices_sha256"], "tokens_sha256": stats["tokens_sha256"],
        "tokenizer": cfg.model.name, "packages": package_versions(["transformers", "tokenizers", "datasets"]),
        "indices": indices,
    })
    write_json(PROBE_FILE, make_probe_set(cfg.project.seed))
    stats_path = REPO_ROOT / "results" / "task1" / "data" / "selection_stats.json"
    write_json(stats_path, {**stats, "template_check": template})

    print(f"\nselected {stats['n_selected']} of {stats['rows_scanned']} rows scanned; excluded {stats['excluded']}")
    print(f"leading newlines stripped in {stats['n_responses_with_leading_newlines_stripped']} responses")
    print(f"template render mismatches after fix: {template['n_render_mismatch']} "
          f"(token-boundary differences: {template['n_token_mismatch']}, expected and harmless)")
    print(f"total input tokens {stats['total_input_tokens'] / 1e6:.2f}M, max length {stats['length_max']}")

    if args.push:
        from common.sync import push_results
        push_results([SELECTION_FILE.parent, stats_path.parent], "task1: training data selection and probe set")


if __name__ == "__main__":
    main()
