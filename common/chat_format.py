"""Qwen3 chat formatting with thinking disabled.

Training examples are built as
    tokenize(generation_prompt(question)) + tokenize(response) + [<|im_end|>]
so the prompt tokens seen in training are exactly the tokens the model sees at
inference (the generation prompt is tokenized alone there too). With
enable_thinking=False, Qwen3's generation prompt ends in an empty
``<think>\\n\\n</think>\\n\\n`` block, which is therefore part of the masked
prompt. ``check_template`` verifies this against the tokenizer's own template.
"""
from __future__ import annotations

from .losses import IGNORE_INDEX

END_OF_TURN = "<|im_end|>"


def generation_prompt(tokenizer, user_text: str, system: str | None = None) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def end_of_turn_id(tokenizer) -> int:
    tid = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    if tid is None or tid == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer has no {END_OF_TURN} token")
    return tid


def clean_response(response: str, strip_leading_newlines: bool = True) -> str:
    """Qwen3's chat template renders assistant content with ``content.lstrip('\\n')``; do the same so
    training matches the model's own formatting (some MetaMathQA responses start with a newline)."""
    return response.lstrip("\n") if strip_leading_newlines else response


def build_sft_example(tokenizer, question: str, response: str, strip_leading_newlines: bool = True) -> dict:
    cleaned = clean_response(response, strip_leading_newlines)
    prompt_ids = tokenizer(generation_prompt(tokenizer, question), add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(cleaned, add_special_tokens=False)["input_ids"] + [end_of_turn_id(tokenizer)]
    return {
        "input_ids": prompt_ids + response_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + response_ids,
        "n_prompt": len(prompt_ids),
        "n_response": len(response_ids),
        "stripped_newlines": len(response) - len(cleaned),
    }


def check_template(tokenizer, pairs: list[tuple[str, str]], strip_leading_newlines: bool = True,
                   max_reported: int = 10) -> dict:
    """Compare our construction with the tokenizer's full-conversation rendering.

    ``render`` compares strings with the official template; ``token`` compares our split
    tokenization (prompt and response tokenized separately, as at inference) with tokenizing
    the joined string. Token mismatches at the prompt/response boundary are expected and harmless.
    """
    render_bad, token_bad = [], []
    for i, (q, r) in enumerate(pairs):
        prompt = generation_prompt(tokenizer, q)
        full = tokenizer.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": r}],
                                             tokenize=False, enable_thinking=False)
        expected = prompt + clean_response(r, strip_leading_newlines) + END_OF_TURN
        if full.rstrip("\n") != expected:
            render_bad.append(i)
        ex = build_sft_example(tokenizer, q, r, strip_leading_newlines)
        if tokenizer(expected, add_special_tokens=False)["input_ids"] != ex["input_ids"]:
            token_bad.append(i)
    examples = []
    for i in render_bad[:max_reported]:
        q, r = pairs[i]
        full = tokenizer.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": r}],
                                             tokenize=False, enable_thinking=False)
        expected = generation_prompt(tokenizer, q) + clean_response(r, strip_leading_newlines) + END_OF_TURN
        examples.append({"index": i, "template_tail": full[-200:], "ours_tail": expected[-200:],
                         "template_head": full[:200], "ours_head": expected[:200]})
    q0 = pairs[0][0]
    n = len(pairs)
    return {
        "n_checked": n,
        "strip_leading_newlines": strip_leading_newlines,
        "generation_prompt_example": generation_prompt(tokenizer, q0),
        "prompt_ends_with_empty_think": generation_prompt(tokenizer, q0).endswith("<think>\n\n</think>\n\n"),
        "full_render_match_rate": 1 - len(render_bad) / n,
        "n_render_mismatch": len(render_bad),
        "split_vs_joint_tokenization_match_rate": 1 - len(token_bad) / n,
        "n_token_mismatch": len(token_bad),
        "render_mismatches": examples,
        # kept for the smoke-test printout
        "first_render_mismatch": examples[0] if examples else None,
    }
