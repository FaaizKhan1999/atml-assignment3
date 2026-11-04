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


def build_sft_example(tokenizer, question: str, response: str) -> dict:
    prompt_ids = tokenizer(generation_prompt(tokenizer, question), add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"] + [end_of_turn_id(tokenizer)]
    return {
        "input_ids": prompt_ids + response_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + response_ids,
        "n_prompt": len(prompt_ids),
        "n_response": len(response_ids),
    }


def check_template(tokenizer, pairs: list[tuple[str, str]]) -> dict:
    """Compare our construction with the tokenizer's full-conversation rendering."""
    render_match = token_match = 0
    first_mismatch = None
    for q, r in pairs:
        prompt = generation_prompt(tokenizer, q)
        full = tokenizer.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": r}],
                                             tokenize=False, enable_thinking=False)
        expected = prompt + r + END_OF_TURN
        ok = full.rstrip("\n") == expected
        render_match += ok
        if not ok and first_mismatch is None:
            first_mismatch = {"template": full[-300:], "ours": expected[-300:]}
        ex = build_sft_example(tokenizer, q, r)
        joint = tokenizer(expected, add_special_tokens=False)["input_ids"]
        token_match += joint == ex["input_ids"]
    q0 = pairs[0][0]
    return {
        "n_checked": len(pairs),
        "generation_prompt_example": generation_prompt(tokenizer, q0),
        "prompt_ends_with_empty_think": generation_prompt(tokenizer, q0).endswith("<think>\n\n</think>\n\n"),
        "full_render_match_rate": render_match / len(pairs),
        "split_vs_joint_tokenization_match_rate": token_match / len(pairs),
        "first_render_mismatch": first_mismatch,
    }
