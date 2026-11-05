**Automatic checks over all 24 qualitative prompts (8 questions × 3 formats) per condition. Math markers include LaTeX, equations, “The answer is” and “####”. Template keys complete: every non-optional key filled with a value other than the template's placeholder.**

| Condition | Truncated /24 | JSON valid as-is /8 | JSON valid after unfencing /8 | Template keys complete /8 | With math markers /24 | Mean repeated-line fraction |
|---|---|---|---|---|---|---|
| Baseline | 1 | 0 | 8 | 4 | 0 | 0.053 |
| Full FT (mid) | 6 | 2 | 2 | 7 | 15 | 0.176 |
| Full FT | 7 | 2 | 2 | 7 | 12 | 0.199 |
| Full FT + L2-SP | 2 | 2 | 2 | 7 | 13 | 0.038 |
| LoRA-4 | 1 | 3 | 3 | 6 | 2 | 0.052 |
| LoRA-16 | 4 | 3 | 3 | 6 | 4 | 0.069 |
| OPLoRA-16 | 3 | 3 | 3 | 6 | 9 | 0.048 |
