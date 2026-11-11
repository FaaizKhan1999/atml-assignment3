**Live generation on 50 held-out MATH questions (Qwen2.5-3B-Instruct, FP16, batch 1, one T4; same 3,072-token output allowance). Means per question; the shared initial response is charged to both parallel and refinement, and refinement also pays its longer revision prompt. Time excludes model loading. Revision transitions vs the shared initial answer (initial accuracy 64%): wrong→correct 1, correct→wrong 1, answer changed 6. Paired bootstrap over questions (1,000 resamples).**

| Strategy | Accuracy % | Δ vs single [95% CI] | Input tok. | Output tok. | Total tok. | s / question | Truncated calls % | No extraction % |
|---|---|---|---|---|---|---|---|---|
| Single response (≤3,072) | 66.0 | – | 108 | 734 | 843 | 36.5 | 4.0 | 4.0 |
| Parallel ×3 (≤1,024) + SC | 62.0 | -4 [-16, +6] | 325 | 1821 | 2146 | 89.2 | 16.0 | 16.0 |
| Initial + critique-revision (≤2,048) | 64.0 | -2 [-12, +10] | 837 | 1117 | 1955 | 55.4 | 5.0 | 5.0 |
