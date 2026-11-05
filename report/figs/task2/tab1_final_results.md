**Final-test results (stage-3 model; greedy, 16 new tokens). Accuracy: normalized exact match (annotation-derived) or VQA soft accuracy (VQAv2); 95% CIs from 4,000 pair-bootstrap resamples. By type: Annotation-derived: presence / counting; Natural VQAv2: yes/no / number / other. Pair acc.: both questions of a pair answered with their own canonical answer. Identical: both questions of a pair receive the same answer. Ref. NLL: teacher-forced reference-answer NLL (termination excluded). †Supplementary, not a protocol metric: only the leading yes/no word or first number is scored (closed-form types), separating answer format from content.**

| Suite | Image | Accuracy % [95% CI] | By type % | Pair acc. % | Identical % | Empty % | Ref. NLL | ms/ans. | Lead-ans. %† |
|---|---|---|---|---|---|---|---|---|---|
| Annotation-derived | Correct | 66.4 [57.0, 75.0] | 89 / 44 | 50.0 | 20.3 | 0.0 | 0.93 | 42 | 66.4 |
| (64 pairs) | Complementary | 10.9 [6.2, 16.4] | 11 / 11 | 0.0 | 20.3 | 0.0 | 4.02 | 30 | 10.9 |
|  | Mismatched | 37.5 [30.5, 45.3] | 58 / 17 | 9.4 | 64.1 | 0.0 | 2.38 | 30 | 37.5 |
|  | No image | 0.0 [0.0, 0.0] | 0 / 0 | 0.0 | 100.0 | 0.0 | 6.55 | 111 | 25.0 |
| Natural VQAv2 | Correct | 56.6 [50.0, 63.1] | 61 / 48 / 58 | 27.1 | 40.6 | 0.0 | 1.40 | 35 | 55.9 |
| (96 pairs) | Complementary | 27.1 [20.9, 33.5] | 48 / 19 / 11 | 5.2 | 40.6 | 0.0 | 2.85 | 35 | 36.5 |
|  | Mismatched | 35.4 [29.1, 42.1] | 59 / 30 / 16 | 8.3 | 56.2 | 0.0 | 2.34 | 39 | 47.0 |
|  | No image | 1.0 [0.0, 2.6] | 0 / 4 / 0 | 0.0 | 100.0 | 0.0 | 9.52 | 123 | 29.2 |
