**BoolQ answer balance (validation, 3,270 questions; gold “yes” share 62.2%). Predictions are the higher-likelihood choice; balanced accuracy averages accuracy over gold-yes and gold-no questions.**

| Condition | Predicts “yes” (%) | Acc. on gold-yes (%) | Acc. on gold-no (%) | Balanced acc. (%) | Acc. (%) |
|---|---|---|---|---|---|
| Baseline | 57.7 | 67.8 | 58.8 | 63.3 | 64.4 |
| Full FT (mid) | 33.9 | 47.2 | 88.0 | 67.6 | 62.6 |
| Full FT | 31.5 | 44.4 | 89.7 | 67.1 | 61.5 |
| Full FT + L2-SP | 32.9 | 45.5 | 87.9 | 66.7 | 61.6 |
| LoRA-4 | 43.4 | 57.1 | 79.1 | 68.1 | 65.4 |
| LoRA-16 | 57.7 | 73.7 | 68.6 | 71.1 | 71.7 |
| OPLoRA-16 | 54.8 | 71.4 | 72.4 | 71.9 | 71.8 |
