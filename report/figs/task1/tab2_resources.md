**Training cost and regularization (1,250 updates, 40,000 examples, one T4 per run; two runs shared each machine). Recovery checkpoint includes optimizer state. ‖θ−θ0‖₂ over all unique parameters at the final checkpoint. Final λΩ with λ = 10⁻³; for Full FT, in parentheses, the penalty its weights would incur (not optimized).**

| Method | Trainable params | Train time (h) | Peak GPU mem. (GB) | Recovery ckpt | Exported weights | ‖θ−θ0‖₂ | Final λΩ | Method-specific overhead |
|---|---|---|---|---|---|---|---|---|
| Full FT | 596.05M (100.00%) | 2.36 | 11.95 | 6.66 GB | 2.22 GB | 10.65 | (0.0567) | – |
| Full FT + L2-SP | 596.05M (100.00%) | 2.46 | 11.95 | 6.66 GB | 2.22 GB | 6.26 | 0.0196 | θ0 copy 2.22 GB (pinned CPU); penalty 5.6 min |
| LoRA-4 | 2.52M (0.42%) | 2.68 | 3.14 | 29.3 MB | 9.7 MB | – | – | export 0.0 s |
| LoRA-16 | 10.09M (1.67%) | 2.63 | 3.25 | 115.9 MB | 38.5 MB | – | – | export 0.1 s |
| OPLoRA-16 | 10.09M (1.67%) | 3.50 | 3.56 | 115.9 MB | 346.6 MB | – | – | SVD+QR 38.8 s; bases 308.0 MB; export 0.6 s |
