**Training cost (1,250 updates, 40,000 examples, one T4 per run; two runs shared each machine) and displacement ‖θ−θ0‖₂ over all unique parameters at the final checkpoint.**

| Method | Trainable params | Train time (h) | Peak GPU mem. (GB) | Exported weights | ‖θ−θ0‖₂ | Method-specific overhead |
|---|---|---|---|---|---|---|
| Full FT | 596.05M (100.00%) | 2.36 | 11.95 | 2.22 GB | 10.65 | – |
| Full FT + L2-SP | 596.05M (100.00%) | 2.46 | 11.95 | 2.22 GB | 6.26 | θ0 copy 2.22 GB (CPU); penalty 5.6 min |
| LoRA-4 | 2.52M (0.42%) | 2.68 | 3.14 | 9.7 MB | – | – |
| LoRA-16 | 10.09M (1.67%) | 2.63 | 3.25 | 38.5 MB | – | – |
| OPLoRA-16 | 10.09M (1.67%) | 3.50 | 3.56 | 346.6 MB | – | SVD+QR 39 s; bases 308.0 MB |
