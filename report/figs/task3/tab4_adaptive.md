**Adaptive sampling (held-out; settings chosen on the 200 development questions as the cheapest within 1 pp of fixed-32 self-consistency, then frozen). Both controllers return the current self-consistency answer; checks after 4, 8, 16 candidates, otherwise 32. Quality-aware stopping adds SC = RWC agreement and an RWC margin ≥ γ (scores: self_certainty for MATH, visconf for MathVista, recomputed on revealed candidates). Tokens are generated tokens of all consumed candidates (offline proxy).**

| Dataset | Policy | Setting (dev) | Accuracy % | Δ vs SC@32 [95% CI] | Tokens / q. | Rollouts | Stops at 4 / 8 / 16 / 32 (%) |
|---|---|---|---|---|---|---|---|
| MATH | Fixed SC, N = 4 | – | 69.2 | – | 2680 | 4 | – |
|  | Fixed SC, N = 8 | – | 71.8 | – | 5344 | 8 | – |
|  | Fixed SC, N = 16 | – | 73.1 | – | 10678 | 16 | – |
|  | Fixed SC, N = 32 | – | 73.8 | – | 21347 | 32 | – |
|  | Adaptive: agreement | τ = 0.625 | 73.6 | -0.1 [-0.5, +0.2] | 10140 | 11.8 | 66 / 6 / 1 / 27 |
|  | Adaptive: quality | τ = 0.5, γ = 0.0 | 73.2 | -0.6 [-1.2, -0.1] | 7663 | 9.0 | 76 / 6 / 1 / 16 |
| MathVista | Fixed SC, N = 4 | – | 57.8 | – | 1112 | 4 | – |
|  | Fixed SC, N = 8 | – | 59.9 | – | 2230 | 8 | – |
|  | Fixed SC, N = 16 | – | 60.9 | – | 4462 | 16 | – |
|  | Fixed SC, N = 32 | – | 61.7 | – | 8898 | 32 | – |
|  | Adaptive: agreement | τ = 0.5 | 61.3 | -0.4 [-1.6, +0.7] | 3290 | 9.9 | 71 / 8 / 2 / 19 |
|  | Adaptive: quality | τ = 0.5, γ = 0.0 | 61.6 | -0.1 [-1.1, +0.9] | 3613 | 10.9 | 65 / 10 / 3 / 22 |
