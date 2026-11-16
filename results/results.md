# Results

Test split: 300 variants (P/LP, VUS, B/LB: 100 each).

| System | Macro-F1 [95% CI] | Accuracy [95% CI] | F1 P/LP | F1 VUS | F1 B/LB | Invalid | Median s/variant |
|---|---|---|---|---|---|---|---|
| TF-IDF + LR, name + abstracts | 0.647 [0.592, 0.699] | 0.653 [0.600, 0.707] | 0.75 | 0.58 | 0.62 | 0.0% | - |
| TF-IDF + LR, variant name only | 0.644 [0.590, 0.693] | 0.637 [0.583, 0.690] | 0.77 | 0.57 | 0.59 | 0.0% | - |
| Qwen2.5 7B (no evidence) | 0.499 [0.440, 0.555] | 0.533 [0.477, 0.587] | 0.64 | 0.57 | 0.29 | 0.0% | 6.6 |
| Qwen2.5 7B | 0.473 [0.417, 0.526] | 0.527 [0.470, 0.583] | 0.68 | 0.53 | 0.21 | 0.0% | 11.6 |
| Gemma 4 E4B | 0.415 [0.374, 0.456] | 0.503 [0.447, 0.560] | 0.65 | 0.57 | 0.02 | 0.0% | 9.7 |
| Qwen2.5 3B (no evidence) | 0.384 [0.343, 0.421] | 0.477 [0.420, 0.533] | 0.59 | 0.56 | 0.00 | 0.0% | 3.1 |
| Qwen2.5 3B | 0.351 [0.314, 0.387] | 0.440 [0.387, 0.493] | 0.60 | 0.46 | 0.00 | 0.0% | 5.0 |
| Gemma 4 E4B (no evidence) | 0.292 [0.247, 0.338] | 0.393 [0.340, 0.450] | 0.34 | 0.51 | 0.02 | 0.0% | 4.5 |
| Llama 3.1 8B | 0.229 [0.192, 0.269] | 0.363 [0.307, 0.420] | 0.53 | 0.11 | 0.04 | 0.0% | 13.4 |
| Llama 3.1 8B (no evidence) | 0.167 [0.146, 0.186] | 0.333 [0.280, 0.387] | 0.50 | 0.00 | 0.00 | 0.0% | 6.5 |
| Majority class | 0.167 [0.147, 0.186] | 0.333 [0.283, 0.387] | 0.00 | 0.00 | 0.50 | 0.0% | - |

## Paired comparisons (A - B)

Δ = metric(A) - metric(B) on the same 300 variants. Bootstrap p is one-sided (H1: A > B); McNemar p is two-sided, on per-variant correctness.

| A | B | Δ macro-F1 [95% CI] | Bootstrap p (A > B) | Δ accuracy | McNemar p |
|---|---|---|---|---|---|
| Qwen2.5 7B | TF-IDF + LR, name + abstracts | -0.175 [-0.243, -0.108] | 1.000 | -0.127 | 0.000598 |
| Qwen2.5 7B | TF-IDF + LR, variant name only | -0.171 [-0.239, -0.105] | 1.000 | -0.110 | 0.00269 |
| Gemma 4 E4B | Gemma 4 E4B (no evidence) | +0.123 [+0.072, +0.172] | 0.000 | +0.110 | 1.92e-05 |
| Llama 3.1 8B | Llama 3.1 8B (no evidence) | +0.062 [+0.030, +0.095] | 0.000 | +0.030 | 0.00391 |
| Qwen2.5 3B | Qwen2.5 3B (no evidence) | -0.032 [-0.079, +0.014] | 0.913 | -0.037 | 0.278 |
| Qwen2.5 7B | Qwen2.5 7B (no evidence) | -0.026 [-0.096, +0.039] | 0.790 | -0.007 | 0.913 |
