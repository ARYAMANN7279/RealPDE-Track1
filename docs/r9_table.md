## 70. ROUND 9 RAW RESULTS (Gemini, executed)

| Metric | u | v | Equal Weight |
|---|---|---|---|
| `g_shuffled` | 0.5081 | 0.6810 | **0.5945** |
| `g_ship` | 0.5719 | 0.7679 | **0.6699** |
| `g_ranked` | 0.5702 | 0.7796 | **0.6749** |
| `g_oracle` | 0.7927 | 0.9078 | **0.8503** |
| `g_const` | 0.5847 | 0.7206 | 0.6526 |
| `h_const` | 0.0074 | 0.0053 | - |
| `coverage` | 0.9140 | 0.9507 | - |
| `h_mean` | 0.0147 | 0.0065 | - |
| `h_median` | 0.0132 | 0.0053 | - |
| `spearman(h, e)` | 0.6572 | 0.5972 | - |

**Kit SPS Check:**
- `weighted` SPS score: 0.5081
- `coverage`: 0.9323
- `score_sps`: 50.8059

**Sanity Checks:**
- Processed 8 chunks (120 windows each, total 900) without triggering Trap-B.
- Information ladder assertion: `g_shuffled` ≤ `g_ship` ≤ `g_ranked` ≤ `g_oracle` passed.
