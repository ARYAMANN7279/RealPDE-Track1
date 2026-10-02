## 72. ROUND 10 RAW RESULTS (Gemini, executed)

### 7.1 & 7.2 & 7.3 `g` table (on `re_lohi`, exact quantile matching)

| quantity | shipped | Task A | Task B |
|---|---|---|---|
| `g_u` | 0.5720 | 0.5443 | 0.5702 |
| `g_v` | 0.7691 | 0.7028 | 0.7681 |
| `equal_weight_g` | **0.6705** | **0.6236** | **0.6692** |
| `R` | 1.0000 | 0.9299 | 0.9979 |
| `spearman(h_new, d)` (u) | 0.6653 | 0.4274 | 0.6364 |
| `spearman(h_new, d)` (v) | 0.5923 | 0.1700 | 0.5888 |
| `coverage` (u) | 0.9154 | 0.8837 | 0.9127 |
| `coverage` (v) | 0.9524 | 0.8816 | 0.9513 |
| KS distance (u) | 0.0 | 0.0 | 0.0 |
| KS distance (v) | 0.0 | 0.0 | 0.0 |
| `g_ranked` (ceiling) | u: 0.5799, v: 0.7908 | - | - |

**Sanity check:** `g_ranked` ≥ `g_shipped` holds in BOTH channels now (u: 0.5799 ≥ 0.5720, v: 0.7908 ≥ 0.7691).

### 7.3 Deployable 24-bin LUT (matched to shipped marginal)

| quantity | Task A (LUT) | Task B (LUT) |
|---|---|---|
| `equal_weight_g` | 0.6242 | 0.6694 |
| `R` (vs shipped exact) | 0.9308 | 0.9983 |

**Task B Inference Timing:**
- Without extra U-Net: 0.284 ms/window
- With extra U-Net: 0.416 ms/window
- Cost: +0.131 ms/window

### Task C (Held-out conditions)
- `train_es/soup_honest.pth`: Held out `re_lohi`
- `train_es/soup_lohi_honest.pth`: Held out `re_lohi`
- `train_es/joint_soup_aoa15.pth`: Held out `aoa15`
- `train_es/soup_v3.pth`: Held out `re_lohi`

**Gate Status:**
`R_TaskA` = 0.9299 (< 1.02)
`R_TaskB` = 0.9979 (< 1.02)
Route is closed.
