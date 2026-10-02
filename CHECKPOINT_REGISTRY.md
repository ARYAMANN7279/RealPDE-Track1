# CHECKPOINT PROVENANCE REGISTRY

## Summary Counts
- **ALL-DATA**: 35
- **HONEST**: 43
- **INIT-LEAK**: 4
- **UNKNOWN**: 30
- **EVERY5**: 11
- **EVERY9**: 4
- **KIT**: 2

## HONEST Checkpoints
- `ftaug_r13_wd1e3_s0.pth`
- `ftaug_mem6_3e5_10.pth`
- `ftaug_r13_control_s1.pth`
- `ftaug_J_noaug.pth`
- `ftaug_r2w015.pth`
- `ftaug_r2w008.pth`
- `ftaug_r15_ema_s0.pth`
- `ftaug_mem7_1e5_08.pth`
- `ftaug_mem1_1e5_08.pth`
- `ftaug_r13_control_s0.pth`
- `ftaug_r14_bsp_t01_s0.pth`
- `ftaug_wtke010.pth`
- `ftaug_r13_wd1e3_s1.pth`
- `ftaug_r13_drop01_s0.pth`
- `ftaug_r15_ema_s2.pth`
- `ftaug_wtke015.pth`
- `ftaug_r15_ema_s3.pth`
- `ftaug_r13_ema9999_s0.pth`
- `ftaug_r2w003.pth`
- `ftaug_wtke005.pth`
- `ftaug_r13_ema9999_s1.pth`
- `ftaug_mem3_1e5_10.pth`
- `ftaug_mem5_3e5_05.pth`
- `ftaug_r14c_bsp_0005_s0.pth`
- `ftaug__smoke14.pth`
- `ftaug_r12_fno_official.pth`
- `ftaug_r13_ema999_s0.pth`
- `ftaug_r14c_bsp_002_s0.pth`
- `ftaug_sweep_2.pth`
- `ftaug_sweep_3.pth`
- `ftaug_r27_ema30k_s0.pth`
- `ftaug_wtke000.pth`
- `ftaug_mem2_1e5_05.pth`
- `ftaug_sweep_1.pth`
- `ftaug_I_phasenoise.pth`
- `ftaug__smoke13b.pth`
- `ftaug_r2w005.pth`
- `ftaug__smoke13.pth`
- `ftaug_H_phase.pth`
- `ftaug_r15_ema_s1.pth`
- `ftaug_mem4_3e5_08.pth`
- `ftaug_r14b_bsp_05_s0.pth`
- `ftaug_sweep_4.pth`

## INIT-LEAK Checkpoints (Dangerous)
- `ftaug_sv4_m3.pth`
- `ftaug_sv4_m4.pth`
- `ftaug_sv4_m2.pth`
- `ftaug_sv4_m1.pth`

## Registry

| file | bytes | split | init (raw) | init ROOT | steps | lr | wtke | ema | aug | seed | LEAKAGE | in a scored artifact? | USABLE AS BLEND PARTNER? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| soup_v3b.pth | 402777285 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **ALL-DATA** | no | YES |
| ftaug_r13_wd1e3_s0.pth | 402777845 | re_lohi | kit | KIT | 6000 | 3e-05 | 0.15 | 0.0 | none | 0 | **HONEST** | no | HONEST-ONLY |
| ftaug_sv4_m3.pth | 402777509 | re_lohi | init_sv4.pth | init_sv4.pth -> soup_fno_fp16.pth -> SOUP(soup_fno_fp16.pth) | 6000 | 1e-05 | 1.0 | UNKNOWN | phase | 403 | **INIT-LEAK** | no | NO |
| ftaug_mem6_3e5_10.pth | 402777789 | re_lohi | kit | KIT | 16000 | 3e-05 | 0.1 | UNKNOWN | none | 106 | **HONEST** | no | HONEST-ONLY |
| ftaug_r13_control_s1.pth | 402777957 | re_lohi | kit | KIT | 6000 | 3e-05 | 0.15 | 0.0 | none | 1 | **HONEST** | no | HONEST-ONLY |
| ftaug_sv3_m6.pth | 402777509 | none | kit | KIT | 40000 | 0.0005 | 0.08 | UNKNOWN | phase | 47 | **ALL-DATA** | no | YES |
| ftaug_J_noaug.pth | 402777565 | re_lohi | kit | KIT | 30000 | 3e-05 | 0.15 | UNKNOWN | none | UNKNOWN | **HONEST** | no | HONEST-ONLY |
| e2e_e2e_sps_10k_best.pth | 416443889 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| e2e_e2e_v1_best.pth | 416443319 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ftaug_sv3_m5.pth | 402777509 | none | kit | KIT | 40000 | 0.0005 | 0.08 | UNKNOWN | phase | 46 | **ALL-DATA** | no | YES |
| ftaug_sv3_m4.pth | 402777509 | none | kit | KIT | 40000 | 0.0005 | 0.08 | UNKNOWN | phase | 45 | **ALL-DATA** | no | YES |
| ftaug__sm24.pth | 402777453 | none | kit | KIT | 30 | 3e-05 | 0.1 | 0.0 | none | 1 | **ALL-DATA** | no | YES |
| ftaug_r2w015.pth | 402777509 | re_lohi | kit | KIT | 16000 | 1e-05 | 0.15 | UNKNOWN | none | UNKNOWN | **HONEST** | no | HONEST-ONLY |
| ftaug_sv3_1e5_10.pth | 402777733 | none | kit | KIT | 16000 | 1e-05 | 0.1 | UNKNOWN | none | 203 | **ALL-DATA** | no | YES |
| e2e_sv3_v2.pth | 416442749 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ftaug_r2w008.pth | 402777509 | re_lohi | kit | KIT | 16000 | 1e-05 | 0.08 | UNKNOWN | none | UNKNOWN | **HONEST** | no | HONEST-ONLY |
| ftaug_r15_ema_s0.pth | 402777733 | re_lohi | kit | KIT | 12000 | 3e-05 | 0.15 | 0.9999 | none | 0 | **HONEST** | no | HONEST-ONLY |
| ftaug_sv3_3e5_08.pth | 402777733 | none | kit | KIT | 16000 | 3e-05 | 0.08 | UNKNOWN | none | 204 | **ALL-DATA** | no | YES |
| ftaug_mem7_1e5_08.pth | 402777789 | re_lohi | kit | KIT | 16000 | 1e-05 | 0.08 | UNKNOWN | none | 107 | **HONEST** | no | HONEST-ONLY |
| ftaug_sv3_proper_m6.pth | 402777901 | none | kit | KIT | 40000 | 1e-05 | 0.08 | UNKNOWN | phase | 306 | **ALL-DATA** | member of sv3 | YES |
| r13_A_s0.pth | 402777285 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ftaug_mem1_1e5_08.pth | 402777789 | re_lohi | kit | KIT | 16000 | 1e-05 | 0.08 | UNKNOWN | none | 101 | **HONEST** | no | HONEST-ONLY |
| e2e_e2e_w96_fixed_best.pth | 433331221 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ftaug_r24_a100_m2.pth | 402777789 | none | kit | KIT | 16000 | 3e-05 | 0.05 | 0.0 | none | 502 | **ALL-DATA** | no | YES |
| ftaug_r13_control_s0.pth | 402777957 | re_lohi | kit | KIT | 6000 | 3e-05 | 0.15 | 0.0 | none | 0 | **HONEST** | no | HONEST-ONLY |
| r15_ema_soup.pth | 402776165 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **ALL-DATA** | no | YES |
| ftaug_r14_bsp_t01_s0.pth | 402777957 | re_lohi | kit | KIT | 6000 | 3e-05 | 0.15 | 0.0 | none | 0 | **HONEST** | no | HONEST-ONLY |
| ftaug_wtke010.pth | 402777565 | re_lohi | kit | KIT | 12000 | 3e-05 | 0.1 | UNKNOWN | none | UNKNOWN | **HONEST** | no | HONEST-ONLY |
| ftaug_r13_wd1e3_s1.pth | 402777845 | re_lohi | kit | KIT | 6000 | 3e-05 | 0.15 | 0.0 | none | 1 | **HONEST** | no | HONEST-ONLY |
| soup_sv3.pth | 402775877 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **ALL-DATA** | member of sv3 | YES |
| ftaug_r13_drop01_s0.pth | 402777901 | re_lohi | kit | KIT | 6000 | 3e-05 | 0.15 | 0.0 | none | 0 | **HONEST** | no | HONEST-ONLY |
| ftaug_r15_ema_s2.pth | 402777733 | re_lohi | kit | KIT | 12000 | 3e-05 | 0.15 | 0.9999 | none | 2 | **HONEST** | no | HONEST-ONLY |
| ftaug_sv3_m2.pth | 402777509 | none | kit | KIT | 40000 | 0.0005 | 0.08 | UNKNOWN | phase | 43 | **ALL-DATA** | no | YES |
| ftaug_sv3_m1.pth | 402777509 | none | kit | KIT | 40000 | 0.0005 | 0.08 | UNKNOWN | phase | 42 | **ALL-DATA** | no | YES |
| ftaug_wtke015.pth | 402777565 | re_lohi | kit | KIT | 12000 | 3e-05 | 0.15 | UNKNOWN | none | UNKNOWN | **HONEST** | no | HONEST-ONLY |
| ftaug_sv3_3e5_05.pth | 402777733 | none | kit | KIT | 16000 | 3e-05 | 0.05 | UNKNOWN | none | 205 | **ALL-DATA** | no | YES |
| ftaug_r15_ema_s3.pth | 402777733 | re_lohi | kit | KIT | 12000 | 3e-05 | 0.15 | 0.9999 | none | 3 | **HONEST** | no | HONEST-ONLY |
| soup_v3.pth | 402776653 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **ALL-DATA** | no | YES |
| r33_best.pth | 402775877 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ftaug_r13_ema9999_s0.pth | 402777957 | re_lohi | kit | KIT | 6000 | 3e-05 | 0.15 | 0.9999 | none | 0 | **HONEST** | no | HONEST-ONLY |
| e2e_tv_fno.pth | 402776693 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ftaug_sv4_m4.pth | 402777509 | re_lohi | init_sv4.pth | init_sv4.pth -> soup_fno_fp16.pth -> SOUP(soup_fno_fp16.pth) | 6000 | 1e-05 | 1.0 | UNKNOWN | phase | 404 | **INIT-LEAK** | no | NO |
| ftaug_r24_a100_m1.pth | 402777789 | none | kit | KIT | 16000 | 3e-05 | 0.1 | 0.0 | none | 501 | **ALL-DATA** | no | YES |
| r28_best.pth | 402775941 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ftaug_r2w003.pth | 402777509 | re_lohi | kit | KIT | 16000 | 1e-05 | 0.03 | UNKNOWN | none | UNKNOWN | **HONEST** | no | HONEST-ONLY |
| r31_best.pth | 402775877 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ftaug_wtke005.pth | 402777565 | re_lohi | kit | KIT | 12000 | 3e-05 | 0.05 | UNKNOWN | none | UNKNOWN | **HONEST** | no | HONEST-ONLY |
| ftaug_sv3_m7.pth | 402777509 | none | kit | KIT | 40000 | 0.0005 | 0.12 | UNKNOWN | phase | 48 | **ALL-DATA** | no | YES |
| ftaug_r29_ema100_d9999.pth | 402778133 | none | kit | KIT | 16000 | 3e-05 | 0.15 | 0.9999 | none | 602 | **ALL-DATA** | member of submission_BLIND.zip | YES |
| ftaug_r13_ema9999_s1.pth | 402777957 | re_lohi | kit | KIT | 6000 | 3e-05 | 0.15 | 0.9999 | none | 1 | **HONEST** | no | HONEST-ONLY |
| ftaug_mem3_1e5_10.pth | 402777789 | re_lohi | kit | KIT | 16000 | 1e-05 | 0.1 | UNKNOWN | none | 103 | **HONEST** | no | HONEST-ONLY |
| ftaug_sv3_proper_m3.pth | 402777901 | none | kit | KIT | 40000 | 1e-05 | 0.08 | UNKNOWN | phase | 303 | **ALL-DATA** | member of sv3 | YES |
| soup_honest.pth | 402776045 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **ALL-DATA** | no | YES |
| ftaug_sv3_1e5_05.pth | 402777733 | none | kit | KIT | 16000 | 1e-05 | 0.05 | UNKNOWN | none | 202 | **ALL-DATA** | no | YES |
| r13_A_s1.pth | 402777285 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ftaug_mem5_3e5_05.pth | 402777789 | re_lohi | kit | KIT | 16000 | 3e-05 | 0.05 | UNKNOWN | none | 105 | **HONEST** | no | HONEST-ONLY |
| ftaug_r14c_bsp_0005_s0.pth | 402778133 | re_lohi | kit | KIT | 3000 | 3e-05 | 0.15 | 0.0 | none | 0 | **HONEST** | no | HONEST-ONLY |
| ftaug_r29_ema100_d999.pth | 402778013 | none | kit | KIT | 16000 | 3e-05 | 0.15 | 0.999 | none | 601 | **ALL-DATA** | member of submission_BLIND.zip | YES |
| r25_best.pth | 402775941 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ftaug__smoke14.pth | 402777621 | re_lohi | kit | KIT | 60 | 3e-05 | 0.15 | 0.0 | none | 0 | **HONEST** | no | HONEST-ONLY |
| ftaug_r12_fno_official.pth | 402778133 | re_lohi | kit | KIT | 4000 | 0.0001 | 0.0 | UNKNOWN | none | 0 | **HONEST** | no | HONEST-ONLY |
| r16_greedy_soup.pth | 402776333 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **ALL-DATA** | no | YES |
| ftaug_r13_ema999_s0.pth | 402777901 | re_lohi | kit | KIT | 12000 | 3e-05 | 0.15 | 0.999 | none | 0 | **HONEST** | no | HONEST-ONLY |
| ftaug_r14c_bsp_002_s0.pth | 402778013 | re_lohi | kit | KIT | 3000 | 3e-05 | 0.15 | 0.0 | none | 0 | **HONEST** | no | HONEST-ONLY |
| ftaug_sweep_2.pth | 402777565 | re_lohi | kit | KIT | 8000 | 1e-05 | 0.08 | UNKNOWN | phase | 202 | **HONEST** | no | HONEST-ONLY |
| ftaug_sweep_3.pth | 402777565 | re_lohi | kit | KIT | 8000 | 3e-05 | 0.12 | UNKNOWN | phase | 203 | **HONEST** | no | HONEST-ONLY |
| ftaug_sv4_m2.pth | 402777509 | re_lohi | init_sv4.pth | init_sv4.pth -> soup_fno_fp16.pth -> SOUP(soup_fno_fp16.pth) | 6000 | 1e-05 | 1.0 | UNKNOWN | phase | 402 | **INIT-LEAK** | no | NO |
| ftaug_sv3_proper_m5.pth | 402777901 | none | kit | KIT | 40000 | 1e-05 | 0.08 | UNKNOWN | phase | 305 | **ALL-DATA** | member of sv3 | YES |
| ftaug_r27_ema30k_s0.pth | 402777901 | re_lohi | kit | KIT | 30000 | 3e-05 | 0.15 | 0.9999 | none | 0 | **HONEST** | no | HONEST-ONLY |
| r23_best.pth | 402775877 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ftaug_wtke000.pth | 402777565 | re_lohi | kit | KIT | 12000 | 3e-05 | 0.0 | UNKNOWN | none | UNKNOWN | **HONEST** | no | HONEST-ONLY |
| ftaug_sv3_3e5_10.pth | 402777733 | none | kit | KIT | 16000 | 3e-05 | 0.1 | UNKNOWN | none | 206 | **ALL-DATA** | member of submission_SCREEN.zip | YES |
| ftaug_sv3_1e5_08.pth | 402777733 | none | kit | KIT | 16000 | 1e-05 | 0.08 | UNKNOWN | none | 201 | **ALL-DATA** | no | YES |
| ftaug_mem2_1e5_05.pth | 402777789 | re_lohi | kit | KIT | 16000 | 1e-05 | 0.05 | UNKNOWN | none | 102 | **HONEST** | no | HONEST-ONLY |
| ftaug_sweep_1.pth | 402777565 | re_lohi | kit | KIT | 8000 | 3e-05 | 0.08 | UNKNOWN | phase | 201 | **HONEST** | no | HONEST-ONLY |
| ftaug_I_phasenoise.pth | 402777845 | re_lohi | kit | KIT | 30000 | 3e-05 | 0.15 | UNKNOWN | phase | UNKNOWN | **HONEST** | no | HONEST-ONLY |
| r19_screen_best.pth | 402776269 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ftaug__smoke13b.pth | 402777677 | re_lohi | kit | KIT | 60 | 3e-05 | 0.15 | 0.999 | none | 0 | **HONEST** | no | HONEST-ONLY |
| ftaug_r2w005.pth | 402777509 | re_lohi | kit | KIT | 16000 | 1e-05 | 0.05 | UNKNOWN | none | UNKNOWN | **HONEST** | no | HONEST-ONLY |
| ftaug_r12_armD_simfno_mse.pth | 402778301 | re_lohi | /SML_DISK_24TB/rajeshr/Aryamann/UGP/data/comp_real/sim_fno.pth | sim_fno.pth | 4000 | 0.0001 | 0.0 | UNKNOWN | none | 0 | **UNKNOWN** | no | UNKNOWN |
| ftaug__smoke13.pth | 402777621 | re_lohi | kit | KIT | 60 | 3e-05 | 0.15 | 0.999 | none | 0 | **HONEST** | no | HONEST-ONLY |
| ftaug_r24_a100_m3.pth | 402777789 | none | kit | KIT | 16000 | 1e-05 | 0.15 | 0.0 | none | 503 | **ALL-DATA** | no | YES |
| init_sv4.pth | 402775877 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ftaug_r12_armD_simfno_ours.pth | 402778357 | re_lohi | /SML_DISK_24TB/rajeshr/Aryamann/UGP/data/comp_real/sim_fno.pth | sim_fno.pth | 4000 | 0.0001 | 0.08 | UNKNOWN | none | 0 | **UNKNOWN** | no | UNKNOWN |
| ftaug_sv3_proper_m1.pth | 402777901 | none | kit | KIT | 40000 | 1e-05 | 0.08 | UNKNOWN | phase | 301 | **ALL-DATA** | member of sv3 | YES |
| ftaug_H_phase.pth | 402777565 | re_lohi | kit | KIT | 30000 | 3e-05 | 0.15 | UNKNOWN | phase | UNKNOWN | **HONEST** | no | HONEST-ONLY |
| ftaug_r15_ema_s1.pth | 402777733 | re_lohi | kit | KIT | 12000 | 3e-05 | 0.15 | 0.9999 | none | 1 | **HONEST** | no | HONEST-ONLY |
| fno_sv3_e2e_fp16.pth | 402722693 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ftaug_mem4_3e5_08.pth | 402777789 | re_lohi | kit | KIT | 16000 | 3e-05 | 0.08 | UNKNOWN | none | 104 | **HONEST** | no | HONEST-ONLY |
| ftaug_r14b_bsp_05_s0.pth | 402777957 | re_lohi | kit | KIT | 4000 | 3e-05 | 0.15 | 0.0 | none | 0 | **HONEST** | no | HONEST-ONLY |
| ftaug_sv4_m1.pth | 402777509 | re_lohi | init_sv4.pth | init_sv4.pth -> soup_fno_fp16.pth -> SOUP(soup_fno_fp16.pth) | 6000 | 1e-05 | 1.0 | UNKNOWN | phase | 401 | **INIT-LEAK** | no | NO |
| e2e_tv_best.pth | 416442863 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| r17_best_soup.pth | 402776221 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **ALL-DATA** | no | YES |
| ftaug_sv3_m3.pth | 402777509 | none | kit | KIT | 40000 | 0.0005 | 0.08 | UNKNOWN | phase | 44 | **ALL-DATA** | no | YES |
| ftaug_r24_a100_m4.pth | 402777789 | none | kit | KIT | 16000 | 3e-05 | 0.2 | 0.0 | none | 504 | **ALL-DATA** | no | YES |
| ftaug_sv3_1e5_08_b.pth | 402777845 | none | kit | KIT | 16000 | 1e-05 | 0.08 | UNKNOWN | none | 207 | **ALL-DATA** | no | YES |
| ftaug_sweep_4.pth | 402777565 | re_lohi | kit | KIT | 8000 | 3e-05 | 0.08 | UNKNOWN | none | 204 | **HONEST** | no | HONEST-ONLY |
| soup_lohi_honest.pth | 402776389 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **ALL-DATA** | no | YES |
| ft_w060_best.pth | 402777509 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **EVERY5** | no | YES |
| m55_fno.pth | 402774125 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ft_w015_best.pth | 402777509 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **EVERY5** | no | YES |
| ft_lr1e4_best.pth | 402777565 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **EVERY5** | no | YES |
| ft_all_w33_lr1_final.pth | 402777957 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| soup_moredata.pth | 402776925 | UNKNOWN | kit | KIT | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ft_w15lr3_best.pth | 402777621 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **EVERY5** | no | YES |
| ft_hetero_nll_final.pth | 402780479 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ft_lr1e5_best.pth | 402777565 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **EVERY5** | no | YES |
| ft_m55_w20lr5_best.pth | 402777845 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ft_lr3e5_best.pth | 402777565 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **EVERY5** | no | YES |
| ft_m55_w15lr4_best.pth | 402777845 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| soup_final_candidate.pth | 402777317 | UNKNOWN | kit | KIT | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ft_w010_best.pth | 402777509 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **EVERY5** | no | YES |
| ft_all_w15_lr3_final.pth | 402777957 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ft_md_w20lr2_best.pth | 402777789 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **EVERY9** | no | YES |
| sim_real_fno_unwrapped.pth | 402777429 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **KIT** | no | UNKNOWN |
| ft_md_w15lr1_best.pth | 402777789 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **EVERY9** | no | YES |
| ft_md_w15lr3_best.pth | 402777789 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **EVERY9** | no | YES |
| ft_m55_w15lr3_best.pth | 402777845 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ft_long_w20lr2_best.pth | 402777901 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **EVERY5** | no | YES |
| soup_best.pth | 402776701 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **ALL-DATA** | no | YES |
| ft_w005_best.pth | 402777509 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **EVERY5** | no | YES |
| ft_md_w10lr3_best.pth | 402777789 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **EVERY9** | no | YES |
| ft_long_w10lr1_best.pth | 402777901 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **EVERY5** | no | YES |
| ft_long_w15lr3_best.pth | 402777901 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **EVERY5** | member of submission_LONG80.zip | YES |
| soup_v2.pth | 402775245 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **ALL-DATA** | member of sv3 | YES |
| ft_all_w15_lr1_final.pth | 402777957 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| ft_all_w33_lr3_final.pth | 402777957 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| sim_fno.pth | 402988026 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **UNKNOWN** | no | UNKNOWN |
| sim_real_fno.pth | 402988026 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **KIT** | member of sv3 | UNKNOWN |
