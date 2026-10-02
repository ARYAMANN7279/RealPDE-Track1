# PRE-SUBMISSION AUDIT RESULTS

**Date: 9 Sep 2026. Banked: 79.484440.**

## 3.1 Fingerprint Verification

All three candidates match their gated fingerprints exactly.

| zip | zip md5 | backbone md5 | result |
|---|---|---|---|
| submission_W73.zip | 3bdf77b97c3ef5b5c5c0e79865982d98 | 8d112b78550e87fa91241c15af8eae84 | MATCH |
| submission_CORNER3.zip | 7b6b125dfe27d7453ca82e4753530ea8 | fd4a4231a7402982d06c3917b96e05c8 | MATCH |
| submission_LONG80.zip | 6bae48b8b5ba1478614baf95fa10c6fc | 43c7823d7004c9a4eb4f895e82d5be6a | MATCH |

## 3.2 Duplicate Check

Checked W73's backbone md5 (`8d112b78550e87fa91241c15af8eae84`) against all 123 zips in `submissions/` using CRC32 pre-filter then full md5 confirmation.

- **Zips sharing W73 backbone md5:** NONE
- **Zips sharing all 3 payload md5s (backbone + bounds + submission.py):** NONE

W73 is a genuinely new artifact. No previously-scored zip shares its backbone.

## 3.3 Structural Verification

| check | W73 | CORNER3 | LONG80 |
|---|---|---|---|
| unzip -t | clean | clean | clean |
| Entry list == SV2 | True | True | True |
| .pyc count | 0 | 0 | 0 |
| Extracted size (B) | 244,360,369 | 244,360,593 | 244,360,761 |
| Extracted size (% of 268MB cap) | 91.03% | 91.03% | 91.03% |
| Bounds keys | 186 | 186 | 186 |
| Non-backbone files identical to SV2 | True | True | True |

## 3.4 Registry Cross-check

Both rows verbatim from `CHECKPOINT_REGISTRY.md`:

```
ft_md_w10lr3_best.pth   -> EVERY9, USABLE AS BLEND PARTNER? YES
ft_long_w15lr3_best.pth -> EVERY5, member of submission_LONG80.zip, USABLE AS BLEND PARTNER? YES
```

Confirmed: `ft_long_w15lr3_best.pth` = **EVERY5**, `ft_md_w10lr3_best.pth` = **EVERY9**. Both `USABLE AS BLEND PARTNER? = YES`.

## 4. GO/NO-GO SUMMARY

| zip | verdict |
|---|---|
| **submission_W73.zip** | **GO** |
| **submission_CORNER3.zip** | **GO** |
| **submission_LONG80.zip** | **GO** |

All three pass every check. No slot will be wasted on a duplicate or a structurally broken artifact.

---

## RECIPE Audit (Part B) — 10 Sep 2026

| item | value |
|---|---|
| Zip MD5 | 70eba220f017c8ef7d021325bb6045aa |
| Backbone (sim_real_fno_fp16.pth) MD5 | 3458f305593e3f06292bea6f4e1369e2 |

### Duplicate check
- Zips sharing RECIPE backbone md5: **NONE**
- Zips sharing all 3 payload md5s: **NONE**

### Structural checks
| check | result |
|---|---|
| unzip -t | clean |
| Entry list == SV2 | True |
| .pyc count | 0 |
| Extracted size | 244,360,537 B (91.03%) |
| Bounds keys | 186 |
| Non-backbone files identical to SV2 | True |

## submission_RECIPE.zip: **GO**
