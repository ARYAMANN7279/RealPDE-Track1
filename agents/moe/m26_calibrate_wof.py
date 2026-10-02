import numpy as np
def n_(x): return x / (0.5 + x)
def Wof(dm, tke, mv):
    # la_score is usually 100*(1-err)
    # inv(score) = err
    return 0.5*(1-n_(dm)) + 0.3*(1-n_(tke)) + 0.2*(1-n_(mv))

# Test anchors from compare_subs.py:
# SOUP_v1: (94.05, 76.00, 92.87) -> SPS 34.37
# Let's see what our Wof predicts for these.
dm = 1 - 94.05/100
tke = 1 - 76.00/100
mv = 1 - 92.87/100
wv = Wof(dm, tke, mv)
print(f"SOUP_v1 Wof: {wv:.4f}")
# If real SPS = 100 * Wv * E_local, then E_local = 34.37 / (100 * wv)
e_local = 34.37 / (100 * wv)
print(f"Implied E_local for SOUP_v1: {e_local:.4f}")
