"""Differentiable mvpe, byte-faithful to scoring.mvpe_rel_l2_per_sample.

mvpe is a TIME-AVERAGED relative L2 over a tiny fixed probe set in the near wake:
4 x-stations x 9 y-rows x 2 channels = 72 values out of 32*64*2 = 4096.
It has never appeared in this project's training loss (only in checkpoint
selection, via proj()), which is why every fine-tune to date has cost mvpe
0.05-0.35 locally instead of gaining it.
"""
import torch


def probe_grid(h, w, sub_s_real=2, d=16, center_x=10, center_y=32, n_probe=9):
    """Exactly the index arithmetic in scoring.mvpe_rel_l2_per_sample."""
    probe_center_y = int(center_y / sub_s_real)
    interval_y = min(2, int(h / (n_probe + 1)))
    probe_y = [probe_center_y + interval_y * j
               for j in range(-(n_probe - 1) // 2, n_probe - (n_probe - 1) // 2)]
    probe_y = [y for y in probe_y if 0 <= y < h]
    xs = []
    for i in range(4):
        if int((2 * d + center_x) / sub_s_real) < w:
            px = int(((i + 1) * d + center_x) / sub_s_real)
        else:
            px = int((0.5 * (i + 2) * d + center_x) / sub_s_real)
        if 0 <= px < w:
            xs.append(px)
    return probe_y, xs


def mvpe_per_sample(p, t, sub_s_real=2):
    """p, t: (B, T, H, W, C>=2) torch tensors. Returns (B,) differentiable."""
    B = p.shape[0]
    h, w = p.shape[2], p.shape[3]
    probe_y, xs = probe_grid(h, w, sub_s_real)
    if not probe_y or not xs:
        return torch.zeros(B, device=p.device, dtype=p.dtype)
    yi = torch.as_tensor(probe_y, device=p.device, dtype=torch.long)
    errs = []
    for px in xs:
        # index y then x separately -- avoids relying on mixed list/int
        # advanced-indexing semantics matching numpy's.
        pp = p.index_select(2, yi)[:, :, :, px, :2].mean(dim=1).reshape(B, -1)
        tt = t.index_select(2, yi)[:, :, :, px, :2].mean(dim=1).reshape(B, -1)
        den = torch.linalg.norm(tt, dim=1).clamp(min=1e-8)
        errs.append(torch.linalg.norm(pp - tt, dim=1) / den)
    return torch.stack(errs, 0).mean(0)


def mvpe_loss(p, t, sub_s_real=2):
    return mvpe_per_sample(p, t, sub_s_real).mean()
