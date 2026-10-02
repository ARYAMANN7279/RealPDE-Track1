"""V4 gate: the fast loader must produce EXACTLY the banked weights.
Loads the FNO and all U-Nets through the banked code path (var_base) and through the V4 path
(var_v4) in one process and compares every state_dict entry: dtype, shape, device and
bit-equality (complex compared via view_as_real)."""
import os, sys, importlib.util, time
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location("sub_" + name,
                                                  os.path.join(D, "var_" + name, "submission.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


import torch  # noqa: E402

out = {}
for name in ("base", "v4"):
    m = load(name)
    t = time.perf_counter()
    model, dev = m._get_model()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    net, _, _ = m._get_net(dev)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    sds = {"fno": model.state_dict(), "net": net.state_dict()}
    for j, e in enumerate(m._state["extra"]):
        sds[f"extra{j}"] = e.state_dict()
    out[name] = sds
    print(f"{name}: _get_model {t1 - t:.3f}s _get_net {t2 - t1:.3f}s "
          f"(same process; the second loader benefits from a warm CUDA context)", flush=True)

bad, n = 0, 0
for part in out["base"]:
    a, b = out["base"][part], out["v4"][part]
    if set(a) != set(b):
        print("KEY MISMATCH", part, set(a) ^ set(b))
        bad += 1
        continue
    for k in a:
        n += 1
        x, y = a[k], b[k]
        same = (x.dtype == y.dtype and x.shape == y.shape and x.device == y.device)
        if same:
            if x.is_complex():
                same = torch.equal(torch.view_as_real(x), torch.view_as_real(y))
            else:
                same = torch.equal(x, y)
        if not same:
            bad += 1
            print("DIFF", part, k, x.dtype, y.dtype, tuple(x.shape), tuple(y.shape), x.device, y.device)
print(f"compared {n} tensors, mismatches {bad}")
print("V4 WEIGHTS IDENTICAL" if bad == 0 else "V4 WEIGHTS DIFFER")
print("[done]")
