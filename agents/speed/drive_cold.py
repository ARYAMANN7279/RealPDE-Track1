"""Runs prof_cold.py in FRESH processes, one per emulated ingestion part, strictly sequentially
on the GPU this driver was launched on (CUDA_VISIBLE_DEVICES is inherited).

  $P drive_cold.py --subs screen=<dir>[,cand=<dir>] --ns 48,240,1028,5140 --modes plain,hooks \
      --reps 3 --out logs/cold.json
Order is interleaved (rep-major, then n, then sub, then mode) so drift hits every arm alike."""
import subprocess, sys, json, os, argparse, time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--subs", required=True, help="name=dir,name=dir")
ap.add_argument("--ns", default="48,240,1028,5140")
ap.add_argument("--modes", default="plain,hooks")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--evict", action="store_true")
ap.add_argument("--out", required=True)
args = ap.parse_args()

subs = [s.split("=", 1) for s in args.subs.split(",")]
ns = [int(v) for v in args.ns.split(",")]
modes = args.modes.split(",")
rows = []
for rep in range(args.reps):
    for n in ns:
        for name, d in subs:
            for mode in modes:
                cmd = [sys.executable, os.path.join(HERE, "prof_cold.py"), "--sub", d, "--n", str(n),
                       "--warm", str(args.warm), "--tag", f"{name}|{mode}|{n}|{rep}"]
                if mode == "hooks":
                    cmd.append("--hooks")
                if mode == "ctx":
                    cmd.append("--ctx_first")
                if args.evict:
                    cmd.append("--evict")
                t = time.perf_counter()
                p = subprocess.run(cmd, capture_output=True, text=True)
                wall = time.perf_counter() - t
                js = [ln[5:] for ln in p.stdout.splitlines() if ln.startswith("JSON ")]
                if p.returncode != 0 or not js:
                    print(f"FAIL {name} {mode} n={n} rep={rep} rc={p.returncode}\n{p.stderr[-3000:]}",
                          flush=True)
                    continue
                r = json.loads(js[-1])
                r.update(name=name, mode=mode, rep=rep, subprocess_wall_s=wall)
                rows.append(r)
                wm = min(r["warm_s"]) if r["warm_s"] else float("nan")
                print(f"{name:10s} {mode:6s} n={n:5d} rep={rep} import {r['import_s']:.3f}s "
                      f"cold {r['cold_s']:.3f}s ({r['cold_s'] / n * 1e3:.3f} ms/s) "
                      f"warm {wm:.3f}s ({wm / n * 1e3:.3f} ms/s) "
                      f"overhead {r['cold_s'] - wm:.3f}s ok={r['out_ok']} "
                      f"loaders {json.dumps({k: round(v, 3) for k, v in r['loaders'].items()})}",
                      flush=True)
                with open(args.out, "w") as f:
                    json.dump(rows, f, indent=1)
print("[done]", flush=True)
