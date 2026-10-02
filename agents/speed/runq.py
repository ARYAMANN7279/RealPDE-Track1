"""Sequential job queue for ONE GPU (inherits CUDA_VISIBLE_DEVICES from the launcher).

  setsid nohup env CUDA_VISIBLE_DEVICES=3 $P runq.py jobs.json < /dev/null > logs/runq_X.log 2>&1 & disown

jobs.json: [["logname", {"ENV": "val"} (optional), "script.py", "arg", ...], ...]
Each job writes logs/<logname>. Before every job the GPU-3 thermal state is checked; the queue
waits (does not launch) while temperature >= 90 C or SW thermal slowdown is Active.
Jobs never overlap, so timings are never contaminated by our own work.
"""
import json, subprocess, sys, os, time

D = os.path.dirname(os.path.abspath(__file__))
PHYS_GPU = "3"


def thermal():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,temperature.gpu,clocks.sm,"
                          "clocks_throttle_reasons.sw_thermal_slowdown", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    for ln in out.splitlines():
        p = [s.strip() for s in ln.split(",")]
        if p and p[0] == PHYS_GPU:
            return (int(p[1]) < 90 and p[3] == "Not Active"), ln
    return False, out.strip()


def foreign_procs():
    uu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                        capture_output=True, text=True).stdout
    uuid = [ln.split(",")[1].strip() for ln in uu.splitlines() if ln.split(",")[0].strip() == PHYS_GPU]
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout
    return [ln for ln in apps.splitlines() if uuid and uuid[0] in ln and str(os.getpid()) not in ln]


jobs = json.load(open(sys.argv[1]))
os.makedirs(os.path.join(D, "logs"), exist_ok=True)
for job in jobs:
    log, rest = job[0], job[1:]
    env = dict(os.environ)
    if rest and isinstance(rest[0], dict):
        env.update({k: str(v) for k, v in rest[0].items()})
        rest = rest[1:]
    while True:
        ok, ln = thermal()
        fp = foreign_procs()
        if ok and not fp:
            break
        print(f"[runq] waiting: thermal_ok={ok} ({ln}) other_procs={fp}", flush=True)
        time.sleep(30)
    t = time.time()
    print(f"[runq] start {log}: {' '.join(rest)} | {ln}", flush=True)
    with open(os.path.join(D, "logs", log), "w") as f:
        rc = subprocess.run([sys.executable] + rest, stdout=f, stderr=subprocess.STDOUT, cwd=D,
                            env=env, stdin=subprocess.DEVNULL).returncode
    print(f"[runq] end {log} rc={rc} {time.time() - t:.1f}s", flush=True)
print("[done]", flush=True)
