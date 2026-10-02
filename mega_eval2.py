import os, time, sys, subprocess, json

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
TES = f"{B}/train_es"
os.chdir(TES)

print("Waiting for U-Nets to finish...", flush=True)
while True:
    try:
        out = subprocess.check_output(["cat", "launch_unets.log"]).decode("utf-8")
        if "All U-Nets finished!" in out:
            break
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(10)

print("Packing U-Nets...", flush=True)
subprocess.run(["python3", f"{TES}/pack_unets.py"], check=True)
subprocess.run(["CUDA_VISIBLE_DEVICES=0", "python3", f"{TES}/run_mega_eval.py"], env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"), check=True)
print("MEGA EVAL DONE", flush=True)
