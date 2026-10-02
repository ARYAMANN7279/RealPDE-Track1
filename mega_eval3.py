import os, time, sys, subprocess, json

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
TES = f"{B}/train_es"
os.chdir(TES)

print("Waiting for U-Nets to finish...", flush=True)
while True:
    all_done = True
    for i in range(3):
        for j in range(4):
            if not os.path.exists(f"{TES}/joint_unet_m{i}_{j}.pth"):
                all_done = False
                break
    if all_done:
        break
    time.sleep(10)

print("Packing U-Nets...", flush=True)
subprocess.run(["python3", f"{TES}/pack_unets.py"], check=True)
subprocess.run(["CUDA_VISIBLE_DEVICES=0", "python3", f"{TES}/run_mega_eval.py"], env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"), check=True)
print("MEGA EVAL DONE", flush=True)
