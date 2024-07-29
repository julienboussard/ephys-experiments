import pandas as pd
df = pd.read_csv("/mnt/home/cwindolf/ceph/rs_freeze_2024_03.csv")
strs = [f"{{ source /mnt/home/cwindolf/.bashrc ; mamba activate sp ; timeout 15000 python /mnt/home/cwindolf/ephys-experiments/scripts/bwm_templates.py {pid} --allsyms_dir /mnt/home/cwindolf/rs_syms --data_dir /mnt/home/cwindolf/ceph/rs_freeze_2024_03/ --n_jobs 10 ; }}" for pid in df.pid]
print(strs[0])
with open("rs_templates.sbatch", "w") as f:
    f.write("#! /bin/env bash\n")
    f.write("\n".join(strs))
    f.write("\n")
