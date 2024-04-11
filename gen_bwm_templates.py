import pandas as pd
df = pd.read_csv("~/ceph/2023_12_bwm_release.csv")
strs = [f"{{ source /mnt/home/cwindolf/.bashrc ; mamba activate sp ; timeout 15000 python /mnt/home/cwindolf/ephys-experiments/scripts/bwm_templates.py {pid} --n_jobs 10 --retry_err  ; }}" for pid in df.pid]
print(strs[0])
with open("bwm_templates.sbatch", "w") as f:
    f.write("#! /bin/env bash\n")
    f.write("\n".join(strs))
    f.write("\n")
