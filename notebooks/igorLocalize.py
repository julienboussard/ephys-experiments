# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python [conda env:mysi]
#     language: python
#     name: conda-env-mysi-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'retina'

# %%
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.io import loadmat
from spike_psvae import subtract
import spikeinterface.core as sc
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
from pathlib import Path
import chardet
import shutil
import subprocess

# %%
dsroot = Path("/Volumes/emt_ssd_2/dredge_data/pacman-task/igor/2022-12-07/pacman-task_i_221207_neu_g0/")
outdir = Path("/Users/charlie/data/igor")

# %%
for subdir in dsroot.iterdir():
    print(subdir.stem)
    iout = outdir / subdir.stem
    iout.mkdir(exist_ok=True, parents=True)

# %%
h = loadmat(outdir / "neuropixNHPv1_kilosortChanMap_v1.mat")
geom = np.c_[h["xcoords"], h["ycoords"]].astype("float")
geom.shape, geom.dtype

# %%
print(h.keys())

# %%
eval("1")

# %%
shutil.

# %%
for subdir in dsroot.iterdir():
    if subdir.stem.endswith("imec0"):
        continue
    
    ap_bin = next(subdir.glob("*.ap.bin"))
    print(ap_bin)
    print(ap_bin.name)
    ap_meta = ap_bin.with_suffix(".meta")
    meta_bkp = ap_meta.with_suffix(".meta.bkp")
    if ap_meta.exists():
        shutil.move(ap_meta, meta_bkp)
    
    with open(meta_bkp, "r") as meta:
        for line in meta:
            if line.startswith("imSampRate"):
                fs = eval(line.split("=")[1])
    print(f"{fs=}")
    
    rec = sc.read_binary(ap_bin, fs, 385, np.int16, gain_to_uV=2.3438, is_filtered=False)
    print(rec)
    
    plt.imshow(rec.get_traces(start_frame=1000, end_frame=1500).T, aspect="auto")
    plt.show()
    plt.close(plt.gcf())
    
    plt.plot(rec.get_traces(start_frame=1000, end_frame=1500))
    plt.plot(rec.get_traces(channel_ids=[384], start_frame=1000, end_frame=1500), color="k")
    plt.show()
    plt.close(plt.gcf())

# %%
import spikeglx

# %%
spikeglx.__file__

# %%
for subdir in dsroot.iterdir():
    ap_bin = next(subdir.glob("*.ap.bin"))
    ap_meta = ap_bin.with_suffix(".meta")
    meta_bkp = ap_meta.with_suffix(".meta.bkp")
    if ap_meta.exists() and not meta_bkp.exists():
        shutil.move(ap_meta, meta_bkp)
                
    print(f"{fs=}")
    print(f"{meta_bkp=}")
    print(f"{ap_meta=}")
    
    with open(meta_bkp, "r") as meta:
        with open(ap_meta, "w") as meta_out:
            saw_port = False
            saw_slot = False
    
            for line in meta:
                if line.startswith("imSampRate"):
                    fs = eval(line.split("=")[1])
                if line.startswith("imDatPrb_type"):
                    # mock 3B2
                    line = "imDatPrb_type=0\n"
                if line.startswith("imDatPrb_port"):
                    saw_port = True
                if line.startswith("imDatPrb_slot"):
                    saw_slot = True
                meta_out.write(line)
    
            if not saw_port:
                meta_out.write("imDatPrb_port=mock\n")
            if not saw_slot:
                meta_out.write("imDatPrb_slot=mock\n")
            
    print(f"{fs=}")
    
    rec_orig = sc.read_binary(ap_bin, fs, 385, np.int16, gain_to_uV=2.3438, is_filtered=False)
    print(rec_orig)
    
    destriped_bin = subdir / f"destriped_{ap_bin.name}"
    if not destriped_bin.exists():
        subprocess.run(
            [
                "python",
                Path("/Users/charlie/spike-psvae") / "scripts" / "destripe.py",
                str(ap_bin),
            ],
            check=True,
        )
        
    rec = sc.read_binary(destriped_bin, rec_orig.sampling_frequency, 384, dtype="float32")
    rec.set_dummy_probe_from_locations(geom)
    print(rec)

    sub_h5 = subtract.subtraction(
        rec,
        out_folder=subdir,
        thresholds=[12, 10, 8, 6, 5],
        n_sec_pca=40,
        save_subtracted_tpca_projs=False,
        save_cleaned_tpca_projs=False,
        save_denoised_tpca_projs=False,
        n_jobs=8,
        loc_workers=1,
        overwrite=False,
        n_sec_chunk=1,
        save_cleaned_pca_projs_on_n_channels=5,
        loc_feature=("ptp", "peak"),
    )
    
    destriped_cbin.unlink()

# %%
