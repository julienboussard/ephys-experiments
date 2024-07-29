# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python [conda env:a]
#     language: python
#     name: conda-env-a-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import pickle
import h5py
from pathlib import Path
from spike_psvae import ibme, ibme_corr
import matplotlib.pyplot as plt

# %%
locdir = Path("/Users/charlie/data/sdsc/re_metas")

regdir = Path("/Users/charlie/data/sdsc/re_regs")

figdir = Path("/Users/charlie/data/sdsc/re_reg_figs")
figdir.mkdir(exist_ok=True)

# %%
count = 0
for subdir in locdir.iterdir():
    print(subdir)
    
    meta_pkl = subdir / "metadata.pkl"
    with open(meta_pkl, "rb") as jar:
        metadata = pickle.load(jar)
    
    if "done" in metadata and metadata["done"]:
        continue
    
    print(metadata["probe"], metadata["pid"])
    
    count += 1

count

# %%
for subdir in regdir.iterdir():
    reg_kind = subdir.name
    
    for reg_npz in subdir.iterdir():
        pid = reg_npz.stem
        print(reg_kind, pid)
        
        with np.load(reg_npz) as npz:
            z = npz["z_reg"]
            t = npz["t"]
            maxptp = npz["maxptp"]
        
        r, *_ = ibme.fast_raster(maxptp, z, t)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(r, aspect="auto", cmap=plt.cm.cubehelix, vmax=15)
        plt.colorbar(im, shrink=0.5, label="amplitude")
        ax.set_title(f"{pid} {reg_kind}")
        
        fig.savefig(figdir / f"{pid}_{reg_kind}.png", dpi=200)

# %%
import torch

# %%
torch.arange(25).reshape(5, 5)[torch.arange(5), torch.arange(5)]

# %%
import numpy as np

# %%
type(np.mean(np.arange(2, dtype=np.int16)))

# %%
