# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python [conda env:b]
#     language: python
#     name: conda-env-b-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'retina'

# %%
import matplotlib.pyplot as plt

plt.rc("figure", dpi=300)
plt.rc("figure", figsize=(4, 4))
SMALL_SIZE = 5
MEDIUM_SIZE = 8
BIGGER_SIZE = 10
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "serif",
})
preamble = r"""
\renewcommand{\familydefault}{\sfdefault}
\usepackage[scaled=1]{helvet}
\usepackage[helvet]{sfmath}
\usepackage{textgreek}
"""
plt.rc('text.latex', preamble=preamble)

from spike_psvae import denoise, subtract, spikeio, waveform_utils, snr_templates, relocation
from dartsort.util import waveform_util
from spike_psvae.cluster_viz_index import pgeom
from pathlib import Path
import h5py

import brainbox
import numpy as np
import pandas as pd
import seaborn as sns

# %%
from scipy.spatial.distance import cdist

# %%
import spikeinterface.full as si

# %%
figdir = Path("/Users/charlie/data/spike_sorting_paper_figs/relocation_fig/")

# %%
from scipy.io import loadmat

# %%
ddd1 = si.read_binary("/Users/charlie/data/dataset1-destriped/destriped_p1_g0_t0.imec0.ap.bin", 30000, "float32", 384, is_filtered=True)

# %%
h = loadmat("/Users/charlie/data/dataset1/NP2_kilosortChanMap.mat")

# %%
g = np.c_[h['xcoords'].ravel(), h['ycoords'].ravel()]

# %%
ddd1.set_dummy_probe_from_locations(g)

# %%
ddd1

# %%
ddd1 = ddd1.frame_slice(750 * 30_000, (750 + 600) * 30_000)

# %%
ddd1.save(folder="/Users/charlie/data/dataset1-destriped_snip10_start1000s")

# %%
rec = si.read_binary_folder("/Users/charlie/data/dataset1/destriped_snip_start1000s")

# %%
rec = si.read_binary_folder("/media/cat/data/drift-dataset1-snip/destriped_snip_start1000s")

# %% [markdown]
# ```
# rsync -avP panlab:/media/cat/data/relocation_results/results_07_07/third_deconvolution/merge_labels.npy ~/data/spike_sorting_paper_figs/relocation_fig
# rsync -avP panlab:/media/cat/data/relocation_results/results_07_07/third_deconvolution/deconv_extracted/deconv_results.h5 ~/data/spike_sorting_paper_figs/relocation_fig
# ```

# %%
merge_labels = np.load("/Users/charlie/data/spike_sorting_paper_figs/relocation_fig/merge_labels.npy")

# %%
raw_bin = "/Users/charlie/data/dataset1-destriped_snip_start1000s/traces_cached_seg0.raw"

# %%
1

# %%
temps = np.load("/Users/charlie/Downloads/manual_picked_temp_denoised.npy")

# %%
from dartsort.localize.localize_torch import localize_amplitude_vectors

# %%
import neuropixel

# %%
h = neuropixel.dense_layout()
geom = np.c_[h['x'], h['y']]

# %%
locs_dict = localize_amplitude_vectors(
    templates.ptp(1),
    geom,
    templates.ptp(1).argmax(1),
    radius=100,
)

# %%
plt.scatter(locs_dict['x'], locs_dict['z_abs'], s=5, lw=0, c=np.minimum(templates.ptp(1).max(1), 15))

# %%

# %%

# %%
from spike_psvae.deconvolve import deconv

# %%
rg = np.random.default_rng(0)

# %%
c = rg.normal(size=(1000, 3600000))

# %%
c.shape

# %%
c.dtype

# %%
import time

# %%
del z, c

# %%
tic = time.time()
z = c > 0
toc = time.time()

# %%
toc - tic

# %%

# %%
c = rg.poisson(size=(1000, 3600000))

# %%
c.size

# %%
deconv(raw_bin, "/tmp/hi", temps)

# %%

# %%

# %%
with h5py.File(figdir / "deconv_results.h5") as h5:
    times = h5["spike_index"][:, 0]
    xyza = h5["localizationspeakgpu"][:, :4]
    z_reg = h5["z_reg"][:]
    geom = h5["geom"][:]

# %%
x, y, z, a = xyza.T

# %%
keep = (z > geom[:, 1].min() - 15) & (z < geom[:, 1].max() + 15)
print(keep.sum())
keep &= (x > geom[:, 0].min() - 15) & (x < geom[:, 0].max() + 15)
print(keep.sum())


# %%
merge_labels.shape

# %%
times.shape

# %%
times = times[keep]
xyza = xyza[keep]
x, y, z, a = xyza.T
z_reg = z_reg[keep]

# %%
ids = np.unique(merge_labels)
print(ids)
ids = ids[ids >= 0]

# %%
templates, extra = snr_templates.get_templates(np.c_[times, merge_labels], geom, raw_bin)

# %%
visci = waveform_utils.vertical_dist_channel_index(geom, 61)

# %%
fullci = waveform_utils.full_channel_index(geom.shape[0])

# %%
xyza.min(axis=0)

# %%
which = np.abs(xyza).max(1) < np.abs(geom).max() + 100

# %%
xyza[which].min(axis=0)

# %%
xyza[which].max(axis=0)

# %%
geom[:, 1].min()

# %%
geom[:, 1].max()

# %%
import colorcet as cc

# %%
plt.scatter(times, z, c=merge_labels, cmap=cc.m_glasbey, s=1, lw=0);

# %%
plt.hist(templates.ptp(1).max(1));

# %% tags=[]
nwfs = 150
rg = np.random.default_rng(0)

for label in [78]:
    maxchan = templates[label].ptp(0).argmax()
    
    in_l = np.flatnonzero((merge_labels == label) & which)
    if in_l.size < nwfs:
        continue
    print(label, in_l.size)
    
    choices = rg.choice(in_l, size=min(in_l.size, nwfs))
    maxchans = np.full_like(choices, maxchan)
    
    orig_wfs_full, skipped = spikeio.read_waveforms(times[choices], raw_bin, geom.shape[0])
    if skipped.size:
        continue
    
    orig_wfs = waveform_utils.restrict_wfs_to_chans(
        orig_wfs_full,
        max_channels=maxchans,
        channel_index=fullci,
        dest_channels=visci[maxchan],
    )
    try:
        reloc_wfs = relocation.get_relocated_waveforms_on_channel_subset(
            maxchans,
            orig_wfs_full,
            xyza[choices],
            # np.full_like(xyza[choices, 2], z_reg[choices].mean()),
            # xyza[choices, 2] - (z_reg[choices] - xyza[choices, 2]),
            z_reg[choices],
            fullci, geom, visci[maxchan]
        )
    except Exception as e:
        print(e)
        continue
    maa = 2 * np.abs(templates[label]).max()
    
    
    fig, axes = plt.subplot_mosaic("xx\nab", figsize=(6, 6))
    
    axes["x"].scatter(times[in_l], xyza[in_l, 2], s=1, lw=0, label="unreg")
    axes["x"].scatter(times[in_l], z_reg[in_l], s=1, lw=0, label="reg")
    axes["x"].scatter(times[choices], xyza[choices, 2], marker="x", s=3, label="unreg, shown in wf plot")
    axes["x"].scatter(times[choices], z_reg[choices], marker="x", s=3, label="reg, shown in wf plot")
    axes["x"].legend()
    axes["x"].set_title(f"unit {label}, {in_l.size} spikes, ptp {templates[label].ptp(0).max():0.1f}")
    pgeom(orig_wfs, maxchans, visci, geom, ax=axes["a"], color="k", alpha=0.1, lw=1, max_abs_amp=maa)
    pgeom(reloc_wfs, maxchans, visci, geom, ax=axes["b"], color="k", alpha=0.1, lw=1, max_abs_amp=maa)
    axes["a"].set_title("original raw waveforms")
    axes["b"].set_title("relocated to reg position")
    
    plt.show()
    plt.close(fig)


# %%
