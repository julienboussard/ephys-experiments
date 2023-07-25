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

from spike_psvae import denoise, subtract, spikeio, waveform_utils, newms, before_deconv_merge_split, deconv_resid_merge, snr_templates
from spike_psvae.cluster_viz_index import pgeom
from pathlib import Path
import h5py

import brainbox
from one.api import ONE
import brainbox.io.one
from ibllib import atlas
from ibllib.atlas.regions import BrainRegions
import numpy as np
import pandas as pd
import seaborn as sns

# %%
from matplotlib.patches import Ellipse, Rectangle, ConnectionPatch

# %%
from matplotlib import transforms

# %%
import string

# %%
import colorcet as cc

# %%
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import complete, fcluster, dendrogram
scipy.cluster.hierarchy.set_link_color_palette(cc.glasbey)


# %%
def get_ellipse(x, y, ax, n_std=2, color="k"):
    x_mean, y_mean = np.median(x), np.median(y)
    cov = np.zeros((2, 2))
    if x.size > 2:
        cov = np.cov(x, y)

    vx, vy = cov[0, 0], cov[1, 1]
    if min(vx, vy) <= 0:
        assert False
    rho = np.minimum(1.0, cov[0, 1] / np.sqrt(vx * vy))
    
    ell = Ellipse(
        (0, 0),
        width=2 * np.sqrt(1 + rho),
        height=2 * np.sqrt(1 - rho),
        facecolor=(0, 0, 0, 0),
        edgecolor=color,
        linewidth=1,
    )
    transform = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(n_std * np.sqrt(vx), n_std * np.sqrt(vy))
        .translate(x_mean, y_mean)
    )
    ell.set_transform(transform + ax.transData)
    
    return ell


# %%
from matplotlib.transforms import offset_copy

def inline_xlabel(ax, label):
    t = offset_copy(
        ax.transAxes,
        y=-(ax.xaxis.get_tick_padding() + ax.xaxis.get_tick_space()), 
        fig=fig,
        units='points'
    )
    ax.xaxis.set_label_coords(.5, 0, transform=t)
    ax.set_xlabel(label, va='center')
    ax.xaxis.get_label().set_bbox(dict(facecolor='white', alpha=0.0, linewidth=0))

def inline_ylabel(ax, label):
    t = offset_copy(
        ax.transAxes,
        x=-(ax.yaxis.get_tick_padding() + ax.yaxis.get_tick_space()), 
        fig=fig,
        units='points'
    )
    ax.yaxis.set_label_coords(0, .5, transform=t)
    ax.set_ylabel(label, va='center')
    ax.yaxis.get_label().set_bbox(dict(facecolor='white', alpha=0.0, linewidth=0))


# %%
sub_data_dir = Path("/Users/charlie/data/spike_sorting_paper_figs/subtraction_fig/subtraction_fig_data")
yy_h5 = next(f for f in sub_data_dir.glob("*.h5") if f.name.count("yes") == 2)
yy_clust = next(sub_data_dir.glob("yes*yes*clust"))

# %%
raw_bin = sub_data_dir / "zad5" / "traces_cached_seg0.raw"

# %%
# %ll {yy_clust}

# %%
dnt_fig_dir = Path("/Users/charlie/data/spike_sorting_paper_figs/denoised_templates_fig")
dnt_fig_dir.mkdir(exist_ok=True)

# %%
with h5py.File(yy_h5) as h5:
    si = h5["spike_index"][()]
    st = h5["aligned_spike_train"][()]
    ci = h5["channel_index"][()]
    geom = h5["geom"][()]

# %%
temps, extra = snr_templates.get_templates(st, geom, raw_bin)

# %%
extra.keys()

# %%
tempmcs = temps.ptp(1).argmax(1)

# %%
raw_temps = extra["raw_templates"]
pca_temps = extra["denoised_templates"]
weights = extra["weights"]

# %%
weights.shape

# %%
fullci = waveform_utils.make_contiguous_channel_index(384, n_neighbors=384)
visci = waveform_utils.make_channel_index(geom, 75)

# %%
temps_vis = waveform_utils.channel_subset_by_index(temps, tempmcs, fullci, visci)
raw_temps_vis = waveform_utils.channel_subset_by_index(raw_temps, tempmcs, fullci, visci)
pca_temps_vis = waveform_utils.channel_subset_by_index(pca_temps, tempmcs, fullci, visci)
weights_vis = waveform_utils.channel_subset_by_index(weights, tempmcs, fullci, visci)

# %%

# %%

# %%
weights.min()

# %%
weights.max()

# %%
nunits = 5
seed = 5
choices = np.random.default_rng(seed).choice(len(temps), size=nunits, replace=False)
maa = np.abs(temps[choices]).max() / 1.1
sls = temps.shape[1]
trim = 0

zero_kw = dict(color="gray", lw=0.5, linestyle="--")

fig, axes = plt.subplots(ncols=nunits, figsize=(7, 3.5))
fig.subplots_adjust(top=1, bottom=0.1)

for ax, choice in zip(axes.flat, choices):
    mc = tempmcs[choice]
    temp = temps_vis[choice]
    raw_temp = raw_temps_vis[choice]
    pca_temp = pca_temps_vis[choice]
    weight = weights_vis[choice]
    
    domsbar = choice == choices[-1]
    dobar = choice == choices[-1]
    
    lr = pgeom(raw_temp[None, trim:sls-trim], [mc], visci, geom, ax=ax, color="k", lw=1.5, max_abs_amp=maa, show_zero=False, show_zero_kwargs=zero_kw, zlim=0.5, subar=5 * dobar, msbar=domsbar)
    lp = pgeom(pca_temp[None, trim:sls-trim], [mc], visci, geom, ax=ax, color="blue", lw=1, max_abs_amp=maa, show_zero=False, show_zero_kwargs=zero_kw, zlim=0.5)
    lt = pgeom(temp[None, trim:sls-trim], [mc], visci, geom, ax=ax, color="orange", lw=1, max_abs_amp=maa, show_zero=False, show_zero_kwargs=zero_kw, zlim=0.5)
    lw = pgeom(weight[None, trim:sls-trim], [mc], visci, geom, ax=ax, color="gray", lw=1, max_abs_amp=maa, show_zero=False, show_zero_kwargs=zero_kw, zlim=0.5)
    
    ax.axis("off")

fig.legend(
    handles=[lr[0], lp[0], lt[0], lw[0]],
    labels=["raw template", "PCA template", "denoised template", "raw template mixing weight"],
    ncol=4,
    loc="upper center",
    frameon=False,
)

fig.patch.set_edgecolor([1, 1, 0, 1])
fig.patch.set_linewidth(1)

# %%
