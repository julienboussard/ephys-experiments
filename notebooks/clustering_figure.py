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
clust_fig_dir = Path("/Users/charlie/data/spike_sorting_paper_figs/clustering_fig")
clust_fig_dir.mkdir(exist_ok=True)

# %%
with h5py.File(yy_h5) as h5:
    si = h5["spike_index"][()]
    ci = h5["channel_index"][()]
    geom = h5["geom"][()]

# %%
split_st = np.load(yy_clust / "split_st.npy")[np.load(yy_clust / "split_order.npy")]
merge_st = np.load(yy_clust / "merge_st.npy")[np.load(yy_clust / "merge_order.npy")]

# %%
chans = np.unique(si[:, 1])
splitcounts = np.empty_like(chans)
mergecounts = np.empty_like(chans)
for i, c in enumerate(chans):
    in_c = si[:, 1] == c
    nsplit = np.unique(split_st[in_c, 1]).size
    nmerge = np.unique(merge_st[in_c, 1]).size    
    splitcounts[i] = nsplit
    mergecounts[i] = nmerge

# %%
# chans of interest
chans[(splitcounts > 4) & (mergecounts < splitcounts)]

# %%
chan = 73
in_chan = np.flatnonzero(si[:, 1] == chan)


# %%
def getwfs(ix, maxchans=None, kind="cleaned", n_max=250, origci=ci, visci=None):
    if ix.size > n_max:
        rg = np.random.default_rng(0)
        which = rg.choice(ix.size, size=n_max, replace=False)
        which.sort()
        ix = ix[which]
        if maxchans is not None:
            maxchans = maxchans[which]
    with h5py.File(yy_h5) as h5:
        wfs = h5[f"{kind}_waveforms"][ix]
    if visci is not None:
        wfs = waveform_utils.channel_subset_by_index(wfs, maxchans, origci, visci)
    return wfs, ix, maxchans


# %%
fullci = waveform_utils.make_contiguous_channel_index(384, n_neighbors=384)
fullci.shape

# %%
visci = waveform_utils.make_channel_index(geom, 60)

# %%
visci_big = waveform_utils.make_channel_index(geom, 100)

# %%
# now here is what's gonna happen
# run clustering on this chan without recursive and store the labels, and repeat until it stops

# %%
wfs, ix, mcs = getwfs(in_chan, si[in_chan, 1], visci=visci)
plt.figure(figsize=(1, 2))
pgeom(wfs, mcs, visci, geom, alpha=0.25, color="gray", show_zero=False, lw=0.5);
plt.axis("off")

# %%
triaged_st = si.copy()
triaged_st[:, 1] = -1
triaged_st[in_chan, 1] = 0

# %%
extractor = before_deconv_merge_split.H5Extractor(yy_h5, raw_bin)

# %%
# build splitting tree
cluster_tree = {}
cur_labels = triaged_st[:, 1].copy()
parent_split_results = {}
to_process = [0]
next_label = 1

while to_process:
    parent = to_process.pop(0)
    in_parent = np.flatnonzero(cur_labels == parent)
    is_split, new_labels, in_unit, unit_features = before_deconv_merge_split.herding_split(
        in_parent,
        # hdbscan_kwargs=dict(min_cluster_size=25, min_samples=50, cluster_selection_epsilon=40.),
        hdbscan_kwargs=dict(min_cluster_size=25, min_samples=50, cluster_selection_epsilon=20.),
        # hdbscan_kwargs=dict(min_cluster_size=25, min_samples=25, cluster_selection_epsilon=30.),
        extractor=extractor,
        return_features=True,
    )
    
    if is_split:
        parent_split_results[parent] = dict(
            new_labels=new_labels + next_label * (new_labels >= 0),
            in_parent=in_parent,
            features=unit_features,
        )
        
        new_unique = np.unique(new_labels)
        new_unique = new_unique[new_unique >= 0]
        cluster_tree[parent] = []
        cur_labels[in_parent[new_labels < 0]] = new_labels[new_labels < 0]
        for newu in new_unique:
            cur_labels[in_parent[new_labels == newu]] = next_label
            cluster_tree[parent].append(next_label)
            to_process.append(next_label)
            next_label += 1
final_labels = cur_labels.copy()

# %%
cluster_tree

# %%
{p: np.unique(v['new_labels']) for p, v in parent_split_results.items()}

# %%
np.unique(final_labels, return_counts=True)

# %% tags=[]
# for parent, children in cluster_tree.items():
#     parent_info = parent_split_results[parent]
#     child_labels = parent_info["new_labels"]
#     in_parent = parent_info["in_parent"]
#     features = parent_info["features"]
#     print(f"{parent=} {children=}")
    
#     wfs, ix, mcs = getwfs(in_parent, si[in_parent, 1], visci=visci)
    
#     # plot parent waveforms
#     fig, ax = plt.subplots(figsize=(1, 2))
#     pgeom(wfs, mcs, visci, geom, alpha=0.25, color="gray", show_zero=False, lw=0.5, ax=ax);
#     ax.axis("off")
#     plt.show()
#     plt.close(fig)
    
#     # plot all waveforms
#     fig, ax = plt.subplots(figsize=(1, 2))
#     for u in np.unique(child_labels):
#         color = "gray" if u < 0 else cc.glasbey[u]
#         in_child = np.flatnonzero(child_labels == u)
#         in_child_global = in_parent[in_child]
#         cwfs, cix, cmcs = getwfs(in_child_global, si[in_child_global, 1], visci=visci)
#         pgeom(cwfs, cmcs, visci, geom, alpha=0.25, color=color, show_zero=False, lw=0.5, ax=ax);
#     ax.axis("off")
#     plt.show()
#     plt.close(fig)
    
#     # plot features in color
#     fig, axes = plt.subplots(ncols=3, figsize=(6, 2))
#     colors = np.where(child_labels >= 0, np.array(cc.glasbey)[child_labels], "#999999")
#     order = np.argsort(child_labels)
#     axes[0].scatter(features[order, 0], features[order, 1], c=colors[order], lw=0, s=1)
#     axes[1].scatter(features[order, 0], features[order, 2], c=colors[order], lw=0, s=1)
#     axes[2].scatter(features[order, 3], features[order, 4], c=colors[order], lw=0, s=1)
#     for ax in axes:
#         sns.despine(ax=ax, right=True, top=True)
#         ax.set_xticks([])
#         ax.set_yticks([])
#     axes[0].set_xlabel("$x_n$")
#     axes[0].set_ylabel("$z_n$")
#     axes[1].set_xlabel("scaled log amplitude")
#     axes[1].set_ylabel("$z_n$")
#     axes[2].set_xlabel("PC1")
#     axes[2].set_ylabel("PC2")
#     plt.show()
#     plt.close(fig)

# %%
# 3-layer feature splits vis
cur_parent = 0
n_layers = 3

fig, axes = plt.subplots(nrows=n_layers, ncols=3, figsize=(6, 4))

for ax in axes.flat:
    # sns.despine(ax=ax, right=True, top=True)
    ax.set_xticks([])
    ax.set_yticks([])

for i in range(n_layers):
    row = axes[i]
    
    # unpack clustering result here
    parent_info = parent_split_results[cur_parent]
    child_labels = parent_info["new_labels"]
    in_parent = parent_info["in_parent"]
    features = parent_info["features"]
    
    # plot features
    colors = np.where(child_labels >= 0, np.array(cc.glasbey)[child_labels], "#999999")
    # order scatterplot so that smaller child clusters end up on top
    units, inverse, counts = np.unique(child_labels, return_inverse=True, return_counts=True)
    all_counts = counts[inverse]
    all_counts[child_labels < 0] = child_labels.size + 1
    order = np.argsort(all_counts)[::-1]
    row[0].scatter(features[order, 0], features[order, 1], c=colors[order], lw=0, s=1)
    row[1].scatter(features[order, 2], features[order, 1], c=colors[order], lw=0, s=1)
    row[2].scatter(features[order, 3], features[order, 4], c=colors[order], lw=0, s=1)
    row[0].set_ylabel("$z_n$")
    row[1].set_ylabel("$z_n$")
    row[2].set_ylabel("PC2")
    if i == n_layers - 1:
        row[0].set_xlabel("$x_n$")
        row[1].set_xlabel("scaled log amplitude")
        row[2].set_xlabel("PC1")
    
    # connect the axes
    if i > 0:
        for j, (xaxi, yaxi) in enumerate([(0, 1), (2, 1), (3, 4)]):
            con00 = ConnectionPatch(xyA=[mins[xaxi], mins[yaxi]], coordsA=axes[i - 1, j].transData,
                                   xyB=[0, 1], coordsB=row[j].transAxes, color=cc.glasbey[cur_parent], lw=1, ls="--", capstyle="butt")
            fig.add_artist(con00)
            con01 = ConnectionPatch(xyA=[maxs[xaxi], mins[yaxi]], coordsA=axes[i - 1, j].transData,
                                   xyB=[1, 1], coordsB=row[j].transAxes, color=cc.glasbey[cur_parent], lw=1, ls="--", capstyle="butt")
            fig.add_artist(con01)
        for ax in row:
            for spine in ax.spines.values():
                spine.set_edgecolor(cc.glasbey[cur_parent])
    
    # which one will we do next? the largest child which split
    split_children = [cl for cl in child_labels if cl in parent_split_results]
    split_children = sorted(split_children, key=lambda cl: (child_labels == cl).sum())
    next_parent = split_children[-1]
    
    # get an ellipse in all axes for this guy
    in_next_parent = np.flatnonzero(child_labels == next_parent)
    mins, maxs = features[in_next_parent].min(0), features[in_next_parent].max(0)
    dfs = maxs - mins
    if i < n_layers - 1:
        for j, (xaxi, yaxi) in enumerate([(0, 1), (2, 1), (3, 4)]):
            bbox0 = Rectangle([mins[xaxi], mins[yaxi]], dfs[xaxi], dfs[yaxi], fc=(0, 0, 0, 0), ec=cc.glasbey[next_parent], lw=1, ls="--")
            row[j].add_patch(bbox0)
    # ell0 = get_ellipse(features[in_next_parent, 0], features[in_next_parent, 1], row[0], n_std=5, color=cc.glasbey[next_parent])
    # row[0].add_patch(ell0)

    # set up next time
    cur_parent = next_parent

# %%
# panel c work

# %%
uniq, split_flat_labels = np.unique(final_labels, return_inverse=True)
split_flat_labels[final_labels < 0] = -1
split_flat_labels[final_labels >= 0] = np.unique(final_labels[final_labels >= 0], return_inverse=True)[1]
uniq = uniq[uniq >= 0]

# %%
uniq.size, uniq

# %%
# resid matrix stuff
tpca = waveform_utils.fit_tpca_bin(si, geom, raw_bin)

# %%
templates, extra = snr_templates.get_templates(np.c_[triaged_st[:, 0], final_labels], geom, raw_bin, tpca=tpca)

# %%
templates.shape

# %%
templates_active = templates[uniq]

# %%
templates_visci = waveform_utils.channel_subset_by_index(templates, templates.ptp(1).argmax(1), fullci, visci)

# %%
templates_active.ptp(1) < 1

# %%
ii, kk = np.nonzero(templates_active.ptp(1) < 1)


# %%
templates_active_vis = templates_active.copy()
templates_active_vis[ii[:, None], :, kk[None, :]] = 0

# %%
# merge_resid_threshold = 2.5
# resid_dist_kind = "max"
merge_resid_threshold = 0.25
resid_dist_kind = "rms"

# %%
resids, shifts = deconv_resid_merge.calc_resid_matrix(*(2 * [templates_active_vis, np.arange(len(uniq))]), auto=True, normalized=True, distance_kind=resid_dist_kind)


# %%
resids_viz = resids.copy()
np.fill_diagonal(resids_viz, 0)

# %%
pdist = resids[np.triu_indices(resids.shape[0], k=1)].copy()
# scipy hierarchical clustering only supports finite values, so let's just
# drop in a huge value here
pdist[~np.isfinite(pdist)] = 1_000_000 + pdist[np.isfinite(pdist)].max()
# complete linkage: max dist between all pairs across clusters.
Z = complete(pdist)
# extract flat clustering using our max dist threshold
merge_result_labels = fcluster(Z, merge_resid_threshold, criterion="distance")

# %%
merge_result_labels

# %%
split_flat_labels.shape

# %%
templates.shape, templates_active.shape, 

# %%
spike_train = np.c_[triaged_st[:, 0], split_flat_labels].copy()

# update labels
labels_updated = spike_train[:, 1].copy()
kept = np.flatnonzero(labels_updated >= 0)
labels_updated[kept] = merge_result_labels[labels_updated[kept]]

# update times according to shifts
times_updated = spike_train[:, 0].copy()

# this is done by aligning each unit to the max snr unit in its cluster
maxsnrs = extra["snr_by_channel"].max(axis=1)

# find original labels in each cluster
clust_inverse = {i: [] for i in merge_result_labels}
for orig_label, new_label in enumerate(merge_result_labels):
    clust_inverse[new_label].append(orig_label)

# align to best snr unit
for new_label, orig_labels in clust_inverse.items():
    # we don't need to realign clusters which didn't change
    if len(orig_labels) <= 1:
        continue

    orig_snrs = maxsnrs[orig_labels]
    best_orig = orig_labels[orig_snrs.argmax()]
    for ogl in np.setdiff1d(orig_labels, [best_orig]):
        in_orig_unit = np.flatnonzero(spike_train[:, 1] == ogl)
        # this is like trough[best] - trough[ogl]
        shift_og_best = shifts[ogl, best_orig]
        # if >0, trough of og is behind trough of best.
        # subtracting will move trough of og to the right.
        times_updated[in_orig_unit] -= shift_og_best

# %%
merge_times = times_updated.copy()
merge_labels = labels_updated.copy()

# %%
maxresid = np.abs(resids_viz).max()

# %%
resids_viz.min(), resids_viz.max()

# %%
order

# %%
maxresid

# %%
dist_vmax = 1

# %%
fig = plt.figure(figsize=(3, 2))
gs = fig.add_gridspec(nrows=3, ncols=5, wspace=0.0, hspace=0, height_ratios=[0.5, 1, 0.5], width_ratios=[2, 0.15, 0.7, 0.15, 0.1])
ax_im = fig.add_subplot(gs[:, 0])
ax_cbar = fig.add_subplot(gs[1, 4])
ax_dendro = fig.add_subplot(gs[:, 2], sharey=ax_im)
ax_dendro.axis("off")

dendro = dendrogram(Z, ax=ax_dendro, color_threshold=merge_resid_threshold, distance_sort=True, orientation="right", above_threshold_color="k")
order = np.array(dendro['leaves'])
im = ax_im.imshow(resids_viz[order][:, order], extent=[0, 100, 0, 100], vmin=0, vmax=dist_vmax, cmap=plt.cm.RdGy, origin="lower")
ax_im.set_xticks(10 * np.arange(len(order)) + 5, order)  # uniq[order])
ax_im.set_yticks(10 * np.arange(len(order)) + 5, order)  # uniq[order])
ax_im.set_xlabel("split unit")
ax_im.set_ylabel("split unit")

plt.colorbar(im, cax=ax_cbar, label="template distance")#, orientation="horizontal")
ax_cbar.set_yticks([0, dist_vmax])
ax_cbar.set_ylabel("template distance", labelpad=-5)


# %%
dendro

# %%
merge_result_labels

# %%
merge_result_labels[dendro['leaves']]

# %%
merge_result_colors = dict(zip(merge_result_labels[dendro['leaves']], dendro['leaves_color_list']))
for k, v in merge_result_colors.items():
    if v == "k":
        merge_result_colors[k] = cc.glasbey[k]
merge_result_colors

# %%

# %%

# %%
templates_merge, extra_merge = snr_templates.get_templates(np.c_[merge_times, merge_labels], geom, raw_bin, tpca=tpca)

# %%
templates_merge_visci_big = waveform_utils.channel_subset_by_index(templates_merge, templates_merge.ptp(1).argmax(1), fullci, visci_big)

# %%
templates_merge_visci = waveform_utils.channel_subset_by_index(templates_merge, templates_merge.ptp(1).argmax(1), fullci, visci)

# %%

# %%
ax_im.get_xlim()

# %%
np.unique(merge_labels)

# %%
fig = plt.figure(figsize=(7, 6))
colorset = np.array(cc.glasbey_light)

# -------------------------------------------------- make the panels
top_row, bottom_row = fig.subfigures(nrows=2, wspace=0, hspace=0, height_ratios=[1.5, 1])
panel_a, panel_b, panel_c = top_row.subfigures(ncols=3, wspace=0, hspace=0, width_ratios=[1, 3, 2])
panel_d, panel_e = bottom_row.subfigures(ncols=2, wspace=0, hspace=0, width_ratios=[1, 1])

# panel axes
# a
ax_a = panel_a.subplots()
panel_a.subplots_adjust(top=1, bottom=0)
# b
axes_b = panel_b.subplots(nrows=n_layers, ncols=3)
panel_b.subplots_adjust(top=0.98, bottom=0.15, left=0.075, right=0.98)
# c
axes_c = panel_c.subplots(nrows=4, ncols=3, gridspec_kw=dict(wspace=0.1, hspace=0))
panel_c.subplots_adjust(top=1, bottom=0)
# d
gs = panel_d.add_gridspec(nrows=3, ncols=5, wspace=0.0, hspace=0, height_ratios=[0.5, 1, 0.5], width_ratios=[2, 0.15, 0.425, 0.15, 0.1])
ax_im = panel_d.add_subplot(gs[:, 0])
ax_cbar = panel_d.add_subplot(gs[1, 4])
ax_dendro = panel_d.add_subplot(gs[:, 2], sharey=ax_im)
panel_d.subplots_adjust(bottom=0.12, left=0.1, right=0.95, top=1)
# e
axes_e = panel_e.subplots(nrows=2, ncols=int(np.ceil((np.unique(merge_labels).size - 1) / 2)), gridspec_kw=dict(wspace=0, hspace=0))
panel_e.subplots_adjust(bottom=0, top=0.95, left=0.075, right=0.975)

# gather some details
all_subfigures = [top_row, bottom_row, panel_a, panel_b, panel_c, panel_d, panel_e]
all_panels = [panel_a, panel_b, panel_c, panel_d, panel_e]
any_ax = ax_a

# -------------------------------------------------- A
a_ci = visci_big
# a_ci = None  # for full chan range
a_pci = a_ci if a_ci is not None else ci
wfs, ix, mcs = getwfs(in_chan, si[in_chan, 1], visci=a_ci, n_max=250)
maa = np.abs(wfs).max()
pgeom(wfs, mcs, a_pci, geom, alpha=0.25, color="k", show_zero=False, lw=1, ax=ax_a, rasterized=True);
ax_a.axis("off")

# -------------------------------------------------- B
for ax in axes_b.flat:
    # sns.despine(ax=ax, right=True, top=True)
    ax.set_xticks([])
    ax.set_yticks([])

cur_parent = 0
for i in range(n_layers):
    row = axes_b[i]
    
    # unpack clustering result here
    parent_info = parent_split_results[cur_parent]
    child_labels = parent_info["new_labels"]
    in_parent = parent_info["in_parent"]
    features = parent_info["features"]
    
    # plot features
    colors = np.where(child_labels >= 0, colorset[child_labels], "#999999")
    # order scatterplot so that smaller child clusters end up on top
    units, inverse, counts = np.unique(child_labels, return_inverse=True, return_counts=True)
    all_counts = counts[inverse]
    all_counts[child_labels < 0] = child_labels.size + 1
    order = np.argsort(all_counts)[::-1]
    row[0].scatter(features[order, 0], features[order, 1], c=colors[order], lw=0, s=1, rasterized=True)
    row[1].scatter(features[order, 2], features[order, 1], c=colors[order], lw=0, s=1, rasterized=True)
    row[2].scatter(features[order, 3], features[order, 4], c=colors[order], lw=0, s=1, rasterized=True)
    row[0].set_ylabel("$z_n$")
    row[1].set_ylabel("$z_n$")
    row[2].set_ylabel("PC2")
    if i == n_layers - 1:
        row[0].set_xlabel("$x_n$")
        row[1].set_xlabel("scaled log amplitude")
        row[2].set_xlabel("PC1")
    
    # connect the axes
    if i > 0:
        for j, (xaxi, yaxi) in enumerate([(0, 1), (2, 1), (3, 4)]):
            con00 = ConnectionPatch(
                xyA=[0, 1], coordsA=row[j].transAxes,
                xyB=[mins[xaxi], mins[yaxi]], coordsB=axes_b[i - 1, j].transData,
                color=colorset[cur_parent], lw=1, ls="--", capstyle="butt",
            )
            panel_b.add_artist(con00)
            con01 = ConnectionPatch(
                xyA=[1, 1], coordsA=row[j].transAxes,
                xyB=[maxs[xaxi], mins[yaxi]], coordsB=axes_b[i - 1, j].transData,
                color=colorset[cur_parent], lw=1, ls="--", capstyle="butt",
            )
            panel_b.add_artist(con01)
        for ax in row:
            for spine in ax.spines.values():
                spine.set_edgecolor(colorset[cur_parent])
    
    # which one will we do next? the largest child which split
    split_children = [cl for cl in child_labels if cl in parent_split_results]
    split_children = sorted(split_children, key=lambda cl: (child_labels == cl).sum())
    next_parent = split_children[-1]
    
    # get an ellipse in all axes for this guy
    in_next_parent = np.flatnonzero(child_labels == next_parent)
    mins, maxs = features[in_next_parent].min(0), features[in_next_parent].max(0)
    dfs = maxs - mins
    if i < n_layers - 1:
        for j, (xaxi, yaxi) in enumerate([(0, 1), (2, 1), (3, 4)]):
            bbox0 = Rectangle([mins[xaxi], mins[yaxi]], dfs[xaxi], dfs[yaxi], fc=(0, 0, 0, 0), ec=colorset[next_parent], lw=1, ls="--")
            row[j].add_patch(bbox0)
    # ell0 = get_ellipse(features[in_next_parent, 0], features[in_next_parent, 1], row[0], n_std=5, color=colorset[next_parent])
    # row[0].add_patch(ell0)

    # set up next time
    cur_parent = next_parent

for ax in axes_b[-1]:
    ax.text(0.5, -0.25, "$\\vdots$", verticalalignment="top", fontsize=BIGGER_SIZE, transform=ax.transAxes)

for k, (label, ax) in enumerate(zip(np.setdiff1d(np.unique(final_labels), [-1]), axes_c.flat)):
    ax.text(0.5, 1, f"split unit {k}", transform=ax.transAxes, verticalalignment="top", horizontalalignment="center", fontsize=SMALL_SIZE)
    in_label = np.flatnonzero(final_labels == label)
    wfs, ix, mcs = getwfs(in_label, si[in_label, 1], visci=visci, n_max=50)
    # maa = np.abs(wfs).max()
    plt.figure(figsize=(1, 2))
    pgeom(wfs, mcs, visci, geom, alpha=0.25, color=colorset[label], show_zero=False, lw=0.5, ax=ax, max_abs_amp=maa, rasterized=True);
    pgeom(templates_visci[label][None], [templates[label].ptp(0).argmax()], visci, geom, alpha=1, color="k", max_abs_amp=maa, show_zero=False, lw=1, ax=ax);
for ax in axes_c.flat:
    ax.axis("off")

# -------------------------------------------------- D
ax_dendro.axvline(merge_resid_threshold, color="gray", snap=False)
dendro = dendrogram(Z, ax=ax_dendro, color_threshold=merge_resid_threshold, distance_sort=True, orientation="right", no_labels=True, above_threshold_color="k")
sns.despine(ax=ax_dendro, left=True, top=True, right=True)
for i, (leaf, color) in enumerate(zip(dendro['leaves'], dendro['leaves_color_list'])):
    if color == "k":
        res_label = merge_result_labels[leaf]
        res_color = cc.glasbey[res_label]
        ax_dendro.plot([0, merge_resid_threshold], 2 * [5 + 10 * i], color=res_color, solid_capstyle="butt")
ax_dendro.axes.get_yaxis().set_visible(False)
ax_dendro.set_xlabel("template distance")
order = np.array(dendro['leaves'])
im = ax_im.imshow(resids_viz[order][:, order], extent=[0, 100, 0, 100], vmin=0, vmax=dist_vmax, cmap=plt.cm.bwr, origin="lower")
ax_im.set_xticks(10 * np.arange(len(order)) + 5, order)  # uniq[order])
ax_im.set_yticks(10 * np.arange(len(order)) + 5, order)  # uniq[order])
ax_im.set_xlabel("split unit")
ax_im.set_ylabel("split unit")

plt.colorbar(im, cax=ax_cbar, label="template distance")#, orientation="horizontal")
ax_cbar.set_yticks([0, dist_vmax])
ax_cbar.set_ylabel("template distance", labelpad=-5)

# -------------------------------------------------- E
merge_result_colors = dict(zip(merge_result_labels[dendro['leaves']], dendro['leaves_color_list']))
for k, v in merge_result_colors.items():
    if v == "k":
        merge_result_colors[k] = cc.glasbey[k]
for k, (label, ax) in enumerate(zip(np.setdiff1d(np.unique(merge_labels), [-1]), axes_e.flat)):
    ax.text(0.5, 1, f"merge unit {k}", transform=ax.transAxes, verticalalignment="top", horizontalalignment="center", fontsize=SMALL_SIZE)
    in_label = np.flatnonzero(merge_labels == label)
    wfs, ix, mcs = getwfs(in_label, si[in_label, 1], visci=visci, n_max=50)
    # maa = np.abs(wfs).max()
    plt.figure(figsize=(1, 2))
    pgeom(wfs, mcs, visci, geom, alpha=0.25, color=merge_result_colors[label], show_zero=False, lw=0.5, ax=ax, max_abs_amp=maa, rasterized=True);
    pgeom(templates_merge_visci[label][None], [templates_merge[label].ptp(0).argmax()], visci, geom, alpha=1, color="k", max_abs_amp=maa, show_zero=False, lw=1, ax=ax);
for ax in axes_e.flat:
    ax.axis("off")

# -------------------------------------------------- adjust
for subfig in all_subfigures:
    subfig.set_facecolor([0, 0, 0, 0])
    subfig.patch.set_facecolor([0, 0, 0, 0])
    # show subfigure edges
    # subfig.patch.set_edgecolor([0, 0.5, 0.5, 1])
    # subfig.patch.set_linewidth(1)
for panel, label in zip(all_panels, string.ascii_uppercase):
    any_ax.text(0, 0.995, f"{{\\bf {label}}}", verticalalignment="top", fontsize=BIGGER_SIZE, transform=panel.transSubfigure)
    
fig.savefig(clust_fig_dir / "clusteringfig.pdf")

# %%

# %%
