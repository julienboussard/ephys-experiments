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

from spike_psvae import denoise, subtract, spikeio, waveform_utils
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

# %% [markdown]
# get the data
# ```
# rsync -aP popeye:/mnt/sdceph/users/cwindolf/subtraction_fig_data ~/data/spike_sorting_paper_figs/subtraction_fig/ 
# ```

# %%
# load up a geom
base_dir = Path("/Users/charlie/data/spike_sorting_paper_figs/subtraction_fig/")
data_dir = base_dir / "subtraction_fig_data"

# %%
bin_file = data_dir / "zad5" / "traces_cached_seg0.raw"
bin_file.exists()

# %%
h5s = list(data_dir.glob("*.h5"))
h5s

# %%
with h5py.File(h5s[0]) as h5:
    geom = h5["geom"][:]

# %%

# %%
# figure out how to make these brainbox plots

# %%
one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international', silent=True)

# %%
eid, probe = one.pid2eid("8ca1a850-26ef-42be-8b28-c2e2d12f06d6")

# %%
eid, probe

# %%
probeinfo = brainbox.io.one.load_channel_locations(eid, probe=probe, one=one)

# %%
br = BrainRegions()
ba = atlas.AllenAtlas()

# %%
beryl_ids = br.remap(probeinfo[probe]['atlas_id'])
cosmos_ids = br.remap(probeinfo[probe]['atlas_id'], target_map='Cosmos')


# %%
### i want to do a horizontal version of IBL's plot, so I copy-paste the code here
def plot_brain_regions(channel_ids, channel_depths=None, brain_regions=None, display=True, ax=None,
                       title=None, label='left', vertical=True, **kwargs):
    """
    Plot brain regions along probe, if channel depths is provided will plot along depth otherwise along channel idx
    :param channel_ids: atlas ids for each channel
    :param channel_depths: depth along probe for each channel
    :param brain_regions: BrainRegions object
    :param display: whether to output plot
    :param ax: axis to plot on
    :param title: title for plot
    :param kwargs: additional keyword arguments for bar plot
    :return:
    """

    if channel_depths is not None:
        assert channel_ids.shape[0] == channel_depths.shape[0]
    else:
        channel_depths = np.arange(channel_ids.shape[0])

    br = brain_regions or BrainRegions()

    region_info = br.get(channel_ids)
    boundaries = np.where(np.diff(region_info.id) != 0)[0]
    boundaries = np.r_[0, boundaries, region_info.id.shape[0] - 1]

    regions = np.c_[boundaries[0:-1], boundaries[1:]]
    if channel_depths is not None:
        regions = channel_depths[regions]
    region_labels = np.c_[np.mean(regions, axis=1), region_info.acronym[boundaries[1:]]]
    region_colours = region_info.rgb[boundaries[1:]]
    

    if display:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        for reg, col in zip(regions, region_colours):
            height = np.abs(reg[1] - reg[0])
            bar_kwargs = dict(edgecolor='w', width=1)
            bar_kwargs.update(**kwargs)
            color = col / 255
            if vertical:
                ax.bar(x=0.5, height=height, color=color, bottom=reg[0], **kwargs)
            else:
                ax.barh(y=0.5, width=height, color=color, left=reg[0], **kwargs)
        if label == 'right':
            ax.yaxis.tick_right()
        if vertical:
            ax.set_yticks(region_labels[:, 0].astype(int))
            ax.set_ylim(np.nanmin(channel_depths), np.nanmax(channel_depths))
            ax.get_xaxis().set_visible(False)
            ax.set_yticklabels(region_labels[:, 1])
            if label == 'right':
                ax.yaxis.tick_right()
                ax.spines['left'].set_visible(False)
            else:
                ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        else:
            ax.set_xticks(region_labels[:, 0].astype(int))
            ax.set_xlim(np.nanmin(channel_depths), np.nanmax(channel_depths))
            ax.get_yaxis().set_visible(False)
            ax.set_xticklabels(region_labels[:, 1])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
        if title:
            ax.set_title(title)

        return fig, ax
    else:
        return regions, region_labels, region_colours

# %%
ap = probeinfo[probe]['y']

# %%
fig, (aa, ab) = plt.subplots(nrows=2, figsize=(3, 2), gridspec_kw=dict(height_ratios=[2.8, 0.2]))
                             
# -- plot the coronal section                             
ba.plot_cslice(ap.mean(), volume='annotation', mapping='Beryl', ax=aa)

# this is some code i hacked up to make the outside of the brain white instead of black
# and to turn the "root" region gray
transparent_white_gray = np.array([[0, 0, 0, 0], [255, 255, 255, 255], [127, 127, 127, 127]], dtype=np.uint8)
is_void = ba.regions.acronym == "void"
is_root = ba.regions.acronym == "root"
color_code = np.zeros(is_void.shape, dtype=int)
color_code[is_void] = 1
color_code[is_root] = 2
region_values = transparent_white_gray[color_code]
ba.plot_cslice(ap.mean(), volume='value', mapping='Beryl', ax=aa, region_values=region_values)

# i may have x and z flipped
# and they are in meters so we scale to microns
# aa.plot(1e6*probeinfo[probe]['x'], 1e6*probeinfo[probe]['z'], color="k", lw=2)
aa.plot(1e6*probeinfo[probe]['x'][[0, -1]], 1e6*probeinfo[probe]['z'][[0, -1]], color="k", lw=2)

# -- plot the regions
# brainbox.ephys_plots.plot_brain_regions(beryl_ids, channel_depths=geom[:, 1], brain_regions=br, display=True, ax=ab, title=None, label='left')
plot_brain_regions(beryl_ids, channel_depths=geom[:, 1], brain_regions=br, display=True, ax=ab, title=None, vertical=False)

fig.tight_layout()

# %%
# testing out the layout

# %%
fig = plt.figure(figsize=(6.5, 5))
# A: heatmap iterations
# B: histology data + scatter plot variations
# C: specific wf example
# D: metric
panel_a, panel_b, rest = fig.subfigures(ncols=3, wspace=0, hspace=0, width_ratios=[1.75, 3.25, 1.5])
panel_c, panel_d = rest.subfigures(nrows=2, wspace=0, hspace=0)

# panel A has 5 axes and then a ...
axes_a = panel_a.subplots(nrows=5, sharex=True, sharey=True)
panel_a.subplots_adjust(top=0.95, bottom=0.05, left=0.15)

# panel B has histology on the left and then 3x2 scatters
panel_b_1, panel_b_2 = panel_b.subfigures(nrows=2, wspace=0, hspace=0, height_ratios=[1, 6])
axes_b_1 = panel_b_1.subplots(ncols=2, gridspec_kw=dict(width_ratios=[2, 4]))
axes_b_2 = panel_b_2.subplots(nrows=3, ncols=2, sharey=True)

# panel C I still think about
axes_c = panel_c.subplots()

# panel D same
axes_d = panel_d.subplots()

panels = [panel_a, panel_b, panel_c, panel_d]
fcs = "rgby"
labels = "ABCD"
axes_in_panels = [axes_a[0], axes_b_1[0], axes_c, axes_d]
for panel, fc, ax, lab in zip(panels, fcs, axes_in_panels, labels):
    panel.set_facecolor(fc)
    panel.set_edgecolor("gray")
    ax.text(0, 1, f"{{\\bf {lab}}}", verticalalignment="top", fontsize=BIGGER_SIZE, transform=panel.transSubfigure)
    
# let the green through
panel_b_1.set_facecolor([0,0,0,0])
panel_b_2.set_facecolor([0,0,0,0])

# %%
# panel A work


# %%

# %%
# panel B work
use_5 = False
# use_5 = True

if use_5:
    the_h5s = [h5 for h5 in h5s if str(h5).endswith("_5.h5")]
else:
    the_h5s = [h5 for h5 in h5s if "_5" not in str(h5)]
the_h5s = list(sorted(the_h5s, key=lambda x: str(x).count("yes")))
the_h5s

# %%
# ROIs
rois = ["PO", "CA3", "SSP-tr"]
rois = ["LD", "CA3"]
rois = ["PO", "CA3"]

roi2roi = {}
for roi in rois:
    whichid = beryl_ids[probeinfo[probe]['acronym'] == roi][0]
    chans = beryl_ids == whichid
    zs = geom[chans, 1]
    low = zs.min()
    high = zs.max()
    roi2roi[roi] = (low, high)

# %%
h5_names = {0: "no NN, single threshold", 1: "NN, single threshold", 2: "NN, multi-threshold"}
locs_by_roi = {}
for h5 in the_h5s:
    name = h5_names[str(h5).count("yes")]
    
    with h5py.File(h5) as h5:
        x, y, z = h5["localizations"][:, :3].T
        a = h5["maxptps"][:]
    
        by_roi = {}
        for roi, (low, high) in roi2roi.items():
            in_roi = np.flatnonzero((z > low) & (z < high))
            in_roi = in_roi[np.argsort(a[in_roi])]
            by_roi[roi] = (x[in_roi], z[in_roi], a[in_roi])
    locs_by_roi[name] = by_roi

# %%
roi2roi

# %%
# panel A work

# %%
yesyes_h5 = the_h5s[-1]
start_s = 100.01666
len_samples = 500
pad = 121
with h5py.File(yesyes_h5) as h5:
    si = h5["spike_index"][:]
    fs = h5["fs"][()]
    start_sample = int(np.floor(fs * start_s))
    end_sample = start_sample + len_samples
    which = np.flatnonzero((si[:, 0] > start_sample) & (si[:, 0] < end_sample))
    assert np.all(np.diff(which) == 1)
    si = si[which]
    si[:, 0] -= start_sample - pad
    subwfs = h5["subtracted_waveforms"][which[0]:which[-1] + 1]
    ci = h5["channel_index"][()]

# %%
si.shape, subwfs.shape, ci.shape

# %%
raw = spikeio.read_data(bin_file, "float32", start_sample - pad, end_sample + pad, 384)
raw.shape

# %%
v = np.abs(raw[si[:, 0], si[:, 1]])
v

# %%
thresholds = [np.inf, 12, 10, 8, 6, 5] + ([] if use_5 else [4])

# get heatmaps
sub_levels = []
res_levels = []
res = raw.copy()
for tu, tl in zip(thresholds, thresholds[1:]):
    ws = np.flatnonzero((v >= tl) & (v < tu))
    tix = si[ws, 0, None] + np.arange(-42, 79)[None, :]
    cix = ci[si[ws, 1]]
    sub_level = np.zeros_like(raw)
    sub_level = np.pad(sub_level, [(0, 0), (0, 1)], constant_values=np.nan)
    np.add.at(
        sub_level,
        (tix[:, :, None], cix[:, None, :]),
        subwfs[ws],
    )
    sub_level = sub_level[:, :-1]
    sub_levels.append(sub_level)
    res = res - sub_level
    res_levels.append(res)
    
# get cleaned wfs
final_res = res_levels[-1]
final_res = np.pad(final_res, [(0, 0), (0, 1)], constant_values=np.nan)
tix = si[:, 0, None] + np.arange(-42, 79)[None, :]
cix = ci[si[:, 1]]
padded_raw = np.pad(raw, [(0, 0), (0, 1)], constant_values=np.nan)
rawwfs = padded_raw[tix[:, :, None], cix[:, None, :]]
reswfs = final_res[tix[:, :, None], cix[:, None, :]]
cleanedwfs = subwfs + reswfs

# %%

# %%

# %%
vmax_a = np.abs(raw).max()
vmin_a = -vmax_a

# %%

# %%

# %%
# panel c work

# %%
diffs = np.nansum((rawwfs - cleanedwfs), axis=(1, 2))

# %% tags=[]
for ix in np.argsort(diffs)[::-1]:
    print(ix)
    fig, ax = plt.subplots(figsize=(1, 2))
    rawwf = rawwfs[ix]
    maa = np.abs(rawwf).max()
    subwf = subwfs[ix]
    cleanedwf = cleanedwfs[ix]
    for wf, color in zip([rawwf, subwf, cleanedwf], "krb"):
        l = pgeom(
            wf[None],
            si[ix, 1],
            channel_index=ci,
            geom=geom,
            ax=ax,
            color=color,
            show_zero=False,
            lw=0.5,
            max_abs_amp=maa,
        )
    ax.axis("off")
    plt.show()
    plt.close(fig)

# %%
# panel_c_ixs_5 = [9, 28, 34, 38]
# panel_c_ixs_4 = [16, 38, 45, 9]
panel_c_ixs_4 = [24, 9, 6, 26]
panel_c_ixs = panel_c_ixs_5 if use_5 else panel_c_ixs_4

# %%
panel_c_ixs

# %%
# visci = waveform_utils.make_channel_index(geom, 60)
visci = waveform_utils.vertical_dist_channel_index(geom, 21)
visrawwfs = waveform_utils.channel_subset_by_index(rawwfs, si[:, 1], ci, visci)
viscleanedwfs = waveform_utils.channel_subset_by_index(cleanedwfs, si[:, 1], ci, visci)
vissubwfs = waveform_utils.channel_subset_by_index(subwfs, si[:, 1], ci, visci)

# %%

# %%

# %%
# panel D work
the_snrs = {}
the_ptps = {}
the_maxamps = {}
for h5 in the_h5s:
    name = h5_names[str(h5).count("yes")]
    with h5py.File(h5) as h5:
        the_snrs[name] = h5["template_snrs"][:]
        the_ptps[name] = h5["templates"][:].ptp(1).max(1)
        the_maxamps[name] = np.abs(h5["templates"][:]).max((1, 2))        

# %%
records = []
for (name, snrs), ptps, maxamps in zip(the_snrs.items(), the_ptps.values(), the_maxamps.values()):
    for snr, ptp, maxamp in zip(snrs, ptps, maxamps):
        records.append(dict(name=name, snr=snr, logsnr=np.log(snr), maxamp=maxamp, shortname=name.replace("single threshold", "ST").replace("multi-threshold", "MT")))
snrdf = pd.DataFrame.from_records(records)

# %%
kruskal(*[df[1]["snr"].values for df in snrdf.groupby("name")])

# %%
plt.figure(figsize=(2, 2))
ax = sns.histplot(
    snrdf, x="snr", hue="shortname", element="step", log_scale=True, legend=True
)
sns.move_legend(
    ax, "lower center",
    bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
)

# %%
plt.figure(figsize=(2, 2))
ax = sns.kdeplot(snrdf.query("maxamp>=5"), x="snr", hue="shortname", multiple="fill", log_scale=True, legend=True)
sns.move_legend(
    ax, "lower center",
    bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
)

# %%
plt.figure(figsize=(2, 2))
sns.violinplot(snrdf.query("maxamp>=5"), x="name", y="logsnr", hue="name")

# %%
# need to look into the clustering.
# kruskal says no
# are they all horrible?

# %%
# make the actual figure

# %%
fig = plt.figure(figsize=(7, 5))
# A: heatmap iterations
# B: histology data + scatter plot variations
# C: specific wf example
# D: metric
panel_a, rest = fig.subfigures(ncols=2, wspace=0, hspace=0, width_ratios=[2, 3.25 + 1.5])
panel_b, rest = rest.subfigures(nrows=2, wspace=0, hspace=0, height_ratios=[1.5, 1])
panel_c, panel_d = rest.subfigures(ncols=2, wspace=0, hspace=0, width_ratios=[2.5, 1])

# panel A has 5 axes and then a ...
axes_a = panel_a.subplot_mosaic("a.\nb.\ncx\nd.\ne.", gridspec_kw=dict(hspace=0.3, wspace=0.1, width_ratios=[1, 0.05]))
axes_a_heatmap = [axes_a[k] for k in "abcde"]
panel_a.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.875)

# panel B has histology on the left and then 3x2 scatters
panel_b_1, panel_b_2, panel_b_3 = panel_b.subfigures(ncols=3, wspace=0, hspace=0, width_ratios=[18, 1, 1])
# axes_b_1 = panel_b_1.subplot_mosaic("a.\nab\na.", gridspec_kw=dict(width_ratios=[1.5, 4], wspace=0))
# panel_b_1.subplots_adjust(top=0.95, bottom=0)
axes_b_2 = panel_b_1.subplots(nrows=len(rois), ncols=3, sharex=True, sharey="row", gridspec_kw=dict(wspace=0.0, hspace=0.15))
panel_b_1.subplots_adjust(left=0.075, right=0.975)
ax_b_cbar = panel_b_2.subplot_mosaic("...\n.z.\n...", gridspec_kw=dict(hspace=0, wspace=0, width_ratios=[0.1, 2, 1]))["z"]
panel_b_2.subplots_adjust(left=0.075, right=0.7)
panel_b.subplots_adjust(left=0.075, right=0.8)

# panel C I still think about
axes_c = panel_c.subplots(ncols=4, gridspec_kw=dict(hspace=0, wspace=0.15))
panel_c.subplots_adjust(left=0.0, right=0.9, bottom=0.0)

# panel D same
axes_d = panel_d.subplots()
panel_d.subplots_adjust(left=0.1, top=0.85, bottom=0.25)

# -------------------------------------------------- A
croi = slice(0, 128)

vm = 6
imkw = dict(cmap=plt.cm.RdGy, vmin=-vm, vmax=vm, aspect="auto")
im0 = axes_a_heatmap[0].imshow(raw[pad:-pad, croi].T, **imkw)
axes_a_heatmap[1].imshow(sub_levels[0][pad:-pad, croi].T, **imkw)
axes_a_heatmap[2].imshow(res_levels[0][pad:-pad, croi].T, **imkw)
axes_a_heatmap[3].imshow(sub_levels[1][pad:-pad, croi].T, **imkw)
axes_a_heatmap[4].imshow(res_levels[1][pad:-pad, croi].T, **imkw)
plt.colorbar(im0, cax=axes_a["x"], shrink=0.5, label="voltage (s.u.)")
heatmap_titles = ["original", "first threshold waveforms", "first residual", "second threshold waveforms", "second residual"]
for i, (ax, title) in enumerate(zip(axes_a_heatmap, heatmap_titles)):
    ax.set_title(title, fontsize=MEDIUM_SIZE, y=0.98)
    ax.set_yticks([0, croi.stop - 1], [1, croi.stop])
    inline_ylabel(ax, "channel")
    if i == 4:
        ax.set_xticks([0, len_samples - 1], [0, len_samples])
        inline_xlabel(ax, "time (samples)")
    else:
        ax.set_xticks([])
axes_a["x"].set_yticks([-vm, vm])
axes_a["x"].set_ylabel("voltage (s.u.)", labelpad=-10)
axes_a["a"].text(0.5, 0, "$\\vdots$", fontsize=BIGGER_SIZE, transform=panel_a.transSubfigure)

# -------------------------------------------------- B

# -------------------------------------------------- B2
geom_offset = geom.copy()
geom_offset[:, 0] -=  geom[:, 0].mean()
for i, (h5name, by_roi) in enumerate(locs_by_roi.items()):
    col = axes_b_2[:, i]
    col[0].set_title(h5name, fontsize=MEDIUM_SIZE)
    col[-1].set_xticks([geom_offset[:, 0].min(), geom_offset[:, 0].max()])
    inline_xlabel(col[-1], "x (\\textmu{}m)")
    for j, (roi, (x, z, a)) in enumerate(by_roi.items()):
        low, high = roi2roi[roi]
        scatter = col[j].scatter(x - geom[:, 0].mean(), z, lw=0, c=np.minimum(a, 15), vmin=0, vmax=15, alpha=0.5, s=1, rasterized=True)
        col[j].set_xlim([geom_offset[:, 0].min() - 60, geom_offset[:, 0].max() + 60])
        col[j].scatter(*geom_offset.T, marker="s", color="w", edgecolor="k", s=3, lw=0.5)
        col[j].set_ylim([low, high])
        col[j].set_yticks([low, high])
        if i == 0:
            inline_ylabel(col[j], f"depth (\\textmu{{}}m, {roi} detail)")
        # if i == 0:
        #     row[j].set_title(roi)
cbar = plt.colorbar(scatter, cax=ax_b_cbar, label="amplitude (s.u.)")
cbar.solids.set_alpha(1)
ax_b_cbar.set_yticks([0, 15])

        
# -------------------------------------------------- C
for ix, ax in zip(panel_c_ixs, axes_c.flat):
    rawwf = visrawwfs[ix]
    maa = np.abs(np.nan_to_num(rawwf)).max()
    subwf = vissubwfs[ix]
    cleanedwf = viscleanedwfs[ix]
    ls = []
    for wf, color in zip([rawwf, subwf, cleanedwf], "krb"):
        lines = pgeom(
            wf[:, :][None],
            si[ix, 1],
            channel_index=visci,
            geom=geom,
            ax=ax,
            color=color,
            show_zero=False,
            lw=0.5,
            max_abs_amp=maa,
            msbar=ix==panel_c_ixs[-1],
        )
        ls.append(lines[0])
    ax.axis("off")
panel_c.legend(
    ls, ["original", "subtracted", "collision-cleaned"],
    ncols=3,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=[0, 0.8, 0.85, 0.1],
)

# -------------------------------------------------- D

# axes_d.boxplot(boxplots, labels=[k.replace("single threshold", "ST").replace("multi-threshold", "MT") for k in the_snrs.keys()])
# axes_d.set_ylabel("template snr (high amplitude units)")
# sns.kdeplot(snrdf, x="snr", hue="name", ax=axes_d, multiple="fill")
g = sns.histplot(
    snrdf,
    x="snr",
    hue="shortname",
    element="step",
    log_scale=True,
    legend=True,
    ax=axes_d,
)
sns.move_legend(
    g,
    "upper right",
    # "lower center",
    # bbox_to_anchor=(.5, 1),
    # ncol=3,
    markerfirst=False,
    title=None,
    frameon=False,
    fontsize=SMALL_SIZE,
)
axes_d.set_xlabel("template SNR")
axes_d.set_ylabel("number of clusters")


# -------------------------------------------------- adjust
panels = [panel_a, panel_b, panel_c, panel_d]
fcs = "rgby"
labels = "ABCD"
axes_in_panels = [axes_a["a"], axes_b_2.flat[0], axes_c.flat[0], axes_d]
for panel, fc, ax, lab in zip(panels, fcs, axes_in_panels, labels):
    panel.set_facecolor([0, 0, 0, 0])
    panel.patch.set_facecolor([0, 0, 0, 0])
    # panel.set_edgecolor("gray")
    ax.text(0, 0.98, f"{{\\bf {lab}}}", verticalalignment="top", fontsize=BIGGER_SIZE, transform=panel.transSubfigure)
panel_b_3.set_facecolor([0, 0, 0, 0])
panel_b_3.patch.set_facecolor([0, 0, 0, 0])

fig.savefig(base_dir / "subtractionfig.pdf")

# %%

# %%

# %%

# %%

# %%

# %%
# old version

# %%
fig = plt.figure(figsize=(6.5, 5))
# A: heatmap iterations
# B: histology data + scatter plot variations
# C: specific wf example
# D: metric
panel_a, panel_b, rest = fig.subfigures(ncols=3, wspace=0, hspace=0, width_ratios=[1.75, 3.25, 1.5])
panel_c, panel_d = rest.subfigures(nrows=2, wspace=0, hspace=0, height_ratios=[3, 1])

# panel A has 5 axes and then a ...
axes_a = panel_a.subplot_mosaic("a.\nb.\ncx\nd.\ne.", gridspec_kw=dict(hspace=0.3, wspace=0.1, width_ratios=[1, 0.05]))
axes_a_heatmap = [axes_a[k] for k in "abcde"]
panel_a.subplots_adjust(top=0.95, bottom=0.075, left=0.15)

# panel B has histology on the left and then 3x2 scatters
panel_b_1, panel_b_2 = panel_b.subfigures(nrows=2, wspace=0, hspace=0, height_ratios=[1, 6])
axes_b_1 = panel_b_1.subplot_mosaic("a.\nab\na.", gridspec_kw=dict(width_ratios=[1.5, 4], wspace=0))
panel_b_1.subplots_adjust(top=0.95, bottom=0)
axes_b_2 = panel_b_2.subplots(nrows=3, ncols=len(rois), sharex=True, gridspec_kw=dict(wspace=0.3, hspace=0.3))

# panel C I still think about
axes_c = panel_c.subplots(nrows=2, ncols=2, sharex=True, gridspec_kw=dict(hspace=0))
laxes_c = panel_c.subplots()
panel_c.subplots_adjust(bottom=0.05)

# panel D same
axes_d = panel_d.subplots()

# -------------------------------------------------- A
imkw = dict(cmap=plt.cm.RdGy, vmin=-10, vmax=10, aspect="auto")
im0 = axes_a_heatmap[0].imshow(raw[pad:-pad].T, **imkw)
axes_a_heatmap[1].imshow(sub_levels[0][pad:-pad].T, **imkw)
axes_a_heatmap[2].imshow(res_levels[0][pad:-pad].T, **imkw)
axes_a_heatmap[3].imshow(sub_levels[1][pad:-pad].T, **imkw)
axes_a_heatmap[4].imshow(res_levels[1][pad:-pad].T, **imkw)
plt.colorbar(im0, cax=axes_a["x"], shrink=0.5, label="voltage (s.u.)")
heatmap_titles = ["original", "first threshold waveforms", "first residual", "second threshold waveforms", "second residual"]
for i, (ax, title) in enumerate(zip(axes_a_heatmap, heatmap_titles)):
    ax.set_title(title, fontsize=MEDIUM_SIZE, y=0.98)
    ax.set_yticks([0, 383], [1, 384])
    inline_ylabel(ax, "channel")
    if i == 4:
        ax.set_xticks([0, len_samples - 1], [0, len_samples])
        inline_xlabel(ax, "time (samples)")
    else:
        ax.set_xticks([])
axes_a["x"].set_yticks([-10, 10])
axes_a["x"].set_ylabel("voltage (s.u.)", labelpad=-10)
axes_a["a"].text(0.5, 0, "$\\vdots$", fontsize=BIGGER_SIZE, transform=panel_a.transSubfigure)

# -------------------------------------------------- B
# -------------------------------------------------- B1
sec_axis = axes_b_1["a"]
reg_axis = axes_b_1["b"]

# -- plot the coronal section                              
ba.plot_cslice(ap.mean(), volume='annotation', mapping='Beryl', ax=sec_axis)

# this is some code i hacked up to make the outside of the brain white instead of black
# and to turn the "root" region gray
transparent_white_gray = np.array([[0, 0, 0, 0], [255, 255, 255, 255], [127, 127, 127, 127]], dtype=np.uint8)
is_void = ba.regions.acronym == "void"
is_root = ba.regions.acronym == "root"
color_code = np.zeros(is_void.shape, dtype=int)
color_code[is_void] = 1
color_code[is_root] = 2
region_values = transparent_white_gray[color_code]
ba.plot_cslice(ap.mean(), volume='value', mapping='Beryl', ax=sec_axis, region_values=region_values)

# hide coords
sec_axis.axis("off")

# i may have x and z flipped
# and they are in meters so we scale to microns
# aa.plot(1e6*probeinfo[probe]['x'], 1e6*probeinfo[probe]['z'], color="k", lw=2)
sec_axis.plot(1e6*probeinfo[probe]['x'][[0, -1]], 1e6*probeinfo[probe]['z'][[0, -1]], color="k", lw=1.5)

# -- plot the regions
# brainbox.ephys_plots.plot_brain_regions(beryl_ids, channel_depths=geom[:, 1], brain_regions=br, display=True, ax=ab, title=None, label='left')
plot_brain_regions(beryl_ids, channel_depths=geom[:, 1], brain_regions=br, display=True, ax=reg_axis, title=None, vertical=False)
# tick surgery
xticks = reg_axis.get_xticklabels()
xticks = [xt for xt in xticks if xt._text != "root"]
reg_axis.set_xticks([xt._x for xt in xticks], [xt._text for xt in xticks])
# for xt in reg_axis.get_xticklabels():
#     if xt._text == "PO":
#         xt.set_color("g")
#         xt.set_backgroundcolor("orange")
#         xt.get_bbox_patch().set_edgecolor("k")

# -------------------------------------------------- B2
axes_b_2title = panel_b_2.subplots(nrows=3, ncols=1, sharex=True, gridspec_kw=dict(hspace=0.3))


for i, ((h5name, by_roi), tax) in enumerate(zip(locs_by_roi.items(), axes_b_2title)):
    row = axes_b_2[i]
    tax.set_title(h5name, fontsize=MEDIUM_SIZE)
    tax.axis("off")
    for j, (roi, (x, z, a)) in enumerate(by_roi.items()):
        low, high = roi2roi[roi]
        row[j].scatter(x, z, lw=0, c=np.minimum(a, 15), alpha=0.5, s=1)
        row[j].set_xlim([geom[:, 0].min() - 60, geom[:, 0].max() + 60])
        row[j].scatter(*geom.T, marker="s", color="k", s=3, lw=0)
        row[j].set_ylim([low, high])
        row[j].set_yticks([low, high])
        inline_ylabel(row[j], f"depth (\\textmu{{}}m, {roi} detail)")
        # if i == 0:
        #     row[j].set_title(roi)

        
# -------------------------------------------------- D
for ix, ax in zip(panel_c_ixs, axes_c.flat):
    rawwf = rawwfs[ix]
    maa = np.abs(rawwf).max()
    subwf = subwfs[ix]
    cleanedwf = cleanedwfs[ix]
    ls = []
    for wf, color in zip([rawwf, subwf, cleanedwf], "krb"):
        lines = pgeom(
            wf[None],
            si[ix, 1],
            channel_index=ci,
            geom=geom,
            ax=ax,
            color=color,
            show_zero=False,
            lw=0.5,
            max_abs_amp=maa,
        )
        ls.append(lines[0])
    ax.axis("off")
laxes_c.axis("off")
laxes_c.legend(
    ls, ["orig.", "subtracted", "cleaned"], ncols=3, frameon=False,
    loc="upper center",
    bbox_to_anchor=[0, 0.925, 1, 0.1],
)

# -------------------------------------------------- D
axes_d.boxplot(the_snrs.values(), labels=the_snrs.keys())


# -------------------------------------------------- adjust
panels = [panel_a, panel_b, panel_c, panel_d]
fcs = "rgby"
labels = "ABCD"
axes_in_panels = [axes_a["a"], sec_axis, axes_c.flat[0], axes_d]
for panel, fc, ax, lab in zip(panels, fcs, axes_in_panels, labels):
    panel.set_facecolor([0,0,0,0])
    # panel.set_edgecolor("gray")
    ax.text(0, 0.98, f"{{\\bf {lab}}}", verticalalignment="top", fontsize=BIGGER_SIZE, transform=panel.transSubfigure)
panel_b_1.set_facecolor([0,0,0,0])
panel_b_2.set_facecolor([0,0,0,0])


# %%
len(axes_c.flat)

# %%
len(panel_c_ixs)

# %%
