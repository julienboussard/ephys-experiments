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
1

# %%
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import numpy as np
import h5py
import spikeinterface.full as si
from spike_psvae import snr_templates, waveform_utils, spike_train_utils
from spike_psvae.cluster_viz_index import pgeom

# %%
import pandas as pd

# %%
from pathlib import Path

# %%
import seaborn as sns

# %%
from tqdm.auto import tqdm, trange

# %% tags=[]
import matplotlib.pyplot as plt

# %%
from spikeinterface.preprocessing.hybrid_recording import hybrid_recording

# %%
import string

# %%
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
# ## denoised templates
#
# are they better estimates of the mean?

# %% [markdown]
# ### mearec version

# %%
# !rm -rf /tmp/mearectmp

# %% tags=[]
# mearec_h5 = "/moto/stats/users/ch3676/recordings_static/Neuropixels-64_static_uniform_homogeneous_static_noise_5.h5"
mearec_h5 = "/Users/charlie/data/spike_sorting_paper_figs/denoised_templates_fig/Neuropixels-64_static_uniform_homogeneous_static_noise_5.h5"

with h5py.File(mearec_h5) as h5:
    print(h5.keys())
    # orig_temps = h5["templates"][:]
    # print(h5["recordings"].keys())

rec, sorting = si.read_mearec(mearec_h5)

rec = si.bandpass_filter(rec)
rec = si.common_reference(rec)
rec = si.zscore(rec, num_chunks_per_segment=200)

# !rm -rf /tmp/mearectmp
rec_saved = rec.save_to_folder(folder="/tmp/mearectmp")
geom = rec_saved.get_channel_locations()

rec_saved

# %% tags=[]
st = sorting.get_all_spike_trains()

times, orig_labels = st[0]

orig_units, labels, counts = np.unique(orig_labels, return_inverse=True, return_counts=True)

sls = 181
tgh = 72

st = np.c_[times, labels]
ast, order, temps0, shifts = spike_train_utils.clean_align_and_get_templates(st, 64, "/tmp/mearectmp/traces_cached_seg0.raw", max_shift=50, spike_length_samples=sls, trough_offset=tgh)

times = ast[:, 0].copy()
labels = ast[:, 1].copy()

# %% [markdown] tags=[]
# ### hybrid version

# %% [markdown]
# y axis variables
#
#  - cleaned template err vs gt template
#  - raw template err vs gt template
#
# x axis variables
#
#  - GT template maxptp
#  - number of spikes used to compute template
#  
# 5 repeats
#
# within each repeat:
# 2 dataframes:
#  - for n spikes varying: max_n_spikes [50, 100, 150, 200, 250]
#  - for ptp varying: max_n_spikes 250

# %%

# %%
T

# %%
# check:
#  - rank=5 vs 10
#  - threshold = 10 vs 50
#  - inf vs min(...) vs just snr with no nothin

# %%

# %%
times

# %%
times.shape, labels.shape

# %%
st = np.c_[times, labels]

# %%
st.shape

# %%
ns_records = []
chan_records = []

nss = (5, 10, 25, 50, 100, 250)
# nss = (300,)
# nss = (50, 150, 250)
sls = 121
tgh = 42
rg = np.random.default_rng(0)
templates0 = np.load("/Users/charlie/data/spike_sorting_paper_figs/denoised_templates_fig/templates_for_simulation.npy")
# templates0 = templates0[templates0.ptp(1).max(1) > 5]
edge_behavior = "saturate"

for fold in trange(3):
    rec = si.read_binary_folder("/Users/charlie/data/spike_sorting_paper_figs/subtraction_fig/subtraction_fig_data/zad5/")
    T = rec.get_num_samples()

    n_units = 100
    rr = int(1e-3 * rec.sampling_frequency)

    frs = rg.uniform(0.5, 4, size=n_units) * T / 30000
    times = []
    labels = []
    for u, fr in enumerate(frs):
        ns = rg.poisson(fr)
        while True:
            st = rg.integers(T, size=ns)
            st.sort()
            if np.diff(st).min() >= rr:
                break
        times.extend(st)
        labels.extend([u] * ns)
    times = np.array(times)
    order = np.argsort(times)
    times = times[order]
    labels = np.array(labels)[order]
    templates = templates0[rg.choice(len(templates0), size=n_units, replace=False)]
    gtptps = templates.ptp(1).max(1)

    rec = hybrid_recording(rec, times, labels, templates)
    # !rm -rf /tmp/mearectmp
    rec_saved = rec.save_to_folder(folder="/tmp/mearectmp", n_jobs=5)
    geom = rec_saved.get_channel_locations()
    
    # which = np.zeros(len(times), dtype=bool)
    # which[rg.choice(len(times), size=len(times)//5, replace=False)] = 1

    tin = times#[which]
    lin = labels#[which]
    # tout = times[~which]
    # lout = labels[~which]

    tpca_rank = 5

    for ns in tqdm(nss, leave=False):
        temps_in, extra_in = snr_templates.get_templates(
            np.c_[tin, lin],
            geom,
            "/tmp/mearectmp/traces_cached_seg0.raw",
            tpca_rank=5,
            spike_length_samples=sls,
            trough_offset=tgh,
            max_spikes_per_unit=ns,
            n_jobs=5,
            do_temporal_decrease=False,
            snr_threshold=50.0,
            zero_radius_um=10000,
            tpca_centered=False,
            edge_behavior=edge_behavior,
        )
        raw_temps_in = extra_in["raw_templates"]
        chan_snrs = extra_in["snr_by_channel"]
        cerr = np.linalg.norm(templates - temps_in, axis=(1, 2))
        rerr = np.linalg.norm(templates - raw_temps_in, axis=(1, 2))
        cerrm = np.abs(templates - temps_in).max(axis=(1, 2))
        rerrm = np.abs(templates - raw_temps_in).max(axis=(1, 2))
        for u, (ce, re, cem, rem, gtptp) in enumerate(zip(cerr, rerr, cerrm, rerrm, gtptps)):
            truens = min(np.sum(labels == u), ns)
            ns_records.append(
                dict(u=u, kind="cleaned", maxns=ns, ns=truens, l2dist=ce, maxdist=cem, gtptp=gtptp, fold=fold, beh=edge_behavior)
            )
            ns_records.append(
                dict(u=u, kind="raw", maxns=ns, ns=truens, l2dist=re, maxdist=rem, gtptp=gtptp, fold=fold, beh=edge_behavior)
            )
            tptps = templates[u].ptp(0)

            for c in range(384):
                csnr = chan_snrs[u, c]
                cptp = tptps[c]
                ccerr = np.linalg.norm(templates[u][:, c] - temps_in[u][:, c])
                rcerr = np.linalg.norm(templates[u][:, c] - raw_temps_in[u][:, c])
                ccerrm = np.abs(templates[u][:, c] - temps_in[u][:, c]).max()
                rcerrm = np.abs(templates[u][:, c] - raw_temps_in[u][:, c]).max()
                chan_records.append(
                    dict(u=u, kind="cleaned", maxns=ns, ns=truens, snr=csnr, l2dist=ccerr, maxdist=ccerrm, ptp=cptp, fold=fold, beh=edge_behavior)
                )
                chan_records.append(
                    dict(u=u, kind="raw", maxns=ns, ns=truens, snr=csnr, l2dist=rcerr, maxdist=rcerrm, ptp=cptp, fold=fold, beh=edge_behavior)
                )

    # temps_out, extra_out = snr_templates.get_templates(np.c_[tout, lout], geom, "/tmp/mearectmp/traces_cached_seg0.raw", tpca_rank=tpca_rank, spike_length_samples=sls, trough_offset=tgh)
    # raw_temps_out = extra_out["raw_templates"]

# %%
nsdf = pd.DataFrame.from_records(ns_records)

# %%
nsdf.to_csv("/Users/charlie/data/spike_sorting_paper_figs/denoised_templates_fig/nsdf.csv")

# %%
chandf = pd.DataFrame.from_records(chan_records)

# %%
nsdf["snr"] = np.sqrt(nsdf["ns"]) * nsdf["gtptp"]

# %%
nsdf.head()

# %%
chandf.head()

# %%
sns.relplot(data=nsdf, x="gtptp", hue="kind", y="l2dist", col="maxns", s=3, lw=0)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
# sns.scatterplot(data=nsdf, x="maxns", hue="kind", y="err", s=3, lw=0, ax=ax)
sns.lineplot(data=nsdf, x="maxns", hue="kind", y="l2dist", errorbar="sd", ax=ax)
plt.semilogx()
plt.ylabel("squared error")
plt.xlabel("max spike count")

# %%
nsdf["snr_quantized"] = 20 * (nsdf["snr"] // 20)

# %%
nsdf["ptp_quantized"] = 5 * (nsdf["gtptp"] // 5)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
# sns.scatterplot(data=nsdf, x="maxns", hue="kind", y="err", s=3, lw=0, ax=ax)
sns.lineplot(data=nsdf, x="snr_quantized", hue="kind", y="l2dist", errorbar="sd", ax=ax)
plt.semilogx()
plt.ylabel("squared error")
plt.xlabel("quantized snr (20*floor(snr/20))")

# %%
fig, ax = plt.subplots(figsize=(2, 2))
# sns.scatterplot(data=nsdf, x="maxns", hue="kind", y="err", s=3, lw=0, ax=ax)
sns.lineplot(data=nsdf, x="ptp_quantized", hue="kind", y="l2dist", errorbar="sd", ax=ax)
# plt.semilogx()
plt.ylabel("squared error")
plt.xlabel("quantized ptp (5*floor(ptp/5))")

# %%

# %%
# new hybrid recording for showing individual templates

# %%
n_units = 400

rg = np.random.default_rng(1)
rec = si.read_binary_folder("/Users/charlie/data/spike_sorting_paper_figs/subtraction_fig/subtraction_fig_data/zad5/")
T = rec.get_num_samples()
rr = int(1e-3 * rec.sampling_frequency)

frs = rg.uniform(0.001, 1, size=n_units) * T / 30000
times = []
labels = []
for u, fr in enumerate(frs):
    ns = max(4, rg.poisson(fr))
    while True:
        st = rg.integers(T, size=ns)
        st.sort()
        if np.diff(st).min() >= rr:
            break
    times.extend(st)
    labels.extend([u] * ns)
times = np.array(times)
order = np.argsort(times)
times = times[order]
labels = np.array(labels)[order]
templates0 = np.load("/Users/charlie/data/spike_sorting_paper_figs/denoised_templates_fig/templates_for_simulation.npy")
templates = templates0[rg.choice(len(templates0), size=n_units, replace=False)]
gtptps = templates.ptp(1).max(1)
uniq, counts = np.unique(labels, return_counts=True)

rec_hybrid = hybrid_recording(rec, times, labels, templates)
# !rm -rf /tmp/mearectmp
rec_saved = rec_hybrid.save_to_folder(folder="/tmp/mearectmp", n_jobs=5)
geom = rec_saved.get_channel_locations()

temps, extra = snr_templates.get_templates(
    np.c_[times, labels],
    geom,
    "/tmp/mearectmp/traces_cached_seg0.raw",
    tpca_rank=5,
    # spike_length_samples=sls,
    # trough_offset=tgh,
    # max_spikes_per_unit=ns,
    n_jobs=5,
    do_temporal_decrease=False,
    snr_threshold=50.0,
    zero_radius_um=10000,
    tpca_centered=False,
)

# %%
plt.figure(figsize=(2, 2))
plt.scatter(counts, gtptps, s=1, lw=0)

# %%
tempmcs = temps.ptp(1).argmax(1)

raw_temps = extra["raw_templates"]
pca_temps = extra["denoised_templates"]
weights = extra["weights"]

weights.shape

fullci = waveform_utils.make_contiguous_channel_index(384, n_neighbors=384)
visci = waveform_utils.make_pitch_channel_index(geom, pitch=20)
# visci = waveform_utils.make_channel_index(geom, 50)

templates_vis = waveform_utils.channel_subset_by_index(templates, tempmcs, fullci, visci)
temps_vis = waveform_utils.channel_subset_by_index(temps, tempmcs, fullci, visci)
raw_temps_vis = waveform_utils.channel_subset_by_index(raw_temps, tempmcs, fullci, visci)
pca_temps_vis = waveform_utils.channel_subset_by_index(pca_temps, tempmcs, fullci, visci)
weights_vis = waveform_utils.channel_subset_by_index(weights, tempmcs, fullci, visci)


# %%
def plot_choice(choice):
    
    # fig, (ax, ay) = plt.subplots(ncols=2, figsize=(4, 2))
    fig, ax = plt.subplots(ncols=1, figsize=(2, 2))
    
    mc = tempmcs[choice]
    temp = temps_vis[choice]
    raw_temp = raw_temps_vis[choice]
    pca_temp = pca_temps_vis[choice]
    weight = weights_vis[choice]
    
    trim=0
    domsbar = False
    dobar = False
    maa = np.abs(templates).max()
    zero_kw=None
    
    # lp = pgeom(pca_temp[None, trim:sls-trim], [mc], visci, geom, ax=ax, color="blue", lw=1, max_abs_amp=maa, show_zero=False, show_zero_kwargs=zero_kw, zlim=0.5)
    lr = pgeom(raw_temp[None, trim:sls-trim], [mc], visci, geom, ax=ax, color="k", lw=1, max_abs_amp=maa, show_zero=False, show_zero_kwargs=zero_kw, zlim=0.5, subar=5 * dobar, msbar=domsbar, show_chan_label=True)
    lt = pgeom(temp[None, trim:sls-trim], [mc], visci, geom, ax=ax, color="green", lw=1, max_abs_amp=maa, show_zero=False, show_zero_kwargs=zero_kw, zlim=0.5, show_chan_label=True)
    lt = pgeom(templates_vis[ix][None, trim:sls-trim], [mc], visci, geom, ax=ax, color="orange", lw=1, max_abs_amp=maa, show_zero=False, show_zero_kwargs=zero_kw, zlim=0.5, show_chan_label=True)
    # lt = pgeom(templates_vis[ix][None, trim:sls-trim], [mc], visci, geom, ax=ay, color="purple", lw=1, max_abs_amp=maa, show_zero=False, show_zero_kwargs=zero_kw, zlim=0.5)
    # lw = pgeom(weight[None, trim:sls-trim], [mc], visci, geom, ax=ax, color="gray", lw=1, max_abs_amp=maa, show_zero=False, show_zero_kwargs=zero_kw, zlim=0.5)
    
    ax.axis("off")
    # ay.axis("off")
    plt.show()
    plt.close(fig)


# %%
# find cases of interest
#  - 

# %% tags=[]
for ix in np.flatnonzero((counts < 100)):
    print(ix)
    print(counts[ix], gtptps[ix])
    plot_choice(ix)

# %%
weights.shape

# %%
mw = weights.max(axis=(1, 2))

# %%
mw[mw < 0.7]

# %% tags=[]
for ix in np.flatnonzero((gtptps < 6) & (counts < 20)):
    print(ix)
    print(counts[ix], gtptps[ix])
    plot_choice(ix)

# %% tags=[]
for ix in np.flatnonzero(mw < 0.7):
    print(ix)
    print(counts[ix], gtptps[ix], mw[ix])
    plot_choice(ix)

# %%

# %%

# %%

# %%
# figure plan:
#  - schematic
#  - 4 cases of interest on 6 channels, 2x2 grid
#  - error by max n spikes
#  - error by ptp at maxns=largest <not clear rn>

# %%
temps_fig_dir = Path("/Users/charlie/data/spike_sorting_paper_figs/denoised_templates_fig/")

# %%
nsdf["rms"] = nsdf["l2dist"] / np.sqrt(384 * 121)

# %%
nsdf["kind2"] = np.where(nsdf["kind"] == "cleaned", "denoised", "raw median")

# %%

# %%
# figure
fig = plt.figure(figsize=(7, 5))
panel_a, panel_b, panel_c = fig.subfigures(ncols=3, wspace=100, hspace=100, width_ratios=[1.75, 3.75, 1.5])

# axes
axes_a = panel_a.subplots(ncols=2, nrows=4, gridspec_kw=dict(hspace=0.2, wspace=0), sharey="row")
panel_a.subplots_adjust(left=0.075, right=1, bottom=0.1, top=0.95)
axes_b = panel_b.subplots(ncols=2, nrows=2, gridspec_kw=dict(hspace=0, wspace=0))
panel_b.subplots_adjust(left=0.1, right=0.95, bottom=0.0, top=1)
axes_c = panel_c.subplots(nrows=2, gridspec_kw=dict(hspace=0.25))
panel_c.subplots_adjust(left=0.15, right=1, bottom=0.075, top=0.95)

all_panels = [panel_a, panel_b, panel_c]
all_subfigures = [*all_panels, ]
any_ax = axes_a.flat[0]

# -------------------------------------------------- A

# show_a_tix = (61, 319)
# show_a_cix = (155, 343)
show_a_tix = (9, 319)
show_a_cix = (172, 343)
titles = ("low SNR", "middle SNR")

for col, tix, cix, title in zip(axes_a.T, show_a_tix, show_a_cix, titles):
    gt_ta = templates[tix][:, cix]
    dnt_ta = temps[tix][:, cix]
    raw_ta = raw_temps[tix][:, cix]
    pca_ta = pca_temps[tix][:, cix]
    w_ta = weights[tix][:, cix]

    hr, = col[0].plot(raw_ta, color="k")
    hg, = col[0].plot(gt_ta, color="orange")
    col[1].plot(raw_ta, color="k")
    hp, = col[1].plot(pca_ta, color="b")
    col[2].axhline(0, color="k", lw=0.8)
    col[2].axhline(1, color="k", lw=0.8)
    hw, = col[2].plot(w_ta, color="purple")
    col[3].plot(gt_ta, color="orange")
    hd, = col[3].plot(dnt_ta, color="green")
    
    col[0].set_title(title)
    
    for ax in col[[0, 1, 3]]:
        ax.axis("off")
    sns.despine(ax=col[2], left=True, right=True, top=True, bottom=True)
    col[2].set_xticks([])
    col[2].set_yticks([0, 1])

panel_a.legend(
    [hr, hg, hp, hw, hd],
    ["raw median", "ground truth", "low rank median", "weight", "denoised"],
    ncols=2,
    loc="lower center",
    frameon=False
)

# -------------------------------------------------- B

b_choices = [239, 61, 309, 18]
trim = 0
maa = np.abs(templates[b_choices]).max()

for tix, ax in zip(b_choices, axes_b.flat):
    dobar = tix == b_choices[-1]
    domsbar = tix == b_choices[-1]
    
    mc = tempmcs[tix]
    temp = temps_vis[tix]
    raw_temp = raw_temps_vis[tix]
    pca_temp = pca_temps_vis[tix]
    weight = weights_vis[tix]
    
    lr = pgeom(
        raw_temp[None, trim:sls-trim], [mc],
        visci,
        geom,
        ax=ax,
        color="k",
        lw=1,
        max_abs_amp=maa,
        show_zero=False,
        zlim=0.5,
        subar=5 * dobar,
        msbar=domsbar,
        show_chan_label=False,
    )
    lt = pgeom(
        templates_vis[tix][None, trim:sls-trim], [mc],
        visci,
        geom,
        ax=ax,
        color="orange",
        lw=1,
        max_abs_amp=maa,
        show_zero=False,
        zlim=0.5,
        show_chan_label=False,
    )
    lt = pgeom(
        temp[None, trim:sls-trim], [mc],
        visci,
        geom,
        ax=ax,
        color="green",
        lw=1,
        max_abs_amp=maa,
        show_zero=False,
        zlim=0.5,
        show_chan_label=False,
    )
    ax.set_xticks([])
    ax.set_yticks([])

sns.despine(ax=axes_b[0, 0], right=False, left=True)
sns.despine(ax=axes_b[0, 1])
sns.despine(ax=axes_b[1, 0], right=False, top=False, left=True, bottom=True)
sns.despine(ax=axes_b[1, 1], right=True, top=False, left=False, bottom=True)
# for r in range(len(axes_b)):
#     sns.despine(ax=axes_b[r, 0], top=False, right=False, left=True)
#     sns.despine(ax=axes_b[r, -1], right=True)
# for c in range(len(axes_b.T)):
#     sns.despine(ax=axes_b[0, c], right=False, top=True)
#     sns.despine(ax=axes_b[-1, c], right=False, top=False, bottom=True)
   

# -------------------------------------------------- C
ax_nsverr = axes_c[0]
sns.lineplot(data=nsdf, x="maxns", hue="kind2", y="rms", errorbar="sd", ax=ax_nsverr)
ax_nsverr.semilogx()
ax_nsverr.set_yticks([0.0, 0.5])
# ax_nsverr.set_ylabel("RMSE (s.u.)")
inline_ylabel(ax_nsverr, "RMSE (s.u.)")
ax_nsverr.set_xlabel("max spike count")
sns.move_legend(
    ax_nsverr, "upper left",
    bbox_to_anchor=(0.4, 1), ncol=1, title=None, frameon=False,
)

ax_snrverr = axes_c[1]
sns.lineplot(data=nsdf, x="snr_quantized", hue="kind2", y="rms", errorbar="sd", ax=ax_snrverr)
ax_snrverr.semilogx()
ax_snrverr.set_yticks([0.0, 0.5])
# ax_snrverr.set_ylabel("RMSE (s.u.)")
inline_ylabel(ax_snrverr, "RMSE (s.u.)")
ax_snrverr.set_xlabel("unit SNR")
sns.move_legend(
    ax_snrverr, "upper left",
    bbox_to_anchor=(0.4, 1), ncol=1, title=None, frameon=False,
)


# -------------------------------------------------- adjust
for subfig in all_subfigures:
    subfig.set_facecolor([0, 0, 0, 0])
    subfig.patch.set_facecolor([0, 0, 0, 0])
#     # show subfigure edges
#     subfig.patch.set_edgecolor([0, 0.5, 0.5, 1])
#     subfig.patch.set_linewidth(1)
for panel, label in zip(all_panels, string.ascii_uppercase):
    any_ax.text(0 + 0.05 * (label == "B"), 0.995, f"{{\\bf {label}}}", verticalalignment="top", fontsize=BIGGER_SIZE, transform=panel.transSubfigure)
    
fig.savefig(temps_fig_dir / "denoisedtemplatesfig.pdf")


# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
sns.catplot(data=chandf.query("ptp>=1"), x="kind", hue="kind", col="maxns", y="l2dist", kind="box", height=2, aspect=5/5, fliersize=2, flierprops=dict(marker="."))
plt.gcf().suptitle("MSE on channels with ptp at least 1", y=1.05)

# %%
sns.catplot(data=chandf.query("ptp>=1"), x="kind", hue="kind", col="maxns", y="maxdist", kind="box", height=2, aspect=5/5, fliersize=2, flierprops=dict(marker="."))
plt.gcf().suptitle("max dist on channels with ptp at least 1", y=1.05)

# %%
# cerrors = np.linalg.norm(templates - temps_in, axis=1)
# rerrors = np.linalg.norm(templates - raw_temps_in, axis=1)
cerrors = np.abs(templates - temps_in).max(axis=1)
rerrors = np.abs(templates - raw_temps_in).max(axis=1)

# %%
worset, worsec = np.nonzero((cerrors > rerrors) & (temps_in.ptp(1) > 1)[:, :])
order = np.argsort((cerrors - rerrors)[worset, worsec])[::-1]
worset = worset[order]
worsec = worsec[order]

# %%
for tix, cix in zip(worset[:10], worsec[:10]):
    fig, ax = plt.subplots(figsize=(2, 1))
    ax.plot(temps_in[tix][:, cix], label="denoised")
    ax.plot(raw_temps_in[tix][:, cix], label="raw median")
    ax.plot(templates[tix][:, cix], label="GT")
    fig.legend(loc="center left", bbox_to_anchor=(0.9, 0, 1, 1))
    ax.set_title(f"template {tix} channel {cix}")
    plt.show()
    plt.close(fig)

# %%

# %%

# %%
sns.relplot(data=nsdf, x="gtptp", hue="maxns", y="err", col="kind", s=3, lw=0)

# %%
sns.relplot(data=nsdf, x="snr", hue="kind", y="err", col="maxns", s=3, lw=0)

# %%
# fig, ax = plt.subplots(figsize=(2, 2))
# sns.scatterplot(data=nsdf, x="ns", hue="kind", y="err", s=3, lw=0, ax=ax)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
sns.scatterplot(data=nsdf, x="maxns", hue="kind", y="err", s=3, lw=0, ax=ax)
sns.lineplot(data=nsdf, x="maxns", hue="kind", y="err", errorbar="sd", ax=ax)

# %% tags=[]
pca_temps_in = extra_in["denoised_templates"]
weights_in = extra_in["weights"]
tempmcs = raw_temps_in.ptp(1).argmax(1)

fullci = waveform_utils.make_contiguous_channel_index(geom.shape[0], n_neighbors=geom.shape[0])
visci = waveform_utils.make_channel_index(geom, 35)

temps_in_vis = waveform_utils.channel_subset_by_index(temps_in, tempmcs, fullci, visci)
raw_temps_in_vis = waveform_utils.channel_subset_by_index(raw_temps_in, tempmcs, fullci, visci)
pca_temps_in_vis = waveform_utils.channel_subset_by_index(pca_temps_in, tempmcs, fullci, visci)
weights_in_vis = waveform_utils.channel_subset_by_index(weights_in, tempmcs, fullci, visci)

# %% tags=[]
nunits = 5
seed = 5
choices = np.random.default_rng(seed).choice(len(temps_in), size=nunits, replace=False)
maa = np.abs(temps_in[choices]).max() / 1.1
sls = temps_in.shape[1]
trim = 0

zero_kw = dict(color="gray", lw=0.5, linestyle="--")

fig, axes = plt.subplots(ncols=nunits, figsize=(7, 3.5))
fig.subplots_adjust(top=1, bottom=0.1)

for ax, choice in zip(axes.flat, choices):
    mc = tempmcs[choice]
    temp = temps_in_vis[choice]
    raw_temp = raw_temps_in_vis[choice]
    pca_temp = pca_temps_in_vis[choice]
    weight = weights_in_vis[choice]
    
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

# %% tags=[]
raw_temps_out.shape

# %%
cerr0 = cerr
rerr0 = rerr

# %%
cerr1 = cerr
rerr1 = rerr

# %% tags=[]
cerr = np.linalg.norm(templates - temps_in, axis=(1, 2))
rerr = np.linalg.norm(templates - raw_temps_in, axis=(1, 2))

# %%
plt.hist(cerr - cerr1, bins=32);

# %% tags=[]
mn = min(cerr.min(), rerr.min())
mx = max(cerr.max(), rerr.max())
# plt.scatter(cerr0, rerr0, s=3, lw=0)
# plt.scatter(cerr1, rerr1, s=3, lw=0)
plt.scatter(cerr, rerr, s=3, lw=0)
plt.plot([mn, mx], [mn, mx], lw=0.8, c="k")

# %% tags=[]
plt.hist(cerr - rerr)

# %%

# %%

# %% tags=[]
st = np.c_[*st[0]]

# %%

# %% tags=[]
times, units = st[0]

# %% tags=[]
labels, counts = np.unique(units, return_counts=True)

# %% tags=[]
counts

# %%
