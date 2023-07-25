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
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import resample
import colorcet as cc
import spikeinterface.full as si
import h5py
from tqdm.auto import tqdm, trange
from scipy import linalg as la
import pickle

# %%
from spike_psvae import (
    subtract,
    cluster_utils,
    cluster_viz_index,
    ibme,
    ibme_corr,
    newms,
    waveform_utils,
    chunk_features,
    drifty_deconv,
    spike_train_utils,
    snr_templates,
    extract_deconv,
    localize_index,
    outliers,
)

# %%
from spike_psvae.hybrid_analysis import Sorting

# %%
# from reglib import ap_filter, lfpreg

# %%
# %matplotlib inline
plt.rc("figure", dpi=200)
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# %% [markdown] tags=[]
# ## Paths / config

# %%
humanX = "human6"

# %%
dsroot = Path("/Users/charlie/data/beeStinger") / humanX

# %%
bsout = Path("/Users/charlie/data/beeStingerFeb")
dsout = bsout / humanX
dsout.mkdir(exist_ok=True, parents=True)


# %%
fs = 20_000
nc = 128

# %%
chanmap_mat = dsroot / ".." / "chanMap.mat"
h = loadmat(chanmap_mat)
cm0i = h["chanMap0ind"].squeeze().astype(int)
geom = np.c_[h['xcoords'].squeeze(), h['ycoords'].squeeze()]
geom.shape

# %%
icm0i = np.zeros_like(cm0i)
for i, ix in enumerate(cm0i):
    icm0i[ix] = i

# %%
t_start = 700
t_end = None

# %%
trough_offset = 30
spike_length_samples = 82

# %% [markdown] tags=[]
# ## SpikeInterface preprocessing

# %%
recorded_geom = np.array([0, geom[:, 1].max()]) - geom[icm0i]

# %%
rec = si.read_neuroscope_recording(dsroot / "amplifier.xml")
rec.set_dummy_probe_from_locations(geom, shape_params=dict(radius=10))
probe = rec.get_probe()
probe.annotate(name="beeStinger")
probe.set_contacts(recorded_geom)
# probe.set_device_channel_indices(cm0i)
probe.set_contact_ids([str(i) for i in range(128)])
probe.create_auto_shape()
rec.set_probe(probe, in_place=True)
rec

# %%
plt.figure(figsize=(3, 2))
noise_levels = si.get_noise_levels(rec)
plt.hist(noise_levels, bins=32);
dead_level = 86.4e-6
plt.axvline(dead_level, color="k");

# %% tags=[]
fig, ax = plt.subplots(figsize=(4, 25))
si.plot_probe_map(rec, with_contact_id=True, with_channel_ids=True, ax=ax)
plt.title("beeStinger Probe Map")
plt.savefig(dsout / "probemap.png")

# %%
rec_filtered = si.bandpass_filter(rec, freq_min=300., freq_max=6000., dtype='float32')
rec = si.common_reference(rec_filtered, reference='global', operator='median')

# %%
dead_chans = np.flatnonzero(noise_levels < dead_level)

# %% tags=[]
for s in range(0, int(rec.get_total_duration()), 250):
    fig = plt.figure()
    chunk = rec.get_traces(start_frame=s*fs, end_frame=s*fs+1000)
    plt.imshow(chunk.T, aspect="auto", interpolation="none")
    plt.colorbar(shrink=0.3)
    for ix in dead_chans:
        y = ix
        plt.axhline(y, ls=":", lw=1, color="orange")
        # ab.axhline(icm0i[ix], ls=":", lw=1, color="orange")
    plt.title(f"Filtered/CMR raw recording with dead channels marked, start time {s}s")
    plt.xlabel("time (samples)")
    plt.ylabel("channel")
    plt.show()
    fig.savefig(dsout / f"raw_rec_{s}.png")
    plt.close("all")


# %%
alive_chans = np.setdiff1d(np.arange(nc), dead_chans)

# %%
depthsort = np.argsort(recorded_geom[alive_chans, 1])
rec_sliced = rec_filtered.channel_slice([str(c) for c in alive_chans[depthsort]])
rec = rec_sliced.frame_slice(t_start * fs, t_end * fs if t_end else None)
rec = si.common_reference(rec, reference='global', operator='median')
rec = si.zscore(rec)

# %%
rec

# %%

# %%

# %%

# %% tags=[]
fig, ax = plt.subplots(figsize=(4, 25))
si.plot_probe_map(rec, with_contact_id=True, with_channel_ids=True, ax=ax)
plt.title("beeStinger Probe Map: alive channels")
plt.savefig(dsout / "probemap_alivechans.png")

# %%
# %rm -rf {dsout / "sippx"}

# %%
binrec = rec.save(folder=dsout / "sippx", n_jobs=8, chunk_size=fs)

# %% tags=[]
fig, ax = plt.subplots(figsize=(4, 25))
si.plot_probe_map(binrec, with_contact_id=True, with_channel_ids=True, ax=ax)
plt.title("beeStinger Probe Map: channel subset binary")
plt.savefig(dsout / "probemap_channel_subset_binary.png")

# %% tags=[]
for s in range(0, int(binrec.get_total_duration()), 250):
    fig = plt.figure()
    chunk = binrec.get_traces(start_frame=s*fs, end_frame=s*fs+1000)
    plt.imshow(chunk.T, aspect="auto")
    plt.colorbar(shrink=0.3)
    plt.title(f"Channel-subset, re-filtered, CMRd, and standardized binary. Start time {s}s")
    plt.xlabel("time (samples)")
    plt.ylabel("channels (subset)")
    plt.show()
    fig.savefig(dsout / f"ppx_rec_{s}.png")
    plt.close("all")

# %% [markdown]
# # Detection / featurization

# %%
geom = np.load(dsout / "sippx" / "properties" / "location.npy")

# %%
trough_offset, spike_length_samples

# %% tags=[]
sub_h5 = subtract.subtraction_binary(
    dsout / "sippx" / "traces_cached_seg0.raw",
    dsout / "sub",
    geom=geom,
    # save_waveforms=True,
    save_residual=False,
    n_sec_pca=80,
    peak_sign="both",
    enforce_decrease_kind="radial",
    neighborhood_kind="circle",
    sampling_rate=fs,
    do_nn_denoise=False,
    thresholds=[5],
    n_jobs=8,
    overwrite=True,
    localize_radius=100,
    save_subtracted_tpca_projs=False,
    save_cleaned_tpca_projs=True,
    save_denoised_tpca_projs=False,
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
)

# %%
overwrite = True
# overwrite = False

with h5py.File(next((dsout / "sub").glob("sub*h5")), "r+" if overwrite else "r") as h5:
    if "z_reg" in h5:
        print("already done, skip")

    samples = h5["spike_index"][:, 0] - h5["start_sample"][()]
    x, y, z_abs, alpha, _ = h5["localizations"][:].T
    maxptp = maxptps = h5["maxptps"][:]
    spike_index = h5["spike_index"][:]
    maxchans = spike_index[:, 1]
    t = spike_index[:, 0] / fs
    z = z_abs
    
    # dispmap -= dispmap.mean()
    if "z_reg" in h5 and not overwrite:
        z_reg = h5["z_reg"][:]
        pap = p = h5["p"][:]
    else:
        if overwrite and "z_reg" in h5:
            del h5["z_reg"]
            del h5["p"]
        
        # z_reg, dispmap = ibme.register_nonrigid(
        z_reg, p = ibme.register_rigid(
            maxptps,
            z_abs - z_abs.min(),
            (samples - samples.min()) / 20000,
            # robust_sigma=1,
            # corr_threshold=0.3,
            adaptive_mincorr_percentile=5,
            disp=300,
            denoise_sigma=0.1,
            # max_dt=100,
            prior_lambda=1,
            batch_size=64,
        )
        z_reg -= (z_reg - z_abs).mean()
        h5.create_dataset("z_reg", data=z_reg)
        h5.create_dataset("p", data=p)
    pap = p
    t = spike_index[:, 0] / fs

# %%
fig, ax = plt.subplots(figsize=(8, 6))
for zz in np.unique(geom[:, 1]):
    ax.axhline(zz, lw=1, color="k", alpha=0.2)
ax.scatter(t, geom[maxchans, 1], c=np.clip(maxptp, 0, 15), s=50, alpha=1, linewidths=0, marker=".")
plt.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(0, 15), plt.cm.viridis),
    label="denoised peak-to-peak amp.",
    shrink=0.5,
    pad=0.025,
    ax=ax,
)
tt = np.arange(0, 100 * (t.max() // 100) + 1, 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("detection channel depth (um)")
plt.title(f"{humanX}: time vs. maxchan.  {len(maxptp)} spikes.")
fig.savefig(dsout / "scatter_t_v_maxchan.png")

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(t, z, c=np.clip(maxptp, 0, 15), s=5, alpha=0.5, linewidths=0, marker=".")
plt.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(0, 15), plt.cm.viridis),
    label="denoised peak-to-peak amp.",
    shrink=0.5,
    pad=0.025,
    ax=ax,
)
ax.plot(geom.max() / 2 + p, color="k", lw=2, label="drift est.")
for zz in np.unique(geom[:, 1]):
    ax.axhline(zz, lw=1, color="k", alpha=0.2)
ax.legend()
tt = np.arange(0, 100 * (t.max() // 100) + 1, 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y.  {len(maxptp)} spikes.")
fig.savefig(dsout / "scatter_t_v_y.png")

# %%
fig = plt.figure(figsize=(8, 6))
plt.scatter(t, z_reg, c=np.clip(maxptp, 0, 15), s=5, alpha=0.5, marker=".", linewidths=0)
plt.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(0, 15), plt.cm.viridis),
    label="denoised peak-to-peak amp.",
    shrink=0.5,
    pad=0.025,
    ax=plt.gca(),
)
tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. registered y.  {len(maxptp)} spikes.")
fig.savefig(dsout / "scatter_t_v_regy.png")

# %%
fig, (aa, ab) = plt.subplots(ncols=2, figsize=(6, 6), gridspec_kw=dict(wspace=0.05))
ordd = np.argsort(maxptp)
aa.scatter(x[ordd], z_abs[ordd], c=np.clip(maxptp[ordd], 0, 15), s=1, alpha=0.25, linewidths=0)
ab.scatter(np.log(5+maxptp[ordd]), z_abs[ordd], c=np.clip(maxptp[ordd], 0, 15), s=1, alpha=0.25, linewidths=0)
plt.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(0, 15), plt.cm.viridis),
    label="denoised peak-to-peak amp.",
    shrink=0.5,
    pad=0.025,
    ax=ab,
)
aa.set_xlabel("x (um)")
ab.set_xlabel("log(5 + amplitude)")
ab.set_yticks([])
aa.set_ylabel("depth (um)")
fig.suptitle(f"{humanX}: x vs. y.  {len(maxptp)} spikes.", y=0.92)
fig.savefig(dsout / "scatter_x_v_y.png")

# %%
fig, (aa, ab) = plt.subplots(ncols=2, figsize=(6, 6), gridspec_kw=dict(wspace=0.05))
ordd = np.argsort(maxptp)
aa.scatter(x[ordd], z_reg[ordd], c=np.clip(maxptp[ordd], 0, 15), s=1, alpha=0.25, linewidths=0)
ab.scatter(np.log(5+maxptp[ordd]), z_reg[ordd], c=np.clip(maxptp[ordd], 0, 15), s=1, alpha=0.25, linewidths=0)
plt.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(0, 15), plt.cm.viridis),
    label="denoised peak-to-peak amp.",
    shrink=0.5,
    pad=0.025,
    ax=ab,
)
aa.set_xlabel("x (um)")
ab.set_xlabel("log(5 + amplitude)")
ab.set_yticks([])
aa.set_ylabel("depth (um)")
fig.suptitle(f"{humanX}: x vs. registered y.  {len(maxptp)} spikes.", y=0.92)
fig.savefig(dsout / "scatter_x_v_regy.png")

# %%
fig, (aa, ab) = plt.subplots(ncols=2, figsize=(8, 3))

counts, edges, _ = aa.hist(pap, bins=128, color="k")
aa.set_xlabel("est. displacement")
aa.set_ylabel("frequency")

ab.plot(t_start + np.arange(len(pap)), pap, c="gray", lw=1)
ab.scatter(t_start + np.arange(len(pap)), pap, c="k", s=1, zorder=12)
ab.set_ylabel("est. displacement")
ab.set_xlabel("time (s)")
fig.suptitle(f"{humanX}, start time {t_start}", y=0.95, fontsize=10)

fig.savefig(bsout / f"{humanX}_disp_tstart{t_start}.png", dpi=300)
np.savetxt(bsout / f"{humanX}_disp_tstart{t_start}.csv", pap, delimiter=",")

# %%
pitch = waveform_utils.get_pitch(geom)

# %%
pitch

# %%
fig, (aa, ab) = plt.subplots(ncols=2, figsize=(8, 3))

counts, edges, _ = aa.hist(p, bins=128, color="k")

p_mode = edges[counts.argmax():counts.argmax()+2].mean()
lo = p_mode - pitch / 2
hi = p_mode + pitch / 2
p_good = (lo < p) & (hi > p)

aa.set_xlabel("est. displacement")
aa.set_ylabel("frequency")

aa.axvline(edges[counts.argmax():counts.argmax()+2].mean(), color=plt.cm.Greens(0.4), label="mode est.", lw=2, ls="--")
aa.legend()
ab.plot(t_start + np.arange(len(pap)), pap, c="gray", lw=1)
ab.scatter(t_start + np.arange(len(p))[~p_good], p[~p_good], c="k", s=1, label=f"disp too far {(~p_good).sum()}s", zorder=12)
ab.scatter(t_start + np.arange(len(p))[p_good], p[p_good], color=plt.cm.Greens(0.4), s=1, label=f"disp within pitch/2 of mode {(p_good).sum()}s", zorder=12)
ab.set_ylabel("est. displacement")
ab.set_xlabel("time (s)")
ab.legend()

fig.savefig(dsout / "stable_bins.png", dpi=300)

# %%
sub_h5 = next((dsout / "sub").glob("sub*h5"))

# %%
(dsout / "clust").mkdir(exist_ok=True)

# %%
geom = np.load(dsout / "sippx" / "properties" / "location.npy")
geom.shape

# %%
spike_index.shape

# %%
si0 = spike_index.copy()

# %% tags=[]
st = spike_index.copy()
# st[:, 1] = newms.registered_maxchan(st, p, geom, pfs=fs)
good_times = np.isin(st[:, 0] // fs, np.flatnonzero(p_good))
st[~good_times, 1] = -1
spike_train, templates, order = newms.new_merge_split(
    st,
    geom.shape[0],
    dsout / "sippx" / "traces_cached_seg0.raw",
    sub_h5,
    geom,
    dsout / "clust",
    n_workers=1,
    # merge_resid_threshold=2.5,
    merge_resid_threshold=3.0,
    relocated=True,
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    # exp_split=True,
    # herding_npcs=2,
    # extra_pc_split=True,
)

# %% tags=[]
for k in ("split", "merge"):
    visst = np.load(dsout / "clust" / f"{k}_st.npy")
    vissort = Sorting(
        dsout / "sippx" / "traces_cached_seg0.raw",
        geom,
        visst[:, 0],
        spike_train_utils.make_labels_contiguous(visst[:, 1]),
        name="StableClust" + k.capitalize(),
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )
    od = dsout / "summaries_orig_mt3.0"
    vissort.make_unit_summaries(out_folder=od / f"stableclust_{k}_summaries_raw")
    with h5py.File(sub_h5) as h5:
        vissort.make_unit_summaries(
            out_folder=od / f"stableclust_{k}_summaries_cleaned",
            stored_maxchans=h5["spike_index"][:, 1],
            stored_order=order,
            stored_channel_index=h5["channel_index"][:],
            stored_tpca_projs=h5["cleaned_tpca_projs"],
            stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
            show_scatter=False,
            relocated=False,
            n_jobs=1,
        )

# %%
# sort spike train by Z
tx, ty, _, tz, ta = localize_index.localize_ptps_index(
    templates.ptp(1), geom, templates.ptp(1).argmax(1),
    np.array([np.arange(geom.shape[0])] * geom.shape[0]),
    radius=100,
)
zord = np.argsort(tz)
templates = templates[zord]
zord_inv = np.arange(len(zord))[zord]
zord_inv = np.concatenate([[-1], zord_inv], axis=0)
spike_train = np.c_[spike_train[:, 0], spike_train_utils.make_labels_contiguous(zord_inv[1 + spike_train[:, 1]])]
spike_train[:, 1].max() + 1

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw=dict(wspace=0.1))
cluster_viz_index.array_scatter(
    spike_train[:, 1],
    geom,
    x[order],
    z_reg[order],
    maxptp[order],
    axes=axes,
    zlim=(geom.min() - 25, geom.max() + 25),
    c=5,
)
cax = axes[2].inset_axes([0.05, 0.02, 0.075, 0.2])
cbar = plt.colorbar(axes[0].collections[0], ax=axes[1], cax=cax, label="amplitude")
cbar.solids.set(alpha=1)
cax.set_yticks([0, 1], labels=[3, 15])
axes[0].set_ylabel("depth (um)")
axes[1].set_yticks([])
axes[1].set_xlabel("log peak-to-peak amplitude")
axes[2].set_yticks([])
nunits = np.setdiff1d(np.unique(spike_train[:, 1]), [-1]).size
axes[1].set_title(f"Spatial view of clustered and triaged spikes. {nunits} units.")
fig.savefig(dsout / "reloc_cluster_scatter.png")

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw=dict(wspace=0.1))
cluster_viz_index.array_scatter(
    spike_train[:, 1],
    geom,
    x[order],
    z_reg[order],
    maxptp[order],
    axes=axes,
    zlim=(500, 1000),
    c=5,
)
cax = axes[2].inset_axes([0.05, 0.02, 0.075, 0.2])
cbar = plt.colorbar(axes[0].collections[0], ax=axes[1], cax=cax, label="amplitude")
cbar.solids.set(alpha=1)
cax.set_yticks([0, 1], labels=[3, 15])
axes[0].set_ylabel("depth (um)")
axes[1].set_yticks([])
axes[1].set_xlabel("log peak-to-peak amplitude")
axes[2].set_yticks([])
axes[1].set_title("Spatial view of clustered and triaged spikes. {nunits} units.")
fig.savefig(dsout / "reloc_cluster_scatter_detail.png")

# %%
kept = spike_train[:, 1] >= 0
triaged = spike_train[:, 1] < 0

fig = plt.figure(figsize=(8, 6))
plt.scatter(t[order][triaged & ~good_times], z_reg[order][triaged & ~good_times], color="k", s=5, alpha=0.5, marker=".", linewidths=0, label="outside stable")
plt.scatter(t[order][triaged & good_times], z_reg[order][triaged & good_times], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[order][kept], z_reg[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
plt.legend(markerscale=2.5)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. registered y.  {kept.sum()} sorted spikes.")
fig.savefig(dsout / "reloc_scatter_sorted_t_v_regy.png", dpi=300)

# %%
kept = spike_train[:, 1] >= 0
triaged = spike_train[:, 1] < 0

fig = plt.figure(figsize=(8, 6))
plt.scatter(t[order][triaged & ~good_times], z[order][triaged & ~good_times], color="k", s=5, alpha=0.5, marker=".", linewidths=0, label="outside stable")
plt.scatter(t[order][triaged & good_times], z[order][triaged & good_times], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[order][kept], z[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
plt.legend(markerscale=2.5)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y.  {kept.sum()} sorted spikes.")
fig.savefig(dsout / "reloc_scatter_sorted_t_v_y.png")

# %%
# with h5py.File(sub_h5) as h5:
#     sorting_orig.make_unit_summaries(
#         out_folder=dsout / "summaries" / "stableclust_summaries_cleanedtpca",
#         stored_channel_index=h5["channel_index"][:],
#         stored_maxchans=h5["spike_index"][:, 1],
#         stored_order=order,
#         stored_tpca_projs=h5["cleaned_tpca_projs"],
#         stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
#         show_scatter=False,
#         n_jobs=1,
#         nchans=12,
#     )

# %%
1

# %% [markdown]
# ## Deconv1

# %%
sub_h5 = next((dsout / "sub").glob("sub*h5"))
with h5py.File(sub_h5) as h5:
    geom = h5["geom"][:]
    p = h5["p"][:]
    
    plt.figure(figsize=(1,1))
    counts, edges, _ = plt.hist(p, bins=128, color="k")

    p_mode = edges[counts.argmax():counts.argmax()+2].mean()
    lo = p_mode - 25 / 4
    hi = p_mode + 25 / 4
    p_good = (lo < p) & (hi > p)

# %%
spike_train_al, order, templates, template_shifts = spike_train_utils.clean_align_and_get_templates(
    np.load(dsout / "clust" / "merge_st.npy"),
    geom.shape[0],
    dsout / "sippx" / "traces_cached_seg0.raw",
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
)

# %%
assert (order == np.arange(len(order))).all()

# %%
# !rm -rf {dsout / "deconv1"}/*

# %% tags=[]
with h5py.File(sub_h5) as h5:
    merge_order = np.load(dsout / "clust" / "merge_order.npy")
    superres = drifty_deconv.superres_deconv(
        dsout / "sippx" / "traces_cached_seg0.raw",
        geom,
        h5["localizations"][:, 2][merge_order],
        h5["p"][:],
        spike_train=spike_train_al,
        # templates=templates,
        reference_displacement=p_mode,
        bin_size_um=25,
        pfs=fs,
        n_jobs=8,
        deconv_dir=dsout / "deconv1",
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
        # max_upsample=4,
        # refractory_period_frames=7,
    )

# %% tags=[]
extract_h5, extra = drifty_deconv.extract_superres_shifted_deconv(
    superres,
    save_cleaned_waveforms=True,
    save_cleaned_tpca_projs=True,
    save_residual=False,
    sampling_rate=fs,
    subtraction_h5=sub_h5,
    nn_denoise=False,
    geom=geom,
    n_jobs=1,
    # save_reassignment_residuals=True,
    # pairs_method="radius",
    max_resid_dist=20,
    do_reassignment_tpca=True,
    do_reassignment=True,
    n_sec_train_feats=80,
    tpca_weighted=True,
)

# %%
extract_h5

# %%
(dsout / "deconv1").exists()

# %%
# import pickle
# with open(dsout / "deconv1" / "deconv_info.pkl", "wb") as jar:
#     pickle.dump(dict(superres=superres, extra=extra), jar)

# %%
1

# %% tags=[]
with h5py.File(extract_h5, "r") as h5:
    orig_dst = h5["deconv_spike_train"][:]
    reas_labels = h5["reassigned_unit_labels"][:]
    
    stable_visst = orig_dst.copy()
    unstable = ~np.isin((stable_visst[:, 0] - trough_offset) // fs, np.flatnonzero(p_good))
    stable_visst[unstable, 1] = -1
    stable_reas_labels = np.where(unstable, -1, reas_labels)
    
    vissort = Sorting(
        dsout / "sippx" / "traces_cached_seg0.raw",
        geom,
        stable_visst[:, 0],
        stable_visst[:, 1],
        name="StableDeconv1",
        spike_maxchans=h5["spike_index"][:, 1],
        # do_cleaned_templates=True,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )
    vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"{vissort.name_lo}_raw")
    vissort.make_unit_summaries(
        out_folder=dsout / "summaries" / f"{vissort.name_lo}_cleaned",
        stored_maxchans=h5["spike_index"][:, 1],
        # stored_order=order,
        stored_channel_index=h5["channel_index"][:],
        stored_waveforms=h5["cleaned_waveforms"],
        show_scatter=False,
        n_jobs=1,
    )
    
    vissort = Sorting(
        dsout / "sippx" / "traces_cached_seg0.raw",
        geom,
        stable_visst[:, 0],
        stable_reas_labels,
        name="StableDeconv1Reas",
        spike_maxchans=h5["spike_index"][:, 1],
        # do_cleaned_templates=True,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )
    vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"{vissort.name_lo}_raw")
    vissort.make_unit_summaries(
        out_folder=dsout / "summaries" / f"{vissort.name_lo}_cleaned",
        stored_maxchans=h5["spike_index"][:, 1],
        # stored_order=order,
        stored_channel_index=h5["channel_index"][:],
        stored_waveforms=h5["cleaned_waveforms"],
        show_scatter=False,
        n_jobs=1,
    )
    
    # vissort.make_unit_summaries(
    #     out_folder=dsout / "summaries" / f"stabledeconv2_cleanedtpca_relocated",
    #     stored_maxchans=h5["spike_index"][:, 1],
    #     # stored_order=order,
    #     stored_channel_index=h5["channel_index"][:],
    #     stored_tpca_projs=h5["cleaned_tpca_projs"],
    #     stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
    #     show_scatter=False,
    #     relocated=True,
    #     n_jobs=1,
    # )
    # vissort.make_resid_studies(out_folder=dsout / "resid_studies" / f"stabledeconv2_resid_studies")

# %%
1

# %%

# %%

# %%

# %%

# %%
with h5py.File(extract_h5, "r") as h5:
    storig = spike_train_orig = h5["deconv_spike_train"][:]
    in_stable = ~np.isin((spike_train_orig[:, 0] - trough_offset) // fs, np.flatnonzero(p_good))
    stable_sto = spike_train_orig.copy()
    stable_sto[in_stable, 1] = -1
    
    new_labels = h5["reassigned_unit_labels"][:]
    stable_new_labels = new_labels.copy()
    stable_new_labels[in_stable] = -1
    
    spike_train_sup = h5["superres_deconv_spike_train"][:]
    stable_suporig_labels = spike_train_sup[:, 1].copy()
    stable_suporig_labels[in_stable] = -1

    supnew_labels = h5["reassigned_superres_labels"][:]
    stable_supnew_labels = supnew_labels.copy()
    stable_supnew_labels[in_stable] = -1
    
    shuporig_labels = h5["superres_deconv_spike_train_shifted_upsampled"][:, 1]
    stable_shuporig_labels = shuporig_labels.copy()
    stable_shuporig_labels[in_stable] = -1
    
    shupnew_labels = h5["reassigned_shifted_upsampled_labels"][:]
    stable_shupnew_labels = shupnew_labels.copy()
    stable_shupnew_labels[in_stable] = -1

    print(f"{(stable_supnew_labels != stable_suporig_labels).mean()=}")
    print(f"{(stable_new_labels != stable_sto[:, 1]).mean()=}")
    
    superres_templates = h5["superres_templates"][:]
    superres_label_to_orig_label = h5["superres_label_to_orig_label"][:]
    superres_label_to_bin_id = h5["superres_label_to_bin_id"][:]

    if "reassigned_scores" in h5:
        reassco = h5["reassigned_scores"][:]
    
    unit_raw_templates = snr_templates.get_raw_templates(stable_sto, geom, dsout / "sippx" / "traces_cached_seg0.raw", trough_offset=trough_offset, spike_length_samples=spike_length_samples)

# %%
1

# %%
overwrite = True
# overwrite = False

with h5py.File(extract_h5, "r+") as h5:
    samples = h5["spike_index"][:, 0] #- h5["start_sample"][()]
    x, y, z_abs, alpha, _ = h5["localizations"][:].T
    maxptp = maxptps = h5["maxptps"][:]
    spike_index = h5["spike_index"][:]
    maxchans = spike_index[:, 1]
    t = spike_index[:, 0] / fs
    z = z_abs
    
    # dispmap -= dispmap.mean()
    if "z_reg" in h5 and not overwrite:
        z_reg = h5["z_reg"][:]
        pap = p = h5["p"][:]
    else:
        if overwrite and "z_reg" in h5:
            del h5["z_reg"]
            del h5["p"]
        
        # z_reg, dispmap = ibme.register_nonrigid(
        z_reg, p = ibme.register_rigid(
            maxptps,
            z_abs - z_abs.min(),
            (samples - samples.min()) / 20000,
            # robust_sigma=1,
            # corr_threshold=0.3,
            adaptive_mincorr_percentile=5,
            disp=300,
            denoise_sigma=0.1,
            # max_dt=100,
            prior_lambda=1,
            batch_size=64,
        )
        z_reg -= (z_reg - z_abs).mean()
        h5.create_dataset("z_reg", data=z_reg)
        h5.create_dataset("p", data=p)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(t, z, c=np.clip(maxptp, 0, 15), s=5, alpha=0.5, linewidths=0, marker=".")
plt.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(0, 15), plt.cm.viridis),
    label="denoised peak-to-peak amp.",
    shrink=0.5,
    pad=0.025,
    ax=ax,
)
ax.plot(geom.max() / 2 + p, color="k", lw=2, label="drift est.")
for zz in np.unique(geom[:, 1]):
    ax.axhline(zz, lw=1, color="k", alpha=0.2)
ax.legend()
tt = np.arange(0, 100 * (t.max() // 100) + 1, 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y.  {len(maxptp)} spikes.")
fig.savefig(dsout / "scatter_t_v_y_postdeconv.png")

# %%
extract_h5

# %%
with h5py.File(extract_h5, "r") as h5:
    superres_templates = h5["superres_templates"][:]
    superres_templates = h5["superres_templates"][:]
    superres_label_to_orig_label = h5["superres_label_to_orig_label"][:]
    superres_label_to_bin_id = h5["superres_label_to_bin_id"][:]
    
    shup_templates = h5["all_shifted_upsampled_temps"][:]
    print(shup_templates.shape)
    shup_shifts = h5["shifted_upsampled_idx_to_shift_id"][:]
    stable_shup_temps = shup_templates[shup_shifts == 1]
    print(stable_shup_temps.shape)
    shup_supids = h5["shifted_upsampled_idx_to_superres_id"][:]
    stable_shup_supids = shup_supids[shup_shifts == 1]
    
    
    storig = h5["deconv_spike_train"][:]
    cluster_viz_index.superres_templates_viz(
        superres_templates,
        superres_label_to_orig_label,
        superres_label_to_bin_id,
        storig,
        dsout / "superres_templates" / "deconv1",
        geom,
        radius=200,
    )
    
    storig = h5["deconv_spike_train"][:]
    cluster_viz_index.superres_templates_viz(
        stable_shup_temps,
        superres_label_to_orig_label[stable_shup_supids],
        superres_label_to_bin_id[stable_shup_supids],
        storig,
        dsout / "superres_templates" / "deconv1_shup",
        geom,
        radius=200,
    )

# %%
1

# %%
# with open(dsout / "deconv1" / "deconv_info.pkl", "wb") as jar:
#     pkl = pickle.load(jar)
#     superres = pkl["superres"]
#     # dict(superres=superres, extra=extra)

# %% tags=[]
# exps = {
#     # "Deconv1ReasInf": (False, np.inf),
#     # "Deconv1Reas2": (False, 2.0),
#     # "Deconv1ReasInfPC": (True, np.inf),
#     "Deconv1Reas2PC": (True, 2.0),
# }

# for name, (do_tpca, norm_p) in exps.items():
#     print("-" * 80)
#     print(name)
    
#     extract_h5, extra = drifty_deconv.extract_superres_shifted_deconv(
#         superres,
#         save_cleaned_waveforms=True,
#         save_cleaned_tpca_projs=True,
#         save_residual=False,
#         sampling_rate=fs,
#         subtraction_h5=sub_h5,
#         nn_denoise=False,
#         geom=geom,
#         n_jobs=1,
#         do_reassignment=True,
#         # save_reassignment_residuals=True,
#         tpca_weighted=True,
#         do_reassignment_tpca=do_tpca,
#         reassignment_norm_p=norm_p,
#     )
    
#     with h5py.File(extract_h5) as h5:    
#         orig_dst = h5["deconv_spike_train"][:]
#         reas_labels = h5["reassigned_unit_labels"][:]


#         stable_visst = orig_dst.copy()
#         stable_reas_labels = reas_labels.copy()
#         unstable = ~np.isin((stable_visst[:, 0] - trough_offset) // fs, np.flatnonzero(p_good))
#         stable_reas_labels[unstable] = -1
#         stable_visst[unstable, 1] = -1
#         vissort = Sorting(
#             dsout / "sippx" / "traces_cached_seg0.raw",
#             geom,
#             stable_visst[:, 0],
#             stable_visst[:, 1],
#             name=f"Stable{name}",
#             spike_maxchans=h5["spike_index"][:, 1],
#             trough_offset=trough_offset,
#             spike_length_samples=spike_length_samples,
#         )
#         vissort.make_unit_summaries(out_folder=dsout / "reassign_exps_summaries" / f"{vissort.name_lo}_raw", n_jobs=1)
#         vissort.make_unit_summaries(
#             out_folder=dsout / "reassign_exps_summaries" / f"{vissort.name_lo}_cleaned",
#             stored_maxchans=h5["spike_index"][:, 1],
#             # stored_order=order,
#             stored_channel_index=h5["channel_index"][:],
#             stored_waveforms=h5["cleaned_waveforms"],
#             show_scatter=False,
#             n_jobs=1,
#         )

# #         spike_train_sup = h5["superres_deconv_spike_train"][:]
# #         stable_suporig_labels = spike_train_sup[:, 1].copy()
# #         stable_suporig_labels[unstable] = -1

# #         supnew_labels = h5["reassigned_superres_labels"][:]
# #         stable_supnew_labels = supnew_labels.copy()
# #         stable_supnew_labels[unstable] = -1

# #         cluster_viz_index.reassignments_viz(
# #             np.c_[orig_dst[:, 0], stable_suporig_labels],
# #             stable_supnew_labels,
# #             dsout / "sippx" / "traces_cached_seg0.raw",
# #             dsout / "reassign_exps_reasfigs" / f"{vissort.name_lo}_reassign_sup",
# #             geom,
# #             radius=200,
# #             z_extension=1.5,
# #             trough_offset=trough_offset,
# #             spike_length_samples=spike_length_samples,
# #             templates=superres["superres_templates"],
# #             proposed_pairs=extra["superres_pairs"],
# #             reassigned_scores=h5["reassigned_scores"][:],
# #             # reassigned_resids=h5["reassigned_residuals"],
# #             reas_channel_index=h5["channel_index"][:],
# #             max_channels=h5["spike_index"][:, 1],
# #         )
#     import gc; gc.collect()

# %%
1

# %% tags=[]
# with h5py.File(extract_h5, "r") as h5:
#     disp_units = np.flatnonzero(np.isin(superres["shifted_upsampled_idx_to_orig_id"], [0, 1, 2]))
#     print(f"{disp_units=} {superres['shifted_upsampled_idx_to_orig_id'][disp_units]=}")
    
#     vissort = Sorting(
#         dsout / "sippx" / "traces_cached_seg0.raw",
#         geom,
#         # stable_visst[:, 0],
#         # spike_train_utils.make_labels_contiguous(stable_visst[:, 1]),
#         storig[:, 0],
#         stable_shuporig_labels,
#         name="StableDeconv1SHUP",
#         spike_xyza=h5["localizations"][:, :4],
#         spike_maxchans=h5["spike_index"][:, 1],
#         spike_z_reg=h5["z_reg"][:],
#         spike_maxptps=h5["maxptps"][:],
#         # do_cleaned_templates=True,
#         trough_offset=trough_offset,
#         spike_length_samples=spike_length_samples,
#     )
    
#     vissort.make_unit_summaries(
#         out_folder=dsout / "summaries" / f"stabledeconv1shup_summaries_cleaned2",
#         stored_maxchans=h5["spike_index"][:, 1],
#         # stored_order=order,
#         stored_channel_index=h5["channel_index"][:],
#         stored_waveforms=h5["cleaned_waveforms"],
#         show_scatter=False,
#         relocated=True,
#         n_jobs=1,
#         units=disp_units,
#     )
#     vissort.make_unit_summaries(
#         out_folder=dsout / "summaries" / f"stabledeconv1shup_summaries_raw",
#         units=disp_units,
#     )
#     vissort.make_unit_summaries(
#         out_folder=dsout / "summaries" / f"stabledeconv1shup_summaries_cleaned",
#         stored_maxchans=h5["spike_index"][:, 1],
#         # stored_order=order,
#         stored_channel_index=h5["channel_index"][:],
#         stored_waveforms=h5["orig_cleaned_waveforms"],
#         show_scatter=False,
#         relocated=True,
#         n_jobs=1,
#         units=disp_units,
#     )
#     vissort.make_unit_summaries(
#         out_folder=dsout / "summaries" / f"stabledeconv1shup_summaries_cleanedtpca",
#         stored_maxchans=h5["spike_index"][:, 1],
#         # stored_order=order,
#         stored_channel_index=h5["channel_index"][:],
#         stored_tpca_projs=h5["cleaned_tpca_projs"],
#         stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
#         show_scatter=False,
#         n_jobs=1,
#         units=disp_units,
#     )
    
#     vissort = Sorting(
#         dsout / "sippx" / "traces_cached_seg0.raw",
#         geom,
#         # stable_visst[:, 0],
#         # stable_supnew_labels,
#         # spike_train_utils.make_labels_contiguous(stable_supnew_labels),
#         storig[:, 0],
#         stable_shupnew_labels,
#         name="StableDeconv1SHUPReas",
#         spike_xyza=h5["localizations"][:, :4],
#         spike_maxchans=h5["spike_index"][:, 1],
#         spike_z_reg=h5["z_reg"][:],
#         spike_maxptps=h5["maxptps"][:],
#         # do_cleaned_templates=True,
#         trough_offset=trough_offset,
#         spike_length_samples=spike_length_samples,
#     )
#     vissort.make_unit_summaries(
#         out_folder=dsout / "summaries" / f"stabledeconv1shupreas_summaries_raw",
#         units=disp_units,
#     )
#     vissort.make_unit_summaries(
#         out_folder=dsout / "summaries" / f"stabledeconv1shupreas_summaries_cleaned",
#         stored_maxchans=h5["spike_index"][:, 1],
#         # stored_order=order,
#         stored_channel_index=h5["channel_index"][:],
#         stored_waveforms=h5["orig_cleaned_waveforms"],
#         show_scatter=False,
#         relocated=True,
#         n_jobs=1,
#         units=disp_units,
#     )
#     vissort.make_unit_summaries(
#         out_folder=dsout / "summaries" / f"stabledeconv1shupreas_summaries_cleaned2",
#         stored_maxchans=h5["spike_index"][:, 1],
#         # stored_order=order,
#         stored_channel_index=h5["channel_index"][:],
#         stored_waveforms=h5["cleaned_waveforms"],
#         show_scatter=False,
#         relocated=True,
#         n_jobs=1,
#         units=disp_units,
#     )
#     vissort.make_unit_summaries(
#         out_folder=dsout / "summaries" / f"stabledeconv1shupreas_summaries_cleanedtpca",
#         stored_maxchans=h5["spike_index"][:, 1],
#         # stored_order=order,
#         stored_channel_index=h5["channel_index"][:],
#         stored_tpca_projs=h5["cleaned_tpca_projs"],
#         stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
#         show_scatter=False,
#         n_jobs=1,
#         units=disp_units,
#     )

# %% tags=[]
# with h5py.File(extract_h5, "r") as h5:
    
#     # vissort = Sorting(
#     #     dsout / "sippx" / "traces_cached_seg0.raw",
#     #     geom,
#     #     storig[:, 0],
#     #     stable_sto[:, 1],
#     #     name="StableDeconv1",
#     #     spike_xyza=h5["localizations"][:, :4],
#     #     spike_maxchans=h5["spike_index"][:, 1],
#     #     spike_z_reg=h5["z_reg"][:],
#     #     spike_maxptps=h5["maxptps"][:],
#     #     # do_cleaned_templates=True,
#     #     trough_offset=trough_offset,
#     #     spike_length_samples=spike_length_samples,
#     # )
#     # vissort.make_unit_summaries(
#     #     out_folder=dsout / "summaries" / f"stabledeconv1_raw",
#     # )
#     # vissort.make_unit_summaries(
#     #     out_folder=dsout / "summaries" / f"stabledeconv1_cleaned",
#     #     stored_maxchans=h5["spike_index"][:, 1],
#     #     # stored_order=order,
#     #     stored_channel_index=h5["channel_index"][:],
#     #     stored_waveforms=h5["orig_cleaned_waveforms"],
#     #     show_scatter=False,
#     #     n_jobs=1,
#     # )
#     # vissort.make_unit_summaries(
#     #     out_folder=dsout / "summaries" / f"stabledeconv1_cleanedtpca",
#     #     stored_maxchans=h5["spike_index"][:, 1],
#     #     # stored_order=order,
#     #     stored_channel_index=h5["channel_index"][:],
#     #     stored_tpca_projs=h5["cleaned_tpca_projs"],
#     #     stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
#     #     show_scatter=False,
#     #     n_jobs=1,
#     # )
    
#     vissort = Sorting(
#         dsout / "sippx" / "traces_cached_seg0.raw",
#         geom,
#         storig[:, 0],
#         stable_new_labels,
#         name="StableDeconv1Reas",
#         spike_xyza=h5["localizations"][:, :4],
#         spike_maxchans=h5["spike_index"][:, 1],
#         spike_z_reg=h5["z_reg"][:],
#         spike_maxptps=h5["maxptps"][:],
#         # do_cleaned_templates=True,
#         trough_offset=trough_offset,
#         spike_length_samples=spike_length_samples,
#     )
#     vissort.make_unit_summaries(
#         out_folder=dsout / "summaries" / f"stabledeconv1reas_raw",
#     )
#     vissort.make_unit_summaries(
#         out_folder=dsout / "summaries" / f"stabledeconv1reas_cleaned",
#         stored_maxchans=h5["spike_index"][:, 1],
#         # stored_order=order,
#         stored_channel_index=h5["channel_index"][:],
#         stored_waveforms=h5["orig_cleaned_waveforms"],
#         show_scatter=False,
#         relocated=True,
#         n_jobs=1,
#     )
#     vissort.make_unit_summaries(
#         out_folder=dsout / "summaries" / f"stabledeconv1reas_cleanedtpca",
#         stored_maxchans=h5["spike_index"][:, 1],
#         # stored_order=order,
#         stored_channel_index=h5["channel_index"][:],
#         stored_tpca_projs=h5["cleaned_tpca_projs"],
#         stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
#         show_scatter=False,
#         n_jobs=1,
#     )

# %%

# %% tags=[]
# with h5py.File(extract_h5, "r") as h5:
#     cluster_viz_index.reassignments_viz(
#         np.c_[storig[:, 0], stable_shuporig_labels],
#         stable_shupnew_labels,
#         dsout / "sippx" / "traces_cached_seg0.raw",
#         dsout / "deconv1_reassign_shup",
#         geom,
#         radius=200,
#         z_extension=1.5,
#         trough_offset=trough_offset,
#         spike_length_samples=spike_length_samples,
#         templates=superres["all_shifted_upsampled_temps"],
#         proposed_pairs=extra["shifted_upsampled_pairs"],
#         reassigned_scores=h5["reassigned_scores"][:],
#         reassigned_resids=h5["reassigned_residuals"],
#         reas_channel_index=h5["channel_index"][:],
#         max_channels=h5["spike_index"][:, 1],
#         units=disp_units,
#     )
    
#     cluster_viz_index.reassignments_viz(
#         np.c_[storig[:, 0], stable_suporig_labels],
#         stable_supnew_labels,
#         dsout / "sippx" / "traces_cached_seg0.raw",
#         dsout / "deconv1_reassign_sup",
#         geom,
#         radius=200,
#         z_extension=1.5,
#         trough_offset=trough_offset,
#         spike_length_samples=spike_length_samples,
#         templates=superres["superres_templates"],
#         proposed_pairs=extra["shifted_upsampled_pairs"],
#         reassigned_scores=h5["reassigned_scores"][:],
#         reassigned_resids=h5["reassigned_residuals"],
#         reas_channel_index=h5["channel_index"][:],
#         max_channels=h5["spike_index"][:, 1],
#         units=np.flatnonzero(np.isin(superres_label_to_orig_label, [0, 1, 2])),
#     )
    
#     cluster_viz_index.reassignments_viz(
#         np.c_[storig[:, 0], stable_sto[:, 1]],
#         stable_new_labels,
#         dsout / "sippx" / "traces_cached_seg0.raw",
#         dsout / "deconv1_reassign_unit",
#         geom,
#         radius=200,
#         z_extension=1.5,
#         trough_offset=trough_offset,
#         spike_length_samples=spike_length_samples,
#         templates=unit_raw_templates,
#         proposed_pairs=extra["unit_pairs"],
#         reassigned_scores=h5["reassigned_scores"][:],
#         reassigned_resids=h5["reassigned_residuals"],
#         reas_channel_index=h5["channel_index"][:],
#         max_channels=h5["spike_index"][:, 1],
#         units=[0, 1, 2],
#     )

# %%

# %% [markdown]
# ## Moving on

# %%
1

# %%
extract_h5 = next((dsout / "deconv1").glob("*.h5"))
print(f"{extract_h5=}")

with h5py.File(extract_h5) as h5:
    print(type(h5), isinstance(h5, h5py.File))
    orig_dst = h5["deconv_spike_train"][:]
    reas_labels = h5["reassigned_unit_labels"][:]
    dcx, dcy, dcz, dcalpha = dcxyza = h5["localizations"][:, :4].T
    dca = h5["maxptps"][:]
    outlier_scores = h5["outlier_scores"][:]
    superres_temps = h5["superres_templates"][:]
    superres_reas_labels = h5["reassigned_superres_labels"][:]
dst = np.c_[orig_dst[:, 0], reas_labels]
dstsup = np.c_[orig_dst[:, 0], superres_reas_labels]
dct = dst[:, 0] / fs

# %% tags=[]
# # (dsout / "deconv1_resid_outliers").mkdir(exist_ok=True)
# with h5py.File(extract_h5) as h5:
#     scores = h5["outlier_scores"][:]
#     ci = h5["channel_index"][:]
#     maxchans = h5["templates_up_maxchans"][:][h5["spike_train_up"][:, 1]]
#     cleaned_st = dst.copy()
#     # cleaned_st[:, 1] = -1

# #     for u in np.unique(dst[:, 1]):
# #         print(u, end=", ")
# #         in_u = dst[:, 1] == u
# #         scores_u = scores[in_u]
# #         med = np.median(scores_u)
# #         mad = np.median(np.abs(scores_u - med))
# #         thresh = med + 4 * mad / 0.6745
        
# #         inliers = scores_u <= thresh
# #         in_u_1 = np.flatnonzero(in_u)[inliers]
# #         keeps = outliers.enforce_refractory_by_score(
# #             in_u_1,
# #             dst[:, 0],
# #             scores,
# #             min_dt_frames=21,
# #         )
# #         in_u_2 = in_u_1[keeps]
        
# #         inliers = np.isin(np.flatnonzero(in_u), in_u_2)
# #         fig, ax = outliers.outlier_viz_mini(
# #             scores,
# #             inliers,
# #             thresh,
# #             dst,
# #             u,
# #             cleaned_tpca_projs=h5["cleaned_tpca_projs"],
# #             cleaned_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
# #             geom=geom,
# #             channel_index=ci,
# #             maxchans=maxchans,
# #             keep_lt=True,
# #             ms_frames=20,
# #         )
# #         fig.tight_layout()
        
# #         fig.savefig(dsout / "deconv1_resid_outliers" / f"{u:03d}.png")
# #         # plt.show()
# #         plt.close(fig)

# #         cleaned_st[in_u_2, 1] = u  

# %% tags=[]
# with h5py.File(extract_h5, "r") as h5:
#     visst = cleaned_st.copy()
#     vissort = Sorting(
#         dsout / "sippx" / "traces_cached_seg0.raw",
#         geom,
#         visst[:, 0],
#         spike_train_utils.make_labels_contiguous(visst[:, 1]),
#         name="Deconv1OutlierReas",
#         spike_xyza=h5["localizations"][:, :4],
#         spike_maxchans=h5["spike_index"][:, 1],
#         spike_z_reg=h5["z_reg"][:],
#         spike_maxptps=h5["maxptps"][:],
#         do_cleaned_templates=True,
#         trough_offset=trough_offset,
#         spike_length_samples=spike_length_samples,
#     )
#     # vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"deconv1outlierreas_summaries_raw")
#     vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"deconv1outlierreas_summaries_raw_relocated", relocated=True)
    
    
#     stable_visst = visst.copy()
#     stable_visst[~np.isin(stable_visst[:, 0] // fs, np.flatnonzero(p_good))] = -1
#     vissort = Sorting(
#         dsout / "sippx" / "traces_cached_seg0.raw",
#         geom,
#         stable_visst[:, 0],
#         spike_train_utils.make_labels_contiguous(stable_visst[:, 1]),
#         name="StableDeconv1OutlierReas",
#         spike_xyza=h5["localizations"][:, :4],
#         spike_maxchans=h5["spike_index"][:, 1],
#         spike_z_reg=h5["z_reg"][:],
#         spike_maxptps=h5["maxptps"][:],
#         do_cleaned_templates=True,
#         trough_offset=trough_offset,
#         spike_length_samples=spike_length_samples,
#     )
#     vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"stabledeconv1outlierreas_summaries_raw")
#     vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"stabledeconv1outlierreas_summaries_raw_relocated", relocated=True)
#     vissort.make_resid_studies(out_folder=dsout / "resid_studies" / f"stabledeconv1outlierreas_resid_studies")

# %% tags=[]
st = dst.copy()
good_times = np.isin(st[:, 0] // fs, np.flatnonzero(p_good))
st[~good_times, 1] = -1
st, *_ = spike_train_utils.clean_align_and_get_templates(st, geom.shape[0], dsout / "sippx" / "traces_cached_seg0.raw", min_n_spikes=5)
(dsout / "clust_post_deconv1").mkdir(exist_ok=True)
spike_train, templates, order = newms.new_merge_split(
    st,
    geom.shape[0],
    dsout / "sippx" / "traces_cached_seg0.raw",
    extract_h5,
    geom,
    dsout / "clust_post_deconv1",
    # n_workers=1,
    # merge_resid_threshold=2.5,
    merge_resid_threshold=3.0,
    relocated=True,
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    # extra_pc_split=False,
    pc_only=True,
    # threshold_diptest=1.5,
)

# %% tags=[]
for k in ("split", "merge"):
    visst = np.load(dsout / "clust_post_deconv1" / f"{k}_st.npy")
    vissort = Sorting(
        dsout / "sippx" / "traces_cached_seg0.raw",
        geom,
        visst[:, 0],
        spike_train_utils.make_labels_contiguous(visst[:, 1]),
        name="PostDeconv1StableClust" + k.capitalize(),
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
        do_cleaned_templates=True,
    )
    od = dsout / "summaries"
    vissort.make_unit_summaries(out_folder=od / f"{vissort.name_lo}_raw")
    with h5py.File(extract_h5) as h5:
        vissort.make_unit_summaries(
            out_folder=od / f"{vissort.name_lo}_cleaned",
            stored_maxchans=h5["spike_index"][:, 1],
            stored_order=order,
            stored_channel_index=h5["channel_index"][:],
            stored_tpca_projs=h5["cleaned_tpca_projs"],
            stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
            show_scatter=False,
            relocated=False,
            n_jobs=1,
        )

# %% tags=[]
# with h5py.File(extract_h5) as h5:
#     # for k in ("split", "merge"):
#     for k in ("merge",):
#         visst = np.load(dsout / "clust_post_deconv1" / f"{k}_st.npy")
#         visst_order = np.load(dsout / "clust_post_deconv1" / f"{k}_order.npy")
#         vissort = Sorting(
#             dsout / "sippx" / "traces_cached_seg0.raw",
#             geom,
#             visst[:, 0],
#             # spike_train_utils.make_labels_contiguous(visst[:, 1]),
#             visst[:, 1],
#             name="PostDeconvStable" + k.capitalize(),
#             # do_cleaned_templates=True,
#             trough_offset=trough_offset,
#             spike_length_samples=spike_length_samples,
#         )
#         vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"postdeconvstable{k}_raw")
#         # vissort.make_resid_studies(out_folder=dsout / "resid_studies" / f"postdeconvstable{k}_resid_studies")
#         vissort.make_unit_summaries(
#             out_folder=dsout / "summaries" / f"postdeconvstable{k}_cleaned",
#             stored_maxchans=h5["spike_index"][:, 1],
#             stored_order=visst_order,
#             stored_channel_index=h5["channel_index"][:],
#             stored_waveforms=h5["cleaned_waveforms"],
#             show_scatter=False,
#             n_jobs=1,
#         )
#         vissort.make_unit_summaries(
#             out_folder=dsout / "summaries" / f"postdeconvstable{k}_cleanedtpca",
#             stored_maxchans=h5["spike_index"][:, 1],
#             stored_order=visst_order,
#             stored_channel_index=h5["channel_index"][:],
#             stored_tpca_projs=h5["cleaned_tpca_projs"],
#             stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
#             show_scatter=False,
#             n_jobs=1,
#         )

# %% [markdown]
# # Deconv2

# %% tags=[]
superres2 = drifty_deconv.superres_deconv(
    dsout / "sippx" / "traces_cached_seg0.raw",
    geom,
    dcxyza.T[np.load(dsout / "clust_post_deconv1" / f"merge_order.npy"), 2],
    p,
    spike_train=np.load(dsout / "clust_post_deconv1" / f"merge_st.npy"),
    reference_displacement=p_mode,
    bin_size_um=25,
    pfs=fs,
    n_jobs=8,
    deconv_dir=dsout / "deconv2",
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
)

# %% tags=[]
extract2_h5, extra2 = drifty_deconv.extract_superres_shifted_deconv(
    superres2,
    save_cleaned_waveforms=True,
    save_cleaned_tpca_projs=True,
    save_residual=False,
    sampling_rate=fs,
    subtraction_h5=sub_h5,
    nn_denoise=False,
    geom=geom,
    n_jobs=1,
    do_reassignment=True,
    save_reassignment_residuals=True,
    tpca_weighted=True,
)

# %%
1

# %% tags=[]
overwrite = True
# overwrite = False

with h5py.File(extract2_h5, "r+") as h5:
    samples = h5["spike_index"][:, 0] #- h5["start_sample"][()]
    x, y, z_abs, alpha, _ = h5["localizations"][:].T
    maxptp = maxptps = h5["maxptps"][:]
    spike_index = h5["spike_index"][:]
    maxchans = spike_index[:, 1]
    t = spike_index[:, 0] / fs
    z = z_abs
    
    # dispmap -= dispmap.mean()
    if "z_reg" in h5 and not overwrite:
        z_reg = h5["z_reg"][:]
        pap = p = h5["p"][:]
    else:
        if overwrite and "z_reg" in h5:
            del h5["z_reg"]
            del h5["p"]
        
        # z_reg, dispmap = ibme.register_nonrigid(
        z_reg, p = ibme.register_rigid(
            maxptps,
            z_abs - z_abs.min(),
            (samples - samples.min()) / 20000,
            # robust_sigma=1,
            # corr_threshold=0.3,
            adaptive_mincorr_percentile=5,
            disp=300,
            denoise_sigma=0.1,
            # max_dt=100,
            prior_lambda=1,
            batch_size=64,
        )
        z_reg -= (z_reg - z_abs).mean()
        h5.create_dataset("z_reg", data=z_reg)
        h5.create_dataset("p", data=p)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(t, z, c=np.clip(maxptp, 0, 15), s=5, alpha=0.5, linewidths=0, marker=".")
plt.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(0, 15), plt.cm.viridis),
    label="denoised peak-to-peak amp.",
    shrink=0.5,
    pad=0.025,
    ax=ax,
)
ax.plot(geom.max() / 2 + p, color="k", lw=2, label="drift est.")
for zz in np.unique(geom[:, 1]):
    ax.axhline(zz, lw=1, color="k", alpha=0.2)
ax.legend()
tt = np.arange(0, 100 * (t.max() // 100) + 1, 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y.  {len(maxptp)} spikes.")
fig.savefig(dsout / "scatter_t_v_y_postdeconv2.png")

# %%
fig, (aa, ab) = plt.subplots(ncols=2, figsize=(8, 3))

counts, edges, _ = aa.hist(p, bins=128, color="k")

p_mode = edges[counts.argmax():counts.argmax()+2].mean()
lo = p_mode - pitch / 2
hi = p_mode + pitch / 2
p_good = (lo < p) & (hi > p)

aa.set_xlabel("est. displacement")
aa.set_ylabel("frequency")

aa.axvline(edges[counts.argmax():counts.argmax()+2].mean(), color=plt.cm.Greens(0.4), label="mode est.", lw=2, ls="--")
aa.legend()
ab.plot(t_start + np.arange(len(pap)), pap, c="gray", lw=1)
ab.scatter(t_start + np.arange(len(p))[~p_good], p[~p_good], c="k", s=1, label=f"disp too far {(~p_good).sum()}s", zorder=12)
ab.scatter(t_start + np.arange(len(p))[p_good], p[p_good], color=plt.cm.Greens(0.4), s=1, label=f"disp within pitch/2 of mode {(p_good).sum()}s", zorder=12)
ab.set_ylabel("est. displacement")
ab.set_xlabel("time (s)")
ab.legend()

fig.savefig(dsout / "stable_bins_deconv2.png", dpi=300)

# %%

# %% tags=[]
with h5py.File(extract2_h5) as h5:
    orig_dst = h5["deconv_spike_train"][:]
    reas_labels = h5["reassigned_unit_labels"][:]
dst = np.c_[orig_dst[:, 0], reas_labels]
st = dst.copy()
good_times = np.isin(st[:, 0] // fs, np.flatnonzero(p_good))
st[~good_times, 1] = -1
st, *_ = spike_train_utils.clean_align_and_get_templates(
    st, geom.shape[0], dsout / "sippx" / "traces_cached_seg0.raw", min_n_spikes=5
)
(dsout / "clust_post_deconv1").mkdir(exist_ok=True)
spike_train, templates, order = newms.new_merge_split(
    st,
    geom.shape[0],
    dsout / "sippx" / "traces_cached_seg0.raw",
    extract2_h5,
    geom,
    dsout / "clust_post_deconv2",
    # n_workers=1,
    # merge_resid_threshold=2.5,
    merge_resid_threshold=10.0,
    relocated=True,
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    # extra_pc_split=False,
    pc_only=True,
    # threshold_diptest=1.5,
)

# %% tags=[]
for k in ("split", "merge"):
    visst = np.load(dsout / "clust_post_deconv2" / f"{k}_st.npy")
    vissort = Sorting(
        dsout / "sippx" / "traces_cached_seg0.raw",
        geom,
        visst[:, 0],
        spike_train_utils.make_labels_contiguous(visst[:, 1]),
        name="PostDeconv2StableClust" + k.capitalize(),
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
        do_cleaned_templates=True,
    )
    od = dsout / "summaries"
    vissort.make_unit_summaries(out_folder=od / f"{vissort.name_lo}_raw")
    with h5py.File(extract_h5) as h5:
        vissort.make_unit_summaries(
            out_folder=od / f"{vissort.name_lo}_cleaned",
            stored_maxchans=h5["spike_index"][:, 1],
            stored_order=order,
            stored_channel_index=h5["channel_index"][:],
            stored_tpca_projs=h5["cleaned_tpca_projs"],
            stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
            show_scatter=False,
            relocated=False,
            n_jobs=1,
        )

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% tags=[]
with h5py.File(extract2_h5, "r") as h5:
    orig_dst = h5["deconv_spike_train"][:]
    reas_labels = h5["reassigned_unit_labels"][:]
    
    stable_visst = orig_dst.copy()
    unstable = ~np.isin((stable_visst[:, 0] - trough_offset) // fs, np.flatnonzero(p_good))
    stable_visst[unstable, 1] = -1
    stable_reas = reas_labels.copy()
    stable_reas[unstable] = -1
    vissort = Sorting(
        dsout / "sippx" / "traces_cached_seg0.raw",
        geom,
        stable_visst[:, 0],
        stable_visst[:, 1],
        name="StableDeconv2",
        # spike_xyza=h5["localizations"][:, :4],
        spike_maxchans=h5["spike_index"][:, 1],
        # spike_z_reg=h5["z_reg"][:],
        # spike_maxptps=h5["maxptps"][:],
        # do_cleaned_templates=True,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )
    vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"stabledeconv2_raw", n_jobs=1)
    vissort.make_unit_summaries(
        out_folder=dsout / "summaries" / f"stabledeconv2_cleaned",
        stored_maxchans=h5["spike_index"][:, 1],
        # stored_order=order,
        stored_channel_index=h5["channel_index"][:],
        stored_waveforms=h5["cleaned_waveforms"],
        show_scatter=False,
        n_jobs=1,
    )
    
    
    vissort = Sorting(
        dsout / "sippx" / "traces_cached_seg0.raw",
        geom,
        stable_visst[:, 0],
        stable_reas,
        name="StableDeconv2Reas",
        # spike_xyza=h5["localizations"][:, :4],
        spike_maxchans=h5["spike_index"][:, 1],
        # spike_z_reg=h5["z_reg"][:],
        # spike_maxptps=h5["maxptps"][:],
        # do_cleaned_templates=True,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )
    vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"stabledeconv2reas_raw", n_jobs=1)
    vissort.make_unit_summaries(
        out_folder=dsout / "summaries" / f"stabledeconv2reas_cleaned",
        stored_maxchans=h5["spike_index"][:, 1],
        # stored_order=order,
        stored_channel_index=h5["channel_index"][:],
        stored_waveforms=h5["cleaned_waveforms"],
        show_scatter=False,
        n_jobs=1,
    )
    
    visst = orig_dst.copy()
    vissort = Sorting(
        dsout / "sippx" / "traces_cached_seg0.raw",
        geom,
        visst[:, 0],
        visst[:, 1],
        name="Deconv2",
        spike_xyza=h5["localizations"][:, :4],
        spike_maxchans=h5["spike_index"][:, 1],
        spike_z_reg=h5["z_reg"][:],
        spike_maxptps=h5["maxptps"][:],
        # do_cleaned_templates=True,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )
    vissort.make_unit_summaries(
        out_folder=dsout / "summaries" / f"deconv2_cleanedtpca_relocated",
        stored_maxchans=h5["spike_index"][:, 1],
        # stored_order=order,
        stored_channel_index=h5["channel_index"][:],
        stored_tpca_projs=h5["cleaned_tpca_projs"],
        stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
        show_scatter=False,
        relocated=True,
        n_jobs=1,
    )
    # vissort.make_resid_studies(out_folder=dsout / "resid_studies" / f"stabledeconv2_resid_studies")

# %%
with h5py.File(extract2_h5, "r") as h5:
    orig_dst = h5["deconv_spike_train"][:]
    reas_labels = h5["reassigned_unit_labels"][:]
    final_st = np.c_[orig_dst[:, 0], reas_labels]

# %%
1

# %%

# %%

# %% tags=[]
# exps = {
#     "Deconv2ReasInf": (False, np.inf),
#     "Deconv2Reas2": (False, 2.0),
#     "Deconv2ReasInfPC": (True, np.inf),
#     "Deconv2Reas2PC": (True, 2.0),
# }

# for name, (do_tpca, norm_p) in exps.items():
#     print("-" * 80)
#     print(name)
    
#     extract2_h5, extra2 = drifty_deconv.extract_superres_shifted_deconv(
#         superres2,
#         save_cleaned_waveforms=True,
#         save_cleaned_tpca_projs=True,
#         save_residual=False,
#         sampling_rate=fs,
#         subtraction_h5=sub_h5,
#         nn_denoise=False,
#         geom=geom,
#         n_jobs=1 if do_tpca else 4,
#         do_reassignment=True,
#         # save_reassignment_residuals=True,
#         tpca_weighted=True,
#         do_reassignment_tpca=do_tpca,
#         reassignment_norm_p=norm_p,
#     )
    
#     with h5py.File(extract2_h5) as h5:    
#         orig_dst = h5["deconv_spike_train"][:]
#         reas_labels = h5["reassigned_unit_labels"][:]

#         unstable = ~np.isin((stable_visst[:, 0] - trough_offset) // fs, np.flatnonzero(p_good))

#         stable_reas_labels = reas_labels.copy()
#         stable_reas_labels[unstable] = -1
#         stable_visst = orig_dst.copy()
#         stable_visst[unstable, 1] = -1
#         vissort = Sorting(
#             dsout / "sippx" / "traces_cached_seg0.raw",
#             geom,
#             stable_visst[:, 0],
#             stable_visst[:, 1],
#             name=f"Stable{name}",
#             spike_maxchans=h5["spike_index"][:, 1],
#             trough_offset=trough_offset,
#             spike_length_samples=spike_length_samples,
#         )
#         vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"{vissort.name_lo}_raw")
#         vissort.make_unit_summaries(
#             out_folder=dsout / "reassign_exps_summaries" / f"{vissort.name_lo}_cleaned",
#             stored_maxchans=h5["spike_index"][:, 1],
#             # stored_order=order,
#             stored_channel_index=h5["channel_index"][:],
#             stored_waveforms=h5["cleaned_waveforms"],
#             show_scatter=False,
#             n_jobs=1,
#         )

#         spike_train_sup = h5["superres_deconv_spike_train"][:]
#         stable_suporig_labels = spike_train_sup[:, 1].copy()
#         stable_suporig_labels[unstable] = -1

#         supnew_labels = h5["reassigned_superres_labels"][:]
#         stable_supnew_labels = supnew_labels.copy()
#         stable_supnew_labels[unstable] = -1

#         cluster_viz_index.reassignments_viz(
#             np.c_[orig_dst[:, 0], stable_suporig_labels],
#             stable_supnew_labels,
#             dsout / "sippx" / "traces_cached_seg0.raw",
#             dsout / "reassign_exps_reasfigs" / f"{vissort.name_lo}_reassign_sup",
#             geom,
#             radius=200,
#             z_extension=1.5,
#             trough_offset=trough_offset,
#             spike_length_samples=spike_length_samples,
#             templates=superres2["superres_templates"],
#             proposed_pairs=extra2["superres_pairs"],
#             reassigned_scores=h5["reassigned_scores"][:],
#             # reassigned_resids=h5["reassigned_residuals"],
#             reas_channel_index=h5["channel_index"][:],
#             max_channels=h5["spike_index"][:, 1],
#         )

# %%

# %%

# %%

# %%

# %%

# %%

# %%
with h5py.File(extract2_h5) as h5:
    orig_dst = h5["deconv_spike_train"][:]
    reas_labels = h5["reassigned_unit_labels"][:]
    dcx, dcy, dcz, dcalpha = dcxyza = h5["localizations"][:, :4].T
    dca = h5["maxptps"][:]
    outlier_scores = h5["outlier_scores"][:]
    superres_temps = h5["superres_templates"][:]
    superres_reas_labels = h5["reassigned_superres_labels"][:]
dst = np.c_[orig_dst[:, 0], reas_labels]
dstsup = np.c_[orig_dst[:, 0], superres_reas_labels]
dct = dst[:, 0] / fs

# %%
vissort.was_sorted

# %% tags=[]
with h5py.File(extract2_h5, "r") as h5:
    visst = orig_dst.copy()
    vissort = Sorting(
        dsout / "sippx" / "traces_cached_seg0.raw",
        geom,
        visst[:, 0],
        spike_train_utils.make_labels_contiguous(visst[:, 1]),
        name="Deconv2",
        spike_xyza=h5["localizations"][:, :4],
        spike_maxchans=h5["spike_index"][:, 1],
        spike_z_reg=h5["z_reg"][:],
        spike_maxptps=h5["maxptps"][:],
        do_cleaned_templates=True,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )
    vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"deconv2_summaries_raw")
    vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"deconv2_summaries_raw_relocated", relocated=True)
    
    
    stable_visst = visst.copy()
    stable_visst[~np.isin(stable_visst[:, 0] // fs, np.flatnonzero(p_good))] = -1
    vissort = Sorting(
        dsout / "sippx" / "traces_cached_seg0.raw",
        geom,
        stable_visst[:, 0],
        spike_train_utils.make_labels_contiguous(stable_visst[:, 1]),
        name="StableDeconv2",
        spike_xyza=h5["localizations"][:, :4],
        spike_maxchans=h5["spike_index"][:, 1],
        spike_z_reg=h5["z_reg"][:],
        spike_maxptps=h5["maxptps"][:],
        do_cleaned_templates=True,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )
    vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"stabledeconv2_summaries_raw")
    vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"stabledeconv2_summaries_raw_relocated", relocated=True)
    vissort.make_resid_studies(out_folder=dsout / "resid_studies" / f"stabledeconv2_resid_studies")

# %% tags=[]
with h5py.File(extract2_h5, "r") as h5:
    visst = dst.copy()
    vissort = Sorting(
        dsout / "sippx" / "traces_cached_seg0.raw",
        geom,
        visst[:, 0],
        spike_train_utils.make_labels_contiguous(visst[:, 1]),
        name="Deconv2Reas",
        spike_xyza=h5["localizations"][:, :4],
        spike_maxchans=h5["spike_index"][:, 1],
        spike_z_reg=h5["z_reg"][:],
        spike_maxptps=h5["maxptps"][:],
        do_cleaned_templates=True,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )
    vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"deconv2reas_summaries_raw")
    vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"deconv2reas_summaries_raw_relocated", relocated=True)
    
    
    stable_visst = visst.copy()
    stable_visst[~np.isin(stable_visst[:, 0] // fs, np.flatnonzero(p_good))] = -1
    vissort = Sorting(
        dsout / "sippx" / "traces_cached_seg0.raw",
        geom,
        stable_visst[:, 0],
        spike_train_utils.make_labels_contiguous(stable_visst[:, 1]),
        name="StableDeconv2Reas",
        spike_xyza=h5["localizations"][:, :4],
        spike_maxchans=h5["spike_index"][:, 1],
        spike_z_reg=h5["z_reg"][:],
        spike_maxptps=h5["maxptps"][:],
        do_cleaned_templates=True,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )
    vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"stabledeconv2reas_summaries_raw")
    vissort.make_unit_summaries(out_folder=dsout / "summaries" / f"stabledeconv2reas_summaries_raw_relocated", relocated=True)
    vissort.make_resid_studies(out_folder=dsout / "resid_studies" / f"stabledeconv2reas_resid_studies")

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
# dct = deconv_spike_train[:, 0] / fs
# dc_template_z = tz[deconv_spike_train[:, 1]]
dct = superres2["superres_deconv_spike_train"][:, 0] / fs
dc_template_z = tz[superres2["superres_deconv_spike_train"][:, 1]]

fig = plt.figure(figsize=(8, 6))
plt.scatter(dct, tz[superres2["superres_deconv_spike_train"][:, 1]], c=superres2["superres_deconv_spike_train"][:, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)

tt = np.arange(0, 100 * (dct.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y. {dct.size} deconvolved spikes.")
fig.savefig(dsout / "reloc_scatter_deconv_t_v_y.png")

# %%
with h5py.File(extract_h5, "r") as h5:
    spike_train_orig = h5["deconv_spike_train"][:]
    stable_sto = spike_train_orig[np.isin(spike_train_orig[:, 0] // fs, np.flatnonzero(p_good))]
    new_labels = h5["reassigned_unit_labels"][:]
    stable_new_labels = new_labels[np.isin(spike_train_orig[:, 0] // fs, np.flatnonzero(p_good))]
    superres_templates = h5["superres_templates"][:]
    superres_label_to_orig_label = h5["superres_label_to_orig_label"][:]
    superres_label_to_bin_id = h5["superres_label_to_bin_id"][:]
    storig = h5["deconv_spike_train"][:]
    cluster_viz_index.reassignments_viz(
        # spike_train_orig,
        stable_sto,
        # new_labels,
        stable_new_labels,
        dsout / "sippx" / "traces_cached_seg0.raw",
        dsout / "reassign2",
        geom,
        radius=200,
        z_extension=1.5,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )

# %%
with h5py.File(extract_h5, "r") as h5:
    superres_templates = h5["superres_templates"][:]
    superres_templates = h5["superres_templates"][:]
    superres_label_to_orig_label = h5["superres_label_to_orig_label"][:]
    superres_label_to_bin_id = h5["superres_label_to_bin_id"][:]
    storig = h5["deconv_spike_train"][:]
    cluster_viz_index.superres_templates_viz(
        superres_templates,
        superres_label_to_orig_label,
        superres_label_to_bin_id,
        storig,
        dsout / "superres2",
        geom,
        radius=200,
    )

# %%
with h5py.File(extract_h5) as h5:
    print(type(h5), isinstance(h5, h5py.File))
    orig_dst = h5["deconv_spike_train"][:]
    reas_labels = h5["reassigned_unit_labels"][:]
    dcx, dcy, dcz, dca, _ = h5["localizations"][:].T
    dca = h5["maxptps"][:]
    outlier_scores = h5["outlier_scores"][:]
    superres_temps = h5["superres_templates"][:]
    superres_reas_labels = h5["reassigned_superres_labels"][:]
dst = np.c_[orig_dst[:, 0], reas_labels]
dstsup = np.c_[orig_dst[:, 0], superres_reas_labels]
dct = dst[:, 0] / fs

# %%
# dct = deconv_spike_train[:, 0] / fs
# dc_template_z = tz[deconv_spike_train[:, 1]]

fig = plt.figure(figsize=(8, 6))
plt.scatter(dct, tz[dstsup[:, 1]], c=dstsup[:, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)

tt = np.arange(0, 100 * (dct.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y. deconv2. {dct.size} deconvolved + reassigned spikes.")
fig.savefig(dsout / "reloc_scatter_deconv2reas_t_v_y.png")

# %%

# %%
sorting_superres2 = Sorting(
    dsout / "sippx" / "traces_cached_seg0.raw",
    geom,
    superres2["deconv_spike_train"][:, 0],
    spike_train_utils.make_labels_contiguous(superres2["deconv_spike_train"][:, 1]),
    "Superres2",
    fs=20_000,
)

# %%
sorting_superres2.make_unit_summaries(out_folder=dsout / "superres2_summaries")

# %%
sorting_superres2.make_unit_summaries(out_folder=dsout / "superres2_summaries_reloc", relocated=True)

# %% tags=[]
(dsout / "resid_outliers2").mkdir(exist_ok=True)
with h5py.File(extract_h5) as h5:
    scores = h5["outlier_scores"][:]
    ci = h5["channel_index"][:]
    maxchans = h5["templates_up_maxchans"][:][h5["spike_train_up"][:, 1]]
    cleaned_st = dst.copy()
    cleaned_st[:, 1] = -1

    for u in np.unique(dst[:, 1]):
        print(u, end=", ")
        in_u = dst[:, 1] == u
        scores_u = scores[in_u]
        med = np.median(scores_u)
        mad = np.median(np.abs(scores_u - med))
        thresh = med + 4 * mad / 0.6745
        
        inliers = scores_u <= thresh
        fig, ax = outliers.outlier_viz_mini(
            scores,
            inliers,
            thresh,
            dst,
            u,
            cleaned_waveforms=h5["cleaned_waveforms"],
            geom=geom,
            channel_index=ci,
            maxchans=maxchans,
            keep_lt=True,
            ms_frames=20,
        )
        fig.tight_layout()
        
        fig.savefig(dsout / "resid_outliers2" / f"{u:03d}.png")
        # plt.show()
        plt.close(fig)

        cleaned_st[in_u_2, 1] = u  

# %%

# %%
dst_ = dst
dst = cleaned_st
dstsup_ = dstsup
dstsup = dstsup.copy()
dstsup[cleaned_st[:, 1] < 0, 1] = -1

# %%
# dct = deconv_spike_train[:, 0] / fs
# dc_template_z = tz[deconv_spike_train[:, 1]]
# dct = superres["superres_deconv_spike_train"][:, 0] / fs
# dc_template_z = tz[superres["superres_deconv_spike_train"][:, 1]]

fig = plt.figure(figsize=(8, 6))
plt.scatter(dct, tz[dstsup[:, 1]], c=reas_labels, cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)

tt = np.arange(0, 100 * (dct.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y. {dct.size} deconvolved spikes.")
fig.savefig(dsout / "reloc_scatter_deconv_t_v_y.png")

# %%
fig = plt.figure(figsize=(8, 6))
plt.scatter(dct, dcz, c=orig_dst[:, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)

tt = np.arange(0, 100 * (dct.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y. {dct.size} deconvolved spikes.")
fig.savefig(dsout / "reloc_scatter_deconv_t_v_loc_y.png")

# %%
fig = plt.figure(figsize=(8, 6))
plt.scatter(dct, dcz, c=reas_labels, cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)

tt = np.arange(0, 100 * (dct.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y. {dct.size} deconvolved spikes, reassigned.")
fig.savefig(dsout / "reloc_scatter_deconv_reas_t_v_loc_y.png")

# %%
dczreg = dcz - p[dst[:, 0] // fs]

fig = plt.figure(figsize=(8, 6))
plt.scatter(dct, dczreg, c=orig_dst[:, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)

tt = np.arange(0, 100 * (dct.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y. {dct.size} deconvolved spikes.")
fig.savefig(dsout / "reloc_scatter_deconv_t_v_reg_loc_y.png")

# %%
dczreg = dcz - p[dst[:, 0] // fs]

fig = plt.figure(figsize=(8, 6))
plt.scatter(dct, dczreg, c=dst[:, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)

tt = np.arange(0, 100 * (dct.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y. {dct.size} deconvolved spikes, reassigned.")
fig.savefig(dsout / "reloc_scatter_deconv_reas_t_v_reg_loc_y.png")

# %%
order = np.argsort([dcz[cleaned_st[:, 1] == u].mean() for u in np.unique(cleaned_st[:, 1])])

# %%
(dsout / "relocovertime").mkdir(exist_ok=True)

# %% tags=[]
colors = cc.m_glasbey(np.unique(cleaned_st[:, 1]))
for zord, u in enumerate(order):
    lo = max(u - 1, 0)
    hi = min(lo + 3, len(order))
    
    fig, (aa, ab, ac) = plt.subplots(nrows=3, figsize=(8, 6), sharex=True, gridspec_kw=dict(hspace=0.05))
    meanzs = []
    for uu in range(lo, hi):
        uu = order[uu]
        in_uu = np.flatnonzero(superres["deconv_spike_train"][:, 1] == uu)
        meanzs.append(dcz[in_uu].mean())
        aa.scatter(dct[in_uu], dca[in_uu], color=colors[uu], s=1, lw=0, label=uu)
        ab.scatter(dct[in_uu], dcz[in_uu], color=colors[uu], s=1, lw=0)
        ac.scatter(dct[in_uu], dczreg[in_uu], color=colors[uu], s=1, lw=0)
    ab.plot(np.arange(len(pap)), pap + np.mean(meanzs), color="k", lw=1)
    aa.legend(markerscale=2.5, loc="upper left")
        
    ac.set_xlabel("time (s)")
    aa.set_title(f"z-neighbor units {tuple([order[uu] for uu in range(lo, hi)])}")
    aa.set_ylabel(f"cleaned+TPCA denoised PTP")
    ab.set_ylabel(f"z")
    ac.set_ylabel(f"reg z")
    # plt.show()
    fig.savefig(dsout / "relocovertime" / f"zorder{zord:03d}_unit{order[u]:03d}.png", dpi=300)
    plt.close(fig)

# %%
deconv_merge_st[:, 1].max() + 1

# %%
fig = plt.figure(figsize=(8, 6))
plt.scatter(deconv_merge_st[:, 0] / fs, dcz[order], c=deconv_merge_st[:, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y. {(deconv_merge_st[:, 1] >= 0).sum()} deconvolved + reassigned + cleaned + post-hoc merged spikes.")
fig.savefig(dsout / "reloc_scatter_deconvmerge_t_v_loc_y.png")

# %%
fig = plt.figure(figsize=(8, 6))
plt.scatter(deconv_merge_st[:, 0] / fs, dczreg[order], c=deconv_merge_st[:, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. reg y. {(deconv_merge_st[:, 1] >= 0).sum()} deconvolved + reassigned + cleaned + post-hoc merged spikes")
fig.savefig(dsout / "reloc_scatter_deconvmerge_t_v_reg_loc_y.png")

# %%
fig, (aa, ab) = plt.subplots(nrows=2, figsize=(8, 10), sharex=True, gridspec_kw=dict(hspace=0))

big = dca[order] > 8

aa.scatter(deconv_merge_st[big, 0] / fs, dczreg[order][big], c=deconv_merge_st[big, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
tt = np.arange(0, 100 * (t.max() // 100) , 100)
aa.set_ylabel("reg depth (um)")
aa.text(0.05, 0.95, "amp > 8", transform=aa.transAxes)

ab.scatter(deconv_merge_st[~big, 0] / fs, dczreg[order][~big], c=deconv_merge_st[~big, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
tt = np.arange(0, 100 * (t.max() // 100) , 100)
ab.set_xticks(tt, t_start + tt)
ab.text(0.05, 0.95, "amp < 8", transform=ab.transAxes)
ab.set_xlabel("time (s)")
aa.set_ylabel("reg depth (um)")

aa.set_title(f"{humanX}: time vs. reg y. {(deconv_merge_st[:, 1] >= 0).sum()} deconvolved + reassigned + cleaned + post-hoc merged spikes")
fig.savefig(dsout / "reloc_scatter_deconvmerge_t_v_reg_loc_y_double.png")

# %%

# %%

# %%
plt.scatter(normsq, templates.ptp(1).max(1))
plt.loglog()

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Phy export

# %%
# times = spike_index_c[:, 0]
# labels = clusterer.labels_
# times = deconv_spike_train[:, 0]
# labels = deconv_spike_train[:, 1]
# times, labels = deconv_merge_st.T
times, labels = final_st.T

times = times[labels >= 0]
# x_k, z_k, maxptps_k = x_c[labels >= 0], z_c[labels >= 0], maxptps_c[labels >= 0]
labels = labels[labels >= 0]

# %%
# %rm -rf {dsout / "sorting"}

# %%
sorting = si.NumpySorting.from_times_labels(times, labels, sampling_frequency=fs)
sorting = sorting.save(folder=dsout / "sorting")
sorting

# %%
binrec = si.read_binary_folder(dsout / "sippx")
binrec.annotate(is_filtered=True)
binrec

# %%
# %rm -rf {dsout / "wfs"}

# %%
we = si.extract_waveforms(binrec, sorting, dsout / "wfs")

# %%
# %rm -rf {dsout / "phy"}

# %%
si.export_to_phy(we, dsout / "phy", n_jobs=8, chunk_size=fs)

# %%
# %ll /Users/charlie/data/beeStingerOutput/human6/phy/

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

# %% [markdown]
# # LFP stuff

# %%
lfprec = si.read_neuroscope_recording(dsroot / "amplifier.xml")
lfprec.set_dummy_probe_from_locations(geom, shape_params=dict(radius=10))
probe = lfprec.get_probe()
probe.annotate(name="beeStinger")
probe.set_contacts(recorded_geom)
# probe.set_device_channel_indices(cm0i)
probe.set_contact_ids([str(i) for i in range(128)])
probe.create_auto_shape()
lfprec.set_probe(probe, in_place=True)
lfprec

# %%
lfprec = lfprec.frame_slice(t_start * fs, t_end * fs if t_end else None)
# rec_filtered = si.bandpass_filter(rec, freq_min=0.5, freq_max=6000., dtype='float32')
# rec = si.common_reference(rec_filtered, reference='global', operator='median')

# %%
depthsort = np.argsort(recorded_geom[alive_chans, 1])
lfprec = lfprec.channel_slice([str(c) for c in alive_chans[depthsort]])
# rec = si.common_reference(rec, reference='global', operator='median')
# rec = si.zscore(rec)
lfprec

# %%
lfpbinrec = lfprec.save(folder=dsout / "lfpraw", n_jobs=8, chunk_size=fs)

# %%
lfprec = si.read_binary_folder(dsout / "lfpraw")

# %%
geom = lfprec.get_channel_locations()
geom.shape

# %%
binfile = lfprec.get_binary_description()["file_paths"][0]
binfile

# %%
(dsout / "lfpppx").mkdir(exist_ok=True)
rmss = ap_filter.run_preprocessing(binfile, dsout / "lfpppx" / "csd.bin", geom=geom, n_channels=geom.shape[0], bp=(0.5, 250), csd=False, avg_depth=True, fs=fs, resample_to=250)

# %%
v = np.zeros(5)

# %%
v

# %%
vk = v + 0

# %%
v - vk

# %%
float(False)

# %%
all(v - vk) < 1e-5

# %%
np.unique(geom[:, 1]).size

# %%
csd = np.memmap(dsout / "lfpppx" / "csd.bin", dtype=np.float32).reshape(-1, np.unique(geom[:, 1]).size)

# %%
csd.shape

# %%
a = 450 * 250
im = plt.imshow((csd[a:a+10000] - np.median(csd[a:a+10000], axis=1, keepdims=True)).T, aspect="auto")
plt.colorbar(im)
plt.xticks([0, 9999], [a / 250, (a + 10000) / 250])
plt.xlabel("time (s)")
plt.title("LFP")

# %%
a=6000
fig, (aa, ab) = plt.subplots(nrows=2, figsize=(10, 5), sharex=True)
aa.set_title(f"Raw LFP starting at t={a/250}s")
im = aa.imshow(csd[a:a+1000].T, aspect="auto")
plt.colorbar(im, ax=aa)
ab.set_title(f"Common referenced LFP starting at t={a/250}s")
im = ab.imshow((csd[a:a+1000] - np.median(csd[a:a+1000], axis=1, keepdims=True)).T, aspect="auto")
plt.colorbar(im, ax=ab)
ab.set_xticks([0, 999], [a / 250, (a + 1000) / 250])
ab.set_xlabel("time (s)")

# %%
plf = p.copy()

# %%
p = lfpreg.online_register_rigid(csd.T, disp=50, prior_lambda=1, adaptive_mincorr_percentile=0.1)

# %%
with h5py.File(next((dsout / "sub").glob("sub*h5")), "r+") as h5:
    pap = h5["p"][:]
