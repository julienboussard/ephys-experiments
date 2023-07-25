# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
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
from sklearn.cluster import SpectralClustering, AffinityPropagation

# %%
from spike_psvae import subtract, cluster_utils, cluster_viz_index, ibme, ibme_corr, newms

# %%
from reglib import ap_filter, lfpreg

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

# %% [markdown]
# ## Paths / config

# %%
humanX = "human1"

# %%
dsroot = Path("/Users/charlie/data/beeStinger") / humanX

# %%
bsout = Path("/Users/charlie/data/beeStingerDeconv")
dsout = bsout / humanX
dsout.mkdir(exist_ok=True, parents=True)


# %%
fs = 20_000
nc = 128

# %%
chanmap_mat = dsroot.parent / "chanMap.mat"
h = loadmat(chanmap_mat)
cm0i = h["chanMap0ind"].squeeze().astype(int)
geom = np.c_[h['xcoords'].squeeze(), h['ycoords'].squeeze()]
geom.shape

# %%
icm0i = np.zeros_like(cm0i)
for i, ix in enumerate(cm0i):
    icm0i[ix] = i

# %%
t_start = 75
t_end = None

# %% [markdown] tags=[]
# ## SpikeInterface preprocessing

# %%
recorded_geom = np.array([0, geom[:, 1].max()]) -  geom[icm0i]

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
rec_orig = rec
rec

# %% tags=[]
fig, ax = plt.subplots(figsize=(4, 25))
si.plot_probe_map(rec, with_contact_id=True, with_channel_ids=True, ax=ax)
plt.title("beeStinger Probe Map")
plt.savefig(dsout / "probemap.png")

# %%
noise_levels = si.get_noise_levels(rec)

# %%
plt.figure(figsize=(3, 2))
plt.hist(noise_levels, bins=32);
dead_level = 33.4e-6
plt.axvline(dead_level, color="k");

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
rec = rec_orig.channel_slice([str(c) for c in alive_chans[depthsort]])
rec = si.bandpass_filter(rec, freq_min=300., freq_max=6000., dtype='float32')
rec = rec.frame_slice(t_start * fs, t_end * fs if t_end else None)
rec = si.common_reference(rec, reference='global', operator='median')
rec = si.zscore(rec)

# %%
rec

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
for s in range(0, int(binrec.get_total_duration()), 50):
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
# !rm -rf {dsout / "lfpraw"}

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
rmss = ap_filter.run_preprocessing(binfile, dsout / "lfpppx" / "csd.bin", geom=geom, n_channels=geom.shape[0], bp=(0.5, 250), csd=True, avg_depth=False, fs=fs, resample_to=250)

# %%
csd = np.memmap(dsout / "lfpppx" / "csd.bin", dtype=np.float32).reshape(-1, np.unique(geom[:, 1]).size)

# %%
csd.shape

# %%
a = 24 * 250
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
plf = lfpreg.online_register_rigid(csd[:, :55].T, disp=25, prior_lambda=1, adaptive_mincorr_percentile=0.1)

# %%
plt.plot(plf[20000:30000], lw=1, color="k")

# %% [markdown]
# # Detection / featurization

# %%
geom = np.load(dsout / "sippx" / "properties" / "location.npy")

# %%
subtract.subtraction(
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
)

# %%
# !h5ls /Users/charlie/data/beeStingerOutput/human6/sub/subtraction_traces_cached_seg0_t_0_None.h5

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

# %%
plt.plot(*geom[:10].T, marker="s", markerfacecolor="orange")
plt.xlabel("probe electrode x")
plt.ylabel("probe electrode z")
plt.title("First 10 channels of beeStinger")

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
# ax.plot(geom.max() / 2 + p, color="k", lw=2, label="drift est.")
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
fig, (aa, ab) = plt.subplots(ncols=2, figsize=(8, 3))

counts, edges, _ = aa.hist(p, bins=128, color="k")
aa.set_xlabel("est. displacement")
aa.set_ylabel("frequency")

aa.axvline(edges[counts.argmax():counts.argmax()+2].mean(), color=plt.cm.Greens(0.4), label="mode est.", lw=2, ls="--")
aa.legend()

p_mode = 0#edges[counts.argmax():counts.argmax()+2].mean()
lo = p_mode - 25 / 4
hi = p_mode + 25 / 4
p_good = (lo < p) & (hi > p)

ab.plot(t_start + np.arange(len(pap)), pap, c="gray", lw=1)
ab.scatter(t_start + np.arange(len(p))[~p_good], p[~p_good], c="k", s=1, label=f"disp too far {(~p_good).sum()}s", zorder=12)
ab.scatter(t_start + np.arange(len(p))[p_good], p[p_good], color=plt.cm.Greens(0.4), s=1, label=f"disp within pitch/2 of mode {(p_good).sum()}s", zorder=12)
ab.set_ylabel("est. displacement")
ab.set_xlabel("time (s)")
ab.legend()

fig.savefig(dsout / "stable_bins.png", dpi=300)

# %%
# clusterer, cluster_centers, spike_index_c, x_c, z_c, maxptps_c, original_spike_ids = cluster_utils.cluster_spikes(
#     x[:],
#     z_reg[:],
#     maxptp[:],
#     spike_index[:],
#     split_big=True,
#     do_remove_dups=False,
# )

# %%
sub_h5 = next((dsout / "sub").glob("sub*h5"))

# %%
(dsout / "clust").mkdir(exist_ok=True)

# %%
geom = np.load(dsout / "sippx" / "properties" / "location.npy")
geom.shape

# %%
import numba

# %%
from spike_psvae.relocation import restrict_wfs_to_chans

# %% tags=[]
st = spike_index.copy()
st[:, 1] = newms.registered_maxchan(st, p, geom, pfs=fs)
good_times = np.isin(st[:, 0] // fs, np.flatnonzero(p_good))
st[~good_times, 1] = -1
spike_train, templates, order = newms.new_merge_split(
    st,
    geom.shape[0],
    dsout / "sippx" / "traces_cached_seg0.raw",
    sub_h5,
    geom,
    dsout / "clust",
    n_workers=0,
    # merge_resid_threshold=2.5,
    merge_resid_threshold=3.0,
    relocated=True,
)

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw=dict(wspace=0.1))
cluster_viz_index.array_scatter(
    spike_train[:, 1],
    geom,
    x,
    z_reg,
    maxptp,
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
axes[1].set_title("Spatial view of clustered and triaged spikes")
fig.savefig(dsout / "reloc_cluster_scatter.png")

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw=dict(wspace=0.1))
cluster_viz_index.array_scatter(
    spike_train[:, 1],
    geom,
    x,
    z_reg,
    maxptp,
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
axes[1].set_title("Spatial view of clustered and triaged spikes")
fig.savefig(dsout / "reloc_cluster_scatter_detail.png")

# %%
t = spike_index[:, 0] / fs

# %%
kept = spike_train[:, 1] >= 0
triaged = spike_train[:, 1] < 0

fig = plt.figure(figsize=(8, 6))
plt.scatter(t[triaged & ~good_times], z_reg[triaged & ~good_times], color="k", s=5, alpha=0.5, marker=".", linewidths=0, label="outside stable")
plt.scatter(t[triaged & good_times], z_reg[triaged & good_times], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[kept], z_reg[kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
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
plt.scatter(t[triaged & ~good_times], z[triaged & ~good_times], color="k", s=5, alpha=0.5, marker=".", linewidths=0, label="outside stable")
plt.scatter(t[triaged & good_times], z[triaged & good_times], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[kept], z[kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
plt.legend(markerscale=2.5)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y.  {kept.sum()} sorted spikes.")
fig.savefig(dsout / "reloc_scatter_sorted_t_v_y.png")

# %%
spike_train[:, 1].max()

# %% [markdown]
# ## Deconv

# %%
from spike_psvae import drifty_deconv, spike_train_utils, snr_templates

# %%
spike_train_al, order, templates, template_shifts = spike_train_utils.clean_align_and_get_templates(
    spike_train,
    geom.shape[0],
    dsout / "sippx" / "traces_cached_seg0.raw",
    # trough_offset=31,
    # spike_length_samples=111,
)

# %%
# templates, extra = snr_templates.get_templates(
#     spike_train_al,
#     geom,
#     dsout / "sippx" / "traces_cached_seg0.raw",
#     templates.ptp(1).argmax(1),
#     n_jobs=8,
#     trough_offset=31,
#     spike_length_samples=111,
# )

# %%
# extra["tpca"]

# %%
(dsout / "deconv").mkdir(exist_ok=True)

# %%
from spike_psvae import deconvolve, localize_index, extract_deconv, drifty_deconv

# %%
z_abs.shape, spike_train.shape

# %%
# !rm -rf {dsout / "deconv"}/*

# %%
1000 / 20000

# %% tags=[]
superres = drifty_deconv.superres_deconv(
    dsout / "sippx" / "traces_cached_seg0.raw",
    geom,
    z_abs,
    p,
    spike_train=spike_train_al,
    # templates=templates,
    reference_displacement=p_mode,
    bin_size_um=25,
    pfs=fs,
    n_jobs=8,
    deconv_dir=dsout / "deconv",
    # max_upsample=4,
    refractory_period_frames=15,
)

# %%
# deconv_spike_train, deconv_spike_train_up, deconv_scalings, all_up_temps = drifty_deconv.rigid_int_shift_deconv(
#     dsout / "sippx" / "traces_cached_seg0.raw",
#     geom,
#     spike_train,
#     p,
#     templates=templates,
#     reference_displacement=p_mode,
#     pfs=fs,
#     n_jobs=8,
#     deconv_dir=dsout / "deconv",
#     max_upsample=4,
# )

# %%
from spike_psvae import waveform_utils

# %%
st_up = superres["superres_deconv_spike_train_shifted_upsampled"]
temps_up = superres["all_shifted_upsampled_temps"]

max_channels = temps_up.ptp(1).argmax(1)[st_up[:, 1]]
tpca = waveform_utils.fit_tpca_bin(
    np.c_[st_up[:, 0], max_channels],
    geom,
    dsout / "sippx" / "traces_cached_seg0.raw",
    tpca_rank=5,
    tpca_n_wfs=50000,
    spike_length_samples=121,
    spatial_radius=75,
    seed=0,
)

# %% tags=[]
# np.save(dsout / "deconv" / "all_up_temps.npy", all_up_temps)
# np.save(dsout / "deconv" / "deconv_spike_train_up.npy", deconv_spike_train_up)
# np.save(dsout / "deconv" / "all_up_temps.npy", superres["all_up_temps"])
# np.save(dsout / "deconv" / "deconv_spike_train_up.npy", superres["deconv_spike_train_up"])
# ci = subtract.make_channel_index(geom, 100)
# extract_h5 = extract_deconv.extract_deconv(
#     dsout / "deconv" / "all_up_temps.npy",
#     dsout / "deconv" / "deconv_spike_train_up.npy",
#     dsout / "deconv",
#     dsout / "sippx" / "traces_cached_seg0.raw",
#     channel_index=ci,
#     save_cleaned_waveforms=True,
#     save_residual=False,
#     sampling_rate=fs,
#     # trough_offset=31,
#     n_jobs=8,
#     subtraction_h5=sub_h5,
#     nn_denoise=False,
#     # tpca=extra["tpca"],
#     tpca=tpca,
# )

extract_h5 = drifty_deconv.extract_superres_shifted_deconv(
    superres,
    save_cleaned_waveforms=True,
    save_residual=False,
    sampling_rate=fs,
    # trough_offset=31,
    subtraction_h5=sub_h5,
    nn_denoise=False,
    # tpca=extra["tpca"],
    tpca=tpca,
    geom=geom,
    n_jobs=8,
)

# %%
with h5py.File(extract_h5) as h5:
    orig_dst = h5["deconv_spike_train"][:]
    reas_labels = h5["reassigned_unit_labels"][:]
    dcx, dcy, _, dcz = h5["localizations"][:, :4].T
    dca = h5["maxptps"][:]
    outlier_scores = h5["outlier_scores"][:]
    superres_temps = h5["superres_templates"][:]
    superres_reas_labels = h5["reassigned_superres_labels"][:]
dst = np.c_[orig_dst[:, 0], reas_labels]
dstsup = np.c_[orig_dst[:, 0], superres_reas_labels]
dct = dst[:, 0] / fs

# %%
outlier_scores.min(), outlier_scores.max()

# %%
tx, ty, _, tz, ta = localize_index.localize_ptps_index(
    superres_temps.ptp(1), geom, superres_temps.ptp(1).argmax(1),
    np.array([np.arange(geom.shape[0])] * geom.shape[0]),
    radius=100,
)

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
from spike_psvae import outliers

# %%
(dsout / "resid_outliers").mkdir(exist_ok=True)
with h5py.File(extract_h5) as h5:
    scores = h5["outlier_scores"][:]
    ci = h5["channel_index"][:]
    maxchans = h5["templates_up_maxchans"][:][h5["spike_train_up"][:, 1]]
    cleaned_st = dst.copy()
    cleaned_st[:, 1] = -1

    for u in np.unique(dst[:, 1]):
        print(u)
        in_u = dst[:, 1] == u
        scores_u = scores[in_u]
        med = np.median(scores_u)
        mad = np.median(np.abs(scores_u - med))
        thresh = med + 4 * mad / 0.6745
        
        inliers = scores_u <= thresh
        in_u_1 = np.flatnonzero(in_u)[inliers]
        keeps = outliers.enforce_refractory_by_score(
            in_u_1,
            dst[:, 0],
            scores,
            min_dt_frames=21,
        )
        in_u_2 = in_u_1[keeps]
        
        # fig = outliers.outlier_viz(
        #     score, inl, outl,
        #     c_ratios,
        #     p_values,
        #     dst,
        #     dca,
        #     # outlier_scores,
        #     u,
        #     ms_frames=20,
        #     metric_name="max resid amp",
        #     cleaned_waveforms=h5["cleaned_waveforms"],
        #     maxchans=maxchans,
        #     residual_path=None,
        #     geom=geom,
        #     channel_index=channel_index,
        # )
        inliers = np.isin(np.flatnonzero(in_u), in_u_2)
        
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
        
        fig.savefig(dsout / "resid_outliers" / f"{u:03d}.png")
        # plt.show()
        plt.close(fig)

        cleaned_st[in_u_2, 1] = u  

# %%
order = np.argsort([dcz[cleaned_st[:, 1] == u].mean() for u in np.unique(cleaned_st[:, 1])])

# %%
(dsout / "relocovertime").mkdir(exist_ok=True)

# %%
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
from spike_psvae import deconv_resid_merge

# %%
# post-hoc merge
(
    deconv_st_aligned,
    order,
    templates,
    _,
) = spike_train_utils.clean_align_and_get_templates(
    cleaned_st,
    geom.shape[0],
    dsout / "sippx" / "traces_cached_seg0.raw",
    max_shift=0,
)

# kept = aligned_spike_train3[:, 1] >= 0
times_updated, labels_updated = deconv_resid_merge.run_deconv_merge(
    deconv_st_aligned,
    geom,
    dsout / "sippx" / "traces_cached_seg0.raw",
    templates.ptp(1).argmax(1),
    merge_resid_threshold=2.5,
)
deconv_merge_st = np.c_[times_updated, labels_updated]

(
    deconv_merge_st,
    reorder,
    templates,
    _,
) = spike_train_utils.clean_align_and_get_templates(
    deconv_merge_st,
    geom.shape[0],
    dsout / "sippx" / "traces_cached_seg0.raw",
    max_shift=0,
)
order = order[reorder]

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

# %% [markdown]
# ## Phy export

# %%
# times = spike_index_c[:, 0]
# labels = clusterer.labels_
# times = deconv_spike_train[:, 0]
# labels = deconv_spike_train[:, 1]
times, labels = deconv_merge_st.T

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

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
