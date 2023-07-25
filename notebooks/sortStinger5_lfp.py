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
import shutil

# %%
from dredge.ibme_corr import online_register_rigid

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
    before_deconv_merge_split,
)

# %%
from spike_psvae.hybrid_analysis import Sorting

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
humanX = "human5"

# %%
t_start = 850
t_end = None

# %%
fs = 20_000
nc = 128

# %%
trough_offset = 30
spike_length_samples = 82
deconv_thresh = 64
merge_thresh_early = 3.0
merge_thresh_end = 7.0
tpca_weighted = False
do_reg = "precomputed"
do_reloc = False 

# %%
dsroot = Path("/Users/charlie/data/beeStinger") / humanX

# %%
bsout = Path("/Users/charlie/data/beeStinger64")
dsout = bsout / humanX
dsout.mkdir(exist_ok=True, parents=True)


# %%
(dsout / "probe_and_raw_data").mkdir(exist_ok=True)
(dsout / "scatterplots").mkdir(exist_ok=True)
(dsout / "drift").mkdir(exist_ok=True)

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

# %% [markdown] tags=[]
# ## SpikeInterface preprocessing

# %%
recorded_geom = np.array([0, geom[:, 1].max()]) - geom[icm0i]

# %%
pitch = waveform_utils.get_pitch(recorded_geom)
pitch

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
rec = rec.frame_slice(t_start * fs, t_end * fs if t_end else None)

# %%
rec_orig_sliced = rec

# %%
rec_show = si.bandpass_filter(rec, freq_min=300., freq_max=6000., dtype='float32')
rec_show = si.common_reference(rec_show, reference='global', operator='median')

# %%
plt.figure(figsize=(3, 2))

noise_levels = si.get_noise_levels(rec)
# noise_levels = si.get_noise_levels(rec_show)
plt.hist(noise_levels, bins=32);
# dead_level = 6e-6
dead_level = 0.00010
plt.axvline(dead_level, color="k");
plt.savefig(dsout / "probe_and_raw_data" / "noise_levels.png")

# %% tags=[]
fig, ax = plt.subplots(figsize=(4, 25))
si.plot_probe_map(rec, with_contact_id=True, with_channel_ids=True, ax=ax)
plt.title("beeStinger Probe Map")
plt.savefig(dsout / "probe_and_raw_data" / "probemap.png")

# %%
dead_chans = np.flatnonzero(noise_levels < dead_level)

# %%

# %%

# %%

# %%

# %%
depthsort = np.argsort(recorded_geom[:, 1])
rec_show = rec_show.channel_slice([str(c) for c in depthsort])

# %% tags=[]
for s in range(0, int(rec.get_total_duration()), 100):
    fig = plt.figure()
    chunk = rec_show.get_traces(start_frame=s*fs, end_frame=s*fs+1000)
    plt.imshow(chunk.T, aspect="auto", interpolation="none")
    plt.colorbar(shrink=0.3)
    for ix in np.argsort(depthsort)[dead_chans]:
        y = ix
        plt.axhline(y, ls=":", lw=1, color="orange")
        # ab.axhline(icm0i[ix], ls=":", lw=1, color="orange")
    plt.title(f"Filtered/CMR raw recording with dead channels marked, start time {s}s")
    plt.xlabel("time (samples)")
    plt.ylabel("channel")
    plt.show()
    fig.savefig(dsout / "probe_and_raw_data" / f"raw_rec_{s}.png")
    plt.close("all")


# %%
alive_chans = np.setdiff1d(np.arange(nc), dead_chans)
depthsort = np.argsort(recorded_geom[alive_chans, 1])
rec = rec.channel_slice([str(c) for c in alive_chans[depthsort]])

# %%
# rec = si.phase_shift(rec)
rec = si.bandpass_filter(rec, dtype='float32')
rec = si.common_reference(rec, reference='global', operator='median')
rec = si.zscore(rec)

# %% tags=[]
for s in range(0, int(rec.get_total_duration()), 100):
    fig = plt.figure()
    chunk = rec.get_traces(start_frame=s*fs, end_frame=s*fs+1000)
    plt.imshow(chunk.T, aspect="auto", interpolation="none")
    plt.colorbar(shrink=0.3)
    plt.title(f"Filtered/CMR raw recording with dead channels marked, start time {s}s")
    plt.xlabel("time (samples)")
    plt.ylabel("channel")
    plt.show()
    fig.savefig(dsout / "probe_and_raw_data" / f"alive_chans_filt_rec_{s}.png")
    plt.close("all")

# %%
if (dsout / "sippx").exists(): shutil.rmtree(dsout / "sippx")
rec = rec.save(folder=dsout / "sippx", n_jobs=8, chunk_size=fs)

# %%
lfs = 125

# %%
rec_orig_sliced.get_sampling_frequency()

# %%
lf = si.bandpass_filter(rec_orig_sliced, freq_min=0.5, freq_max=250.0)
lf = si.resample(lf, lfs)
lf = lf.channel_slice(lf.get_channel_ids()[alive_chans])
lf2 = si.depth_order(lf)
# lf2 = lf2.channel_slice(lf2.get_channel_ids()[])
lf2 = si.directional_derivative(lf2, order=1)
lf2 = si.average_across_direction(lf2)

# %%
# chunk0 = lf2.get_traces(start_frame=0, end_frame=5000)
chunk0 = lf2.get_traces(start_frame=15000, end_frame=20000)

# %%
lf2.channel_ids

# %%
ints = np.random.randint(0, 10, len(lf2.channel_ids))

# %%
ints

# %%
np.array(["-".join(lf2.channel_ids[ints == i]) for i in range(10)])

# %%
p0, D, C = ibme.register_raster_rigid(
    chunk0.T,
    mincorr=0.7,
    prior_lambda=1,
    batch_size=256,
    disp=20,
)

# %%
np.percentile(np.diagonal(C, 1), 1)

# %%
plt.imshow(chunk0.T, aspect="auto",)
# plt.plot(p0 + chunk0.shape[1] / 2)

# %%
p = online_register_rigid(lf2, batch_length=5000, mincorr=0.7, disp=20, batch_size=128, prior_lambda=1)

# %%
po = p

# %%
po.shape

# %%
po.size * 125

# %%
rec_orig_sliced.get_num_samples()

# %% tags=[]
for ts in np.arange(0, lf2.get_num_samples(), 40 * lf2.get_sampling_frequency()):
    plt.figure(figsize=(4,4))
    plt.imshow(lf2.get_traces(start_frame=ts, end_frame=ts + 5 * lfs).T, aspect="auto")
    plt.plot(p[ts:ts+5*lfs] - p[ts:ts+5*lfs].mean() + 50, color="w")
    plt.show()
    plt.close("all")

# %%
from scipy import signal, interpolate

# %%
rec

# %%
# pof = signal.savgol_filter(po, 121, 1)
pof = po.copy()
good_times = (
    ((np.arange(len(pof)) > 5000) & (pof >= 15))
    | (np.arange(len(pof)) <= 5000)
)
good_times = (
    good_times
    & (
        ((np.arange(len(pof)) > 25000) & (pof > 65) & (pof < 71))
        | (np.arange(len(pof)) <= 25000)
    )
)
good_times = np.flatnonzero(good_times)
pof = interpolate.interp1d(good_times, pof[good_times], fill_value="extrapolate")(np.arange(len(pof)))
# = np.nan
# pof = np.array([np.nanmedian(pof[s:s+250]) for s in range(len(pof) - 125)])
pof2 = signal.medfilt(pof, 251)


# %%
def median_changepoint(pp):
    mad = np.inf
    changepoint = -1
    best = None
    for i in trange(1, len(pp) - 1):
        med0 = np.median(pp[:i])
        med1 = np.median(pp[i:])
        est = np.where(np.arange(len(pp)) < i, med0, med1)
        # madi = np.median(np.abs(pp - est))
        madi = np.mean(np.square(pp - est))
        if madi < mad:
            mad = madi
            changepoint = i
            best = est
    return changepoint, best


# %%
medcp, meds_est = median_changepoint(pof2[230 * lfs:])

# %%
# plt.plot(pof2[np.arange(len(pof)) > 280 * lfs])
# plt.plot(meds_est)
# plt.plot(means_est)

# %%
pof3 = np.concatenate((pof2[:230 * lfs], meds_est))

# %%
po.shape, pof3.shape

# %%
plt.plot(po)
plt.plot(pof)
plt.plot(pof2)
plt.plot(pof3)
plt.axhline(15, color="k")
plt.axhline(65, color="k")
plt.axhline(71, color="k")

# %%
p_precomputed = signal.resample(np.pad(pof3, 50 * lfs, mode="reflect"), 101 + int(np.ceil(t.max())))
p_precomputed = p_precomputed[50:-50] * 25

# %%
# p_precomputed = signal.resample(pof3, int(np.ceil(t.max()))) * 25
# p_precomputed = p_precomputed[50:-50] * 25

# %% tags=[]
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(t, z, c=np.clip(maxptp, 0, 15), s=5, alpha=0.5, linewidths=0, marker=".")
plt.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(0, 15), plt.cm.viridis),
    label="denoised peak-to-peak amp.",
    shrink=0.5,
    pad=0.025,
    ax=ax,
)
ax.scatter(np.arange(len(po))/lfs, 300 + geom.max() / 2 + (po - np.median(po)) * 25, s=1, color="k", lw=0, label="drift est.")
ax.scatter(np.arange(len(pof3))/lfs, 300 + geom.max() / 2 + (pof3 - np.median(po)) * 25, s=1, color="r", lw=0, label="hand-filtered drift est.")
ax.plot(300 + geom.max() / 2 + (p_precomputed - 25 * np.median(po)), color="b", lw=1, label="downsampled to 1Hz")
for zz in np.unique(geom[:, 1]):
    ax.axhline(zz, lw=1, color="k", alpha=0.2)
ax.legend(markerscale=3, loc="upper left")
tt = np.arange(0, 100 * (t.max() // 100) + 1, 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y.  {len(maxptp)} spikes.")
fig.savefig(dsout / "scatterplots" / "initial_driftest_t_v_y.png")
# plt.close(fig)

# %% [markdown]
# # Detection / featurization

# %% tags=[]
sub_h5 = subtract.subtraction(
    rec,
    dsout / "sub",
    save_residual=False,
    n_sec_pca=80,
    peak_sign="both",
    enforce_decrease_kind="radial",
    neighborhood_kind="circle",
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

    samples = h5["spike_index"][:, 0]
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
        if do_reg == "precomputed":
            print('hi')
            p = p_precomputed.copy()
            z_reg = ibme.warp_rigid(z_abs, t, np.arange(len(p)), p)
            z_reg -= (z_reg - z_abs).mean()
        elif do_reg:
            # z_reg, dispmap = ibme.register_nonrigid(
            z_reg, p = ibme.register_rigid(
                maxptps,
                z_abs - z_abs.min(),
                (samples - samples.min()) / fs,
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
        else:
            z_reg = z_abs
            p = np.zeros(np.ceil(t.max()).astype(int))
        h5.create_dataset("z_reg", data=z_reg)
        h5.create_dataset("p", data=p)
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
fig.savefig(dsout / "scatterplots" / "initial_detection_t_v_channel.png")
plt.close(fig)

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
fig.savefig(dsout / "scatterplots" / "initial_detection_t_v_y.png")
plt.close(fig)

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
fig.savefig(dsout / "scatterplots" / "initial_detection_t_v_regy.png")
plt.close(fig)

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
fig.savefig(dsout / "scatterplots" / "initial_detection_x_v_y.png")
plt.close(fig)

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
fig.savefig(dsout / "scatterplots" / "initial_detection_x_v_regy.png")
plt.close(fig)

# %%
fig, (aa, ab) = plt.subplots(ncols=2, figsize=(8, 3))

counts, edges, _ = aa.hist(p, bins=128, color="k")
aa.set_xlabel("est. displacement")
aa.set_ylabel("frequency")

ab.plot(t_start + np.arange(len(p)), p, c="gray", lw=1)
ab.scatter(t_start + np.arange(len(p)), p, c="k", s=1, zorder=12)
ab.set_ylabel("est. displacement")
ab.set_xlabel("time (s)")
fig.suptitle(f"{humanX}, start time {t_start}", y=0.95, fontsize=10)

fig.savefig(dsout / "drift" / f"{humanX}_initial_detection_disp_tstart{t_start}.png", dpi=300)
np.savetxt(dsout / "drift" / f"{humanX}_initial_detection_disp_tstart{t_start}.csv", p, delimiter=",")
plt.close(fig)

fig, (aa, ab) = plt.subplots(ncols=2, figsize=(8, 3))
counts, edges, _ = aa.hist(p, bins=128, color="k")
p_mode = edges[counts.argmax():counts.argmax()+2].mean()
lo = p_mode - pitch
hi = p_mode + pitch
p_good = (lo < p) & (hi > p)
aa.set_xlabel("est. displacement")
aa.set_ylabel("frequency")
aa.axvline(edges[counts.argmax():counts.argmax()+2].mean(), color=plt.cm.Greens(0.4), label="mode est.", lw=2, ls="--")
aa.legend()
ab.plot(t_start + np.arange(len(p)), p, c="gray", lw=1)
ab.scatter(t_start + np.arange(len(p))[~p_good], p[~p_good], c="k", s=1, label=f"disp too far {(~p_good).sum()}s", zorder=12)
ab.scatter(t_start + np.arange(len(p))[p_good], p[p_good], color=plt.cm.Greens(0.4), s=1, label=f"disp within pitch of mode {(p_good).sum()}s", zorder=12)
ab.set_ylabel("est. displacement")
ab.set_xlabel("time (s)")
ab.legend()

fig.savefig(dsout / "drift" / f"{humanX}_initial_detection_stable_bins.png", dpi=300)

# %% [markdown]
# # Initial clustering

# %%
sub_h5 = next((dsout / "sub").glob("sub*h5"))
(dsout / "clust").mkdir(exist_ok=True)
geom = np.load(dsout / "sippx" / "properties" / "location.npy")

# %%
with h5py.File(sub_h5) as h5:
    spike_index = h5["spike_index"][:]

# %%
# hdbscan_kwargs=dict(
#     min_cluster_size=15,
#     # min_samples=5,
#     cluster_selection_epsilon=20.0,
# ),

# %%
st = spike_index.copy()
# st[:, 1] = newms.registered_maxchan(st, p, geom, pfs=fs)
good_times = np.isin((st[:, 0]) // fs, np.flatnonzero(p_good))
print(f"{good_times.sum()=}")
st[~good_times, 1] = -1
spike_train, templates, order = newms.new_merge_split(
    st,
    geom.shape[0],
    dsout / "sippx" / "traces_cached_seg0.raw",
    sub_h5,
    geom,
    dsout / "clust",
    n_workers=1,
    merge_resid_threshold=merge_thresh_early,
    relocated=do_reloc,
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    split_kwargs=dict(
        split_steps=(
            before_deconv_merge_split.herding_split,
        ),
        recursive_steps=(True,),
        split_step_kwargs=(
            dict(
                hdbscan_kwargs=dict(
                    min_cluster_size=15,
                    # min_samples=5,
                    cluster_selection_epsilon=20.0,
                ),
            ),
        ),
    )
)

# %%
(spike_train[:, 1] >= 0).sum(), (spike_train[good_times, 1]>=0).mean()

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
    od = dsout / "summaries"
    vissort.make_unit_summaries(out_folder=od / f"{vissort.name_lo}_raw")
    # with h5py.File(sub_h5) as h5:
    #     vissort.make_unit_summaries(
    #         out_folder=od / f"{vissort.name_lo}_cleaned",
    #         stored_maxchans=h5["spike_index"][:, 1],
    #         stored_order=order,
    #         stored_channel_index=h5["channel_index"][:],
    #         stored_tpca_projs=h5["cleaned_tpca_projs"],
    #         stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
    #         show_scatter=False,
    #         relocated=False,
    #         n_jobs=1,
    #     )

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
fig.savefig(dsout / "scatterplots" / "initclust_scatter.png")
plt.close(fig)

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
axes[1].set_title(f"Spatial view (zoom) of clustered and triaged spikes. {nunits} units.")
fig.savefig(dsout / "scatterplots" / "initclust_scatter_detail.png")

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
fig.savefig(dsout / "scatterplots" / "initclust_scatter_sorted_t_v_regy.png", dpi=300)

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
fig.savefig(dsout / "scatterplots" / "initclust_scatter_sorted_t_v_y.png")

# %% [markdown]
# ## Deconv 1

# %%
spike_train, order, templates, template_shifts = spike_train_utils.clean_align_and_get_templates(
    np.load(dsout / "clust" / "merge_st.npy"),
    geom.shape[0],
    dsout / "sippx" / "traces_cached_seg0.raw",
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    max_shift=0,
    min_n_spikes=5,
)

# %%
assert (order == np.arange(len(order))).all()

# %%
if (dsout / "deconv1").exists():
    shutil.rmtree(dsout / "deconv1")

# %% tags=[]
merge_order = np.load(dsout / "clust" / "merge_order.npy")
with h5py.File(sub_h5) as h5:
    z = h5["localizations"][:, 2][merge_order]
    p = h5["p"][:]
superres = drifty_deconv.superres_deconv(
    dsout / "sippx" / "traces_cached_seg0.raw",
    geom,
    z,
    p[:-1],
    spike_train=spike_train,
    reference_displacement=p_mode,
    bin_size_um=pitch / 2,
    pfs=fs,
    n_jobs=8,
    deconv_dir=dsout / "deconv1",
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    threshold=deconv_thresh,
)

# %% tags=[]
extract_deconv1_h5, extract_deconv1_extra = drifty_deconv.extract_superres_shifted_deconv(
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
    do_reassignment=False,
    n_sec_train_feats=80,
    tpca_weighted=tpca_weighted,
)

# %%
overwrite = True
# overwrite = False
rereg = False

with h5py.File(extract_deconv1_h5, "r+" if overwrite else "r") as h5:
    if "z_reg" in h5:
        print("already done, skip")

    samples = h5["spike_index"][:, 0] #- h5["start_sample"][()]
    x, y, z_abs, alpha, _ = h5["localizations"][:].T
    maxptp = maxptps = h5["maxptps"][:]
    spike_index = h5["spike_index"][:]
    maxchans = spike_index[:, 1]
    t = spike_index[:, 0] / fs
    z = z_abs
    
    
    if "z_reg" in h5 and not overwrite:
        z_reg = h5["z_reg"][:]
        pap = p = h5["p"][:]
    else:
        if rereg:
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
        else:
            # *_, tt = ibme.fast_raster(
            #     maxptp, z_abs - z_abs.min(), t
            # )
            z_reg = ibme.warp_rigid(z_abs, t, np.arange(len(p)), p)
        z_reg -= np.median(z_reg - z_abs)
        h5.create_dataset("z_reg", data=z_reg)
        h5.create_dataset("p", data=p)

# %%
order = slice(None)
spike_train = superres["deconv_spike_train"]

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
fig.savefig(dsout / "scatterplots" / "deconv1_scatter.png")
plt.close(fig)

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
axes[1].set_title(f"Spatial view (zoom) of clustered and triaged spikes. {nunits} units.")
fig.savefig(dsout / "scatterplots" / "deconv1_scatter_detail.png")

kept = spike_train[:, 1] >= 0
triaged = spike_train[:, 1] < 0

fig = plt.figure(figsize=(8, 6))
plt.scatter(t[order][triaged], z_reg[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[order][kept], z_reg[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
plt.legend(markerscale=2.5)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. registered y.  {kept.sum()} sorted spikes.")
fig.savefig(dsout / "scatterplots" / "deconv1_scatter_sorted_t_v_regy.png", dpi=300)

fig = plt.figure(figsize=(8, 6))
plt.scatter(t[order][triaged], z[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[order][kept], z[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
plt.legend(markerscale=2.5)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y.  {kept.sum()} sorted spikes.")
fig.savefig(dsout / "scatterplots" / "deconv1_scatter_sorted_t_v_y.png")

# %%
# fig, (aa, ab) = plt.subplots(ncols=2, figsize=(8, 3))
# counts, edges, _ = aa.hist(p, bins=128, color="k")
# p_mode = edges[counts.argmax():counts.argmax()+2].mean()
# lo = p_mode - pitch / 2
# hi = p_mode + pitch / 2
# p_good = (lo < p) & (hi > p)
# aa.set_xlabel("est. displacement")
# aa.set_ylabel("frequency")
# aa.axvline(edges[counts.argmax():counts.argmax()+2].mean(), color=plt.cm.Greens(0.4), label="mode est.", lw=2, ls="--")
# aa.legend()
# ab.plot(t_start + np.arange(len(p)), p, c="gray", lw=1)
# ab.scatter(t_start + np.arange(len(p))[~p_good], p[~p_good], c="k", s=1, label=f"disp too far {(~p_good).sum()}s", zorder=12)
# ab.scatter(t_start + np.arange(len(p))[p_good], p[p_good], color=plt.cm.Greens(0.4), s=1, label=f"disp within pitch/2 of mode {(p_good).sum()}s", zorder=12)
# ab.set_ylabel("est. displacement")
# ab.set_xlabel("time (s)")
# ab.legend()

# fig.savefig(dsout / "drift" / f"{humanX}_deconv1_stable_bins.png", dpi=300)

# %% tags=[]
st = spike_train.copy()
# st[:, 1] = newms.registered_maxchan(st, p, geom, pfs=fs)
good_times = np.isin((st[:, 0]) // fs, np.flatnonzero(p_good))
print(f"{good_times.sum()=}")
st[~good_times, 1] = -1
spike_train, templates, order = newms.new_merge_split(
    st,
    geom.shape[0],
    dsout / "sippx" / "traces_cached_seg0.raw",
    extract_deconv1_h5,
    geom,
    dsout / "deconv1clust",
    n_workers=1,
    merge_resid_threshold=merge_thresh_early,
    relocated=do_reloc,
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    split_kwargs=dict(
        split_steps=(
            before_deconv_merge_split.herding_split,
        ),
        recursive_steps=(True,),
        split_step_kwargs=(
            dict(
                hdbscan_kwargs=dict(
                    min_cluster_size=15,
                    # min_samples=5,
                    cluster_selection_epsilon=20.0,
                ),
            ),
        ),
    )
)

# %%
(spike_train[:, 1] >= 0).sum(), (spike_train[good_times, 1]>=0).mean()

# %%
# from spike_psvae.hybrid_analysis import Sorting
# vissort = Sorting(raw_binary, geom, *spike_train.T, name="SortingName")
# vissort.make_unit_summaries(
#     out_folder="/out/folder",
#     stored_maxchans=h5["spike_index"][:, 1],
#     stored_order=order,
#     stored_channel_index=h5["channel_index"][:],
#     stored_tpca_projs=h5["cleaned_tpca_projs"],
#     stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
#     show_scatter=False,
#     relocated=False,
#     n_jobs=1,
# )

# %% tags=[]
for k in ("split", "merge"):
    visst = np.load(dsout / "deconv1clust" / f"{k}_st.npy")
    a
    od = dsout / "summaries"
    vissort.make_unit_summaries(out_folder=od / f"{vissort.name_lo}_raw")
    # with h5py.File(extract_deconv1_h5) as h5:
    #     vissort.make_unit_summaries(
    #         out_folder=od / f"{vissort.name_lo}_cleaned",
    #         stored_maxchans=h5["spike_index"][:, 1],
    #         stored_order=order,
    #         stored_channel_index=h5["channel_index"][:],
    #         stored_tpca_projs=h5["cleaned_tpca_projs"],
    #         stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
    #         show_scatter=False,
    #         relocated=False,
    #         n_jobs=1,
    #     )

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
fig.savefig(dsout / "scatterplots" / "deconv1clust_scatter.png")
plt.close(fig)

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
axes[1].set_title(f"Spatial view (zoom) of clustered and triaged spikes. {nunits} units.")
fig.savefig(dsout / "scatterplots" / "deconv1clust_scatter_detail.png")

kept = spike_train[:, 1] >= 0
triaged = spike_train[:, 1] < 0

fig = plt.figure(figsize=(8, 6))
plt.scatter(t[order][triaged], z_reg[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[order][kept], z_reg[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
plt.legend(markerscale=2.5)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. registered y.  {kept.sum()} sorted spikes.")
fig.savefig(dsout / "scatterplots" / "deconv1clust_scatter_sorted_t_v_regy.png", dpi=300)

fig = plt.figure(figsize=(8, 6))
plt.scatter(t[order][triaged], z[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[order][kept], z[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
plt.legend(markerscale=2.5)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y.  {kept.sum()} sorted spikes.")
fig.savefig(dsout / "scatterplots" / "deconv1clust_scatter_sorted_t_v_y.png")

# %%

# %% [markdown]
# ## Deconv 2

# %%
spike_train, order, templates, template_shifts = spike_train_utils.clean_align_and_get_templates(
    np.load(dsout / "deconv1clust" / "merge_st.npy"),
    geom.shape[0],
    dsout / "sippx" / "traces_cached_seg0.raw",
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    max_shift=0,
    min_n_spikes=5,
)

# %%
assert (order == np.arange(len(order))).all()

# %%
if (dsout / "deconv2").exists():
    shutil.rmtree(dsout / "deconv2")

# %% tags=[]
merge_order = np.load(dsout / "deconv1clust" / "merge_order.npy")
with h5py.File(extract_deconv1_h5) as h5:
    z = h5["localizations"][:, 2][merge_order]
    p = h5["p"][:]
superres2 = drifty_deconv.superres_deconv(
    dsout / "sippx" / "traces_cached_seg0.raw",
    geom,
    z,
    p[:-1],
    spike_train=spike_train,
    reference_displacement=p_mode,
    bin_size_um=pitch / 2,
    pfs=fs,
    n_jobs=8,
    deconv_dir=dsout / "deconv2",
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    threshold=deconv_thresh,
)

# %% tags=[]
extract_deconv2_h5, extract_deconv2_extra = drifty_deconv.extract_superres_shifted_deconv(
    superres2,
    save_cleaned_waveforms=True,
    save_cleaned_tpca_projs=True,
    save_residual=False,
    sampling_rate=fs,
    subtraction_h5=extract_deconv1_h5,
    nn_denoise=False,
    geom=geom,
    n_jobs=1,
    # save_reassignment_residuals=True,
    # pairs_method="radius",
    max_resid_dist=20,
    do_reassignment_tpca=True,
    do_reassignment=False,
    n_sec_train_feats=80,
    tpca_weighted=tpca_weighted,
)

# %%
overwrite = True
# overwrite = False
rereg = False

with h5py.File(extract_deconv2_h5, "r+" if overwrite else "r") as h5:
    if "z_reg" in h5:
        print("already done, skip")

    samples = h5["spike_index"][:, 0] #- h5["start_sample"][()]
    x, y, z_abs, alpha, _ = h5["localizations"][:].T
    maxptp = maxptps = h5["maxptps"][:]
    spike_index = h5["spike_index"][:]
    maxchans = spike_index[:, 1]
    t = spike_index[:, 0] / fs
    z = z_abs
    
    
    if "z_reg" in h5 and not overwrite:
        z_reg = h5["z_reg"][:]
        pap = p = h5["p"][:]
    else:
        if rereg:
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
        else:
            z_reg = ibme.warp_rigid(z_abs, t, np.arange(len(p)), p)
        z_reg -= np.median(z_reg - z_abs)
        h5.create_dataset("z_reg", data=z_reg)
        h5.create_dataset("p", data=p)

# %%
order = slice(None)
spike_train = superres2["deconv_spike_train"]

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
fig.savefig(dsout / "scatterplots" / "deconv2_scatter.png")
plt.close(fig)

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
axes[1].set_title(f"Spatial view (zoom) of clustered and triaged spikes. {nunits} units.")
fig.savefig(dsout / "scatterplots" / "deconv2_scatter_detail.png")

kept = spike_train[:, 1] >= 0
triaged = spike_train[:, 1] < 0

fig = plt.figure(figsize=(8, 6))
plt.scatter(t[order][triaged], z_reg[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[order][kept], z_reg[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
plt.legend(markerscale=2.5)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. registered y.  {kept.sum()} sorted spikes.")
fig.savefig(dsout / "scatterplots" / "deconv2_scatter_sorted_t_v_regy.png", dpi=300)

fig = plt.figure(figsize=(8, 6))
plt.scatter(t[order][triaged], z[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[order][kept], z[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
plt.legend(markerscale=2.5)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y.  {kept.sum()} sorted spikes.")
fig.savefig(dsout / "scatterplots" / "deconv2_scatter_sorted_t_v_y.png")

# %%
# fig, (aa, ab) = plt.subplots(ncols=2, figsize=(8, 3))
# counts, edges, _ = aa.hist(p, bins=128, color="k")
# p_mode = edges[counts.argmax():counts.argmax()+2].mean()
# lo = p_mode - pitch / 2
# hi = p_mode + pitch / 2
# p_good = (lo < p) & (hi > p)
# aa.set_xlabel("est. displacement")
# aa.set_ylabel("frequency")
# aa.axvline(edges[counts.argmax():counts.argmax()+2].mean(), color=plt.cm.Greens(0.4), label="mode est.", lw=2, ls="--")
# aa.legend()
# ab.plot(t_start + np.arange(len(p)), p, c="gray", lw=1)
# ab.scatter(t_start + np.arange(len(p))[~p_good], p[~p_good], c="k", s=1, label=f"disp too far {(~p_good).sum()}s", zorder=12)
# ab.scatter(t_start + np.arange(len(p))[p_good], p[p_good], color=plt.cm.Greens(0.4), s=1, label=f"disp within pitch/2 of mode {(p_good).sum()}s", zorder=12)
# ab.set_ylabel("est. displacement")
# ab.set_xlabel("time (s)")
# ab.legend()

# fig.savefig(dsout / "drift" / f"{humanX}_deconv2_stable_bins.png", dpi=300)

# %% tags=[]
st = spike_train.copy()
# st[:, 1] = newms.registered_maxchan(st, p, geom, pfs=fs)
good_times = np.isin((st[:, 0]) // fs, np.flatnonzero(p_good))
print(f"{good_times.sum()=}")
st[~good_times, 1] = -1
spike_train, templates, order = newms.new_merge_split(
    st,
    geom.shape[0],
    dsout / "sippx" / "traces_cached_seg0.raw",
    extract_deconv2_h5,
    geom,
    dsout / "deconv2clust",
    n_workers=1,
    merge_resid_threshold=merge_thresh_end,
    relocated=do_reloc,
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    split_kwargs=dict(
        split_steps=(
            before_deconv_merge_split.herding_split,
        ),
        recursive_steps=(True,),
        split_step_kwargs=(
            dict(
                hdbscan_kwargs=dict(
                    min_cluster_size=25,
                    # min_samples=5,
                    cluster_selection_epsilon=20.0,
                ),
            ),
        ),
    )
)

# %%
(spike_train[:, 1] >= 0).sum(), (spike_train[good_times, 1]>=0).mean()

# %% tags=[]
for k in ("split", "merge"):
    visst = np.load(dsout / "deconv2clust" / f"{k}_st.npy")
    vissort = Sorting(
        dsout / "sippx" / "traces_cached_seg0.raw",
        geom,
        visst[:, 0],
        spike_train_utils.make_labels_contiguous(visst[:, 1]),
        name="Deconv2Clust" + k.capitalize(),
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )
    od = dsout / "summaries"
    vissort.make_unit_summaries(out_folder=od / f"{vissort.name_lo}_raw")
    # with h5py.File(extract_deconv2_h5) as h5:
    #     vissort.make_unit_summaries(
    #         out_folder=od / f"{vissort.name_lo}_cleaned",
    #         stored_maxchans=h5["spike_index"][:, 1],
    #         stored_order=order,
    #         stored_channel_index=h5["channel_index"][:],
    #         stored_tpca_projs=h5["cleaned_tpca_projs"],
    #         stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
    #         show_scatter=False,
    #         relocated=False,
    #         n_jobs=1,
    #     )

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
fig.savefig(dsout / "scatterplots" / "deconv2clust_scatter.png")
plt.close(fig)

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
axes[1].set_title(f"Spatial view (zoom) of clustered and triaged spikes. {nunits} units.")
fig.savefig(dsout / "scatterplots" / "deconv2clust_scatter_detail.png")

kept = spike_train[:, 1] >= 0
triaged = spike_train[:, 1] < 0

fig = plt.figure(figsize=(8, 6))
plt.scatter(t[order][triaged], z_reg[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[order][kept], z_reg[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
plt.legend(markerscale=2.5)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. registered y.  {kept.sum()} sorted spikes.")
fig.savefig(dsout / "scatterplots" / "deconv2clust_scatter_sorted_t_v_regy.png", dpi=300)

fig = plt.figure(figsize=(8, 6))
plt.scatter(t[order][triaged], z[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[order][kept], z[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
plt.legend(markerscale=2.5)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y.  {kept.sum()} sorted spikes.")
fig.savefig(dsout / "scatterplots" / "deconv2clust_scatter_sorted_t_v_y.png")

# %% [markdown]
# ## Deconv 3

# %%
spike_train, order, templates, template_shifts = spike_train_utils.clean_align_and_get_templates(
    np.load(dsout / "deconv2clust" / "merge_st.npy"),
    geom.shape[0],
    dsout / "sippx" / "traces_cached_seg0.raw",
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    max_shift=0,
    min_n_spikes=5,
)

# %%
assert (order == np.arange(len(order))).all()

# %%
if (dsout / "deconv3").exists():
    shutil.rmtree(dsout / "deconv3")

# %% tags=[]
merge_order = np.load(dsout / "deconv2clust" / "merge_order.npy")
with h5py.File(extract_deconv2_h5) as h5:
    z = h5["localizations"][:, 2][merge_order]
    p = h5["p"][:]
superres3 = drifty_deconv.superres_deconv(
    dsout / "sippx" / "traces_cached_seg0.raw",
    geom,
    z,
    p[:-1],
    spike_train=spike_train,
    reference_displacement=p_mode,
    bin_size_um=pitch / 2,
    pfs=fs,
    n_jobs=8,
    deconv_dir=dsout / "deconv3",
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    threshold=deconv_thresh,
)

# %% tags=[]
extract_deconv3_h5, extract_deconv3_extra = drifty_deconv.extract_superres_shifted_deconv(
    superres3,
    save_cleaned_waveforms=True,
    save_cleaned_tpca_projs=True,
    save_residual=False,
    sampling_rate=fs,
    subtraction_h5=extract_deconv2_h5,
    nn_denoise=False,
    geom=geom,
    n_jobs=1,
    # save_reassignment_residuals=True,
    # pairs_method="radius",
    max_resid_dist=20,
    do_reassignment_tpca=True,
    do_reassignment=False,
    n_sec_train_feats=80,
    tpca_weighted=tpca_weighted,
)

# %%
overwrite = True
# overwrite = False
rereg = False

with h5py.File(extract_deconv3_h5, "r+" if overwrite else "r") as h5:
    if "z_reg" in h5:
        print("already done, skip")

    samples = h5["spike_index"][:, 0] #- h5["start_sample"][()]
    x, y, z_abs, alpha, _ = h5["localizations"][:].T
    maxptp = maxptps = h5["maxptps"][:]
    spike_index = h5["spike_index"][:]
    maxchans = spike_index[:, 1]
    t = spike_index[:, 0] / fs
    z = z_abs
    
    
    if "z_reg" in h5 and not overwrite:
        z_reg = h5["z_reg"][:]
        pap = p = h5["p"][:]
    else:
        if rereg:
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
        else:
            z_reg = ibme.warp_rigid(z_abs, t, np.arange(len(p)), p)
        z_reg -= np.median(z_reg - z_abs)
        h5.create_dataset("z_reg", data=z_reg)
        h5.create_dataset("p", data=p)

# %%
order = slice(None)
spike_train = superres3["deconv_spike_train"]

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
fig.savefig(dsout / "scatterplots" / "deconv3_scatter.png")
plt.close(fig)

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
axes[1].set_title(f"Spatial view (zoom) of clustered and triaged spikes. {nunits} units.")
fig.savefig(dsout / "scatterplots" / "deconv3_scatter_detail.png")

kept = spike_train[:, 1] >= 0
triaged = spike_train[:, 1] < 0

fig = plt.figure(figsize=(8, 6))
plt.scatter(t[order][triaged], z_reg[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[order][kept], z_reg[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
plt.legend(markerscale=2.5)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. registered y.  {kept.sum()} sorted spikes.")
fig.savefig(dsout / "scatterplots" / "deconv3_scatter_sorted_t_v_regy.png", dpi=300)

fig = plt.figure(figsize=(8, 6))
plt.scatter(t[order][triaged], z[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[order][kept], z[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
plt.legend(markerscale=2.5)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"{humanX}: time vs. y.  {kept.sum()} sorted spikes.")
fig.savefig(dsout / "scatterplots" / "deconv3_scatter_sorted_t_v_y.png")

# %% tags=[]
visst = spike_train.copy()
# visst[:, 1] = newms.registered_maxchan(visst, p, geom, pfs=fs)
good_times = np.isin((visst[:, 0]) // fs, np.flatnonzero(p_good))
visst[~good_times, 1] = -1
vissort = Sorting(
    dsout / "sippx" / "traces_cached_seg0.raw",
    geom,
    visst[:, 0],
    spike_train_utils.make_labels_contiguous(visst[:, 1]),
    name="Deconv3Stable",
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
)
od = dsout / "summaries"
vissort.make_unit_summaries(out_folder=od / f"{vissort.name_lo}_raw")
# with h5py.File(extract_deconv3_h5) as h5:
#     vissort.make_unit_summaries(
#         out_folder=od / f"{vissort.name_lo}_cleaned",
#         stored_maxchans=h5["spike_index"][:, 1],
#         stored_order=order,
#         stored_channel_index=h5["channel_index"][:],
#         stored_tpca_projs=h5["cleaned_tpca_projs"],
#         stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
#         show_scatter=False,
#         relocated=False,
#         n_jobs=1,
#     )

# %% [markdown] tags=[]
# ## Phy export

# %%
import spikeinterface.core as sc

# %%
times, labels = spike_train.T
times = times[labels >= 0]
labels = labels[labels >= 0]

# %%
if (dsout / "sisorting").exists():
    (dsout / "sisorting").unlink()
sorting = si.NumpySorting.from_times_labels(times, labels, sampling_frequency=fs)
sorting = sorting.save(folder=dsout / "sisorting")
sorting

# %%
binrec = si.read_binary_folder(dsout / "sippx")
binrec.annotate(is_filtered=True)
binrec

# %%
binrec.get_traces(start_frame=0, end_frame=1000).min()

# %%
binrec.get_traces(start_frame=0, end_frame=1000).max()

# %%
rec_orig_sliced.get_traces(start_frame=0, end_frame=1000).min()

# %%
-31287 * .195

# %%
rec_orig_sliced.get_traces(0, 1000).max()

# %%
if (dsout / "siwfs").exists():
    (dsout / "siwfs").unlink()

# %%
we = si.extract_waveforms(binrec, sorting, dsout / "siwfs")

# %%
metrics_df = si.compute_quality_metrics(we, skip_pc_metrics=True)
metrics_df.to_csv(dsout / f"{humanX}_deconv3_qualitymetrics.csv")

# %%
if (dsout / "phy").exists():
    (dsout / "phy").unlink()

# %%
si.export_to_phy(we, dsout / "phy", n_jobs=8, chunk_size=fs)

# %%
# !ls {dsout / "phy"}

# %%
rec_phy = si.read_p(dsout / "phy")

# %%
rec_phy

# %%
rec_phy.get_tra
