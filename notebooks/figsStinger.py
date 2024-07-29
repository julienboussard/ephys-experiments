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

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
import spikeinterface.full as si
import pandas as pd
from spike_psvae import subtract

# %%
from tqdm.auto import tqdm

# %%
from spike_psvae import cluster_viz_index

# %%
import colorcet as cc

# %%
plt.rc("savefig", dpi=300)
plt.rc("figure", figsize=(4, 4))
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
# plt.rcParams.update({
#     "text.usetex": True,
#     # "font.family": "sans-serif",
# })
# preamble = r"""
# \usepackage{textgreek}
# %\renewcommand{\familydefault}{\sfdefault}
# \usepackage{helvet}
# \usepackage{sansmathfonts}
# """
# plt.rc('text.latex', preamble=preamble)

# %%
# %config InlineBackend.figure_format = 'retina'

# %%
plt.plot([0, 1], [0, 1])

# %%
origdir = Path("/Users/charlie/data/beeStinger")
datadir = Path("/Users/charlie/data/beeStingerFullSub")
base = Path("/Users/charlie/data/beeStinger64/")
figdir = Path("/Users/charlie/data/beeStinger64Figs/")

# %%
figdir.mkdir(exist_ok=True)

# %%
trough_offset = 30
spike_length_samples = 82

# %%
for humanX_dir in base.iterdir():
    if not (humanX_dir / "phy").exists():
        continue  

    humanX = humanX_dir.stem
    print(humanX)

# %% tags=[]
chanmap_mat = origdir / "chanMap.mat"
h = loadmat(chanmap_mat)
cm0i = h["chanMap0ind"].squeeze().astype(int)
icm0i = np.zeros_like(cm0i)
for i, ix in enumerate(cm0i):
    icm0i[ix] = i
geom = np.c_[h['xcoords'].squeeze(), h['ycoords'].squeeze()]
recorded_geom = np.array([0, geom[:, 1].max()]) - geom[icm0i]


for humanX_dir in base.iterdir():
    if not (humanX_dir / "phy").exists():
        continue  

    humanX = humanX_dir.stem
    if humanX == "human4":
        continue
    humanX_stem = humanX.split("_")[0]
    print(humanX)
    
    rec_ppx = si.read_binary_folder(humanX_dir / "sippx")
    t_start_s = rec_ppx.get_times()[0]
    t_start_samples = rec_ppx.get_times()[0] * rec_ppx.get_sampling_frequency()
    
    drift_est = pd.read_csv(next((humanX_dir / "drift").glob("*.csv")))
    
    # get original recording
    rec = si.read_neuroscope_recording(Path("/Users/charlie/data/beeStinger") / humanX_stem / "amplifier.xml")
    rec.set_dummy_probe_from_locations(geom, shape_params=dict(radius=10))
    probe = rec.get_probe()
    probe.annotate(name="beeStinger")
    probe.set_contacts(recorded_geom)
    # probe.set_device_channel_indices(cm0i)
    probe.set_contact_ids([str(i) for i in range(128)])
    probe.create_auto_shape()
    rec.set_probe(probe, in_place=True)
    rec = rec.channel_slice(rec_ppx.get_channel_ids())
    
    # determine the scale on the sorted region
    rec_sliced = rec.frame_slice(t_start_samples, None)
    assert rec_sliced.get_num_frames() == rec_ppx.get_num_frames()
    rec_sliced = si.bandpass_filter(rec_sliced, dtype='float32')
    rec_sliced = si.common_reference(rec_sliced, reference='global', operator='median')
    rec_zscore = si.zscore(rec_sliced)
    offset = rec_zscore._recording_segments[0].offset
    gain = rec_zscore._recording_segments[0].gain

    # apply this to the whole recording
    rec = si.bandpass_filter(rec, dtype='float32')
    rec = si.common_reference(rec, reference='global', operator='median')
    rec = si.scale(rec, gain=gain, offset=offset)
    
    # ok now we can do our thing
    subdir = datadir / humanX
    subdir.mkdir(exist_ok=True, parents=True)
    print(subdir / "subtraction.h5", (subdir / "subtraction.h5").exists())
    if not (subdir / "subtraction.h5").exists():
        sub_h5 = subtract.subtraction(
            rec,
            subdir,
            # save_waveforms=True,
            save_residual=False,
            n_sec_pca=80,
            pca_t_start=t_start_s,
            peak_sign="both",
            enforce_decrease_kind="radial",
            neighborhood_kind="circle",
            do_nn_denoise=False,
            thresholds=[5],
            n_jobs=8,
            overwrite=True,
            localize_radius=100,
            save_subtracted_tpca_projs=False,
            save_cleaned_tpca_projs=False,
            save_denoised_tpca_projs=False,
            trough_offset=trough_offset,
            spike_length_samples=spike_length_samples,
        )
        print(sub_h5)
        print(subdir / "subtraction.h5")

# %%
for humanX_dir in base.iterdir():
    if not (humanX_dir / "phy").exists():
        continue  

    humanX = humanX_dir.stem
    if humanX == "human4":
        continue
    
    rec_ppx = si.read_binary_folder(humanX_dir / "sippx")
    t_start_s = rec_ppx.get_times()[0]
    t_start_samples = rec_ppx.get_times()[0] * rec_ppx.get_sampling_frequency()
    
    drift_est = pd.read_csv(next((humanX_dir / "drift").glob("*.csv")))
    drift_est = drift_est.values
    
    # subdir = datadir / humanX
    # subdir = base / humanX
    subdir = datadir / humanX
    
    # with h5py.File(subdir / "sub" / "subtraction.h5", "r") as h5:
    with h5py.File(subdir/ "subtraction.h5", "r", locking=False) as h5:
        x, y, z = h5["localizations"][:, :3].T
        maxptp = h5["maxptps"][:]
        t_samples = h5["spike_index"][:, 0]
        t = t_samples / h5["fs"]
        geom = h5["geom"][:]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    s = ax.scatter(t, z, s=1, c=maxptp, lw=0, vmax=15, alpha=0.5, rasterized=True)
    cbar = plt.colorbar(s, ax=ax, shrink=0.5, label="amplitude (s.u.)")
    cbar.solids.set(alpha=1)
    
    ax.plot(t_start_s + np.arange(len(drift_est)), geom[:, 1].mean() + drift_est - drift_est.mean(), color="r", label="drift est.")
    ax.set_ylim([geom[:, 1].min() - 50, geom[:, 1].max() + 50])
    plt.xlabel("time (s)")
    plt.ylabel("depth along probe ($\\mu$m)")
    plt.title(f"{humanX.split('_')[0]}: initial spike times vs. depths, drift estimate in sorted region", size=12)
    plt.show()
    fig.savefig(figdir / f"{humanX}_initial_tvy.pdf")
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_start_s + np.arange(len(drift_est)), geom[:, 1].mean() + drift_est - drift_est.mean(), color="k", label="drift est.")
    plt.xlabel("time (s)")
    plt.ylabel("depth along probe ($\\mu$m)")
    plt.title(f"{humanX.split('_')[0]}: drift estimate in sorted region", size=12)
    plt.show()
    fig.savefig(figdir / f"{humanX}_drift_est.pdf")
    plt.close(fig)
    


# %%

# %%
from matplotlib.patches import Rectangle

# %%
with h5py.File(base / "human6" / "deconv3" / "deconv_results.h5") as h5:
    print(list(h5.keys()))
    g = h5["cleaned_tpca"]
    print(g.keys())
    mean = g["tpca_mean"][:]
    c = g["tpca_components"][:]

# %%
plt.plot(mean)

# %%
np.around(c @ c.T)

# %%
plt.plot(c.T)

# %%
for humanX_dir in base.iterdir():
    if not (humanX_dir / "phy").exists():
        continue  

    humanX = humanX_dir.stem
    
    rec_ppx = si.read_binary_folder(humanX_dir / "sippx")
    t_start_s = rec_ppx.get_times()[0]
    t_start_samples = rec_ppx.get_times()[0] * rec_ppx.get_sampling_frequency()
    
    drift_est = pd.read_csv(next((humanX_dir / "drift").glob("*.csv")))
    drift_est = drift_est.values
    
    subdir = datadir / humanX
    
    deconv3_h5 = base / humanX / "deconv3" / "deconv_results.h5"
    
    with h5py.File(deconv3_h5, "r") as h5:
        x, y, z = h5["localizations"][:, :3].T
        z_reg = h5["z_reg"][:]
        maxptp = h5["maxptps"][:]
        t_samples = h5["spike_index"][:, 0]
        labels = h5["deconv_spike_train"][:, 1]
        t = t_samples / 20000
        geom = h5["geom"][:]
    
    fig = plt.figure(figsize=(7, 4))
    plt.scatter(t, z, c=labels, cmap=cc.m_glasbey, s=1, alpha=0.5, linewidths=0, rasterized=True)

    xt = plt.xticks()[0][1:-1]
    plt.xticks(xt, [xtt + t_start_s for xtt in xt])
    plt.xlabel("time (s)")
    plt.ylabel("depth (um)")
    plt.title(f"{humanX} deconv3: time vs. depth for {len(labels)} sorted spikes.")
    plt.show()
    fig.savefig(figdir / f"{humanX}_deconv3_sorted_tvy.pdf")
    
    plt.close(fig)
    
    which = (x > geom[:,0].min() - 100) & (x < geom[:,0].max() + 100)
    
    fig, axes = plt.subplots(1, 3, figsize=(7, 7), gridspec_kw=dict(wspace=0.1))
    cluster_viz_index.array_scatter(
        labels[which],
        geom,
        x[which],
        z_reg[which],
        maxptp[which],
        axes=axes,
        zlim=(geom.min() - 25, geom.max() + 25),
        c=5,
    )
    ccc = axes[2].add_patch(Rectangle([0.00, 0.00], 0.3, 0.25, color="w", linewidth=0, alpha=0.5)) 
    ccc.set_transform(axes[2].transAxes)
    cax = axes[2].inset_axes([0.05, 0.02, 0.075, 0.2])
    cbar = plt.colorbar(axes[0].collections[0], ax=axes[1], cax=cax)
    cbar.set_label(label="amplitude", labelpad=-12)
    cbar.solids.set(alpha=1)
    cax.set_yticks([0, 1], labels=[3, 15])
    axes[0].set_xlabel("x ($\\mu$m)")
    axes[2].set_xlabel("x ($\\mu$m)")
    axes[0].set_ylabel("depth ($\\mu$m)")
    axes[0].set_xlim([geom[:, 0].min() - 100, geom[:, 0].max() + 100])
    axes[2].set_xlim([geom[:, 0].min() - 100, geom[:, 0].max() + 100])
    axes[1].set_yticks([])
    axes[1].set_xlabel("log trough-to-peak amplitude")
    axes[2].set_yticks([])
    nunits = np.setdiff1d(np.unique(labels), [-1]).size
    axes[1].set_title(f"Spatial view, {nunits} units.")
    plt.show()
    fig.savefig(figdir / f"{humanX}_deconv3_spatial.pdf")
    plt.close(fig)
    
    dz = 500
    bestz = 0
    dptp = 0
    for z in range(0, int(geom[:, 1].max()) - dz + 1, 100):
        wh = which & (z_reg > z) & (z_reg < z + dz)
        if not wh.any():
            continue
        dptpz = np.median([(maxptp[which & (z_reg > z + i) & (z_reg < z + i + 100)].ptp() if (which & (z_reg > z + i) & (z_reg < z + i + 100)).any() else 0) for i in range(0, dz, 100) ])
        if dptpz > dptp:
            bestz = z
            dptp = dptpz
    wh = which & (z_reg > bestz) & (z_reg < bestz + dz)
    
    fig, axes = plt.subplots(1, 3, figsize=(7, 7), gridspec_kw=dict(wspace=0.1))
    cluster_viz_index.array_scatter(
        labels[which],
        geom[(geom[:, 1] >= bestz) & (geom[:, 1] <= bestz + dz)],
        x[which],
        z_reg[which],
        maxptp[which],
        axes=axes,
        zlim=(bestz, bestz + dz),
        c=5,
    )
    ccc = axes[2].add_patch(Rectangle([0.00, 0.00], 0.3, 0.25, color="w", linewidth=0, alpha=0.5)) 
    ccc.set_transform(axes[2].transAxes)
    cax = axes[2].inset_axes([0.05, 0.02, 0.075, 0.2])
    cbar = plt.colorbar(axes[0].collections[0], ax=axes[1], cax=cax)
    cbar.set_label(label="amplitude", labelpad=-12)
    cbar.solids.set(alpha=1)
    cax.set_yticks([0, 1], labels=[3, 15])
    axes[0].set_xlabel("x ($\\mu$m)")
    axes[2].set_xlabel("x ($\\mu$m)")
    axes[0].set_ylabel("depth ($\\mu$m), detail")
    axes[0].set_xlim([geom[:, 0].min() - 100, geom[:, 0].max() + 100])
    axes[2].set_xlim([geom[:, 0].min() - 100, geom[:, 0].max() + 100])
    axes[1].set_yticks([])
    axes[1].set_xlabel("log trough-to-peak amplitude")
    axes[2].set_yticks([])
    nunits = np.setdiff1d(np.unique(labels), [-1]).size
    axes[1].set_title(f"Spatial view, detail")
    plt.show()
    fig.savefig(figdir / f"{humanX}_deconv3_spatial_detail.pdf")
    plt.close(fig)


# %%
