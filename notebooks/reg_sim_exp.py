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
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import matplotlib.pyplot as plt
import spikeinterface.full as si
from spikeinterface.sortingcomponents import motion_estimation, peak_detection, peak_localization

# %%
import neuropixel

# %%
from pathlib import Path

# %%

# %%
import numpy as np

# %%
from scipy.io import loadmat
h = loadmat("/Volumes/paninski-locker/data/trautmann/dredge_data/neuropixNHPv1_kilosortChanMap_v1.mat")
geom = np.c_[h['xcoords'], h['ycoords']]

# %%
geom.shape

# %%
name = "dd1"

# %%
# figdir = Path("/moto/stats/users/ciw2107/mysi_reg_test")
figdir = Path("/Users/charlie/data/mysi_reg_test")
figdir.mkdir(exist_ok=True)

# %%
job_kw = dict(chunk_size=30_000, n_jobs=10)

# %%
rec = si.read_binary(
    # "/moto/stats/users/ciw2107/icassp2023/drift-dataset1/destriped_p1_g0_t0.imec0.ap.bin",
    "/Users/charlie/data/icassp2023/drift-dataset1/destriped_p1_g0_t0.imec0.ap.bin",
    # "/local/p1_g0_t0.imec0.ap.bin",
    sampling_frequency=30000,
    num_chan=384,
    dtype=np.int16,
)

h = neuropixel.dense_layout(version=2)
geom = np.c_[h['x'], h['y']]
rec.set_dummy_probe_from_locations(geom, shape_params=dict(radius=10))

# %%
# # !rsync -avP /moto/stats/users/ciw2107/dataset1/p1_g0_t0.imec0.ap.* /local/

# %%
# # !ls /local

# %%
# rec = si.read_binary(
#     # "/moto/stats/users/ciw2107/icassp2023/drift-dataset1/destriped_p1_g0_t0.imec0.ap.bin",
#     # "/local/p1_g0_t0.imec0.ap.bin",
#     # "/Users/charlie/data/icassp2023/drift-dataset1/destriped_p1_g0_t0.imec0.ap.bin",
#     "/Users/charlie/data/icassp2023/drift-dataset1/destriped_p1_g0_t0.imec0.ap.bin",
#     sampling_frequency=30000,
#     num_chan=384,
#     dtype=np.float32,
#     # dtype=np.int16,
# )
# rec = si.read_spikeglx("/local/")
rec = si.read_spikeglx("/Users/charlie/data/dataset1/")
# h = neuropixel.dense_layout(version=2)
# geom = np.c_[h['x'], h['y']]
# rec.set_dummy_probe_from_locations(geom, shape_params=dict(radius=10))
# rec = si.
rec

# %%
fs = rec.get_sampling_frequency()
rec = si.highpass_filter(rec, dtype=np.float32)
rec = si.phase_shift(rec)
bad_channel_ids, channel_labels = si.detect_bad_channels(rec, num_random_chunks=100)
print(f"{bad_channel_ids=}")
rec = si.interpolate_bad_channels(rec, bad_channel_ids)
rec = si.highpass_spatial_filter(rec)
# we had been working with this before -- should switch to MAD,
# but we need to rethink the thresholds
rec = si.zscore(rec, mode="mean+std", num_chunks_per_segment=100, margin_frames=100*fs)

# %%
t = rec.get_traces(start_frame=410000, end_frame=420000)
t.min(), t.max(), t.std(axis=0).mean()

# %%
overwrite = False

peaks_npy = figdir / f"{name}_peaks.npy"
if overwrite or not peaks_npy.exists():
    peaks = peak_detection.detect_peaks(
        rec,
        method="locally_exclusive",
        local_radius_um=100,
        peak_sign="both",
        detect_threshold=5,
        **job_kw,
    )
    np.save(peaks_npy, peaks)
peaks = np.load(peaks_npy)

peak_locations_npy = figdir / f"{name}_peak_locations.npy"
if overwrite or not peak_locations_npy.exists():
    peak_locations = peak_localization.localize_peaks(
        rec,
        peaks,
        ms_before=0.3,
        ms_after=0.6,
        method='monopolar_triangulation',
        **{'local_radius_um': 100., 'max_distance_um': 1000., 'optimizer': 'minimize_with_log_penality'},
        **job_kw,
    )
    np.save(peak_locations_npy, peak_locations)
peak_locations = np.load(peak_locations_npy)

# %%
bin_um = 5.0
bin_s = 2.0
margin_um = -1300

# %%
plt.hist(np.abs(peaks["amplitude"]), bins=128, log=True);

# %%
plt.hist(np.log1p(np.abs(peaks["amplitude"])), bins=128, log=True);

# %%
spatial_bin_edges = motion_estimation.get_spatial_bin_edges(rec, "y", margin_um, bin_um)
# non_rigid_windows, non_rigid_window_centers = get_windows(True, bin_um, contact_pos, spatial_bin_edges,
#                                                               margin_um, win_step_um, win_sigma_um, win_shape)
motion_histogram, temporal_hist_bin_edges, spatial_hist_bin_edges = \
            motion_estimation.make_2d_motion_histogram(rec, peaks,
                                     peak_locations,
                                     direction="y",
                                     bin_duration_s=bin_s,
                                     spatial_bin_edges=spatial_bin_edges,
                                     weight_with_amplitude=True,
                                    )
plt.imshow(motion_histogram.T, aspect="auto", vmax=15)

# %%
subsample_p = 0.02 + 0.98 * (1 - peaks["sample_ind"].astype(float) / peaks["sample_ind"].max())
subsample_p = 0.02 + 0.98 * (0.5 + np.sin(np.pi + np.pi * peaks["sample_ind"].astype(float) / peaks["sample_ind"].max()) / 2)
rg = np.random.default_rng(0)
subsamp = rg.binomial(1, subsample_p)
subsamp = np.flatnonzero(subsamp)
peaks_sub = peaks[subsamp]
peak_locations_sub = peak_locations[subsamp]

# %%
spatial_bin_edges = motion_estimation.get_spatial_bin_edges(rec, "y", margin_um, bin_um)
# non_rigid_windows, non_rigid_window_centers = get_windows(True, bin_um, contact_pos, spatial_bin_edges,
#                                                               margin_um, win_step_um, win_sigma_um, win_shape)
motion_histogram, temporal_hist_bin_edges, spatial_hist_bin_edges = \
            motion_estimation.make_2d_motion_histogram(rec, peaks_sub,
                                     peak_locations_sub,
                                     direction="y",
                                     bin_duration_s=bin_s,
                                     spatial_bin_edges=spatial_bin_edges,
                                     weight_with_amplitude=True,
                                    )
plt.imshow(motion_histogram.T, aspect="auto", vmax=15)

# %%
motion_ks, temporal_bins_ks, non_rigid_window_centers_ks, extra_check_ks = motion_estimation.estimate_motion(
    rec, peaks_sub, peak_locations_sub,
    direction='y', bin_duration_s=bin_s, bin_um=bin_um, margin_um=margin_um,
    rigid=True,
    post_clean=False, speed_threshold=30, sigma_smooth_s=None,
    method='iterative_template',
    output_extra_check=True,
    progress_bar=True,
    upsample_to_histogram_bin=False,
    verbose=False,
)

# %%
motion_dec, temporal_bins_dec, non_rigid_window_centers_dec, extra_check_dec = motion_estimation.estimate_motion(
    rec, peaks_sub, peak_locations_sub,
    direction='y', bin_duration_s=bin_s, bin_um=bin_um, margin_um=margin_um,
    rigid=True,
    post_clean=False, 
    method='decentralized',
    convergence_method="lsqr_robust",
    lsqr_robust_n_iter=1,
    output_extra_check=True,
    progress_bar=True,
    upsample_to_histogram_bin=False,
    time_horizon_s=None,
    verbose=False,
)

# %%
extent = [*extra_check_dec['temporal_hist_bin_edges'][[0, -1]], *extra_check_dec['spatial_hist_bin_edges'][[-1, 0]]]
plt.figure(figsize=(10,10))
plt.imshow(extra_check_dec['motion_histogram'].T, aspect="auto", vmax=15, extent=extent)
plt.plot(extra_check_dec['temporal_hist_bin_edges'][:-1], -100 + motion_dec + non_rigid_window_centers_dec, color="w", lw=1, label="ours");
plt.plot(extra_check_ks['temporal_hist_bin_edges'][:-1], 100 + motion_ks + non_rigid_window_centers_ks, color="y", lw=1, label="ks");
plt.legend(fancybox=False, framealpha=1);

# %%
D_unnorm, S_unnorm = motion_estimation.compute_pairwise_displacement(motion_histogram, bin_um,
                                                  window=np.ones(motion_histogram.shape[1], dtype=motion_histogram.dtype),
                                                  method="conv",
                                                  weight_scale="linear",
                                                  error_sigma=0,
                                                  conv_engine="torch",
                                                  torch_device=None,
                                                  batch_size=64,
                                                  max_displacement_um=100.0,
                                                  normalized_xcorr=False,
                                                  centered_xcorr=False,
                                                  corr_threshold=0.0,
                                                  time_horizon_s=None,
                                                  bin_duration_s=bin_s,
                                                  progress_bar=False)
D_norm, S_norm = motion_estimation.compute_pairwise_displacement(motion_histogram, bin_um,
                                                  window=np.ones(motion_histogram.shape[1], dtype=motion_histogram.dtype),
                                                  method="conv",
                                                  weight_scale="linear",
                                                  error_sigma=0,
                                                  conv_engine="torch",
                                                  torch_device=None,
                                                  batch_size=64,
                                                  max_displacement_um=100.0,
                                                  normalized_xcorr=True,
                                                  centered_xcorr=False,
                                                  corr_threshold=0.0,
                                                  time_horizon_s=None,
                                                  bin_duration_s=bin_s,
                                                  progress_bar=False)
D_corr, S_corr = motion_estimation.compute_pairwise_displacement(motion_histogram, bin_um,
                                                  window=np.ones(motion_histogram.shape[1], dtype=motion_histogram.dtype),
                                                  method="conv",
                                                  weight_scale="linear",
                                                  error_sigma=0,
                                                  conv_engine="torch",
                                                  torch_device=None,
                                                  batch_size=64,
                                                  max_displacement_um=100.0,
                                                  normalized_xcorr=True,
                                                  centered_xcorr=True,
                                                  corr_threshold=0.0,
                                                  time_horizon_s=None,
                                                  bin_duration_s=bin_s,
                                                  progress_bar=False)
T = D_unnorm.shape[0]

# %%
Dc_wm = (D_corr * S_corr).sum(1) / S_corr.sum(1)
Dc_wvar = (np.square(D_corr - Dc_wm) * S_corr).sum(1) / S_corr.sum(1)
Dc_wstd = np.sqrt(Dc_wvar)
Sc_wstd = 1/(Dc_wstd[None, :] * Dc_wstd[:, None])

# %%
D_corr_std = D_corr.std(axis=1)
D_corr_std = D_corr_std[:, None] * D_corr_std[None, :]
D_corr_std = 1/D_corr_std
np.fill_diagonal(D_corr_std, 1)

# %%
S_corr_rescaled = S_corr.copy()
S_corr_rescaled = np.einsum("ij,i,j->ij", S_corr, np.sqrt(D_corr.std(1))/np.sqrt(np.median(S_corr,1)), np.sqrt(D_corr.std(1))/np.sqrt(np.median(S_corr,1)))

# %%
S_corr_rescaled = S_corr * (1 / (np.triu(S_corr).std() * np.triu(D_corr).std()))

# %%
1

# %%
from sklearn.gaussian_process.kernels import Matern

# %%
from spike_psvae.ibme_corr import calc_corr_decent_pair


# %%
def shifted_template(R, Di, Si):
    assert (R.shape[0],) == Di.shape == Si.shape
    pad = int(np.abs(Di // bin_um).max())
    Rshifted = np.empty_like(R)
    for i, d in enumerate((Di // bin_um).astype(int)):
        Rshifted[i, :] = np.pad(R[i, :], [(pad, pad)])[pad + d : R.shape[1] + pad + d]
    return (Rshifted * Si[:, None]).sum(0) / Si.sum()

def shifted_templates(R, D, S):
    sts = np.empty_like(R)
    for i, (Di, Si) in enumerate(zip(D.T, S.T)):
        sts[i] = shifted_template(R, Di, Si)
    return sts

def centralize_shifted_templates(R, local_templates, axis=1):
    D, C = calc_corr_decent_pair(R.T, local_templates.T)
    return (D * C).sum(axis=axis) / C.sum(axis=axis)


# %%
motion_histogram.shape, D_corr.shape, T

# %%
mo_temps_corr = shifted_templates(motion_histogram, D_corr, S_corr)
mo_temps_unnorm = shifted_templates(motion_histogram, D_unnorm, S_unnorm)

# %%
p_temps_corr_weightavg0 = centralize_shifted_templates(motion_histogram, mo_temps_corr, axis=0)
p_temps_corr_weightavg1 = centralize_shifted_templates(motion_histogram, mo_temps_corr, axis=1)

# %%
plt.imshow(motion_histogram.T, aspect="auto"); plt.colorbar(); plt.title("original raster")

# %%
plt.imshow(mo_temps_corr.T, aspect="auto"); plt.colorbar(); plt.title("corr weighted local templates")

# %%
D_temp_corr, S_temp_corr = motion_estimation.compute_pairwise_displacement(mo_temps_corr, bin_um,
                                                  window=np.ones(motion_histogram.shape[1], dtype=motion_histogram.dtype),
                                                  method="conv",
                                                  weight_scale="linear",
                                                  error_sigma=0,
                                                  conv_engine="torch",
                                                  torch_device=None,
                                                  batch_size=64,
                                                  max_displacement_um=100.0,
                                                  normalized_xcorr=True,
                                                  centered_xcorr=True,
                                                  corr_threshold=0.0,
                                                  time_horizon_s=None,
                                                  bin_duration_s=bin_s,
                                                  progress_bar=False)


# %%
def newt_solve(D, S, Sigma0inv, normalize=None):
    """D is TxT displacement, S is TxT subsampling or soft weights matrix"""
    
    if normalize == "sym":
        uu = 1/np.sqrt((S + S.T).sum(1))
        S = np.einsum("i,j,ij->ij",uu,uu,S)
    if normalize == "rescale_std":
        S = S / (np.triu(S_corr).std() * np.triu(D_corr).std())
    
    # forget the factor of 2, we'll put it in later
    # HS = (S + S.T) - np.diag((S + S.T).sum(1))
    HS = S.copy()
    HS += S.T
    np.fill_diagonal(HS, np.diagonal(HS) - S.sum(1) - S.sum(0))
    # grad_at_0 = (S * D - S.T * D.T).sum(1)
    SD = S * D
    grad_at_0 = SD.sum(1) - SD.sum(0)
    # Next line would be (Sigma0inv ./ 2 .- HS) \ grad in matlab
    p = la.solve(Sigma0inv / 2 - HS, grad_at_0)
    return p, HS


# %%

# %%
Sigma0inv_bm = np.eye(T) - np.diag(0.5 * np.ones(T - 1), k=1) - np.diag(0.5 * np.ones(T - 1), k=-1) 

ker = Matern(length_scale=10, nu=0.5)
Sigma0_mat05 = ker(np.arange(T)[:, None])
Sigma0inv_mat05 = la.inv(Sigma0_mat05)

Sigma0invs = {"bm": Sigma0inv_bm, "heavy_mat": 0.5 * T * Sigma0inv_mat05}

# %%
pmat, HS = newt_solve(D_corr, S_corr, 0.5 * T * Sigma0inv_mat05)
postcov = la.inv(Sigma0inv_bm - 2 * HS)
postvarmat = np.diagonal(postcov)

pog, HS = newt_solve(D_corr, S_corr, Sigma0inv_bm)
postcov = la.inv(Sigma0inv_bm - 2 * HS)
postvarog = np.diagonal(postcov)


ptemp, HS = newt_solve(D_temp_corr, S_temp_corr, Sigma0inv_bm)
postcov = la.inv(Sigma0inv_bm - 2 * HS)
postvartemp = np.diagonal(postcov)

pstd, HS = newt_solve(D_corr, D_corr_std, Sigma0inv_bm)
postcov = la.inv(Sigma0inv_bm - 2 * HS)
postvarstd = np.diagonal(postcov)

pmatstd, HS = newt_solve(D_corr, D_corr_std, 1 * Sigma0inv_mat05)
postcov = la.inv(Sigma0inv_bm - 2 * HS)
postvarmatstd = np.diagonal(postcov)

# %%
pks = motion_ks.squeeze()

# %%
plt.plot(-p_temps_corr_weightavg0, color="b")
plt.plot(p_temps_corr_weightavg1, color="r")

# %%
plt.plot(p_temps_corr_weightavg1, color="r")
plt.plot(-p_temps_corr_weightavg0, color="b")


# %%
plt.plot(pks - pks[0], label="ks")
plt.plot(pog - pog[0], label="dredge S=corr, bm prior")
plt.plot(ptemp - ptemp[0], label="decentralizing local templates, S=corr, bm prior")
plt.plot(-5 * (p_temps_corr_weightavg0 - p_temps_corr_weightavg0[0]), label="local templates weighted avg, S=corr, bm prior")
plt.legend()

# %%
plt.plot(pks - pks[0], label="ks")
plt.plot(100 + pog - pog[0], label="dredge S=corr, bm prior")
plt.plot(200 + ptemp - ptemp[0], label="decentralizing local templates, S=corr, bm prior")
plt.plot(200 + ptemp - ptemp[0], label="decentralizing local templates, S=corr, bm prior")
plt.plot(300 + -5 * (p_temps_corr_weightavg0 - p_temps_corr_weightavg0[0]), label="local templates weighted avg, S=corr, bm prior")

plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1, 0, 0))

# %%
plt.plot(pmatstd)

# %%
# plt.plot(pog)
# plt.plot(pstd)
plt.plot(pks)
plt.plot(pmat)
plt.plot(pmatstd)


# %%
plt.plot(pks - pmat)

# %%
plt.plot(postvarog)
plt.plot(postvarstd)
plt.plot(postvarmat)

# %%
Sigma0inv_bm = np.eye(T) - np.diag(0.5 * np.ones(T - 1), k=1) - np.diag(0.5 * np.ones(T - 1), k=-1) 

ker = Matern(length_scale=10, nu=0.5)
Sigma0_mat05 = ker(np.arange(T)[:, None])
Sigma0inv_mat05 = la.inv(Sigma0_mat05)

Sigma0invs = {"bm": Sigma0inv_bm, "mat": Sigma0inv_mat05, "heavy_mat": 0.5 * T * Sigma0inv_mat05}

# %%
dss = {
    # "unnorm": (D_unnorm, S_unnorm),
    # "norm": (D_norm, S_norm),
    "corr": (D_corr, S_corr),
    "corr_dstd": (D_corr, D_corr_std),
}

# %% tags=[]
for dstype, (D, S) in dss.items():
    for prior, Sigma0inv in Sigma0invs.items():
        for normtype in (None, "rescale_std"):
            pp, HS = newt_solve(D, S, Sigma0inv, normalize=normtype)
            postcov = la.inv(Sigma0inv - 2 * HS)
            postvar = np.diagonal(postcov)
            
            extent = [*extra_check_dec['temporal_hist_bin_edges'][[0, -1]], *extra_check_dec['spatial_hist_bin_edges'][[-1, 0]]]
            fig, (aa, ab) = plt.subplots(ncols=2, figsize=(10,5))
            aa.imshow(extra_check_dec['motion_histogram'].T, aspect="auto", vmax=15, extent=extent)
            aa.plot(extra_check_dec['temporal_hist_bin_edges'][:-1], -100 + motion_dec + non_rigid_window_centers_dec, color="w", lw=1, label="ours");
            aa.plot(extra_check_dec['temporal_hist_bin_edges'][:-1], 0 + pp[:,None] + non_rigid_window_centers_dec, color="b", lw=1, label="test");
            aa.fill_between(
                extra_check_dec['temporal_hist_bin_edges'][:-1],
                (0 + pp[:,None] + non_rigid_window_centers_dec - postvar[:, None] * (10 / postvar.mean())).squeeze(),
                (0 + pp[:,None] + non_rigid_window_centers_dec + postvar[:, None] * (10 / postvar.mean())).squeeze(),
                color="w",
                alpha=0.5,
            )
            aa.plot(extra_check_ks['temporal_hist_bin_edges'][:-1], 100 + motion_ks + non_rigid_window_centers_ks, color="y", lw=1, label="ks");
            aa.set_title(f"{dstype=} {prior=} {normtype=}")
            aa.legend(fancybox=False, framealpha=1);
            
            ab.plot(postvar)
            ab.set_ylabel("post var")
            
            plt.show()

# %%

# %%
# centered brownian motion cov
Sigma_bm = np.minimum(np.arange(T)[:, None], np.arange(T)[None, :])
centering = np.full((T, T), -1/T)
np.fill_diagonal(centering, (T-1)/T)
Sigma0_bm = centering @ Sigma_bm @ centering
Sigma0inv_bm = np.eye(T) - np.diag(0.5 * np.ones(T - 1), k=1) - np.diag(0.5 * np.ones(T - 1), k=-1) 

# Matern cov
ker = Matern(length_scale=10, nu=0.5)
Sigma0_mat05 = ker(np.arange(T)[:, None])
Sigma0inv_mat05 = la.inv(Sigma0_mat05)

ker = Matern(length_scale=10, nu=1.5)
Sigma0_mat15 = ker(np.arange(T)[:, None])
Sigma0inv_mat15 = la.inv(Sigma0_mat15)

# weird idea
Scov = S_unnorm + S_unnorm.T
Svar = np.diagonal(Scov)
Sstd = np.sqrt(Svar)
Ostd = Sstd[:, None] * Sstd[None, :]
Scorr = Scov / Ostd
Sigma0_S_unnorm = np.square(np.maximum(Sstd[:, None], Sstd[None, :])) * Scorr - Scov
np.fill_diagonal(Sigma0_S_unnorm, 1)
Sigma0inv_S_unnorm = la.inv(Sigma0_S_unnorm)

# weird idea
# Sigma0_S_corr = S_corr + S_corr.T
# Sigma0inv_S_corr = la.inv(Sigma0_S_corr)

# other needed matrices
_1 = np.ones(T)
eye = sp.eye(T)

# %%
plt.plot(postvar);

# %%
extent = [*extra_check_dec['temporal_hist_bin_edges'][[0, -1]], *extra_check_dec['spatial_hist_bin_edges'][[-1, 0]]]
plt.figure(figsize=(10,10))
plt.imshow(extra_check_dec['motion_histogram'].T, aspect="auto", vmax=15, extent=extent)
plt.plot(extra_check_dec['temporal_hist_bin_edges'][:-1], -100 + motion_dec + non_rigid_window_centers_dec, color="w", lw=1, label="ours");
plt.plot(extra_check_dec['temporal_hist_bin_edges'][:-1], 0 + pp[:,None] + non_rigid_window_centers_dec, color="b", lw=1, label="test");
plt.fill_between(
    extra_check_dec['temporal_hist_bin_edges'][:-1],
    (0 + pp[:,None] + non_rigid_window_centers_dec - postvar[:, None] * (10 / postvar.mean())).squeeze(),
    (0 + pp[:,None] + non_rigid_window_centers_dec + postvar[:, None] * (10 / postvar.mean())).squeeze(),
    color="w",
    alpha=0.5,
)
plt.plot(extra_check_ks['temporal_hist_bin_edges'][:-1], 100 + motion_ks + non_rigid_window_centers_ks, color="y", lw=1, label="ks");
plt.legend(fancybox=False, framealpha=1);

# %%

# %%

# %%
