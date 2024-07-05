# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python [conda env:a]
#     language: python
#     name: conda-env-a-py
# ---

# %%

# %%
1

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import dartsort
import numpy as np
import dartsort.vis as dartvis
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA, TruncatedSVD
import spikeinterface.full as si
from dartsort.config import *
from dartsort.cluster import initial, density
import dataclasses
from dartsort.util import drift_util
import warnings
from tqdm.auto import trange, tqdm
from scipy.stats import chi2
from ephysx import spike_gmm, spike_lrgmm, spike_basic, ppca
from matplotlib import colors
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster

# %%
# %config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import offset_copy
from matplotlib.patches import Ellipse, Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import contextlib
import colorcet as cc

plt.rc("figure", dpi=300)
plt.rc("figure", figsize=(2, 2))
SMALL_SIZE = 5
MEDIUM_SIZE = 7
BIGGER_SIZE =  8
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# %%
# global
model_radius = 15.0
cfg = DARTsortConfig(
    matching_iterations=2,
    subtraction_config=SubtractionConfig(
        detection_thresholds=(12, 9, 6, 5),
        extract_radius=75.0,
        max_waveforms_fit=20_000,
        subtraction_denoising_config=FeaturizationConfig(
            denoise_only=True,
            input_waveforms_name="raw",
            output_waveforms_name="subtracted",
            tpca_fit_radius=model_radius,
        ),
        residnorm_decrease_threshold=20.0,
    ),
    matching_config=MatchingConfig(
        threshold=2500.0,
        max_waveforms_fit=20_000,
        extract_radius=75.0,
    ),
    template_config=TemplateConfig(
        denoising_fit_radius=model_radius,
        denoising_snr_threshold=100.0,
        superres_templates=False,
    ),
    clustering_config=ClusteringConfig(
        cluster_strategy="density_peaks",
        sigma_regional=25.0,
        noise_density=1.0,
        ensemble_strategy=None,
    ),
    split_merge_config=SplitMergeConfig(
        min_spatial_cosine=0.0,
        linkage="single",
        # linkage="weighted_template",
        split_strategy_kwargs=dict(
            channel_selection_radius=model_radius,
            max_spikes=10_000,
        ),
        merge_template_config=TemplateConfig(
            denoising_fit_radius=model_radius,
            denoising_snr_threshold=100.0,
            superres_templates=False,
        )
    ),
    featurization_config=FeaturizationConfig(
        tpca_fit_radius=model_radius,
        localization_radius=50.0,
        localization_model="dipole",
    ),
    motion_estimation_config=MotionEstimationConfig(
        max_dt_s=1000,
        window_scale_um=250,
        window_step_um=75,
        window_margin_um=-150,
        min_amplitude=15.0,
    ),
)

# %%
rec = si.read_binary_folder("/home/charlie/data/uhdzigzagzye57cmr/")

# %%
sub_st = dartsort.DARTsortSorting.from_peeling_hdf5(
    "/home/charlie/data/uhdzigzagzye57cmr_sub/subtraction.h5",
    load_simple_features=False,
)
sub_st

# %%
motion_est = dartsort.estimate_motion(rec, sub_st, sub_st.parent_h5_path.parent, **dataclasses.asdict(cfg.motion_estimation_config))

# %%
chunk_time_ranges = initial.chunk_time_ranges(rec, chunk_length_samples=30_000 * 300)
chunk11_dpc = initial.cluster_chunk(sub_st.parent_h5_path, cfg.clustering_config, chunk_time_range_s=chunk_time_ranges[10], motion_est=motion_est)

# %%
mask = chunk11_dpc.labels >= 0

# %%
h5_path = chunk11_dpc.parent_h5_path

# %%
dataset_name = "collisioncleaned_tpca_features"


# %%

# %%
def _read_by_chunk(mask, dataset):
    """
    mask : boolean array of shape dataset.shape[:1]
    dataset : chunked h5py.Dataset
    """
    out = np.empty((mask.sum(), *dataset.shape[1:]), dtype=dataset.dtype)
    n = 0
    for sli, *_ in dataset.iter_chunks():
        m = np.flatnonzero(mask[sli])
        nm = m.size
        if not nm:
            continue
        x = dataset[sli][m]
        out[n : n + nm] = x
        n += nm
    return out


# %%
def _read_by_chunk2(mask, dataset, axis=0):
    """
    mask : boolean array of shape (dataset.shape[axis],)
    dataset : chunked h5py.Dataset
    """
    out_shape = list(dataset.shape)
    out_shape[axis] = mask.sum()
    out = np.empty(out_shape, dtype=dataset.dtype)
    src_ix = [slice(None)] * dataset.ndim
    n = 0
    for slice_tuple in dataset.iter_chunks():
        ax_slice = slice_tuple[axis]
        m = np.flatnonzero(mask[ax_slice])
        nm = m.size
        if not nm:
            continue
        src_ix[axis] = m
        x = dataset[slice_tuple][tuple(src_ix)]
        dest_ix = (*slice_tuple[:axis], slice(n, n + nm), *slice_tuple[axis + 1 :])
        out[dest_ix] = x
        n += nm
    return out


# %%
mask.shape, mask.sum(), mask.mean()

# %%
indices = np.flatnonzero(mask)

# %%
with h5py.File(h5_path, "r", locking=False) as h5:
    print(h5[dataset_name].shape)
    print(h5[dataset_name].chunks)

# %%
# %%timeit
with h5py.File(h5_path, "r", locking=False) as h5:
    x = _read_by_chunk(mask, h5[dataset_name])
    # y = _read_by_chunk2(mask, h5[dataset_name])

# %%
m = mask.copy()
m[np.flatnonzero(m)[8000:]] = 0

# %%
m.shape, m.sum(), m.mean()

# %%
# %%timeit -r1
with h5py.File(h5_path, "r", locking=False) as h5:
    x = h5[dataset_name][m]

# %%
