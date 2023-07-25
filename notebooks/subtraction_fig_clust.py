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
from spike_psvae import before_deconv_merge_split, newms, cluster_viz_index
from spike_psvae.hybrid_analysis import Sorting
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py

# %%
sub_fig_dir = Path("/Users/charlie/data/spike_sorting_paper_figs/subtraction_fig/")
sub_data_dir = sub_fig_dir / "subtraction_fig_data"

# %%
raw_bin = sub_data_dir / "zad5" / "traces_cached_seg0.raw"

# %%
# %ll {sub_data_dir}

# %%
yy_h5 = sub_data_dir / "subtraction_yes_nn_yes_thresholds.h5"

# %%
yy_clustdir0 = sub_data_dir / "yes_nn_yes_thresholds_clust"
# %ll {yy_clustdir0}

# %%
with h5py.File(yy_h5) as h5:
    geom = h5["geom"][:]
    xzptp = np.c_[h5["localizations"][:, [0, 2]], h5["maxptps"][:]]

# %%
order0 = np.load(yy_clustdir0 / "merge_order.npy")
st0 = np.load(yy_clustdir0 / "merge_st.npy")
temps0 = np.load(yy_clustdir0 / "merge_templates.npy")

# %%
np.array_equal(order0, np.arange(len(order0)))

# %%
sorting0 = Sorting(raw_bin, geom, *st0.T, "sorting0", templates=temps0, spike_xzptp=xzptp[order0])

# %%
sorting0.array_scatter(zlim=(2000, 2500))

# %%
yy_h5.stem

# %%
sub_h5 = yy_h5

# %% tags=[]
# run initial clustering on all of these
clustdir = sub_data_dir / f"{yy_h5.stem}_clust_test"
name = sub_h5.stem.split("subtraction_")[-1]
with h5py.File(sub_h5, "r") as h5:
    spind = h5["spike_index"][:]
    geom = h5["geom"][:]
    # if "z_reg" not in h5:
    #     h5.create_dataset("z_reg", data=h5["localizations"][:, 2])
    # if "ast_order" in h5:
    #     print("Already done.")
    #     continue
ast1, templates1, order1 = newms.new_merge_split(
    spind,
    len(geom),
    raw_bin,
    sub_h5,
    geom,
    clustdir,
    n_workers=5,
    merge_resid_threshold=5.0,
    threshold_diptest=0.5,
    relocated=False,
    trough_offset=42,
    spike_length_samples=121,
    # extra_pc_split=True,
    # pc_only=False,
    # exp_split=False,
    load_split=False,
    split_kwargs=dict(
        split_steps=(
            before_deconv_merge_split.herding_split,
        ),
        recursive_steps=(True,),
        split_step_kwargs=(
            dict(
                hdbscan_kwargs=dict(
                    min_cluster_size=50,
                    # min_samples=5,
                    cluster_selection_epsilon=30.0,
                ),
            ),
        ),
    ),
)
# with h5py.File(sub_h5, "r+") as h5:
#     h5.create_dataset("aligned_spike_train", data=ast)
#     h5.create_dataset("templates", data=templates)
#     h5.create_dataset("ast_order", data=order)
sorting1 = Sorting(raw_bin, geom, *ast1.T, "sorting1", templates=templates1, spike_xzptp=xzptp[order1])

# %%
sorting1.array_scatter(zlim=(2000, 2500))
