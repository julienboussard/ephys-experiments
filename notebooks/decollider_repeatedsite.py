# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python [conda env:sp]
#     language: python
#     name: conda-env-sp-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from dartsort.templates import TemplateData, get_templates
from dartsort import DARTsortSorting, TemplateConfig
from dartsort.templates.template_util import get_realigned_sorting
from dartsort.transform import decollider
from dartsort.util import decollider_util, data_util, waveform_util
import spikeinterface.full as si
from one.api import ONE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ephysx import ibl_util, decollider_ibl_tests
import shutil
import pickle
import numpy as np
from dartsort.peel import ThresholdAndFeaturize
import torch
import h5py
import pandas as pd
from dartsort.util import multiprocessing_util
import dartsort
import string
import cloudpickle
from tqdm.auto import trange, tqdm

# %%
dartsort.__file__

# %%
# %config InlineBackend.figure_format = 'retina'
import colorcet as cc

plt.rc("figure", dpi=300)
plt.rc("figure", figsize=(7, 4))
SMALL_SIZE = 6.5
MEDIUM_SIZE = 8
BIGGER_SIZE = 10
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)


plt.rc('svg', fonttype='none')
plt.rc('ps', usedistiller='xpdf')
plt.rc('pdf', fonttype=42)

# %%
snip_length_s = 600

# %%
template_config = TemplateConfig(superres_templates=False, registered_templates=False)

# %%
1

# %%
one = ONE()
one

# %%
pids = open(Path("~/ceph/ibl_re_include.csv").expanduser()).read().split()
pids[:5]

# %%
rg = np.random.default_rng(0)
pids = list(rg.choice(pids, size=10, replace=False))
pids

# %%
templates_dir = Path("~/ceph/rs_templates").expanduser()
templates_dir.mkdir(exist_ok=True)

# %%
data_dir = Path("~/ceph/rs_snips").expanduser()
data_dir.mkdir(exist_ok=True)

# %%
scratch_dir = Path("/tmp/cwscratch")
scratch_dir.mkdir(exist_ok=True)

# %% [markdown]
# ## Compute templates and store preprocessed recordings

# %%
for pid in pids:
    print("-" * len(pid))
    print(pid)

    symlink_dir = data_dir / f"syms{pid}"
    tmp_rec_ppx_dir = data_dir / f"tmprecppx{pid}"
    rec_ppx_dir = data_dir / f"recppx{pid}"
    temps_dir = templates_dir / f"temps{pid}"
    sorting_pkl = data_dir / f"sorting{pid}.pkl"
    detections_h5 = data_dir / f"detect{pid}.h5"
    detections4_h5 = data_dir / f"det4ect{pid}.h5"

    if tmp_rec_ppx_dir.exists():
        shutil.rmtree(tmp_rec_ppx_dir)

    rec0 = ibl_util.read_popeye_cbin_ibl(pid, symlink_dir)
    center_sample = rec0.get_num_samples() // 2
    center_time = rec0._recording_segments[0].sample_index_to_time(
        center_sample
    )
    start_time = center_time - snip_length_s // 2
    end_time = center_time + snip_length_s // 2
    start_sample = rec0._recording_segments[0].time_to_sample_index(start_time)
    end_sample = rec0._recording_segments[0].time_to_sample_index(end_time)
    print(f"{start_sample=}, {end_sample=}")

    if rec_ppx_dir.exists():
        rec = si.read_binary_folder(str(rec_ppx_dir))
    else:
        # rec = ibl_util.read_and_destripe_popeye_cbin_ibl(
        #     pid, symlink_dir, one=one
        # )
        rec = ibl_util.read_and_catgt_popeye_cbin_ibl(pid, symlink_dir, one=one)
        print(rec)
        rec = rec.frame_slice(start_sample, end_sample)
        rec.save_to_folder(folder=str(tmp_rec_ppx_dir), n_jobs=8)
        tmp_rec_ppx_dir.rename(rec_ppx_dir)
        rec = si.read_binary_folder(str(rec_ppx_dir))
    print(rec)

    if False and sorting_pkl.exists():
        with open(sorting_pkl, "rb") as jar:
            sorting = pickle.load(jar)
    else:
        sorting = ibl_util.get_ks_sorting(pid, one=one)
        print("orig", sorting)
        print("orig", sorting.times_samples.min(), sorting.times_samples.max())
        print("orig", sorting.labels.min(), sorting.labels.max())
        sorting = data_util.subset_sorting_by_time_samples(
            sorting, start_sample=start_sample + 42, end_sample=end_sample - 79
        )
        print("subset time", sorting)
        print(
            "subset time",
            sorting.times_samples.min(),
            sorting.times_samples.max(),
        )
        print(
            "subset time",
            sorting.labels.min(),
            sorting.labels.max(),
        )
        sorting = data_util.subset_sorting_by_spike_count(
            sorting, min_spikes=50
        )
        print("subset spikes", sorting)
        print(
            "subset spikes",
            sorting.times_samples.min(),
            sorting.times_samples.max(),
        )
        print(
            "subset spikes",
            sorting.labels.min(),
            sorting.labels.max(),
        )
        sorting = data_util.reindex_sorting_labels(sorting)
        print("reindex", sorting)
        print(
            "reindex", sorting.times_samples.min(), sorting.times_samples.max()
        )
        print(
            "reindex", sorting.labels.min(), sorting.labels.max()
        )
        sorting = get_realigned_sorting(
            rec,
            sorting,
            realign_max_sample_shift=60,
            n_jobs=8,
            units_per_job=32,
            # device="cuda:1",
        )
        print("realign", sorting)
        print(
            "realign", sorting.times_samples.min(), sorting.times_samples.max()
        )
        print(
            "realign", sorting.labels.min(), sorting.labels.max()
        )

        sorting = data_util.subset_sorting_by_time_samples(
            sorting,
            start_sample=42,
            end_sample=(end_sample - start_sample) - 79,
            reference_to_start_sample=False,
        )
        print("final subset", sorting)
        print(
            "final subset",
            sorting.times_samples.min(),
            sorting.times_samples.max(),
        )
        print(
            "final subset",
            sorting.labels.min(),
            sorting.labels.max(),
        )

        with open(sorting_pkl, "wb") as jar:
            pickle.dump(sorting, jar)
    print(sorting)

    td = TemplateData.from_config(
        rec,
        sorting,
        template_config=template_config,
        save_folder=temps_dir,
        units_per_job=32,
        n_jobs=8,
        overwrite=True,
    )

    thresh = ThresholdAndFeaturize(
        rec,
        featurization_pipeline=None,
        channel_index=torch.from_numpy(
            waveform_util.make_channel_index(rec.get_channel_locations(), 200.0)
        ),
        spatial_dedup_channel_index=torch.from_numpy(
            waveform_util.make_channel_index(rec.get_channel_locations(), 150.0)
        ),
    )
    thresh.peel(
        detections_h5,
        n_jobs=8,
    )

    thresh4 = ThresholdAndFeaturize(
        rec,
        detection_threshold=4,
        featurization_pipeline=None,
        channel_index=torch.from_numpy(
            waveform_util.make_channel_index(rec.get_channel_locations(), 200.0)
        ),
        spatial_dedup_channel_index=torch.from_numpy(
            waveform_util.make_channel_index(rec.get_channel_locations(), 150.0)
        ),
    )
    thresh4.peel(
        detections4_h5,
        n_jobs=4,
    )

# %%
np.random.default_rng(0).choice(range(5), size=2)

# %%
rec.has_time_vector()

# %%
rec._recording_segments[0].t_start

# %% [markdown]
# # Experiment 1: Fully unsupervised individual fits

# %%
pid = pids[1]

# %%
rec_ppx_dir = data_dir / f"recppx{pid}"
temps_dir = templates_dir / f"temps{pid}"
sorting_pkl = data_dir / f"sorting{pid}.pkl"
detections_h5 = data_dir / f"detect{pid}.h5"
detections4_h5 = data_dir / f"det4ect{pid}.h5"

# %%
net_data_dir = data_dir / f"net{pid}"
net_data_dir.mkdir(exist_ok=True)

# %%
with open(sorting_pkl, "rb") as jar:
    ks_sorting = pickle.load(jar)

# %%
ks_sorting.__post_init__()

# %%
ks_sorting

# %%
kstd = TemplateData.from_npz(temps_dir / "template_data.npz")

# %%
kstd.templates.shape

# %%
mc = kstd.templates.ptp(1).argmax(1)
tr = kstd.templates[np.arange(len(mc)), :, mc]
tr.shape

# %%
# !echo {rec_ppx_dir}

# %%
# !echo {scratch_dir}

# %%
# !rsync -avP {rec_ppx_dir} {scratch_dir}/

# %%
# !ls -lah {scratch_dir}

# %%
rec0 = si.read_binary_folder(scratch_dir / rec_ppx_dir.name)

# %%
rec0.sample_index_to_time(0)

# %%
rec1 = rec0.save_to_memory()

# %%
rec0._recording_segments[0].t_start

# %%
rec1._recording_segments[0].t_start = rec0._recording_segments[0].t_start
rec1._kwargs["t_starts"] = [rec0._recording_segments[0].t_start]

# %%
rec1.sample_index_to_time(0)

# %%
rec = rec1

# %%
with h5py.File(detections_h5, "r") as h5:
    times = h5["times_samples"][()]
    channels = h5["channels"][()]

# %%
channel_index = waveform_util.make_channel_index(rec.get_channel_locations(), 200.0, fill_holes=True)
channel_index.shape

# %%
channel_index

# %%
np.unique(channel_index).size

# %%
channel_jitter_index = waveform_util.make_channel_index(
    rec.get_channel_locations(), 25.0
)

# %%
np.array_equal(rec0.get_times(), rec1.get_times())

# %%
splitrg = np.random.default_rng(0)
rec_seconds = np.unique(np.floor(rec.get_times()))
assignments = splitrg.choice(
    3, p=[0.5, 0.25, 0.25], size=rec_seconds.size
)
train_seconds = rec_seconds[assignments == 0]
val_seconds = rec_seconds[assignments == 1]
test_seconds = rec_seconds[assignments == 2]
train_seconds.shape, val_seconds.shape, test_seconds.shape

# %%
# subset the KS spike train to the test seconds
kstest = np.isin(np.floor(
    rec0.sample_index_to_time(ks_sorting.times_samples)), test_seconds)
testtimes = ks_sorting.times_samples[kstest]
testlabels = ks_sorting.labels[kstest]
testtimes.shape

# %%
nets = {
    "singlechan_conv": lambda fa="relu": decollider.ConvToLinearSingleChannelDecollider(
        final_activation=fa,
    ),
    "singlechan_convdeeper2": lambda fa="relu": decollider.ConvToLinearSingleChannelDecollider(
        kernel_lengths=(5, 9, 13, 17, 21),
        out_channels=(64, 64, 64, 64, 64),
        # hidden_linear_dims=(256, 256),
        final_activation=fa,
    ),
    "singlechan_convdeeper2wide": lambda fa="relu": decollider.ConvToLinearSingleChannelDecollider(
        kernel_lengths=(5, 9, 13, 17, 21),
        out_channels=(128, 128, 128, 128, 128),
        # hidden_linear_dims=(256, 256),
        final_activation=fa,
    ),
    # "singlechan_convdeepest": lambda fa="relu": decollider.ConvToLinearSingleChannelDecollider(
    #     kernel_lengths=(11, 11, 11, 11, 11, 11, 11, 11, 11),
    #     out_channels=(64, 64, 64, 64, 64, 64, 64, 64, 64),
    #     # hidden_linear_dims=(256, 256),
    #     final_activation=fa,
    # ),
    "singlechan_convdeeper2mlp": lambda fa="relu": decollider.ConvToLinearSingleChannelDecollider(
        kernel_lengths=(5, 9, 13, 17),
        out_channels=(64, 64, 64, 64),
        hidden_linear_dims=(256,),
        final_activation=fa,
    ),
    # "singlechan_convwider": lambda fa="relu": decollider.ConvToLinearSingleChannelDecollider(
    #     kernel_lengths=(5, 7, 11, 13),
    #     out_channels=(64, 64, 64, 64),
    #     final_activation=fa,
    # ),
    # "singlechan_mlp": lambda fa="relu": decollider.MLPSingleChannelDecollider(
    #     final_activation=fa,
    # ),
    "multichan_conv": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
        n_channels=channel_index.shape[1],
        final_activation=fa,
    ),
    # "multichan_wideconv": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
    #     n_channels=channel_index.shape[1],
    #     out_channels=(16, 32, 64),
    #     kernel_heights=(4, 4, 8),
    #     kernel_lengths=(5, 7, 11),
    #     final_activation=fa,
    # ),
    # "multichan_wideconv2": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
    #     n_channels=channel_index.shape[1],
    #     out_channels=(16, 64),
    #     kernel_heights=(4, 8),
    #     kernel_lengths=(5, 11),
    #     final_activation=fa,
    # ),
    # "multichan_deepconv": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
    #     n_channels=channel_index.shape[1],
    #     kernel_lengths=(5, 5, 11),
    #     kernel_heights=(4, 4, 8),
    #     hidden_linear_dims=(512, 512, 512),
    #     final_activation=fa,
    # ),
    # "multichan_deepconv2": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
    #     n_channels=channel_index.shape[1],
    #     kernel_lengths=(5, 9, 14),
    #     kernel_heights=(4, 8, 12),
    #     out_channels=(64, 64, 64),
    #     # hidden_linear_dims=(512, 512, 512),
    #     final_activation=fa,
    # ),
    # "multichan_deepconv2mlp": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
    #     n_channels=channel_index.shape[1],
    #     kernel_lengths=(5, 9, 14),
    #     kernel_heights=(4, 8, 12),
    #     out_channels=(64, 64, 64),
    #     hidden_linear_dims=(1024,),
    #     final_activation=fa,
    # ),
    # "multichan_deeperconv2": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
    #     n_channels=channel_index.shape[1],
    #     kernel_lengths=(5, 9, 13, 17),
    #     kernel_heights=(4, 8, 12, 16),
    #     out_channels=(64, 64, 64, 64),
    #     # hidden_linear_dims=(512, 512, 512),
    #     final_activation=fa,
    # ),
    # "multichan_deeperconv2mlp": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
    #     n_channels=channel_index.shape[1],
    #     kernel_lengths=(5, 9, 13, 17),
    #     kernel_heights=(4, 8, 12, 16),
    #     out_channels=(64, 64, 64, 64),
    #     hidden_linear_dims=(1024,),
    #     final_activation=fa,
    # ),
    # "multichan_deepestconv2": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
    #     n_channels=channel_index.shape[1],
    #     kernel_lengths=(5, 9, 13, 17, 21),
    #     kernel_heights=(4, 8, 12, 16, 4),
    #     out_channels=(64, 64, 64, 64, 64),
    #     # hidden_linear_dims=(512, 512, 512),
    #     final_activation=fa,
    # ),
    "multichan_bigconv": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
        n_channels=channel_index.shape[1],
        kernel_lengths=[23] * 5,
        kernel_heights=[8] * 5,
        out_channels=[64] * 5,
        hidden_linear_dims=(),
        # hidden_linear_dims=(512, 512, 512),
        final_activation=fa,
    ),
    # "multichan_bigconvwide": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
    #     n_channels=channel_index.shape[1],
    #     kernel_lengths=[23] * 5,
    #     kernel_heights=[8] * 5,
    #     out_channels=[128] * 5,
    #     hidden_linear_dims=(),
    #     # hidden_linear_dims=(512, 512, 512),
    #     final_activation=fa,
    # ),
    "multichan_deepestconv4": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
        n_channels=channel_index.shape[1],
        kernel_lengths=[5] * 13,
        kernel_heights=[4] * 13,
        out_channels=[64] * 13,
        hidden_linear_dims=(),
        # hidden_linear_dims=(512, 512, 512),
        final_activation=fa,
    ),
    "multichan_testmlp": lambda fa="relu": decollider.MLPMultiChannelDecollider(
        n_channels=channel_index.shape[1],
        hidden_sizes=(256,),
        # hidden_linear_dims=(512, 512, 512),
        final_activation=fa,
    ),
    "multichan_testmlpwide": lambda fa="relu": decollider.MLPMultiChannelDecollider(
        n_channels=channel_index.shape[1],
        hidden_sizes=(1024,),
        # hidden_linear_dims=(512, 512, 512),
        final_activation=fa,
    ),
    "multichan_testmlpwide2": lambda fa="relu": decollider.MLPMultiChannelDecollider(
        n_channels=channel_index.shape[1],
        hidden_sizes=(1024, 1024,),
        # hidden_linear_dims=(512, 512, 512),
        final_activation=fa,
    ),
    # "multichan_testmlpwider2": lambda fa="relu": decollider.MLPMultiChannelDecollider(
    #     n_channels=channel_index.shape[1],
    #     hidden_sizes=(2048, 2048,),
    #     # hidden_linear_dims=(512, 512, 512),
    #     final_activation=fa,
    # ),
    # "multichan_testmlpwide3": lambda fa="relu": decollider.MLPMultiChannelDecollider(
    #     n_channels=channel_index.shape[1],
    #     hidden_sizes=(1024, 1024, 1024),
    #     # hidden_linear_dims=(512, 512, 512),
    #     final_activation=fa,
    # ),
    "multichan_waveletmlp": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
        n_channels=channel_index.shape[1],
        kernel_lengths=[121],
        kernel_heights=[1],
        out_channels=[128],
        hidden_linear_dims=(),
        # hidden_linear_dims=(512, 512, 512),
        final_activation=fa,
    ),
    "multichan_waveletwidemlp": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
        n_channels=channel_index.shape[1],
        kernel_lengths=[121],
        kernel_heights=[1],
        out_channels=[256],
        hidden_linear_dims=(),
        # hidden_linear_dims=(512, 512, 512),
        final_activation=fa,
    ),
    "multichan_waveletdeepmlp2": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
        n_channels=channel_index.shape[1],
        kernel_lengths=[7, 115],
        kernel_heights=[1, 1],
        out_channels=[32, 128],
        hidden_linear_dims=(),
        # hidden_linear_dims=(512, 512, 512),
        final_activation=fa,
    ),
    "multichan_waveletdeepmlp": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
        n_channels=channel_index.shape[1],
        kernel_lengths=[3, 119],
        kernel_heights=[1, 1],
        out_channels=[32, 128],
        hidden_linear_dims=(),
        # hidden_linear_dims=(512, 512, 512),
        final_activation=fa,
    ),
    "multichan_waveletmlp1": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
        n_channels=channel_index.shape[1],
        kernel_lengths=[121],
        kernel_heights=[1],
        out_channels=[128],
        hidden_linear_dims=(1024,),
        # hidden_linear_dims=(512, 512, 512),
        final_activation=fa,
    ),
    # "multichan_deepestconv4wide": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
    #     n_channels=channel_index.shape[1],
    #     kernel_lengths=[5] * 13,
    #     kernel_heights=[4] * 13,
    #     out_channels=[128] * 13,
    #     hidden_linear_dims=(),
    #     # hidden_linear_dims=(512, 512, 512),
    #     final_activation=fa,
    # ),
    # "multichan_deepestconv2mlp": lambda fa="relu": decollider.ConvToLinearMultiChannelDecollider(
    #     n_channels=channel_index.shape[1],
    #     kernel_lengths=(5, 9, 13, 17, 21),
    #     kernel_heights=(4, 8, 12, 16, 4),
    #     out_channels=(64, 64, 64, 64, 64),
    #     hidden_linear_dims=(1024,),
    #     final_activation=fa,
    # ),
    # "multichan_mlp": lambda fa="relu": decollider.MLPMultiChannelDecollider(
    #     n_channels=channel_index.shape[1],
    #     final_activation=fa,
    # ),
    # "multichan_deepmlp": lambda fa="relu": decollider.MLPMultiChannelDecollider(
    #     n_channels=channel_index.shape[1],
    #     hidden_sizes=(2048, 512, 512, 512, 512),
    #     final_activation=fa,
    # ),
    # "multichan_bigmlp": lambda fa="relu": decollider.MLPMultiChannelDecollider(
    #     n_channels=channel_index.shape[1],
    #     hidden_sizes=(1024, 512, 512, 512),
    #     final_activation=fa,
    # ),
}


# %%
def train_and_save(
    model_name,
    net_factory,
    save_folder,
    recording,
    n2_alpha=1.0,
    detection_threshold=5,
    early_stop_decrease_epochs=5,
    noise_same_chans=False,
    final_activation="relu",
    opt="adam",
    weight_decay=0,
    device_index=0,
    irandom=0,
    seed=0,
):
    if (save_folder / "val_df.csv").exists():
        return

    device = torch.device(f"cuda:{device_index}")
    esde = str(early_stop_decrease_epochs)
    max_n_epochs = 500
    if not early_stop_decrease_epochs:
        max_n_epochs = 250
    elif early_stop_decrease_epochs < 0:
        max_n_epochs = abs(early_stop_decrease_epochs)
        early_stop_decrease_epochs = None
    net = net_factory(fa=final_activation)
    print(model_name)
    print(net)
    net.to(device)
    singlechan = "single" in model_name

    det_h5 = detections_h5 if detection_threshold == 5 else detections4_h5
    with h5py.File(det_h5, "r") as h5:
        times = h5["times_samples"][()]
        channels = h5["channels"][()]
    print(f"{times.shape=}")
    print(f"{times.min()=} {times.max()=}")
    print(f"{train_seconds.min()=}")
    print(f"{train_seconds.max()=}")
    print(f"{recording.sample_index_to_time(0)=}")
    print(f"{recording.sample_index_to_time(recording.get_num_samples() - 1)=}")
    print(f"{recording.sample_index_to_time(times).min()=}")
    print(f"{recording.sample_index_to_time(times).max()=}")

    train_mask = np.isin(
        np.floor(recording.sample_index_to_time(times)), train_seconds
    )
    val_mask = np.isin(
        np.floor(recording.sample_index_to_time(times)), val_seconds
    )
    print(f"{train_mask.sum()=} {val_mask.sum()=}")
    times_train = times[train_mask]
    channels_train = channels[train_mask]
    times_val = times[val_mask]
    channels_val = channels[val_mask]

    net, train_df, val_df = decollider_util.train_decollider(
        net,
        recordings=[recording],
        detection_times_train=[times_train],
        detection_channels_train=[channels_train],
        detection_times_val=[times_val],
        detection_channels_val=[channels_val],
        channel_index=None if singlechan else channel_index,
        channel_jitter_index=channel_index
        if singlechan
        else channel_jitter_index,
        noise_same_chans=noise_same_chans,
        noise2_alpha=n2_alpha,
        data_random_seed=seed,
        max_n_epochs=max_n_epochs,
        early_stop_decrease_epochs=early_stop_decrease_epochs,
        device=device,
        show_progress=True,
        opt_class=torch.optim.Adam if opt == "adam" else torch.optim.SGD,
        opt_kw=dict(weight_decay=weight_decay),
        val_every=50,
    )
    net.cpu()
    import gc; gc.collect(); torch.cuda.empty_cache()

    val_df["training_time_noval"] = np.cumsum(
        val_df.epoch_train_wall_dt_s + val_df.epoch_data_load_wall_dt_s
    )
    val_df["total_time"] = np.cumsum(
        val_df.epoch_train_wall_dt_s + val_df.epoch_data_load_wall_dt_s
        + val_df.val_metrics_wall_dt_s + val_df.val_data_load_wall_dt_s
    )
    val_df["model"] = model_name
    val_df["weight_decay"] = weight_decay
    val_df["opt"] = opt
    val_df["noise2_alpha"] = str(n2_alpha)
    val_df["early_stop_decrease_epochs"] = str(esde)
    val_df["final_activation"] = final_activation
    val_df["detection_threshold"] = str(detection_threshold)
    val_df["noise_same_chans"] = str(int(noise_same_chans))
    val_df["irandom"] = str(irandom)

    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)
    net.save(save_folder / "net.pt")
    train_df.to_csv(save_folder / "train_df.csv")
    val_df.to_csv(save_folder / "val_df.csv")


# %%
net_data_dir

# %%
1

# %%
jobs = []
transtable = str.maketrans(string.punctuation + string.whitespace, '_' * len(string.punctuation + string.whitespace))
seeder = np.random.SeedSequence()
for model_name, net_factory in nets.items():
    for opt in ("adam",):
        for noise_same_chans in (False,):
            # for early_stop_decrease_epochs in (None, 5, 25, -500):
            for early_stop_decrease_epochs in (-750,):
                for detection_threshold in (5,):
                    # for n2_alpha in (1.0, 0.5):
                    n2_alpha = 1.0
                    # for final_activation in ("relu", "tanh", "sigmoid"):
                    for final_activation in ("relu",):
                        for weight_decay in (0, 0.001):
                            for irandom in range(3):
                                job_kwargs = dict(
                                    opt=opt,
                                    noise_same_chans=noise_same_chans,
                                    early_stop_decrease_epochs=early_stop_decrease_epochs,
                                    detection_threshold=detection_threshold,
                                    n2_alpha=n2_alpha,
                                    final_activation=final_activation,
                                    weight_decay=weight_decay,
                                    irandom=irandom,
                                )
                                save_folder = net_data_dir / model_name
                                for k, v in job_kwargs.items():
                                    save_folder = save_folder / f"{k}={v}".translate(transtable)
                                job_args = (model_name, net_factory, save_folder)
                                job_kwargs["seed"] = seeder.spawn(1)[0]
                                save_folder.mkdir(exist_ok=True, parents=True)
                                if (save_folder / "val_df.csv").exists():
                                    continue

                                jobs.append((job_args, job_kwargs))

                                with open(save_folder / "args.pkl", "wb") as jar:
                                    pickle.dump({**job_kwargs, "model_name": model_name}, jar)


# %%
import gc; gc.collect()

# %%
len(jobs)


# %%
def runjob(iak):
    rank = multiprocessing_util.rank_init.rank
    i, rec, (args, kwargs) = iak
    print(rec)
    di = rank % torch.cuda.device_count()
    kwargs["device_index"] = di
    print(f"hi from job {i=} {rank=} {di=}")
    kwargs["recording"] = rec
    train_and_save(*args, **kwargs)
    print(f"bye from job {i=} {rank=} {di=}")


# %%
shufrg = np.random.default_rng(0)
order = shufrg.choice(len(jobs), size=len(jobs), replace=False)
jobs = [(i, rec1, jobs[o]) for i, o in enumerate(order)]

# %%
n_jobs, Pool, context, queue = multiprocessing_util.get_pool(
    4,
    cls=multiprocessing_util.CloudpicklePoolExecutor,
    with_rank_queue=True,
    # n_tasks=len(jobs),
    # max_tasks_per_child=8,
    context="spawn",
)
with Pool(
    n_jobs,
    mp_context=context,
    initializer=multiprocessing_util.rank_init,
    initargs=(queue,),
    # max_tasks_per_child=8,
) as pool:
    ndone = 0
    for res in pool.map(runjob, jobs):
        ndone += 1
        print(f"{ndone=} {res=} {len(jobs)=}")

# %%
1

# %%
valcsvs = list(net_data_dir.glob("**/val_df.csv"))
len(valcsvs)

# %%
traincsvs = list(net_data_dir.glob("**/train_df.csv"))
len(traincsvs)

# %%
net_data_dir

# %%
nets.keys()

# %%
mods = []
for traincsv in traincsvs:
    with open(traincsv.parent / "args.pkl", "rb") as jar:
        job_kwargs = pickle.load(jar)
    mods.append(job_kwargs["model_name"])
mods = list(set(mods))

# %%
leg = [{}, {}]
fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True)
for traincsv in traincsvs:
    traindf = pd.read_csv(traincsv)
    with open(traincsv.parent / "args.pkl", "rb") as jar:
        job_kwargs = pickle.load(jar)
    opt = job_kwargs.get("opt", "adam")
    mod = job_kwargs["model_name"]
    wd = job_kwargs["weight_decay"]
    assert wd in (0, 0.001)

#     if "single" not in job_kwargs["model_name"]:
#         continue

#     if "multi" not in job_kwargs["model_name"]:
#         continue
    if "multichan_bigconv" in mod:
        continue

    c = mods.index(mod)
    # ax = axes[0] if opt == "adam" else axes[1]
    i = int("single" in mod)
    j = int(wd > 0)

    l, = axes[j, i].plot(
        # traindf.batch_train_wall_dt_s[1000:].cumsum(),
        traindf.samples[1000:],
        traindf.loss[1000:].ewm(halflife=100).mean(),
        color=cc.glasbey[c],
        # ls=":" if opt != "adam" else "-",
    )
    # if f"{opt=} {mod=}" not in leg:
    #     leg[f"{opt=} {mod=}"] = l
    if mod not in leg[i]:
        leg[i][mod] = l
for ax in axes.flat:
    ax.loglog()
# plt.semilogy()
axes[0, 0].legend(handles=leg[0].values(), labels=leg[0].keys(), loc="upper right")
axes[0, 1].legend(handles=leg[1].values(), labels=leg[1].keys(), loc="upper right")

# %%
leg = [{}, {}]
fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True)
for valcsv in valcsvs:
    valdf = pd.read_csv(valcsv)
    with open(valcsv.parent / "args.pkl", "rb") as jar:
        job_kwargs = pickle.load(jar)
    opt = job_kwargs.get("opt", "adam")
    mod = job_kwargs["model_name"]
    wd = job_kwargs["weight_decay"]
    assert wd in (0, 0.001)

#     if "single" not in job_kwargs["model_name"]:
#         continue

#     if "multi" not in job_kwargs["model_name"]:
#         continue
    if "multichan_bigconv" in mod:
        continue

    c = mods.index(mod)
    # ax = axes[0] if opt == "adam" else axes[1]
    i = int("single" in mod)
    j = int(wd > 0)

    l, = axes[j, i].plot(
        valdf.epoch,
        valdf.val_loss.ewm(halflife=2).mean(),
        color=cc.glasbey[c],
    )
    # if f"{opt=} {mod=}" not in leg:
    #     leg[f"{opt=} {mod=}"] = l
    if mod not in leg[i]:
        leg[i][mod] = l
for ax in axes.flat:
    ax.loglog()
# plt.semilogy()
axes[0, 0].legend(handles=leg[0].values(), labels=leg[0].keys(), loc="upper right")
axes[0, 1].legend(handles=leg[1].values(), labels=leg[1].keys(), loc="upper right")

# %%
valdf = pd.concat([pd.read_csv(csv) for csv in valcsvs])

# %%
valdf.columns

# %%
grid = sns.relplot(
    valdf,
    x="epoch",
    y="val_loss",
    hue="model",
    # style="noise2_alpha",
    # style="final_activation",
    style="early_stop_decrease_epochs",
    # markers="early_stop_decrease_epochs",
    row="detection_threshold",
    # col="noise_same_chans",
    col="final_activation",
    kind="line",
    units="irandom",
    # errorbar=("pi", 100),
    # errorbar="se",
    estimator=None,
    palette=cc.glasbey_light,
)
grid.set(yscale="log", xscale="log")

# %%
channel_index


# %%
def test_model(
    net_pt,
    device_index=0,
):
    prevdn = net_pt.name == "prev_single_chan_denoiser"
    save_folder = net_pt.parent
    
    if (save_folder / "test_df.csv").exists():
        return

    with open(save_folder / "args.pkl", "rb") as jar:
        job_kwargs = pickle.load(jar)
    # if "weight_decay" not in job_kwargs:
    #     print("no weight_decay")
    #     return

    # if (save_folder / "test_df.csv").exists():
    #     return

    device = f"cuda:{device_index}"
    print("device", device)

    if prevdn:
        net = decollider_util.SCDAsDecollider()
        net.load()
    else:
        net = decollider.Decollider.load(save_folder / "net.pt")
    net = net.to(device)

    # with open(save_folder / "args.pkl", "rb") as jar:
    #     job_kwargs = pickle.load(jar)
    model_name = job_kwargs["model_name"]
    print(f"before")

    df = decollider_ibl_tests.test(
        net,
        rec,
        templates=kstd.templates,
        times=testtimes,
        labels=testlabels,
        spike_counts=kstd.spike_counts,
        single_channel="single" in model_name,
        n_channels_full=None,
        recording_channels_subset=None,
        max_count_per_unit=16,
        full_channel_index=channel_index,
        n2n_alpha=job_kwargs["n2_alpha"],
        n2n_samples=(1, 5, 10),
        random_seed=0,
        noise_same_channels=job_kwargs["noise_same_chans"],
        device=device,
    )
    print(f"after")
    net = net.cpu()
    df["model"] = model_name
    df["opt"] = job_kwargs["opt"]
    df["weight_decay"] = job_kwargs.get("weight_decay", 0)
    df["noise2_alpha"] = str(job_kwargs["n2_alpha"])
    df["early_stop_decrease_epochs"] = str(job_kwargs["early_stop_decrease_epochs"])
    df["detection_threshold"] = str(job_kwargs["detection_threshold"])
    df["noise_same_chans"] = str(int(job_kwargs["noise_same_chans"]))
    df["final_activation"] = job_kwargs["final_activation"]
    df["irandom"] = str(job_kwargs.get("irandom", -1))
    df.to_csv(save_folder / "test_df.csv")


# %%
testjobs = list(net_data_dir.glob("**/net.pt"))
# testjobs = []
transtable = str.maketrans(string.punctuation + string.whitespace, '_' * len(string.punctuation + string.whitespace))
model_name = "prev_single_chan_denoiser"
for noise_same_chans in (False,):
    # for early_stop_decrease_epochs in (None, 5, 25, -500):
    # for early_stop_decrease_epochs in (5, 25):
    early_stop_decrease_epochs = -1
        # for detection_threshold in (4, 5):
    detection_threshold = -1
            # for n2_alpha in (1.0, 0.5):
    n2_alpha = 1.0
    final_activation = "relu"
            # for final_activation in ("relu", "tanh", "sigmoid"):
        
    job_kwargs = dict(
        noise_same_chans=noise_same_chans,
        early_stop_decrease_epochs=early_stop_decrease_epochs,
        detection_threshold=detection_threshold,
        n2_alpha=n2_alpha,
        final_activation=final_activation,
        opt="idk",
    )
    save_folder = net_data_dir / model_name
    for k, v in job_kwargs.items():
        save_folder = save_folder / f"{k}={v}".translate(transtable)

    save_folder.mkdir(exist_ok=True, parents=True)
    with open(save_folder / "args.pkl", "wb") as jar:
        pickle.dump({**job_kwargs, "model_name": model_name}, jar)
    
    testjobs.append(save_folder / model_name)

# %%
len(testjobs)

# %%
shufrg = np.random.default_rng(0)
order = shufrg.choice(len(testjobs), size=len(testjobs), replace=False)
testjobs = [(i, testjobs[o]) for i, o in enumerate(order)]


# %%
def runtest(ia):
    rank = multiprocessing_util.rank_init.rank
    i, net_pt = ia
    di = rank % torch.cuda.device_count()
    # di = int(1 + rank % 3)
    # di=3
    print(f"hi from job {i=} {rank=} {di=}")
    test_model(net_pt, device_index=di)
    print(f"bye from job {i=} {rank=} {di=}")


# %%
n_jobs, Pool, context, queue = multiprocessing_util.get_pool(
    8,
    with_rank_queue=True,
    cls=multiprocessing_util.CloudpicklePoolExecutor,
)
with Pool(
    n_jobs,
    mp_context=context,
    initializer=multiprocessing_util.rank_init,
    initargs=(queue,),
) as pool:
    ndone = 0
    for res in pool.map(runtest, testjobs):
        ndone += 1
        print(f"{ndone=} {len(testjobs)=}")

# %%
net_data_dir

# %%
test_csvs = list(net_data_dir.glob("**/test_df.csv"))

# %%
pd.read_csv(test_csvs[0]).head()

# %%
pd.read_csv(test_csvs[-1]).head()


# %%
def quantizedline(ax, x, y, q=3, c="b", e="se"):
    xq = x // q
    d = np.unique(xq)
    m = np.zeros_like(d)
    s = np.zeros_like(d)
    
    for j, xx in enumerate(d):
        w = np.flatnonzero(xq == xx)
        m[j] = np.nanmean(y[w])
        if e == "sd":
            s[j] = np.nanstd(y[w])
        elif e == "se":
            s[j] = np.nanstd(y[w]) / np.sqrt(w.size)
        
    
    ax.fill_between(q * d, m - s, m + s, color=c, alpha=0.2)
    return ax.plot(q * d, m, c=c, lw=1)

# %%
leg = [{}, {}]
fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True, figsize=(8, 8))
for test_csv in test_csvs:
    test_df = pd.read_csv(test_csv)
    with open(test_csv.parent / "args.pkl", "rb") as jar:
        job_kwargs = pickle.load(jar)
    opt = job_kwargs.get("opt", "adam")
    mod = job_kwargs["model_name"]
    wd = job_kwargs.get("weight_decay", 0)
    assert wd in (0, 0.001)

#     if "single" not in job_kwargs["model_name"]:
#         continue

#     if "multi" not in job_kwargs["model_name"]:
#         continue
    if "multichan_bigconv" in mod:
        continue

    c = cc.glasbey[mods.index(mod)] if mod in mods else "k"
    # ax = axes[0] if opt == "adam" else axes[1]
    i = int("single" in mod)
    j = int(wd > 0)

    l, = quantizedline(
        ax=axes[j, i],
        x=test_df.template_maxptp.values,
        y=test_df.naive_l2err.values,
        c=c,
    )
    # if f"{opt=} {mod=}" not in leg:
    #     leg[f"{opt=} {mod=}"] = l
    if mod not in leg[i]:
        leg[i][mod] = l
# for ax in axes.flat:
#     ax.loglog()
# plt.semilogy()
axes[0, 0].legend(handles=leg[0].values(), labels=leg[0].keys(), loc="upper right")
axes[0, 1].legend(handles=leg[1].values(), labels=leg[1].keys(), loc="upper right")

# %%
leg = [{}, {}]
fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True, figsize=(8, 8))
for test_csv in test_csvs:
    test_df = pd.read_csv(test_csv)
    with open(test_csv.parent / "args.pkl", "rb") as jar:
        job_kwargs = pickle.load(jar)
    opt = job_kwargs.get("opt", "adam")
    mod = job_kwargs["model_name"]
    wd = job_kwargs.get("weight_decay", 0)
    assert wd in (0, 0.001)

#     if "single" not in job_kwargs["model_name"]:
#         continue

#     if "multi" not in job_kwargs["model_name"]:
#         continue
    if "multichan_bigconv" in mod:
        continue

    c = cc.glasbey[mods.index(mod)] if mod in mods else "k"
    # ax = axes[0] if opt == "adam" else axes[1]
    i = int("single" in mod)
    j = int(wd > 0)

    l, = quantizedline(
        ax=axes[j, i],
        x=test_df.template_maxptp.values,
        y=test_df["n2n_10samples_l2err"].values,
        c=c,
    )
    # if f"{opt=} {mod=}" not in leg:
    #     leg[f"{opt=} {mod=}"] = l
    if mod not in leg[i]:
        leg[i][mod] = l
# for ax in axes.flat:
#     ax.loglog()
# plt.semilogy()
axes[0, 0].legend(handles=leg[0].values(), labels=leg[0].keys(), loc="upper right")
axes[0, 1].legend(handles=leg[1].values(), labels=leg[1].keys(), loc="upper right")

# %%
leg = [{}, {}]
fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True, figsize=(8, 8))
for test_csv in test_csvs:
    test_df = pd.read_csv(test_csv)
    with open(test_csv.parent / "args.pkl", "rb") as jar:
        job_kwargs = pickle.load(jar)
    opt = job_kwargs.get("opt", "adam")
    mod = job_kwargs["model_name"]
    wd = job_kwargs.get("weight_decay", 0)
    assert wd in (0, 0.001)

#     if "single" not in job_kwargs["model_name"]:
#         continue

#     if "multi" not in job_kwargs["model_name"]:
#         continue
    if "multichan_bigconv" in mod:
        continue

    c = cc.glasbey[mods.index(mod)] if mod in mods else "k"
    # ax = axes[0] if opt == "adam" else axes[1]
    i = int("single" in mod)
    j = int(wd > 0)

    l, = quantizedline(
        ax=axes[j, i],
        x=test_df.template_maxptp.values,
        y=test_df["n2n_1sample_l2err"].values,
        c=c,
    )
    # if f"{opt=} {mod=}" not in leg:
    #     leg[f"{opt=} {mod=}"] = l
    if mod not in leg[i]:
        leg[i][mod] = l
# for ax in axes.flat:
#     ax.loglog()
# plt.semilogy()
axes[0, 0].legend(handles=leg[0].values(), labels=leg[0].keys(), loc="upper right")
axes[0, 1].legend(handles=leg[1].values(), labels=leg[1].keys(), loc="upper right")

# %%

# %%

# %%

# %%

# %%

# %%
testjobs = list(net_data_dir.glob("**/net.pt"))
# testjobs = []
transtable = str.maketrans(string.punctuation + string.whitespace, '_' * len(string.punctuation + string.whitespace))
model_name = "prev_single_chan_denoiser"
for noise_same_chans in (False,):
    # for early_stop_decrease_epochs in (None, 5, 25, -500):
    # for early_stop_decrease_epochs in (5, 25):
    early_stop_decrease_epochs = -1
        # for detection_threshold in (4, 5):
    detection_threshold = -1
            # for n2_alpha in (1.0, 0.5):
    n2_alpha = 1.0
    final_activation = "relu"
            # for final_activation in ("relu", "tanh", "sigmoid"):
        
    job_kwargs = dict(
        noise_same_chans=noise_same_chans,
        early_stop_decrease_epochs=early_stop_decrease_epochs,
        detection_threshold=detection_threshold,
        n2_alpha=n2_alpha,
        final_activation=final_activation,
        opt="idk",
    )
    save_folder = net_data_dir / model_name
    for k, v in job_kwargs.items():
        save_folder = save_folder / f"{k}={v}".translate(transtable)

    save_folder.mkdir(exist_ok=True, parents=True)
    with open(save_folder / "args.pkl", "wb") as jar:
        pickle.dump({**job_kwargs, "model_name": model_name}, jar)
    
    testjobs.append(save_folder / model_name)

# %%
bestnets = {}
pick = ["singlechan_conv", "multichan_waveletmlp", "multichan_waveletdeepmlp2", "prev_single_chan_denoiser"]
for job_kwargs_pkl in net_data_dir.glob("**/args.pkl"):
    save_folder = job_kwargs_pkl.parent
    with open(save_folder / "args.pkl", "rb") as jar:
        job_kwargs = pickle.load(jar)
    mod = job_kwargs["model_name"]
    prevdn = mod == "prev_single_chan_denoiser"
    if mod in pick and mod not in bestnets:
        if prevdn:
            net = decollider_util.SCDAsDecollider()
            net.load()
        else:
            net = decollider.Decollider.load(save_folder / "net.pt")
        job_kwargs["net"] = net
        bestnets[mod] = job_kwargs

# %%
bestnets

# %%
1

# %%
bestwfs = {}
seq = np.random.SeedSequence(100)
rs = seq.spawn(1)[0]
for model_name, job_kwargs in bestnets.items():
    job_kwargs["net"].to("cuda:3")
    _, bestwfs[model_name] = decollider_ibl_tests.test(
        job_kwargs["net"],
        rec,
        templates=kstd.templates,
        times=testtimes,
        labels=testlabels,
        spike_counts=kstd.spike_counts,
        single_channel="single" in model_name,
        n_channels_full=None,
        recording_channels_subset=None,
        max_count_per_unit=1,
        full_channel_index=channel_index,
        n2n_alpha=job_kwargs["n2_alpha"],
        n2n_samples=(1, 100, 1000),
        random_seed=rs,
        noise_same_channels=job_kwargs["noise_same_chans"],
        return_waveforms=True,
        device="cuda:3",
    )
    job_kwargs["net"].cpu()

# %%
1

# %%
figdir = net_data_dir.parent / "netfigs"
figdir.mkdir(exist_ok=True)
figdir

# %%
with open(figdir / "bestwfs.pkl", "wb") as jar:
    pickle.dump(bestwfs, jar)

# %%
with open(figdir / "bestwfs.pkl", "rb") as jar:
    bestwfs = pickle.load(jar)

# %%
1


# %%
def figjob(i):
    fig = decollider_ibl_tests.comparison_vis(
        bestwfs,
        i,
        fig_width=7,
        filt=lambda x: 'med' in x and not x.endswith('_100')
    )
    fig.savefig(figdir / f"unit{i:03d}.pdf")
    plt.close(fig)


# %%
import gc; gc.collect()

# %%
bestwfs['multichan_waveletdeepmlp2'].keys()

# %%
n_jobs, Pool, context = multiprocessing_util.get_pool(
    0, cls=multiprocessing_util.CloudpicklePoolExecutor
)
with Pool(n_jobs, mp_context=context) as pool:
    ndone = 0
    for res in tqdm(pool.map(figjob, range(10)), total=10):
        # ndone += 1
        # print(f"{ndone=} {len(testjobs)=}")
        pass

# %%

# %%
g = sns.FacetGrid(
    meandf,
    # x="model",
    # y="naive_l2err",
    col="early_stop_decrease_epochs",
    row="dtfa",
    # style="final_activation",
    # hue="model",
    # size="irandom",
    # x_jitter=0.1,
)
g.map_dataframe(
    sns.stripplot,
    x="model",
    # y="naive_l2err",
    y="n2n_20samples_l2err",
    # style="final_activation",
    hue="model",
    # size="irandom",
    palette=cc.glasbey_light,
)

# %%
# pick net with best average test L2
