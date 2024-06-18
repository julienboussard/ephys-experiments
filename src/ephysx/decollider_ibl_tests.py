import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from . import decollider_util as dcu
from dartsort.util import spikeio
from dartsort.vis import geomplot
from scipy.spatial import KDTree
import h5py
from pathlib import Path


@torch.no_grad()
def test(
    net,
    recording,
    templates,
    times,
    labels,
    spike_counts=None,
    single_channel=False,
    n_channels_full=None,
    recording_channels_subset=None,
    max_count_per_unit=500,
    full_channel_index=None,
    n2n_alpha=1.0,
    n2n_samples=(1, 3),
    random_seed=0,
    noise_same_channels=False,
    trough_offset_samples=42,
    spike_length_samples=121,
    return_waveforms=False,
    device=None,
):
    rg = np.random.default_rng(random_seed)

    if n_channels_full is None:
        assert recording_channels_subset is None
        n_channels_full = recording.get_num_channels()
    if full_channel_index is None:
        full_channel_index = np.arange(n_channels_full)[:, None]
    if recording_channels_subset is None:
        recording_channels_subset = np.arange(n_channels_full)
    recording_channel_index = dcu.subset_recording_channel_index(
        full_channel_index, recording_channels_subset
    )

    templates_ = np.full(
        (templates.shape[0], n_channels_full + 1, spike_length_samples),
        np.nan,
        dtype=templates.dtype,
    )
    templates_[:, recording_channels_subset] = templates.transpose(0, 2, 1)
    templates = templates_
    maxchans = np.nanargmax(templates.ptp(2), 1)

    # pick spikes
    which = []
    units = np.unique(labels)
    units = units[units > 0]
    for u in units:
        in_unit = np.flatnonzero(labels == u)
        in_unit = in_unit[
            (times[in_unit] >= trough_offset_samples)
            & (
                times[in_unit]
                < recording.get_num_samples()
                - templates.shape[2]
                + trough_offset_samples
            )
        ]
        if in_unit.size > max_count_per_unit:
            in_unit = rg.choice(in_unit, size=max_count_per_unit, replace=False)
        which.append(in_unit)
    print(f"{len(which)=}")
    which = np.concatenate(which)
    which.sort()
    print(f"bq {which.size=} {(which.size/len(templates))=}")

    times = times[which]
    labels = labels[which]
    rec_channels = np.searchsorted(recording_channels_subset, maxchans)[labels]
    full_channels = maxchans[labels]

    # template waveforms
    waveform_channels = full_channel_index[full_channels][:, :, None]
    gt_waveforms = templates[
        labels[:, None, None],
        waveform_channels,
        np.arange(templates.shape[2])[None, None, :],
    ]

    # noisy waveforms
    noisy_waveforms = spikeio.read_waveforms_channel_index(
        recording,
        times,
        recording_channel_index,
        rec_channels,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        fill_value=np.nan,
    )
    print(f"a {noisy_waveforms.shape=}")
    noisy_waveforms = noisy_waveforms.transpose(0, 2, 1)
    print(f"b {noisy_waveforms.shape=}")

    # handle missing channels
    channel_mask = np.isfinite(noisy_waveforms[:, :, 0])
    mask = channel_mask[..., None].astype(noisy_waveforms.dtype)
    gt_waveforms = np.nan_to_num(gt_waveforms)
    noisy_waveforms = np.nan_to_num(noisy_waveforms)

    # we'll build up a dataframe with lots of info
    df = pd.DataFrame()

    # template-related covariates
    df["template_maxptp"] = np.nanmax(templates.ptp(2), axis=1)[labels]
    df["template_l2norm"] = np.sqrt(
        np.nansum(np.square(templates), axis=(1, 2))
    )[labels]
    if spike_counts is not None:
        df["unit_spike_count"] = spike_counts[labels]

    # noise-related covariates
    df["noise1_l2norm"] = np.linalg.norm(
        mask * (noisy_waveforms - gt_waveforms), axis=(1, 2)
    )

    # unsupervised prediction
    naive_pred = dcu.batched_infer(
        net,
        noisy_waveforms,
        channel_masks=channel_mask,
        device=device,
        show_progress=True,
    )
    naive_diff = mask * (naive_pred - gt_waveforms)
    df["naive_l2err"] = np.linalg.norm(naive_diff, axis=(1, 2))
    df["naive_maxerr"] = np.abs(naive_diff).max(axis=(1, 2))
    if return_waveforms:
        waveforms = dict(
            gt_waveforms=gt_waveforms,
            noisy_waveforms=noisy_waveforms,
            naive_pred=naive_pred,
        )
    del naive_pred, naive_diff

    # nodel prediction
    # pure noise waveforms, to assist in making noisier waveforms
    noise2 = []
    for _ in range(max(n2n_samples)):
        n2 = dcu.load_noise_singlerec(
            recording,
            channels=rec_channels if noise_same_channels else None,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            channel_index=recording_channel_index,
            n=which.size,
            alpha=n2n_alpha,
            rg=rg,
            to_torch=False,
        )
        mymask = channel_mask & np.isfinite(n2[:, :, 0])
        n2 = dcu.batched_n2n_infer(
            net,
            np.nan_to_num(noisy_waveforms + n2),
            alpha=n2n_alpha,
            channel_masks=mymask,
            device=device,
            show_progress=True,
        )
        n2[~mymask] = np.nan
        noise2.append(n2)
    print(f"{len(noise2)=} {noise2[0].shape=}")

    for k in n2n_samples:
        model_pred = np.nanmean(noise2[:k], axis=0)
        model_diff = mask * (model_pred - gt_waveforms)
        print(f"{model_pred.shape=} {model_diff.shape=}")
        model_l2 = np.linalg.norm(model_diff, axis=(1, 2))
        df[f"model_{k}sample{'s'*(k>1)}_l2err"] = model_l2
        df[f"model_{k}sample{'s'*(k>1)}_maxerr"] = np.abs(model_diff).max(
            axis=(1, 2)
        )
        if return_waveforms:
            waveforms[f"model_pred_{k}"] = model_pred
        model_diff = model_pred - gt_waveforms
        df[f"model_{k}sample{'s'*(k>1)}_l2err"] = np.linalg.norm(
            model_diff, axis=(1, 2)
        )
        df[f"model_{k}sample{'s'*(k>1)}_maxerr"] = np.abs(model_diff).max(
            axis=(1, 2)
        )

        # model_pred = np.nanmedian(noise2[:k], axis=0)
        # model_diff = mask * (model_pred - gt_waveforms)
        # print(f"{model_pred.shape=} {model_diff.shape=}")
        # model_l2 = np.linalg.norm(model_diff, axis=(1, 2))
        # df[f"modelmed_{k}sample{'s'*(k>1)}_l2err"] = model_l2
        # df[f"modelmed_{k}sample{'s'*(k>1)}_maxerr"] = np.abs(model_diff).max(
        #     axis=(1, 2)
        # )
        # if return_waveforms:
        #     waveforms[f"modelmed_pred_{k}"] = model_pred
        # model_diff = model_pred - gt_waveforms
        # df[f"modelmed_{k}sample{'s'*(k>1)}_l2err"] = np.linalg.norm(
        #     model_diff, axis=(1, 2)
        # )
        # df[f"modelmed_{k}sample{'s'*(k>1)}_maxerr"] = np.abs(model_diff).max(
        #     axis=(1, 2)
        # )
        # del model_pred, model_diff
        print(f"{model_l2.min()=} {model_l2.max()=}")

    if return_waveforms:
        return df, waveforms
    return df


def knn_ciline(ax, x, y, dx=0.1, k=5, alpha=0.1, bandwidth=1, doci=True, **plot_kw):
    domain = np.arange(x.min(), x.max(), dx)
    kdt = KDTree(x[:, None])
    d, i = kdt.query(domain[:, None], k=k)
    weights = np.exp(-(1 / (2 * bandwidth)) * (d * d))
    yi = y[i]
    assert yi.shape[0] == domain.size
    mean = (yi * weights).sum(1) / weights.sum(1)
    std = np.sqrt(np.sum(weights * np.square(yi - mean[:, None]), 1) / weights.sum(1))
    if doci:
        ax.fill_between(domain, mean - std, mean + std, alpha=alpha, color=plot_kw["color"])
    l = ax.plot(domain, mean, **plot_kw)
    return l


def comparison_vis(
    names2wfdicts,
    spike_index,
    top_height=3,
    height=3,
    prev_color="r",
    prev_name="prev_single_chan_denoiser",
    scatter_kw=dict(s=2, lw=0, alpha=0.5),
    fig_width=8.5,
    filt=lambda x: True,
):
    names = list(names2wfdicts.keys())
    assert all(
        np.array_equal(
            list(names2wfdicts[names[0]].keys()),
            list(names2wfdicts[name].keys()),
        )
        for name in names
    )
    assert all(
        np.array_equal(
            names2wfdicts[name]["gt_waveforms"][spike_index],
            names2wfdicts[names[0]]["gt_waveforms"][spike_index],
        )
        for name in names
    )
    gtwf = names2wfdicts[names[0]]["gt_waveforms"][spike_index].T
    noisywf = names2wfdicts[names[0]]["noisy_waveforms"][spike_index].T

    colors = {}
    linestyles = {}
    j = 0
    for name in names:
        if name == prev_name:
            colors[name] = prev_color
            linestyles[name] = "--"
        else:
            colors[name] = cc.glasbey_light[j]
            linestyles[name] = "-"
            j += 1

    # check largest model k
    modelk = max(
        int(k.split("_")[-1])
        for k in names2wfdicts[names[0]].keys()
        if k.startswith("model") and filt(k)
    )

    npreds = len(names2wfdicts[names[0]]) - 2
    nrows = len(names)
    ncols = 1 + 2 * npreds
    nchans = gtwf.shape[1]
    aspect = (top_height + height * nrows) / ncols

    fig = plt.figure(
        figsize=(fig_width, fig_width * aspect),
        layout="constrained",
    )
    top, bottom = fig.subfigures(
        nrows=2, height_ratios=[top_height, height * nrows], hspace=0.05
    )
    for f in (fig, top, bottom):
        f.patch.set_visible(False)
    naive_l2_ax, naive_max_ax, model_l2_ax, model_max_ax = top.subplots(
        ncols=4, sharey=True
    )
    axs = bottom.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey=True,
        gridspec_kw=dict(wspace=0, hspace=0),
    )

    geom = np.c_[np.zeros(nchans), np.arange(float(nchans))]
    max_abs_amp = np.abs(gtwf).max()

    for ax in axs.flat:
        plt.setp(ax.spines.values(), visible=False)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axs[:, :-1:2].flat:
        geomplot(
            noisywf,
            geom=geom,
            ax=ax,
            max_abs_amp=max_abs_amp,
            lw=1,
            color="gray",
        )
        geomplot(
            gtwf, geom=geom, ax=ax, max_abs_amp=max_abs_amp, lw=1, color="k"
        )
    for ax in axs[:, -1]:
        geomplot(
            gtwf, geom=geom, ax=ax, max_abs_amp=max_abs_amp, lw=1, color="k"
        )
        geomplot(
            noisywf,
            geom=geom,
            ax=ax,
            max_abs_amp=max_abs_amp,
            lw=1,
            color="gray",
        )

    names = []
    handles = []
    for j, (name, wf_dict) in enumerate(names2wfdicts.items()):
        naive_pred = wf_dict["naive_pred"][spike_index].T
        modelkeys = [k for k in wf_dict.keys() if k.startswith("model") and filt(k)]
        assert modelkeys[-1].endswith(f"_{modelk}")

        row_axs = axs[j]
        row_axs[0].set_ylabel(name)
        if not j:
            row_axs[0].set_title("naive", fontsize=8)
            row_axs[1].set_title("< gt resid", fontsize=8)
            for n, k in enumerate(modelkeys):
                thek = int(k.split("_")[-1])
                row_axs[2 + 2 * n].set_title(f"model-{thek}", fontsize=8)
                row_axs[3 + 2 * n].set_title(f"< gt resid", fontsize=8)

        color = colors[name]
        ls = linestyles[name]
        gpkw = dict(color=color, max_abs_amp=max_abs_amp, lw=1, geom=geom, linestyle=ls)

        naive_diff = gtwf - naive_pred
        geomplot(naive_pred, ax=row_axs[0], **gpkw)
        geomplot(naive_diff, ax=row_axs[1], **gpkw)

        for n, k in enumerate(modelkeys):
            model_pred = wf_dict[k][spike_index].T
            model_diff = gtwf - model_pred
            geomplot(model_pred, ax=row_axs[2 + 2 * n], **gpkw)
            geomplot(model_diff, ax=row_axs[3 + 2 * n], **gpkw)

        gt_ptp = gtwf.ptp(0)
        naive_l2 = np.linalg.norm(naive_diff, axis=0)
        naive_max = np.abs(naive_diff).max(0)
        model_l2 = np.linalg.norm(model_diff, axis=0)
        model_max = np.abs(model_diff).max(0)
        naive_l2_ax.scatter(gt_ptp, naive_l2, color=color, **scatter_kw)
        naive_max_ax.scatter(gt_ptp, naive_max, color=color, **scatter_kw)
        model_l2_ax.scatter(gt_ptp, model_l2, color=color, **scatter_kw)
        model_max_ax.scatter(gt_ptp, model_max, color=color, **scatter_kw)
        l = knn_ciline(naive_l2_ax, gt_ptp, naive_l2, color=color, ls=ls, doci=False)
        knn_ciline(naive_max_ax, gt_ptp, naive_max, color=color, ls=ls, doci=False)
        knn_ciline(model_l2_ax, gt_ptp, model_l2, color=color, ls=ls, doci=False)
        knn_ciline(model_max_ax, gt_ptp, model_max, color=color, ls=ls, doci=False)
        
        names.append(name)
        handles.append(l[0])
    
    top.legend(
        handles,
        names,
        loc="upper center",
        bbox_to_anchor=(0, -0.2, 1, 0.2),
        frameon=False,
        ncols=4,
    )
        

    naive_l2_ax.set_title("naive l2 err")
    naive_max_ax.set_title("naive max err")
    model_l2_ax.set_title(f"model {modelk} samps l2 err")
    model_max_ax.set_title(f"model {modelk} max err")
    for ax in (naive_l2_ax, naive_max_ax, model_l2_ax, model_max_ax):
        ax.set_xlabel("per-channel gt ptp")

    return fig


def train_and_save(
    model_name,
    net_factory,
    save_folder,
    recording,
    det_h5,
    train_seconds,
    val_seconds,
    channel_index,
    spike_sample_p=None,
    singlechan=True,
    channel_jitter_index=None,
    n2_alpha=1.0,
    detection_threshold=5,
    noise_same_chans=False,
    early_stop_cur_epochs=20,
    final_activation="relu",
    opt="adam",
    weight_decay=0,
    device_index=0,
    irandom=0,
    val_every=1,
    seed=0,
):
    save_folder.mkdir(exist_ok=True)
    if (save_folder / "val_df.csv").exists():
        return

    device = torch.device(f"cuda:{device_index}")
    print(model_name)
    net = net_factory(final_activation=final_activation)
    net.to(device)
    print(net)

    with h5py.File(det_h5, "r") as h5:
        times = h5["times_samples"][()]
        channels = h5["channels"][()]

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
    if spike_sample_p is not None:
        sample_p_train = [spike_sample_p[train_mask]]
        sample_p_val = [spike_sample_p[val_mask]]

    net, train_df, val_df = dcu.train_decollider(
        net,
        recordings=[recording],
        detection_times_train=[times_train],
        detection_channels_train=[channels_train],
        detection_p_train=sample_p_train,
        detection_times_val=[times_val],
        detection_channels_val=[channels_val],
        detection_p_val=sample_p_val,
        channel_index=None if singlechan else channel_index,
        channel_jitter_index=channel_index
        if singlechan
        else channel_jitter_index,
        noise_same_chans=noise_same_chans,
        early_stop_cur_epochs=early_stop_cur_epochs,
        noise2_alpha=n2_alpha,
        data_random_seed=seed,
        device=device,
        show_progress=True,
        opt_class=torch.optim.Adam if opt == "adam" else torch.optim.SGD,
        opt_kw=dict(weight_decay=weight_decay),
        val_every=val_every,
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
    val_df["early_stop"] = True
    val_df["final_activation"] = final_activation
    val_df["detection_threshold"] = str(detection_threshold)
    val_df["noise_same_chans"] = str(int(noise_same_chans))
    val_df["irandom"] = str(irandom)

    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)
    torch.save(net, save_folder / "net.pt")
    train_df.to_csv(save_folder / "train_df.csv")
    val_df.to_csv(save_folder / "val_df.csv")
