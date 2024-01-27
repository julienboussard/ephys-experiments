import numpy as np
import pandas as pd
from dartsort.util import decollider_util as dcu
from dartsort.util import spikeio
import torch


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
    noisy_waveforms = noisy_waveforms.transpose(0, 2, 1)

    # pure noise waveforms, to assist in making noisier waveforms
    noise2 = [
        dcu.load_noise_singlerec(
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
        for _ in range(max(n2n_samples))
    ]

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
    print(f"{df.noise1_l2norm.min()=} {df.noise1_l2norm.max()=}")

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
    del naive_pred, naive_diff
    print(f"{df.naive_l2err.min()=} {df.naive_l2err.max()=}")

    # n2n prediction
    noise2 = [
        dcu.batched_n2n_infer(
            net,
            noisy_waveforms + np.nan_to_num(n2),
            alpha=n2n_alpha,
            channel_masks=channel_mask,
            device=device,
            show_progress=True,
        )
        for n2 in noise2
    ]
    for k in n2n_samples:
        n2n_pred = np.mean(noise2[:k], axis=0)
        n2n_diff = mask * (n2n_pred - gt_waveforms)
        n2n_l2 = np.linalg.norm(n2n_diff, axis=(1, 2))
        df[f"n2n_{k}sample{'s'*(k>1)}_l2err"] = n2n_l2
        df[f"n2n_{k}sample{'s'*(k>1)}_maxerr"] = np.abs(n2n_diff).max(
            axis=(1, 2)
        )
        del n2n_pred, n2n_diff
        print(f"{n2n_l2.min()=} {n2n_l2.max()=}")

    return df
