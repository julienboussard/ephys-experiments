import numpy as np
import pandas as pd
from dartsort.util import decollider_util as dcu
from dartsort.util import spikeio


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

    templates = np.full(
        (templates.shape[0], n_channels_full + 1, spike_length_samples),
        np.nan,
        dtype=templates.dtype,
    )
    templates[:, recording_channels_subset] = templates.transpose(0, 2, 1)
    maxchans = np.nanargmax(templates.ptp(2), 1)

    # pick spikes
    which = []
    units = np.unique(labels)
    units = units[units > 0]
    for u in units:
        in_unit = np.flatnonzero(units == u)
        if in_unit.size > max_count_per_unit:
            in_unit = rg.choice(in_unit, size=max_count_per_unit, replace=False)
        which.append(in_unit)
    which = np.concatenate(which)
    which.sort()

    times = times[which]
    full_channels = maxchans[which]
    rec_channels = np.searchsorted(recording_channels_subset, maxchans)[which]
    labels = labels[which]

    # template waveforms
    waveform_channels = full_channel_index[full_channels][:, :, None]
    gt_waveforms = templates[
        labels[:, None, None],
        waveform_channels,
        np.arange(templates.shape[1])[None, None, :],
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
        for _ in range(n2n_samples)
    ]

    # we'll build up a dataframe with lots of info
    df = pd.DataFrame()

    # template-related covariates
    df["template_maxptp"] = templates.ptp(2).max(1)[labels]
    df["template_norm"] = np.nansum(np.square(templates), axis=(1, 2))[labels]
    if spike_counts is not None:
        df["unit_spike_count"] = spike_counts[labels]

    # noise-related covariates
    df["wf_template_diff_norm"] = np.linalg.norm(
        noisy_waveforms - gt_waveforms, axis=(1, 2)
    )

    # unsupervised prediction
    naive_pred = dcu.batched_infer(net, noisy_waveforms)
    naive_diff = naive_pred - gt_waveforms
    df["naive_l2err"] = np.linalg.norm(naive_diff, axis=(1, 2))
    df["naive_maxerr"] = np.abs(naive_diff).max(axis=(1, 2))
    if return_waveforms:
        waveforms = dict(
            gt_waveforms=gt_waveforms,
            noisy_waveforms=noisy_waveforms,
            naive_pred=naive_pred,
        )
    del naive_pred, naive_diff

    # n2n prediction
    for k in n2n_samples:
        n2n_pred = np.mean(
            [
                dcu.batched_n2n_infer(net, noisy_waveforms + n2, alpha=n2n_alpha)
                for n2 in noise2
            ],
            dim=0,
        )
        if return_waveforms:
            waveforms[f"n2n_pred_{k}"] = n2n_pred
        n2n_diff = n2n_pred - gt_waveforms
        df[f"n2n_{k}sample{'s'*(k>1)}_l2err"] = np.linalg.norm(n2n_diff, axis=(1, 2))
        df[f"n2n_{k}sample{'s'*(k>1)}_maxerr"] = np.abs(n2n_diff).max(axis=(1, 2))
        del n2n_pred, n2n_diff

    if return_waveforms:
        return df, waveforms
    return df
