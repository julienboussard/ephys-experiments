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
import numpy as np
import spikeinterface.full as si

# %%
# mouse AP
mouse_ap_snip = np.fromfile(
    "/Users/charlie/data/spike_sorting_paper_figs/subtraction_fig/subtraction_fig_data/zad5/traces_cached_seg0.raw",
    dtype=np.float32,
    count=30000*384,
).reshape(30000, 384)
np.save("/Users/charlie/data/dredgefigs/mouse_ap_snip_zad05.npy", mouse_ap_snip)

# %%
# human LFP
cutoff_um = None
# start and end times in seconds
t_start = 300.0
t_end = 865.0

# preprocessing, again
lfprec_interp = si.read_spikeglx("/Users/charlie/data/NeuropixelsHumanData/Pt02/raw/", stream_id="imec0.lf")
lfprec_interp = si.astype(lfprec_interp, np.float32)

if cutoff_um is not None:
    geom = lfprec_interp.get_channel_locations()
    lfprec_interp = lfprec_interp.remove_channels(lfprec_interp.channel_ids[geom[:, 1] > cutoff_um])

# is 1000kHz too low?
# a highpass filter would probably be OK too.
lfprec_interp = si.bandpass_filter(
    lfprec_interp,
    freq_min=0.5,
    freq_max=1000,
    margin_ms=1000,
    filter_order=3,
    dtype="float32",
    add_reflect_padding=True,
)

bad_chans, labels = si.detect_bad_channels(lfprec_interp, psd_hf_threshold=1.4, num_random_chunks=100, seed=0)
print("Found bad channels", bad_chans)

# [!!!] a change from above
# here I'm choosing to interpolate the bad channels, rather than remove them.
# so, they are being filled in with some kriging interpolator. this seemed to
# make sense for the interpolation step, since maybe we'd like to keep the signal
# on all of the channels rather than having holes?
lfprec_interp = si.interpolate_bad_channels(lfprec_interp, bad_chans)

# correct for ADC sample shifts
lfprec_interp = si.phase_shift(lfprec_interp)

# common median reference
lfprec_interp = si.common_reference(lfprec_interp)

# temporal slice
lfprec_interp = lfprec_interp.frame_slice(
    start_frame=int(t_start * lfprec_interp.sampling_frequency),
    end_frame=int(t_end * lfprec_interp.sampling_frequency),
)

# %%
np.save("/Users/charlie/data/dredgefigs/human_lfp_snip_Pt02.npy", lfprec_interp.get_traces(0, 0, 5 * 2500))

# %%
