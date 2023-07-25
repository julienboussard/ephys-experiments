# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python [conda env:a]
#     language: python
#     name: conda-env-a-py
# ---

# %%
import numpy as np
import spikeinterface.full as si
from spikeinterface.sortingcomponents import peak_detection
import matplotlib.pyplot as plt
import torch

# %%
x = np.array([0, 0, 1, 1, 1, 2, 1, 2, 3, 4, 5])

# %%
u, i = np.unique(x, return_index=True)

# %%
u

# %%
i

# %%
x[~np.isin(np.arange(len(x)), i)] = -1
x

# %%

# %%

# %%

# %%
from spike_psvae import detect, denoise

# %%
fs = 30000
traces = np.random.default_rng(0).normal(size=10 * fs)

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.plot([0,1], [1000, 1001])
ax.set_xticks([0, 1])
ax.set_yticks([1000, 1001])
ax.set_xlabel("x", labelpad=-8)
ax.set_ylabel("y", labelpad=-18)
fig.savefig("/Users/charlie/Screenshots/labels.png", bbox_inches="tight")

# %%
rec = si.NumpyRecording([traces[:, None]], fs)
rec

# %%
rec =  si.bandpass_filter(rec, freq_min=300., freq_max=6000., dtype='float32')

# %%
rec = si.zscore(rec)

# %%
plt.hist(rec.get_traces().ravel())

# %%
peaks = peak_detection.detect_peaks(rec, peak_sign="neg", detect_threshold=2.5)

# %%
t = peaks["sample_ind"]

# %%
zz = rec.get_traces().squeeze()
zz.shape

# %%
dt = np.arange(-30, 30)

# %%
wfs = zz[t[(t > 40) & (t < t.max() - 60)][:, None] + dt[None, :]]

# %%
wfs.shape

# %%
fig, (aa, ab) = plt.subplots(ncols=2, figsize=(10, 6))
aa.plot(dt, wfs.T, color="k", lw=0.5, alpha=0.1);
aa.set_title("1333 peaks at threshold 2.5")
aa.set_xlabel("time relative to trough (samples)")
aa.set_ylabel('"amplitude"')

ab.hist(np.diff(t) / 3000)
ab.set_title("ISI histogram of these 1333 peaks")
ab.set_xlabel("isi (ms)")
ab.set_ylabel("bin count")

fig.suptitle("troughs detected by thresholding in 10s of white noise which was bandpass filtered at (300, 6000)Hz and zscored again");

# %%
dn = denoise.SingleChanDenoiser()
dnd = detect.DenoiserDetect(dn)

# %%
detect.denoiser_detect_dedup(zz[:, None], 1, dnd)#, channel_index=np.array([[0]]))

# %%
dndrec = dnd.forward_recording(torch.tensor(zz[:, None], dtype=torch.float, requires_grad=False)).detach().numpy().squeeze()

# %%
plt.title("denoised PTP for traces extracted at every possible offset of this fake recording")
plt.hist(dndrec);
plt.xlabel("denoised PTP")
plt.ylabel("bin count")

# %%
