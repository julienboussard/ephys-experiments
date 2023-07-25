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
import numpy as np
from spike_psvae import waveform_utils, spikeio
import spikeinterface.full as si
from dartsort.peel.grab import GrabAndFeaturize
from dartsort import transform

# %%
from tqdm.auto import tqdm, trange

# %%
import torch

# %%
rec = si.read_binary_folder("/Users/charlie/data/spike_sorting_paper_figs/subtraction_fig/subtraction_fig_data/zad5/")
rec

# %%
rec2 = rec.channel_slice(rec.channel_ids[200:250]).frame_slice(0, 100 * 30_000)
rec2

# %%
from pathlib import Path

# %%
rec2.save_to_folder(folder=Path("~/data/testds").expanduser())

# %%
# load 50k waveforms at random times
rg = np.random.default_rng(0)
times = rg.integers(100, rec.get_num_samples() - 100, size=50_000)
max_channels = rg.integers(0, 384, size=50_000)
# do it sparsely
geom = rec.get_channel_locations()
ci = waveform_utils.make_channel_index(geom, 75)

# %%

# %%
ts = (10, 100, 1000, 10000)
reps = range(6)
stats50 = np.zeros((len(ts), len(reps)))
stats10 = np.zeros((len(ts), len(reps)))
for i, total_seconds in enumerate(ts):
    for rep in range(6):
        times = rg.integers(0, total_seconds * 30_000, size=50_000)
        us = np.unique(times // 30_000)
        stats50[i, rep] = us.size / total_seconds
        times = rg.integers(0, total_seconds * 30_000, size=10_000)
        us = np.unique(times // 30_000)
        stats10[i, rep] = us.size / total_seconds

# %%
import matplotlib.pyplot as plt

# %%
plt.plot(ts, stats50.mean(1))
plt.fill_between(ts, stats50.mean(1) - stats50.std(1), stats50.mean(1) + stats50.std(1), alpha=0.5)
plt.plot(ts, stats10.mean(1))
plt.fill_between(ts, stats10.mean(1) - stats10.std(1), stats10.mean(1) + stats10.std(1), alpha=0.5)
plt.semilogx()

# %%

# %%
wfs = spikeio.read_waveforms(times, raw_bin, max_channels=max_channels, channel_index=ci, n_channels=384)

# %%
for _ in trange(3):
    wfs = spikeio.read_waveforms(times, raw_bin, max_channels=max_channels, channel_index=ci, n_channels=384)    

# %%
pipeline = transform.WaveformPipeline([transform.Waveform(ci)])
grab = GrabAndFeaturize(rec, torch.as_tensor(ci), pipeline, torch.as_tensor(times), torch.as_tensor(max_channels))

# %%
grab.peel("/tmp/testgrab.h5", overwrite=True)

# %%
# !h5ls /tmp/testgrab.h5

# %%
