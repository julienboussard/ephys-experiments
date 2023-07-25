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
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# %%
from spike_psvae.multiprocessing_utils import get_pool

# %%
global worker_params

# %%
from sklearn.decomposition import PCA

# %%

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from dredge import dredge_ap, motion_util as mu

# %%
import matplotlib.pyplot as plt

# %%
with np.load("/Users/charlie/Downloads/test_locations.npz") as npz:
    a = npz["amplitude"]
    t = npz["t_seconds"]
    z = npz["depth_um"]

# %%
a.min(), a.mean(), a.max()

# %%

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
plt.plot(np.exp(-((np.arange(121) - 42) / 30) ** 2) - np.exp(-((np.arange(121) - 46) / 30) ** 2))

# %%
import numpy as np
from dartsort.localize import localize_torch
from dartsort.util.waveform_util import make_channel_index
from neuropixel import dense_layout
h = dense_layout()
g = np.c_[h["x"], h["y"]]
padded_g = np.pad(g, [(0, 1), (0, 0)], constant_values=np.nan)
ci = make_channel_index(g, 75)

# %%
padded_g.shape

# %%
padded_g[-1]

# %%
g.shape

# %%
ci

# %%
padded_g[ci[5]]

# %%

# %%

# %%

# %%

# %%

# %%
me.displacement.std()

# %%
z.min(), z.max()

# %%
me, extra = dredge_ap.register(
    a,
    z,
    t,
    bin_um=5,
    bin_s=2,
    max_disp_um=300,
    win_scale_um=1800,
    win_step_um=500,
)


fig, (aa, ab) = plt.subplots(ncols=2, figsize=(10, 5))
mu.show_spike_raster(a, z, t, aa, aspect="auto", vmax=50)
mu.plot_me_traces(me, aa, color="r")

mu.show_registered_raster(me, a, z, t, ab, aspect="auto", vmax=50)

# %%

# %%

# %%

# %%

# %%

# %%

# %%
x = np.random.default_rng(0).normal(size=(100_000, 121))

# %%
p1 = PCA(5, random_state=0).fit(x)

# %%
p2 = PCA(5, random_state=0).fit(x)

# %%
np.array_equal(p1.components_, p2.components_)

# %%
import torch
torch.float

# %%
import numpy as np

# %%
np.dtype(str(torch.float).split(".")[1])

# %%
np.float32

# %%
import spikeinterface.full as si

# %%
z = si.read_binary_folder("/Users/charlie/data/spike_sorting_paper_figs/subtraction_fig/subtraction_fig_data/zad5/")

# %%
z

# %%
# z.get_traces?

# %%
from spike_psvae.multiprocessing_utils import get_pool, CloudpicklePoolExecutor


# %%
class P:
    def __init__(self, x):
        self.x = x
    
    def __str__(self):
        return f"P({self.x})"
    
    def job(self, arg):
        print("Job", self, id(self))
        return self.x * arg
    
    def init(self):
        print("Init", self, id(self))
    
    def run(self, it, n_jobs):
        print("Run", self, id(self))
        Pool, context = get_pool(n_jobs, cls=CloudpicklePoolExecutor)
        
        with Pool(n_jobs, mp_context=context, initializer=self.init, initargs=()) as p:
            results = list(p.map(self.job, it))
        return results
        
        

# %%
p = P(3)

# %%
p.run([1, 2, 3], 2)

# %%
from scipy.stats import zscore

# %%
zscore(np.zeros((10, 10, 10, 10)), axis=(0, 1, 2))

# %%
zz = """\
02cc03e4-8015-4050-bb42-6c832091febb
0851db85-2889-4070-ac18-a40e8ebd96ba
0aafb6f1-6c10-4886-8f03-543988e02d9e
0b8ea3ec-e75b-41a1-9442-64f5fbc11a5a
143dd7cf-6a47-47a1-906d-927ad7fe9117
16799c7a-e395-435d-a4c4-a678007e1550
1a60a6e1-da99-4d4e-a734-39b1d4544fad
1e176f17-d00f-49bb-87ff-26d237b525f1
27bac116-ea57-4512-ad35-714a62d259cd
31f3e083-a324-4b88-b0a4-7788ec37b191
36362f75-96d8-4ed4-a728-5e72284d0995
3d3d5a5e-df26-43ee-80b6-2d72d85668a5
41a3b948-13f4-4be7-90b9-150705d39005
485b50c8-71e1-4654-9a07-64395c15f5ed
523f8301-4f56-4faf-ab33-a9ff11331118
63517fd4-ece1-49eb-9259-371dc30b1dd6
69f42a9c-095d-4a25-bca8-61a9869871d3
6b6af675-e1ef-43a6-b408-95cfc71fe2cc
6e1379e8-3af0-4fc5-8ba8-37d3bb02226b
6fc4d73c-2071-43ec-a756-c6c6d8322c8b
70da415f-444d-4148-ade7-a1f58a16fcf8
7cbecb3f-6a8a-48e5-a3be-8f7a762b5a04
7d999a68-0215-4e45-8e6c-879c6ca2b771
80f6ffdd-f692-450f-ab19-cd6d45bfd73e
84bb830f-b9ff-4e6b-9296-f458fb41d160
8abf098f-d4f6-4957-9c0a-f53685db74cc
8b7c808f-763b-44c8-b273-63c6afbc6aae
8ca1a850-26ef-42be-8b28-c2e2d12f06d6
8d59da25-3a9c-44be-8b1a-e27cdd39ca34
9117969a-3f0d-478b-ad75-98263e3bfacf
a12c8ae8-d5ad-4d15-b805-436ad23e5ad1
ad714133-1e03-4d3a-8427-33fc483daf1a
b25799a5-09e8-4656-9c1b-44bc9cbb5279
b749446c-18e3-4987-820a-50649ab0f826
b83407f8-8220-46f9-9b90-a4c9f150c572
bbe6ebc1-d32f-42dd-a89c-211226737deb
bc1602ba-dd6c-4ae4-bcb2-4925e7c8632a
bf96f6d6-4726-4cfa-804a-bca8f9262721
c07d13ed-e387-4457-8e33-1d16aed3fd92
c17772a9-21b5-49df-ab31-3017addea12e
c4f6665f-8be5-476b-a6e8-d81eeae9279d
ce397420-3cd2-4a55-8fd1-5e28321981f4
dab512bd-a02d-4c1f-8dbc-9155a163efc0
e31b4e39-e350-47a9-aca4-72496d99ff2a
e49f221d-399d-4297-bb7d-2d23cc0e4acc
eeb27b45-5b85-4e5c-b6ff-f639ca5687de
f03b61b4-6b13-479d-940f-d1608eb275cc
f26a6ab1-7e37-4f8d-bb50-295c056e1062
f2a098e7-a67e-4125-92d8-36fc6b606c45
f2ee886d-5b9c-4d06-a9be-ee7ae8381114
f4bd76a6-66c9-41f3-9311-6962315f8fc8
f93bfce4-e814-4ae3-9cdf-59f4dcdedf51
f9d8aacd-b2a0-49f2-bd71-c2f5aadcfdd1
febb430e-2d50-4f83-87a0-b5ffbb9a4943
"""

# %%
qq = """\
0b8ea3ec-e75b-41a1-9442-64f5fbc11a5a
143dd7cf-6a47-47a1-906d-927ad7fe9117
16799c7a-e395-435d-a4c4-a678007e1550
1a60a6e1-da99-4d4e-a734-39b1d4544fad
1e176f17-d00f-49bb-87ff-26d237b525f1
27bac116-ea57-4512-ad35-714a62d259cd
31f3e083-a324-4b88-b0a4-7788ec37b191
3d3d5a5e-df26-43ee-80b6-2d72d85668a5
41a3b948-13f4-4be7-90b9-150705d39005
523f8301-4f56-4faf-ab33-a9ff11331118
63517fd4-ece1-49eb-9259-371dc30b1dd6
6e1379e8-3af0-4fc5-8ba8-37d3bb02226b
6fc4d73c-2071-43ec-a756-c6c6d8322c8b
7d999a68-0215-4e45-8e6c-879c6ca2b771
84bb830f-b9ff-4e6b-9296-f458fb41d160
8b7c808f-763b-44c8-b273-63c6afbc6aae
8ca1a850-26ef-42be-8b28-c2e2d12f06d6
9117969a-3f0d-478b-ad75-98263e3bfacf
ad714133-1e03-4d3a-8427-33fc483daf1a
b749446c-18e3-4987-820a-50649ab0f826
"""

# %%
len(zz.split())

# %%
len(qq.split())

# %%
all(q in zz.split() for q in qq.split())

# %%

# %%
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# %%
rgi = RegularGridInterpolator((np.arange(10), 1 + np.arange(5)), np.ones((10, 5)))

# %%
x = np.array([1.5, 2.5, 3.5])
y = np.array([2.5, 3])
x, y = np.meshgrid(x, y, indexing="ij")

# %%
rgi(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)

# %%

# %%
import torch

z = torch.zeros(10)

ix = torch.tensor([0, 0])
add = torch.tensor([1, 2.])
z.scatter_add_(0, ix, add)

z[ix] += add

z

# %%
z.clamp_(0, 2)

# %%
import torch

# %%
from concurrent.futures import ProcessPoolExecutor

# %%
torch.is_grad_enabled()

# %%
torch.set_grad_enabled(False)


# %%
def j


# %%

# %%
# torch.save?

# %%
z

# %%
from one.api import ONE

# %%
from brainbox.io.one import SpikeSortingLoader

# %%
rec = si.read_binary_folder("/Users/charlie/data/spike_sorting_paper_figs/subtraction_fig/subtraction_fig_data/zad5/")

# %%
rec.get_times()

# %%
one = ONE()

# %%
ssl = SpikeSortingLoader(one, pid="8ca1a850-26ef-42be-8b28-c2e2d12f06d6")

# %%
sorting, header, coords = ssl.load_spike_sorting()

# %%
sorting

# %%
sorting["times"].min(), sorting["times"].max()

# %%
times = sorting["times"]

# %%
times_samples = ssl.samples2times(times, direction="reverse")

# %%
start_samples, end_samples = rec.get_times()[[0, -1]] * 30000

# %%
which = (times_samples > start_samples) & (times_samples < end_samples)
times = times[which]
times_samples = times_samples[which]
clusters = sorting["clusters"][which]

# %%
times_samples.min(), times_samples.max()

# %%
times_samples_rounded = np.around(times_samples)
np.abs(times_samples_rounded - times_samples).max()

# %%
t_samples = times_samples_rounded.astype(np.int)


# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
def job(i):
    obj = job.obj
    time.sleep(2)
    print("Worker", i)
    return i, np.square(obj).sum()


# %%
def worker_init(obj):
    print("hi from worker init")
    job.obj = obj


# %%
def f(x):
    return x ** 2


# %%
list(map(f, range(10)))

# %%
Executor, context = get_pool(1)

# %%
expensive_object = np.arange(1000)

# %%
with Executor(4, mp_context=context, initializer=worker_init, initargs=(expensive_object,)) as p:
    for result in p.map(job, range(10)):
        print("Got result", result)

# %%
