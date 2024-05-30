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
import matplotlib.pyplot as plt
from ephysx import spike_gpfa
import torch
import gpytorch

# %%
# %config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import offset_copy
from matplotlib.patches import Ellipse, Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import contextlib

plt.rc("figure", dpi=300)
plt.rc("figure", figsize=(3, 2))
SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE =  10
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "serif",
})
preamble = r"""
\renewcommand{\familydefault}{\sfdefault}
\usepackage[scaled=1]{helvet}
\usepackage[helvet]{sfmath}
\usepackage{textgreek}
"""
plt.rc('text.latex', preamble=preamble)
plt.rc('svg', fonttype='none')
plt.rc('ps', usedistiller='xpdf')
plt.rc('pdf', fonttype=42)

def inline_xlabel(ax, label):
    t = offset_copy(
        ax.transAxes,
        y=-(ax.xaxis.get_tick_padding() + ax.xaxis.majorTicks[0].get_pad()), 
        fig=ax.figure,
        units='points',
    )
    ax.xaxis.set_label_coords(.5, 0, transform=t)
    ax.set_xlabel(label, va='baseline', ha="center")
    ax.xaxis.get_label().set_bbox(dict(facecolor='white', alpha=0.0, linewidth=0))

def inline_ylabel(ax, label):
    t = offset_copy(
        ax.transAxes,
        # changed to xaxis here bc yaxis has too much tick space?
        x=-(ax.yaxis.get_tick_padding() + ax.yaxis.majorTicks[0].get_pad()), 
        fig=ax.figure,
        units='points',
    )
    ax.yaxis.set_label_coords(0, .5, transform=t)
    ax.set_ylabel(label, va='top', ha="center")
    ax.yaxis.get_label().set_bbox(dict(facecolor='white', alpha=0.0, linewidth=0))
    
def x_scalebar(ax, length, label=None, unit="s", lw=2, loc="lower left"):
    if label is None:
        label = f"{length}{unit}"

    t_line = offset_copy(
        trans=ax.get_xaxis_transform(),
        y=-(ax.xaxis.get_tick_padding()), 
        fig=ax.figure,
        units='points',
    )
    t_text = offset_copy(
        trans=ax.get_xaxis_transform(),
        y=-(ax.xaxis.get_tick_padding() + SMALL_SIZE), 
        fig=ax.figure,
        units='points',
    )
    ax.figure.set_facecolor([0, 0, 0, 0])
    ax.figure.patch.set_facecolor([0, 0, 0, 0])
    
    line_x = [ax.get_xlim()[0], ax.get_xlim()[0] + length]
    line_y = [0, 0]
    
    line = Line2D(line_x, line_y, color="k", linewidth=lw, transform=t_line, solid_capstyle="butt")
    ax.figure.add_artist(line)
    
    ax.text(sum(line_x) / 2, 0, label, ha="center", va="center", transform=t_text)
    
def y_scalebar(ax, length, label=None, unit="\\textmu{}m", lw=2, loc="lower left"):
    if label is None:
        label = f"{length}{unit}"

    t_line = offset_copy(
        trans=ax.get_yaxis_transform(),
        x=-(ax.yaxis.get_tick_padding()), 
        fig=ax.figure,
        units='points',
    )
    t_text = offset_copy(
        trans=ax.get_yaxis_transform(),
        x=-(ax.yaxis.get_tick_padding() + SMALL_SIZE / 2), 
        # y=- SMALL_SIZE / 2,
        fig=ax.figure,
        units='points',
    )
    ax.figure.set_facecolor([0, 0, 0, 0])
    ax.figure.patch.set_facecolor([0, 0, 0, 0])
    
    line_x = [0, 0]
    line_y = [ax.get_ylim()[0], ax.get_ylim()[0] + length]

    line = Line2D(line_x, line_y, color="k", linewidth=lw, transform=t_line, solid_capstyle="butt")
    ax.figure.add_artist(line)
    
    ax.text(0, sum(line_y) / 2, label, ha="right", va="center", transform=t_text, rotation="vertical")
    

@contextlib.contextmanager
def subplots(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    try:
        yield fig, axes
    finally:
        plt.show()
        plt.close(fig)


def clearpanel(figure):
    figure.set_facecolor([0, 0, 0, 0])
    figure.patch.set_facecolor([0, 0, 0, 0])


# %%

# %%
rg = np.random.default_rng(0)

# %%
ntrain = 1024
train_x = rg.uniform(0, 5, size=ntrain)[:, None]
train_x.sort(0)
train_z = np.sin(train_x) * np.exp(-(train_x-2)**2)
mu0 = np.array([-5, 5])
W0 = np.array([[2, -0.5]])
train_noise = rg.normal(size=(ntrain, 2), scale=0.1)
train_y = mu0 + train_z @ W0 + train_noise

# %%
plt.scatter(train_x, train_z)

# %%
with subplots(ncols=2, sharey=True) as (fig, (aa, ab)):
    aa.scatter(train_x, train_y[:, 0], s=1)
    ab.scatter(train_x, train_y[:, 1], s=1)

# %%
# test cubic interp
test_grid = np.linspace(-1, 1, num=40)
test_fn = lambda x: x ** 3 - x ** 2 + 3
grid_fn = test_fn(test_grid)
x_eval = np.linspace(-1, 1, num=1000)
true_eval = test_fn(x_eval)

# %%
left_interp_matrix = spike_gpfa.left_cubic_interpolation_matrix(
    torch.from_numpy(test_grid),
    torch.from_numpy(x_eval),
)

# %%
interp_eval = left_interp_matrix @ grid_fn

# %%
plt.plot(x_eval, interp_eval)
plt.plot(x_eval, true_eval)

# %%
plt.plot(x_eval, torch.tensor(true_eval) - interp_eval)

# %%
torch.abs(interp_eval - torch.tensor(true_eval)).max()

# %%
# assert torch.isclose(interp_eval, torch.tensor(true_eval)).all()

# %%

# %%
gpfa = spike_gpfa.GridGPFA(
    (0, 5),
    2,
    lengthscale=0.5,
    grid_size=10,
    loss_on_interp=True,
    learn_lengthscale=True,
    # prior_noiselogit=-10.0,
    learn_prior_noise_fraction=True,
    latent_update="embed_uninterp",
)

# %% tags=[]
losses = gpfa.fit(
    torch.tensor(train_x, dtype=torch.float),
    torch.tensor(train_y, dtype=torch.float),
    n_iter=200,
    lr=0.1,
    eps=1e-5,
)

# %%
gpfa.obs_logstd.exp()

# %%
gpfa.lengthscale()

# %%
gpfa.noise_fraction()

# %%
mu0, gpfa.net.bias

# %%
W0, gpfa.net.weight

# %%
plt.plot(losses)

# %%
plt.plot(losses[-10:])

# %%
gridz, gridpreds = gpfa(gpfa.grid)

# %%
plt.plot(gpfa.grid.numpy(), gpfa.grid_z.numpy(force=True))
plt.plot(gpfa.grid.numpy(), gridz.numpy(force=True)[:,0])

# %%
z, pred = gpfa(torch.tensor(train_x))

# %%
plt.scatter(train_x, train_z)
plt.scatter(train_x, z.numpy(force=True))

# %%
with subplots(ncols=2, sharey=True) as (fig, (aa, ab)):
    aa.scatter(train_x, train_y[:, 0], s=1)
    ab.scatter(train_x, train_y[:, 1], s=1)
    aa.scatter(train_x, pred[:, 0].numpy(force=True), s=1)
    ab.scatter(train_x, pred[:, 1].numpy(force=True), s=1)

# %%

# %%

# %%
