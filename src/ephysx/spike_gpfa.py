import dataclasses
from typing import Union

from tqdm.auto import tqdm, trange

import gpytorch
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from dartsort.cluster import density
from dartsort.util import data_util, drift_util, waveform_util
from gpytorch.distributions import Delta, MultivariateNormal
from gpytorch.kernels import InducingPointKernel, RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean
# from gpytorch.utils.interpolation import Interpolation
from linear_operator.operators import DiagLinearOperator
from linear_operator.utils.interpolation import left_interp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import eigh
from scipy.sparse import dok_array
from sklearn.gaussian_process.kernels import RBF

tqdm_kw = dict(smoothing=0, mininterval=1 / 24)


# -- helper classes


@dataclasses.dataclass
class SpikePCAData:
    """Data bag to keep things tidy."""

    keepers: np.array
    original_labels: np.array
    channels: np.array
    times_seconds: np.array
    n_reg_chans: int
    tpca_rank: int
    n_chans_cluster: int
    n_wf_chans: int
    n_spikes: int
    original_channel_index: np.array
    wf_channel_index: np.array
    cluster_channel_index: np.array
    static_amp_vecs: torch.Tensor
    tpca_embeds: Union[np.array, h5py.Dataset]
    spike_static_channels: np.array
    registered_geom: np.array
    in_memory: bool

    def getwf(self, index):
        if self.in_memory:
            return self.tpca_embeds[index]

        scalar = np.isscalar(index)
        index = np.atleast_1d(index)
        indices = self.keepers[index]
        mask = np.zeros(self.tpca_embeds.shape[0], dtype=bool)
        mask[indices] = 1
        wf = _channel_subset_by_chunk(
            mask,
            self.tpca_embeds,
            self.channels[index],
            self.original_channel_index,
            self.wf_channel_index,
        )

        # grab static channels
        # wf = _read_by_chunk(mask, self.tpca_embeds)
        # wf = waveform_util.channel_subset_by_index(
        #     wf, self.channels[index], self.original_channel_index, self.wf_channel_index
        # )
        if scalar:
            wf = wf[0]
        return wf


class GPFAUnit(torch.nn.Module):
    pass


class GPFAClusterer(torch.nn.Module):
    """Mixture of GPFAs."""


# -- core classes


class GridGPFA(torch.nn.Module):
    def __init__(
        self,
        t_bounds,
        output_dim,
        latent_dim=1,
        lengthscale=100.0,
        prior_noiselogit=-6.0,
        obs_logstd=0.0,
        points_per_lengthscale=2,
        grid_size=None,
        learn_lengthscale=False,
        learn_prior_noise_fraction=False,
        learn_obsstd=True,
        loss_on_interp=False,
        latent_update="gradient",
    ):
        super().__init__()
        self.t_bounds = t_bounds
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.loss_on_interp = loss_on_interp

        # make grid
        if grid_size is None:
            grid_size = (
                (t_bounds[1] - t_bounds[0]) / lengthscale
            ) * points_per_lengthscale
            grid_size = int(np.ceil(grid_size))
        self.grid_size = grid_size
        self.register_buffer(
            "grid",
            torch.linspace(
                torch.tensor(t_bounds[0], dtype=torch.float),
                torch.tensor(t_bounds[1], dtype=torch.float),
                steps=grid_size,
            )[:, None],
        )
        grid_z = torch.zeros((grid_size, latent_dim))
        self.latent_update = latent_update
        if latent_update == "gradient":
            self.register_parameter("grid_z", torch.nn.Parameter(grid_z))
        if latent_update == "embed_uninterp":
            self.register_buffer("grid_z", grid_z)
        else:
            assert False

        # kernel
        self.learn_lengthscale = learn_lengthscale
        self.init_kernel(lengthscale)
        # scale_tril = np.linalg.cholesky(K)
        self.register_buffer("prior_mean", torch.zeros(()))
        self.learn_prior_noise_fraction = learn_prior_noise_fraction
        prior_noiselogit = prior_noiselogit + torch.zeros(())
        if learn_prior_noise_fraction:
            self.register_parameter(
                "prior_noiselogit", torch.nn.Parameter(prior_noiselogit)
            )
        else:
            self.register_buffer("prior_noiselogit", prior_noiselogit)
        self._interp_id = None
        self._cached_priordist = None

        # observation model
        self.net = torch.nn.Linear(latent_dim, self.output_dim)
        obs_logstd = obs_logstd + torch.zeros(output_dim)
        if learn_obsstd:
            self.register_parameter(
                "obs_logstd",
                torch.nn.Parameter(obs_logstd),
            )
        else:
            self.register_buffer(
                "obs_logstd",
                obs_logstd,
            )

    def init_kernel(self, lengthscale):
        if not self.learn_lengthscale:
            Kuu = RBF(lengthscale)(self.grid)
            grid_scale_left = np.linalg.cholesky(Kuu)
            self.register_buffer(
                "_grid_scale_left", torch.tensor(grid_scale_left, dtype=torch.float)
            )
        else:
            self.register_buffer(
                "_half_sq_dgrid", -0.5 * torch.square(self.grid - self.grid.T)
            )

        lengthscale = torch.tensor(lengthscale)
        if self.learn_lengthscale:
            if lengthscale < 20.0:
                # invert softplus
                lengthscale = lengthscale.expm1().log()
            self.register_parameter("_lengthscale", torch.nn.Parameter(lengthscale))
        else:
            self.register_buffer("_lengthscale", lengthscale)

    def lengthscale(self):
        if self.learn_lengthscale:
            return F.softplus(self._lengthscale)
        return self._lengthscale

    def grid_cov(self):
        if not self.learn_lengthscale:
            return self._grid_scale_left @ self._grid_scale_left.T

        Kuu = self._half_sq_dgrid / self.lengthscale().square()
        Kuu = Kuu.exp()
        return Kuu

    def grid_scale_left(self, eps=1e-4):
        Kuu = self.grid_cov()
        # Kuu.diagonal().add_(eps)
        # Kuu = Kuu + eps * torch.eye(self.grid_size, device=Kuu.device)
        scale_left = torch.linalg.cholesky(Kuu)
        return scale_left

    def noise_fraction(self):
        return F.sigmoid(self.prior_noiselogit)

    @torch.no_grad()
    def _compute_grid_matrix(self, inputs):
        left_interp_matrix = left_cubic_interpolation_matrix(self.grid, inputs)
        return left_interp_matrix

    def forward(self, t, left_interp_matrix=None):
        if left_interp_matrix is None:
            left_interp_matrix = self._compute_grid_matrix(t)
        z = left_interp_matrix @ self.grid_z
        preds = self.net(z)
        return z, preds

    def get_prior_distribution(self, left_interp_matrix=None, eps=1e-4):
        if not (self.learn_lengthscale or self.learn_prior_noise_fraction):
            if self._cached_priordist is not None:
                if self._interp_id == id(left_interp_matrix):
                    return self._cached_priordist
                else:
                    self._cached_priordist = None

        scale_left = self.grid_scale_left(eps)
        if self.loss_on_interp:
            scale_left = left_interp_matrix @ scale_left

        n_prior = left_interp_matrix.shape[0] if self.loss_on_interp else self.grid_size

        lambd = self.noise_fraction()
        prior_dist = torch.distributions.LowRankMultivariateNormal(
            self.prior_mean.broadcast_to((n_prior,)),
            (1.0 - lambd) * scale_left,
            lambd.add(eps).broadcast_to((n_prior,)),
        )
        if not (self.learn_lengthscale or self.learn_prior_noise_fraction):
            self._cached_priordist = prior_dist
            self._interp_id = id(left_interp_matrix)
        return prior_dist

    def log_prior(self, z, left_interp_matrix=None, eps=1e-4):
        prior_dist = self.get_prior_distribution(left_interp_matrix, eps)
        if self.loss_on_interp:
            logprior = prior_dist.log_prob(z.T).sum()
        else:
            logprior = prior_dist.log_prob(self.grid_z.T).sum()
        return logprior, prior_dist

    @torch.no_grad()
    def update_z_embed_uninterp(
        self, y, prior_dist=None, left_interp_matrix=None, left_interp_pinv=None
    ):
        if prior_dist is None:
            prior_dist = self.get_prior_distribution(left_interp_matrix)
        if left_interp_pinv is None:
            left_interp_pinv = torch.linalg.pinv(left_interp_matrix)
        z = self.embed(y)
        z = left_interp_pinv @ z
        self.grid_z.copy_(z)

    def embed(self, y):
        return (y - self.net.bias) @ torch.linalg.pinv(self.net.weight).T

    def log_likelihood(self, preds, targets, mask_tuple):
        obs_var = (2 * self.obs_logstd).exp()
        # recon_err = torch.nansum(torch.square(preds - targets))
        recon_err = torch.square(preds[mask_tuple] - targets[mask_tuple])
        recon_err = (recon_err / obs_var[mask_tuple[1]]).sum()
        # denom = (mask * self.obs_logstd).sum()
        denom = (self.obs_logstd[mask_tuple[1]]).sum()
        loglik = -0.5 * recon_err - denom

        return loglik

    def loss(self, train_t, train_y, left_interp_matrix, mask_tuple, eps=1e-4):
        z, pred = self(train_t, left_interp_matrix)
        loss = -self.log_likelihood(pred, train_y, mask_tuple)
        prior_dist = None
        if self.latent_update == "gradient" or (
            self.learn_lengthscale or self.learn_prior_noise_fraction
        ):
            logprior, prior_dist = self.log_prior(z, left_interp_matrix, eps)
            loss = loss - logprior
        return loss, prior_dist

    def initialize_svd_smoothed(
        self, train_t, train_y, left_interp_matrix, missing=None, empty=None
    ):
        if missing is None:
            missing = torch.isnan(train_y)
        if empty is None:
            empty = missing.all(1)
        loadings, mean, components, svs = fit_pcas(
            train_y, missing, empty, self.latent_dim, max_iter=1
        )
        weights = left_interp_matrix @ self.grid_cov()
        weights = weights / weights.sum(0)
        zinit = weights.T @ loadings

        with torch.no_grad():
            self.net.bias.copy_(mean)
            self.net.weight.copy_(components)
            self.grid_z.copy_(zinit)
        print(f"{self.net.bias=}")
        print(f"{self.net.weight=}")

    def fit(
        self,
        train_t,
        train_y,
        lr=0.05,
        eps=1e-8,
        show_progress=True,
        n_iter=100,
        loss_converged=1e-2,
    ):
        # precompute cubic interpolation kernel info
        left_interp_matrix = self._compute_grid_matrix(train_t)
        if self.latent_update == "embed_uninterp":
            left_interp_pinv = torch.linalg.pinv(left_interp_matrix)

        # missing values info
        mask = torch.isfinite(train_y)
        mask_tuple = torch.nonzero(mask)
        missing = torch.logical_not(mask)
        empty = missing.all(1)

        # initialize with pca
        self.initialize_svd_smoothed(
            train_t, train_y, left_interp_matrix, missing, empty
        )

        # optimize
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        losses = torch.zeros(n_iter, device=self.net.bias.device, requires_grad=False)
        for i in (trange if show_progress else range)(n_iter):
            opt.zero_grad()
            loss, prior_dist = self.loss(
                train_t, train_y, left_interp_matrix, mask_tuple, eps
            )
            losses[i] = loss.detach()
            if i > 10 and loss + loss_converged >= losses[i - 1]:
                break

            loss.backward()
            opt.step()

            if self.latent_update == "embed_uninterp":
                self.update_z_embed_uninterp(
                    train_y,
                    prior_dist=prior_dist,
                    left_interp_matrix=left_interp_matrix,
                    left_interp_pinv=left_interp_pinv,
                )

        return losses[: i + 1].numpy(force=True)


# -- core functions


@torch.no_grad()
def fit_pcas(
    X,
    missing,
    empty,
    rank,
    max_iter=100,
    check_every=5,
    n_oversamples=10,
    atol=1e-3,
    show_progress=False,
    centered=True,
):
    """
    X : (nu, n, dim_in)
    missing : (nu, n, dim_in)
    empty : (nu, n)
    """
    # single = X.ndim == 2
    # if single:
    #     assert missing.ndim == 2
    #     assert empty.ndim == 1
    #     X = X[None]
    #     missing = missing[None]
    #     empty = empty[None]

    # initialize mean
    # Xc = torch.where(ignore, torch.nan, X)
    Xc = X.clone()
    if centered:
        mean = torch.nan_to_num(Xc.nanmean(dim=-2, keepdims=True))
    else:
        shape = list(Xc.shape)
        shape[-1] = 1
        mean = torch.zeros(shape, device=Xc.device, dtype=Xc.dtype)
    # after this line, isnan(Xc) === empty.
    Xc[missing] = mean.broadcast_to(X.shape)[missing]
    if centered:
        mean = Xc.nanmean(dim=-2, keepdims=True)
    else:
        Xc = torch.where(empty[..., None], 0, Xc)

    ###
    filled = torch.logical_not(empty)
    no_missing = not missing[filled].any()

    # iterate svds
    it = trange(max_iter, desc="SVD") if show_progress else range(max_iter)
    # svd_storage = None
    q = min(rank + n_oversamples, X.shape[-1])
    for j in it:
        # update svd
        if centered:
            Xin = torch.where(empty[..., None], 0, Xc - mean)
        else:
            Xin = Xc
        # svd_storage = torch.linalg.svd(Xin, full_matrices=False, out=svd_storage)
        # U, S, Vh = svd_storagev

        # faster in my experience despite the note in torch.linalg.svd docs
        U, S, V = torch.svd_lowrank(Xin, q=q)
        Vh = V.mT

        U = U[..., :rank]
        S = S[..., :rank]
        Vh = Vh[..., :rank, :]

        if no_missing:
            break

        # impute
        recon = torch.baddbmm(mean, U, S[..., None] * Vh)
        check = not (j % check_every)
        if check:
            dx = (Xc[missing] - recon[missing]).abs().max().numpy(force=True)
            dx = float(dx)
            if show_progress:
                it.set_description(f"{dx=:0.5f}")
        Xc[missing] = recon[missing]
        if centered:
            mean = Xc.nanmean(dim=-2, keepdims=True)

        if check and dx < atol:
            break

    # svd -> pca
    loadings = U * S[..., None, :]
    mean = mean[..., 0, :]
    components = Vh.mT
    svs = S

    # if single:
    #     loadings = loadings[0]
    #     mean = mean[0]
    #     components = components[0]
    #     svs = svs[0]

    return loadings, mean, components, svs


def _load_data(
    sorting,
    motion_est,
    fit_radius,
    wf_radius=None,
    in_memory=False,
    whiten_input=False,
    initially_clustered_only=False,
):
    # load up labels
    labels = sorting.labels
    if initially_clustered_only:
        keep_mask = labels >= 0
    else:
        keep_mask = np.ones(labels.shape, dtype=bool)
    keepers = np.flatnonzero(keep_mask)
    labels = labels[keepers]
    channels = sorting.channels[keepers]
    times_seconds = sorting.times_seconds[keepers]

    # load waveforms and subset by radius, retaining the new index for later
    h5 = h5py.File(sorting.parent_h5_path, "r", locking=False)
    geom = h5["geom"][:]
    original_channel_index = h5["channel_index"][:]

    # amplitude vectors on channel subset
    amp_vecs = _read_by_chunk(keep_mask, h5["denoised_ptp_amplitude_vectors"])
    amp_vecs, channel_index = waveform_util.channel_subset_by_radius(
        amp_vecs,
        channels,
        original_channel_index,
        geom,
        radius=wf_radius,
    )

    # tpca embeds on channel subset
    tpca_embeds = h5["collisioncleaned_tpca_features"]
    if in_memory:
        tpca_embeds = _channel_subset_by_chunk(
            keep_mask, tpca_embeds, channels, original_channel_index, channel_index
        )
        h5.close()
        if whiten_input:
            s = np.nanstd(tpca_embeds, axis=0, keepdims=True)
            s[s == 0] = 1.0
            tpca_embeds /= s
        tpca_embeds.setflags(write=False)

    # static channels logic
    pitch = drift_util.get_pitch(geom)
    registered_geom = drift_util.registered_geometry(geom, motion_est=motion_est)
    registered_kdtree = drift_util.KDTree(registered_geom)
    match_distance = drift_util.pdist(geom).min() / 2
    cluster_channel_index = waveform_util.make_channel_index(
        registered_geom, fit_radius
    )
    n_reg_chans = len(registered_geom)
    n_wf_chans = channel_index.shape[1]
    n_chans_cluster = cluster_channel_index.shape[1]
    tpca_rank = tpca_embeds.shape[1]
    n_spikes = keepers.size
    n_pitches_shift = drift_util.get_spike_pitch_shifts(
        sorting.point_source_localizations[:, 2][keepers],
        geom=geom,
        motion_est=motion_est,
        times_s=times_seconds,
    )

    # where a channel is not present, this has n_reg_chans
    spike_static_channels = drift_util.static_channel_neighborhoods(
        geom,
        channels,
        channel_index,
        pitch=pitch,
        n_pitches_shift=n_pitches_shift,
        registered_geom=registered_geom,
        target_kdtree=registered_kdtree,
        match_distance=match_distance,
        workers=4,
    )

    static_amp_vecs = drift_util.grab_static(
        torch.from_numpy(amp_vecs),
        spike_static_channels,
        n_reg_chans,
    )

    return SpikePCAData(
        keepers=keepers,
        original_labels=labels,
        channels=channels,
        times_seconds=times_seconds,
        tpca_rank=tpca_rank,
        n_reg_chans=n_reg_chans,
        n_chans_cluster=n_chans_cluster,
        n_wf_chans=n_wf_chans,
        n_spikes=n_spikes,
        original_channel_index=original_channel_index,
        wf_channel_index=channel_index,
        cluster_channel_index=cluster_channel_index,
        static_amp_vecs=static_amp_vecs,
        tpca_embeds=tpca_embeds,
        spike_static_channels=spike_static_channels,
        registered_geom=registered_geom,
        in_memory=in_memory,
    )


# -- helpers


def cubic_interpolation_kernel(x, xeval):
    assert x.shape[0] == x.numel()
    assert xeval.shape[0] == xeval.numel()
    x = x.ravel()
    xeval = xeval.ravel()

    h = x[1] - x[0]
    n = x.numel()
    left_inds = torch.searchsorted(x, xeval, right=True) - 1
    s = (xeval - x[left_inds]) / h

    # in parentheses
    s3 = s**3
    s2 = s**2
    w0 = 0.5 * (-s3 + 2 * s2 - s)
    w3 = 0.5 * (s3 - s2)

    # main case
    inds = left_inds[:, None] + torch.arange(-1, 3, device=left_inds.device)
    weights = torch.empty(inds.shape, dtype=x.dtype, device=x.device)
    weights[:, 0] = w0
    weights[:, 1] = 0.5 * (3 * s3 - 5 * s2 + 2)
    weights[:, 2] = 0.5 * (-3 * s3 + 4 * s2 + s)
    weights[:, 3] = w3

    # points on left boundary
    left = left_inds == 0
    # inds[:, 0] += left.to(inds.dtype)  # -1 -> 0
    dw_left = torch.column_stack((-w0, 3 * w0, -3 * w0, w0))
    weights[left] += dw_left[left]

    # points on right boundary
    right = left_inds == (n - 2)
    # inds[:, 3] += right.to(inds.dtype)
    dw_right = torch.column_stack((w3, -3 * w3, 3 * w3, -w3))
    weights[right] += dw_right[right]

    # points really on the right boundary
    right = left_inds == (n - 1)
    weights[right] = 0
    weights[right, 1] = 1.0
    # inds[:, 3] += right.to(inds.dtype)

    keep = torch.logical_not(torch.logical_or(inds < 0, inds >= n))
    keep = inds_eval, _ = torch.nonzero(keep, as_tuple=True)
    inds_grid = inds[keep]
    weights = weights[keep]

    return inds_grid, inds_eval, weights


def left_cubic_interpolation_matrix(x, xeval, dim=0):
    inds_grid, inds_eval, weights = cubic_interpolation_kernel(x, xeval)
    indices = torch.row_stack((inds_eval, inds_grid))
    left_interp_matrix = torch.sparse_coo_tensor(
        indices,
        weights,
        size=(xeval.shape[dim], x.shape[dim]),
    )
    # can't csr, no grads in torch for csr, but they have coo grad
    # might be better to do dense?
    # left_interp_matrix = left_interp_matrix.to_sparse_csr()
    left_interp_matrix = left_interp_matrix.to_dense()
    return left_interp_matrix


def _read_by_chunk(mask, dataset):
    """
    mask : boolean array of shape dataset.shape[:1]
    dataset : chunked h5py.Dataset
    """
    out = np.empty((mask.sum(), *dataset.shape[1:]), dtype=dataset.dtype)
    n = 0
    for sli, *_ in dataset.iter_chunks():
        m = np.flatnonzero(mask[sli])
        nm = m.size
        if not nm:
            continue
        x = dataset[sli][m]
        # x = dataset[np.arange(sli.start, sli.stop)[m]]
        out[n : n + nm] = x
        n += nm
    return out


def _channel_subset_by_chunk(
    mask, dataset, channels, original_channel_index, new_channel_index
):
    out = np.empty(
        (mask.sum(), dataset.shape[1], new_channel_index.shape[1]), dtype=dataset.dtype
    )
    n = 0
    for sli, *_ in dataset.iter_chunks():
        m = np.flatnonzero(mask[sli])
        nm = m.size
        if not nm:
            continue
        x = dataset[sli][m]
        # x = dataset[np.arange(sli.start, sli.stop)[m]]
        out[n : n + nm] = waveform_util.channel_subset_by_index(
            x, channels[n : n + nm], original_channel_index, new_channel_index
        )
        n += nm
    return out
