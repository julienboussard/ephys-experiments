import dataclasses
from typing import Optional, Union

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


class SpikeData(torch.nn.Module):
    """Data bag to keep things tidy."""

    def __init__(
        self,
        n_chans_full: int,
        waveform_rank: int,
        n_chans_unit: int,
        n_chans_waveform: int,
        n_spikes: int,
        keepers: np.array,
        original_labels: np.array,
        channels: np.array,
        times_seconds: np.array,
        original_channel_index: np.array,
        waveform_channel_index: np.array,
        cluster_channel_index: np.array,
        static_amp_vecs: torch.Tensor,
        tpca_embeds: Union[np.array, h5py.Dataset],
        spike_static_channels: np.array,
        registered_geom: np.array,
        in_memory: bool,
        pin: bool = True,
    ):
        super().__init__()
        self.n_chans_full = n_chans_full
        self.waveform_rank = waveform_rank
        self.n_chans_unit = n_chans_unit
        self.n_chans_waveform = n_chans_waveform
        self.n_spikes = n_spikes

        # arrays not needed in torch
        self.original_labels = original_labels
        self.original_channel_index = original_channel_index
        self.waveform_channel_index = waveform_channel_index

        # CPU tensors
        self.keepers = torch.from_numpy(keepers)
        self.static_amp_vecs = torch.tensor(static_amp_vecs)
        if pin:
            self.static_amp_vecs = self.static_amp_vecs.pin_memory()
        self.tpca_embeds = tpca_embeds
        if pin and self.in_memory:
            self.tpca_embeds = self.tpca_embeds.pin_memory()

        # GPU
        self.register_buffer("registered_geom", torch.tensor(registered_geom))
        self.register_buffer(
            "cluster_channel_index", torch.tensor(cluster_channel_index)
        )
        self.register_buffer("times_seconds", torch.tensor(times_seconds))
        self.register_buffer("channels", torch.tensor(channels))

    def get_waveforms(self, index):
        if self.in_memory:
            waveforms = self.tpca_embeds[index]
        else:
            scalar = np.isscalar(index)
            index = np.atleast_1d(index)
            indices = self.keepers[index]
            mask = np.zeros(self.tpca_embeds.shape[0], dtype=bool)
            mask[indices] = 1
            waveforms = _channel_subset_by_chunk(
                mask,
                self.tpca_embeds,
                self.channels[index],
                self.original_channel_index,
                self.waveform_channel_index,
            )
            if scalar:
                waveforms = waveforms[0]
        return waveforms


default_fa_kwargs = dict(
    latent_update="embed_uninterp",
    do_prior=False,
)

default_fa_fit_kwargs = dict(
    lr=0.05,
    eps=1e-6,
    show_progress=True,
    n_iter=100,
    loss_converged=1e-2,
)

default_residual_pca_kwargs = dict(
    centered=False,
    atol=0.1,
    max_iter=25,
)


class InterpUnit(torch.nn.Module):
    """InterpUnit

    This is the middleman between the overall clusterer and the actual models.
    It knows how to deal with the channels the unit lives on, converting wfs
    to those channels before passing them to the model.
    """

    def __init__(
        self,
        t_bounds,
        waveform_rank,
        n_chans_unit,
        n_chans_full,
        min_overlap=0.5,
        residual_pca_rank=2,
        do_interp=True,
        amplitude_scaling_std=np.sqrt(0.001),
        amplitude_scaling_limit=1.2,
        fa_kwargs=default_fa_kwargs,
        residual_pca_kwargs=default_residual_pca_kwargs,
        fa_fit_kwargs=default_fa_fit_kwargs,
    ):
        super().__init__()
        self.residual_pca_rank = residual_pca_rank
        self.do_interp = do_interp
        self.min_overlap = min_overlap
        self.n_chans_full = n_chans_full
        self.n_chans_unit = n_chans_unit
        self.waveform_rank = waveform_rank
        self.input_dim = waveform_rank * self.n_chans_unit

        pca_centered = True
        if do_interp:
            self.interp = CubicInterpFactorAnalysis(
                t_bounds, output_dim=self.input_dim, **fa_kwargs
            )
            self.fa_fit_kwargs = fa_fit_kwargs
            pca_centered = False
        pca_kwargs = residual_pca_kwargs | dict(centered=pca_centered)
        self.pca = MaskedPCA(
            input_dim=self.input_dim, rank=residual_pca_rank, **pca_kwargs
        )

        # unit channels logic
        self.needs_fit = True

        self.register_buffer(
            "inv_lambda", torch.tensor(1.0 / (amplitude_scaling_std**2))
        )
        self.register_buffer(
            "scale_clip_low", torch.tensor(1.0 / amplitude_scaling_limit)
        )
        self.register_buffer("scale_clip_high", amplitude_scaling_limit)

    def _needs_to_be_fitted(self):
        assert not self.needs_fit

    def determine_position_(self, static_amp_vecs, geom, cluster_channel_index):
        assert cluster_channel_index.shape == (self.n_chans_full, self.n_chans_uni)

        count = torch.sqrt(torch.isfinite(static_amp_vecs).sum(0))
        snr = torch.nan_to_num(torch.nanmean(static_amp_vecs, dim=0)) * count
        self.com = (snr * geom[:, 1]).sum() / snr.sum()
        self.max_channel = snr.argmax()

        channel_reindexer = torch.full(
            self.n_chans_full, self.n_chans_unit, device=cluster_channel_index.device
        )
        my_chans = cluster_channel_index[self.max_channel]
        self.channels = my_chans.clone()
        my_valid = my_chans < self.n_chans_full
        my_ixs = torch.arange(self.n_chans_unit)[my_valid]
        channel_reindexer[my_chans[my_valid]] = my_ixs
        if hasattr(self, "channel_reindexer"):
            self.channel_reindexer.copy_(channel_reindexer)
        else:
            self.register_buffer("channel_reindexer", channel_reindexer)

    def overlaps(self, static_channels):
        """
        Arguments
        ---------
        static_channels : (n_spikes, n_chans_wf)

        Returns
        -------
        overlaps : (n_spikes)
        rel_ix : (n_spikes, n_chans_unit)
        """
        rel_ix = self.rel_ix(static_channels)
        overlap_num = (rel_ix < self.n_chans_unit).sum(1)
        overlap_den = (static_channels < self.n_chans_full).sum(1)
        overlaps = overlap_num / overlap_den
        return overlaps, rel_ix

    def rel_ix(self, static_channels):
        """
        Arguments
        ---------
        static_channels : (n_spikes, n_chans_wf)

        Returns
        -------
        rel_ix : (n_spikes, n_chans_unit)
        """
        rel_ix = torch.take_along_dim(
            self.unit_reindexers[None],
            static_channels,
            axis=1,
        )
        return rel_ix

    def get_means(self, times=None, padded=False, constant_value=torch.nan):
        self._needs_to_be_fitted()
        single = times is None
        if not single:
            n = len(times)
        if self.do_interp:
            assert not single
            _, means_flat = self.interp(times)
            means = means_flat.view(n, self.waveform_rank, self.n_chans_unit)
            if padded:
                means = F.pad(means, (0, 1), value=constant_value)
        else:
            mean_flat = self.pca.mean
            mean = mean_flat.view(self.waveform_rank, self.n_chans_unit)
            if padded:
                mean = F.pad(mean, (0, 1), value=constant_value)
            if not single:
                means = mean[None].broadcast_to(n, *mean.shape).contiguous()

        return means

    def to_unit_channels(
        self,
        waveforms,
        times=None,
        waveform_channels=None,
        rel_ix=None,
        fill_mode="mean",
        constant_value=torch.nan,
    ):
        """Shift waveform from its channels to this unit's

        How to fill in the blanks? fill_mode="mean" uses this unit's possibly time-varying
        mean. fill_mode="constant" uses constant_value.
        """
        if rel_ix is None:
            rel_ix = self.rel_ix(waveform_channels)

        if fill_mode == "mean":
            waveforms_rel = self.get_means(
                times, padded=True, constant_value=constant_value
            )
        elif fill_mode == "constant":
            waveforms_rel = torch.full(
                (len(waveforms), self.waveform_rank, self.n_chans_unit + 1),
                constant_value,
                dtype=waveforms.dtype,
                device=waveforms.device,
            )

        n, r, c = waveforms.shape
        rel_ix_scatter = rel_ix[:, None, :].broadcast_to((n, r, rel_ix.shape[-1]))
        waveforms_rel.scatter_(src=waveforms, dim=2, index=rel_ix_scatter)
        return waveforms_rel

    def to_waveform_channels(self, waveforms_rel, waveform_channels=None, rel_ix=None):
        if rel_ix is None:
            rel_ix = self.rel_ix(waveform_channels)

        n = waveforms_rel.shape[0]
        waveforms_rel = waveforms_rel.reshape(n, -1, self.n_chans_unit)
        waveforms_rel = F.pad(waveforms_rel, (0, 1))
        rel_ix = rel_ix[:, None, :].broadcast_to(
            (n, self.waveform_rank, rel_ix.shape[-1])
        )
        return torch.gather(waveforms_rel, dim=2, index=rel_ix)

    def residual_embed(
        self,
        times,
        waveforms,
        waveform_channels,
        overlaps=None,
        rel_ix=None,
        batch_size=256,
        out=None,
    ):
        self._needs_to_be_fitted()
        if rel_ix is None:
            overlaps, rel_ix = self.overlaps(waveform_channels)
        n = len(waveforms)
        if out is None:
            out = torch.empty(
                (n, self.residual_pca_rank),
                dtype=waveforms.dtype,
                device=waveforms.device,
            )

        for j in range(0, n, batch_size):
            sl = slice(j, min(j + batch_size, n))
            means_rel = self.get_means(times[sl], padded=True)
            resids = self.to_unit_channels(waveforms=waveforms[sl], rel_ix=rel_ix[sl])
            resids.sub_(means_rel)
            self.pca.forward_precentered(resids, out=out[sl])

        return out

    def spike_badnesses(
        self,
        times,
        waveforms,
        waveform_channels,
        kinds=("l2", "1-r^2", "1-scaledr^2"),
        overlaps=None,
        rel_ix=None,
    ):
        self._needs_to_be_fitted()

        # a client may already know exactly which spikes they want to compare
        spike_ix = slice(None)
        if rel_ix is None:
            overlaps, rel_ix = self.overlaps(waveform_channels)
            (spike_ix,) = torch.nonzero(overlaps >= self.min_overlap, as_tuple=True)
            waveforms = waveforms[spike_ix]
            rel_ix = rel_ix[spike_ix]
            overlaps = overlaps[spike_ix]

        # masks are all or nothing along axis=1
        mask = torch.isfinite(waveforms[:, 0, :]).unsqueeze(1).to(waveforms)
        waveforms = torch.nan_to_num(waveforms)

        # try to reconstruct spikes
        recons_rel = self.get_means(times, padded=True)
        if times is None:
            recons_rel = recons_rel[None]
        recons = self.to_waveform_channels(recons_rel, rel_ix=rel_ix)
        recons = recons * mask

        badnesses = {}
        l2 = waveforms.sub(recons).square().sum(dim=(1, 2))
        if "l2" in kinds:
            badnesses["l2"] = l2
        if any("r^2" in k for k in kinds):
            wf_l2 = waveforms.square().sum(dim=(1, 2))
        if "1-r^2" in kinds:
            badnesses["1-r^2"] = l2 / wf_l2
        if "1-scaledr^2" in kinds:
            dots = recons.mul(waveforms).sum(dim=(1, 2))
            recons_sumsq = recons.square().sum(dim=(1, 2))
            scalings = (dots + self.inv_lambda).div_(recons_sumsq + self.inv_lambda)
            scalings = scalings.clip_(self.scale_clip_low, self.scale_clip_high)
            scaled_l2 = waveforms.sub(scalings * recons).square().sum(dim=(1, 2))
            badnesses["1-scaledr^2"] = scaled_l2 / wf_l2

        return spike_ix, overlaps, badnesses

    def divergence(
        self,
        other,
        kind="1-scaledr^2",
        aggregate=torch.amax,
        min_overlap=0.5,
        subset_channel_index=None,
    ):
        """Try to explain other units' mean (traces)"""
        other_channels = other.channels
        if subset_channel_index is not None:
            other_channels = subset_channel_index[other.max_channel]

        overlaps, rel_ix = self.overlaps(other_channels[None])
        if overlaps.squeeze() < min_overlap:
            return torch.inf

        if not self.do_interp:
            # simple static case
            other_waveform = other.get_means()[None]
            if subset_channel_index is not None:
                other_waveform = other.to_waveform_channels(
                    other_waveform,
                    waveform_channels=other_channels,
                )
            _, _, badnesses = self.spike_badnesses(
                times=None,
                waveforms=other_waveform,
                kinds=(kind,),
                overlaps=overlaps,
                rel_ix=rel_ix,
            )
            div = badnesses[kind].squeeze()
            return div

        # find grid times in common
        common_grid = torch.logical_and(
            self.interp.grid_fitted,
            other.interp.grid_fitted,
        )
        if not common_grid.any():
            return torch.inf
        common_times = self.grid[common_grid]

        # compare
        other_waveforms = other.get_means(common_times)
        if subset_channel_index is not None:
            other_waveforms = other.to_waveform_channels(
                other_waveforms,
                waveform_channels=other_channels,
            )
        _, _, badnesses = self.spike_badnesses(
            times=common_times,
            waveforms=other_waveforms,
            waveform_channels=other_channels[None],
            kinds=(kind,),
            overlaps=overlaps,
            rel_ix=rel_ix,
        )
        div = badnesses[kind].squeeze()

        # aggregate over time
        div = aggregate(div)

        return div

    def fit(
        self,
        times,
        waveforms,
        waveform_channels,
        static_amp_vecs,
        geom,
        cluster_channel_index,
    ):
        # transfer waveform -> unit channels, filling with nans
        self.determine_position_(static_amp_vecs, geom, cluster_channel_index)
        n = len(times)
        rel_ix = self.rel_ix(waveform_channels)
        waveforms_rel = self.to_unit_channels(
            waveforms,
            times,
            rel_ix=rel_ix,
            fill_mode="constant",
            constant_value=torch.nan,
        )
        waveforms_flat = waveforms_rel.reshape(n, -1)

        # fit/transform with the interpolator
        residuals_flat = waveforms_flat
        if self.do_interp:
            self.interp.fit(
                times,
                waveforms_flat,
                **self.fa_fit_kwargs,
            )
            _, means_flat = self.interp(times)
            residuals_flat = waveforms_flat - means_flat

        self.pca.fit(residuals_flat)
        self.needs_fit = False


@dataclasses.dataclass
class DPCSplitKwargs:
    rank: int = 2
    sigma_local: Union[str, float] = "rule_of_thumb"
    sigma_regional: Optional[float] = None
    n_neighbors_search: int = 500
    allow_single_cluster_outlier_removal: bool = True
    recursive: bool = True
    # split_on_train: bool = False


class InterpClusterer(torch.nn.Module):
    """Mixture of InterpUnits."""

    def __init__(
        self,
        sorting,
        t_bounds,
        motion_est=None,
        fit_radius=35.0,
        waveform_radius=25.0,
        min_overlap=0.5,
        residual_pca_rank=2,
        min_cluster_size=50,
        n_spikes_fit=2048,
        do_interp=True,
        fa_kwargs=default_fa_kwargs,
        residual_pca_kwargs=default_residual_pca_kwargs,
        split_kwargs=DPCSplitKwargs(),
        in_memory=True,
        keep_spikes="byamp",
        max_n_spikes=5000000,
        reassign_metric="1-scaledr^2",
        merge_metric="1-scaledr^2",
        merge_threshold=0.25,
        merge_sym_function=torch.maximum,
        merge_linkage="complete",
        merge_on_waveform_radius=True,
        outlier_explained_var=0.0,
        rg=0,
    ):
        self.min_cluster_size = min_cluster_size
        self.n_spikes_fit = n_spikes_fit
        self.rg = np.random.default_rng(rg)

        self.data = _load_data(
            sorting,
            motion_est,
            fit_radius,
            wf_radius=waveform_radius,
            in_memory=in_memory,
            whiten_input=False,
            keep=keep_spikes,
            max_n_spikes=max_n_spikes,
            rg=self.rg,
        )
        self.residual_pca_rank = residual_pca_rank
        self.unit_kw = dict(
            t_bounds=t_bounds,
            n_chans_unit=self.data.n_chans_unit,
            waveform_rank=self.data.waveform_rank,
            min_overlap=min_overlap,
            residual_pca_rank=residual_pca_rank,
            do_interp=do_interp,
            fa_kwargs=fa_kwargs,
            residual_pca_kwargs=residual_pca_kwargs,
        )

        # torch.manual_seed(self.rg.bit_generator.random_raw())
        self.labels = torch.tensor(self.data.original_labels)
        self.models = torch.nn.ModuleDict()
        self.register_buffer("_device", torch.tensor(0))
        self.split_kw = split_kwargs

        self.reassign_metric = reassign_metric
        self.merge_metric = merge_metric
        self.merge_threshold = merge_threshold
        self.merge_linkage = merge_linkage
        self.min_overlap = min_overlap
        self.merge_sym_function = merge_sym_function
        self.merge_on_waveform_radius = merge_on_waveform_radius
        self.outlier_explained_var = outlier_explained_var

        self.m_step()
        self.order_by_depth()

    @property
    def device(self):
        return self._device.device

    def update_labels(self, old_labels, new_labels=None, flat=False):
        """
        Re-label units. This should not split labels, but merging is OK.

        Arguments
        ---------
        old_labels : (n_units,)
        new_labels : (n_units,)
        """
        # invariant:
        # labels in self.labels **always** match keys in self.models
        # this means that things which change the meaning of labels
        # must update both structures
        if new_labels is None:
            new_labels = torch.arange(len(old_labels))

        if self.models:
            new_models = {}
            for oldk, newk in zip(old_labels, new_labels):
                if newk not in new_models:
                    new_models[newk] = self.models[oldk]
            self.models.clear()
            self.models.update(new_models)

        if flat:
            kept = self.labels >= 0
            label_indices = self.labels[kept]
        else:
            kept = self.labels >= 0
            label_indices = torch.searchsorted(old_labels, self.labels[kept])
            kept = torch.logical_and(
                kept,
                self.labels[kept]
                == old_labels[label_indices.clip(0, old_labels.shape[0] - 1)],
            )
        self.labels[kept] = new_labels[label_indices[kept]]
        self.labels[torch.logical_not(kept)] = -1

    def cleanup(self):
        """Remove small units and make labels contiguous."""
        old_labels, counts = torch.unique(self.labels, return_counts=True)
        big_enough = counts >= self.min_cluster_size
        n_removed = torch.logical_not(big_enough).sum()
        if n_removed:
            print(f"Removed {n_removed} too-small units.")
        self.update_labels(old_labels[big_enough], flat=False)

    def order_by_depth(self):
        """Reorder labels by unit CoM depth."""
        if not self.models:
            return

        order = np.argsort([u.com for u in self.models])
        # this second argsort never fails to impress (my brain into a small cube)
        self.update_labels(
            self.unit_ids(),
            torch.argsort(torch.from_numpy(order)),
            flat=True,
        )

    def m_step(self, force=False):
        """Fit all models that need fitting."""
        for uid in self.unit_ids():
            if uid > len(self.models):
                unit = InterpUnit(**self.unit_kw)
                unit.to(self.device)
                self.models.append(unit)
            model = self.models[uid]

            if not (force or model.needs_fit):
                continue

            model.fit(**self.get_training_data(uid))

    def residual_dpc_split(self):
        unit_ids_to_split = list(self.unit_ids())

        while unit_ids_to_split:
            next_ids_to_split = []
            for uid in unit_ids_to_split:
                next_ids_to_split.extend(self.dpc_split_unit(uid))
            self.m_step()
        self.cleanup()
        self.order_by_depth()

    def dpc_split_unit(self, uid):
        """
        Updates state by adding new models and updating labels etc after splitting a unit.

        Returns
        -------
        list of new IDs to split
        """
        # invariant: maintains contiguous label space of big-enough units

        # -- featurize full set of spikes for this unit
        (in_unit,) = (self.labels == uid).nonzero(as_tuple=True)
        n = in_unit.numel()
        unit = self.models[uid]
        features = torch.empty((n, self.residual_pca_rank), device=self.device)
        for sl, data in self.batches(in_unit):
            unit.residual_embed(**data, out=features[sl])

        # -- back to CPU for DPC
        features = features[:, : self.split_kw.rank].numpy(force=True)
        try:
            split_labels = density.density_peaks_clustering(
                features,
                sigma_local=self.split_kw.sigma_local,
                n_neighbors_search=self.split_kw.n_neighbors_search,
                remove_clusters_smaller_than=self.min_cluster_size,
            )
        except ValueError as e:
            print(e)
            return []
        del features

        # -- deal with relabeling
        split_units, counts = np.unique(split_labels, return_counts=True)
        n_split_full = split_units.size
        counts = counts[split_units >= 0]
        assert (counts >= self.min_cluster_size).all()
        split_units = split_units[split_units >= 0]
        n_split = split_units.size
        # case 0: nothing happened.
        if n_split == n_split_full == 1:
            return []
        # case 1: single-unit outlier removal. re-fit but don't re-split.
        if n_split_full - 1 == n_split == 1:
            if self.split_kw.allow_single_cluster_outlier_removal:
                self.labels[in_unit[split_labels < 0]] = -1
                unit.needs_fit = True
                return []
        # case 2: something legitimately took place.
        # here, split_unit 0 retains label uid. split units >=1 get new labels.
        assert n_split in (n_split_full, n_split_full + 1)
        assert n_split > 1
        assert split_units[0] == 0
        self.labels[in_unit] = -1
        new_unit_ids = (uid, *(self.labels.max() + np.arange(1, n_split)))
        for split_label, new_label in zip(split_units, new_unit_ids):
            in_split = in_unit[split_labels == split_label]
            self.labels[in_split] = new_label
        return split_units

    def central_divergences(self, kind=None, min_overlap=0.5):
        if kind is None:
            kind = self.merge_metric
        subset_channel_index = None
        if self.merge_on_waveform_radius:
            subset_channel_index = (self.data.waveform_channel_index,)
        units = self.unit_ids()
        nu = units.numel()
        divergences = torch.full((nu, nu), torch.nan)
        for i, ua in enumerate(units):
            for j, ub in enumerate(units):
                if ua == ub:
                    divergences[i, j] = 0
                    continue
                divergences[i, j] = self.models[ua].divergence(
                    self.models[ub],
                    kind=kind,
                    min_overlap=min_overlap,
                    subset_channel_index=subset_channel_index,
                )
        return divergences

    def merge(self):
        merge_dists = self.central_divergences(
            kind=self.merge_metric,
            min_overlap=self.min_overlap,
        )
        merge_dists = self.merge_sym_function(merge_dists, merge_dists.T)
        merge_dists = merge_dists.numpy(force=True)
        merge_dists[np.isinf(merge_dists)] = merge_dists.max() + 10
        d = merge_dists[np.triu_indices(merge_dists.shape[0], k=1)]
        Z = linkage(d, method=self.merge_linkage)
        new_labels = fcluster(Z, self.merge_threshold, criterion="distance")
        unique_new_labels = np.unique(new_labels)
        print(f"Merge: {merge_dists.shape[0]} -> {unique_new_labels.size}")

        # update state
        self.update_labels(self.unit_ids(), new_labels)
        self.order_by_depth()

    def reassignment_divergences(self):
        unit_ids = self.unit_ids()
        nu = unit_ids.size

        dtype = self.data.tpca_embeds.dtype
        shape = (nu, self.data.n_spikes)
        divergences = dok_array(shape, dtype=dtype)

        for j, uid in enumerate(self.models.keys()):
            unit = self.models[uid]
            overlaps, rel_ix = unit.overlaps(self.data.spike_static_channels)
            (which,) = torch.nonzero(overlaps >= self.min_overlap)
            if not which.numel():
                continue
            overlaps = overlaps[which]
            rel_ix = rel_ix[which]

            for sl, batch in self.batches(which):
                res = unit.spike_badnesses(
                    **batch,
                    overlaps=overlaps[sl],
                    rel_ix=rel_ix[sl],
                    kinds=(self.reassign_metric,),
                )
                divergences[j, which[sl]] = res[self.reassign_metric]

        return divergences

    def reassign(self):
        divergences = self.reassignment_divergences().tocsc()
        # sparse nonzero rows. this is CSC format specific.
        has_match = np.diff(divergences.indptr) > 0
        match_threshold = 1.0 - self.outlier_explained_var

        # we want sparse 0s to mean infinite err, and divs>thresh
        # to be infinite as well. right now, [0, 1] with 0 as best
        # subtract M to go to [-M, -M + 1].
        errs = divergences.copy()
        # errs.data[errs.data > match_threshold] += 212.0
        errs.data[errs.data <= match_threshold] -= match_threshold - 212.0
        new_labels = np.where(
            has_match,
            errs.argmin(0),
            -1,
        )
        kept = np.flatnonzero(has_match)
        new_labels[errs[new_labels[kept], kept] >= 0] = -1

        self.labels = new_labels
        self.cleanup()

    def unit_ids(self):
        ids = torch.unique(self.labels)
        ids = ids[ids >= 0]
        assert torch.equal(ids, torch.arange(ids.numel()))
        assert torch.equal(
            ids, torch.sort(torch.tensor([uid for uid in self.models.keys()]))
        )
        return ids

    def get_training_data(self, uid):
        (in_unit,) = (self.labels == uid).nonzero(as_tuple=True)
        ns = in_unit.numel()
        if ns > self.n_spikes_fit:
            which = self.rg.choice(ns, size=self.n_spikes_fit, replace=False)
            which.sort()
            in_unit = in_unit[torch.from_numpy(which)]
        train_data = self.spike_data(in_unit)
        train_data["static_amp_vecs"] = self.data.static_amp_vecs[in_unit]
        train_data["geom"] = self.data.registered_geom
        train_data["cluster_channel_index"] = self.data.cluster_channel_index
        return train_data

    def spike_data(self, which):
        return dict(
            times=self.data.times_seconds[which],
            waveforms=self.data.get_waveforms(which, device=self.device),
            waveform_channels=self.data.spike_static_channels[which],
        )

    def batches(self, indices, batch_size=256):
        for j in range(0, len(indices), batch_size):
            sl = slice(j, min(j + batch_size, len(indices)))
            yield sl, self.spike_data(indices[sl])


# -- core classes


class MaskedPCA(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        rank,
        max_iter=100,
        check_every=5,
        n_oversamples=10,
        atol=1e-3,
        show_progress=False,
        centered=True,
        transform_iter=0,
    ):
        super().__init__()
        self.fit_kw = dict(
            max_iter=max_iter,
            check_every=check_every,
            n_oversamples=n_oversamples,
            atol=atol,
            show_progress=show_progress,
            centered=centered,
        )
        self.rank = rank
        self.transform_iter = transform_iter
        self.train_loadings = None
        self.centered = centered
        if centered:
            self.register_buffer("mean", torch.empty(input_dim))
        self.register_buffer("weight", torch.empty((rank, input_dim)))
        self.register_buffer("svs", torch.empty((rank)))

    def fit(self, waveforms):
        missing = torch.isnan(waveforms)
        empty = missing.all(1)
        loadings, mean, components, svs = fit_pcas(
            waveforms,
            missing,
            empty,
            **self.fit_kw,
        )
        self.train_loadings = loadings
        self.weight.copy_(components)
        self.mean.copy_(mean)
        self.svs.copy_(svs)

    def forward_precentered(self, waveforms, out=None):
        torch.matmul(waveforms, self.weight.T, out=out)

    def forward(self, waveforms):
        if self.centered:
            waveforms = waveforms - self.mean
        return waveforms @ self.weight.T

    def backward(self, embeds):
        if self.centered:
            return torch.addmm(self.mean, embeds, self.weight)
        return embeds @ self.weight

    def transform(self, waveforms):
        mask = torch.isfinite(waveforms)
        waveforms = torch.where(
            mask,
            waveforms,
            self.mean,
        )

        if self.transform_iter == 0:
            return self.forward(waveforms)

        for j in range(self.transform_iter):
            embeds = self.forward(waveforms)
            recons = self.backward(embeds)
            waveforms = torch.where(
                mask,
                waveforms,
                recons,
                out=waveforms,
            )

        return embeds


class CubicInterpFactorAnalysis(torch.nn.Module):
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
        min_lengthscale=1e-5,
        learn_prior_noise_fraction=False,
        learn_obsstd=True,
        loss_on_interp=False,
        latent_update="gradient",
        do_prior=True,
        fitted_point_count=25,
        fitted_point_fraction="grid",
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
        elif latent_update == "embed_uninterp":
            self.register_buffer("grid_z", grid_z)
        else:
            assert False

        self.fitted_point_count = fitted_point_count
        self.fitted_point_fraction = fitted_point_fraction
        if fitted_point_fraction == "grid":
            self.fitted_point_fraction = 1 / self.grid_size

        self.do_prior = do_prior
        self.learn_lengthscale = learn_lengthscale
        self.learn_prior_noise_fraction = learn_prior_noise_fraction
        self.compute_prior_in_loss = False
        self.init_kernel(lengthscale, min_lengthscale)
        if self.do_prior:
            self.register_buffer("prior_mean", torch.zeros(()))
            prior_noiselogit = prior_noiselogit + torch.zeros(())
            if learn_prior_noise_fraction:
                self.register_parameter(
                    "prior_noiselogit", torch.nn.Parameter(prior_noiselogit)
                )
            else:
                self.register_buffer("prior_noiselogit", prior_noiselogit)
            self._interp_id = None
            self._cached_priordist = None
            if self.learn_lengthscale or self.learn_prior_noise_fraction:
                self.compute_prior_in_loss = True
            if self.latent_update == "gradient":
                self.compute_prior_in_loss = True

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
        self._unbias = None
        self._unweight = None

    def init_kernel(self, lengthscale, min_lengthscale):
        assert lengthscale >= min_lengthscale
        if not self.learn_lengthscale:
            Kuu = RBF(lengthscale)(self.grid)
            self._grid_cov = Kuu
            if self.do_prior:
                grid_scale_left = np.linalg.cholesky(Kuu)
                self.register_buffer(
                    "_grid_scale_left", torch.tensor(grid_scale_left, dtype=torch.float)
                )
        else:
            self.register_buffer(
                "_half_sq_dgrid", -0.5 * torch.square(self.grid - self.grid.T)
            )

        lengthscale = torch.tensor(lengthscale)
        min_lengthscale = torch.tensor(min_lengthscale)
        self.register_buffer("min_lengthscale", min_lengthscale)
        if self.learn_lengthscale:
            lengthscale = lengthscale - self.min_lengthscale
            if lengthscale < 20.0:
                # invert softplus
                lengthscale = lengthscale.expm1().log()
            self.register_parameter("_lengthscale", torch.nn.Parameter(lengthscale))
        else:
            self.register_buffer("_lengthscale", lengthscale)

    def lengthscale(self):
        if self.learn_lengthscale:
            return F.softplus(self._lengthscale) + self.min_lengthscale
        return self._lengthscale

    def grid_cov(self):
        if not self.learn_lengthscale:
            return self._grid_cov

        Kuu = self._half_sq_dgrid / self.lengthscale().square()
        Kuu = Kuu.exp()
        return Kuu

    def grid_scale_left(self, eps=1e-4):
        if not self.learn_lengthscale:
            return self._grid_scale_left
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
        assert self.training

        if self.do_prior and prior_dist is None:
            prior_dist = self.get_prior_distribution(left_interp_matrix)
        if left_interp_pinv is None:
            left_interp_pinv = torch.linalg.pinv(left_interp_matrix)
        z = self.embed(y)
        z = left_interp_pinv @ z
        if self.do_prior:
            assert False
            # z = torch.linalg.solve_triangular(prior_dist._capacitance_tril, z, upper=False)
            # z = torch.cholesky_solve(z, prior_dist._capacitance_tril, )
            # z = left_interp_matrix @ z
            # z = left_interp_pinv @ z
        self.grid_z.copy_(z)

    def embed(self, y):
        if self.training:
            unweight = torch.linalg.pinv(self.net.weight).T
        else:
            if self._unweight is None:
                self._unweight = torch.linalg.pinv(self.net.weight).T
            unweight = self._unweight
        return (y - self.net.bias) @ unweight

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
        if self.compute_prior_in_loss:
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

    def fit(
        self,
        train_t,
        train_y,
        lr=0.05,
        eps=1e-6,
        show_progress=True,
        n_iter=100,
        loss_converged=1e-2,
    ):
        assert self.training

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

        # check which grid points had enough spikes
        grid_neighbors = torch.cdist(
            train_t[:, None],
            self.grid[:, None],
            p=1,
        ).argmin(dim=1)
        histogram = torch.zeros_like(self.grid)
        histogram.scatter_add_(
            0,
            grid_neighbors,
            torch.ones(1).broadcast_to(grid_neighbors.shape),
        )
        self.register_buffer(
            "grid_fitted",
            torch.logical_or(
                histogram >= self.fitted_point_count,
                histogram / histogram.sum() >= self.fitted_point_fraction,
            ),
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
    keep="all",
    max_n_spikes=5000000,
    rg=0,
):
    rg = np.random.default_rng(rg)

    # load up labels
    labels = sorting.labels
    if keep == "labeled":
        keep_mask = labels >= 0
    elif keep == "all":
        keep_mask = np.ones(labels.shape, dtype=bool)
    elif keep == "byamp":
        keep_mask = labels >= 0
        a = sorting.denoised_ptp_amplitudes
        keep_mask = np.logical_or(
            keep_mask,
            a >= np.median(a[keep_mask]),
        )
    keepers = np.flatnonzero(keep_mask)

    if max_n_spikes and keepers.size > max_n_spikes:
        print(f"Subsampling to {max_n_spikes} ({100*(max_n_spikes/labels.size):0.1f}%)")
        keepers = rg.choice(keepers, size=max_n_spikes, replace=False)
        keepers.sort()
        keep_mask = np.zeros_like(keep_mask)
        keep_mask[keepers] = 1

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
    n_chans_full = len(registered_geom)
    n_chans_waveform = channel_index.shape[1]
    n_chans_unit = cluster_channel_index.shape[1]
    waveform_rank = tpca_embeds.shape[1]
    n_spikes = keepers.size
    n_pitches_shift = drift_util.get_spike_pitch_shifts(
        sorting.point_source_localizations[:, 2][keepers],
        geom=geom,
        motion_est=motion_est,
        times_s=times_seconds,
    )

    # where a channel is not present, this has n_chans_full
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
        n_chans_full,
    )

    return SpikeData(
        keepers=keepers,
        original_labels=labels,
        channels=channels,
        times_seconds=times_seconds,
        waveform_rank=waveform_rank,
        n_chans_full=n_chans_full,
        n_chans_unit=n_chans_unit,
        n_chans_waveform=n_chans_waveform,
        n_spikes=n_spikes,
        original_channel_index=original_channel_index,
        waveform_channel_index=channel_index,
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
