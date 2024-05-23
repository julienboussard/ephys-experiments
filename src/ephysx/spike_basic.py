from typing import Union
import dataclasses

from tqdm.auto import tqdm, trange

import h5py
import numpy as np
from scipy.sparse import dok_array
import torch
import torch.nn.functional as F
from dartsort.cluster import density
from dartsort.util import drift_util, waveform_util, data_util
from scipy.cluster.hierarchy import linkage, fcluster
from torch import nn

tqdm_kw = dict(smoothing=0, mininterval=1 / 24)


@dataclasses.dataclass
class SpikePCAData:
    """Data bag to keep things tidy."""

    keepers: np.array
    channels: np.array
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
        # wf = data_util.batched_h5_read(self.tpca_embeds, indices)
        wf = _read_by_chunk(mask, self.tpca_embeds)

        # grab static channels
        wf = waveform_util.channel_subset_by_index(
            wf, self.channels[index], self.original_channel_index, self.wf_channel_index
        )
        if scalar:
            wf = wf[0]
        return wf


def average_sym(A, AT):
    return 0.5 * (A + AT)


class BasicSpikePCAClusterer(nn.Module):
    def __init__(
        self,
        sorting,
        motion_est=None,
        rank=5,
        svd_atol=0.01,
        svd_max_iter=250,
        svd_n_oversamples=15,
        fit_radius=25.0,
        wf_radius=25.0,
        min_unit_size=50,
        n_wfs_fit=8192,
        min_overlap=0.5,
        spike_r2_threshold=0.0,
        buffer_growth_factor=1.5,
        train_units_batch_size=1,
        spike_batch_size=64,
        embedding_damping=0.0,
        time_varying=False,
        time_scale=100.0,
        time_regularization=10.0,
        time_rank=1,
        initially_clustered_only=False,
        centered=True,
        whiten_input=False,
        in_memory=True,
        split_rank=2,
        merge_rank=1,
        reassign_rank=1,
        split_impute_iters=5,
        merge_impute_iters=5,
        reassign_impute_iters=5,
        split_on_train=False,
        rg=0,
    ):
        super().__init__()
        self.rank = rank
        self.svd_atol = svd_atol
        self.svd_max_iter = svd_max_iter
        self.rg = np.random.default_rng(rg)
        # seed torch rg for svd_lowrank reproducibility
        torch.manual_seed(self.rg.bit_generator.random_raw())
        self.n_wfs_fit = n_wfs_fit
        self.min_unit_size = max(2 * self.rank + svd_n_oversamples, min_unit_size)
        self.buffer_growth_factor = buffer_growth_factor
        self.train_units_batch_size = train_units_batch_size
        self.min_overlap = min_overlap
        self.spike_batch_size = spike_batch_size
        self.svd_n_oversamples = svd_n_oversamples
        self.centered = centered
        self.spike_r2_threshold = spike_r2_threshold
        self.embedding_damping = embedding_damping
        self.split_rank = split_rank
        self.merge_rank = merge_rank
        self.reassign_rank = reassign_rank
        self.split_impute_iters = split_impute_iters
        self.split_on_train = split_on_train
        self.merge_impute_iters = merge_impute_iters
        self.reassign_impute_iters = reassign_impute_iters
        self.time_varying = time_varying
        self.time_scale = time_scale
        self.time_regularization = time_regularization
        self.time_rank = time_rank

        self.labels, self.data = _load_data(
            sorting,
            motion_est,
            fit_radius,
            wf_radius,
            in_memory,
            whiten_input,
            initially_clustered_only,
        )
        self.dim_input = self.data.tpca_rank * self.data.n_chans_cluster

        self.initialize()
        self.reindex()
        self.drop_small()

    def initialize(self):
        n_units = self.unit_ids().max() + 1
        U = int(np.ceil(n_units * self.buffer_growth_factor))
        self.register_buffer("components", torch.zeros((U, self.dim_input, self.rank)))
        self.register_buffer("mean", torch.zeros((U, self.dim_input)))
        self.register_buffer("svs", torch.zeros((U, self.rank)))
        self.channel_had_observations = np.zeros((U, self.data.n_chans_cluster), dtype=bool)
        self.wf_to_pc_reindexers = get_channel_structures(
            self.data.cluster_channel_index
        )
        self.train_loadings = np.full((U, self.n_wfs_fit, self.rank), np.nan)
        self.train_spike_indices = np.full((U, self.n_wfs_fit), -1)
        self.snrs = np.full((U, self.data.n_reg_chans), np.nan)
        self.main_channels = np.full(U, self.data.n_reg_chans)
        self.needs_fit = np.ones(U, dtype=bool)
        self.needs_split = np.ones(U, dtype=bool)
        self.chis = None

    def check_grow(self):
        unit_ids = self.unit_ids()
        n_units = unit_ids.max() + 1
        if n_units <= self.mean.shape[0]:
            return
        U = int(np.ceil(n_units * self.buffer_growth_factor))
        self.mean.resize_(U, *self.mean.shape[1:])
        self.components.resize_(U, *self.components.shape[1:])
        self.svs.resize_(U, *self.svs.shape[1:])
        self.train_loadings.resize(U, *self.train_loadings.shape[1:])
        self.channel_had_observations.resize(U, *self.channel_had_observations.shape[1:])
        self.train_spike_indices.resize(U, *self.train_spike_indices.shape[1:])
        self.snrs.resize(U, *self.snrs.shape[1:])
        self.main_channels.resize(U)
        self.needs_fit.resize(U)
        self.needs_split.resize(U)

    def reorder(self, oldix):
        k = len(oldix)
        self.mean[:k] = self.mean[oldix] + 0
        self.components[:k] = self.components[oldix] + 0
        self.svs[:k] = self.svs[oldix] + 0
        self.train_loadings[:k] = self.train_loadings[oldix] + 0
        self.channel_had_observations[:k] = self.channel_had_observations[oldix] + 0
        self.main_channels[:k] = self.main_channels[oldix] + 0
        self.main_channels[k:] = self.data.n_reg_chans
        self.snrs[:k] = self.snrs[oldix] + 0
        self.train_spike_indices[:k] = self.train_spike_indices[oldix] + 0
        self.train_spike_indices[k:] = -1
        self.needs_fit[:k] = self.needs_fit[oldix] + 0
        self.needs_split[:k] = self.needs_split[oldix] + 0
        if self.chis is not None:
            self.chis = self.chis[oldix]

    def drop_small(self):
        unit_ids, counts = np.unique(self.labels, return_counts=True)
        small = np.flatnonzero(counts < self.min_unit_size)
        if small.size:
            too_small = np.isin(self.labels, unit_ids[small])
            print(f"{too_small.mean()*100:0.1f}% in {small.size} too-small units")
            self.labels[too_small] = -1
        self.pack_units()

    def reindex(self):
        kept = np.flatnonzero(self.labels >= 0)
        flatix, flat_labels = np.unique(self.labels[kept], return_inverse=True)
        self.labels[kept] = flat_labels
        self.reorder(flatix)

    def pack_units(self):
        device = self.mean.device
        self.cpu()
        self.reindex()
        self.check_grow()

        # order by depth
        snrs = self.unit_channel_snrs()
        sums = snrs.sum(axis=1)
        empty = sums == 0
        sums[empty] = 1
        coms = (snrs * self.data.registered_geom[:, 1]).sum(1) / sums
        coms[empty] = self.data.registered_geom[:, 1].max() + 100
        order = np.argsort(coms)

        kept = np.flatnonzero(self.labels >= 0)
        self.labels[kept] = np.argsort(order)[self.labels[kept]]
        self.reorder(order)
        self.to(device)

    def unit_ids(self):
        uids = np.unique(self.labels)
        uids = uids[uids >= 0]
        return uids

    def unit_ids_to_fit(self):
        uids = self.unit_ids()
        return uids[self.needs_fit[: len(uids)]]

    def unit_ids_to_split(self):
        uids = self.unit_ids()
        return uids[self.needs_split[: len(uids)]]

    def unit_channel_snrs(self, unit_ids=None, no_count=False):
        if unit_ids is None:
            unit_ids = self.unit_ids()
        snrs = torch.zeros((unit_ids.size, self.data.n_reg_chans))
        for j, uid in enumerate(unit_ids):
            avs = self.data.static_amp_vecs[self.labels == uid]
            if no_count:
                count = 1
            else:
                count = torch.sqrt(torch.isfinite(avs).sum(0))
            snrs[j] = torch.nan_to_num(torch.nanmean(avs, dim=0)) * count
        return snrs.numpy(force=True)

    def m_step(self, unit_ids=None, force=False):
        #TODO: nans!!!!
        if unit_ids is None:
            if force:
                unit_ids = self.unit_ids()
            else:
                unit_ids = self.unit_ids_to_fit()
        bs = self.train_units_batch_size
        train_buffer = np.empty(
            (bs, self.n_wfs_fit, self.data.tpca_rank, self.data.n_chans_cluster + 1),
            dtype=np.float32,
        )
        self.chis = None
        for batch_start in trange(0, unit_ids.size, bs, desc="Fit"):
            batch = unit_ids[batch_start : min(unit_ids.size, batch_start + bs)]
            n_wfs_fit = self.n_wfs_fit
            if batch.size == 1:
                n_wfs_fit = -1

            self.snrs[batch] = self.unit_channel_snrs(batch)
            self.main_channels[batch] = self.snrs[batch].argmax(1)

            X, missing, empty, inds, n_wfs_fit, channel_had_observations = self.get_training_data(
                batch, n_wfs_fit=n_wfs_fit, train_buffer=train_buffer[: len(batch)]
            )
            loadings, mean, components, svs = fit_pcas(
                X,
                missing,
                empty,
                self.rank,
                max_iter=self.svd_max_iter,
                n_oversamples=self.svd_n_oversamples,
                atol=self.svd_atol,
                show_progress=False,
                centered=self.centered,
            )
            self.mean[batch] = mean
            self.svs[batch] = svs
            self.channel_had_observations[batch] = channel_had_observations
            self.components[batch] = components
            self.train_loadings[batch, :n_wfs_fit] = loadings.numpy(force=True)
            self.train_spike_indices[batch, n_wfs_fit:] = -1
            self.train_spike_indices[batch, :n_wfs_fit] = inds
            self.needs_fit[batch] = False

    def embed_spikes(self, unit_id, waveforms, rel_ix, return_X=False, damping=None, impute_iters=0, rank=None):
        n, r, c = waveforms.shape
        m = self.mean[unit_id]
        # intialize empties with the means
        # TODO: only impute rows with empties
        X = F.pad(m.reshape(r, self.data.n_chans_cluster), (0, 1))
        X = X[None].broadcast_to(n, *X.shape).contiguous()
        rel_ix_scatter = rel_ix[:, None, :].broadcast_to((n, r, rel_ix.shape[-1]))
        W = self.components[unit_id]
        
        X.scatter_(src=waveforms, dim=2, index=rel_ix_scatter)
        Xflat = X[..., :-1].reshape(n, self.dim_input)
        embeds = (Xflat - m) @ W
        for i in range(impute_iters):
            X = self.reconstruct_spikes(
                unit_id, embeds, rel_ix, return_rel=True
            )
            X.scatter_(src=waveforms, dim=2, index=rel_ix_scatter)
            Xflat = X[..., :-1].reshape(n, self.dim_input)
            embeds = (Xflat - m) @ W
    
        if damping is None:
            damping = self.embedding_damping

        if damping:
            nu = (self.train_spike_indices[unit_id] >= 0).sum()
            sds = self.svs[unit_id] / float(np.sqrt(nu))
            damp = torch.square(sds / (sds + self.embedding_damping))
            embeds *= damp
        
        if rank is not None:
            embeds[:, rank:] = 0.0

        if return_X:
            return X, embeds

        return embeds

    def embed_at(
        self,
        unit_id,
        spike_indices,
        overlaps=None,
        rel_ix=None,
        return_X=False,
        impute_iters=0,
        damping=None,
        rank=None
    ):
        # todo: this should call embed_spikes
        if overlaps is None:
            overlaps, rel_ix = self.spike_overlaps(unit_id, which_spikes=spike_indices)
        rel_ix = torch.tensor(rel_ix, device=self.mean.device)
        waveforms = self.data.getwf(spike_indices)
        waveforms = torch.tensor(waveforms, device=self.mean.device)
        
        return overlaps, self.embed_spikes(
            unit_id, waveforms, rel_ix, damping=damping, impute_iters=impute_iters, rank=rank
        )

    def reconstruct_spikes(
        self, unit_id, loadings, rel_ix, return_rel=False, recons_rel=None
    ):
        n = len(loadings)
        r = self.data.tpca_rank
        recons_rel = torch.addmm(
            self.mean[unit_id], loadings, self.components[unit_id].T, out=recons_rel
        )
        recons_rel = recons_rel.reshape(n, -1, self.data.n_chans_cluster)
        recons_rel = F.pad(recons_rel, (0, 1))
        if return_rel:
            return recons_rel
        rel_ix = rel_ix[:, None, :].broadcast_to((n, r, rel_ix.shape[-1]))
        return torch.gather(recons_rel, dim=2, index=rel_ix)

    def centroid_to_spike_channels(self, unit_id, rel_ix, with_components=False):
        n = len(rel_ix)
        r = self.data.tpca_rank
        ncc = self.data.n_chans_cluster
        ncw = rel_ix.shape[-1]

        rel_ix = rel_ix.unsqueeze(1).broadcast_to(n, r, ncw)

        m = self.mean[unit_id].reshape(r, ncc)
        m = F.pad(m, (0, 1))
        m = m.unsqueeze(0).broadcast_to(n, r, ncc + 1)
        centroid_rel = torch.gather(m, dim=2, index=rel_ix)
        if not with_components:
            return centroid_rel

        comps = self.components[unit_id].reshape(r, ncc, self.rank)
        comps = F.pad(comps, (0, 0, 0, 1, 0, 0))
        comps = comps.unsqueeze(0).broadcast_to(n, r, ncc + 1, self.rank)
        rel_ix = rel_ix.unsqueeze(3).broadcast_to(n, r, ncw, self.rank)
        comps_rel = torch.gather(comps, dim=2, index=rel_ix)

        return centroid_rel, comps_rel

    def inspect_spike(self, index):
        wf = self.data.getwf(index)
        sc = self.data.spike_static_channels[index]
        unit_ids = self.unit_ids()
        snrs = self.snrs[unit_ids]
        main_channels = snrs.argmax(1)
        unit_reindexers = np.atleast_2d(self.wf_to_pc_reindexers[main_channels])

        overlaps = np.zeros(unit_ids.size)
        embeds = np.zeros((unit_ids.size, self.rank), dtype=np.float32)
        wfs_rel = np.zeros(
            (unit_ids.size, wf.shape[0], self.dim_input // wf.shape[0]),
            dtype=np.float32,
        )
        recons = np.zeros((unit_ids.size, *wf.shape), dtype=np.float32)
        for uid in unit_ids:
            rel_ix = unit_reindexers[uid][sc]
            unc = np.sum(unit_reindexers[uid] < self.data.n_chans_cluster)
            overlaps[uid] = (rel_ix < self.data.n_chans_cluster).sum() / unc

            rel_ix = torch.as_tensor(rel_ix[None])
            x, e = self.embed_spikes(uid, torch.tensor(wf[None]), rel_ix, return_X=True, impute_iters=self.reassign_impute_iters)
            wfs_rel[uid] = x[0].reshape(wfs_rel[uid].shape)
            embeds[uid] = e[0]
            recons[uid] = self.reconstruct_spikes(
                uid, torch.tensor(embeds[uid][None]), rel_ix
            )[0].reshape(wf.shape)

        return dict(
            overlaps=overlaps,
            wf=wf,
            sc=sc,
            wfs_rel=wfs_rel,
            embeds=embeds,
            recons=recons,
            errs=np.nansum(np.square(wf[None] - recons), axis=(1, 2)),
        )

    def calc_errors(self, kind="unexplained_var", centroid_only=False, sparse=False, rank=None):
        unit_ids = self.unit_ids()
        n_units = unit_ids.size
        if rank is None:
            rank = self.reassign_rank

        # this is typically sparse
        if sparse:
            # if you use this just beware that the empties are infs, not 0.
            errs = dok_array(
                (n_units, self.data.n_spikes), dtype=self.data.tpca_embeds.dtype
            )
            nobs = dok_array(
                (n_units, self.data.n_spikes), dtype=self.data.tpca_embeds.dtype
            )
            max_err = 0
        else:
            errs = np.full((n_units, self.data.n_spikes), np.inf)
            nobs = np.full((n_units, self.data.n_spikes), np.inf)

        for uid in tqdm(unit_ids, desc="Unit distances"):
            overlaps, rel_ix = self.spike_overlaps(uid)
            which = np.flatnonzero(overlaps >= self.min_overlap)
            if not which.size:
                print(f"No spikes overlap unit {uid}.")
                continue
            rel_ix = torch.tensor(rel_ix[which], device=self.mean.device)
            for bs in range(0, which.size, self.spike_batch_size):
                be = min(which.size, bs + self.spike_batch_size)
                batch = which[bs:be]
                wfs = self.data.getwf(batch)

                if centroid_only:
                    wfs = torch.from_numpy(wfs).to(self.mean)
                    crels = self.centroid_to_spike_channels(uid, rel_ix[bs:be])
                    err = crels
                else:
                    wfs = torch.from_numpy(wfs).to(self.mean)
                    loadings = self.embed_spikes(uid, wfs, rel_ix[bs:be], impute_iters=self.reassign_impute_iters, rank=rank)
                    recons = self.reconstruct_spikes(uid, loadings, rel_ix[bs:be])
                    err = recons

                err -= wfs
                obs = torch.isfinite(err).sum(dim=(1, 2))
                # if p == 2:
                torch.square(err, out=err)
                err = torch.nansum(err, dim=(1, 2))
                if kind == "unexplained_var":
                    err /= torch.square(wfs).nansum(dim=(1, 2))
                # elif p == np.inf:
                #     err = err.abs().max(dim=(1, 2)).values

                if sparse:
                    batch_max_err = err.max().numpy(force=True)
                    max_err = max(max_err, batch_max_err)
                errs[uid, batch] = err.numpy(force=True)
                nobs[uid, batch] = obs.numpy(force=True)

        if sparse:
            # spikes are columns and we want to operate quickly on spikes.
            errs = errs.tocsc()
            nobs = nobs.tocsc()
            return max_err, errs, nobs

        return errs, nobs

    def calc_liks(self, sparse=False, centroid_only=False):
        unit_ids = self.unit_ids()
        n_units = unit_ids.size

        # if you use this just beware that the empties are infs, not 0.
        dtype = self.data.tpca_embeds.dtype
        shape = (n_units, self.data.n_spikes)
        if sparse:
            mahals = dok_array(shape, dtype=dtype)
            logliks = dok_array(shape, dtype=dtype)
            chis = dok_array(shape, dtype=dtype)
            nobs = dok_array(shape, dtype=dtype)
        else:
            mahals = np.full(shape, np.inf, dtype=dtype)
            logliks = np.full(shape, -np.inf, dtype=dtype)
            chis = np.full(shape, np.inf, dtype=dtype)
            nobs = np.full(shape, np.inf, dtype=dtype)

        for uid in tqdm(unit_ids, desc="Unit distances"):
            overlaps, rel_ix = self.spike_overlaps(uid)
            which = np.flatnonzero(overlaps >= self.min_overlap)
            if not which.size:
                print(f"No spikes overlap unit {uid}.")
                continue
            rel_ix = torch.tensor(rel_ix[which], device=self.mean.device)
            for bs in range(0, which.size, self.spike_batch_size):
                be = min(which.size, bs + self.spike_batch_size)
                batch = which[bs:be]

                wfs = torch.tensor(self.data.getwf(batch)).to(self.mean)

                if centroid_only:
                    centroid_rel = self.centroid_to_spike_channels(uid, rel_ix[bs:be])
                    comps_rel = None
                    sds = None
                else:
                    centroid_rel, comps_rel = self.centroid_to_spike_channels(
                        uid, rel_ix[bs:be], with_components=True
                    )

                rc = np.prod(wfs.shape[1:])
                X = wfs.reshape(be - bs, rc)
                masks = torch.isfinite(X)
                centroid_rel = centroid_rel.reshape(be - bs, rc)
                if not centroid_only:
                    comps_rel = comps_rel.reshape(be - bs, rc, self.rank)
                    nu = (self.train_spike_indices[uid] >= 0).sum()
                    sds = self.svs[uid] / float(np.sqrt(nu))
                m, l, c = fast_gaussian(X, centroid_rel, masks, comps_rel, sds=sds, v=1)
                mahals[uid, batch] = m.numpy(force=True)
                logliks[uid, batch] = l.numpy(force=True)
                chis[uid, batch] = c.numpy(force=True)
                nobs[uid, batch] = masks.sum(1).numpy(force=True)

        if sparse:
            # spikes are columns and we want to operate quickly on spikes.
            mahals = mahals.tocsc()
            logliks = logliks.tocsc()
            nobs = nobs.tocsc()
            # except the chis. those are kind of "within-unit".
            chis = chis.tocsr()

        return mahals, logliks, chis

    def reassign(self, kind="unexplained_var", centroid_only=False):
        if kind.endswith("unexplained_var") or kind.endswith("reconstruction"):
            centroid_only = centroid_only or (kind.startswith("centroid"))
            max_err, errs, nobs = self.calc_errors(
                kind=kind.removeprefix("centroid"),
                centroid_only=centroid_only,
                sparse=True,
            )
            # sparse nonzero rows. this is CSC format specific.
            # has_match = np.isfinite(errs).any(0)
            has_match = np.diff(errs.indptr) > 0
            # we want sparse 0s to mean infinite err.
            # let M>max err. then 0 > max err - M.
            adj_errs = errs.copy()
            adj_errs.data -= max_err + 212.0
            new_labels = np.where(
                has_match,
                adj_errs.argmin(0),
                -1,
            )
            if kind.endswith("unexplained_var"):
                uv_thresh = 1.0 - self.spike_r2_threshold
                kept = np.flatnonzero(has_match)
                uvs = errs[new_labels[kept], kept]
                new_labels[kept[uvs > uv_thresh]] = -1
        elif kind.endswith("likelihood"):
            centroid_only = centroid_only or (kind == "centroidlikelihood")
            mahals, logliks, chis = self.calc_liks(
                sparse=True, centroid_only=centroid_only
            )
            has_match = np.diff(logliks.indptr) > 0
            # handle sparse 0s being -inf
            min_lik = logliks.data.min()
            logliks.data += 212.0 + np.abs(min_lik)
            new_labels = np.where(
                has_match,
                logliks.argmax(0),
                -1,
            )
            self.chis = chis

        print(f"{(new_labels < 0).mean()*100:0.1f}% of spikes unmatched after reassignment.")
        self.labels = new_labels
        self.drop_small()
        self.needs_fit[: len(self.unit_ids())] = 1

    def dpc_split(self, recursive=True, **kwargs):
        unit_ids = self.unit_ids()
        orig_nunits = len(unit_ids)
        i = 1
        self.needs_split[: len(self.unit_ids())] = True
        while self.needs_split.any():
            ids_to_refit = []
            ids_to_resplit = []
            for uid in tqdm(
                self.unit_ids_to_split(),
                desc=f"Split round {i}" if recursive else "Split",
                **tqdm_kw,
            ):
                if recursive:
                    split_ids, fit_ids = self.dpc_split_unit(uid, **kwargs)
                    ids_to_refit.extend(fit_ids)
                    ids_to_resplit.extend(split_ids)
            assert len(ids_to_resplit) <= len(ids_to_refit)
            self.check_grow()
            self.needs_fit[ids_to_refit] = True
            self.needs_split[:] = False
            self.needs_split[ids_to_resplit] = True
            assert self.needs_split.sum() <= self.needs_fit.sum()
            self.drop_small()
            assert self.needs_split.sum() <= self.needs_fit.sum()
            self.m_step()
            i += 1
        print(f"Split: {orig_nunits} -> {self.unit_ids().size}")

    def dpc_split_unit(
        self,
        unit_id,
        rank=2,
        sigma_local="rule_of_thumb",
        sigma_regional=None,
        n_neighbors_search=500,
        remove_clusters_smaller_than=50,
        allow_single_cluster_outlier_removal=True,
    ):
        """Split the unit. Reassign to sub-units. Update state variables."""
        # TODO set rank, embed full unit
        # run the clustering
        if self.split_on_train:
            X = self.train_loadings[unit_id]
            in_unit = self.train_spike_indices[unit_id]
            valid = np.flatnonzero(in_unit >= 0)
            in_unit = in_unit[valid]
            X = X[valid, : self.split_rank]
        else:
            in_unit = np.flatnonzero(self.labels == unit_id)
            overlaps, rel_ix = self.spike_overlaps(unit_id, which_spikes=in_unit)
            valid = overlaps >= self.min_overlap
            in_unit = in_unit[valid]
            overlaps = overlaps[valid]
            rel_ix = rel_ix[valid]
            overlaps, X = self.embed_at(
                unit_id,
                in_unit,
                overlaps=overlaps,
                rel_ix=rel_ix,
                impute_iters=self.split_impute_iters,
            )
            X = X[:, : self.split_rank].numpy(force=True)

        # assert np.isin(in_unit, np.flatnonzero(self.labels == unit_id)).all()
        try:
            split_labels = density.density_peaks_clustering(
                X,
                sigma_local=sigma_local,
                n_neighbors_search=n_neighbors_search,
                remove_clusters_smaller_than=max(
                    self.min_unit_size, remove_clusters_smaller_than
                ),
            )
        except ValueError as e:
            print(e)
            print(f"{X[:10]=}")
            return [], []
        split_units_full, counts = np.unique(split_labels, return_counts=True)
        split_units = split_units_full[split_units_full >= 0]
        if split_units.size == split_units_full.size == 1:
            return [], []
        if split_units.size <= 1 and not allow_single_cluster_outlier_removal:
            return [], []
        cleaning = split_units.size <= 1

        # todo: if single cluster outlier case, re-fit but don't re-split.
        # replace labels with split labels, update params in those groups
        self.labels[self.labels == unit_id] = -1
        new_unit_ids = (unit_id, *(self.labels.max() + np.arange(1, split_units.size)))
        for split_label, new_label in zip(split_units, new_unit_ids):
            in_split = in_unit[split_labels == split_label]
            self.labels[in_split] = new_label

        fit_ids = new_unit_ids
        split_ids = [] if cleaning else new_unit_ids
        return split_ids, fit_ids

    def get_training_data(self, unit_ids, n_wfs_fit=None, train_buffer=None):
        """
        Returns
        -------
        X : shape (n_wfs_fit, tpca_dim, n_chans_cluster)
        missing : shape (n_wfs_fit, n_chans_cluster)
        """
        n_units = unit_ids.size
        main_channels = self.main_channels[unit_ids]
        unit_reindexers = np.atleast_2d(self.wf_to_pc_reindexers[main_channels])
        nc_pca = self.data.n_chans_cluster
        nc_full = self.data.n_reg_chans
        in_units = [np.flatnonzero(self.labels == uid) for uid in unit_ids]
        if n_wfs_fit is None:
            n_wfs_fit = self.n_wfs_fit
        elif n_wfs_fit == -1:
            n_wfs_fit = min(self.n_wfs_fit, max(iu.size for iu in in_units))

        if train_buffer is None:
            X = np.full(
                (n_units, n_wfs_fit, self.data.tpca_rank, nc_pca + 1),
                np.nan,
                dtype=np.float32,
            )
        else:
            X = train_buffer[:, :n_wfs_fit]
            X.fill(np.nan)
        missing = np.ones((n_units, n_wfs_fit, nc_pca + 1), dtype=bool)
        empty = np.ones((n_units, n_wfs_fit), dtype=bool)
        spike_indices = np.full((n_units, n_wfs_fit), -1)

        for j, uid in enumerate(unit_ids):
            in_unit = in_units[j]
            if in_unit.size > n_wfs_fit:
                in_unit = self.rg.choice(in_unit, size=n_wfs_fit, replace=False)
                in_unit.sort()
            nj = in_unit.size

            chans = self.data.spike_static_channels[in_unit]
            embeds = self.data.getwf(in_unit)

            reindexer = unit_reindexers[j]
            train_ix = reindexer[chans]

            np.put_along_axis(X[j, :nj], train_ix[:, None, :], embeds, axis=-1)
            np.put_along_axis(missing[j, :nj], train_ix, chans == nc_full, axis=-1)
            empty[j, :nj] = 0
            spike_indices[j, :nj] = in_unit

        missing[empty] = 0
        X = torch.from_numpy(X[..., :-1])
        channel_had_observations = np.logical_not(missing[..., :-1].all(1))
        missing = torch.from_numpy(missing[..., :-1])
        missing = missing[..., None, :].broadcast_to(X.shape)
        X = X.reshape(n_units, n_wfs_fit, -1)
        missing = missing.reshape(n_units, n_wfs_fit, -1)
        empty = torch.from_numpy(empty)

        X = X.to(self.mean.device)
        missing = missing.to(self.mean.device)
        empty = empty.to(self.mean.device)

        return X, missing, empty, spike_indices, n_wfs_fit, channel_had_observations

    def spike_overlaps(self, unit_ids=None, which_spikes=None):
        """
        Returns
        -------
        overlaps : (n_units, n_spikes)
        rel_ix : (n_units, n_spikes, n_chans_cluster)
        """
        if unit_ids is None:
            unit_ids = self.unit_ids()
        if which_spikes is None:
            which_spikes = slice(None)
        single = np.isscalar(unit_ids)
        unit_ids = np.atleast_1d(unit_ids)

        main_channels = self.main_channels[unit_ids]
        unit_reindexers = np.atleast_2d(self.wf_to_pc_reindexers[main_channels])

        spike_static_channels = self.data.spike_static_channels[which_spikes]

        rel_ix = np.take_along_axis(
            unit_reindexers[:, None],
            spike_static_channels[None],
            axis=2,
        )
        overlap_num = (rel_ix < self.data.n_chans_cluster).sum(2)
        overlap_den = (spike_static_channels < self.data.n_reg_chans).sum(1)
        overlaps = overlap_num / overlap_den
        if single:
            assert overlaps.shape[0] == rel_ix.shape[0] == 1
            overlaps = overlaps[0]
            rel_ix = rel_ix[0]
        return overlaps, rel_ix

    def centroid_overlaps(self, unit_ids_a=None, unit_ids_b=None):
        """
        Returns
        -------
        overlaps : (n_units, n_spikes)
        rel_ix : (n_units, n_spikes, n_chans_cluster)
        """
        if unit_ids_a is None:
            unit_ids_a = self.unit_ids()
        if unit_ids_b is None:
            unit_ids_b = self.unit_ids()
        unit_ids_a = np.atleast_1d(unit_ids_a)
        unit_ids_b = np.atleast_1d(unit_ids_b)

        main_channels_a = self.main_channels[unit_ids_a]
        unit_reindexers_a = np.atleast_2d(self.wf_to_pc_reindexers[main_channels_a])

        main_channels_b = self.main_channels[unit_ids_b]
        static_channels_b = self.data.cluster_channel_index[main_channels_b]
        static_channels_b = np.where(
            self.channel_had_observations[unit_ids_b],
            static_channels_b,
            self.data.n_reg_chans,
        )

        rel_ix = np.take_along_axis(
            unit_reindexers_a[:, None],
            static_channels_b[None, :],
            axis=2,
        )
        overlap_num = (rel_ix < self.data.n_chans_cluster).sum(2)
        overlap_den = (static_channels_b < self.data.n_reg_chans).sum(1)
        overlaps = overlap_num / overlap_den
        return overlaps, rel_ix, static_channels_b

    def merge(
        self,
        kind="reconstruction",
        measure="unexplained_var",
        sym_function=average_sym,
        threshold=0.25,
        link="complete",
        centroid_only=False,
    ):
        dist_res = self.centroid_dists(kind, centroid_only=centroid_only, rank=self.merge_rank)
        D = dist_res[measure]
        D = sym_function(D, D.T)
        D[np.isinf(D)] = D.max() + 10
        d = D[np.triu_indices(D.shape[0], k=1)]
        Z = linkage(d, method=link)
        # extract flat clustering using our max dist threshold
        new_labels = fcluster(Z, threshold, criterion="distance")
        unique_new_labels, new_counts = np.unique(new_labels, return_counts=True)
        print(f"Merge: {self.unit_ids().size} -> {unique_new_labels.size}")

        # update labels
        labels_updated = np.full_like(self.labels, -1)
        kept = np.flatnonzero(self.labels >= 0)
        labels_updated[kept] = new_labels[self.labels[kept]]

        self.labels = labels_updated
        new_to_fit = unique_new_labels[new_counts > 1]
        # self.needs_fit[new_to_fit] = True
        self.drop_small()
        self.needs_fit[self.unit_ids()] = True

    def centroid_dists(
        self,
        kind="reconstruction",
        centroid_only=False,
        unit_ids_a=None,
        unit_ids_b=None,
        return_reconstructions=False,
        rank=None,
    ):
        if unit_ids_a is None:
            unit_ids_a = self.unit_ids()
        if unit_ids_b is None:
            unit_ids_b = self.unit_ids()
        unit_ids_a = np.atleast_1d(unit_ids_a)
        unit_ids_b = np.atleast_1d(unit_ids_b)

        overlaps, rel_ix, static_channels_b = self.centroid_overlaps(
            unit_ids_a, unit_ids_b
        )
        pairs = overlaps > self.min_overlap
        if return_reconstructions:
            reconstructions = torch.zeros(
                (self.unit_ids().size, self.unit_ids().size, *self.mean.shape[1:]),
                device=self.mean.device,
                dtype=self.mean.dtype,
            )

        if kind.endswith("likelihood"):
            mahals = np.full((unit_ids_a.size, unit_ids_b.size), np.inf)
            logliks = np.full((unit_ids_a.size, unit_ids_b.size), -np.inf)
            chis = np.full((unit_ids_a.size, unit_ids_b.size), np.inf)
            nobs = np.full((unit_ids_a.size, unit_ids_b.size), 0)
        if kind == "reconstruction":
            errors = np.full((unit_ids_a.size, unit_ids_b.size), np.inf)
            unexplained_var = np.ones((unit_ids_a.size, unit_ids_b.size))

        for unit_a, row_a in enumerate(pairs):
            units_b = np.flatnonzero(row_a)
            nb = units_b.size
            if not nb:
                continue

            centroids_b = self.mean[units_b]
            rel_ix_b = torch.tensor(rel_ix[unit_a, units_b]).to(centroids_b.device)
            scb = static_channels_b[units_b]

            if kind == "likelihood":
                if centroid_only:
                    centroid_a_rel = self.centroid_to_spike_channels(unit_a, rel_ix_b)
                    comps_a_rel = None
                    sds = None
                else:
                    centroid_a_rel, comps_a_rel = self.centroid_to_spike_channels(
                        unit_a, rel_ix_b, with_components=True
                    )

                rc = np.prod(centroids_b.shape[1:])
                X = centroids_b.reshape(nb, rc)
                masks = torch.tensor(scb < self.data.n_reg_chans, device=X.device)
                masks = (
                    masks.unsqueeze(1)
                    .broadcast_to(nb, self.data.tpca_rank, self.data.n_chans_cluster)
                    .reshape(X.shape)
                )
                centroid_a_rel = centroid_a_rel.reshape(nb, rc)
                if not centroid_only:
                    comps_a_rel = comps_a_rel.reshape(nb, rc, self.rank)
                    nu = (self.train_spike_indices[unit_a] >= 0).sum()
                    sds = self.svs[unit_a] / float(np.sqrt(nu))
                m, l, c = fast_gaussian(
                    X, centroid_a_rel, masks, comps_a_rel, sds=sds, v=1
                )
                mahals[unit_a, units_b] = m.numpy(force=True)
                logliks[unit_a, units_b] = l.numpy(force=True)
                chis[unit_a, units_b] = c.numpy(force=True)
                nobs[unit_a, units_b] = masks.sum(1).numpy(force=True)

            elif kind == "reconstruction" and not centroid_only:
                wfs = centroids_b.reshape(
                    nb, self.data.tpca_rank, self.data.n_chans_cluster
                )
                loadings = self.embed_spikes(unit_a, wfs, rel_ix_b, impute_iters=self.merge_impute_iters, rank=rank)
                recons = self.reconstruct_spikes(unit_a, loadings, rel_ix_b)
                if return_reconstructions:
                    reconstructions[unit_a, units_b] = recons.reshape(len(units_b), -1)
                err = wfs - recons
                ignore = torch.from_numpy(scb == self.data.n_reg_chans).to(err.device)
                ignore = ignore[:, None, :].broadcast_to(err.shape)
                err[ignore] = 0.0
                err = torch.square(err).sum(dim=(1, 2))
                unexvar = err / torch.square(wfs).sum(dim=(1, 2))
                errors[unit_a, units_b] = err.numpy(force=True)
                unexplained_var[unit_a, units_b] = unexvar.numpy(force=True)
            elif kind == "reconstruction" and centroid_only:
                wfs = centroids_b.reshape(
                    nb, self.data.tpca_rank, self.data.n_chans_cluster
                )
                crels = self.centroid_to_spike_channels(unit_a, rel_ix_b)
                if return_reconstructions:
                    reconstructions[unit_a, units_b] = crels.reshape(len(units_b), -1)
                err = wfs - crels
                ignore = torch.from_numpy(scb == self.data.n_reg_chans).to(err.device)
                ignore = ignore[:, None, :].broadcast_to(err.shape)
                err[ignore] = 0.0
                err = torch.square(err).sum(dim=(1, 2))
                unexvar = err / torch.square(wfs).sum(dim=(1, 2))
                errors[unit_a, units_b] = err.numpy(force=True)
                unexplained_var[unit_a, units_b] = unexvar.numpy(force=True)

        results = dict(
            overlaps=overlaps,
        )
        if kind == "likelihood":
            results["mahals"] = mahals
            results["logliks"] = logliks
            results["chis"] = chis
            results["nobs"] = nobs
        else:
            results["errors"] = errors
            results["unexplained_var"] = unexplained_var
            if return_reconstructions:
                results["reconstructions"] = reconstructions.numpy(force=True)
        return results


def get_channel_structures(cluster_channel_index):
    """
    Helps us to convert each spike's static channels into each unit's space.

    So, later, we'll want to put spikes living on `static_chans` shape (n, n_wf_chans)
    into a training buffer `X` shape (n, :, n_chans_cluster). A user can do this like:

    ```
    spike_to_unit = wf_to_pc_reindexers[unit_main_channel, static_chans]
    X.scatter_(src=spikes, dim=2, index=spike_to_unit)
    ```

    The trick is that X actually needed to have shape (n, :, n_chans_cluster + 1), and the
    extra channel eats up the invalid stuff.

    We also can get observed masks. O has shape (n, n_chans_cluster + 1). Then we set

    ```
    O[torch.arange(n)[:, None], spike_to_unit] = 1
    ```

    Then this means that
        wf_to_pc_reindexers[c, d] =
            index of d (a registered chan) in cluster_channel_index[c], if present
            n_chans_cluster (==cluster_channel_index.shape[1]) otherwise

    Returns
    -------
    wf_to_pc_reindexers : int tensor, shape (n_units, n_reg_chans, n_chans_cluster)
        wf_to_pc_reindexers[u, c] is an array of indices in [0, ..., n_wf_chans].
        Inclusive on the right -- waveforms need to be padded (with 0s) to be
        indexed by it.
    """
    nc, n_chans_cluster = cluster_channel_index.shape
    wf_to_pc_reindexers = np.full((nc, nc + 1), n_chans_cluster)
    for c in range(nc):
        for i, d in enumerate(cluster_channel_index[c]):
            if d < nc:
                wf_to_pc_reindexers[c, d] = i
    return wf_to_pc_reindexers


@torch.no_grad()
def fit_time_varying(
    X,
    times,
    missing,
    empty,
    rank,
    time_rank,
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
    z = torch.ones((len(X), time_rank))

    ###
    filled = torch.logical_not(empty)
    no_missing = not missing[filled].any()

    # iterate svds
    it = trange(max_iter, desc="SVD") if show_progress else range(max_iter)
    # svd_storage = None
    for j in it:
        # update svd
        if centered:
            Xin = torch.where(empty[..., None], 0, Xc - mean)
        else:
            Xin = Xc
        # svd_storage = torch.linalg.svd(Xin, full_matrices=False, out=svd_storage)
        # U, S, Vh = svd_storage

        # faster in my experience despite the note in torch.linalg.svd docs
        U, S, V = torch.svd_lowrank(Xin, q=rank + n_oversamples)
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

    return loadings, mean, components, svs



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
    for j in it:
        # update svd
        if centered:
            Xin = torch.where(empty[..., None], 0, Xc - mean)
        else:
            Xin = Xc
        # svd_storage = torch.linalg.svd(Xin, full_matrices=False, out=svd_storage)
        # U, S, Vh = svd_storagev

        # faster in my experience despite the note in torch.linalg.svd docs
        U, S, V = torch.svd_lowrank(Xin, q=rank + n_oversamples)
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

    # load waveforms and subset by radius, retaining the new index for later
    h5 = h5py.File(sorting.parent_h5_path, "r", locking=False)
    tpca_embeds = None
    if in_memory:
        tpca_embeds = _read_by_chunk(keep_mask, h5["collisioncleaned_tpca_features"])
    else:
        tpca_embeds = h5["collisioncleaned_tpca_features"]
    amp_vecs = _read_by_chunk(keep_mask, h5["denoised_ptp_amplitude_vectors"])
    geom = h5["geom"][:]
    oci = channel_index = h5["channel_index"][:]
    if in_memory:
        h5.close()
    # TODO: whiten_input when not in_memory means load up tpca info

    if wf_radius is not None:
        if in_memory:
            tpca_embeds, channel_index = waveform_util.channel_subset_by_radius(
                tpca_embeds,
                channels,
                channel_index,
                geom,
                radius=wf_radius,
            )
            amp_vecs = waveform_util.channel_subset_by_index(
                amp_vecs, channels, oci, channel_index
            )
            if whiten_input:
                s = np.nanstd(tpca_embeds, axis=0, keepdims=True)
                s[s == 0] = 1.0
                tpca_embeds /= s
            tpca_embeds.setflags(write=False)
        else:
            amp_vecs, channel_index = waveform_util.channel_subset_by_radius(
                amp_vecs,
                channels,
                channel_index,
                geom,
                radius=wf_radius,
            )

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
        times_s=sorting.times_seconds[keepers],
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

    return labels, SpikePCAData(
        keepers=keepers,
        channels=channels,
        n_reg_chans=n_reg_chans,
        tpca_rank=tpca_rank,
        n_chans_cluster=n_chans_cluster,
        n_wf_chans=n_wf_chans,
        n_spikes=n_spikes,
        original_channel_index=oci,
        wf_channel_index=channel_index,
        cluster_channel_index=cluster_channel_index,
        static_amp_vecs=static_amp_vecs,
        tpca_embeds=tpca_embeds,
        spike_static_channels=spike_static_channels,
        registered_geom=registered_geom,
        in_memory=in_memory,
    )


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


def fast_gaussian(X, mu, masks, W=None, sds=None, v=1):
    """
    Quickly evaluate many mahalanobis distances and Gaussian likelihoods
    in diag plus low rank model (Sigma = sigma I + W diag(sds^2) W')

    This runs something like O(npr^2)? The idea is to multiply the Woodbury
    identity by x'()x, and this avoids storing the pxp precision matrix or
    doing any p^2 stuff.

    To wit. Let V = W * sds be pxr.
    inv(vIp + VV') = Ip/v - (1/v^2) V inv(Ir + V'V/v) V'

    We want to evaluate x' inv(vI + VV') x. This can be efficiently parenthesized:
        Let u = xV, which is a r-vector.
        x' inv(vI + VV') x = x'x/v - (1/v^2) u' inv(Ir + V'V/v) u.

    For the likelihood, we need to evaluate the determinant. Matrix determinant
    lemma reads
       det(vIp + VV') = v^p det(Ir + V'V)
       logdet(vIp + VV') = p log(v) + logdet(Ir + V'V)

    Arguments
    ---------
    X : (n, p)
    mu : (p,)
    W : (p, r)
    sds : (r,)
        Standard deviation of something low rank
    v : float
        Variance of observations

    Returns
    -------
    mahals, logliks, chis
    """
    dx = torch.where(masks, X - mu, 0)
    nobs = masks.sum(1)
    # assert not torch.isnan(dx).any()
    if W is None:
        mahals = torch.square(dx).sum(1) / v
        logliks = -0.5 * mahals
        if v != 1:
            logliks.subtract_(float(np.log(v)) * nobs.sum(1))
        chis = mahals / nobs
        return mahals, logliks, chis

    V = W
    if sds is not None:
        V = V * sds

    dots = dx.square().sum(1) / v

    inner = torch.bmm(V.mT, V) / v
    inner.diagonal(dim1=-2, dim2=-1).add_(1)
    # inner_inv = torch.linalg.inv(inner)

    u = torch.bmm(V.mT, dx[..., None])[..., 0]
    # solve should be lsq sense bc V can have rank < r even
    inner_inv_u = torch.linalg.lstsq(inner, u).solution
    inner_inv_q = (u * inner_inv_u).sum(1)
    # inner_inv_q = torch.einsum("nr,nrs,ns->n", u, inner_inv, u) / (v * v)

    mahals = dots - inner_inv_q
    logcovdets = inner.logdet()
    if v != 1:
        logcovdets.add_(float(np.log(v)) * nobs.sum(1))
    logliks = -0.5 * (mahals + logcovdets)  # ok, up to the 2pis.

    chis = mahals / nobs

    return mahals, logliks, chis


# this would be nice but it's bugged, need to rework the
# dest_ix logic.
#
# def _read_by_chunk2(mask, dataset, axis=0):
#     """
#     mask : boolean array of shape (dataset.shape[axis],)
#     dataset : chunked h5py.Dataset
#     """
#     out_shape = list(dataset.shape)
#     out_shape[axis] = mask.sum()
#     out = np.empty(out_shape, dtype=dataset.dtype)
#     src_ix = [slice(None)] * dataset.ndim
#     n = 0
#     for slice_tuple in dataset.iter_chunks():
#         ax_slice = slice_tuple[axis]
#         m = np.flatnonzero(mask[ax_slice])
#         nm = m.size
#         if not nm:
#             continue
#         src_ix[axis] = m
#         x = dataset[slice_tuple][tuple(src_ix)]
#         dest_ix = (*slice_tuple[:axis], slice(n, n + nm), *slice_tuple[axis + 1 :])
#         out[dest_ix] = x
#         n += nm
#     return out
