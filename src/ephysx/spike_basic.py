import dataclasses

from tqdm.auto import tqdm, trange

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from dartsort.cluster import density
from dartsort.util import drift_util, waveform_util
from torch import nn

tqdm_kw = dict(smoothing=0)


@dataclasses.dataclass
class SpikePCAData:
    """Data bag to keep things tidy."""

    keepers: np.array
    n_reg_chans: int
    tpca_rank: int
    n_chans_cluster : int
    n_wf_chans: int
    n_spikes: int
    cluster_channel_index: np.array
    static_amp_vecs: torch.Tensor
    tpca_embeds: np.array
    spike_static_channels: np.array


class BasicSpikePCAClusterer(nn.Module):

    def __init__(
        self,
        sorting,
        motion_est=None,
        rank=5,
        svd_atol=0.01,
        svd_max_iter=100,
        fit_radius=25.0,
        wf_radius=25.0,
        min_unit_size=50,
        n_wfs_fit=8192,
        min_overlap=0.5,
        buffer_growth_factor=1.5,
        train_units_batch_size=16,
        spike_batch_size=256,
        rg=0,
    ):
        self.rank = rank
        self.svd_atol = svd_atol
        self.svd_max_iter = svd_max_iter
        self.rg = np.random.default_rng(rg)
        self.n_wfs_fit = n_wfs_fit
        self.min_unit_size = max(2 * self.rank, min_unit_size)
        self.buffer_growth_factor = buffer_growth_factor
        self.train_units_batch_size = train_units_batch_size
        self.min_overlap = min_overlap
        self.spike_batch_size = spike_batch_size

        self.labels, self.data = _load_data(sorting, motion_est, fit_radius, wf_radius)
        self.dim_input = self.data.tpca_rank * self.data.n_tpca_chans

        self.drop_small()
        self.initialize()

    def initialize(self):
        n_units = self.unit_ids().size
        U = int(np.ceil(n_units * self.buffer_growth_factor))
        self.register_buffer(
            "components", torch.zeros((U, self.dim_input, self.svd_rank))
        )
        self.register_buffer("mean", torch.zeros((U, self.dim_input)))
        self.wf_to_pc_reindexers = get_channel_structures(self.data.cluster_channel_index)
        # self.register_buffer("_channel_reindexer", torch.tensor(self.wf_to_pc_reindexers))
        self.train_loadings = np.full((U, self.n_wfs_fit, self.rank), np.nan)
        self.train_spike_indices = np.full((U, self.n_wfs_fit), -1)

    def check_grow(self):
        n_units = self.unit_ids().size
        if n_units > self.mean.shape[0]:
            U = int(np.ceil(n_units * self.buffer_growth_factor))
            self.mean.resize_(U, *self.mean.shape[1:])
            self.components.resize_(U, *self.components.shape[1:])
            self.train_loadings.resize(U, self.train_loadings.shape[1:])
            self.train_spike_indices.resize(U, self.train_spike_indices.shape[1:])

    def drop_small(self):
        unit_ids, counts = np.unique(self.labels, return_counts=True)
        small = np.flatnonzero(counts < self.min_unit_size)
        if small.size:
            print(f"{small.size} too-small units")
            too_small = np.isin(self.labels, unit_ids[small])
            self.labels[too_small] = -1
        self.pack_units()

    def pack_units(self):
        # order by depth
        snrs = self.unit_channel_snrs()
        sums = snrs.sum(dim=1)
        empty = sums == 0
        sums[empty] = 1
        coms = (snrs * torch.arange(self.data.n_reg_chans)).sum(1) / sums
        coms[empty] = self.data.n_reg_chans + 100
        order = torch.argsort(coms)

        kept = np.flatnonzero(self.labels >= 0)
        flatix, flat_labels = np.unique(self.labels[kept])
        self.labels[kept] = order[flat_labels]

        self.mean[order[flatix]] = self.mean[flatix]
        self.components[order[flatix]] = self.components[flatix]
        self.train_loadings[order[flatix]] = self.train_loadings[flatix]
        self.train_spike_indices[order[flatix]] = self.train_spike_indices[flatix]

    def unit_ids(self):
        uids = np.unique(self.labels)
        uids = uids[uids >= 0]
        assert uids == np.arange(uids.size)
        return uids

    def unit_channel_snrs(self, unit_ids=None):
        uids = self.unit_ids()
        snrs = torch.zeros((uids.size, self.data.n_reg_chans))
        for j, uid in enumerate(uids):
            avs = self.data.static_amp_vecs[self.labels == uid]
            count = torch.sqrt(torch.isfinite(avs).sum(0))
            snrs[j] = torch.nan_to_num(torch.nanmean(avs, dim=0)) * count
        return snrs

    def m_step(self, unit_ids=None):
        if unit_ids is None:
            unit_ids = self.unit_ids()
        bs = self.train_units_batch_size
        for batch_start in range(0, unit_ids.size, bs):
            batch = np.arange(batch_start, min(unit_ids.size, batch_start + bs))
            assert unit_ids[batch] == batch
            X, missing, empty, inds = self.get_training_data(batch)
            loadings, mean, components = fit_pcas(
                X,
                missing,
                empty,
                self.rank,
                max_iter=self.svd_max_iter,
                atol=self.svd_atol,
                show_progress=True,
            )

            self.mean[batch].copy_(mean)
            self.components[batch].copy_(components)
            self.train_loadings[batch] = loadings.numpy(force=True)
            self.train_spike_indices[batch] = inds

    def dpc_split(self, recursive=True, **kwargs):
        unit_ids = self.unit_ids()
        orig_nunits = len(unit_ids)
        i = 1
        while unit_ids:
            ids_to_refit = []
            for uid in tqdm(
                unit_ids, desc=f"Split round {i}" if recursive else "Split", **tqdm_kw
            ):
                ids_to_refit.extend(self.dpc_split_unit(uid, **kwargs))
            self.m_step(np.array(ids_to_refit))
            if recursive:
                unit_ids = ids_to_refit
            else:
                break
        print(f"Split: {orig_nunits} -> {self.unit_ids.size}")

    def embed_spikes(self, unit_id, waveforms, rel_ix):
        n, r, c = waveforms.shape
        m = self.mean[unit_id]
        # intialize empties with the mean
        # TODO: impute here?
        X = F.pad(m, (0, 1))
        X = X[None, None].broadcast_to(n, r, self.data.n_chans_cluster + 1).contiguous()
        rel_ix = rel_ix[:, None, :].broadcast_to((n, r, self.data.n_chans_cluster))
        X.scatter_(waveforms, dim=2, index=rel_ix)
        W = self.components[unit_id]
        return (X - m) @ W.T

    def reconstruct_spikes(self, unit_id, loadings, rel_ix):
        n = len(loadings)
        recons_rel = torch.addmm(self.mean[unit_id], loadings, self.components[unit_id])
        recons_rel = recons_rel.reshape(n, -1 , self.data.n_chans_cluster)
        recons_rel = F.pad(recons_rel, (0, 1))
        rel_ix = rel_ix[:, None, :].broadcast_to((n, -1, self.data.n_chans_cluster))
        return torch.gather(recons_rel, dim=2, index=rel_ix)

    def calc_reconstruction_errors(self):
        unit_ids = self.unit_ids
        n_units = unit_ids.size

        recon_errs = np.full((n_units, self.data.n_spikes), np.inf)

        for uid in unit_ids:
            overlaps, rel_ix = self.spike_overlaps(uid)
            which = np.flatnonzero(overlaps >= self.min_overlap)
            if not which.size:
                print(f"No spikes overlap unit {uid}.")
                continue
            rel_ix = torch.tensor(rel_ix[which], device=self.mean.device)
            for bs in range(0, which.size, self.spike_batch_size):
                be = min(which.size, bs + self.spike_batch_size)
                batch = which[bs:be]

                wfs = self.data.tpca_embeds[batch].to(self.mean)
                loadings = self.embed_spikes(uid, wfs, rel_ix[bs:be])
                recons = self.reconstruct_spikes(uid, loadings, rel_ix[bs:be])

                err = recons
                err -= wfs
                torch.square(err, out=err)
                err = err.sum(1)

                recon_errs[uid, batch] = err

        return recon_errs

    def dpc_split_unit(
        self,
        unit_id,
        rank=2,
        sigma_local="rule_of_thumb",
        sigma_regional=None,
        n_neighbors_search=500,
        remove_clusters_smaller_than=50,
    ):
        """Split the unit. Reassign to sub-units. Update state variables."""
        # run the clustering
        X = self.train_loadings[unit_id]
        in_unit = self.train_spike_indices[unit_id]
        valid = np.flatnonzero(in_unit >= 0)
        in_unit = in_unit[valid]
        X = X[valid, :rank]
        split_labels = density.density_peaks_clustering(
            X,
            sigma_local=sigma_local,
            n_neighbors_search=n_neighbors_search,
            remove_clusters_smaller_than=remove_clusters_smaller_than,
        )
        split_units = np.unique(split_labels)
        split_units = split_units[split_units >= 0]
        if split_units.size <= 1:
            return []

        # replace labels with split labels, update params in those groups
        self.labels[in_unit] = -1
        new_unit_ids = [unit_id, *(self.labels.max() + split_units[split_units >= 1])]
        for split_label, new_label in zip(split_units, new_unit_ids):
            in_split = in_unit[split_labels == split_label]
            self.labels[in_split] = new_label

        return new_unit_ids

    def get_training_data(self, unit_ids):
        """
        Returns
        -------
        X : shape (n_wfs_fit, tpca_dim, n_chans_cluster)
        missing : shape (n_wfs_fit, n_chans_cluster)
        """
        n_units = unit_ids.size
        snrs = self.unit_channel_snrs(unit_ids)
        main_channels = snrs.argmax(1)
        unit_reindexers = self.wf_to_pc_reindexers[main_channels]
        nc_pca = self.data.n_chans_cluster
        nc_full = self.data.n_reg_chans

        X = np.full(
            (n_units, self.n_wfs_fit, self.data.tpca_rank, nc_pca + 1),
            np.nan,
            dtype=np.float32,
        )
        missing = np.ones((n_units, self.n_wfs_fit, nc_pca + 1), dtype=np.bool)
        empty = np.ones((n_units, self.n_wfs_fit), dtype=np.bool)
        spike_indices = np.full((n_units, self.n_wfs_fit), -1)

        for j, uid in enumerate(unit_ids):
            in_unit = np.flatnonzero(self.labels == uid)
            if in_unit.size > self.n_wfs_fit:
                in_unit = self.rg.choice(in_unit, size=self.n_wfs_fit, replace=False)
                in_unit.sort()
            nj = in_unit.size

            chans = self.data.spike_static_channels[in_unit]
            embeds = self.data.tpca_embeds[in_unit]

            reindexer = unit_reindexers[j]
            train_ix = reindexer[chans]

            np.put_along_axis(X[j, :nj], train_ix, embeds, axis=-1)
            np.put_along_axis(missing[j, :nj], train_ix, chans == nc_full, axis=-1)
            empty[j, :nj] = 0
            spike_indices[j, :nj] = in_unit

        missing[empty] = 0
        X = torch.from_numpy(X[..., :-1])
        missing = torch.from_numpy(missing[..., -1])
        missing = missing[..., None, :].broadcast_to(X.shape)
        X = X.reshape(n_units, self.n_wfs_fit, -1)
        missing = missing.reshape(n_units, self.n_wfs_fit, -1)
        empty = torch.from_numpy(empty)

        X = X.to(self.mean.device)
        missing = missing.to(self.mean.device)
        empty = empty.to(self.mean.device)

        return X, missing, empty, spike_indices

    def spike_overlaps(self, unit_ids=None):
        """
        Returns
        -------
        overlaps : (n_units, n_spikes)
        rel_ix : (n_units, n_spikes, n_chans_cluster)
        """
        if unit_ids is None:
            unit_ids = self.unit_ids()
        single = np.isscalar(unit_ids)
        unit_ids = np.atleast_1d(unit_ids)

        snrs = self.unit_channel_snrs(unit_ids)
        main_channels = snrs.argmax(1)
        unit_reindexers = self.wf_to_pc_reindexers[main_channels]

        rel_ix = np.take_along_axis(
            unit_reindexers[:, None],
            self.data.spike_static_channels[None, :],
            axis=2,
        )
        overlapping = rel_ix < self.data.n_chans_cluster
        overlaps = overlapping.mean(2)
        if single:
            assert overlaps.shape[0] == rel_ix.shape[0] == 1
            overlaps = overlaps[0]
            rel_ix = rel_ix[0]
        return overlaps, rel_ix


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


@torch.no_grad
def fit_pcas(
    X,
    missing,
    empty,
    rank,
    max_iter=100,
    check_every=10,
    atol=1e-3,
    show_progress=False,
):
    """
    X : (nu, n, dim_in)
    missing : (nu, n, dim_in)
    empty : (nu, n)
    """
    # ignore = torch.logical_or(missing, empty)
    rows_with_missing = missing.any(dim=-1, keepdims=True)
    missing_in_rows = missing[rows_with_missing]

    # initialize mean
    # Xc = torch.where(ignore, torch.nan, X)
    Xc = X.clone()
    mean = Xc.nanmean(dim=-2, keepdims=True)
    # after this line, isnan(Xc) === empty.
    Xc[missing] = mean.broadcast_to(X.shape)[missing]
    mean = Xc.nanmean(dim=-2, keepdims=True)

    # iterate svds
    it = trange(max_iter, desc="SVD") if show_progress else range(max_iter)
    svd_storage = None
    for j in it:
        # update svd
        Xin = torch.where(empty, 0, Xc - mean)
        svd_storage = torch.linalg.svd(Xin, full_matrices=False, out=svd_storage)
        U, S, Vh = svd_storage
        U = U[..., :rank]
        S = S[..., :rank]
        Vh = Vh[..., :rank, :]

        # impute
        recon_rows = torch.baddbmm(mean, U[rows_with_missing], S[..., None] * Vh)
        dx = (Xc[missing] - recon_rows[missing_in_rows]).abs().max()
        Xc[missing] = recon_rows[missing_in_rows]
        mean = Xc.nanmean(dim=-2, keepdims=True)

        check = not (j + 1 % check_every)
        if check and dx < atol:
            break

    # svd -> pca
    loadings = U * S[..., None, :]
    mean = mean[..., 0, :]
    components = Vh.mT

    return loadings, mean, components


def _load_data(sorting, motion_est, fit_radius, wf_radius=None):
    # load up labels
    labels = sorting.labels
    keepers = np.flatnonzero(labels >= 0)
    labels = labels[keepers]
    channels = sorting.channels[keepers]

    # load waveforms and subset by radius, retaining the new index for later
    with h5py.File(sorting.parent_h5_path, "r", locking=False) as h5:
        tpca_embeds = h5["collisioncleaned_tpca_embeds"][:][keepers]
        amp_vecs = h5["denoised_ptp_amplitude_vectors"][:][keepers]
        geom = h5["geom"][:]
        oci = channel_index = h5["channel_index"][:]
    if wf_radius is not None:
        tpca_embeds, channel_index = waveform_util.channel_subset_by_radius(
            tpca_embeds,
            channels,
            channel_index,
            geom,
        )
        amp_vecs = waveform_util.channel_subset_by_index(
            amp_vecs,
            channels,
            oci,
            channel_index
        )

    pitch = drift_util.get_pitch(geom)
    registered_geom = drift_util.registered_geometry(
        geom, motion_est=motion_est
    )
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
        sorting.channels[keepers],
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
        n_reg_chans=n_reg_chans,
        tpca_rank=tpca_rank,
        n_chans_cluster=n_chans_cluster,
        n_wf_chans=n_wf_chans,
        n_spikes=n_spikes,
        cluster_channel_index=cluster_channel_index,
        static_amp_vecs=static_amp_vecs,
        tpca_embeds=tpca_embeds,
        spike_static_channels=spike_static_channels,
    )
