import warnings
from contextlib import nullcontext

from tqdm.auto import tqdm, trange

import h5py
import numpy as np
import torch
from dartsort.cluster import density
from dartsort.util import drift_util, waveform_util
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import chi2
from sklearn.decomposition import PCA
import torch.nn.functional as F

from .ppca import PPCA, VBPCA

"""

# 2 alternatives for the reassignment, and i guess cluster mahal

1: common visible chans
    Say point i and cluster q overlap on k chans
    Then mahal(i,q) is chi^2_k*rank
    Then mahal(i,q)/k can maybe be used for assignment?
    Or is this just a fancy way of writing 1.
2: full bayes
    just like in ppca, we can get a distribution of the mahals when we have missing chans (or with all chans?)
"""

tqdm_kw = dict(smoothing=0)


class SpikeLrGmm:
    def __init__(
        self,
        sorting,
        svd_rank=20,
        svd_atol=0.2,
        svd_max_iter=100,
        fit_radius=75.0,
        wfs_radius=50.0,
        max_wfs_svd=25000,
        # geom=None,
        motion_est=None,
        learn_vy=True,
        rg=0,
        device="cpu",
    ):
        self.svd_rank = svd_rank
        self.svd_atol = svd_atol
        self.svd_max_iter = svd_max_iter
        self.rg = np.random.default_rng(rg)
        self.max_wfs_svd = max_wfs_svd
        self.learn_vy = learn_vy
        self.device = device

        # reindex labels -- this class will work in contiguous label space always
        # the torch.LongTensor call below will copy
        labels = sorting.labels
        self.keepers = np.flatnonzero(labels >= 0)
        self.labels = labels[self.keepers]
        # need to have enough spikes for svd
        unit_ids, counts = np.unique(self.labels, return_counts=True)
        small = np.flatnonzero(counts < 2 * svd_rank)
        if small.size:
            print(f"{small.size} too-small units")
            too_small = np.isin(self.labels, unit_ids[small])
            self.labels[too_small] = -1
        channels = sorting.channels[self.keepers]

        # load up stuff we'll need
        with h5py.File(sorting.parent_h5_path, "r", locking=False) as h5:
            self.tpca_features = h5["collisioncleaned_tpca_features"][:][self.keepers]
            self.amp_vecs = h5["denoised_ptp_amplitude_vectors"][:][self.keepers]
            self.channel_index = h5["channel_index"][:]
            geom = h5["geom"][:]
        if wfs_radius:
            self.tpca_features, new_channel_index = waveform_util.channel_subset_by_radius(
                self.tpca_features,
                channels,
                self.channel_index,
                geom,
                radius=wfs_radius,
            )
            self.amp_vecs = waveform_util.channel_subset_by_index(
                self.amp_vecs, channels, self.channel_index, new_channel_index
            )
            self.channel_index = new_channel_index
        self.n_pitches_shift = drift_util.get_spike_pitch_shifts(
            sorting.point_source_localizations[:, 2][self.keepers],
            geom=geom,
            motion_est=motion_est,
            times_s=sorting.times_seconds[self.keepers],
        )
        self.registered_geom = drift_util.registered_geometry(
            geom, motion_est=motion_est
        )
        self.n_reg_chans = len(self.registered_geom)
        self.tpca_rank = self.tpca_features.shape[1]
        self.max_n_features = self.tpca_rank * self.n_reg_chans
        self.full_index_set = torch.arange(self.max_n_features).reshape(
            self.tpca_rank, self.n_reg_chans
        )

        # need to figure out what channels things land on
        self.geom = geom
        self.pitch = drift_util.get_pitch(geom)
        self.registered_kdtree = drift_util.KDTree(self.registered_geom)
        self.match_distance = drift_util.pdist(geom).min() / 2
        self.registered_channel_index = waveform_util.make_channel_index(
            self.registered_geom, fit_radius
        )
        self.channels = torch.from_numpy(channels)

        # this might be too big to store for real
        # n_spikes, chans_per_spike
        self.spike_static_channels = drift_util.static_channel_neighborhoods(
            geom,
            sorting.channels[self.keepers],
            self.channel_index,
            pitch=self.pitch,
            n_pitches_shift=self.n_pitches_shift,
            registered_geom=self.registered_geom,
            target_kdtree=self.registered_kdtree,
            match_distance=self.match_distance,
            workers=4,
        )
        self.spike_static_channels = torch.from_numpy(self.spike_static_channels)
        self.observed = self.spike_static_channels < self.n_reg_chans
        self.observed_features = self.observed[:, None, :].broadcast_to(
            self.tpca_features.shape
        )
        self.tpca_features = torch.from_numpy(self.tpca_features)
        self.rfull_amp_vecs = drift_util.grab_static(
            torch.from_numpy(self.amp_vecs),
            self.spike_static_channels,
            self.n_reg_chans,
        )
        # self.rfull_amp_vecs = drift_util.get_waveforms_on_static_channels(
        #     self.amp_vecs,
        #     geom,
        #     main_channels=self.channels,
        #     n_pitches_shift=self.n_pitches_shift,
        #     channel_index=self.channel_index,
        #     registered_geom=self.registered_geom,
        # )

        # initialize params
        self.m_step()

    def reset(self):
        self.inus = {}
        self.loadings = {}
        self.pcas = {}
        self.chans = {}
        self.train_feats = {}
        self.rfull_centroids = {}

        self.get_unit_ids()
        self.get_inus()

    def m_step(self):
        """Update params in each cluster"""
        self.reset()
        self.get_unit_max_channels()
        self.fit_pcas()

    def get_unit_ids(self):
        uids = np.unique(self.labels)
        self.unit_ids = uids[uids >= 0]

    def get_inus(self):
        self.inus = {}
        self.inus_pca = {}
        for uid in self.unit_ids:
            self.inus_pca[uid] = self.inus[uid] = np.flatnonzero(self.labels == uid)

    def get_unit_max_channels(self, unit_ids=None):
        if unit_ids is None:
            unit_ids = self.unit_ids
        maxchans = {}
        for uid in unit_ids:
            avs = self.rfull_amp_vecs[self.inus[uid]]
            snr = torch.nan_to_num(torch.nanmean(avs, axis=0)) * torch.sqrt(
                torch.isfinite(avs).sum(0)
            )
            maxchans[uid] = snr.argmax()
        self.maxchans = maxchans

    def fit_pcas(self, unit_ids=None, show_progress=True):
        """TODO: Restrict to inliers when chi2 scores are computed."""
        if unit_ids is None:
            unit_ids = self.unit_ids

        if show_progress:
            unit_ids = tqdm(unit_ids, desc="PCA", **tqdm_kw)

        for uid in unit_ids:
            inu = self.inus[uid]
            active_channels = self.registered_channel_index[self.maxchans[uid]]
            active_channels = active_channels[active_channels < self.n_reg_chans]
            self.chans[uid] = active_channels
            which = inu
            if inu.size > self.max_wfs_svd:
                which = self.rg.choice(inu, size=self.max_wfs_svd, replace=False)
                which.sort()
            self.inus_pca[uid] = which
            full_feats = drift_util.grab_static(
                self.tpca_features[which],
                self.spike_static_channels[which],
                self.n_reg_chans,
            )
            self.train_feats[uid] = full_feats[:, :, active_channels]
            # TODO: random_state in PCA
            X = self.train_feats[uid].reshape(which.size, -1)
            self.pcas[uid] = VBPCA(
                X.shape[1],
                self.svd_rank,
                active_indices=self.full_index_set[:, active_channels].reshape(-1),
                max_feature_ind=self.max_n_features,
                learn_vy=self.learn_vy,
            )
            self.pcas[uid].to(self.device)
            self.loadings[uid], _ = self.pcas[uid].fit_transform(
                torch.as_tensor(X, device=self.device),
                max_iter=self.svd_max_iter,
                atol=self.svd_atol,
            )
            self.pcas[uid].to("cpu")
            self.loadings[uid] = self.loadings[uid].numpy(force=True)
            c = torch.full((self.tpca_rank, self.n_reg_chans), torch.nan)
            c[:, self.chans[uid]] = (
                self.pcas[uid]
                .mbar.reshape(self.tpca_rank, self.chans[uid].size)
                .cpu()
                .detach()
            )
            self.rfull_centroids[uid] = c

    def dpc_split(self, recursive=True, **kwargs):
        unit_ids = list(self.unit_ids)
        orig_nunits = len(unit_ids)
        i = 1
        while unit_ids:
            new_ids = []
            for uid in tqdm(
                unit_ids, desc=f"Split round {i}" if recursive else "Split", **tqdm_kw
            ):
                new_ids.extend(self.dpc_split_unit(uid, **kwargs))
            if recursive:
                print(f"Split round {i}: {len(new_ids)} new units.")
                unit_ids = new_ids
        print(f"Split: {orig_nunits} -> {self.unit_ids.size}")

    def dpc_split_unit(
        self,
        unit_id,
        rank=2,
        sigma_local="rule_of_thumb",
        sigma_regional=None,
        n_neighbors_search=500,
        remove_clusters_smaller_than=50,
        min_bin_size=0.01,
    ):
        """Split the unit. Reassign to sub-units. Update state variables."""
        # run the clustering
        X = self.loadings[unit_id][:, :rank]
        split_labels = density.density_peaks_clustering(
            X,
            sigma_local=sigma_local,
            n_neighbors_search=n_neighbors_search,
            remove_clusters_smaller_than=remove_clusters_smaller_than,
            min_bin_size=min_bin_size,
        )
        split_units = np.unique(split_labels)
        split_units = split_units[split_units >= 0]
        if split_units.size <= 1:
            return []

        # remove all points from cluster
        self.labels[self.inus[unit_id]] = -1

        # re-do labels with split labels, update params in those groups
        inu = self.inus_pca[unit_id]
        new_unit_ids = np.concatenate(
            ([unit_id], self.labels.max() + split_units[split_units >= 1])
        )
        for split_label, new_label in zip(split_units, new_unit_ids):
            in_split = inu[split_labels == split_label]
            self.labels[in_split] = new_label
            self.inus_pca[new_label] = self.inus[new_label] = in_split
        self.unit_ids = np.unique(np.concatenate((self.unit_ids, new_unit_ids)))
        self.get_unit_max_channels(new_unit_ids)
        self.fit_pcas(new_unit_ids, show_progress=False)

        return new_unit_ids
    
    def embedding_reassign(self, batch_size=32, min_size=50, outlier_quantile=0.9):
        unit_ids = self.unit_ids
        self.embed_norms = np.full((self.keepers.size, unit_ids.size), np.nan)
        self.embed_mahals = np.full((self.keepers.size, unit_ids.size), np.nan)
        self.embed_overlaps = np.full((self.keepers.size, unit_ids.size), np.nan)

        for batch_start in trange(
            0, self.keepers.size, batch_size, desc="Reassign", **tqdm_kw
        ):
            batch_end = min(self.keepers.size, batch_start + batch_size)
            bsl = slice(batch_start, batch_end)
            nbatch = batch_end - batch_start

            X = self.tpca_features[bsl].reshape(nbatch, -1)
            chans = self.spike_static_channels[bsl]
            Xix = F.pad(self.full_index_set, (0, 1, 0, 0), value=self.max_n_features)
            Xix = Xix[None].broadcast_to(nbatch, *Xix.shape)
            Xix = Xix[
                torch.arange(nbatch)[:, None, None],
                torch.arange(self.tpca_rank)[None, :, None],
                chans[:, None, :],
            ]
            Xix = Xix.reshape(nbatch, -1)
            O = torch.logical_and(torch.isfinite(X), Xix < self.max_n_features)

            for j, uid in enumerate(unit_ids):
                self.pcas[uid].to(self.device)
                x, S, O = self.pcas[uid](X.to(self.device), O.to(self.device), Xix.to(self.device), return_O=True)
                self.pcas[uid].to("cpu")

                self.embed_norms[bsl, j] = torch.square(x).sum(dim=1)
                self.embed_mahals[bsl, j] = torch.einsum("ni,nij,nj", x, torch.linalg.inv(S), x)
                self.embed_overlaps[bsl, j] = O.sum(1)

        # assert np.isfinite(self.chi_quantiles).all()
        # self.old_labels = self.labels
        # best_inds = np.nanargmax(self.logliks, axis=1)
        # self.labels = unit_ids[best_inds]
        # if outlier_quantile:
        #     q = self.chi_quantiles[np.arange(best_inds.size), best_inds]
        #     outliers = (q > outlier_quantile) | np.isnan(q)
        #     print(f"Outlier fraction: {outliers.mean()}")
        #     self.labels[outliers] = -1
        # if min_size:
        #     unit_ids, counts = np.unique(self.labels, return_counts=True)
        #     small = np.flatnonzero(counts < min_size)
        # if min_size and small.size:
        #     print(f"{small.size} small units")
        #     too_small = np.isin(self.labels, unit_ids[small])
        #     self.labels[too_small] = -1
        # self.reset()
    
    def pair_embedding_distances(self):
        # return mahalanobis distance matrix between centroids?
        # do we need to use "standard error" flavored covariances here (i.e n^{-1/2})
        # for this to be in nice chi units?
        # could also look at other distances? this one is not symmetric etc.
        unit_ids = self.unit_ids
        pair_embed_norms = np.full((unit_ids.size, unit_ids.size), np.nan)
        pair_embed_mahals = np.full((unit_ids.size, unit_ids.size), np.nan)
        overlaps = np.full((unit_ids.size, unit_ids.size), np.nan)

        for i, uid in enumerate(tqdm(unit_ids, desc="Pairwise embed", **tqdm_kw)):
            # self.pcas[uid].to(self.device)
            # for j, ujd in enumerate(unit_ids[i + 1:], start=i + 1):
            for j, ujd in enumerate(unit_ids):
                x, S = self.pcas[uid](
                    self.pcas[ujd].mbar[None], None, self.pcas[ujd].active_indices[None]
                )
                x = x[0]
                S = S[0]
                pair_embed_norms[i, j] = torch.square(x).sum()
                pair_embed_mahals[i, j] = torch.einsum("i,ij,j", x, torch.linalg.inv(S), x)
                overlaps[i, j] = np.intersect1d(
                    self.pcas[uid].active_indices,
                    self.pcas[ujd].active_indices,
                ).size
            # self.pcas[uid].to("cpu")

        return pair_embed_norms, pair_embed_mahals, overlaps

    def mahalanobis_reassign(self, batch_size=256, min_size=50, outlier_quantile=0.9, min_overlap=0.5):
        """Reassign points to new cluster labels

        TODO: currently looks at all clusters as candidates. let's not.

        For each cluster, get some neighbors (Mahal? Mahal Chi2? L2?). For all
        points in those neighbors, get their Mahalanobis to this cluster and
        their chi2 score.

        Now, for each point, assign it to its best cluster according to chi2.
        Store new labels and the chi2 cdf values.

        Note: points that don't belong to clusters can't be reassigned.
        Maybe a reassign_outliers is necessary if we want to get those back.
        """
        unit_ids = self.unit_ids
        self.mahals = np.full((self.keepers.size, unit_ids.size), np.nan)
        self.nsamps = np.full((self.keepers.size, unit_ids.size), np.nan)
        self.overlaps = np.full((self.keepers.size, unit_ids.size), np.nan)
        self.logliks = np.full((self.keepers.size, unit_ids.size), np.nan)
        self.chi_quantiles = np.full((self.keepers.size, unit_ids.size), np.nan)
        for pca in self.pcas.values():
            pca.to(self.device)
        for batch_start in trange(
            0, self.keepers.size, batch_size, desc="Reassign", **tqdm_kw
        ):
            batch_end = min(self.keepers.size, batch_start + batch_size)
            bsl = slice(batch_start, batch_end)
            nbatch = batch_end - batch_start

            X = self.tpca_features[bsl].reshape(nbatch, -1)
            chans = self.spike_static_channels[bsl]
            Xix = F.pad(self.full_index_set, (0, 1, 0, 0), value=self.max_n_features)
            Xix = Xix[None].broadcast_to(nbatch, *Xix.shape)
            Xix = Xix[
                torch.arange(nbatch)[:, None, None],
                torch.arange(self.tpca_rank)[None, :, None],
                chans[:, None, :],
            ]
            Xix = Xix.reshape(nbatch, -1)
            O = torch.logical_and(torch.isfinite(X), Xix < self.max_n_features)
            X = X.to(self.device)
            O = O.to(self.device)
            Xix = Xix.to(self.device)

            results = []
            for j, uid in enumerate(unit_ids):
                # self.pcas[uid].to(self.device)
                res = self.pcas[uid].predict(
                    X, O, Xix, return_overlaps=True
                )
                results.append(res)
            for j, (mahals, logliks, chi_qs, nobs, overlaps) in enumerate(results):
                # self.pcas[uid].to("cpu")
                self.mahals[bsl, j] = mahals.numpy(force=True)
                self.logliks[bsl, j] = logliks.numpy(force=True)
                self.nsamps[bsl, j] = nobs.numpy(force=True)
                self.overlaps[bsl, j] = overlaps.numpy(force=True)
                self.chi_quantiles[bsl, j] = chi_qs.numpy(force=True)

        for pca in self.pcas.values():
            pca.to("cpu")
        # assert np.isfinite(self.chi_quantiles).all()
        self.old_labels = self.labels
        if min_overlap:
            self.logliks[self.overlaps < min_overlap] = -np.inf
        best_inds = np.nanargmax(self.logliks, axis=1)
        self.labels = unit_ids[best_inds]
        if outlier_quantile:
            q = self.chi_quantiles[np.arange(best_inds.size), best_inds]
            outliers = (q > outlier_quantile) | np.isnan(q)
            print(f"Outlier fraction: {outliers.mean()}")
            self.labels[outliers] = -1
        if min_size:
            unit_ids, counts = np.unique(self.labels, return_counts=True)
            small = np.flatnonzero(counts < min_size)
        if min_size and small.size:
            print(f"{small.size} small units")
            too_small = np.isin(self.labels, unit_ids[small])
            self.labels[too_small] = -1
        self.reset()

    def mahalanobis_matrix(self, sym_function=np.minimum, batch_size=8):
        # return mahalanobis distance matrix between centroids?
        # do we need to use "standard error" flavored covariances here (i.e n^{-1/2})
        # for this to be in nice chi units?
        # could also look at other distances? this one is not symmetric etc.
        unit_ids = self.unit_ids
        mahal_dists = np.full((unit_ids.size, unit_ids.size), np.nan)
        pair_chi_quantiles = np.full((unit_ids.size, unit_ids.size), np.nan)
        pair_nsamps = np.full((unit_ids.size, unit_ids.size), np.nan)
        overlaps = np.full((unit_ids.size, unit_ids.size), np.nan)

        for i, uid in enumerate(tqdm(unit_ids, desc="Pairwise Mahalanobis", **tqdm_kw)):
            # self.pcas[uid].to(self.device)
            # for j, ujd in enumerate(unit_ids[i + 1:], start=i + 1):
            for j, ujd in enumerate(unit_ids):
                mahals, logliks, chi_qs, nobs = self.pcas[uid].predict(
                    self.pcas[ujd].mbar[None],
                    Onew=None,
                    Y_indices_new=self.pcas[ujd].active_indices[None],
                )
                mahal_dists[i, j] = mahals.numpy(force=True)
                pair_nsamps[i, j] = nobs.numpy(force=True)
                pair_chi_quantiles[i, j] = chi_qs.numpy(force=True)
                overlaps[i, j] = np.intersect1d(
                    self.pcas[uid].active_indices,
                    self.pcas[ujd].active_indices,
                ).size
            # self.pcas[uid].to("cpu")

        mahal_dists = sym_function(mahal_dists, mahal_dists.T)
        pair_chi_quantiles = sym_function(pair_chi_quantiles, pair_chi_quantiles.T)
        return mahal_dists, pair_chi_quantiles, pair_nsamps, overlaps

    def mahalanobis_merge(
        self, sym_function=np.maximum, link="complete", chi2_quantile=0.05
    ):
        mahal_dists, pair_chi_quantiles, nsamps, overlaps = self.mahalanobis_matrix(
            sym_function=sym_function
        )
        from scipy.stats import chi2

        chi_quantiles2 = chi2.cdf(mahal_dists, df=overlaps)

        D = sym_function(chi_quantiles2, chi_quantiles2.T)
        D[overlaps == 0] = 1
        D = np.nan_to_num(D, nan=1.0)
        np.fill_diagonal(D, 0)

        n_units = pair_chi_quantiles.shape[0]

        # hierarchical clustering
        pdist = D[np.triu_indices(D.shape[0], k=1)]
        Z = linkage(pdist, method=link)
        new_labels = fcluster(Z, D, criterion="distance")

        # update labels
        kept = np.flatnonzero(self.labels >= 0)
        _, flat_labels = np.unique(self.labels[kept], return_inverse=True)
        self.labels[kept] = new_labels[flat_labels]

        self.reset()
        print(f"Merge: {n_units} -> {self.unit_ids.size}")

    def pair_embed(self, uid, ujd):
        nj = len(self.train_feats[ujd])
        ni = len(self.train_feats[uid])
        return dict(
            ii=self.loadings[uid],
            ij=self.pcas[uid](
                self.train_feats[ujd].reshape(nj, -1),
                None,
                self.pcas[ujd].active_indices[None].broadcast_to(nj, -1).contiguous(),
            )[0],
            ij0=self.pcas[uid](
                self.pcas[ujd].mbar[None], None, self.pcas[ujd].active_indices[None]
            )[0][0],
            ij0s=self.pcas[uid](
                self.pcas[ujd].mbar[None], None, self.pcas[ujd].active_indices[None]
            )[1][0],
            jj=self.loadings[ujd],
            ji=self.pcas[ujd](
                self.train_feats[uid].reshape(ni, -1),
                None,
                self.pcas[uid].active_indices[None].broadcast_to(ni, -1).contiguous(),
            )[0],
            ji0=self.pcas[ujd](
                self.pcas[uid].mbar[None], None, self.pcas[uid].active_indices[None]
            )[0][0],
            ji0s=self.pcas[ujd](
                self.pcas[uid].mbar[None], None, self.pcas[uid].active_indices[None]
            )[1][0],
        )
