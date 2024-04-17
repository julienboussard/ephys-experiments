import warnings
from contextlib import nullcontext

from tqdm.auto import tqdm, trange

import h5py
import numpy as np
from dartsort.cluster import density
from dartsort.util import drift_util, waveform_util
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import chi2
from sklearn.decomposition import PCA

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


class SlowSpikePCAiterGMMThing:

    def __init__(
        self,
        sorting,
        svd_rank=20,
        svd_atol=0.2,
        svd_max_iter=100,
        fit_radius=75.0,
        max_wfs_svd=25000,
        geom=None,
        motion_est=None,
        rg=0,
    ):
        self.svd_rank = svd_rank
        self.svd_atol = svd_atol
        self.svd_max_iter = svd_max_iter
        self.rg = np.random.default_rng(rg)
        self.max_wfs_svd = max_wfs_svd

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

        # load up stuff we'll need
        with h5py.File(sorting.parent_h5_path, "r", locking=False) as h5:
            self.tpca_features = h5["collisioncleaned_tpca_features"][:][self.keepers]
            self.amp_vecs = h5["denoised_ptp_amplitude_vectors"][:][self.keepers]
            self.channel_index = h5["channel_index"][:]
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
        self.registered_kdtree = drift_util.KDTree(self.registered_geom)
        self.match_distance = drift_util.pdist(geom).min() / 2
        self.registered_channel_index = waveform_util.make_channel_index(
            self.registered_geom, fit_radius
        )
        self.channels = sorting.channels[self.keepers]

        # this might be too big to store for real
        self.rfull_amp_vecs = drift_util.get_waveforms_on_static_channels(
            self.amp_vecs,
            geom,
            main_channels=self.channels,
            n_pitches_shift=self.n_pitches_shift,
            channel_index=self.channel_index,
            registered_geom=self.registered_geom,
        )
        self.rfull_tpca_features = drift_util.get_waveforms_on_static_channels(
            self.tpca_features,
            geom,
            main_channels=self.channels,
            n_pitches_shift=self.n_pitches_shift,
            channel_index=self.channel_index,
            registered_geom=self.registered_geom,
        )

        # initialize params
        self.m_step()

    def reset(self):
        self.inus = {}
        self.loadings = {}
        self.pcas = {}
        self.chans = {}
        self.precisions = {}
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
            snr = np.nan_to_num(nanmean(avs, axis=0)) * np.sqrt(np.isfinite(avs).sum(0))
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
            c = self.registered_channel_index[self.maxchans[uid]]
            self.chans[uid] = c[c < self.n_reg_chans]
            which = inu
            if inu.size > self.max_wfs_svd:
                which = self.rg.choice(inu, size=self.max_wfs_svd, replace=False)
                which.sort()
            self.inus_pca[uid] = which
            self.train_feats[uid] = self.rfull_tpca_features[which][
                :, :, self.chans[uid]
            ]
            # TODO: random_state in PCA
            self.loadings[uid], self.pcas[uid] = pca_iter_impute(
                self.train_feats[uid].reshape(which.size, -1),
                rank=self.svd_rank,
                atol=self.svd_atol,
                max_iter=self.svd_max_iter,
                show_progress=False,
            )
            self.precisions[uid] = get_precision_full(
                self.pcas[uid],
                self.loadings[uid],
                self.train_feats[uid],
                self.chans[uid],
                self.n_reg_chans,
            )
            c = np.full((self.train_feats[uid].shape[1], self.n_reg_chans), np.nan)
            c[:, self.chans[uid]] = self.pcas[uid].mean_.reshape(
                -1, self.chans[uid].size
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
    ):
        """Split the unit. Reassign to sub-units. Update state variables."""
        # run the clustering
        X = self.loadings[unit_id][:, :rank]
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

        # TODO: reassign, replace labels again, update PCA again???

    def reassign(self, min_size=50, outlier_quantile=0.9):
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
        self.logliks = np.full((self.keepers.size, unit_ids.size), np.nan)
        self.chi_quantiles = np.full((self.keepers.size, unit_ids.size), np.nan)
        for j, uid in enumerate(tqdm(unit_ids, desc="Mahalanobis", **tqdm_kw)):
            # mahal, ns = ppca_mahal_mychans(
            #     self.pcas[uid],
            #     self.loadings[uid],
            #     self.train_feats[uid],
            #     rfull_tpca_features=self.rfull_tpca_features,
            #     chans=self.chans[uid],
            #     n_reg_chans=self.n_reg_chans,
            # )
            mahal, ns, loglik = svd_mahal_mychans(
                self.pcas[uid],
                self.loadings[uid],
                self.train_feats[uid],
                self.rfull_tpca_features,
                chans=self.chans[uid],
                n_reg_chans=self.n_reg_chans,
                max_iter=self.svd_max_iter,
                atol=self.svd_atol,
                # show_progress=False,
            )
            self.mahals[:, j] = mahal
            self.nsamps[:, j] = ns
            self.chi_quantiles[:, j] = chi2.cdf(mahal, ns)
            self.logliks[:, j] = loglik
            print(f"{self.mahals[:, j].min()=} {self.mahals[:, j].max()=}")
            print(f"{self.nsamps[:, j].min()=} {self.nsamps[:, j].max()=}")
            print(
                f"{self.chi_quantiles[:, j].min()=} {self.chi_quantiles[:, j].max()=}"
            )
            print(f"{self.logliks[:, j].min()=} {self.logliks[:, j].max()=}")
        # assert np.isfinite(self.chi_quantiles).all()
        self.old_labels = self.labels
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

    def mahalanobis_matrix(self, sym_function=np.minimum):
        # return mahalanobis distance matrix between centroids?
        # do we need to use "standard error" flavored covariances here (i.e n^{-1/2})
        # for this to be in nice chi units?
        # could also look at other distances? this one is not symmetric etc.
        unit_ids = self.unit_ids
        mahal_dists = np.full((unit_ids.size, unit_ids.size), np.nan)
        pair_chi_quantiles = np.full((unit_ids.size, unit_ids.size), np.nan)
        pair_nsamps = np.full((unit_ids.size, unit_ids.size), np.nan)
        rfull_centroids = np.stack(list(self.rfull_centroids.values()), axis=0)
        mask = np.all(rfull_centroids == 0, axis=1)
        rfull_centroids[np.broadcast_to(mask[:, None, :], rfull_centroids.shape)] = (
            np.nan
        )
        for j, uid in enumerate(tqdm(unit_ids, desc="Pairwise Mahalanobis", **tqdm_kw)):
            # mahal, nsamps = ppca_mahal_mychans(
            #     self.pcas[uid],
            #     self.loadings[uid],
            #     self.train_feats[uid],
            #     rfull_centroids,
            #     self.chans[uid],
            #     self.n_reg_chans,
            # )
            mahal, nsamps = svd_mahal_mychans(
                self.pcas[uid],
                self.loadings[uid],
                self.train_feats[uid],
                rfull_centroids,
                chans=self.chans[uid],
                n_reg_chans=self.n_reg_chans,
                max_iter=self.svd_max_iter,
                atol=self.svd_atol,
            )
            mahal_dists[:, j] = mahal
            pair_nsamps[:, j] = nsamps
            pair_chi_quantiles[:, j] = chi2.cdf(mahal, nsamps)
        mahal_dists = sym_function(mahal_dists, mahal_dists.T)
        pair_chi_quantiles = sym_function(pair_chi_quantiles, pair_chi_quantiles.T)
        return mahal_dists, pair_chi_quantiles, pair_nsamps

    def mahalanobis_merge(
        self, sym_function=np.minimum, link="complete", chi2_quantile=0.2
    ):
        mahal_dists, pair_chi_quantiles, nsamps = self.mahalanobis_matrix(
            sym_function=sym_function
        )
        n_units = pair_chi_quantiles.shape[0]

        # hierarchical clustering
        pdist = pair_chi_quantiles[np.triu_indices(pair_chi_quantiles.shape[0], k=1)]
        Z = linkage(pdist, method=link)
        new_labels = fcluster(Z, chi2_quantile, criterion="distance")

        # update labels
        kept = np.flatnonzero(self.labels >= 0)
        _, flat_labels = np.unique(self.labels[kept], return_inverse=True)
        self.labels[kept] = new_labels[flat_labels]

        self.reset()
        print(f"Merge: {n_units} -> {self.unit_ids.size}")


def nanmean(x, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            category=RuntimeWarning, action="ignore", message="Mean of empty slice"
        )
        return np.nanmean(x, **kwargs)


def pca_iter_impute(
    X, rank=10, max_iter=1000, atol=1e-4, whiten=False, show_progress=True
):
    """See the lovely JMLR paper of Ilin and Raiko, 2010."""
    missing = np.isnan(X)
    any_missing = np.flatnonzero(missing.any(axis=1))
    mm = missing[any_missing]
    mean0 = np.nan_to_num(nanmean(X, axis=0))

    # initialize imputed vals with mean0
    Xc = np.where(missing, mean0[None, :], X)
    pca = PCA(n_components=rank, whiten=whiten)

    zprev = np.inf
    cm = trange(max_iter) if show_progress else nullcontext(range(max_iter))
    with cm as pbar:
        for it in pbar:
            z = pca.fit_transform(Xc)
            dz = np.abs(z - zprev).max()
            if show_progress:
                pbar.set_description(f"{dz=:0.6f}")
            if dz < atol:
                break
            zprev = z

            Xc[missing] = pca.inverse_transform(z[any_missing])[mm]

    return z, pca


def apply_pca_impute(pca, X, max_iter=1000, atol=1e-4, show_progress=False):
    missing = np.isnan(X)
    any_missing = missing.any(axis=1)

    # initialize imputed vals with mean0
    Xc = np.where(missing, pca.mean_[None, :], X)
    zprev = pca.transform(Xc)
    if not any_missing.any():
        return zprev, Xc, missing
    mm = missing[any_missing]
    zprevm = zprev[any_missing]
    Xm = Xc[any_missing]

    cm = (
        trange(max_iter, desc="PCA reconstruct")
        if show_progress
        else nullcontext(range(max_iter))
    )
    with cm as pbar:
        for it in pbar:
            zm = pca.transform(Xm)
            dz = np.abs(zm - zprevm).max()
            if show_progress:
                pbar.set_description(f"{dz=:0.6f}")
            if it > 0 and dz < atol:
                break
            zprevm = zm
            Xm[mm] = pca.inverse_transform(zm)[mm]
    z = zprev
    z[any_missing] = zm
    Xc[any_missing] = Xm

    return z, Xc, missing


# def ppcad(X, rank=10, max_iter=1000, atol=1e-4, show_progress=True):
#


def get_var(pca, z, wfs):  # , var_kind="scalar"):
    # if var_kind == "scalar":
    return np.nanvar(wfs.reshape(wfs.shape[0], -1) - pca.inverse_transform(z))
    # elif var_kind == "diag"
    #     return np.nanvar(wfs.reshape(wfs.shape[0], -1) - pca.inverse_transform(z), axis=0)


def get_precision_full(pca, z, wfs, chans, n_reg_chans):
    assert wfs.shape[2] == chans.size
    v = get_var(pca, z, wfs)
    rank = pca.components_.shape[0]
    W = pca.components_.reshape(rank, -1, chans.size)
    W_ = np.zeros((rank, W.shape[1], n_reg_chans), dtype=W.dtype)
    W_[:, :, chans] = W
    W = W_.T.reshape(-1, rank)
    d, r = W.shape
    Iinv = np.eye(d) / v
    Minv = Iinv - (W @ np.linalg.inv(np.eye(r) + W.T @ W / v) @ W.T) / (v * v)
    return Minv


def ppca_mahal_mychans(
    pca, z, wfs_in, rfull_tpca_features, chans, n_reg_chans, which_out=slice(None)
):
    wfs_out_in = rfull_tpca_features[which_out]
    wfs_out_in = wfs_out_in.reshape(len(wfs_out_in), -1)
    Minv = get_precision_full(pca, z, wfs_in, chans, n_reg_chans)

    finite = np.isfinite(wfs_out_in)
    wfs_out_in = np.nan_to_num(wfs_out_in)
    mu = pca.mean_.reshape(-1, len(chans))
    mu_ = np.zeros((mu.shape[0], n_reg_chans), dtype=mu.dtype)
    mu_[:, chans] = mu
    mu = mu_.ravel()
    dx = wfs_out_in - mu
    dx[~finite] = 0.0
    mahal = dx @ Minv
    mahal *= dx
    mahal = mahal.sum(1)
    # mahal = np.einsum("iq,iq->i", dx @ Minv, dx)
    # mahal = np.einsum("ip,pq,iq->i", dx, Minv, dx)
    nsamps = finite.sum(1)
    return mahal, nsamps


def svd_mahal_mychans(
    pca,
    z_in,
    wfs_in,
    rfull_tpca_features,
    chans,
    n_reg_chans,
    which_out=slice(None),
    max_iter=1000,
    atol=1e-4,
    show_progress=True,
):
    wfs_out_in = rfull_tpca_features[which_out]
    wfs_out_in_pca = rfull_tpca_features[which_out, :, chans]
    wfs_out_in = wfs_out_in.reshape(len(wfs_out_in), -1)

    train_resids = wfs_in - pca.inverse_transform(z_in).reshape(wfs_in.shape)
    var = np.nanvar(train_resids)

    entirely_missing = np.isnan(wfs_out_in_pca).all(axis=(1, 2))
    not_entirely_missing = np.flatnonzero(np.logical_not(entirely_missing))
    n_fit = not_entirely_missing.size
    embeds, Xc, missing = apply_pca_impute(
        pca,
        wfs_out_in_pca[not_entirely_missing].reshape(n_fit, -1),
        max_iter=max_iter,
        atol=atol,
        show_progress=show_progress,
    )
    resids = wfs_out_in.copy().reshape(len(wfs_out_in), *rfull_tpca_features.shape[1:])
    resids[
        not_entirely_missing[:, None, None],
        np.arange(resids.shape[1])[None, :, None],
        chans[None, None, :],
    ] -= pca.inverse_transform(embeds).reshape(resids[:, :, chans].shape)
    resids = resids.reshape(len(resids), -1)
    # var = np.nanvar(resids)
    nsamps = np.zeros(len(resids), dtype=int)
    nsamps[not_entirely_missing] = np.sum(np.logical_not(missing), axis=1)
    mahals = np.nansum(np.square(resids) / var, axis=1)
    logliks = -0.5 * mahals - 0.5 * nsamps * (np.log(var) + np.log(2 * np.pi))
    logliks[nsamps == 0] = -np.inf
    return mahals, nsamps, logliks
