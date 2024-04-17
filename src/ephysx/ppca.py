from tqdm.auto import trange

import torch
import torch.nn.functional as F
from torch import nn


class PPCA(torch.nn.Module):
    def __init__(self, d, c, learn_vy=True):
        super().__init__()
        self.d = d
        self.c = c
        self.m = nn.Parameter(torch.zeros(d), requires_grad=False)
        self.w = nn.Parameter(torch.zeros(d, c), requires_grad=False)
        self.vy = nn.Parameter(torch.ones(()), requires_grad=False)
        self.learn_vy = learn_vy
        self.register_buffer("Ic", torch.eye(c))

    def initialize_from(self, Y, O):
        with torch.no_grad():
            m = torch.nan_to_num(torch.nanmean(Y, dim=0))
            self.m.copy_(m)
            Yc = torch.nan_to_num(Y - m)
            u, s, vh = torch.linalg.svd(Yc, full_matrices=False)
            vh = vh[: self.c]
            s = s[: self.c]
            self.w.copy_(vh.T)

    def forward(self, Y, O=None):
        if O is None:
            O = torch.isfinite(Y)
        O_ = O.to(self.w)
        Y = torch.nan_to_num(Y)

        Oww = torch.einsum("ni,ik,il->nkl", O_, self.w, self.w)
        Sigma_xn = self.vy * torch.linalg.inv(self.vy * self.Ic[None] + Oww)
        xbarn = torch.einsum(
            "nlk,ni,ik,ni->nl", Sigma_xn / self.vy, O_, self.w, Y - self.m
        )

        return xbarn, Sigma_xn

    def m_step(self, Y, O, xbarn, Sigma_xn):
        O_ = O.to(self.w)
        m = torch.nanmean(torch.addmm(Y, xbarn, self.w.T, alpha=-1), axis=0)

        # xxTSigmax = xbarn[:, :, None] * xbarn[:, None, :] + Sigma_xn
        xxTSigmax = torch.baddbmm(Sigma_xn, xbarn[:, :, None], xbarn[:, None, :])
        xxTSigmax = O_.T @ xxTSigmax.reshape(-1, self.c * self.c)
        xxTSigmax = xxTSigmax.reshape(-1, self.c, self.c)
        Ym = torch.nan_to_num(Y - m)
        xbarym = torch.einsum("ni,nk,ni->ik", O_, xbarn, Ym)
        w = torch.linalg.solve(xxTSigmax, xbarym)
        wSw = torch.einsum("ik,nkl,il->ni", w, Sigma_xn, w)
        vy = torch.mean(torch.square(torch.addmm(Ym, xbarn, w.T, alpha=-1)[O]) + wSw[O])

        return m, w, vy

    def fit_transform(self, Y, O=None, max_iter=100, show_progress=True, atol=1e-3):
        if O is None:
            O = torch.isfinite(Y)
        self.initialize_from(Y, O)
        pbar = trange(max_iter)
        for i in pbar:
            xbarn, Sigma_xn = self(Y, O)
            m, w, vy = self.m_step(Y, O, xbarn, Sigma_xn)
            dw = torch.max(torch.abs(w - self.w))
            # dw = torch.mean(torch.square(w - self.w))
            pbar.set_description(f"dw={dw.numpy(force=True):0.5f}")
            if torch.isnan(dw).any() or dw < atol:
                break
            self.m.copy_(m)
            self.w.copy_(w)
            if self.learn_vy:
                self.vy.copy_(vy)
        return xbarn, Sigma_xn


class VBPCA(torch.nn.Module):
    def __init__(self, d, c, active_indices=None, max_feature_ind=None, learn_vy=True):
        super().__init__()
        self.d = d
        self.c = c
        self.mbar = nn.Parameter(torch.zeros(d), requires_grad=False)
        self.mtilde = nn.Parameter(torch.ones(d), requires_grad=False)
        self.wbar = nn.Parameter(torch.zeros(d, c), requires_grad=False)
        eyes_dcc = torch.eye(c)[None].repeat(d, 1, 1)
        self.Sigmaw = nn.Parameter(eyes_dcc, requires_grad=False)
        self.vy = nn.Parameter(torch.ones(()), requires_grad=False)
        self.learn_vy = learn_vy
        self.vm = nn.Parameter(torch.ones(()), requires_grad=False)
        self.vwk = nn.Parameter(torch.ones(c), requires_grad=False)
        self.register_buffer("Ic", torch.eye(c))
        self.active_indices = active_indices
        self.max_feature_ind = max_feature_ind
        if active_indices is not None:
            self.active_indices = torch.tensor(active_indices)

    def initialize_from(self, Y, O):
        with torch.no_grad():
            m = torch.nan_to_num(torch.nanmean(Y, dim=0))
            self.mbar.copy_(m)
            Yc = torch.nan_to_num(Y - m)
            u, s, vh = torch.linalg.svd(Yc, full_matrices=False)
            vh = vh[: self.c]
            s = s[: self.c]
            self.wbar.copy_(vh.T)

    def forward(self, Y, O=None, Y_indices=None):
        if O is None:
            O = torch.isfinite(Y)
        O_ = O.to(self.wbar)
        Y = torch.nan_to_num(Y)
        if Y_indices is not None:
            # re-order O, Y s.t. Y_indices matches active_indices
            rel_inds, not_found = searchfor(self.active_indices, Y_indices, invalid=0)
            Y = torch.take_along_dim(Y, rel_inds, dim=1)
            O_ = torch.take_along_dim(O_, rel_inds, dim=1)
            O_[not_found] = 0
            pass

        OwwSw = torch.baddbmm(self.Sigmaw, self.wbar[:, :, None], self.wbar[:, None, :])
        OwwSw = (O_ @ OwwSw.reshape(self.d, self.c * self.c)).reshape(
            O.shape[0], self.c, self.c
        )
        Sigma_xn = self.vy * torch.linalg.inv(self.vy * self.Ic[None] + OwwSw)
        xbarn = torch.einsum(
            "nlk,ni,ik,ni->nl", Sigma_xn / self.vy, O_, self.wbar, Y - self.mbar
        )

        return xbarn, Sigma_xn

    def m_step(self, Y, O, xbarn, Sigma_xn):
        O_ = O.to(self.wbar)
        Oicard = O_.sum(0)

        # -- M
        vm_OvvO = torch.where(
            Oicard > 0,
            self.vm / (Oicard * (self.vm + self.vy / Oicard)),
            self.vm / self.vy,
        )
        # mbar
        mbar = vm_OvvO * torch.nansum(
            torch.addmm(Y, xbarn, self.wbar.T, alpha=-1), axis=0
        )
        # mtilde
        mtilde = self.vy * vm_OvvO

        # -- W
        # xxTSigmax = xbarn[:, :, None] * xbarn[:, None, :] + Sigma_xn
        xxTSigmax = torch.baddbmm(Sigma_xn, xbarn[:, :, None], xbarn[:, None, :])
        xxTSigmax = (O_.T @ xxTSigmax.reshape(-1, self.c * self.c)).reshape(
            -1, self.c, self.c
        )
        # Sigmaw
        Sigmaw = self.vy * torch.linalg.inv(self.vy * torch.diag(self.vwk) + xxTSigmax)
        Ym = torch.nan_to_num(Y - mbar)
        xbarym = torch.einsum("ni,nk,ni->ik", O_, xbarn, Ym)
        # wbar
        wbar = torch.bmm(Sigmaw, xbarym[:, :, None])[:, :, 0] / self.vy

        # -- vs
        wSxw = torch.einsum("ik,nkl,il->ni", wbar, Sigma_xn, wbar)
        xSwx = torch.einsum("nk,ikl,nl->ni", xbarn, Sigmaw, xbarn)
        trSS = torch.einsum("nkl,ikl->ni", Sigma_xn, Sigmaw)
        # vy
        vy = torch.mean(
            torch.square(torch.addmm(Ym, xbarn, wbar.T, alpha=-1)[O])
            + wSxw[O]
            + xSwx[O]
            + trSS[O]
        )
        # vwk
        vwk = torch.mean(
            torch.square(wbar) + torch.diagonal(Sigmaw, dim1=-2, dim2=-1), dim=0
        )
        # vm
        vm = torch.mean(torch.square(mbar) + mtilde, dim=0)

        return mbar, mtilde, wbar, Sigmaw, vy, vwk, vm

    def fit_transform(self, Y, O=None, max_iter=100, show_progress=True, atol=1e-3):
        if O is None:
            O = torch.isfinite(Y)
        self.initialize_from(Y, O)
        pbar = trange(max_iter)
        for i in pbar:
            xbarn, Sigma_xn = self(Y, O)
            mbar, mtilde, wbar, Sigmaw, vy, vwk, vm = self.m_step(Y, O, xbarn, Sigma_xn)
            dw = torch.max(torch.abs(wbar - self.wbar))
            # dw = torch.mean(torch.square(w - self.w))
            pbar.set_description(f"dw={dw.numpy(force=True):0.5f}")
            if torch.isnan(dw).any() or dw < atol:
                break
            self.mbar.copy_(mbar)
            self.mtilde.copy_(mtilde)
            self.wbar.copy_(wbar)
            self.Sigmaw.copy_(Sigmaw)
            if self.learn_vy:
                self.vy.copy_(vy)
            self.vwk.copy_(vwk)
            self.vm.copy_(vm)
        return xbarn, Sigma_xn

    def predictive_dists(self, Ynew, Onew=None, Y_indices_new=None):
        if Onew is None:
            Onew = torch.isfinite(Ynew)

        # posteriors for xs, holding everything fixed
        # not sure how kosher this is.
        xbarnew, Sigma_xnew = self(Ynew, Onew, Y_indices_new)

        if Y_indices_new is None:
            # channels not moving around case

            # posterior predictive means
            Ynewhat = xbarnew @ self.wbar.T + self.mbar

            # posterior predictive variance
            wSxw = torch.einsum("ik,nkl,jl->nij", self.wbar, Sigma_xnew, self.wbar)
            SwSx = torch.einsum("nkl,ikl->ni", Sigma_xnew, self.Sigmaw)
            Sigma_Ynew = wSxw
            diag = self.vy + SwSx + self.mtilde
            Sigma_Ynew.diagonal(dim1=-2, dim2=-1).add_(diag)

            Lambda_xnew = torch.linalg.inv(Sigma_xnew)
            diag_inv = torch.reciprocal(diag)
            Lambda_Ynew = torch.diag_embed(diag_inv)
            inner_prod = torch.einsum("ik,ni,il->nkl", self.wbar, diag_inv, self.wbar)
            inner = Lambda_xnew + inner_prod
            Lambda_Ynew.sub_(torch.einsum("ni,ik,nkl,jl,nj", diag_inv, self.wbar, inner, self.wbar, diag_inv))

            log_cov_det = torch.logdet(Lambda_xnew) + torch.logdet(inner_prod) + torch.log(diag).sum(1)
            return Onew, Ynewhat, Sigma_Ynew, Lambda_Ynew, log_cov_det

        # reverse lookup
        my_chans_to_new, my_chans_found = searchfor(Y_indices_new, self.active_indices, invalid=self.d)
        my_chans_to_new[Y_indices_new == self.max_feature_ind] = self.d + 1
        Nnew = len(Ynew)
        mbar_pad = F.pad(self.mbar, (0, 2))
        mbar = torch.take_along_dim(
            mbar_pad[None].broadcast_to(Nnew, self.d),
            my_chans_to_new,
            dim=1,
        )
        wbar_pad = F.pad(self.wbar, (0, 2, 0, 0))
        wbar = torch.take_along_dim(
            wbar_pad[None].broadcast_to(Nnew, self.d, self.c),
            my_chans_to_new,
            dim=1,
        )
        Sigmaw_pad = F.pad(self.Sigmaw, (0, 2, 0, 0, 0, 0))
        Sigmaw_pad[self.d] = self.vwk * torch.eye(self.c)
        Sigmaw_pad[self.d + 1] = torch.eye(self.c)
        Sigmaw = torch.take_along_dim(
            Sigmaw_pad[None].broadcast_to(Nnew, self.d, self.c, self.c),
            my_chans_to_new,
            dim=1,
        )
        mtilde_pad = F.pad(self.mtilde, (0, 2))
        mtilde_pad[self.d] = self.vm
        mtilde_pad[self.d + 1] = 1.0
        mtilde = torch.take_along_dim(
            mtilde_pad[None].broadcast_to(Nnew, self.d),
            my_chans_to_new,
            dim=1,
        )

        Ynewhat = torch.baddbmm(mbar, xbarnew, wbar.mT)
        wSxw = torch.einsum("nik,nkl,njl->nij", wbar, Sigma_xnew, wbar)
        SwSx = torch.einsum("nkl,nikl->ni", Sigma_xnew, Sigmaw)
        Sigma_Ynew = wSxw
        diag = self.vy + SwSx + mtilde
        Sigma_Ynew.diagonal(dim1=-2, dim2=-1).add_(diag)

        Lambda_xnew = torch.linalg.inv(Sigma_xnew)
        diag_inv = torch.reciprocal(diag)
        Lambda_Ynew = torch.diag_embed(diag_inv)
        inner_prod = torch.einsum("nik,ni,nil->nkl", wbar, diag_inv, wbar)
        inner = Lambda_xnew + inner_prod
        Lambda_Ynew.sub_(torch.einsum("ni,nik,nkl,njl,nj", diag_inv, wbar, inner, wbar, diag_inv))

        log_cov_det = torch.logdet(Lambda_xnew) + torch.logdet(inner_prod) + torch.log(diag).sum(1)

        return Onew, Ynewhat, Sigma_Ynew, Lambda_Ynew, log_cov_det

        # dists = torch.distributions.MultivariateNormal(
        #     Ynewhat, covariance_matrix=Sigma_Ynew
        # )
        # return dists

    def predict(self, Ynew, Onew=None, Y_indices_new=None):
        Onew, Ynewhat, Sigma_Ynew, Lambda_Ynew, log_cov_det = self.predictive_dists(Ynew, Onew, Y_indices_new)
        dY = torch.nan_to_num(Ynew - Ynewhat)
        mahals = torch.einsum("ni,nij,nj->n", dY, Lambda_Ynew, dY)
        nobs = Onew.sum(1)
        logliks = -0.5 * (log_cov_det + nobs * torch.log(2 * torch.pi)) - 0.5 * mahals
        chi_qs = torch.distributions.Chi2(nobs).cdf(mahals)
        return mahals, logliks, chi_qs, nobs


def searchfor(indices, new_indices, invalid=0):
    if new_indices.ndim == 1:
        new_indices = torch.broadcast_to(new_indices[None], indices.shape).contiguous()
    assert new_indices.shape[-1] == indices.shape[-1]
    ix = torch.searchsorted(indices, new_indices)
    ix[ix == new_indices.shape[1]] = 0
    if indices.ndim == 1:
        not_found = indices[ix] != new_indices
    else:
        not_found = torch.take_along_dim(indices, ix, dim=1) != new_indices
    return torch.where(not_found, invalid, ix), not_found
