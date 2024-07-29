from tqdm.auto import trange

import torch
import torch.nn.functional as F
from torch import nn
from dartsort.util import spiketorch


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
    def __init__(self, d, c, active_indices=None, max_feature_ind=None, learn_vy=True, principal_transform=True):
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
        self.max_feature_ind = max_feature_ind
        self.principal_transform = principal_transform
        self.register_buffer("log2pi", torch.log(torch.tensor(2 * torch.pi)))
        if active_indices is not None:
            self.register_buffer("active_indices", torch.as_tensor(active_indices))
        else:
            self.active_indices = active_indices

    def initialize_from(self, Y, O):
        with torch.no_grad():
            m = torch.nan_to_num(torch.nanmean(Y, dim=0))
            self.mbar.copy_(m)
            Yc = torch.nan_to_num(Y - m)
            u, s, vh = torch.linalg.svd(Yc, full_matrices=False)
            vh = vh[: self.c]
            s = s[: self.c]
            self.wbar.copy_(vh.T)

    def forward(self, Y, O=None, Y_indices=None, batch_size=256, return_O=False):
        if O is None:
            O = torch.isfinite(Y)
        O_ = O.to(self.wbar)
        del O
        Y = torch.nan_to_num(Y)
        if Y_indices is not None:
            # re-order O, Y s.t. Y_indices matches active_indices
            
            # get my active indices in y's index space
            # when going forward, we want indices into Y which conform it to my index set
            rel_inds, not_found = searchfor(Y_indices, self.active_indices, invalid=0, shape="right")
            # Ys = torch.zeros((len(Y), self.d + 1), dtype=Y.dtype, device=Y.device)
            # Os = torch.zeros((len(Y), self.d + 1), dtype=O_.dtype, device=O_.device)
            # Ys.scatter_(dim=1, index=rel_inds, src=Y)
            # Os.scatter_(dim=1, index=rel_inds, src=O_)
            # Y = Ys[:, :self.d]
            # O_ = Os[:, :self.d]
            Y = torch.take_along_dim(Y, rel_inds, dim=1)
            O_ = torch.take_along_dim(O_, rel_inds, dim=1)
            O_[not_found] = 0
            pass

        xbarn = torch.empty(
            (len(Y), self.c), dtype=Y.dtype, layout=Y.layout, device=Y.device
        )
        Sigma_xn = torch.empty(
            (len(Y), self.c, self.c), dtype=Y.dtype, layout=Y.layout, device=Y.device
        )

        OwwSw0 = torch.baddbmm(
            self.Sigmaw, self.wbar[:, :, None], self.wbar[:, None, :]
        )
        OwwSw0 = OwwSw0.reshape(self.d, self.c * self.c)
        for batch_start in range(0, len(Y), batch_size):
            batch_end = min(batch_start + batch_size, len(Y))
            bsl = slice(batch_start, batch_end)
            bn = batch_end - batch_start
            OwwSw = (O_[bsl] @ OwwSw0).reshape(
                bn, self.c, self.c
            )
            bS = self.vy * torch.linalg.inv(self.vy * self.Ic[None] + OwwSw)
            Sigma_xn[bsl] = bS
            xbarn[bsl] = torch.einsum(
                "nlk,ni,ik,ni->nl", bS / self.vy, O_[bsl], self.wbar, Y[bsl] - self.mbar
            )

        if return_O:  
            return xbarn, Sigma_xn, O_
        return xbarn, Sigma_xn
    
    def apply_principal_transform(self, x, Sigmax):
        mu = x.mean(0)
        x = x - mu
        

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
            Lambda_Ynew.sub_(
                torch.einsum(
                    "ni,ik,nkl,jl,nj", diag_inv, self.wbar, inner, self.wbar, diag_inv
                )
            )

            log_prec_det = (
                torch.logdet(Lambda_xnew)
                + torch.logdet(inner_prod)
                + torch.log(diag).sum(1)
            )
            return Onew, Ynewhat, Sigma_Ynew, Lambda_Ynew, log_prec_det

        # reverse lookup
        # when coming back from my index set into Y's, we want
        # indices which conform my index set to Y indices
        my_chans_to_new, my_chans_found = searchfor(
            Y_indices_new, self.active_indices, invalid=0, shape="right"
        )
        found = my_chans_found.to(self.mbar)
        Nnew = len(Ynew)

        mbar_fill = self.mbar[None].broadcast_to(Nnew, *self.mbar.shape)
        mbar = torch.zeros_like(Ynew)
        mbar.scatter_(src=mbar_fill, dim=1, index=my_chans_to_new)
        
        # wbar_fill = F.pad(self.wbar, (0, 0, 0, 2))
        # wbar_fill = wbar_fill[None].broadcast_to(Nnew, self.d + 2, self.c)
        wbar_fill = self.wbar[None] * found[..., None]
        wbar = torch.zeros((*Ynew.shape, self.c), dtype=Ynew.dtype, device=Ynew.device)
        Nix = torch.arange(Nnew)
        cix = torch.arange(self.c)
        wbar[Nix[:, None], my_chans_to_new[:, :]].copy_(wbar_fill)
        
        Sigmaw_fill = self.Sigmaw[None] * found[..., None, None]
        Sigmaw = torch.zeros((*Ynew.shape, self.c, self.c), dtype=Ynew.dtype, device=Ynew.device)
        # fill vwks for observed chans where model does not overlap
        # fill eyes for nonexistent chans so that determinants dont get messed up
        Sigmaw.diagonal(dim1=-2, dim2=-1).copy_(torch.where(Onew[..., None], self.vwk, 1.0))
        # add in so that the zeros don't overwrite ^ where there are no observations
        Sigmaw[Nix[:, None], my_chans_to_new[:, :]].add_(Sigmaw_fill)
        
        mtilde_fill = self.mtilde[None] * found
        mtilde = torch.zeros_like(Ynew)
        mtilde[Onew] = self.vm
        mtilde[Nix[:, None], my_chans_to_new[:, :]].add_(mtilde_fill)
        mtilde[torch.logical_not(Onew)] = 1.0

        Ynewhat = torch.baddbmm(mbar[:, None], xbarnew[:, None], wbar.mT)[:, 0]
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
        Lambda_Ynew.sub_(
            torch.einsum("ni,nik,nkl,njl,nj", diag_inv, wbar, inner, wbar, diag_inv)
        )
        
        # print(f"{Lambda_xnew.shape=}")
        # print(f"{torch.logdet(Lambda_xnew)=}")
        # print(f"{inner_prod.shape=}")
        # print(f"{torch.logdet(inner_prod)=}")
        # inner_prod_det = torch.einsum("nik,ni,nil->ni", wbar, diag_inv, wbar)
        ipsvs = torch.linalg.svdvals(inner_prod)
        # print(f"{ipsvs=}")

        log_prec_det = (
            torch.logdet(Lambda_xnew)
            + torch.log(torch.where(ipsvs > 0, ipsvs, 1.0)).sum()
            + torch.log(diag).sum(1)
        )

        return Onew, Ynewhat, Sigma_Ynew, Lambda_Ynew, log_prec_det
    
    
    def predictive_dists_lowrank(self, Ynew, Onew=None, Y_indices_new=None):
        if Onew is None:
            Onew = torch.isfinite(Ynew)

        # posteriors for xs, holding everything fixed
        # not sure how kosher this is.
        xbarnew, Sigma_xnew = self(Ynew, Onew, Y_indices_new)
        assert Y_indices_new is not None

        # reverse lookup
        # when coming back from my index set into Y's, we want
        # indices which conform my index set to Y indices
        my_chans_to_new, my_chans_found = searchfor(
            Y_indices_new, self.active_indices, invalid=0, shape="right"
        )
        found = my_chans_found.to(self.mbar)
        Nnew = len(Ynew)

        mbar_fill = self.mbar[None].broadcast_to(Nnew, *self.mbar.shape)
        mbar = torch.zeros_like(Ynew)
        mbar.scatter_(src=mbar_fill, dim=1, index=my_chans_to_new)
        
        # wbar_fill = F.pad(self.wbar, (0, 0, 0, 2))
        # wbar_fill = wbar_fill[None].broadcast_to(Nnew, self.d + 2, self.c)
        wbar_fill = self.wbar[None] * found[..., None]
        wbar = torch.zeros((*Ynew.shape, self.c), dtype=Ynew.dtype, device=Ynew.device)
        Nix = torch.arange(Nnew)
        cix = torch.arange(self.c)
        wbar[Nix[:, None], my_chans_to_new[:, :]].copy_(wbar_fill)
        
        Sigmaw_fill = self.Sigmaw[None] * found[..., None, None]
        Sigmaw = torch.zeros((*Ynew.shape, self.c, self.c), dtype=Ynew.dtype, device=Ynew.device)
        # fill vwks for observed chans where model does not overlap
        # fill eyes for nonexistent chans so that determinants dont get messed up
        Sigmaw.diagonal(dim1=-2, dim2=-1).copy_(torch.where(Onew[..., None], self.vwk, 1.0))
        # add in so that the zeros don't overwrite ^ where there are no observations
        Sigmaw[Nix[:, None], my_chans_to_new[:, :]].add_(Sigmaw_fill)
        
        mtilde_fill = self.mtilde[None] * found
        mtilde = torch.zeros_like(Ynew)
        mtilde[Onew] = self.vm
        mtilde[Nix[:, None], my_chans_to_new[:, :]].add_(mtilde_fill)
        mtilde[torch.logical_not(Onew)] = 1.0

        Ynewhat = torch.baddbmm(mbar[:, None], xbarnew[:, None], wbar.mT)[:, 0]
        # wSxw = torch.einsum("nik,nkl,njl->nij", wbar, Sigma_xnew, wbar)
        SwSx = torch.einsum("nkl,nikl->ni", Sigma_xnew, Sigmaw)
        # Sigma_Ynew = wSxw
        # Sigma_Ynew.diagonal(dim1=-2, dim2=-1).add_(diag)

        Lambda_xnew = torch.linalg.inv(Sigma_xnew)
        diag = self.vy + SwSx + mtilde
        Lambda_diag = torch.reciprocal(diag)
        
        inner_prod = torch.einsum("nik,ni,nil->nkl", wbar, Lambda_diag, wbar)
        inner = Lambda_xnew + inner_prod
        Lambda_left = torch.einsum("ni,nik,nkl->nil", Lambda_diag, wbar, inner)
        Lambda_right = torch.einsum("nik,ni->nki", wbar, Lambda_diag)
        
        ipsvs = torch.linalg.svdvals(inner)

        log_cov_det = (
            torch.logdet(Sigma_xnew)
            + torch.log(torch.where(ipsvs > 0, ipsvs, 1.0)).sum()
            + torch.log(diag).sum(1)
        )

        return Onew, Ynewhat, Lambda_left, Lambda_diag, Lambda_right, log_cov_det 

        # dists = torch.distributions.MultivariateNormal(
        #     Ynewhat, covariance_matrix=Sigma_Ynew
        # )
        # return dists

    def predict(self, Ynew, Onew=None, Y_indices_new=None, return_overlaps=False):
        # Onew, Ynewhat, Sigma_Ynew, Lambda_Ynew, log_prec_det = self.predictive_dists(
        #     Ynew, Onew, Y_indices_new
        # )
        if return_overlaps:
            old_counts = Onew.sum(1)
        Onew, Ynewhat, Lambda_left, Lambda_diag, Lambda_right, log_cov_det = self.predictive_dists_lowrank(
            Ynew, Onew, Y_indices_new
        )
        # print(f"{log_prec_det=}")
        dY = torch.nan_to_num(Ynew - Ynewhat)
        mahals = (
            torch.einsum("ni,ni,ni->n", dY, Lambda_diag, dY)
            - torch.einsum("ni,nik,nkj,nj->n", dY, Lambda_left, Lambda_right, dY)
        )
        nobs = Onew.sum(1)
        logliks = - 0.5 * (log_cov_det + nobs * self.log2pi) - 0.5 * mahals
        chi_qs = torch.distributions.Chi2(nobs).cdf(mahals)
        if return_overlaps:
            return mahals, logliks, chi_qs, nobs, nobs / old_counts
            
        return mahals, logliks, chi_qs, nobs


def searchfor(indices, new_indices, invalid=0, shape="min"):
    """
    searchsorted(indices, new_indices) returns indices of indices.
    """
    if new_indices.ndim == 1:
        new_indices = torch.broadcast_to(new_indices[None], (indices.shape[0], -1)).contiguous()
    ix = torch.searchsorted(indices, new_indices)
    ix[ix == indices.shape[-1]] = 0
    if indices.ndim == 1:
        not_found = indices[ix] != new_indices
    else:
        not_found = torch.take_along_dim(indices, ix, dim=1) != new_indices
    ixs = torch.where(not_found, invalid, ix)
    if shape == "min" and indices.shape[-1] != new_indices.shape[-1]:
        min_size = min(indices.shape[-1], new_indices.shape[-1])
        keep_me = torch.sort(not_found.to(torch.int8), stable=True)
        assert torch.take_along_dim(not_found, keep_me.indices[..., min_size:], dim=1).all()
        keep_me = keep_me.indices[..., :min_size]
        ixs = torch.take_along_dim(ixs, keep_me, dim=1)
        not_found = torch.take_along_dim(not_found, keep_me, dim=1)
        return ixs, not_found, keep_me
        
    return ixs, not_found
