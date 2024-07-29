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
import torch
from torch import vmap

# %%
from torch.func import grad, hessian, grad_and_value

# %%

# %%
from torch.optim.lbfgs import _cubic_interpolate, _strong_wolfe

# %%
import numpy as np


# %%
def norm(g):
    return torch.linalg.norm(g)


# %%
g_and_v = grad_and_value(norm)

# %%
norm(torch.ones(3))

# %%
bgv = grad_and_value(vmap(norm))

# %%
bgv(torch.ones(10, 3))

# %% [markdown] tags=[]
# # make and test batched cubic interpolator

# %%
do_print = False
def dprint(*args, **kwargs):
    if do_print: print(*args, **kwargs)


# %%
def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    dprint(f"/// {xmin_bound=} {xmax_bound=} {d1=} {d2_square=} {g1.numpy()=} {g2.numpy()=}")
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.


# %%
def batched_cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    """Argmin of cubic interpolation of a scalar function (batched)
    """
    # bounds logic -- make sure x1 <= x2
    correct_order = x1 <= x2
    if bounds is None:
        xmin_bound = torch.where(correct_order, x1, x2)
        xmax_bound = torch.where(correct_order, x2, x1)
    else:
        xmin_bound, xmax_bound = bounds

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3.0 * (f1 - f2) / (x1 - x2)
    d2_square = d1.square() - g1 * g2
    d2 = d2_square.abs().sqrt()
    dprint(f"/// {xmin_bound=} {xmax_bound=} {d1=} {d2_square=} {g1.numpy()=} {g2.numpy()=}")
    min_pos = torch.where(
        correct_order,
        x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2)),
        x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2)),
    )
    min_pos = torch.where(
        d2_square >= 0,
        min_pos.clamp_(xmin_bound, xmax_bound),
        (xmin_bound + xmax_bound) / 2.,
    )
    return min_pos


# %%
# x1s = torch.arange(30.).reshape(10, 3)
x1s = torch.normal(torch.zeros(10, 3), torch.ones(10, 3))
f1s = vmap(norm)(x1s)
g1s = vmap(grad(norm))(x1s)
d1s = -g1s
gtd1s = vmap(torch.dot)(g1s, d1s)

# %%
t1s = torch.zeros(10)
t2s = torch.normal(t1s, torch.ones(10))

# %%
x2s = x1s + t2s[:, None] * d1s
f2s = vmap(norm)(x2s)
g2s = vmap(grad(norm))(x2s)
d2s = -g2s
gtd2s = vmap(torch.dot)(g2s, d2s)

# %%
batched_cubic_interpolate(t1s, f1s, gtd1s, t2s, f2s, gtd2s).numpy().tolist()

# %%
[_cubic_interpolate(t1, f1, gtd1, t2, f2, gtd2) for t1, f1, gtd1, t2, f2, gtd2 in zip(t1s, f1s, gtd1s, t2s, f2s, gtd2s)]

# %% [markdown]
# # full line search

# %%
batched_g_and_v = vmap(g_and_v)


# %%
def obj_func(x, t, d):
    grad, val = g_and_v(x + t * d)
    return val, grad


# %%
def batched_strong_wolfe(batched_grad_and_obj,
                  x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):
    """Documented and batched version of torch's strong Wolfe line search routine
    
    torch's original lua is much better documented:
    https://github.com/torch/optim/blob/master/lswolfe.lua
    
    acknowledgements to the nice library and useful reference
    @rfeinman/pytorch-minimize/blob/master/torchmin/line_search.py
    
    Arguments
    ---------
    batched_grad_and_obj : function
    x : tensor
        Current center / starting point
    t : float
        Initial step size. In torch LBFGS they set this to `lr` (i.e. 1 for Newton methods)
        except on the first iterate, when they use `lr*min(1,l1norm(grad))`.
    d : tensor
        Descent direction
    f : initial objective function value
    g : gradient at x
    gtd : tensor
        Directional derivative at starting point (grad @ d)
    c1, c2, tolerance_change : floats
        parameters: sufficient decrease, curvature, minimum allowed step length
    max_ls : int
        Allowed number of iterations
    
    Returns
    -------
    f : function value at x+t*d
    g : gradient value at x+t*d
    t : step length
    ls_nevals : number of objective function evals
    """
    B, in_dim = x.shape
    print("B", t.shape)
    t = t.clone()
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    # via https://github.com/pytorch/pytorch/blob/main/torch/optim/lbfgs.py
    
    # to make this work in batched setting, we do two structural changes:
    # - none of these length-1 brackets. bracket is always two numbers.
    # - "active set" idea. breaks after condition checks are implemented 
    #   by deactivating some batch members.

    d_norm, _ = d.abs().max(dim=1)
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    g_new, f_new = batched_grad_and_obj(x + t[:, None] * d)
    g_new = g_new.clone(memory_format=torch.contiguous_format)
    ls_func_evals = torch.ones(B, dtype=torch.int, device=t.device)
    gtd_new = (g_new * d).sum(dim=1)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev = torch.zeros_like(t)
    f_prev = f
    g_prev = g
    gtd_prev = gtd

    done = torch.zeros(B, dtype=torch.bool, device=t.device)
    active_backtrack = torch.arange(B, device=t.device)
    ls_iter = torch.zeros(B, dtype=torch.int, device=t.device)
    
    # allocate brackets
    bracket_l = torch.empty(B, dtype=t.dtype, device=t.device)
    bracket_u = torch.empty(B, dtype=t.dtype, device=t.device)
    bracket_fl = torch.empty(B, dtype=t.dtype, device=t.device)
    bracket_fu = torch.empty(B, dtype=t.dtype, device=t.device)
    bracket_gl = torch.empty((B, in_dim), dtype=t.dtype, device=t.device)
    bracket_gu = torch.empty((B, in_dim), dtype=t.dtype, device=t.device)
    bracket_gtdl = torch.empty(B, dtype=t.dtype, device=t.device)
    bracket_gtdu = torch.empty(B, dtype=t.dtype, device=t.device)
    
    # cur active set
    fa = f
    gtda = gtd
    ta = t
    t_preva = t_prev
    f_preva = f_prev
    f_newa = f_new
    g_preva = g_prev
    g_newa = g_new
    gtd_preva = gtd_prev
    gtd_newa = gtd_new
    
    while (ls_iter[active_backtrack] < max_ls).any() and active_backtrack.numel():
        print(f"bt {ls_iter[active_backtrack]=} {active_backtrack=} {t=} {f_new=} {g_new=}")
        # check conditions
        cond1 = (f_newa > (fa + c1 * ta * gtda)) | ((ls_iter[active_backtrack] > 1) & (f_newa >= f_preva))
        cond2 = gtd_newa.abs() <= -c2 * gtda
        cond3 = gtd_newa >= 0
        conda = cond1 | cond2 | cond3
        
        # -- set brackets
        # conds 1 + 3
        conds = cond1 | cond3
        if conds.any():
            oinds = active_backtrack[conds]
            bracket_l[oinds] = t_preva[conds]
            bracket_u[oinds] = ta[conds]
            bracket_fl[oinds] = f_preva[conds]
            bracket_fu[oinds] = f_newa[conds]
            bracket_gl[oinds] = g_preva[conds]
            bracket_gu[oinds] = g_newa[conds]
            bracket_gtdl[oinds] = gtd_preva[conds]
            bracket_gtdu[oinds] = gtd_newa[conds]
        # cond 2
        if cond2.any():
            oinds = active_backtrack[cond2]
            done[oinds] = True
            bracket_l[oinds] = bracket_u[oinds] = ta[cond2]
            bracket_fl[oinds] = bracket_fu[oinds] = f_newa[cond2]
            bracket_gl[oinds] = bracket_gu[oinds] = g_newa[cond2]
        
        # update active set
        unconda = ~conda
        active_backtrack = active_backtrack[unconda]
        if not active_backtrack.numel():
            break
        fa = fa[unconda]
        gtda = gtda[unconda]
        f_newa = f_newa[unconda]
        g_newa = g_newa[unconda]
        gtd_newa = gtd_newa[unconda]
        
        # interpolate to find new argmin
        ta = ta[unconda]
        min_step = ta + 0.01 * (ta - t_preva[unconda])
        max_step = ta * 10
        tmpa = ta.clone()
        print(f"before interp {ls_iter[active_backtrack]=} {active_backtrack=} {t=} {f_new=} {g_new=}")
        print(f"{t_preva[unconda]=} {f_preva[unconda]=} {gtd_preva[unconda]=} {ta=} {f_newa=} {gtd_newa=} {min_step=} {max_step=}")
        ta = t[active_backtrack] = batched_cubic_interpolate(
            t_preva[unconda],
            f_preva[unconda],
            gtd_preva[unconda],
            ta,
            f_newa,
            gtd_newa,
            bounds=(min_step, max_step))
        print(f"after interp {ls_iter[active_backtrack]=} {active_backtrack=} {t=} {f_new=} {g_new=}")

        # next step
        t_preva = tmpa
        f_preva = f_newa.clone(memory_format=torch.contiguous_format)
        g_preva = g_newa.clone(memory_format=torch.contiguous_format)
        gtd_preva = gtd_newa.clone(memory_format=torch.contiguous_format)
        g_newa, f_newa = g_new[active_backtrack], f_new[active_backtrack] = batched_grad_and_obj(
            x[active_backtrack] + ta[:, None] * d[active_backtrack]
        )
        ls_func_evals[active_backtrack] += 1
        gtd_newa = gtd_new[active_backtrack] = (g_new[active_backtrack] * d[active_backtrack]).sum(1)
        ls_iter[active_backtrack] += 1
        print(f"new_eval {f_newa=}")
    print(f"post bt {t=} {f_new=} {g_new=}")
    
    # reached max number of iterations?
    reached_max = ls_iter == max_ls
    if reached_max.any():
        bracket_l[reached_max] = 0
        bracket_u[reached_max] = t[reached_max]
        bracket_fl[reached_max] = f[reached_max]
        bracket_fu[reached_max] = f_new[reached_max]
        bracket_gl[reached_max] = g[reached_max]
        bracket_gu[reached_max] = g_new[reached_max]
        # these aren't set here for some reason yet unclear
        # bracket_gtdl[reached_max] = gtd_prev[reached_max]
        # bracket_gtdu[reached_max] = gtd_new[reached_max]
    
    # stack the brackets so that the high_pos / low_pos logic works
    bracket = torch.column_stack((bracket_l, bracket_u))
    bracket_f = torch.column_stack((bracket_fl, bracket_fu))
    bracket_g = torch.column_stack((bracket_gl[:, None], bracket_gu[:, None]))
    bracket_gtd = torch.column_stack((bracket_gtdl, bracket_gtdu))
    gtda = gtd
    # just making sure we don't touch these again
    del bracket_l, bracket_u, bracket_fl, bracket_fu, bracket_gl, bracket_gu, bracket_gtdl, bracket_gtdu

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = torch.zeros(B, dtype=torch.bool, device=t.device)
    active_zoom = torch.arange(B, device=t.device)
    # find high and low points in bracket
    high_pos_ac = high_pos = (torch.diff(bracket_f, dim=1)[:, 0] >= 0).to(torch.int)
    low_pos_ac = low_pos = 1 - high_pos
    print(f"{low_pos=} {high_pos=} {bracket=} {bracket_f=} {bracket_g=} {bracket_gtd=} {t=}")

    while not done.all() and (ls_iter[active_zoom] < max_ls).any():
        print(f"{ls_iter=} {active_zoom=} {high_pos=}")
        # line-search bracket is so small
        dbracket = torch.diff(bracket[active_zoom], dim=1).abs()[:, 0]
        smallbracket = dbracket * d_norm[active_zoom] < tolerance_change
        smalliter = ls_iter[active_zoom] < max_ls
        newmask = smalliter & ~smallbracket
        print(f"{smallbracket=}")
        active_zoom = active_zoom[newmask]
        if not active_zoom.numel():
            print("small")
            break
        
        bracket_la, bracket_ua = bracket[active_zoom].T
        high_pos_ac = high_pos_ac[newmask]
        low_pos_ac = low_pos_ac[newmask]
        gtda = gtda[newmask]
        
        # compute new trial value
        ta = t[active_zoom] = batched_cubic_interpolate(
            bracket_la, bracket_f[active_zoom, 0], bracket_gtd[active_zoom, 0],
            bracket_ua, bracket_f[active_zoom, 1], bracket_gtd[active_zoom, 1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        bracket_max = torch.max(bracket_la, bracket_ua)
        bracket_min = torch.min(bracket_la, bracket_ua)
        eps = 0.1 * (bracket_max - bracket_min)
        dt_max = bracket_max - ta
        dt_min = ta - bracket_min
        
        print(f"before insuf {ta=} {f_newa=}")
        cond1 = torch.min(dt_max, dt_min) < eps
        print("A1", cond1)
        cond2 = insuf_progress[active_zoom] | (ta >= bracket_max) | (ta <= bracket_min)
        print("A2", cond1 & cond2)
        cond3 = dt_max.abs() < dt_min.abs()
        conda = cond1 & cond2 & cond3
        print("A", conda)
        condb = cond1 & cond2 & ~cond3
        print("B", condb)
        t[active_zoom[conda]] = bracket_max[conda] - eps[conda]
        t[active_zoom[condb]] = bracket_min[condb] + eps[condb]
        insuf_progress[active_zoom] = cond1 & ~cond2
        ta = t[active_zoom]
        print(f"after insuf {ta=} {f_newa=}")

        # Evaluate new point
        g_newa, f_newa = g_new[active_zoom], f_new[active_zoom] = batched_grad_and_obj(
            x[active_zoom] + ta[:, None] * d[active_zoom]
        )
        ls_func_evals[active_zoom] += 1
        gtd_newa = gtd_new[active_zoom] = (g_newa * d[active_zoom]).sum(dim=1)
        ls_iter[active_zoom] += 1
        print(f"after new point {ta=} {f_newa=}")
        
        # condx: Armijo condition not satisfied or not lower than lowest point
        cond1 = (f_newa > f[active_zoom] + c1 * ta * gtda) | (f_newa >= bracket_f[active_zoom, low_pos_ac])
        # condy: wolfe conditions not satisfied
        cond2 = gtd_newa.abs() <= -c2 * gtda
        # condz: old high becomes new low
        cond3 = gtd_newa * (bracket[active_zoom, high_pos_ac] - bracket[active_zoom, low_pos_ac]) >= 0
        # condw: new point becomes new low
        
        print(f"bX {f_newa=} {fa=} {ta=} {gtda=} {f_newa=} {low_pos_ac=} {bracket_f[active_zoom, low_pos_ac]=}")
        condx = cond1
        print("X", condx)
        condy = ~cond1 & cond2
        print("Y", condy)
        condz = ~cond1 & ~cond2 & cond3
        print("Z", condz)
        condw = ~cond1
        
        # condx
        indsx = active_zoom[condx]
        high_pos_aca = high_pos_ac[condx]
        bracket[indsx, high_pos_aca] = t[indsx]
        bracket_f[indsx, high_pos_aca] = f_new[indsx]
        bracket_g[indsx, high_pos_aca] = g_new[indsx]
        bracket_gtd[indsx, high_pos_aca] = gtd_new[indsx]
        high_pos_ac[condx] = high_pos[indsx] = (torch.diff(bracket_f[indsx], dim=1)[:, 0] >= 0).to(torch.int)
        low_pos_ac[condx] = low_pos[indsx] = 1 - high_pos_ac[condx]
        
        # condz
        indsz = active_zoom[condz]
        high_pos_acc = high_pos_ac[condz]
        low_pos_acc = low_pos_ac[condz]
        bracket[indsz, high_pos_acc] = bracket[indsz, low_pos_acc]
        bracket_f[indsz, high_pos_acc] = bracket_f[indsz, low_pos_acc]
        bracket_g[indsz, high_pos_acc] = bracket_g[indsz, low_pos_acc]
        bracket_gtd[indsz, high_pos_acc] = bracket_gtd[indsz, low_pos_acc]
        
        #condw
        indsw = active_zoom[condw]
        low_pos_acd = low_pos_ac[condw]
        bracket[indsw, low_pos_acd] = ta[condw]
        bracket_f[indsw, low_pos_acd] = f_newa[condw]
        bracket_g[indsw, low_pos_acd] = g_newa[condw]
        bracket_gtd[indsw, low_pos_acd] = gtd_newa[condw]
        
        # condy last bc it updates active set
        done[active_zoom[condy]] = True
        active_zoom = active_zoom[~condy]
        
        print(f"{active_zoom=} {gtda=} {condy=}")
        bracket_la, bracket_ua = bracket[active_zoom].T
        high_pos_ac = high_pos_ac[~condy]
        low_pos_ac = low_pos_ac[~condy]
        gtda = gtda[~condy]

    # return stuff
    print(f"ret {bracket=} {bracket_f=} {bracket_g=} {low_pos=}")
    t = bracket[torch.arange(B), low_pos]
    f_new = bracket_f[torch.arange(B), low_pos]
    g_new = bracket_g[torch.arange(B), low_pos]
    return f_new, g_new, t, ls_func_evals


# %%
# # x1s = torch.arange(30.).reshape(10, 3)
x1s = torch.normal(torch.zeros(10000, 3), torch.arange(30000).reshape(10000, 3))
x1s = x1s.double()
f1s = vmap(norm)(x1s)
g1s = vmap(grad(norm))(x1s)
d1s = -g1s
gtd1s = vmap(torch.dot)(g1s, d1s)
t1s = torch.ones_like(f1s)
# t1s, f1s, g1s, d1s, gtd1s

# %%
sl = slice(None)
# sl = slice(25, 28)

# %% tags=[]
f_new, g_new, t, ls_func_evals = batched_strong_wolfe(batched_g_and_v, x1s[sl].double(), t1s[sl].double(), d1s[sl].double(), f1s[sl].double(), g1s[sl].double(), gtd1s[sl].double())

# %% tags=[]
do_print = False
res = [my_strong_wolfe(obj_func, x1, t1, d1, f1, g1, gtd1) for x1, t1, d1, f1, g1, gtd1 in zip(x1s[sl], t1s[sl], d1s[sl], f1s[sl], g1s[sl], gtd1s[:])]
f_new_o, g_new_o, t_o, lsfe_o = zip(*res)

# %%
f_new, g_new, t, ls_func_evals

# %%
f_new, f_new_o

# %%
np.flatnonzero(np.equal(f_new_o, f_new) == 0)

# %%
np.equal(f_new_o, f_new).numpy().astype(float).mean()

# %%
res_b = list(zip(f_new, g_new, t, ls_func_evals))

# %%
list(zip(list(zip(*res_b))[0], list(zip(*res))[0]))

# %%
list(zip(list(zip(*res_b))[1], list(zip(*res))[1]))

# %%
list(zip(list(zip(*res_b))[2], list(zip(*res))[2]))

# %%
list(zip(list(zip(*res_b))[3], list(zip(*res))[3]))


# %%
def dprint(*args, **kwargs):
    if do_print:
        print(*args, **kwargs)

def my_strong_wolfe(obj_func, x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        dprint(f"bt {ls_iter=} {t=} {f_new=}")
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        dprint(f"before interp {ls_iter=} {t=} {f_new=}")
        dprint(f"{t_prev=} {f_prev=} {gtd_prev=} {t=} {f_new=} {gtd_new=} {min_step=} {max_step=}")
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))
        dprint(f"after interp {ls_iter=} {t=} {f_new=}")

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1
        dprint(f"new eval {ls_iter=} {t=} {f_new=}")
    dprint(f"post bt {t=} {f_new=} {g_new=}")

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    dprint(f"{low_pos=} {high_pos=} {bracket=} {bracket_f=} {bracket_g=} {bracket_gtd=} {t=}")
    
    while not done and ls_iter < max_ls:
        dprint(ls_iter)
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            dprint("small")
            break

        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        dprint(f"before insuf {t=} {f_new=}")
        if min(max(bracket) - t, t - min(bracket)) < eps:
            dprint("A1")
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                dprint("A2")
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    dprint("A")
                    t = max(bracket) - eps
                else:
                    dprint("B")
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False
        dprint(f"after insuf {t=} {f_new=}")

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1
        dprint(f"after new point {t=} {f_new=}")

        dprint(f"bX {f_new=} {f=} {t=} {gtd=} {f_new=} {low_pos=} {bracket_f[low_pos]=}")
        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            dprint("X")
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                dprint("Y")
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                dprint("Z")
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    dprint(f"ret {bracket=} {bracket_f=} {bracket_g=} {low_pos=}")
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals


# %%

# %%

# %%

# %%

# %%
[_strong_wolfe(obj_func, x1, t1, d1, f1, g1, gtd1) for x1, t1, d1, f1, g1, gtd1 in zip(x1s, t1s, d1s, f1s, g1s, gtd1s)]

# %%
# grads correct
# t looks flipped? 0 where 1 and so on


# %%
# f_new[6]

# %%
t1s

# %%
f_new

# %%
res = [backtrack(obj_func, x1, t1, d1, f1, g1, gtd1) for x1, t1, d1, f1, g1, gtd1 in zip(x1s, t1s, d1s, f1s, g1s, gtd1s)]

# %%
f_new

# %%
f1s, g1s, d1s, gtd1s

# %%
list(zip(x1s, t1s, d1s, f1s, g1s, gtd1s))[0]

# %%
f1s, g1s, d1s, gtd1s

# %%

# %%

# %%
f_new.shape

# %% [markdown]
# # full newton routine

# %%
vmse = vmap(norm)
vg = vmap(grad(norm))
vfg = vmap(grad_and_value(norm))
vhess = vmap(hessian(norm))

# %%
mse = (norm)
gg = (grad(norm))
fg = (grad_and_value(norm))
hess = (hessian(norm))


# %%
def objw(x, t, d):
    return mse(x + t * d), gg(x + t * d)


# %%
def newtlssingle(x, lr=1.0, nsteps=15000, convergence_f=1e-10):
    nevalstot = 1
    f = mse(x)
    g = gg(x)
    for i in range(nsteps):
        H = hess(x)
        dprint("H", H)
        dprint("g", g)
        d = torch.linalg.solve(H + 1e-5 * torch.eye(3), -g)
        gtd = g.dot(d)
        dprint("A", d, gtd)
        fnew, gnew, t, nevals = _strong_wolfe(objw, x, lr, d, f, g, gtd)
        dprint("B", fnew, gnew, t)
        nevalstot += nevals
        dx = t * d
        print(t, dx, f, fnew)
        if torch.linalg.norm(dx) < convergence_f:
            break
        x += dx
        dprint("x", x)
        f = fnew
        g = gnew
    return x, nevalstot, i


# %%
res = [newtlssingle(x.clone()) for x in x1s[:10]]

# %%
res

# %%
x1s[5], newtlssingle(x1s[5].clone())

# %%
gg(x1s[5])

# %%
hess(x1s[5])


# %%
def batched_newton(locs, nptps, nan_mask, local_geoms, lr=1.0, max_steps=100, convergence_x=1e-5):
    n, p = locs.shape
    nfevals = torch.ones(n, dtype=torch.int)
    nsteps = torch.zeros(n, dtype=torch.int)
    active = torch.arange(n)

    g, f = vfg(locs, nptps, nan_mask, local_geoms)
    locsa = locs
    nptpsa = nptps
    nan_maska = nan_mask
    local_geomsa = local_geoms
    lra = torch.full_like(f, lr)
    damp = 1e-8 * torch.eye(p, dtype=f.dtype, device=f.device)[None]
    
    for _ in range(max_steps):
        # Newton search direction
        h = vhess(locsa, nptpsa, nan_maska, local_geomsa)
        h = h + damp
        d = torch.linalg.solve(h, -g)
        gtd = (g * d).sum(1)
        
        # backtracking line search
        batched_grad_and_obj = lambda x, a: vfg(x, nptpsa[a], nan_maska[a], local_geomsa[a])
        fnew, gnew, t, nevals = batched_strong_wolfe(batched_grad_and_obj, locsa, lra, d, f, g, gtd)
        nfevals[active] += nevals
        dx = t[:, None] * d
        locs[active] += dx
        nsteps[active] += 1
        
        # check stopping conditions
        converged = torch.linalg.norm(dx, dim=1) < convergence_x
        # will never happen...
        # reached_max = nsteps >= max_steps
        
        # update active set
        # remain = ~(converged | reached_max)
        remain = ~converged
        active = active[remain]
        if not active.numel():
            break
        locsa = locs[active]
        nptpsa = nptpsa[remain]
        nan_maska = nan_maska[remain]
        local_geomsa = local_geomsa[remain]
        lra = lra[remain]
        f = fnew[remain]
        g = gnew[remain]

    return locs, nfevals, nsteps

# %%

# %%
