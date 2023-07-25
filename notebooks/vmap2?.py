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
import torch
from torch import vmap
import torch.nn.functional as F
from torch.func import grad, hessian, grad_and_value
from torch.autograd.functional import jacobian, hessian as hessian_ag


# %%
@torch.compile
def norm(g):
    return torch.linalg.norm(g)


# %%
xs = torch.ones(10, 3)

# %%
norm(torch.ones(3))

# %%
vmap_norm = vmap(norm)

# %%
vmap_norm(xs)

# %%

# %%

# %%
from torch._dynamo import allow_in_graph
from functools import wraps

def traceable(f):
    f = allow_in_graph(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


# %%
def objective(x):
    return torch.square(x).sum()

# g_and_f = grad_and_value(objective)
gradient = traceable(grad(objective))
hess = traceable(hessian(objective))
# @torch.compile
# def gradient(x):
#     return jacobian(objective, x)
# @torch.compile
# def hess(x):
#     return hessian_ag(objective, x)

@torch.compile
def g_and_f(x):
    return gradient(x), objective(x)


# %%
g_and_f(torch.tensor([1., 1.]))


# %%
@torch.compile
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
@torch.compile
def baked_strong_wolfe(x,
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
    g_new, f_new = g_and_f(x + t * d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
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
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        g_new, f_new = g_and_f(x + t * d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

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
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
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
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        g_new, f_new = g_and_f(x + t * d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
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
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals


# %%
def newtlssingle(x, lr=1.0, nsteps=100, convergence_x=1e-5):
    nevalstot = 1
    f = objective(x)
    g = gradient(x)
    for i in range(nsteps):
        H = hess(x)
        d = torch.linalg.solve(H, -g)
        
        print(g.shape, d.shape)
        gtd = g.dot(d)
        fnew, gnew, t, nevals = baked_strong_wolfe(x, lr, d, f, g, gtd)
        
        
        nevalstot += nevals
        dx = t * d
        if torch.linalg.norm(dx) < convergence_x:
            break
        x += dx
        f = fnew
        g = gnew
    return x, nevalstot, i

# %%

# %%

# %%

# %%
newtlssingle(torch.tensor([1., 1., 1.]))

# %%
x = torch.ones(3)
f = objective(x)
g = gradient(x)
h = hess(x)
d = -g
gtd = g.dot(d)

# %%
baked_strong_wolfe(x, 1.0, d, f, g, gtd)

# %%
vmap_obj = traceable(vmap(objective))
# vmap_grad = vmap(grad(objective))
vmap_grad = traceable(vmap(gradient))
# vmap_hess = vmap(hessian(objective))
vmap_hess = traceable(vmap(hess))
vmap_baked_strong_wolfe = vmap(baked_strong_wolfe, (0, None, 0, 0, 0, 0), (0, 0, 0, 0))

# %%
xs = torch.ones(10, 3)
fs = vmap_obj(xs)

# %%
gs = vmap_grad(xs)

# %%
ds = -gs
gtds = torch.einsum("ij,ij->i", gs, ds)

# %%
gs

# %%
vmap_baked_strong_wolfe(xs, 1.0, ds, fs, gs, gtds)


# %%
def newtlsmulti(xs, lr=1.0, nsteps=100, convergence_x=1e-5):
    nevalstot = 1
    
    for i in range(nsteps):
        fs = vmap_obj(xs)
        gs = vmap_grad(xs)
        Hs = vmap_hess(xs)
        print(Hs.shape)
        ds = torch.linalg.solve(Hs, -gs)
        
        print(gs.shape, ds.shape)
        gtds = gs @ ds.T
        
        fnew, gnew, t, nevals = vmap_baked_strong_wolfe(xs, lr, ds, fs, gs, gtds)

        nevalstot += nevals
        dx = t * ds
        if torch.linalg.norm(dx, dim=1).max() < convergence_x:
            break
        x += dx
    return x, nevalstot, i


# %%
newtlsmulti(torch.ones((10, 3), requires_grad=True))

# %%
