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
from torch.optim import Adam
from torch import vmap
import torch.nn.functional as F


# %%
def obj(x):
    return torch.square(x).sum()


# %%
x0 = torch.arange(80.).reshape(40, 2)

# %%
vmap(obj)(x0)

# %%
vmap(obj)(x0).shape


# %%
def adamminimizer(x):
    def minme(i):
        opt = Adam([x])
        for _ in range(100):
            opt.zero_grad()
            loss = obj(x[i])
            loss.backward()
            opt.step()
    return minme


# %%
x1 = x0.clone().detach().requires_grad_()

# %%
minme = adamminimizer(x1)

# %%
[minme(i) for i in range(40)];

# %%
(x0 - x1).mean()

# %%
x2 = x0.clone().detach().requires_grad_()

# %%
minme2 = adamminimizer(x2)

# %%
vmap(minme2)(torch.arange(40))


# %%
def minme3(x):
    opt = Adam([x])
    for _ in range(100):
        opt.zero_grad()
        loss = obj(x)
        loss.backward()
        opt.step()


# %%
x3 = x0.clone().detach().requires_grad_()

# %%
vmap(minme3)(x3)

# %%
from torch.func import grad, hessian

# %%
hess = hessian(obj)
g = grad(obj)

# %%
from torch.optim.lbfgs import _strong_wolfe


def strong_wolfe(obj_func, x, t, d, f, g, gtd, c1=0.0001, c2=0.9, tolerance_change=1e-09, max_ls=25):
    """Documented wrapper around torch's strong Wolfe line search routine
    
    torch's original lua is much better documented:
    https://github.com/torch/optim/blob/master/lswolfe.lua
    
    acknowledgements to the nice library and useful reference
    @rfeinman/pytorch-minimize/blob/master/torchmin/line_search.py
    
    Arguments
    ---------
    obj_func : function
        Takes arguments x, t, d (see below) and returns the objective and its gradient
        at `x+t*d`.
    x : tensor
        Current center / starting point
    t : float
        Step size. But wait I thought the line search finds the step size?
        Well you're right. This is the initial value. In torch LBFGS they set this
        to `lr` except on the first iterate, when they use `lr*min(1,l1norm(grad))`
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
    return _strong_wolfe(obj_func, x, t, d, f, g, gtd, c1=c1, c2=c2, tolerance_change=tolerance_change, max_ls=max_ls)


def ls_wrap_objective(objective):
    g = grad(objective)
    def wrapped(x, t, d):
        return objective(x + t * d), g(x + t * d)
    return wrapped


def gdlssingle(objective, x, lr=1.0, nsteps=100, convergence_x=1e-5):
    wrapped = ls_wrap_objective(objective)
    f, g = wrapped(x, 0, 0)
    nevalstot = 1
    for i in range(nsteps):
        d = -g
        gtd = g.dot(d)
        fnew, gnew, t, nevals = strong_wolfe(wrapped, x, lr, d, f, g, gtd)
        nevalstot += nevals
        dx = t * d
        if torch.linalg.norm(dx) < convergence_x:
            break
        x += dx
        f = fnew
        g = gnew
    return x, nevalstot, i


def newtlssingle(objective, x, lr=1.0, nsteps=100, convergence_x=1e-5):
    wrapped = ls_wrap_objective(objective)
    gg = grad(objective)
    HH = hessian(objective)
    nevalstot = 1
    f = objective(x)
    g = gg(x)
    for i in range(nsteps):
        H = HH(x)
        d = torch.linalg.solve(H, -g)
        gtd = g.dot(d)
        fnew, gnew, t, nevals = strong_wolfe(wrapped, x, lr, d, f, g, gtd)
        nevalstot += nevals
        dx = t * d
        if torch.linalg.norm(dx) < convergence_x:
            break
        x += dx
        f = fnew
        g = gnew
    return x, nevalstot, i


def newtlsmulti(objective, xs, lr=1.0, nsteps=100, convergence_x=1e-5):
    wrapped = ls_wrap_objective(objective)
    gg = grad(objective)
    HH = hessian(objective)
    
    def optsingle(x):
        nevalstot = 1
        f = objective(x)
        g = gg(x)
        for i in range(nsteps):
            H = HH(x)
            d = torch.linalg.solve(H, -g)
            gtd = g.dot(d)
            fnew, gnew, t, nevals = strong_wolfe(wrapped, x, lr, d, f, g, gtd)
            nevalstot += nevals
            dx = t * d
            if torch.linalg.norm(dx) < convergence_x:
                break
            x += dx
            f = fnew
            g = gnew
        return x, nevalstot, i
    
    return vmap(optsingle)(xs)


# %%
x41 = x0[1].clone().detach()

# %%
x41

# %%
gdlssingle(obj, x41)

# %%
x41

# %%
x42 = x0[1].clone().detach()
x42

# %%
newtlssingle(obj, x42)

# %%
x5 = x0.clone().detach()

# %%
newtlsmulti(obj, x5)


# %%

# %%
def gdme(x):
    print(x.shape)
    for _ in range(20):
        x -= 0.1 * g(x)
    return x


# %%
def newtme(x):
    # for _ in range(50):
    #     x = x - 0.1 * g(x)
    print('g', x)
    for _ in range(20):
        hh = hess(x)
        x = x + torch.linalg.solve(hess(x) + 1e-10 * torch.eye(x.numel()), -g(x))
        print('h', x)
    return x


# %%
import numpy as np
hh = np.array([[ 1.2902, -0.8017],
        [-0.8017,  0.4982]])

# %%
# newtme(x41)

# %%
x4 = x0.clone().detach()

# %%
gd4 = vmap(gdme)(x4)

# %%
(x0 - gd4).mean()

# %%
x4 = x0.clone().detach().requires_grad_()

# %%
newt4 = vmap(newtme)(x4);

# %%
(x0 - newt4).mean()

# %%
