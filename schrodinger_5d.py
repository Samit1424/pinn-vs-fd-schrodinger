"""
Same oscillator, but in 5 dimensions: V = |x|^2/2, E0 = 5/2,
psi0 = pi^(-5/4) exp(-|x|^2/2).

This is where grids get into trouble. N points per axis means N^5
unknowns, so N=16 is already a million. The PINN only ever looks at
random points, so it doesn't care nearly as much about dimension.

    python schrodinger_5d.py      (a few minutes on a laptop CPU)

Plots go into ./figures.
"""
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import matplotlib.pyplot as plt

D = 5
L = 5.0                   # box is [-L, L]^5
E0 = D / 2

# where the sample points come from (see sample() for why two values)
SIG_TRAIN = 1.0
SIG_TEST = 2**-0.5        # = the width of |psi0|^2 itself

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)


def psi0(x):
    return np.pi**(-D / 4) * np.exp(-np.sum(x**2, axis=-1) / 2)


# ---- finite differences on an N^5 grid ----

def solve_fd(n):
    x = np.linspace(-L, L, n + 2)[1:-1]
    h = x[1] - x[0]
    h1 = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n)) * (-0.5 / h**2) + sp.diags(0.5 * x**2)

    # the 5D Hamiltonian is just five copies of the 1D one added together
    H = h1
    for _ in range(D - 1):
        H = sp.kronsum(H, h1, format="csr")

    t0 = time.process_time()
    E, v = spla.eigsh(H, k=1, which="SA", tol=1e-10)
    t = time.process_time() - t0

    grid = np.stack(np.meshgrid(*([x] * D), indexing="ij"), axis=-1)
    ref = psi0(grid).ravel()
    psi = v[:, 0] * np.sign(v[:, 0] @ ref)
    psi *= np.linalg.norm(ref) / np.linalg.norm(psi)
    l2 = np.linalg.norm(psi - ref) / np.linalg.norm(ref)

    # rough memory: the sparse matrix plus the ~20 vectors Lanczos keeps around
    mem = H.data.nbytes + H.indices.nbytes + H.indptr.nbytes + 20 * n**D * 8
    return h, E[0], l2, t, mem


# ---- PINN ----

class PINN(torch.nn.Module):
    def __init__(self, width=64, depth=3):
        super().__init__()
        layers, n_in = [], D
        for _ in range(depth):
            layers += [torch.nn.Linear(n_in, width), torch.nn.Tanh()]
            n_in = width
        layers.append(torch.nn.Linear(width, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        # First version was envelope * net(x) and it happily converged to
        # E = 3.5 -- the first excited state, which is also a perfect solution.
        # The ground state never changes sign, so writing psi = envelope * exp(net)
        # rules the excited states out. No Gaussian is built in.
        envelope = torch.prod(1 - (x / L)**2, dim=1, keepdim=True)
        return envelope * torch.exp(self.net(x))


def sample(n, sigma=SIG_TRAIN, weighted=False):
    """Gaussian points, clipped to the box.

    Uniform points are useless in 5D -- almost all of them land in the
    corners where psi is ~0. Gaussian points fix that.

    weighted=True also returns 1/p(x) importance weights so averages turn
    into proper integrals. I only use that for testing: in training the
    weights out in the tails get enormous (e^40-ish) and a handful of
    points end up running the whole loss.
    """
    x = torch.randn(n, D) * sigma
    x = x[(x.abs() < L).all(dim=1)]
    if not weighted:
        return x, torch.ones(len(x), 1)
    logw = (x**2).sum(dim=1, keepdim=True) / (2 * sigma**2)
    w = torch.exp(logw - logw.max())
    return x, w / w.mean()


def loss_fn(model, x, w, e_weight):
    x = x.clone().requires_grad_(True)
    psi = model(x)
    g = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
    lap = sum(torch.autograd.grad(g[:, i].sum(), x, create_graph=True)[0][:, i:i + 1]
              for i in range(D))
    Hpsi = -0.5 * lap + 0.5 * (x**2).sum(dim=1, keepdim=True) * psi

    norm = (w * psi**2).mean()
    E = (w * psi * Hpsi).mean() / norm
    # divide by the norm so shrinking psi towards 0 can't cheat the loss
    resid = (w * (Hpsi - E * psi)**2).mean() / norm
    return resid + e_weight * E, E


def rel_l2(model, x, w):
    psi = model(x).detach().numpy().ravel()
    ref = psi0(x.numpy())
    w = w.numpy().ravel()
    c = np.sum(w * psi * ref) / np.sum(w * psi**2)    # best-fit scale (also fixes the sign)
    return np.sqrt(np.sum(w * (c * psi - ref)**2) / np.sum(w * ref**2))


def train(n_adam=2000, n_lbfgs=30, batch=512, log_every=100):
    model = PINN()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=700, gamma=0.3)

    x_test, w_test = sample(20000, SIG_TEST, weighted=True)
    hist, cpu = [], 0.0      # cpu time, |E - E0|, L2 error

    for ep in range(n_adam + 1):
        t0 = time.process_time()
        x, w = sample(batch)
        loss, E = loss_fn(model, x, w, 0.05 if ep < 700 else 0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        cpu += time.process_time() - t0
        if ep % log_every == 0:
            hist.append((cpu, abs(E.item() - E0), rel_l2(model, x_test, w_test)))
            print(f"  adam {ep:5d}  E={E.item():.5f}  L2={hist[-1][2]:.2e}", flush=True)

    x_fix, w_fix = sample(4 * batch)
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=25,
                              history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss, _ = loss_fn(model, x_fix, w_fix, 0.0)
        loss.backward()
        return loss

    for r in range(n_lbfgs):
        t0 = time.process_time()
        lbfgs.step(closure)
        cpu += time.process_time() - t0
        # check E on held-out points, not the ones L-BFGS has been fitting
        _, E = loss_fn(model, x_test[:8000], w_test[:8000], 0.0)
        hist.append((cpu, abs(E.item() - E0), rel_l2(model, x_test, w_test)))
        print(f"  lbfgs {r:3d}  E={E.item():.6f}  L2={hist[-1][2]:.2e}", flush=True)

    return model, np.array(hist)


# ---- plots ----

BLUE, RED = "#1f6feb", "#d1495b"


def plot_cost(fd, hist, model, l2_nn, n_need, p, c, bytes_per_pt):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    a.loglog(fd[:, 4], fd[:, 3], "o-", c=BLUE, label="Finite difference ($N^5$ grid)")
    for n, _, _, e, t, _ in fd:
        a.annotate(f"N={int(n)}", (t, e), textcoords="offset points", xytext=(6, 4), fontsize=9, color=BLUE)
    a.loglog(hist[1:, 0], hist[1:, 2], c=RED, label="PINN (Adam, then L-BFGS)")
    a.set(xlabel="CPU time [s]", ylabel=r"Relative $L_2$ error of $\psi_0$",
          title="5D oscillator: accuracy vs cost")
    a.legend(frameon=False)

    # how much memory a grid would need, extending the measured h^p trend
    ns = np.arange(6, max(n_need, 20) + 1)
    hs = 2 * L / (ns + 1)
    b.loglog(ns**D * bytes_per_pt / 1e9, np.exp(c) * hs**p, "--", c=BLUE, alpha=0.6, label="FD, extrapolated")
    b.loglog(fd[:, 5] / 1e9, fd[:, 3], "o", c=BLUE, label="FD, measured")
    n_par = sum(q.numel() for q in model.parameters())
    b.loglog([n_par * 8 * 3 / 1e9], [l2_nn], "*", ms=16, c=RED, label="PINN weights")
    b.axhline(l2_nn, c=RED, lw=0.8, ls=":")
    b.axvline(16, c="#999", lw=0.8)
    b.text(16, b.get_ylim()[1], " 16 GB laptop", va="top", fontsize=9, color="#666")
    b.set(xlabel="Memory [GB]", ylabel=r"Relative $L_2$ error", title="Memory needed for a given accuracy")
    b.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.9))

    fig.tight_layout()
    fig.savefig(FIGDIR / "highdim_accuracy_vs_cost.png", dpi=160)


def plot_slice(model):
    # can't plot 5D, so look along x1 with the other four coordinates at 0
    s = np.linspace(-L, L, 400)
    pts = np.zeros((400, D))
    pts[:, 0] = s
    psi = model(torch.tensor(pts)).detach().numpy().ravel()
    ref = psi0(pts)
    psi *= np.sum(psi * ref) / np.sum(psi**2)

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.plot(s, ref, c="#333", lw=3, alpha=0.35, label="Exact")
    ax.plot(s, psi, ":", c=RED, lw=2, label="PINN")
    ax.set(xlabel=r"$x_1$  (with $x_2=\dots=x_5=0$)", ylabel=r"$\psi_0$",
           title="Slice through the 5D ground state")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "highdim_slice.png", dpi=160)


def main():
    fd = []
    for n in [6, 8, 10, 12, 14, 16]:     # 16^5 ~ 1M unknowns is about my laptop's limit
        h, E, l2, t, mem = solve_fd(n)
        fd.append((n, h, abs(E - E0), l2, t, mem))
        print(f"FD  N={n:3d}  unknowns={n**D:>9,d}  cpu={t:7.2f} s  mem={mem/1e6:8.1f} MB  "
              f"|dE|={fd[-1][2]:.2e}  L2={l2:.2e}", flush=True)
    fd = np.array(fd)

    model, hist = train()
    l2_nn = hist[-1, 2]
    print(f"PINN cpu={hist[-1, 0]:.1f} s  |dE|={hist[-1, 1]:.2e}  L2={l2_nn:.2e}")

    # FD error should go like h^2. Fit the last three points and ask how
    # fine the grid would have to be to match the PINN.
    p, c = np.polyfit(np.log(fd[-3:, 1]), np.log(fd[-3:, 3]), 1)
    h_need = np.exp((np.log(l2_nn) - c) / p)
    n_need = int(np.ceil(2 * L / h_need)) - 1
    bytes_per_pt = fd[-1, 5] / fd[-1, 0]**D
    print(f"FD fit: L2 ~ h^{p:.2f}.  Matching the PINN needs N={n_need} per axis -> "
          f"{n_need**D:.2e} unknowns, ~{n_need**D * bytes_per_pt / 1e9:.0f} GB")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    plot_cost(fd, hist, model, l2_nn, n_need, p, c, bytes_per_pt)
    plot_slice(model)
    print(f"Figures saved to {FIGDIR}")


if __name__ == "__main__":
    main()
