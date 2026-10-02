"""
1D harmonic oscillator: finite differences vs a PINN.

I use hbar = m = omega = 1, so V = x^2/2 and the ground state is
E0 = 0.5, psi0 = pi^(-1/4) exp(-x^2/2). Having the exact answer is the
whole point -- every error below is measured against it.

    python schrodinger_pinn_vs_fd.py

Plots go into ./figures.
"""
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import matplotlib.pyplot as plt

L = 6.0          # box is [-L, L], psi = 0 at the walls
E0 = 0.5

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# float32 was fine for Adam but L-BFGS kept stalling at ~1e-4, so float64 everywhere
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)


def psi0(x):
    return np.pi**-0.25 * np.exp(-x**2 / 2)


def rel_l2(x, psi):
    # eigenvectors come back with an arbitrary sign, so flip it to match first
    ref = psi0(x)
    psi = psi * np.sign(np.sum(psi * ref))
    return np.sqrt(np.trapezoid((psi - ref)**2, x) / np.trapezoid(ref**2, x))


def normalise(x, psi):
    return psi / np.sqrt(np.trapezoid(psi**2, x))


# ---- finite differences ----

def solve_fd(n):
    """Standard 3-point stencil, n interior points, Dirichlet walls."""
    x = np.linspace(-L, L, n + 2)[1:-1]
    h = x[1] - x[0]
    T = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n)) * (-0.5 / h**2)
    H = (T + sp.diags(0.5 * x**2)).tocsc()

    # these solves take well under a ms, so the timer is noisy -- take the best of 5
    best = np.inf
    for _ in range(5):
        t0 = time.process_time()
        E, v = spla.eigsh(H, k=1, sigma=0.0)
        best = min(best, time.process_time() - t0)

    return x, normalise(x, v[:, 0]), E[0], best


# ---- PINN ----

class PINN(torch.nn.Module):
    def __init__(self, width=32, depth=3):
        super().__init__()
        layers, n_in = [], 1
        for _ in range(depth):
            layers += [torch.nn.Linear(n_in, width), torch.nn.Tanh()]
            n_in = width
        layers.append(torch.nn.Linear(width, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        # multiplying by (1 - x^2/L^2) pins psi to zero at the walls,
        # so I don't need a separate boundary term in the loss
        return (1 - (x / L)**2) * self.net(x)


def loss_fn(model, x, e_weight):
    psi = model(x)
    dpsi = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
    d2psi = torch.autograd.grad(dpsi.sum(), x, create_graph=True)[0]
    Hpsi = -0.5 * d2psi + 0.5 * x**2 * psi

    E = (psi * Hpsi).mean() / (psi**2).mean()    # Rayleigh quotient
    resid = ((Hpsi - E * psi)**2).mean()
    norm = (2 * L * (psi**2).mean() - 1)**2      # without this, psi = 0 is a perfect "solution"
    return resid + norm + e_weight * E, E


def train(n_adam=2000, n_lbfgs=40, n_pts=256, lr=2e-3, log_every=100):
    model = PINN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=800, gamma=0.3)

    x_eval = torch.linspace(-L, L, 1001).reshape(-1, 1)
    xe = x_eval.numpy().ravel()
    hist = []      # cpu time, |E - E0|, L2 error, loss, iteration
    cpu, it = 0.0, 0

    def record(E, loss):
        pe = normalise(xe, model(x_eval).detach().numpy().ravel())
        hist.append((cpu, abs(E - E0), rel_l2(xe, pe), loss, it))

    # stage 1: Adam, new random points every step
    for ep in range(n_adam + 1):
        t0 = time.process_time()
        x = ((torch.rand(n_pts, 1) * 2 - 1) * L).requires_grad_(True)
        # a small push on E early on so it settles in the ground state,
        # not an excited one. Turned off later so it doesn't bias E.
        loss, E = loss_fn(model, x, 0.1 if ep < 1000 else 0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        cpu += time.process_time() - t0
        it += 1
        if ep % log_every == 0:
            record(E.item(), loss.item())

    # stage 2: L-BFGS. Its line search needs the same loss every call,
    # so the points are fixed here (random points just confuse it)
    x_fix = torch.linspace(-L, L, 2 * n_pts).reshape(-1, 1).requires_grad_(True)
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=25,
                              history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss, _ = loss_fn(model, x_fix, 0.0)
        loss.backward()
        return loss

    for _ in range(n_lbfgs):
        t0 = time.process_time()
        lbfgs.step(closure)
        cpu += time.process_time() - t0
        it += 25
        loss, E = loss_fn(model, x_fix, 0.0)
        record(E.item(), loss.item())

    pe = normalise(xe, model(x_eval).detach().numpy().ravel())
    pe *= np.sign(np.sum(pe * psi0(xe)))
    return xe, pe, E.item(), np.array(hist), n_adam


# ---- plots ----

BLUE, RED, GREY = "#1f6feb", "#d1495b", "#333333"


def plot_all(fd, x_fd, psi_fd, x_nn, psi_nn, hist, n_adam):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    # the wavefunctions themselves + where each method goes wrong
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(x_nn, psi0(x_nn), c=GREY, lw=3, alpha=0.35, label="Exact")
    a.plot(x_fd, psi_fd, "--", c=BLUE, label="Finite difference (N=200)")
    a.plot(x_nn, psi_nn, ":", c=RED, lw=2, label="PINN")
    a.set(xlabel="x", ylabel=r"$\psi_0(x)$", title="Ground state")
    a.legend(frameon=False)
    b.semilogy(x_fd, np.abs(psi_fd - psi0(x_fd)) + 1e-16, c=BLUE, label="Finite difference")
    b.semilogy(x_nn, np.abs(psi_nn - psi0(x_nn)) + 1e-16, c=RED, label="PINN")
    b.set(xlabel="x", ylabel="|error|", title="Pointwise error")
    b.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "wavefunction_comparison.png", dpi=160)

    # the plot that actually matters: error vs cost
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(fd[:, 1], fd[:, 3], "o-", c=BLUE, label="Finite difference (varying N)")
    for n, t, _, e in fd[::2]:
        ax.annotate(f"N={int(n)}", (t, e), textcoords="offset points", xytext=(6, 4),
                    fontsize=9, color=BLUE)
    ax.loglog(hist[1:, 0], hist[1:, 2], c=RED, label="PINN (Adam, then L-BFGS)")
    ax.set(xlabel="CPU time [s]", ylabel=r"Relative $L_2$ error of $\psi_0$", title="Accuracy vs cost")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "accuracy_vs_time.png", dpi=160)

    # training history -- the jump at the Adam -> L-BFGS switch is the interesting bit
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    it = hist[:, 4]
    ax.semilogy(it, hist[:, 3], c="#888888", label="Loss")
    ax.semilogy(it, hist[:, 1], c=RED, label=r"$|E - E_0|$")
    ax.semilogy(it, hist[:, 2], c="#6a4c93", label=r"$L_2$ error")
    ax.axhline(fd[3, 2], ls="--", c=BLUE, label="FD energy error, N=200")
    ax.axvline(n_adam, c="#bbbbbb", lw=1)
    ax.text(n_adam, ax.get_ylim()[1], " Adam → L-BFGS", va="top", fontsize=9, color="#666")
    ax.set(xlabel="Optimizer iteration", title="PINN training")
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "pinn_training.png", dpi=160)


def main():
    fd = []
    for n in [25, 50, 100, 200, 400, 800, 1600, 3200]:
        x, psi, E, t = solve_fd(n)
        fd.append((n, t, abs(E - E0), rel_l2(x, psi)))
        print(f"FD   N={n:5d}  time={t*1e3:7.2f} ms  |dE|={fd[-1][2]:.2e}  L2={fd[-1][3]:.2e}")
    fd = np.array(fd)

    x_fd, psi_fd, _, _ = solve_fd(200)
    psi_fd *= np.sign(np.sum(psi_fd * psi0(x_fd)))

    x_nn, psi_nn, E_nn, hist, n_adam = train()
    print(f"PINN train={hist[-1, 0]:.1f} s  E={E_nn:.6f}  |dE|={hist[-1, 1]:.2e}  L2={hist[-1, 2]:.2e}")

    plot_all(fd, x_fd, psi_fd, x_nn, psi_nn, hist, n_adam)
    print(f"Figures saved to {FIGDIR}")


if __name__ == "__main__":
    main()
