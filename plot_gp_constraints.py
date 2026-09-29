"""GP realizations of the model bias, unconstrained vs constrained, at l_bias = 0.51.

Squared-exp GP prior on epsilon_bias(x) with variance sigma_bias^2 and length scale
l_bias. The constrained GP is conditioned on epsilon_bias(0) = 0 (no bias at the
die exit) and epsilon_bias'(x) = 0 on XD (flat far downstream). Panels:
gp_unconstrained.pdf and gp_constrained.pdf, each with a +/- one std band.

Run: python plot_gp_constraints.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

SIGMA2_BIAS = 1.0                   # sigma_bias^2 : marginal variance
ELL_BIAS = 0.51                     # l_bias       : mean ACF correlation length
X0 = np.array([0.0])                # epsilon_bias(x)  = 0 here
XD = np.linspace(3.5, 5.0, 10)      # epsilon_bias'(x) = 0 here
N_SAMPLES = 10
X_GRID = np.linspace(0.0, 5.0, 200)


def style():
    plt.rcParams.update({
        "font.size": 30, "text.usetex": True, "font.family": "serif",
        "font.weight": "light", "lines.linewidth": 1.5, "legend.fontsize": 26,
        "axes.labelsize": 30, "legend.loc": "best", "lines.markersize": 8,
        "legend.frameon": True,
    })


def k_ff(x, y, ell, s2=SIGMA2_BIAS):
    r = np.subtract.outer(x, y)
    return s2 * np.exp(-0.5 * (r / ell) ** 2)


def k_fd(x, y, ell, s2=SIGMA2_BIAS):
    r = np.subtract.outer(x, y)
    return s2 * (r / ell**2) * np.exp(-0.5 * (r / ell) ** 2)


def k_dd(x, y, ell, s2=SIGMA2_BIAS):
    r = np.subtract.outer(x, y)
    return s2 * (1.0 / ell**2 - r**2 / ell**4) * np.exp(-0.5 * (r / ell) ** 2)


def constrained_covariance(x, ell):
    """Cov of the GP conditioned on f(X0) = 0 and f'(XD) = 0."""
    Kff = k_ff(x, x, ell)
    Kfg = np.hstack([k_ff(x, X0, ell), k_fd(x, XD, ell)])
    Kgg = np.block([
        [k_ff(X0, X0, ell),   k_fd(X0, XD, ell)],
        [k_fd(X0, XD, ell).T, k_dd(XD, XD, ell)]])
    A = Kfg @ np.linalg.pinv(Kgg, rcond=1e-10)
    Kc = Kff - A @ Kfg.T
    return 0.5 * (Kc + Kc.T)


def draw(K, n, rng):
    """Sample from N(0, K) via eigendecomposition (robust to tiny negatives)."""
    w, V = np.linalg.eigh(K)
    L = V * np.sqrt(np.clip(w, 0.0, None))
    return rng.normal(size=(n, K.shape[0])) @ L.T


def draw_panel(ax, K, rng, constrained):
    x = X_GRID
    std = np.sqrt(np.clip(np.diag(K), 0.0, None))
    if constrained:
        ax.axvspan(XD[0], XD[-1], color="tab:red", alpha=0.12, lw=0, zorder=0)
    ax.fill_between(x, -std, std, color="0.85", lw=0, zorder=0)
    ax.plot(x, draw(K, N_SAMPLES, rng).T, color="#4C72B0", lw=1.2, alpha=0.7, zorder=2)
    ax.axhline(0.0, color="0.3", lw=0.8, zorder=1)
    if constrained:
        ax.plot(X0, [0.0], "ko", ms=10, zorder=5, clip_on=False)
    ax.set_xlim(0, 5)
    ax.set_ylim(-3, 3)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\varepsilon_{\mathrm{bias}}(x)$")


def main():
    style()
    rng = np.random.default_rng(1)
    for name, K, constrained in (
            ("gp_unconstrained.pdf", k_ff(X_GRID, X_GRID, ELL_BIAS), False),
            ("gp_constrained.pdf", constrained_covariance(X_GRID, ELL_BIAS), True)):
        fig, ax = plt.subplots(figsize=(8, 6))
        draw_panel(ax, K, rng, constrained)
        out = HERE / name
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
