"""Oldroyd-B model bias delta(x) over all Giesekus runs, and its correlation length.

For every simulation folder in datas/infer-oldroyd-rom1, model_bias.txt holds
delta(x) = y_true(x) - y_model(x; theta_MLE) of the Oldroyd-B ROM fit (written by
Model.compute_model_bias). Five figures are produced:

* model_bias_curves.pdf -- all epsilon_bias(x) curves, coloured by the Giesekus alpha.
* l_bias_hist.pdf       -- histogram of l_bias, the correlation length of each
  delta(x), from a squared-exp fit exp(-t^2 / 2 l^2) to the early lags of its
  sample ACF (same recipe as Model.fix_l_bias_from_acf).
* model_bias.pdf        -- both panels side by side with (a)/(b) captions.
* acf_fit.pdf           -- sample ACF of the median-l_bias curve with its kernel fit.
* acf_all.pdf           -- sample ACFs of all curves and the kernel at the mean l_bias.

Run: python plot_model_bias.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "datas" / "infer-oldroyd-rom1"
CMAP = "plasma"


def style():
    plt.rcParams.update({
        "font.size": 30, "text.usetex": True, "font.family": "serif",
        "font.weight": "light", "lines.linewidth": 1.5, "legend.fontsize": 26,
        "axes.labelsize": 30, "legend.loc": "best", "lines.markersize": 8,
        "legend.frameon": True,
    })


def alpha_of(folder):
    """Giesekus alpha from a folder name like giesekus_lam_1_beta_0p5_alpha_0p05."""
    return float(re.search(r"alpha_([0-9p]+)$", folder.name).group(1).replace("p", "."))


def kernel(t, ell):
    """Squared-exp correlation exp(-t^2 / 2 l^2)."""
    return np.exp(-0.5 * (t / ell) ** 2)


def sample_acf(x, delta):
    """Lags and normalized sample ACF of delta(x) on a uniform grid."""
    xu = np.linspace(x.min(), x.max(), x.size)
    du = np.interp(xu, x, delta - delta.mean())
    acf = np.correlate(du, du, "full")[du.size - 1:]
    return np.arange(acf.size) * (xu[1] - xu[0]), acf / acf[0]


def l_bias(x, delta):
    """Squared-exp correlation length of delta(x) and the number of lags fitted."""
    lags, acf = sample_acf(x, delta)
    dxu, xr = float(lags[1]), float(x.max() - x.min())

    below = np.where(acf <= np.exp(-0.5))[0]  # e^-1/2 crossing: initial guess / fallback
    if below.size and below[0] > 0:
        i = below[0]
        l_est = float((i - 1 + (np.exp(-0.5) - acf[i - 1]) / (acf[i] - acf[i - 1])) * dxu)
    else:
        l_est = 0.1 * xr

    nfit = int(np.clip(round(3 * l_est / dxu), 5, acf.size))  # fit early lags only
    try:
        l_hat = float(curve_fit(kernel, lags[:nfit], acf[:nfit], p0=[max(l_est, dxu)], maxfev=10000)[0][0])
    except (RuntimeError, ValueError):
        l_hat = l_est
    return float(np.clip(l_hat, dxu, xr)), nfit


def load():
    """(alpha, x, delta) for every folder, sorted by alpha."""
    runs = []
    for f in sorted(DATA_DIR.glob("giesekus_*")):
        x, delta = np.loadtxt(f / "model_bias.txt", unpack=True)
        order = np.argsort(x)
        runs.append((alpha_of(f), x[order], delta[order]))
    return sorted(runs, key=lambda r: r[0])


def draw_curves(fig, ax, runs):
    alphas = np.array([a for a, _, _ in runs])
    norm = LogNorm(alphas.min(), alphas.max())
    cmap = plt.get_cmap(CMAP)
    for a, x, d in runs:  # ascending alpha: largest biases drawn on top
        ax.plot(x, d, color=cmap(norm(a)), lw=0.8, alpha=0.6)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.025, 0.032)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_yticks([-0.02, -0.01, 0, 0.01, 0.02, 0.03])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\varepsilon_{\mathrm{bias}}(x)$")
    ax.grid(True, alpha=0.3)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cb.set_label(r"$\alpha$")


def draw_hist(ax, ls):
    mean = ls.mean()
    ax.hist(ls, bins=20, color="#4C72B0", edgecolor="white", linewidth=0.8, zorder=2)
    ax.axvline(mean, color="tab:red", ls="--", lw=2.5, zorder=3, label=rf"mean $={mean:.2f}$")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 60)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel(r"$\ell_{\mathrm{bias}}$")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")


def draw_acf_fit(ax, run, ell, nfit):
    """Sample ACF of one curve, the fitted kernel and l_bias at the e^-1/2 level."""
    _, x, d = run
    lags, acf = sample_acf(x, d)
    lvl = np.exp(-0.5)
    t = np.linspace(0, 2, 400)
    ax.plot(lags[:nfit:2], acf[:nfit:2], "o", color="#4C72B0", ms=6, label="sample ACF")
    ax.plot(t, kernel(t, ell), "--", color="tab:red", lw=2.5,
            label=r"$e^{-r^2/2\ell_{\mathrm{bias}}^2}$")
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.plot([0, ell, ell], [lvl, lvl, -0.5], ":", color="k", lw=1.5)
    ax.text(ell + 0.04, -0.4, rf"$\ell_{{\mathrm{{bias}}}}={ell:.2f}$", va="bottom")
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.5, 1.05)
    ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticks([-0.5, 0, 0.5, 1])
    ax.set_xlabel(r"lag $r$")
    ax.set_ylabel("ACF")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")


def draw_acf_all(fig, ax, runs, ell_mean):
    """Sample ACFs of all curves, coloured by alpha, with the kernel at the mean l_bias."""
    alphas = np.array([a for a, _, _ in runs])
    norm = LogNorm(alphas.min(), alphas.max())
    cmap = plt.get_cmap(CMAP)
    for a, x, d in runs:
        lags, acf = sample_acf(x, d)
        ax.plot(lags, acf, color=cmap(norm(a)), lw=0.8, alpha=0.5)
    t = np.linspace(0, 2, 400)
    ax.plot(t, kernel(t, ell_mean), "--", color="k", lw=3,
            path_effects=[pe.Stroke(linewidth=5.5, foreground="white"), pe.Normal()])
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.5, 1.05)
    ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticks([-0.5, 0, 0.5, 1])
    ax.set_xlabel(r"lag $r$")
    ax.set_ylabel("ACF")
    ax.grid(True, alpha=0.3)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cb.set_label(r"$\alpha$")


def save(fig, name):
    out = HERE / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")


def main():
    style()
    runs = load()
    fits = [l_bias(x, d) for _, x, d in runs]
    ls = np.array([l for l, _ in fits])
    print(f"l_bias over {ls.size} curves: mean = {ls.mean():.3f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    draw_curves(fig, ax, runs)
    save(fig, "model_bias_curves.pdf")

    fig, ax = plt.subplots(figsize=(8, 6))
    draw_hist(ax, ls)
    save(fig, "l_bias_hist.pdf")

    # side-by-side panel with sub-captions under each axis
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(17, 6), width_ratios=[1.12, 1])
    draw_curves(fig, a0, runs)
    draw_hist(a1, ls)
    fig.subplots_adjust(wspace=0.3)
    for ax, cap in ((a0, r"(a) Model bias $\varepsilon_{\mathrm{bias}}(x)$"),
                    (a1, r"(b) Correlation length $\ell_{\mathrm{bias}}$")):
        ax.text(0.5, -0.3, cap, transform=ax.transAxes, ha="center", va="top")
    save(fig, "model_bias.pdf")

    k = int(np.argmin(np.abs(ls - np.median(ls))))  # representative curve: median l_bias
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_acf_fit(ax, runs[k], *fits[k])
    save(fig, "acf_fit.pdf")

    fig, ax = plt.subplots(figsize=(8, 6))
    draw_acf_all(fig, ax, runs, ls.mean())
    save(fig, "acf_all.pdf")
    plt.show()


if __name__ == "__main__":
    main()
