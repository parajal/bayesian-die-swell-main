"""Steady swell ratio h(L)/H vs Wi, Oldroyd-B and Giesekus.

Giesekus: every folder in datas/infer-oldroyd-rom1 with the chosen beta and alpha;
lambda, beta, eta_0 and U_avg come from its input.txt and the steady free surface
h(x) from curve4_y.txt (first entry = outlet x = L_extr, last = die exit, h = H).
Oldroyd-B: the runs in datas/oldroyd-rom1 with the same beta (parameters.txt: lambda, beta,
one row per pair, the swell being independent of eta_0; curve4_y_all.txt: one free surface per row).
Both sets use U_avg = 0.1 in a cylindrical die of radius R = 1, so
Wi = lambda * 4 U_avg / R = 0.4 lambda.

Run: python plot_swell_vs_wi.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
GIESEKUS_DIR = ROOT / "datas" / "infer-oldroyd-rom1"
OLDROYD_DIR = ROOT / "datas" / "oldroyd-rom1"
BETA = 0.5
ETA0 = 1.0
ALPHAS = (0.01, 0.05, 0.1, 0.2)
RADIUS = 1.0      # die radius H
U_AVG = 0.1       # mean inlet velocity of the Oldroyd-B sweep (read from input.txt for Giesekus)
OLDROYD_EXTRA = {11.0: 1.858}  # lambda: swell height h(L) of Oldroyd-B runs not in oldroyd-rom1


def style():
    """Serif LaTeX, as in plot_n1_vs_wi.py."""
    plt.rcParams.update({
        "font.size": 30, "text.usetex": True, "font.family": "serif",
        "font.weight": "light", "lines.linewidth": 1.5, "legend.fontsize": 26,
        "axes.labelsize": 30, "legend.loc": "best", "lines.markersize": 8,
        "legend.frameon": True,
    })


def read_input(folder):
    """lambda, betav, mobility, eta_0 and U_avg from a folder's Fortran namelist input.txt."""
    text = (folder / "input.txt").read_text()
    get = lambda key: float(re.search(rf"^\s*{key}\s*=\s*([-+0-9.eEdD]+)", text, re.M).group(1).replace("d", "e"))
    return {k: get(k) for k in ("lambda", "betav", "mobility", "eta_0", "U_avg")}


def giesekus_swell(alpha):
    """(Wi, swell ratio) of every Giesekus folder with beta = BETA and this alpha, sorted by Wi."""
    pts = []
    for folder in GIESEKUS_DIR.glob("giesekus_lam_*_beta_*_alpha_*"):
        p = read_input(folder)
        if np.isclose(p["betav"], BETA) and np.isclose(p["mobility"], alpha) and np.isclose(p["eta_0"], ETA0):
            h = np.loadtxt(folder / "curve4_y.txt")
            pts.append((p["lambda"] * 4 * p["U_avg"] / RADIUS, h[0] / RADIUS))
    return np.array(sorted(pts)).T


def oldroyd_swell():
    """(Wi, swell ratio) of the Oldroyd-B sweep at beta = BETA, sorted by Wi."""
    P = np.atleast_2d(np.loadtxt(OLDROYD_DIR / "parameters.txt"))   # lambda, beta
    H = np.loadtxt(OLDROYD_DIR / "curve4_y_all.txt", ndmin=2)
    rows = np.isclose(P[:, 1], BETA)
    lam = np.concatenate([P[rows, 0], list(OLDROYD_EXTRA)])
    h = np.concatenate([H[rows, 0], list(OLDROYD_EXTRA.values())])
    order = np.argsort(lam)
    return lam[order] * 4 * U_AVG / RADIUS, h[order] / RADIUS


def main():
    series = {0.0: oldroyd_swell()}
    for alpha in ALPHAS:
        wi, sr = giesekus_swell(alpha)
        if len(wi):
            series[alpha] = (wi, sr)
        else:
            print(f"alpha = {alpha}: no folders found")

    print(f"beta = {BETA}, eta_0 = {ETA0}")
    for alpha in sorted(series):
        wi, sr = series[alpha]
        print(f"alpha = {alpha:<5}  Wi = {np.round(wi, 3)}  swell ratio = {np.round(sr, 4)}")

    style()
    fig, ax = plt.subplots(figsize=(8, 6))
    for alpha in sorted(series):
        wi, sr = series[alpha]
        label = r"$\alpha = 0$" if alpha == 0 else rf"$\alpha = {alpha:g}$"
        ax.plot(wi, sr, "o-", label=label)
    ax.set(xlabel="$Wi$", ylabel=r"$h/H$")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 4.4)
    ax.set_xticks(np.linspace(0, 4.4, 5))
    ax.set_ylim(1, 2.0)
    ax.set_yticks(np.linspace(1, 2, 5))
    ax.legend(loc="best")  # beside the axes: inside it covers the curves

    out = ROOT / f"swell_vs_wi_beta_{BETA:g}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)
    plt.show()


if __name__ == "__main__":
    main()
