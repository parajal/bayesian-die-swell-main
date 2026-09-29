"""First normal stress difference N1 at the die wall vs Wi, Oldroyd-B and Giesekus.

For every simulation folder in datas/infer-oldroyd-rom1 with the chosen beta and
alpha, lambda, beta, eta_0 and U_avg are read from its input.txt. N1 is evaluated
in steady simple shear at the wall shear rate of the cylindrical die,
gamma_dot = 4 U_avg / R (R = 1), so Wi = lambda * gamma_dot = 0.4 lambda.
Oldroyd-B (alpha = 0) uses the same lambda, beta and eta_0 as the Giesekus runs.

Run: python plot_n1_vs_wi.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compute_n1_giesekus_function import compute_n1_giesekus
from compute_n1_oldroyd_b_function import compute_n1_oldroyd_b

DATA_DIR = Path(__file__).resolve().parent / "datas" / "infer-oldroyd-rom1"
BETA = 0.5
ALPHAS = (0.01, 0.05, 0.1, 0.2)
RADIUS = 1.0  # die radius

def style():
    """Undo romlab's global rcParams; serif LaTeX to match figure4."""
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


def runs(alpha):
    """Namelist values of every folder with beta = BETA and this alpha, sorted by lambda."""
    found = [read_input(f) for f in DATA_DIR.glob("giesekus_lam_*_beta_*_alpha_*")]
    found = [p for p in found if np.isclose(p["betav"], BETA) and np.isclose(p["mobility"], alpha)]
    return sorted(found, key=lambda p: p["lambda"])


def main():
    series = {}
    for alpha in ALPHAS:
        pts = runs(alpha)
        if not pts:
            print(f"alpha = {alpha}: no folders found")
            continue
        wi, n1 = [], []
        for p in pts:
            r = compute_n1_giesekus(U_avg=p["U_avg"], radius=RADIUS, lam=p["lambda"],
                                    beta=p["betav"], eta0=p["eta_0"], alpha=alpha)
            wi.append(p["lambda"] * r["rates"][0])
            n1.append(r["N1"][0])
        series[alpha] = (np.array(wi), np.array(n1))

    # Oldroyd-B on the same (lambda, beta, eta_0, U_avg) as the Giesekus runs
    pts = runs(ALPHAS[0])
    wi, n1 = [], []
    for p in pts:
        r = compute_n1_oldroyd_b(U_avg=p["U_avg"], radius=RADIUS, lam=p["lambda"], beta=p["betav"], eta0=p["eta_0"])
        wi.append(p["lambda"] * r["rates"][0])
        n1.append(r["N1"][0])
    series[0.0] = (np.array(wi), np.array(n1))

    print(f"beta = {BETA}")
    for alpha in sorted(series):
        wi, n1 = series[alpha]
        print(f"alpha = {alpha:<5}  Wi = {np.round(wi, 3)}  N1 = {np.round(n1, 4)}")

    style()
    fig, ax = plt.subplots(figsize=(8, 6))
    for alpha in sorted(series):
        wi, n1 = series[alpha]
        label = r"$\alpha = 0$" if alpha == 0 else rf"$\alpha = {alpha:g}$"
        ax.plot(wi, n1, "o-", label=label)
    ax.set(xlabel="$Wi$", ylabel=r"$N_1$")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 4.4)
    ax.set_xticks(np.linspace(0, 4.4, 5))
    ax.set_ylim(0, 2.0)
    ax.set_yticks(np.linspace(0, 2, 5))
    ax.legend()

    out = Path(__file__).resolve().parent / f"n1_vs_wi_beta_{BETA:g}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)
    plt.show()


if __name__ == "__main__":
    main()
