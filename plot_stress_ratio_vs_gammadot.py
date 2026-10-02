"""Inferred vs true wall stress ratio N1/tau_w against the wall shear rate, Oldroyd-B ROM with model bias.

For the Giesekus runs at lambda_true = 5, beta_true = 0.5 and alpha = 0.01, 0.05, 0.1, 0.2, the
inference results in results/bias_true_infer-<U_avg>_giesekus_lam_5_beta_0p5_alpha_<a>.txt give
E[S_R] = E[N1/tau_w] with its 90 % credible interval (inferred) and S_R_true (FEM wall N1 over
tau_w = R |dp/dx| / 2). Both are plotted against the nominal wall shear rate gamma_dot_w = 4 U_avg / R.

The infer-0.4 folders hold the FEM data run at U_avg = 0.2 (DATA_UAVG), so that point is drawn open and
joined by faded lines.

Run: python plot_stress_ratio_vs_gammadot.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_n1_vs_wi import style

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LAMBDA_TRUE, BETA_TRUE = 5, 0.5
ALPHAS = (0.01, 0.05, 0.1, 0.2)
U_AVGS = (0.05, 0.1, 0.2, 0.4)
DATA_UAVG = {0.4: 0.2}  # U_avg whose folder holds FEM data run at another U_avg
RADIUS = 1.0
COLORS = plt.cm.viridis(np.linspace(0, 0.9, len(ALPHAS)))


def tag(v):
    return f"{v:g}".replace(".", "p")


def read_result(u_avg, alpha):
    """(E[S_R], 90 % CI low, high, S_R_true) from one inference results file."""
    f = RESULTS / (f"bias_true_infer-{u_avg:g}_giesekus_lam_{tag(LAMBDA_TRUE)}_beta_{tag(BETA_TRUE)}"
                   f"_alpha_{tag(alpha)}.txt")
    rows = {}
    for line in f.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, *vals = line.split("#")[0].split()
            rows[name] = [float(v) for v in vals]
    return (*rows["E[S_R]"], rows["S_R_true"][0])


def main():
    x = 4 * np.array(U_AVGS) / RADIUS
    own = np.array([u not in DATA_UAVG for u in U_AVGS])  # data run at its own U_avg

    style()
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for alpha, c in zip(ALPHAS, COLORS):
        sr, lo, hi, true = np.array([read_result(u, alpha) for u in U_AVGS]).T
        print(f"alpha = {alpha:<5}  E[S_R] = {np.round(sr, 3)}  S_R_true = {np.round(true, 3)}")

        ax.plot(x[own], sr[own], "-", color=c, lw=1.8)
        ax.plot(x[own], true[own], "--", color=c, lw=1.8)
        for i in np.flatnonzero(~own):  # faded segment to the point with reused data
            ax.plot(x[i - 1:i + 1], sr[i - 1:i + 1], "-", color=c, lw=1.8, alpha=0.35)
            ax.plot(x[i - 1:i + 1], true[i - 1:i + 1], "--", color=c, lw=1.8, alpha=0.35)
        ax.errorbar(x, sr, yerr=[sr - lo, hi - sr], fmt="none", ecolor=c, capsize=5, elinewidth=1.5)
        ax.plot(x[own], sr[own], "o", color=c, ms=6, mec="k", mew=0.6)
        ax.plot(x[~own], sr[~own], "o", color=c, ms=9, mfc="white", mew=1.8)
        ax.plot(x, true, "x", color=c, ms=10, mew=2.5)

    for u, u_data in DATA_UAVG.items():
        ax.text(4 * u / RADIUS, 0.25, f"data run at\n$U_{{avg}} = {u_data:g}$", ha="center", va="bottom",
                color="0.45", fontsize=15)

    handles = [Line2D([], [], marker="o", ls="none", color=c, label=rf"$\alpha = {a:g}$")
               for a, c in zip(ALPHAS, COLORS)]
    handles += [Line2D([], [], color="0.35", marker="o", ms=6, mec="k", mew=0.6, lw=1.8,
                       label=r"inferred (90\% CI)"),
                Line2D([], [], color="0.35", marker="x", ms=10, mew=2.5, ls="--", lw=1.8, label="true (FEM)")]
    ax.legend(handles=handles, loc="upper center", ncol=3, fontsize=17, columnspacing=1.0, handletextpad=0.4)

    ax.set_title(rf"$\lambda_{{\mathrm{{true}}}} = {LAMBDA_TRUE:g},\ \beta_{{\mathrm{{true}}}} = {BETA_TRUE:g}$, "
                 "Oldroyd-B ROM with model bias", fontsize=20)
    ax.set(xlabel=r"$\dot{\gamma}_w = 4U_{avg}/R$", ylabel=r"$N_1/\tau_w$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x])
    ax.set_xlim(0, x[-1] + 0.2)
    ax.set_ylim(0, 4.2)
    ax.grid(alpha=0.3)

    out = HERE / f"stress_ratio_vs_gammadot_lam_{LAMBDA_TRUE:g}_beta_{BETA_TRUE:g}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)
    plt.show()


if __name__ == "__main__":
    main()
