"""Inferred wall stress ratio N1/tau_w vs Wi for ONE Giesekus material at several flow rates.

The material is fixed (lambda = LAMBDA = 1, beta = 0.5, alpha = 0.01 ... 0.2) and Wi = lambda * gammadot_w
is varied through the wall shear rate gammadot_w = 4 U_avg / R, i.e. through the flow rate. At each rate the
Oldroyd-B ROM infers (lambda, beta); its wall stress ratio is N1/tau_w = 2 (1 - beta) lambda gammadot_w.

In creeping flow without surface tension the swell depends on lambda and U_avg only through Wi, so the
Giesekus run at (lambda = 1, U_avg) is the run at (lambda = 10 U_avg, U_avg = 0.1) of datas/infer-oldroyd-rom1.
The posteriors in datas/infer-oldroyd-rom1/n1_tau_inferred_oldroyd.csv (Oldroyd-B ROM trained at U_avg = 0.1)
therefore hold for every rate: N1/tau_w is dimensionless and unchanged, while the inferred lambda is referred
to the actual rate, lambda(gammadot_w) = lambda_ROM * GAMMADOT_TRAIN / gammadot_w.

Figures (points: posterior mean with 95 % credible interval; the four alphas are shifted slightly apart):
  * n1_tau_inferred_vs_wi_beta_0.5[_loglog].pdf : N1/tau vs Wi, lines = analytical Giesekus (the truth);
  * lambda_inferred_vs_gammadot_beta_0.5.pdf    : inferred Oldroyd-B lambda vs gammadot_w, line = lambda.

Run: python plot_n1_tau_inferred_vs_wi.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_n1_vs_wi import style
from plot_wall_stress_ratio import giesekus_ratio

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "datas" / "infer-oldroyd-rom1" / "n1_tau_inferred_oldroyd.csv"
LAMBDA = 1.0            # relaxation time of the Giesekus material
RADIUS = 1.0
GAMMADOT_TRAIN = 0.4    # 4 U_avg / R of the ROM training runs (U_avg = 0.1)
BETA = 0.5
ALPHAS = (0.01, 0.05, 0.1, 0.2)
WI_MAX = 4.4


def load():
    """alpha -> dict of gammadot_w, N1/tau (mean, lo, hi) and lambda at that rate (mean, lo, hi)."""
    d = np.genfromtxt(RESULTS, delimiter=",", names=True)
    out = {}
    for a in ALPHAS:
        r = np.sort(d[np.isclose(d["alpha"], a)], order="Wi_true")
        gd = r["Wi_true"] / LAMBDA
        s = GAMMADOT_TRAIN / gd
        out[a] = {"gd": gd, "sr": (r["Sr_mean"], r["Sr_lo"], r["Sr_hi"]),
                  "lam": (r["lambda_mean"] * s, r["lambda_lo"] * s, r["lambda_hi"] * s)}
    return out


def points(ax, x, y, k, alpha, dodge):
    """Posterior means with 95 % CI error bars, shifted by `dodge` so the alphas do not overlap."""
    mean, lo, hi = y
    ax.errorbar(dodge(x, k - (len(ALPHAS) - 1) / 2), mean, yerr=[mean - lo, hi - mean], fmt="o",
                color=f"C{k}", ms=7, capsize=3, elinewidth=1.2, label=rf"$\alpha = {alpha:g}$")


def uavg_axis(ax):
    """Top axis with the flow rate U_avg = gammadot_w R / 4 = Wi R / (4 lambda)."""
    top = ax.secondary_xaxis("top", functions=(lambda wi: wi * RADIUS / (4 * LAMBDA),
                                               lambda u: 4 * LAMBDA * u / RADIUS))
    top.set_xlabel(r"$U_{\mathrm{avg}}$")


def main():
    data = load()
    style()
    wi_line = np.geomspace(0.1, WI_MAX, 200)

    for scale, suffix in (("linear", ""), ("log", "_loglog")):
        dodge = (lambda x, s: x + 0.03 * s) if scale == "linear" else (lambda x, s: x * (1 + 0.025 * s))
        fig, ax = plt.subplots(figsize=(8, 6.5))
        for k, alpha in enumerate(ALPHAS):
            ax.plot(wi_line, giesekus_ratio(wi_line, alpha, BETA), "-", color=f"C{k}", lw=1.2, alpha=0.6)
            points(ax, LAMBDA * data[alpha]["gd"], data[alpha]["sr"], k, alpha, dodge)
        ax.set(xlabel=rf"$Wi = \lambda \dot{{\gamma}}_w$  ($\lambda = {LAMBDA:g}$)", ylabel=r"$N_1/\tau_{xy}$",
               xscale=scale, yscale=scale)
        ax.set_title("Inferred (Oldroyd-B)", pad=12)
        uavg_axis(ax)
        ax.grid(alpha=0.3, which="both")
        if scale == "linear":
            ax.set_xlim(0, WI_MAX + 0.1)
            ax.set_ylim(bottom=0)
        else:
            ax.set_xlim(0.1, WI_MAX + 0.4)
            ax.set_ylim(0.04, 10)  # headroom for the legend above the points
        ax.legend(loc="upper left")
        out = HERE / f"n1_tau_inferred_vs_wi_beta_{BETA:g}{suffix}.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("saved", out)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.axhline(LAMBDA, color="k", ls="--", lw=1.2)
    for k, alpha in enumerate(ALPHAS):
        points(ax, data[alpha]["gd"], data[alpha]["lam"], k, alpha, lambda x, s: x + 0.03 * s)
    ax.set(xlabel=r"$\dot{\gamma}_w$", ylabel=r"inferred $\lambda$")
    ax.set_title(rf"Oldroyd-B fit, Giesekus $\lambda = {LAMBDA:g}$", pad=12)
    ax.set_xlim(0, WI_MAX / LAMBDA + 0.1)
    ax.set_ylim(0, 1.6 * LAMBDA)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize=20)
    out = HERE / f"lambda_inferred_vs_gammadot_beta_{BETA:g}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)
    plt.show()


if __name__ == "__main__":
    main()
