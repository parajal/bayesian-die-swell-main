"""Inferred wall stress ratio N1/tau_w vs Wi, Oldroyd-B ROM on Giesekus data (oldroyd-rom1).

For the Giesekus runs at beta = 0.5 and alpha = ALPHA (lambda_true = 1, 3, 5, 7, 9, 11) the
Oldroyd-B ROM infers (lambda, beta, eta_0); compute_N1 then gives E[S_R] = E[N1/tau_w] and its
95 % credible interval at the die-wall shear rate gamma_dot_w = 4 U_avg / R = 0.4. Figures:
  * stress_ratio_vs_wi_inferred_alpha_<a>.pdf : with model discrepancy, against
    Wi = lambda_inf gamma_dot_w, with the 95 % CI of (Wi, S_R) drawn as an ellipse, plus the
    analytical Giesekus curve and FEM points;
  * stress_ratio_vs_wi_true_<kind>_alpha_<a>.pdf, kind = bias / no_bias (with / without model
    discrepancy): against Wi = lambda_true gamma_dot_w, with the 95 % CI of S_R as error bars,
    plus the FEM points; no legend.
The FEM N1/tau_w (curve5_wall.txt, pressure_drop.txt) and the analytical curve come from
plot_wall_stress_ratio.py.

Run: python plot_stress_ratio_inferred.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from plot_wall_stress_ratio import BETA, giesekus_ratio, runs, wall_values
from plot_n1_vs_wi import style

ALPHA = 0.2
GAMMA_DOT_W = 0.4  # 4 U_avg / R
WI_MAX = 4.6

# lambda_true: (lambda mean, lambda 95 % CI low, high, E[S_R], S_R 95 % CI low, high),
# from the inference runs in datas/infer-oldroyd-rom1/giesekus_lam_<l>_beta_0p5_alpha_0p2
# (noise 2 % for lambda_true >= 5, 1 % for lambda_true = 1, 3)
POSTERIOR = {
    "bias": {  # with model discrepancy (sigma_bias inferred)
        1:  (0.42701, 0.23348, 0.67245, 0.236474, 0.116992, 0.389024),
        3:  (1.2443, 1.1165, 1.5011, 0.733002, 0.392481, 0.884757),
        5:  (1.7175, 1.5392, 2.1494, 0.993146, 0.631808, 1.1681),
        7:  (2.1907, 1.8787, 2.7461, 1.08358, 0.723248, 1.37028),
        9:  (2.7228, 2.2173, 3.4120, 1.05746, 0.744085, 1.39548),
        11: (3.2000, 2.5910, 4.0326, 1.05390, 0.763252, 1.35678),
    },
    "no_bias": {  # without model discrepancy (sigma_bias=None), same settings otherwise
        1:  (0.39246, 0.35625, 0.42868, 0.224996, 0.203981, 0.246222),  # 30000 steps (10000 unconverged)
        3:  (1.2266, 1.1409, 1.3700, 0.672111, 0.479836, 0.759113),
        5:  (1.6293, 1.5828, 1.7520, 1.10042, 0.955107, 1.16978),
        7:  (1.9701, 1.9164, 2.1074, 1.32793, 1.16488, 1.41146),
        9:  (2.3378, 2.1906, 2.5292, 1.37688, 1.17442, 1.57724),
        11: (2.7864, 2.5913, 3.0213, 1.29573, 1.14371, 1.47561),
    },
}
COLORS = dict(zip(POSTERIOR["bias"], plt.cm.viridis(np.linspace(0, 0.9, len(POSTERIOR["bias"])))))
STAR = dict(marker="*", ms=22, mec="k", mew=0.8, ls="none")
DOT = dict(marker="o", ms=11, mec="k", mew=0.8, ls="none")


def ci_ellipse(x, xlo, xhi, y, ylo, yhi, n=200):
    """Closed curve around (x, y) reaching the (asymmetric) CI bounds on each side."""
    t = np.linspace(0, 2 * np.pi, n)
    c, s = np.cos(t), np.sin(t)
    return np.column_stack([x + np.where(c > 0, xhi - x, x - xlo) * c,
                            y + np.where(s > 0, yhi - y, y - ylo) * s])


def fem():
    """(Wi, N1/tau_w) of the FEM runs at beta = BETA and alpha = ALPHA."""
    wi, s, _, _ = np.array([wall_values(f) for f in runs(ALPHA)]).T
    return wi, s


def finish(ax, xlabel, name, legend=True):
    ax.set(xlabel=xlabel, ylabel=r"$N_1/\tau_w$")
    ax.set_xlim(0, WI_MAX)
    ax.set_ylim(0, 1.6)
    ax.grid(alpha=0.3)
    if legend:
        ax.legend(loc="lower right", ncol=2, fontsize=17, columnspacing=0.8, handletextpad=0.3)
    out = Path(__file__).resolve().parent / f"stress_ratio_vs_wi_{name}_alpha_{ALPHA:g}.pdf"
    ax.figure.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)


def plot_inferred(wi_fem, s_fem):
    """With model discrepancy, against Wi from the inferred lambda: CI ellipses, analytical
    curve and FEM points."""
    fig, ax = plt.subplots(figsize=(9, 7))
    wi_line = np.linspace(1e-3, WI_MAX, 400)
    ax.plot(np.r_[0, wi_line], np.r_[0, giesekus_ratio(wi_line, ALPHA, BETA)], "-", color="0.4",
            lw=1.5, label=rf"Giesekus $\alpha = {ALPHA:g}$ (analytic)")
    ax.plot(wi_fem, s_fem, "s", color="0.3", mfc="none", ms=9, mew=1.5, label="Giesekus FEM")
    for lam, (l, llo, lhi, sr, slo, shi) in POSTERIOR["bias"].items():
        c = COLORS[lam]
        g = GAMMA_DOT_W
        ax.add_patch(Polygon(ci_ellipse(g * l, g * llo, g * lhi, sr, slo, shi),
                             fc=c, ec=c, alpha=0.3, lw=1.2))
        ax.plot(g * l, sr, color=c, label=rf"$\lambda_{{\mathrm{{true}}}} = {lam}$", **STAR)
    finish(ax, r"$Wi = \lambda_{\mathrm{inf}}\,\dot\gamma_w$", "inferred")


def plot_true(wi_fem, s_fem, kind):
    """Against Wi from the true lambda: E[S_R] with its 95 % CI as error bars, and FEM points."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(wi_fem, s_fem, "s", color="0.3", mfc="none", ms=9, mew=1.5)
    for lam, (_, _, _, sr, slo, shi) in POSTERIOR[kind].items():
        ax.errorbar(GAMMA_DOT_W * lam, sr, yerr=[[sr - slo], [shi - sr]], color=COLORS[lam],
                    capsize=5, elinewidth=1.8, **DOT)
    finish(ax, "$Wi$", f"true_{kind}", legend=False)


def main():
    wi_fem, s_fem = fem()
    print(f"alpha = {ALPHA}  FEM Wi = {np.round(wi_fem, 3)}  N1/tau_w = {np.round(s_fem, 3)}")
    for kind, post in POSTERIOR.items():
        for lam, (l, *_, sr, slo, shi) in post.items():
            true = s_fem[np.isclose(wi_fem, GAMMA_DOT_W * lam)]
            print(f"{kind:<8} lambda_true = {lam:<2}  Wi_inf = {GAMMA_DOT_W * l:.3f}  "
                  f"E[S_R] = {sr:.3f} [{slo:.3f}, {shi:.3f}]  FEM S_R = {true[0]:.3f}")
    style()
    plot_inferred(wi_fem, s_fem)
    for kind in POSTERIOR:
        plot_true(wi_fem, s_fem, kind)
    plt.show()


if __name__ == "__main__":
    main()
