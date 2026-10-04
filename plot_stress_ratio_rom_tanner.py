"""Wall stress ratio N1/tau_w vs wall shear rate: simple shear, Oldroyd-B ROM vs FEM (with and without model
bias), and Tanner (four figures).

For the Giesekus runs at lambda_true = 5, beta_true = 0.5 and alpha = 0.01, 0.05, 0.1, 0.2:
  * simple shear: analytical steady-simple-shear Giesekus ratio N1/tau_xy at Wi = lambda_true * gammadot_w
    (giesekus_ratio in plot_wall_stress_ratio.py);
  * Oldroyd-B ROM with model bias: E[S_R] and its 90 % credible interval, with the true (FEM)
    S_R_true = N1_true / tau_w, from
    rom-oldroyd-bias-results/bias_true_infer-<U_avg>_giesekus_lam_5_beta_0p5_alpha_<a>.txt;
  * Oldroyd-B ROM without model bias: the same, from
    rom-oldroyd-nobias-results/bias_false_infer-<U_avg>_giesekus_lam_5_beta_0p5_alpha_<a>.txt;
  * Tanner: E[S_R] = 2 E[N1/(2 tau_w)] and its 90 % credible interval from
    rom-tanner-results/tanner_bias_false_infer-<U_avg>_giesekus_lam_5_beta_0p5_alpha_<a>.txt.
All are plotted against the nominal wall shear rate gammadot_w = 4 U_avg / R.

Run: python plot_stress_ratio_rom_tanner.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from plot_n1_vs_wi import style
from plot_wall_stress_ratio import giesekus_ratio

HERE = Path(__file__).resolve().parent
ROM_DIR, TANNER_DIR = HERE / "rom-oldroyd-bias-results", HERE / "rom-tanner-results"
NOBIAS_DIR = HERE / "rom-oldroyd-nobias-results"
LAMBDA_TRUE, BETA_TRUE = 5, 0.5
ALPHAS = (0.01, 0.05, 0.1, 0.2)
U_AVGS = (0.05, 0.1, 0.2)
RADIUS = 1.0
# blue, orange, aqua, violet: distinct for normal and colour-blind vision (OKLab distance >= 16 / >= 9 for every pair)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
X = 4 * np.array(U_AVGS) / RADIUS


def tag(v):
    return f"{v:g}".replace(".", "p")


def read_result(path):
    """(E[S_R], 90 % CI low, high, S_R_true) from one inference results file."""
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, *vals = line.split("#")[0].split()
            rows[name] = [float(v) for v in vals]
    return (*rows["E[S_R]"], rows["S_R_true"][0])


def series(prefix, folder, alpha):
    """E[S_R], CI low, CI high and S_R_true over U_AVGS for one alpha."""
    name = "giesekus_lam_{}_beta_{}_alpha_{}.txt".format(tag(LAMBDA_TRUE), tag(BETA_TRUE), tag(alpha))
    return np.array([read_result(folder / f"{prefix}infer-{u:g}_{name}") for u in U_AVGS]).T


def new_axes(ylim, loglog, ylabel=r"$N_1/\tau_w$"):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set(xlabel=r"$\dot{\gamma}_w = 4U_{avg}/R$", ylabel=ylabel)
    if loglog:  # decades labelled 10^n, unlabelled minor ticks in between
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1, 1.0)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_xlim(0.1, X[-1] + 0.1)
        ax.set_xticks(X)
        ax.set_xticklabels([f"{v:g}" for v in X])
        ax.minorticks_off()
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    return fig, ax


def legend(ax, extra=()):
    handles = [Line2D([], [], marker="o", ls="none", color=c, label=rf"$\alpha = {a:g}$")
               for a, c in zip(ALPHAS, COLORS)] + list(extra)
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=28, markerscale=1.8,
              columnspacing=1.0, handletextpad=0.4)


def save(fig, name, loglog):
    out = HERE / f"{name}_lam_{LAMBDA_TRUE:g}_beta_{BETA_TRUE:g}{'_loglog' if loglog else ''}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)


def main(loglog=False):
    """The four figures on linear axes, or on log-log axes (files *_loglog.pdf)."""
    style()
    errbar = dict(capsize=11, capthick=2.5, elinewidth=2.5, lw=3, ms=10, mec="k", mew=0.8)
    ylim = lambda linear_max, log_range: log_range if loglog else (0, linear_max)

    fig_s, ax_s = new_axes(ylim(5.5, (0.1, 100)), loglog, r"$N_1/\tau_{xy}$")
    fig_r, ax_r = new_axes(ylim(5.5, (0.1, 100)), loglog)
    fig_n, ax_n = new_axes(ylim(5.5, (0.1, 100)), loglog)
    fig_t, ax_t = new_axes(ylim(10.5, (0.1, 100)), loglog)
    for alpha, c in zip(ALPHAS, COLORS):
        shear = giesekus_ratio(LAMBDA_TRUE * X, alpha, BETA_TRUE)
        rom, rom_lo, rom_hi, true = series("bias_true_", ROM_DIR, alpha)
        nob, nob_lo, nob_hi, _ = series("bias_false_", NOBIAS_DIR, alpha)
        tan, tan_lo, tan_hi, _ = series("tanner_bias_false_", TANNER_DIR, alpha)
        print(f"alpha = {alpha:<5}  simple shear = {np.round(shear, 3)}  true = {np.round(true, 3)}"
              f"  ROM = {np.round(rom, 3)}  ROM no bias = {np.round(nob, 3)}  Tanner = {np.round(tan, 3)}")

        ax_s.plot(X, shear, "-o", color=c, lw=3, ms=10, mec="k", mew=0.8)
        for ax, (m, lo, hi) in ((ax_r, (rom, rom_lo, rom_hi)), (ax_n, (nob, nob_lo, nob_hi))):
            ax.errorbar(X, m, yerr=[m - lo, hi - m], fmt="-", color=c, **errbar)
            ax.plot(X, true, "--x", color=c, lw=3, ms=13, mew=3)
        ax_t.errorbar(X, tan, yerr=[tan - tan_lo, tan_hi - tan], fmt="-^", color=c, **errbar)

    true_handle = Line2D([], [], color="0.35", marker="x", ms=13, mew=3, ls="--", lw=3, label="true (FEM)")
    legend(ax_s)
    legend(ax_r, [true_handle])
    legend(ax_n, [true_handle])
    legend(ax_t)

    save(fig_s, "stress_ratio_simple_shear", loglog)
    save(fig_r, "stress_ratio_rom", loglog)
    save(fig_n, "stress_ratio_rom_nobias", loglog)
    save(fig_t, "stress_ratio_tanner", loglog)
    return fig_s, fig_r, fig_n, fig_t


if __name__ == "__main__":
    main(loglog=False)
    main(loglog=True)
    plt.show()
