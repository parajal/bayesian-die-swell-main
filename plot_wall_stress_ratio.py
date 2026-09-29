"""Stress ratio N1/tau_xy at the die wall vs Wi = lambda * gamma_dot_wall, Giesekus FEM runs.

For every folder in datas/infer-oldroyd-rom1 with beta = BETA and alpha in ALPHAS, gammadot
and N1 are read from the simulation's curve5_wall.txt (columns x, y, pressure, c_xy,
gammadot, c_xx, c_yy, N1) and averaged over the fully developed part of the die wall
(x <= X_DEVELOPED, away from the die exit at x = 0). The wall shear stress comes from the
pressure gradient dp/dx in pressure_drop.txt. With G = eta_p / lambda:

    Wi     = lambda * gammadot_wall
    tau_xy = |dp/dx| R / 2                       (force balance in the die, R = 1)
    N1     = G (c_xx - c_yy)                     (solvent adds no N1 in shear)

For reference, the wall stress from the stress field, G |c_xy| + eta_s gammadot_wall, is printed
next to the pressure-based value.

The analytical steady-simple-shear Giesekus ratio (Bird et al.) is drawn as a line for
comparison. Two figures are saved: linear axes and log-log axes.

Run: python plot_wall_stress_ratio.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_n1_vs_wi import read_input, style

DATA_DIR = Path(__file__).resolve().parent / "datas" / "infer-oldroyd-rom1"
BETA = 0.5          # solvent fraction eta_s / eta_0
ALPHAS = (0.01, 0.05, 0.1, 0.2)
X_DEVELOPED = -1.5  # average wall values over x <= X_DEVELOPED (die exit at x = 0)
RADIUS = 1.0        # die radius
WI_MAX = 4.4


def wall_values(folder):
    """Wi, N1/tau_xy, tau_xy from dp/dx and tau_xy from the stress field for one run."""
    p = read_input(folder)
    eta_s, eta_p = p["betav"] * p["eta_0"], (1 - p["betav"]) * p["eta_0"]
    G = eta_p / p["lambda"]
    x, _, _, cxy, gd, cxx, cyy, _ = np.loadtxt(folder / "curve5_wall.txt").T
    m = x <= X_DEVELOPED
    gd, cxy, n1 = gd[m].mean(), np.abs(cxy[m]).mean(), (G * (cxx[m] - cyy[m])).mean()
    tau_w = abs(np.loadtxt(folder / "pressure_drop.txt").item()) * RADIUS / 2
    wi = 0.4 * p["lambda"]  # nominal Wi = lambda * gamma_dot_nominal (4 U_avg / R = 0.4)
    return wi, n1 / tau_w, tau_w, G * cxy + eta_s * gd


def giesekus_ratio(wi, alpha, beta):
    """Analytical N1/tau_xy in steady simple shear (Giesekus + Newtonian solvent), eta_0 = lambda = 1."""
    chi = np.sqrt((np.sqrt(1 + 16 * alpha * (1 - alpha) * wi**2) - 1) / (8 * alpha * (1 - alpha) * wi**2))
    f = (1 - chi) / (1 + (1 - 2 * alpha) * chi)
    eta = (1 - beta) * (1 - f)**2 / (1 + (1 - 2 * alpha) * f) + beta
    psi1 = 2 * (1 - beta) * f * (1 - alpha * f) / (wi**2 * alpha * (1 - f))
    return psi1 * wi / eta


def runs(alpha):
    """Folders with beta = BETA and this alpha, sorted by lambda."""
    found = [(read_input(f), f) for f in DATA_DIR.glob("giesekus_lam_*_beta_*_alpha_*")]
    found = [(p, f) for p, f in found if np.isclose(p["betav"], BETA) and np.isclose(p["mobility"], alpha)]
    return [f for p, f in sorted(found, key=lambda t: t[0]["lambda"])]


def main():
    series = {}
    print(f"beta = {BETA}")
    for alpha in ALPHAS:
        wi, s, tau_dp, tau_field = np.array([wall_values(f) for f in runs(alpha)]).T
        series[alpha] = (wi, s)
        err = np.abs(s / giesekus_ratio(wi, alpha, BETA) - 1).max()
        print(f"alpha = {alpha:<5}  Wi = {np.round(wi, 3)}  N1/tau_xy = {np.round(s, 3)}  "
              f"max rel. diff. to analytical = {err:.1e}")
        print(f"{'':15}tau_xy = |dp/dx|R/2 = {np.round(tau_dp, 4)}")
        print(f"{'':15}G|c_xy| + eta_s gd = {np.round(tau_field, 4)}")

    style()
    wi_line = np.geomspace(0.1, WI_MAX, 200)
    for scale, suffix in (("linear", ""), ("log", "_loglog")):
        fig, ax = plt.subplots(figsize=(8, 6))
        for k, (alpha, (wi, s)) in enumerate(series.items()):
            keep = wi <= WI_MAX
            ax.plot(wi_line, giesekus_ratio(wi_line, alpha, BETA), "-", color=f"C{k}", lw=1.2, alpha=0.6)
            ax.plot(wi[keep], s[keep], "o", color=f"C{k}", label=rf"$\alpha = {alpha:g}$")
        ax.set(xlabel=r"$Wi = \lambda \dot{\gamma}_w$", ylabel=r"$N_1/\tau_{xy}$", xscale=scale, yscale=scale)
        ax.set_title("FEM")
        ax.grid(alpha=0.3, which="both")
        if scale == "linear":
            ax.set_xlim(0, WI_MAX)
            ax.set_ylim(bottom=0)
        else:
            ax.set_xlim(0.1, WI_MAX)
        ax.legend(loc="upper left" if scale == "linear" else "lower right")
        out = Path(__file__).resolve().parent / f"wall_stress_ratio_beta_{BETA:g}{suffix}.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("saved", out)
    plt.show()


if __name__ == "__main__":
    main()
