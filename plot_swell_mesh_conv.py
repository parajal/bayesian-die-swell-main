"""Swell ratio vs Wi for the Oldroyd-B U_avg sweep in mesh_conv, one line per mesh level.

Each mesh level mesh_conv/m<k> holds a sweep_oldroyd_uavg run (lambda = 1, beta = 1/9) with
U_avg = 0.125, 0.25, 0.5, i.e. Wi = lambda * 4 U_avg / R = 0.5, 1, 2. The swell ratio is the
far-downstream radius end_height / R (R = 1), read from each level's swell_summary.txt
(open markers: the run hit numtimesteps before the steady-height stop);
dx_box, dx_wall and the node count come from the runs' input.txt and simulation.out. The mesh
is refined to dx_wall / 4 at the die exit. Tanner's (1970) round-die estimate with the Oldroyd-B
shear ratio N1/tau = 2 (1 - beta) Wi is drawn for reference.

Run: python plot_swell_mesh_conv.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_n1_vs_wi import style

ROOT = Path(__file__).resolve().parent / "mesh_conv"
LEVELS = ("m2", "m3", "m4", "m5", "m6")
BETA = 1 / 9


def namelist(path, key):
    return float(re.search(rf"^\s*{key}\s*=\s*([-+0-9.eEdD]+)", path.read_text(), re.M).group(1))


def level(name):
    """(Wi, swell ratio, steady flags, legend label) of one mesh level's sweep; a run is
    steady if it stopped on the steady-height test before numtimesteps."""
    summary = next(p for p in sorted((ROOT / name).rglob("swell_summary.txt"))
                   if p.parent.name == "sweep_oldroyd_uavg")
    uavg, deltat, wi, step, hmax, hend = np.loadtxt(summary, ndmin=2).T
    run = next(p for p in summary.parent.iterdir() if (p / "input.txt").is_file())
    dx_box, dx_wall = namelist(run / "input.txt", "dx_box"), namelist(run / "input.txt", "dx_wall")
    steady = step < namelist(run / "input.txt", "numtimesteps")
    nnodes = int(re.search(r"mesh nnodes = (\d+)", (run / "simulation.out").read_text()).group(1))
    label = rf"{name}: $\Delta x_{{\mathrm{{exit}}}} = {dx_wall / 4:g}$, {nnodes} nodes"
    print(f"{name}: dx_box = {dx_box:g}  dx_wall = {dx_wall:g}  nnodes = {nnodes}  "
          f"deltat = {deltat}  Wi = {wi}  swell = {np.round(hend, 4)}  steps = {step.astype(int)}")
    return wi, hend, steady, label


def tanner(wi, beta):
    """Tanner (1970) round-die swell 0.13 + [1 + (N1/tau)^2 / 8]^(1/6), N1/tau = 2 (1 - beta) Wi."""
    return 0.13 + (1 + (2 * (1 - beta) * wi) ** 2 / 8) ** (1 / 6)


def main():
    levels = [level(name) for name in LEVELS]
    style()
    fig, ax = plt.subplots(figsize=(9, 7))
    wi_line = np.linspace(0, 2.2, 200)
    ax.plot(wi_line, tanner(wi_line, BETA), "--", color="0.5", lw=1.5, label="Tanner (Oldroyd-B)")
    for wi, sr, steady, label in levels:
        line, = ax.plot(wi, sr, "-", label=label)
        c = line.get_color()
        ax.plot(wi[steady], sr[steady], "o", ms=8, color=c)
        ax.plot(wi[~steady], sr[~steady], "o", ms=8, color=c, mfc="white", mew=1.8)
    if any((~steady).any() for *_, steady, _ in levels):
        ax.plot([], [], "o", ms=8, color="0.3", mfc="white", mew=1.8, label="not steady (step limit)")
    ax.set(xlabel=r"$Wi = \lambda \dot{\gamma}_w$", ylabel=r"$S_r = R_{\mathrm{extr}}/R$")
    ax.set_xlim(0, 2.2)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=17)
    out = Path(__file__).resolve().parent / "swell_vs_wi_mesh_conv.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)
    plt.show()


if __name__ == "__main__":
    main()
