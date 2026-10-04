"""Mesh error of the free surface, e(x) = h_M4(x) - h_M1(x), for beta = 0.3, 0.5, 0.7 and Wi = 0.5, 2.5, 5.

datas/mesh-conv/beta-<b>/wi-<Wi>/m1_curve4_y_*.txt and m4_curve4_y_*.txt hold the steady free-surface
heights on the M1 and M4 meshes (outlet x = L first, die exit h = 1 last) but not their x. The x of
the free-surface nodes is rebuilt from the Gmsh threshold field of mesh.geo: element size dx_wall
within DIST_MIN of the die exit, growing linearly to dx_box at DIST_MAX; the vertices equidistribute
int ds / size and the P2 midpoints lie halfway. This reproduces the stored curve4_x.txt of the M2
mesh (141 points) and of the oldroyd-rom1 mesh (247 points) exactly. Both curves are then
interpolated linearly onto the M2 observation stations (141 points) and subtracted.

Per case, the stations x, h_M1, h_M4 and e are written to mesh_error_m1_m4.txt in its folder, and
one figure per beta shows e(x) for the three Wi.

Run: python plot_mesh_error_curves.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_n1_vs_wi import style

HERE = Path(__file__).resolve().parent
ROOT = HERE / "datas" / "mesh-conv"
BETAS = (0.3, 0.5, 0.7)
WIS = (0.5, 2.5, 5)
L_EXTR, DIST_MIN, DIST_MAX = 5.0, 0.1, 1.6
MESH = {"M1": (0.2, 0.04), "M2": (0.1, 0.02), "M4": (0.025, 0.005)}   # dx_box, dx_wall
N_OBS = 141                                                            # M2 free-surface nodes
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]                             # Wi = 0.5, 2.5, 5


def free_surface_x(dx_box, dx_wall, npts):
    """x of the npts free-surface nodes (outlet first) of a mesh with these sizes."""
    s = np.linspace(0.0, L_EXTR, 400001)
    size = np.interp(s, [0, DIST_MIN, DIST_MAX, L_EXTR], [dx_wall, dx_wall, dx_box, dx_box])
    F = np.concatenate([[0.0], np.cumsum(0.5 * (1 / size[1:] + 1 / size[:-1]) * np.diff(s))])
    n_el = (npts - 1) // 2
    xv = np.interp(np.linspace(0.0, F[-1], n_el + 1), F, s)
    x = np.empty(npts)
    x[0::2], x[1::2] = xv, 0.5 * (xv[1:] + xv[:-1])
    return x[::-1]


def curve(folder, mesh):
    """(x, h) of one mesh's free surface, sorted by x."""
    h = np.loadtxt(next(folder.glob(f"{mesh.lower()}_curve4_y_*.txt"))).ravel()
    x = free_surface_x(*MESH[mesh], h.size)
    order = np.argsort(x)
    return x[order], h[order]


def main():
    x_obs = np.sort(free_surface_x(*MESH["M2"], N_OBS))
    style()
    for beta in BETAS:
        fig, ax = plt.subplots(figsize=(9, 6.5))
        print(f"\nbeta = {beta:g}:  e = h_M4 - h_M1 at the {N_OBS} M2 stations")
        for wi, c in zip(WIS, COLORS):
            folder = ROOT / f"beta-{beta:g}" / f"wi-{wi:g}"
            h1 = np.interp(x_obs, *curve(folder, "M1"))
            h4 = np.interp(x_obs, *curve(folder, "M4"))
            e = h4 - h1
            np.savetxt(folder / "mesh_error_m1_m4.txt", np.column_stack([x_obs, h1, h4, e]),
                       header="x  h_M1  h_M4  e = h_M4 - h_M1  (M1, M4 interpolated to the M2 stations)")
            w = h4.max() - 1.0
            print(f"  Wi = {wi:<4g} max|e| = {np.abs(e).max():.4g} ({100 * np.abs(e).max() / w:.2f}% of disp), "
                  f"RMS = {np.sqrt(np.mean(e ** 2)):.4g}, e(outlet) = {e[-1]:.4g}")
            ax.plot(x_obs, e, "-", color=c, lw=3)

        ax.axhline(0.0, color="black", lw=1)
        ax.legend(handles=[Line2D([], [], color=c, lw=3, label=rf"$Wi = {wi:g}$") for wi, c in zip(WIS, COLORS)],
                  loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=26, frameon=False,
                  handlelength=1.5, columnspacing=1.2)
        ax.set(xlabel=r"$x$", ylabel=r"$h_{M4} - h_{M1}$", xlim=(0, L_EXTR))
        ax.grid(alpha=0.3)
        out = HERE / f"mesh_error_m1_m4_beta_{beta:g}.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("saved", out)
    plt.show()


if __name__ == "__main__":
    main()
