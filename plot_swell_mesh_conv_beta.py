"""Mesh convergence of the maximum swell ratio vs Wi, Oldroyd-B, beta = 0.3, 0.5 and 0.7 (one figure each).

datas/mesh-conv/beta-<b>/wi-<Wi>/swell_summary.txt lists, per mesh level M0 (coarsest) ... M4 (finest),
dx_box, dx_wall, the node count, max_height and the run status ('steady' or 'completed'; all runs are
drawn alike). The maximum swell ratio is
max_height / R (R = 1); one line per mesh level over Wi = 0.5, 2.5, 5. The (beta, Wi) cases in SKIP
are left out; a summary file that is an exact copy of another Wi's file of the same beta is reported.

Run: python plot_swell_mesh_conv_beta.py
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
RADIUS = 1.0
# M0 ... M4: orange, aqua, blue, violet, black (finest = reference); every pair distinct for normal and
# colour-blind vision
COLORS = ["#eb6834", "#1baf7a", "#2a78d6", "#4a3aa7", "#000000"]
SKIP = set()  # (beta, Wi) cases to leave out


def read_summary(path):
    """{mesh: dict(dx_box, dx_wall, nnodes, max_height, steady)} from one swell_summary.txt."""
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            mesh, dx_box, dx_wall, _, _, nnodes, _, _, hmax, _, _, status = line.split()
            rows[mesh] = dict(dx_box=float(dx_box), dx_wall=float(dx_wall), nnodes=int(nnodes),
                              max_height=float(hmax), steady=status == "steady")
    return rows


def load(beta):
    """{Wi: rows} for one beta, without the SKIP cases; warns about identical summary files."""
    data, texts = {}, {}
    for wi in WIS:
        f = ROOT / f"beta-{beta:g}" / f"wi-{wi:g}" / "swell_summary.txt"
        text = f.read_text().strip()
        twin = next((w for w, t in texts.items() if t == text), None)
        if twin is not None:
            print(f"WARNING: {f.relative_to(HERE)} is identical to the Wi = {twin:g} file")
        texts[wi] = text
        if (beta, wi) in SKIP:
            print(f"skipped beta = {beta:g}, Wi = {wi:g} (SKIP)")
            continue
        data[wi] = read_summary(f)
    return data


def main():
    style()
    for beta in BETAS:
        data = load(beta)
        wis = np.array(sorted(data))
        meshes = list(data[wis[0]])
        colors = COLORS[:len(meshes)]

        fig, ax = plt.subplots(figsize=(9, 6.5))
        print(f"\nbeta = {beta:g}: maximum swell ratio (rel. difference to {meshes[-1]})")
        for mesh, c in zip(meshes, colors):
            sr = np.array([data[w][mesh]["max_height"] / RADIUS for w in wis])
            ref = np.array([data[w][meshes[-1]]["max_height"] / RADIUS for w in wis])
            print(f"  {mesh}: {np.round(sr, 4)}  ({np.round(100 * (sr / ref - 1), 2)} %)")
            ax.plot(wis, sr, "-o", color=c, lw=3, ms=10, mec="k", mew=0.8)

        handles = [Line2D([], [], color=c, lw=3, marker="o", ms=10, mec="k", mew=0.8, label=m)
                   for m, c in zip(meshes, colors)]
        ax.legend(handles=handles, loc="upper left", fontsize=26, markerscale=1.2, handletextpad=0.4)
        ax.set(xlabel=r"$Wi = \lambda \dot{\gamma}_w$", ylabel=r"$h_{\max}/R$")
        ax.set_xticks(WIS)
        ax.set_xticklabels([f"{w:g}" for w in WIS])
        ax.set_xlim(0, 5.5)
        ax.set_ylim(1.0, 2.6)
        ax.grid(alpha=0.3)

        out = HERE / f"swell_mesh_conv_beta_{beta:g}.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("saved", out)
    plt.show()


if __name__ == "__main__":
    main()
