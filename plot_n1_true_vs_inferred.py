"""True vs inferred first normal stress difference N1 against Wi (oldroyd-rom1).

For Giesekus data at beta = 0.5 and alpha = 0.01, 0.05, 0.1 and 0.2, the true N1 is
compared with the N1 inferred by the Oldroyd-B ROM, in two figures:
  * n1_true_vs_inferred_no_bias.pdf : inferred without a model-bias term,
  * n1_true_vs_inferred_bias.pdf    : inferred with the constrained model bias (fixed l_bias).
Values are taken from the oldroyd-rom1 result tables (noise = 1 %, thin = 2);
Wi = lambda * 4 U_avg / R = 0.4 lambda.

Run: python plot_n1_true_vs_inferred.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LAMBDAS = np.array([11, 9, 7, 5, 3, 1])
WI = 0.4 * LAMBDAS

# alpha: {"true", "no_bias", "bias"} N1 in the order of LAMBDAS ("bias" = constrained bias, fixed l_bias)
N1 = {
    0.01: {"true":    [1.104, 1.01, 0.880, 0.697, 0.454, 0.158],
           "no_bias": [1.11, 1.01, 0.863, 0.672, 0.440, 0.158],
           "bias":    [1.105, 1.00, 0.859, 0.669, 0.439, 0.169]},
    0.05: {"true":    [0.577, 0.572, 0.552, 0.501, 0.385, 0.155],
           "no_bias": [0.603, 0.576, 0.539, 0.479, 0.402, 0.094],
           "bias":    [0.571, 0.550, 0.511, 0.456, 0.373, 0.073]},
    0.1:  {"true":    [0.405, 0.412, 0.413, 0.396, 0.334, 0.15],
           "no_bias": [0.438, 0.438, 0.433, 0.429, 0.382, 0.06],
           "bias":    [0.3907, 0.389, 0.383, 0.365, 0.355, 0.08]},
    0.2:  {"true":    [0.277, 0.288, 0.298, 0.301, 0.277, 0.145],
           "no_bias": [0.354, 0.388, 0.405, 0.360, 0.238, 0.087],
           "bias":    [0.274, 0.286, 0.299, 0.324, 0.265, 0.089]},
}
FIGURES = {"no_bias": "inferred (no model bias)", "bias": "inferred (with model bias)"}


def style():
    """Serif LaTeX, as in plot_n1_vs_wi.py."""
    plt.rcParams.update({
        "font.size": 30, "text.usetex": True, "font.family": "serif",
        "font.weight": "light", "lines.linewidth": 1.5, "legend.fontsize": 26,
        "axes.labelsize": 30, "legend.loc": "best", "lines.markersize": 8,
        "legend.frameon": True,
    })


def plot(kind):
    """One figure: true N1 (solid, filled) and the `kind` inferred N1 (dashed, open) for every alpha."""
    fig, ax = plt.subplots(figsize=(8, 6))
    order = np.argsort(WI)
    for i, (alpha, n1) in enumerate(N1.items()):
        c = f"C{i + 1}"  # same colours as alpha in plot_n1_vs_wi.py (C0 is Oldroyd-B there)
        ax.plot(WI[order], np.array(n1["true"])[order], "o-", color=c, label=rf"$\alpha = {alpha:g}$")
        ax.plot(WI[order], np.array(n1[kind])[order], "s--", color=c, mfc="none")
    ax.set(xlabel="$Wi$", ylabel=r"$N_1$")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 4.4)
    ax.set_xticks(np.linspace(0, 4.4, 5))
    ax.set_ylim(0, 1.2)  # same range in both figures, so they can be compared

    # both legends beside the axes: inside they cover the alpha = 0.01 curves
    colours = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(colours)
    kinds = [plt.Line2D([], [], color="k", marker="o", ls="-", label="true"),
             plt.Line2D([], [], color="k", marker="s", ls="--", mfc="none", label=FIGURES[kind])]
    ax.legend(handles=kinds, loc="lower left", bbox_to_anchor=(1.02, 0.0))

    out = Path(__file__).resolve().parent / f"n1_true_vs_inferred_{kind}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)


def main():
    style()
    for kind in FIGURES:
        plot(kind)
    plt.show()


if __name__ == "__main__":
    main()
