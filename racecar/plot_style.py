"""
One chart style for the whole project: one color per agent (colorblind-safe pair, validated
in the dataviz reference palette), recessive grid and axes, plain sans-serif text.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGENT_COLORS = {"A": "#2a78d6", "B": "#eb6834"}     # categorical slots 1 (blue) and 2 (orange)
SEQUENTIAL_BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # magnitude (e.g. speed)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": BASELINE,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2,
        "legend.frameon": False,
    })
