"""Shared helpers for the ECCV benchmark-metric explainer pages."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path


# Shared color palette tokens used across all explainers.
BASE_COLORS = {
    "paper": "#F6F1E8",
    "panel": "#FFFDFC",
    "panel_soft": "#FBF7EF",
    "ink": "#131313",
    "muted": "#66615B",
    "line": "#D8D0C4",
}


def configure_rc(colors: dict[str, str] | None = None) -> None:
    """Apply the common matplotlib rcParams for all explainer figures."""
    c = {**BASE_COLORS, **(colors or {})}
    plt.rcParams.update(
        {
            "figure.facecolor": c["paper"],
            "savefig.facecolor": c["paper"],
            "axes.facecolor": c["panel"],
            "font.family": "DejaVu Serif",
            "axes.edgecolor": c["line"],
            "text.color": c["ink"],
            "axes.labelcolor": c["muted"],
            "xtick.color": c["muted"],
            "ytick.color": c["muted"],
        }
    )


def save(fig: plt.Figure, path: str | Path, dpi: int = 240) -> None:
    """Save a figure and close it."""
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    face: str,
    edge: str,
    radius: float = 0.035,
    lw: float = 1.2,
    alpha: float = 1.0,
) -> FancyBboxPatch:
    """Add a rounded-rectangle patch to *ax* and return it."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch
