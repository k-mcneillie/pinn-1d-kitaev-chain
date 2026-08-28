"""Shared house style for the project's standard figures.

The four experiment notebooks (semi-supervised dual head, Nambu FSM
baseline, structural Nambu, chiral) each open with an identical block of
``matplotlib`` ``rcParams`` and a small set of axis-decoration helpers
(phase shading, transition markers, the AdamW -> L-BFGS split line). That
block lives here now so a script or a notebook gets the same look from one
import, and :mod:`kitaev.visualisation.figures` builds every standard
figure on top of it.

Nothing here knows about a :class:`~sesh.Session` or a model; the functions
take a bare :class:`~matplotlib.axes.Axes` and draw onto it.
"""

from __future__ import annotations

from typing import Any

import matplotlib.style as mpl_style
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

#: Deep navy, used for the exact / ground-truth curve and primary text.
INK = "#1b2a41"
#: Warm red, used for the model / predicted curve.
CORAL = "#e4572e"
#: Teal, used for the topological phase and manifold-density overlays.
TEAL = "#2a9d8f"
#: Muted grey, used for gridlines, trivial-phase labels and secondary axes.
SLATE = "#8d99ae"
#: Gold, used to mark the chain's two edge sites.
GOLD = "#e9c46a"

#: The palette as a mapping, for callers that would rather look colours up
#: by name than import the module-level constants.
PALETTE = {"ink": INK, "coral": CORAL, "teal": TEAL, "slate": SLATE, "gold": GOLD}

_RC_PARAMS: dict[str, Any] = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": SLATE,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e6e8ec",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "600",
    "axes.titlepad": 12,
    "axes.labelsize": 11,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "font.family": "sans-serif",
    "legend.frameon": False,
    "legend.fontsize": 10,
}


def use_house_style() -> None:
    """Apply the project's shared ``rcParams`` to the global ``matplotlib`` state.

    Idempotent and safe to call more than once (each standard figure
    function calls it). Mutates the global ``matplotlib`` rc state rather
    than returning a context manager, matching how the notebooks have
    always done it.
    """
    mpl_style.use(_RC_PARAMS)


def mark_transition(
    ax: Axes,
    *,
    hopping: float,
    mu_max: float,
    two_sided: bool = True,
    major_tick: float = 1.0,
) -> None:
    """Shade the topological phase and draw the transition line(s) on ``ax``.

    Args:
        ax: The axes to decorate. Assumed to have ``mu`` on its x-axis.
        hopping: The hopping amplitude ``t``; the transition sits at
            ``|mu| = 2t``.
        mu_max: The largest ``|mu|`` on the axis, used to set ``xlim``.
        two_sided: When ``True`` the axis spans ``[-mu_max, mu_max]`` and
            both transitions at ``+-2t`` are drawn; when ``False`` the axis
            spans ``[0, mu_max]`` (the folded half-domain) and only the
            ``+2t`` transition is drawn.
        major_tick: Spacing of the x-axis major ticks, in units of ``t``.
    """
    transition = 2.0 * hopping
    if two_sided:
        ax.axvspan(-transition, transition, color=TEAL, alpha=0.07, lw=0)
        for x in (-transition, transition):
            ax.axvline(x, color=SLATE, ls=(0, (4, 3)), lw=1.1)
        ax.set_xlim(-mu_max, mu_max)
    else:
        ax.axvspan(0.0, transition, color=TEAL, alpha=0.07, lw=0)
        ax.axvline(transition, color=SLATE, ls=(0, (4, 3)), lw=1.1)
        ax.set_xlim(0.0, mu_max)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick))


def annotate_phases(
    ax: Axes,
    *,
    hopping: float,
    y: float,
    two_sided: bool = True,
) -> None:
    """Label the topological and trivial regions of ``ax`` inline.

    Args:
        ax: The axes to annotate.
        hopping: The hopping amplitude ``t``.
        y: The y-coordinate (in data units) to place the labels at.
        two_sided: When ``True`` a single ``topological`` label sits at
            ``mu = 0`` and a ``trivial`` label sits at each of ``+-3t``;
            when ``False`` the ``topological`` label sits at ``mu = t`` and
            a single ``trivial`` label at ``mu = 3t``.
    """
    common: dict[str, Any] = {
        "ha": "center",
        "va": "center",
        "fontsize": 9,
        "weight": "600",
    }
    if two_sided:
        ax.text(0.0, y, "topological", color=TEAL, **common)
        for x in (-3 * hopping, 3 * hopping):
            ax.text(x, y, "trivial", color=SLATE, **common)
    else:
        ax.text(hopping, y, "topological", color=TEAL, **common)
        ax.text(3 * hopping, y, "trivial", color=SLATE, **common)


def mark_phase_split(ax: Axes, split_epoch: int, *, top: bool = True) -> None:
    """Draw the AdamW -> L-BFGS hand-over line at ``split_epoch`` on ``ax``.

    Args:
        ax: A loss- or metric-vs-epoch axes.
        split_epoch: The epoch at which the optimiser switches (the AdamW
            epoch count, i.e. the first L-BFGS epoch minus one).
        top: Anchor the ``L-BFGS`` caption to the top of the axis when
            ``True``, otherwise to the bottom. Call after the series are
            plotted so the axis limits are already final.
    """
    ax.axvline(split_epoch, color=SLATE, ls=(0, (2, 2)), lw=1.0)
    ax.text(
        split_epoch,
        ax.get_ylim()[1] if top else ax.get_ylim()[0],
        "  L-BFGS",
        ha="left",
        va="top" if top else "bottom",
        fontsize=8,
        color=SLATE,
    )
