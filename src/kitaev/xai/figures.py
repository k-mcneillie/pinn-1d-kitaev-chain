"""Figures for the interpretability pipeline, in the shared house style.

Four figures, each taking a computed dataclass from elsewhere in
:mod:`kitaev.xai` and returning a :class:`~matplotlib.figure.Figure`.
Passing ``save_path`` also writes it. Styling is reused from
:mod:`kitaev.visualisation.style` so these sit alongside the standard
run figures without a second look-and-feel.

- :func:`plot_transparency_axis` plots accuracy against the number of
  properties a model guarantees by construction.
- :func:`plot_conditioning` plots the residual-operator condition number
  and spectral gap against the chemical potential, per basis.
- :func:`plot_seed_dispersion` plots how far one model's predictions move
  across seeds, against the chemical potential.
- :func:`plot_residual_field` overlays the trained models' per-``mu``
  physics residuals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from kitaev.visualisation.style import (
    CORAL,
    INK,
    SLATE,
    TEAL,
    mark_transition,
    use_house_style,
)

from .conditioning import ConditioningSweep
from .dispersion import SeedDispersion
from .internalisation import InternalisationProfile
from .residual_field import ResidualField

_SERIES_COLOURS = (INK, CORAL, TEAL, SLATE)


def _save(fig: Figure, save_path: str | Path | None, dpi: int) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")


def plot_transparency_axis(
    profiles: Sequence[InternalisationProfile],
    errors: Mapping[str, Mapping[str, float]],
    *,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plot accuracy against the amount of structure internalised.

    Args:
        profiles: One :class:`~kitaev.xai.internalisation.InternalisationProfile`
            per model.
        errors: A mapping from model name to a mapping with at least the
            keys ``"topological"`` and ``"trivial"``, each an energy MAE.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        A two-panel figure, trivial-phase and topological-phase energy
        error against the number of structural guarantees.
    """
    use_house_style()
    ordered = sorted(profiles, key=lambda p: p.n_structural)
    x = [p.n_structural for p in ordered]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, phase, title in (
        (axes[0], "trivial", "Trivial-phase energy MAE"),
        (axes[1], "topological", "Topological-phase energy MAE"),
    ):
        y = [errors[p.name][phase] for p in ordered]
        ax.plot(x, y, color=SLATE, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.scatter(x, y, color=CORAL, s=60, zorder=2)
        for profile, xi, yi in zip(ordered, x, y, strict=True):
            ax.annotate(
                profile.name,
                (xi, yi),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
                color=INK,
            )
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("properties guaranteed by construction")
        ax.set_ylabel("energy MAE")
        ax.set_xticks(sorted(set(x)))

    fig.tight_layout()
    _save(fig, save_path, dpi)
    return fig


def plot_conditioning(
    sweeps: Sequence[ConditioningSweep],
    *,
    hopping: float,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plot the residual-operator condition number and gap against ``mu``.

    Args:
        sweeps: One :class:`~kitaev.xai.conditioning.ConditioningSweep` per
            basis.
        hopping: The hopping amplitude ``t``, for the transition markers.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        A two-panel figure, condition number and smallest gap against
        ``mu``.
    """
    use_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    mu_max = max(float(np.abs(s.mu).max()) for s in sweeps)

    for ax, attr, title in (
        (
            axes[0],
            "condition_number",
            r"Residual condition number $\sigma_{\max}/\sigma_{\min}$",
        ),
        (axes[1], "sigma_min", r"Smallest residual gap $\sigma_{\min}$"),
    ):
        mark_transition(ax, hopping=hopping, mu_max=mu_max, two_sided=True)
        for sweep, colour in zip(sweeps, _SERIES_COLOURS, strict=False):
            ax.plot(
                sweep.mu, getattr(sweep, attr), color=colour, lw=1.8, label=sweep.basis
            )
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(r"$\mu / t$")
        ax.legend()

    fig.tight_layout()
    _save(fig, save_path, dpi)
    return fig


def plot_seed_dispersion(
    dispersion: SeedDispersion,
    *,
    hopping: float,
    model_label: str,
    ylim: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plot how far one model's predictions move across seeds, against ``mu``.

    Args:
        dispersion: The result of
            :func:`kitaev.xai.dispersion.sweep_seed_dispersion`.
        hopping: The hopping amplitude ``t``, for the transition markers.
        model_label: A short name for the model, used in the title.
        ylim: Optional ``(low, high)`` limits for the shared log axis.
            Pass the same tuple to every model's plot so the panels are
            visually comparable; the section's claim is about relative
            magnitude, which auto-scaled axes hide. ``None`` lets
            matplotlib scale to this model's data.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        A single-axis figure with the density, energy and edge-weight
        spreads overlaid on a log scale.
    """
    use_house_style()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    mu_max = float(np.abs(dispersion.mu).max())
    mark_transition(ax, hopping=hopping, mu_max=mu_max, two_sided=True)

    ax.plot(
        dispersion.mu,
        dispersion.density_std_max,
        color=INK,
        lw=1.8,
        label="site density, max over sites",
    )
    ax.plot(
        dispersion.mu,
        dispersion.density_std_mean,
        color=TEAL,
        lw=1.6,
        label="site density, mean over sites",
    )
    ax.plot(
        dispersion.mu,
        dispersion.edge_weight_std,
        color=CORAL,
        lw=1.6,
        ls=(0, (4, 2)),
        label="edge weight",
    )
    ax.plot(
        dispersion.mu,
        dispersion.energy_std,
        color=SLATE,
        lw=1.4,
        ls=(0, (1, 1)),
        label="energy",
    )
    ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(
        f"Cross-seed prediction spread, {model_label} ({dispersion.n_seeds} seeds)"
    )
    ax.set_xlabel(r"$\mu / t$")
    ax.set_ylabel("standard deviation across seeds")
    ax.legend()

    fig.tight_layout()
    _save(fig, save_path, dpi)
    return fig


def plot_residual_field(
    fields: Sequence[ResidualField],
    *,
    hopping: float,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Overlay the models' per-``mu`` physics residuals, median and range.

    Each model contributes a median line over its seeds and a shaded band
    spanning the smallest-to-largest residual across seeds. A single-seed
    field has an invisible band.

    Args:
        fields: One :class:`~kitaev.xai.residual_field.ResidualField` per
            model.
        hopping: The hopping amplitude ``t``, for the transition markers.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        A single-axis figure with one residual curve per model on a log
        scale.
    """
    use_house_style()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    mu_max = max(float(np.abs(f.mu).max()) for f in fields)
    mark_transition(ax, hopping=hopping, mu_max=mu_max, two_sided=True)

    for field, colour in zip(fields, _SERIES_COLOURS, strict=False):
        band_label = (
            field.label
            if field.n_seeds == 1
            else f"{field.label} ({field.n_seeds} seeds)"
        )
        ax.fill_between(
            field.mu,
            field.residual_min,
            field.residual_max,
            color=colour,
            alpha=0.15,
            lw=0,
        )
        ax.plot(field.mu, field.residual_median, color=colour, lw=1.8, label=band_label)
    ax.set_yscale("log")
    ax.set_title("Per-collocation-point physics residual, median over seeds")
    ax.set_xlabel(r"$\mu / t$")
    ax.set_ylabel("residual")
    ax.legend()

    fig.tight_layout()
    _save(fig, save_path, dpi)
    return fig
