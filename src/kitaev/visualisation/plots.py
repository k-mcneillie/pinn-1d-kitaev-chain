"""Seaborn-based plots for evaluating a trained Kitaev-chain PINN against exact
diagonalisation, and for inspecting a training run's loss history.

Every function here takes plain data (a :class:`~kitaev.training.utils.TrainingHistory`
or one of the sweep dataclasses from :mod:`kitaev.visualisation.evaluation`) and
returns a ``matplotlib.figure.Figure`` — none of them know about ``Session`` or
where a run's outputs live, so saving a figure to a particular experiment's
directory is left to the caller (typically ``fig.savefig(session.path() / "...")``,
or the ``save_path`` shortcut below).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from kitaev.training.utils import TrainingHistory

from .evaluation import EnergyEdgeWeightSweep, WavefunctionSweep

_EXACT_COLOR = "black"
_MODEL_COLOR = "crimson"
_TRANSITION_COLOR = "grey"
_LEGEND_KWARGS = {"frameon": True, "facecolor": "white", "edgecolor": "none"}


def _save_if_requested(fig: Figure, save_path: str | Path | None, dpi: int) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")


def plot_loss_curves(
    history: TrainingHistory,
    component_keys: Sequence[str],
    *,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plots total train/val loss alongside the individual training-loss components.

    Args:
        history: A trainer's populated ``TrainingHistory``, expected to
            contain a ``"train_loss"`` series (and, if validation was
            used, ``"val_loss"``), plus one ``"train_{key}"`` series per
            entry in ``component_keys``.
        component_keys: The loss components to plot on the right-hand
            panel, e.g. ``["e", "psi", "res", "ph"]`` for
            :class:`~kitaev.training.loss.losses.SemiSupervisedLoss`.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        The two-panel figure (total loss; loss components).
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    epochs = range(1, len(history["train_loss"]) + 1)
    sns.lineplot(
        x=epochs, y=history["train_loss"], color=_EXACT_COLOR, label="train", ax=axes[0]
    )
    if "val_loss" in history:
        sns.lineplot(
            x=epochs,
            y=history["val_loss"],
            color=_MODEL_COLOR,
            label="val",
            ax=axes[0],
        )
    axes[0].set_yscale("log")
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(**_LEGEND_KWARGS)

    palette = sns.color_palette("deep", len(component_keys))
    for key, colour in zip(component_keys, palette, strict=True):
        series = history[f"train_{key}"]
        sns.lineplot(
            x=range(1, len(series) + 1), y=series, color=colour, label=key, ax=axes[1]
        )
    axes[1].set_yscale("log")
    axes[1].set_title("Training loss components")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(**_LEGEND_KWARGS)

    fig.tight_layout()
    _save_if_requested(fig, save_path, dpi)
    return fig


def plot_energy_and_edge_weight(
    sweep: EnergyEdgeWeightSweep,
    hopping: float,
    *,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plots model-vs-exact energy and combined edge weight across a mu sweep.

    Args:
        sweep: The result of
            :func:`kitaev.visualisation.evaluation.sweep_energy_and_edge_weight`.
        hopping: The hopping amplitude ``t``, used to mark the
            topological transitions at ``mu = +/- 2t``.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        The two-panel figure (energy; combined edge weight).
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    transitions = (-2 * hopping, 2 * hopping)

    sns.lineplot(
        x=sweep.mu_sweep,
        y=sweep.energy_exact,
        color=_EXACT_COLOR,
        label="Exact",
        ax=axes[0],
    )
    sns.lineplot(
        x=sweep.mu_sweep,
        y=sweep.energy_pred,
        color=_MODEL_COLOR,
        linestyle="--",
        label="Model",
        ax=axes[0],
    )
    for boundary in transitions:
        axes[0].axvline(boundary, color=_TRANSITION_COLOR, linestyle=":", linewidth=1)
    axes[0].set_title(r"$E(\mu)$: model vs exact")
    axes[0].set_xlabel(r"$\mu$")
    axes[0].set_ylabel("Energy")
    axes[0].legend(**_LEGEND_KWARGS)

    sns.lineplot(
        x=sweep.mu_sweep,
        y=sweep.edge_weight_exact,
        color=_EXACT_COLOR,
        label="Exact",
        ax=axes[1],
    )
    sns.lineplot(
        x=sweep.mu_sweep,
        y=sweep.edge_weight_pred,
        color=_MODEL_COLOR,
        linestyle="--",
        label="Model",
        ax=axes[1],
    )
    for boundary in transitions:
        axes[1].axvline(boundary, color=_TRANSITION_COLOR, linestyle=":", linewidth=1)
    axes[1].set_title("Combined edge weight: model vs exact")
    axes[1].set_xlabel(r"$\mu$")
    axes[1].set_ylabel("Edge weight")
    axes[1].legend(**_LEGEND_KWARGS)

    fig.tight_layout()
    _save_if_requested(fig, save_path, dpi)
    return fig


def plot_wavefunctions(
    sweep: WavefunctionSweep,
    hopping: float,
    *,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plots model-vs-exact particle/hole probability density profiles.

    One row per probed mu, particle sector on the left and hole sector
    on the right, labelled by which topological regime that mu falls
    in (``|mu| < 2t`` is topological, otherwise trivial).

    Args:
        sweep: The result of
            :func:`kitaev.visualisation.evaluation.sweep_wavefunctions`.
        hopping: The hopping amplitude ``t``, used to classify each
            probed mu as topological or trivial.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        The ``(len(probe_mus), 2)``-panel figure.
    """
    sns.set_style("whitegrid")
    n_probes = len(sweep.probe_mus)
    fig, axes = plt.subplots(n_probes, 2, figsize=(11, 3 * n_probes), sharex=True)
    if n_probes == 1:
        axes = axes[None, :]

    for row, mu in enumerate(sweep.probe_mus):
        ax_particle, ax_hole = axes[row]

        sns.lineplot(
            x=sweep.sites,
            y=sweep.particle_exact[row],
            color=_EXACT_COLOR,
            label="Exact",
            ax=ax_particle,
        )
        sns.lineplot(
            x=sweep.sites,
            y=sweep.particle_pred[row],
            color=_MODEL_COLOR,
            linestyle="--",
            label="Model",
            ax=ax_particle,
        )
        sns.lineplot(
            x=sweep.sites,
            y=sweep.hole_exact[row],
            color=_EXACT_COLOR,
            label="Exact",
            ax=ax_hole,
        )
        sns.lineplot(
            x=sweep.sites,
            y=sweep.hole_pred[row],
            color=_MODEL_COLOR,
            linestyle="--",
            label="Model",
            ax=ax_hole,
        )

        regime = "Topological" if abs(mu) < 2 * hopping else "Trivial"
        ax_particle.set_ylabel(r"$|\psi_n|^2$" + f"\n{regime}, " + r"$\mu=$" + f"{mu}")
        if row == 0:
            ax_particle.set_title("Particle (electron) sector")
            ax_hole.set_title("Hole sector")
            ax_particle.legend(**_LEGEND_KWARGS)
            ax_hole.legend(**_LEGEND_KWARGS)
        else:
            # seaborn's lineplot auto-attaches a per-axes legend on every
            # call; only the top row's should stay, so the rest are dropped.
            for ax in (ax_particle, ax_hole):
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
        if row == n_probes - 1:
            ax_particle.set_xlabel("Site index n")
            ax_hole.set_xlabel("Site index n")
        for ax in (ax_particle, ax_hole):
            ax.grid(alpha=0.3)

    fig.tight_layout()
    _save_if_requested(fig, save_path, dpi)
    return fig
