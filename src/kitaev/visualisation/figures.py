"""The project's standard figures, in the shared house style.

These are the six figures every experiment notebook produces, lifted out
of the notebooks so a script (e.g. ``experiments/four_model_comparison.py``)
can render the same set with one call:

1. :func:`plot_loss_history` -- total train/val loss and its components.
2. :func:`plot_probe_history` -- physical errors vs epoch from the
   :class:`~kitaev.training.probes.BdGEvaluationProbe`.
3. :func:`plot_energy_sweep` -- exact vs model ``E(mu)`` with an error
   inset.
4. :func:`plot_eigenvector_agreement` -- near-zero subspace fidelity and
   combined edge weight vs ``mu``.
5. :func:`plot_wavefunction_grid` -- particle/hole density profiles at
   chosen ``mu`` values, model vs exact, with the ``rho / 2`` manifold
   density where it is defined.
6. :func:`plot_mu_reflection` -- ``|E(+mu)|`` vs ``|E(-mu)|``.

Each takes plain data (a :class:`~kitaev.training.utils.TrainingHistory` or
one of the sweep dataclasses from :mod:`kitaev.visualisation.evaluation`)
and returns a :class:`~matplotlib.figure.Figure`; passing ``save_path``
also writes it. :func:`kitaev.visualisation.save_run_figures` wires all six
together for a finished run.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from kitaev.training.utils import TrainingHistory

from .evaluation import MuReflectionSweep, SpectralSweep, WavefunctionSweep
from .style import (
    CORAL,
    GOLD,
    INK,
    SLATE,
    TEAL,
    annotate_phases,
    mark_phase_split,
    mark_transition,
    use_house_style,
)

_COMPONENT_COLOURS = (INK, TEAL, GOLD, CORAL, SLATE)


def _save(fig: Figure, save_path: str | Path | None, dpi: int) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")


def plot_loss_history(
    history: TrainingHistory,
    *,
    component_keys: Sequence[str],
    weight_key: str | None = None,
    split_epoch: int | None = None,
    total_title: str = "Total loss",
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plot total train/val loss and the individual loss components.

    Args:
        history: A populated history containing ``"train_loss"`` (and,
            if validation ran, ``"val_loss"``) plus one ``"train_{key}"``
            series per entry in ``component_keys``.
        component_keys: Loss-component names to draw on the right panel,
            e.g. ``["fsm", "var"]`` or ``["e", "psi", "res", "ph"]``.
            Missing series are skipped rather than raising.
        weight_key: Optional annealing-weight series (without the
            ``"train_"`` prefix), e.g. ``"pin_wt"`` or ``"physics_wt"``,
            drawn on a secondary right-hand axis.
        split_epoch: If given, the AdamW -> L-BFGS hand-over epoch, marked
            on both panels.
        total_title: Title for the left panel.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        The two-panel figure (total loss; components).
    """
    use_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], color=INK, lw=1.8, label="train")
    if "val_loss" in history:
        axes[0].plot(
            epochs,
            history["val_loss"],
            color=CORAL,
            lw=1.4,
            ls=(0, (3, 2)),
            label="val",
        )
    axes[0].set_yscale("log")
    axes[0].set_title(total_title)
    axes[0].set_xlabel("epoch")
    if split_epoch is not None:
        mark_phase_split(axes[0], split_epoch)
    axes[0].legend()

    for key, colour in zip(
        component_keys, itertools.cycle(_COMPONENT_COLOURS), strict=False
    ):
        series_key = f"train_{key}"
        if series_key not in history:
            continue
        series = history[series_key]
        axes[1].plot(range(1, len(series) + 1), series, color=colour, lw=1.6, label=key)
    axes[1].set_yscale("log")
    axes[1].set_title("Components")
    axes[1].set_xlabel("epoch")

    if weight_key is not None and f"train_{weight_key}" in history:
        weight_axis = axes[1].twinx()
        series = history[f"train_{weight_key}"]
        weight_axis.plot(
            range(1, len(series) + 1), series, color=SLATE, lw=1.0, ls=(0, (1, 1))
        )
        weight_axis.set_ylabel(weight_key, color=SLATE)
        weight_axis.set_ylim(-0.05, 1.05)
        weight_axis.tick_params(axis="y", colors=SLATE)
        weight_axis.grid(False)

    if split_epoch is not None:
        mark_phase_split(axes[1], split_epoch)
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    _save(fig, save_path, dpi)
    return fig


def plot_probe_history(
    history: TrainingHistory,
    *,
    split_epoch: int | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plot the probe's physical errors against epoch.

    Args:
        history: A history populated by
            :class:`~kitaev.training.probes.BdGEvaluationProbe`, i.e.
            carrying ``probe_epoch`` and the ``probe_*`` error series.
        split_epoch: If given, the AdamW -> L-BFGS hand-over epoch.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        The two-panel figure (energy / edge-weight error; subspace
        infidelity).
    """
    use_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    probe_epoch = np.asarray(history["probe_epoch"])

    axes[0].plot(
        probe_epoch,
        history["probe_e_mae_topological"],
        color=TEAL,
        lw=1.8,
        marker="o",
        ms=3,
        label="energy MAE, topological",
    )
    axes[0].plot(
        probe_epoch,
        history["probe_e_mae_trivial"],
        color=SLATE,
        lw=1.8,
        marker="o",
        ms=3,
        label="energy MAE, trivial",
    )
    axes[0].plot(
        probe_epoch,
        history["probe_edge_mae"],
        color=CORAL,
        lw=1.8,
        marker="s",
        ms=3,
        label="edge-weight MAE",
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Energy and edge-weight error vs epoch")
    axes[0].set_xlabel("epoch")
    if split_epoch is not None:
        mark_phase_split(axes[0], split_epoch)
    axes[0].legend()

    axes[1].plot(
        probe_epoch,
        history["probe_subspace_infidelity"],
        color=INK,
        lw=1.8,
        marker="o",
        ms=3,
        label="mean",
    )
    axes[1].plot(
        probe_epoch,
        history["probe_subspace_infidelity_max"],
        color=CORAL,
        lw=1.4,
        ls=(0, (3, 2)),
        marker="s",
        ms=3,
        label="max",
    )
    axes[1].set_yscale("log")
    axes[1].set_title(r"Eigenvector subspace infidelity $1 - \|P\,\psi_{\rm pred}\|$")
    axes[1].set_xlabel("epoch")
    if split_epoch is not None:
        mark_phase_split(axes[1], split_epoch)
    axes[1].legend()

    fig.tight_layout()
    _save(fig, save_path, dpi)
    return fig


def plot_energy_sweep(
    sweep: SpectralSweep,
    *,
    hopping: float,
    model_label: str,
    two_sided: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plot exact vs model ``E(mu)`` with a log-scale absolute-error inset.

    Args:
        sweep: The result of
            :func:`kitaev.visualisation.evaluation.sweep_spectrum`.
        hopping: The hopping amplitude ``t``, for the transition markers.
        model_label: Legend label for the predicted curve, e.g.
            ``"chiral PINN"``.
        two_sided: Passed through to
            :func:`kitaev.visualisation.style.mark_transition` /
            :func:`~kitaev.visualisation.style.annotate_phases`.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        The single-axis figure with an error inset.
    """
    use_house_style()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    mu_max = float(np.abs(sweep.mu).max())
    mark_transition(ax, hopping=hopping, mu_max=mu_max, two_sided=two_sided)

    ax.plot(sweep.mu, sweep.energy_exact, color=INK, lw=2.2, label="exact")
    ax.plot(
        sweep.mu,
        sweep.energy_pred,
        color=CORAL,
        lw=1.8,
        ls=(0, (4, 2)),
        label=model_label,
    )
    ax.fill_between(
        sweep.mu, sweep.energy_exact, sweep.energy_pred, color=CORAL, alpha=0.12, lw=0
    )
    ax.set_title(r"Lowest non-negative eigenvalue $E(\mu)$")
    ax.set_xlabel(r"$\mu / t$")
    ax.set_ylabel(r"$E$")
    annotate_phases(ax, hopping=hopping, y=ax.get_ylim()[1] * 0.88, two_sided=two_sided)
    ax.legend(loc="upper left")

    inset = ax.inset_axes((0.6, 0.16, 0.36, 0.42))
    inset.plot(sweep.mu, sweep.abs_error, color=SLATE, lw=1.3)
    for boundary in (-2 * hopping, 2 * hopping) if two_sided else (2 * hopping,):
        inset.axvline(boundary, color=SLATE, ls=(0, (4, 3)), lw=1.0)
    inset.set_yscale("log")
    inset.set_title(r"$|E_{\rm pred} - E_{\rm exact}|$", fontsize=9, pad=4)
    inset.tick_params(labelsize=8)
    inset.grid(True, alpha=0.4)

    fig.tight_layout()
    _save(fig, save_path, dpi)
    return fig


def plot_eigenvector_agreement(
    sweep: SpectralSweep,
    *,
    hopping: float,
    model_label: str,
    two_sided: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plot near-zero subspace fidelity and combined edge weight vs ``mu``.

    Args:
        sweep: The result of
            :func:`kitaev.visualisation.evaluation.sweep_spectrum`.
        hopping: The hopping amplitude ``t``, for the transition markers.
        model_label: Legend label for the predicted edge-weight curve.
        two_sided: Passed through to
            :func:`kitaev.visualisation.style.mark_transition`.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        The two-panel figure (subspace fidelity; combined edge weight).
    """
    use_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    mu_max = float(np.abs(sweep.mu).max())

    mark_transition(axes[0], hopping=hopping, mu_max=mu_max, two_sided=two_sided)
    axes[0].plot(sweep.mu, sweep.subspace_fidelity, color=INK, lw=2.0)
    axes[0].set_ylim(min(0.95, float(sweep.subspace_fidelity.min()) - 0.01), 1.003)
    axes[0].axhline(1.0, color=SLATE, lw=0.8, ls=":")
    axes[0].set_title(r"subspace fidelity  $\|P\,\psi_{\rm pred}\|$")
    axes[0].set_xlabel(r"$\mu / t$")

    mark_transition(axes[1], hopping=hopping, mu_max=mu_max, two_sided=two_sided)
    axes[1].plot(sweep.mu, sweep.edge_weight_exact, color=INK, lw=2.2, label="exact")
    axes[1].plot(
        sweep.mu,
        sweep.edge_weight_pred,
        color=CORAL,
        lw=1.8,
        ls=(0, (4, 2)),
        label=model_label,
    )
    axes[1].set_title(f"Combined edge weight ({sweep.n_edge_sites} sites / end)")
    axes[1].set_xlabel(r"$\mu / t$")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    _save(fig, save_path, dpi)
    return fig


def plot_wavefunction_grid(
    sweep: WavefunctionSweep,
    *,
    hopping: float,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plot particle/hole density profiles at each probe ``mu``, model vs exact.

    Two rows (particle, hole) by one column per probe ``mu``. Where
    ``sweep.manifold_density`` is defined (topological ``mu``), the
    gauge-invariant ``rho / 2`` is overlaid.

    Args:
        sweep: A :class:`~kitaev.visualisation.evaluation.WavefunctionSweep`,
            typically from
            :func:`kitaev.visualisation.evaluation.sweep_wavefunction_grid`.
        hopping: The hopping amplitude ``t``, used to tag each column as
            topological or trivial.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        The ``(2, len(probe_mus))``-panel figure.
    """
    use_house_style()
    n_probes = len(sweep.probe_mus)
    fig, axes = plt.subplots(
        2, n_probes, figsize=(3.0 * n_probes, 5.4), sharex=True, sharey="row"
    )
    if n_probes == 1:
        axes = axes[:, None]
    sites = sweep.sites

    rows = (
        ("particle", sweep.particle_exact, sweep.particle_pred),
        ("hole", sweep.hole_exact, sweep.hole_pred),
    )
    for col, mu in enumerate(sweep.probe_mus):
        is_topological = abs(mu) < 2 * hopping
        tag_colour = TEAL if is_topological else SLATE
        for row, (label, exact, pred) in enumerate(rows):
            ax = axes[row, col]
            ax.fill_between(sites, exact[col], color=INK, alpha=0.12, lw=0)
            ax.plot(sites, exact[col], color=INK, lw=1.8, label="exact")
            ax.plot(sites, pred[col], color=CORAL, lw=1.5, ls=(0, (3, 2)), label="PINN")
            if (
                sweep.manifold_density is not None
                and not np.isnan(sweep.manifold_density[col, row]).any()
            ):
                ax.plot(
                    sites,
                    sweep.manifold_density[col, row] / 2.0,
                    color=TEAL,
                    lw=1.3,
                    ls=(0, (1, 1)),
                    label=r"$\rho/2$",
                )
            for edge in (int(sites[0]), int(sites[-1])):
                ax.axvline(edge, color=GOLD, lw=1.0, alpha=0.6)
            if row == 0:
                ax.set_title(rf"$\mu = {mu:+.1f}\,t$", color=tag_colour)
            if col == 0:
                ax.set_ylabel(rf"$|\psi_n|^2$  ({label})")
            if row == 1:
                ax.set_xlabel("site $n$")

    axes[0, -1].legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "Particle / hole probability density: model vs exact",
        y=1.02,
        fontsize=13,
        weight="600",
    )
    fig.tight_layout()
    _save(fig, save_path, dpi)
    return fig


def plot_mu_reflection(
    sweep: MuReflectionSweep,
    *,
    hopping: float,
    structural_fold: bool = False,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plot ``|E(+mu)|`` and ``|E(-mu)|`` over ``[0, mu_max]``.

    Args:
        sweep: The result of
            :func:`kitaev.visualisation.evaluation.sweep_mu_reflection`.
        hopping: The hopping amplitude ``t``, for the transition marker.
        structural_fold: When ``True`` the title states evenness in ``mu``
            holds by construction; when ``False`` it is presented as a
            learned property.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        The single-axis figure.
    """
    use_house_style()
    fig, ax = plt.subplots(figsize=(9, 4.0))
    mu_max = float(sweep.mu_half[-1])
    mark_transition(ax, hopping=hopping, mu_max=mu_max, two_sided=False)
    ax.plot(sweep.mu_half, sweep.energy_pos, color=INK, lw=2.0, label=r"$|E(\mu)|$")
    ax.plot(
        sweep.mu_half,
        sweep.energy_neg,
        color=CORAL,
        lw=1.6,
        ls=(0, (4, 2)),
        label=r"$|E(-\mu)|$",
    )
    if structural_fold:
        title = r"Spectrum is even in $\mu$ by construction"
    else:
        title = r"Learned evenness of the spectrum in $\mu$"
    ax.set_title(title)
    ax.set_xlabel(r"$\mu / t$")
    ax.set_ylabel(r"$E$")
    ax.legend(loc="upper left")

    fig.tight_layout()
    _save(fig, save_path, dpi)
    return fig
