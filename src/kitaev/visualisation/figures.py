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
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from kitaev.training.utils import TrainingHistory

from .evaluation import (
    LowSpectrumSweep,
    ModelErrorBand,
    MuReflectionSweep,
    SeedDensitySweep,
    SpectralSweep,
    WavefunctionSweep,
)
from .style import (
    CORAL,
    GOLD,
    INK,
    LEVEL_COLOURS,
    SLATE,
    TEAL,
    annotate_phases,
    mark_phase_split,
    mark_transition,
    use_house_style,
)

_COMPONENT_COLOURS = (INK, TEAL, GOLD, CORAL, SLATE)
_SERIES_COLOURS = (INK, CORAL, TEAL, SLATE)
_LEVEL_COLOURS = LEVEL_COLOURS


def _save(fig: Figure, save_path: str | Path | None, dpi: int) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")


def _rolling_median(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Odd-window centred rolling median, edges shrinking to fit.

    A per-``mu`` error sweep wiggles by up to an order of magnitude
    between neighbouring grid points; smoothing the *plotted* median over
    a few points lets the trivial-vs-topological trend read without
    touching the underlying data or the band.
    """
    if window < 2 or y.size < window:
        return y
    half = window // 2
    return np.array(
        [np.median(y[max(0, i - half) : i + half + 1]) for i in range(y.size)]
    )


def plot_loss_history(
    history: TrainingHistory,
    *,
    component_keys: Sequence[str],
    weight_key: str | None = None,
    split_epoch: int | None = None,
    floor_value: float | None = None,
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
        floor_value: If given, a horizontal reference on the components
            panel marking the analytic lower bound the folded-spectrum
            term approaches (see
            :func:`kitaev.visualisation.evaluation.fsm_convergence_floor`).
            Makes "flat sits on the line, so it converged rather than
            stalled" legible.
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
    if floor_value is not None:
        axes[1].axhline(
            floor_value,
            color=SLATE,
            lw=1.2,
            ls=(0, (1, 1)),
            label=r"analytic floor $\langle E_1^2\rangle$",
        )
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

    Two rows (particle, hole) by one column per probe ``mu``. In the
    topological columns (tagged ``rho/2``) both curves are already the
    gauge-invariant pair density -- a single eigenvector of the
    near-degenerate ``+-lambda_1`` pair is an arbitrary, often one-sided,
    member of the doublet -- so no separate overlay is needed.

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
            for edge in (int(sites[0]), int(sites[-1])):
                ax.axvline(edge, color=GOLD, lw=1.0, alpha=0.6)
            if row == 0:
                title = rf"$\mu = {mu:+.1f}\,t$"
                if is_topological:
                    title += r"  ($\rho/2$)"
                ax.set_title(title, color=tag_colour)
            if col == 0:
                ax.set_ylabel(rf"$|\psi_n|^2$  ({label})")
            if row == 1:
                ax.set_xlabel("site $n$")

    axes[0, -1].legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "Particle / hole probability density: model vs exact",
        y=1.02,
        fontsize=13,
        weight="bold",
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


def plot_model_comparison(
    bands: Sequence[ModelErrorBand],
    *,
    hopping: float,
    two_sided: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Overlay every model's energy error vs ``mu`` plus a trivial/topological bar.

    The headline comparison figure: the left panel makes the
    trivial-phase gap between the models legible at a glance, the right
    panel reduces it to two numbers per model with the per-seed spread
    scattered over the bars.

    Args:
        bands: One
            :class:`~kitaev.visualisation.evaluation.ModelErrorBand` per
            model, all on the same ``mu`` grid.
        hopping: The hopping amplitude ``t``, for the transition markers.
        two_sided: Passed to
            :func:`kitaev.visualisation.style.mark_transition`.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        The two-panel figure (error vs ``mu``; MAE bars).
    """
    use_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    colours = list(itertools.islice(itertools.cycle(_SERIES_COLOURS), len(bands)))

    mu_max = max(float(np.abs(b.mu).max()) for b in bands)
    mark_transition(axes[0], hopping=hopping, mu_max=mu_max, two_sided=two_sided)
    floor = min(float(b.abs_error_lo[b.abs_error_lo > 0].min()) for b in bands)
    for band, colour in zip(bands, colours, strict=True):
        axes[0].fill_between(
            band.mu,
            band.abs_error_lo,
            band.abs_error_hi,
            color=colour,
            alpha=0.15,
            lw=0,
        )
        label = f"{band.label} ({band.n_seeds} seeds)"
        axes[0].plot(
            band.mu,
            _rolling_median(band.abs_error_median),
            color=colour,
            lw=1.7,
            label=label,
        )
    axes[0].set_yscale("log")
    # Follow the data down: the chiral model reaches a few 1e-8 in the
    # trivial phase, well below the old fixed 5e-7 clamp, which cropped its
    # median line and IQR band at the axis edge.
    axes[0].set_ylim(bottom=max(floor * 0.5, 1e-8))
    axes[0].set_title(
        r"$|E_{\rm pred} - E_{\rm exact}|(\mu)$: median, IQR band over seeds"
    )
    axes[0].set_xlabel(r"$\mu / t$")
    axes[0].set_ylabel("absolute energy error")
    axes[0].legend(loc="upper center", fontsize=9)

    x = np.arange(len(bands))
    width = 0.38
    rng = np.random.default_rng(0)
    for offset, phase, attr in (
        (-width / 2, "trivial", "mae_trivial"),
        (+width / 2, "topological", "mae_topological"),
    ):
        phase_colour = SLATE if phase == "trivial" else TEAL
        medians = [float(np.median(getattr(b, attr))) for b in bands]
        axes[1].bar(
            x + offset,
            medians,
            width,
            color=phase_colour,
            alpha=0.5,
            edgecolor=phase_colour,
            lw=0.8,
            label=f"{phase} (median)",
        )
        for i, band in enumerate(bands):
            seeds = np.asarray(getattr(band, attr))
            jitter = (rng.random(seeds.size) - 0.5) * width * 0.6
            axes[1].scatter(
                np.full(seeds.size, x[i] + offset) + jitter,
                seeds,
                s=16,
                color=INK,
                zorder=3,
            )
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([b.label for b in bands], rotation=20, ha="right")
    axes[1].set_ylabel("energy MAE")
    axes[1].set_title("Energy MAE by phase (points are seeds)")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    _save(fig, save_path, dpi)
    return fig


def plot_wavefunction_waterfall(
    sweep: WavefunctionSweep,
    *,
    hopping: float,
    model_label: str,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Image the particle and hole densities as model, exact and difference.

    Two rows, particle sector then hole sector; three columns, model then
    exact then ``|model - exact|``. A good Majorana end mode has the two
    rows matching site by site. The continuous form of the discrete probe
    columns, and the natural per-``N`` panel for the neural-operator work.

    In the topological band (``|mu| < 2t``) both the model and the exact
    panels show the gauge-invariant pair density ``rho/2`` rather than a
    single arbitrary eigenvector of the near-degenerate ``+-lambda_1``
    pair (see :func:`~kitaev.visualisation.evaluation.sweep_wavefunctions`).

    Args:
        sweep: A
            :class:`~kitaev.visualisation.evaluation.WavefunctionSweep`
            over a dense ``probe_mus`` grid (e.g. from
            :func:`kitaev.visualisation.evaluation.sweep_wavefunctions`).
        hopping: The hopping amplitude ``t``, for the transition markers.
        model_label: Column title for the predicted panels.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        The ``(2, 3)``-panel figure.
    """
    use_house_style()
    mu = np.asarray(sweep.probe_mus, dtype=float)
    n_sites = int(sweep.sites[-1]) + 1
    extent = (float(mu.min()), float(mu.max()), -0.5, n_sites - 0.5)

    rows = (
        (r"particle", sweep.particle_pred, sweep.particle_exact),
        (r"hole", sweep.hole_pred, sweep.hole_exact),
    )
    peak = float(
        max(
            sweep.particle_pred.max(),
            sweep.particle_exact.max(),
            sweep.hole_pred.max(),
            sweep.hole_exact.max(),
        )
    )

    # A robust shared ceiling for the two difference panels: a handful of
    # single-seed gauge stripes must not wash the colour scale out.
    diff_peak = float(
        max(
            np.percentile(np.abs(sweep.particle_pred - sweep.particle_exact), 99.0),
            np.percentile(np.abs(sweep.hole_pred - sweep.hole_exact), 99.0),
        )
    )

    fig, axes = plt.subplots(
        2, 3, figsize=(14, 7.2), sharex=True, sharey=True, layout="constrained"
    )
    density_images: list[Any] = []
    diff_images: list[Any] = []
    for r, (sector, pred, exact) in enumerate(rows):
        difference = np.abs(pred - exact)
        panels = (
            (f"{model_label}", pred, "viridis", 0.0, peak),
            ("exact", exact, "viridis", 0.0, peak),
            (
                r"$|\,$model $-$ exact$\,|$",
                difference,
                "magma",
                0.0,
                diff_peak,
            ),
        )
        for c, (title, data, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[r, c]
            image = ax.imshow(
                data.T,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            (density_images if c < 2 else diff_images).append(image)
            ax.grid(False)
            ax.set_yticks(np.arange(0, n_sites, max(1, n_sites // 5)))
            for boundary in (-2 * hopping, 2 * hopping):
                if extent[0] <= boundary <= extent[1]:
                    ax.axvline(boundary, color="white", ls=(0, (4, 3)), lw=1.0)
            if r == 0:
                ax.set_title(title)
            if r == 1:
                ax.set_xlabel(r"$\mu / t$")
        axes[r, 0].set_ylabel(f"{sector}\nsite $n$")

    fig.colorbar(
        density_images[-1],
        ax=axes[:, :2].ravel().tolist(),
        location="right",
        shrink=0.85,
        pad=0.015,
        label=r"$|\psi_n|^2$",
    )
    fig.colorbar(
        diff_images[-1],
        ax=axes[:, 2].ravel().tolist(),
        location="right",
        shrink=0.85,
        pad=0.015,
        label=r"absolute density error  $|\Delta\,|\psi_n|^2|$",
    )
    fig.suptitle(
        r"Particle / hole density $|\psi_n(\mu)|^2$", fontsize=13, weight="bold"
    )
    _save(fig, save_path, dpi)
    return fig


def plot_pair_density_waterfall(
    sweep: WavefunctionSweep,
    *,
    hopping: float,
    model_label: str = "model",
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    r"""Image the gauge-invariant pair density
    :math:`\rho_n(\mu) = |\psi^p_n|^2 + |\psi^h_n|^2`.

    Inside the topological phase the :math:`\pm\sigma_1` pair is
    near-degenerate, so the individual particle/hole density of one
    representative is gauge-dependent and not comparable between models.
    The site-resolved density of the *pair* :math:`\{\psi, \Xi\psi\}` --
    the diagonal of the projector onto their 2D span -- is
    gauge-invariant: exactly so for the chiral model, where
    :math:`\Xi\psi` is structural, and to within the model's own branch
    error for the Nambu-basis models. This panel is therefore a fair
    cross-model view where the raw particle/hole waterfall is not.

    Args:
        sweep: A
            :class:`~kitaev.visualisation.evaluation.WavefunctionSweep`
            over a dense ``probe_mus`` grid.
        hopping: The hopping amplitude ``t``, for the transition markers.
        model_label: Column title for the predicted panel.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        The ``(1, 3)``-panel figure (model, exact, ``|model - exact|``).
    """
    use_house_style()
    mu = np.asarray(sweep.probe_mus, dtype=float)
    n_sites = int(sweep.sites[-1]) + 1
    extent = (float(mu.min()), float(mu.max()), -0.5, n_sites - 0.5)

    rho_pred = sweep.particle_pred + sweep.hole_pred
    rho_exact = sweep.particle_exact + sweep.hole_exact
    difference = np.abs(rho_pred - rho_exact)
    peak = float(max(rho_pred.max(), rho_exact.max()))
    diff_peak = float(np.percentile(difference, 99.0)) or float(difference.max() or 1.0)

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4.0), sharex=True, sharey=True, layout="constrained"
    )
    panels = (
        (model_label, rho_pred, "viridis", peak),
        ("exact", rho_exact, "viridis", peak),
        (r"$|\,$model $-$ exact$\,|$", difference, "magma", diff_peak),
    )
    images: list[Any] = []
    for ax, (title, data, cmap, vmax) in zip(axes, panels, strict=True):
        images.append(
            ax.imshow(
                data.T,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
            )
        )
        ax.grid(False)
        ax.set_yticks(np.arange(0, n_sites, max(1, n_sites // 5)))
        for boundary in (-2 * hopping, 2 * hopping):
            if extent[0] <= boundary <= extent[1]:
                ax.axvline(boundary, color="white", ls=(0, (4, 3)), lw=1.0)
        ax.set_title(title)
        ax.set_xlabel(r"$\mu / t$")
    axes[0].set_ylabel("site $n$")

    fig.colorbar(
        images[1],
        ax=list(axes[:2]),
        location="right",
        shrink=0.85,
        pad=0.015,
        label=r"$\rho_n = |\psi^p_n|^2 + |\psi^h_n|^2$",
    )
    fig.colorbar(
        images[2],
        ax=[axes[2]],
        location="right",
        shrink=0.85,
        pad=0.015,
        label=r"absolute error  $|\Delta\rho_n|$",
    )
    fig.suptitle(
        r"Gauge-invariant pair density $\rho_n(\mu)$", fontsize=13, weight="bold"
    )
    _save(fig, save_path, dpi)
    return fig


_DEFAULT_SLICE_MUS_IN_T = (0.0, 0.6, 1.4, 2.6)


def _seed_ramp(n_seeds: int) -> list[Any]:
    """Distinct per-seed colours from viridis, matching the fan animation."""
    if n_seeds == 1:
        return [plt.get_cmap("viridis")(0.25)]
    ramp = plt.get_cmap("viridis")
    return [ramp(0.08 + 0.84 * i / (n_seeds - 1)) for i in range(n_seeds)]


def plot_seed_density_dispersion_maps(
    sweep: SeedDensitySweep,
    *,
    hopping: float,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    r"""Image the inter-seed spread of one model's site density, two ways.

    Two stacked ``site`` :math:`\times` :math:`\mu` heatmaps on a shared
    colour scale: the standard deviation across seeds of the **raw**
    particle-plus-hole density, then of the **gauge-invariant** pair
    density :math:`\rho_n/2`. Where the training objective leaves the
    near-degenerate Majorana pair unresolved the raw panel carries a
    bright band inside :math:`|\mu| < 2t` while the gauge-invariant panel
    stays dark -- the seeds disagree only on the representative, not on
    the physical subspace. The static counterpart of one model's cross-
    seed fan animation.

    Args:
        sweep: A
            :class:`~kitaev.visualisation.evaluation.SeedDensitySweep`
            with two or more seeds.
        hopping: The hopping amplitude ``t``, for the transition markers.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        The ``(2, 1)``-panel figure.
    """
    use_house_style()
    mu = np.asarray(sweep.mu, dtype=float)
    n_sites = len(sweep.sites)
    extent = (float(mu.min()), float(mu.max()), -0.5, n_sites - 0.5)

    raw_std = sweep.raw_density_std()
    pair_std = sweep.pair_density_std()
    vmax = float(np.percentile(np.concatenate([raw_std, pair_std]), 99.0)) or 1.0

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 6.4), sharex=True, sharey=True, layout="constrained"
    )
    panels = (
        (r"raw $|\psi^p_n|^2 + |\psi^h_n|^2$", raw_std),
        (r"gauge-invariant $\rho_n/2$", pair_std),
    )
    images: list[Any] = []
    for ax, (title, data) in zip(axes, panels, strict=True):
        images.append(
            ax.imshow(
                data.T,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
            )
        )
        ax.grid(False)
        ax.set_yticks(np.arange(0, n_sites, max(1, n_sites // 5)))
        for boundary in (-2 * hopping, 2 * hopping):
            if extent[0] <= boundary <= extent[1]:
                ax.axvline(boundary, color="white", ls=(0, (4, 3)), lw=1.0)
        ax.set_title(title)
        ax.set_ylabel("site $n$")
    axes[1].set_xlabel(r"$\mu / t$")

    fig.colorbar(
        images[0],
        ax=list(axes),
        location="right",
        shrink=0.9,
        pad=0.015,
        label="std across seeds",
    )
    fig.suptitle(
        f"Cross-seed spread of the site density — {sweep.model_label} "
        f"({sweep.n_seeds} seeds)",
        fontsize=13,
        weight="bold",
    )
    _save(fig, save_path, dpi)
    return fig


def plot_seed_density_slices(
    sweep: SeedDensitySweep,
    *,
    hopping: float,
    mu_values_in_t: Sequence[float] = _DEFAULT_SLICE_MUS_IN_T,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    r"""Per-seed site-density profiles at a few fixed ``mu``, three rows.

    Columns are the chemical-potential slices (nearest grid points to
    ``mu_values_in_t`` times ``t``); rows are the raw particle sector, the
    raw hole sector, and the gauge-invariant pair density
    :math:`\rho_n/2`. Each panel draws one line per seed over the grey
    exact :math:`\rho_n/2` reference. A frozen snapshot of the fan
    animation: inside :math:`|\mu| < 2t` the raw rows spread between
    seeds, the gauge-invariant row collapses onto the reference.

    Args:
        sweep: A
            :class:`~kitaev.visualisation.evaluation.SeedDensitySweep`.
        hopping: The hopping amplitude ``t``.
        mu_values_in_t: Slice positions in units of ``t``; the nearest
            available grid point is used for each.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        The ``(3, len(mu_values_in_t))``-panel figure.
    """
    use_house_style()
    mu = np.asarray(sweep.mu, dtype=float)
    sites = np.asarray(sweep.sites)
    targets = [v * hopping for v in mu_values_in_t]
    idxs = [int(np.argmin(np.abs(mu - t))) for t in targets]
    colours = _seed_ramp(sweep.n_seeds)

    topological = np.abs(mu) < 2 * hopping
    # Particle/hole rows: reference is rho_n/2 inside the phase (a single
    # representative is arbitrary there) and the lowest eigenvector's own
    # density outside it. The pair-density row is rho_n/2 everywhere.
    particle_ref = np.where(
        topological[:, None], sweep.pair_particle_exact, sweep.raw_particle_exact
    )
    hole_ref = np.where(
        topological[:, None], sweep.pair_hole_exact, sweep.raw_hole_exact
    )
    rows = (
        (
            r"particle $|\psi^p_n|^2$" + "\n(raw, per seed)",
            sweep.raw_particle,
            particle_ref,
        ),
        (r"hole $|\psi^h_n|^2$" + "\n(raw, per seed)", sweep.raw_hole, hole_ref),
        (
            r"pair density $\rho_n/2$" + "\n(gauge-invariant, per seed)",
            sweep.pair_particle + sweep.pair_hole,
            sweep.pair_particle_exact + sweep.pair_hole_exact,
        ),
    )

    fig, axes = plt.subplots(
        3,
        len(idxs),
        figsize=(3.3 * len(idxs), 8.2),
        sharex=True,
        sharey="row",
        layout="constrained",
        squeeze=False,
    )
    for c, i in enumerate(idxs):
        phase = "topological" if topological[i] else "trivial"
        axes[0, c].set_title(f"$\\mu = {mu[i]:+.2f}\\,t$\n({phase})", fontsize=10)
        for r, (_, per_seed, exact) in enumerate(rows):
            ax = axes[r, c]
            ax.fill_between(
                sites,
                exact[i],
                color=INK,
                alpha=0.14,
                lw=0,
                label="exact" if (r, c) == (0, 0) else None,
            )
            for s in range(sweep.n_seeds):
                ax.plot(
                    sites,
                    per_seed[s, i],
                    color=colours[s],
                    lw=1.2,
                    label=f"seed {s}" if (r, c) == (0, 0) else None,
                )
            ax.grid(True, alpha=0.3)
    for r, (label, _, _) in enumerate(rows):
        axes[r, 0].set_ylabel(label, fontsize=9)
    for c in range(len(idxs)):
        axes[-1, c].set_xlabel("site $n$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(6, sweep.n_seeds + 1),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        f"Per-seed site density at fixed $\\mu$ — {sweep.model_label} "
        f"({sweep.n_seeds} seeds)",
        fontsize=13,
        weight="bold",
    )
    _save(fig, save_path, dpi)
    return fig


def plot_seed_edge_weight_envelope(
    sweeps: Sequence[SeedDensitySweep],
    *,
    hopping: float,
    n_edge_sites: int = 2,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    r"""Cross-seed envelope of the left-end weight, raw vs gauge-invariant.

    One panel per model, shading the full inter-seed range (min to max
    over seeds) of the weight on the ``n_edge_sites`` sites of the **left**
    end against :math:`\mu`, once for the **raw** per-sector density and
    once for the **gauge-invariant** pair density :math:`\rho_n/2`, with
    the exact :math:`\rho_n/2` value on top. The *combined* two-end edge
    weight is nearly gauge-invariant by itself -- a rotation within the
    Majorana pair just moves weight between the ends -- so the single-end
    weight is the scalar that fans: for a model that does not resolve the
    representative the raw band spans much of :math:`[0, 1]` inside
    :math:`|\mu| < 2t`, while the :math:`\rho_n/2` band stays pinned near
    :math:`1/2`.

    Args:
        sweeps: One
            :class:`~kitaev.visualisation.evaluation.SeedDensitySweep`
            per model, in plot order.
        hopping: The hopping amplitude ``t``.
        n_edge_sites: Sites counted at the left end of the chain.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        The figure, one panel per model.
    """
    use_house_style()
    sweeps = list(sweeps)
    n = len(sweeps)
    ncols = min(n, 2)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.6 * ncols, 3.6 * nrows),
        sharex=True,
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    flat = axes.ravel()
    left = np.arange(n_edge_sites)
    for ax, sweep in zip(flat, sweeps, strict=False):
        mu = np.asarray(sweep.mu, dtype=float)
        mu_max = float(np.abs(mu).max())
        for which, colour, name in (
            ("raw", CORAL, "raw sectors"),
            ("pair", TEAL, r"gauge-invariant $\rho_n/2$"),
        ):
            ew = sweep.edge_weight(which, n_edge_sites=n_edge_sites, end="left")
            ax.fill_between(
                mu,
                ew.min(axis=0),
                ew.max(axis=0),
                color=colour,
                alpha=0.25,
                lw=0,
                label=f"{name} (seed range)",
            )
            ax.plot(mu, np.median(ew, axis=0), color=colour, lw=1.4)
        exact_ew = sweep.pair_particle_exact[:, left].sum(
            axis=1
        ) + sweep.pair_hole_exact[:, left].sum(axis=1)
        ax.plot(mu, exact_ew, color=INK, ls=(0, (4, 3)), lw=1.3, label="exact")
        mark_transition(
            ax, hopping=hopping, mu_max=mu_max, two_sided=bool(mu.min() < 0)
        )
        ax.set_title(sweep.model_label, fontsize=11)
        ax.set_xlabel(r"$\mu / t$")
        ax.set_ylabel("left-end weight")
    for ax in flat[n:]:
        ax.set_visible(False)
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Cross-seed spread of the left-end weight — "
        "raw sectors vs gauge-invariant $\\rho_n/2$",
        fontsize=13,
        weight="bold",
    )
    _save(fig, save_path, dpi)
    return fig


def plot_spectral_fan(
    low: LowSpectrumSweep,
    *,
    hopping: float,
    predicted: SpectralSweep | None = None,
    model_label: str = "model",
    two_sided: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plot the lowest exact ``|E_k(mu)|`` levels, optionally with a model branch.

    Shows the ``+-lambda_1`` pair collapsing towards zero inside the
    topological phase and the bulk gap closing at ``|mu| = 2t`` -- the
    context for why the topological eigenvector is under-determined.

    Args:
        low: The result of
            :func:`kitaev.visualisation.evaluation.sweep_low_spectrum`.
        hopping: The hopping amplitude ``t``, for the transition markers.
        predicted: Optional
            :class:`~kitaev.visualisation.evaluation.SpectralSweep` whose
            ``energy_pred`` is overlaid as the model's tracked branch.
        model_label: Legend label for that overlay.
        two_sided: Passed to
            :func:`kitaev.visualisation.style.mark_transition`.
        save_path: If given, the figure is also saved here.
        dpi: Resolution used when saving.

    Returns:
        The single-axis figure.
    """
    use_house_style()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    mu_max = float(np.abs(low.mu).max())
    mark_transition(ax, hopping=hopping, mu_max=mu_max, two_sided=two_sided)

    n_levels = low.levels.shape[1]
    for k in range(n_levels):
        colour = _LEVEL_COLOURS[k % len(_LEVEL_COLOURS)]
        ax.plot(
            low.mu,
            low.levels[:, k],
            color=colour,
            lw=1.8 if k == 0 else 1.2,
            label=rf"exact $\sigma_{{{k + 1}}}$",
        )
    if predicted is not None:
        ax.plot(
            predicted.mu,
            _rolling_median(predicted.energy_pred),
            color=CORAL,
            lw=1.7,
            ls=(0, (4, 2)),
            label=f"{model_label} branch",
        )
    ax.set_yscale("log")
    ax.set_title(r"Lowest exact levels $|E_k(\mu)|$")
    ax.set_xlabel(r"$\mu / t$")
    ax.set_ylabel(r"$|E_k|$")
    ax.legend(loc="lower left", fontsize=9, ncol=2)

    fig.tight_layout()
    _save(fig, save_path, dpi)
    return fig
