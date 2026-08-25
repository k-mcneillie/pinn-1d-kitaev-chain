"""Tests for kitaev.visualisation.plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

from kitaev.training.utils import TrainingHistory
from kitaev.visualisation.evaluation import EnergyEdgeWeightSweep, WavefunctionSweep
from kitaev.visualisation.plots import (
    plot_energy_and_edge_weight,
    plot_loss_curves,
    plot_wavefunctions,
)

matplotlib.use("Agg")  # headless backend, must be set before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    yield
    plt.close("all")


@pytest.fixture
def history() -> TrainingHistory:
    history = TrainingHistory()
    for epoch in range(5):
        history.record("train_loss", 1.0 / (epoch + 1))
        history.record("val_loss", 1.2 / (epoch + 1))
        history.record("train_e", 0.5 / (epoch + 1))
        history.record("train_psi", 0.4 / (epoch + 1))
        history.record("train_res", 0.3 / (epoch + 1))
        history.record("train_ph", 0.3 / (epoch + 1))
    return history


@pytest.fixture
def energy_edge_weight_sweep() -> EnergyEdgeWeightSweep:
    mu = np.linspace(-3, 3, 10)
    return EnergyEdgeWeightSweep(
        mu_sweep=mu,
        energy_exact=np.abs(mu),
        energy_pred=np.abs(mu) + 0.01,
        edge_weight_exact=1.0 - np.abs(mu) / 3,
        edge_weight_pred=1.0 - np.abs(mu) / 3 + 0.01,
        n_edge_sites=2,
    )


@pytest.fixture
def wavefunction_sweep() -> WavefunctionSweep:
    sites = np.arange(4)
    probe_mus = [-1.0, 1.0]
    shape = (len(probe_mus), len(sites))
    return WavefunctionSweep(
        probe_mus=probe_mus,
        sites=sites,
        particle_exact=np.full(shape, 0.25),
        hole_exact=np.full(shape, 0.25),
        particle_pred=np.full(shape, 0.24),
        hole_pred=np.full(shape, 0.26),
    )


def test_plot_loss_curves_returns_figure(history) -> None:
    fig = plot_loss_curves(history, ["e", "psi", "res", "ph"])
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_plot_loss_curves_saves_to_path(history, tmp_path: Path) -> None:
    save_path = tmp_path / "loss.png"
    plot_loss_curves(history, ["e", "psi", "res", "ph"], save_path=save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_energy_and_edge_weight_returns_figure(energy_edge_weight_sweep) -> None:
    fig = plot_energy_and_edge_weight(energy_edge_weight_sweep, hopping=1.0)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_plot_energy_and_edge_weight_saves_to_path(
    energy_edge_weight_sweep, tmp_path: Path
) -> None:
    save_path = tmp_path / "sweep.png"
    plot_energy_and_edge_weight(
        energy_edge_weight_sweep, hopping=1.0, save_path=save_path
    )
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_wavefunctions_returns_figure_with_one_row_per_probe(
    wavefunction_sweep,
) -> None:
    fig = plot_wavefunctions(wavefunction_sweep, hopping=1.0)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2 * len(wavefunction_sweep.probe_mus)


def test_plot_wavefunctions_saves_to_path(wavefunction_sweep, tmp_path: Path) -> None:
    save_path = tmp_path / "wavefunctions.png"
    plot_wavefunctions(wavefunction_sweep, hopping=1.0, save_path=save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_wavefunctions_only_the_first_row_keeps_a_legend(
    wavefunction_sweep,
) -> None:
    # seaborn's lineplot auto-attaches a per-axes legend on every call; only
    # the top row should keep one, or every row would repeat the same
    # Exact/Model key.
    fig = plot_wavefunctions(wavefunction_sweep, hopping=1.0)

    top_particle_ax, top_hole_ax = fig.axes[0], fig.axes[1]
    assert top_particle_ax.get_legend() is not None
    assert top_hole_ax.get_legend() is not None
    for ax in fig.axes[2:]:
        assert ax.get_legend() is None


def test_plot_wavefunctions_handles_a_single_probe_mu() -> None:
    # plt.subplots(1, 2, ...) returns a 1D axes array, unlike the 2D array
    # returned for more than one row -- this must not raise.
    sites = np.arange(4)
    sweep = WavefunctionSweep(
        probe_mus=[0.0],
        sites=sites,
        particle_exact=np.full((1, 4), 0.25),
        hole_exact=np.full((1, 4), 0.25),
        particle_pred=np.full((1, 4), 0.24),
        hole_pred=np.full((1, 4), 0.26),
    )

    fig = plot_wavefunctions(sweep, hopping=1.0)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
