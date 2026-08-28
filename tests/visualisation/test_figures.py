"""Tests for kitaev.visualisation.figures (the house-style standard figures)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

from kitaev.training.utils import TrainingHistory
from kitaev.visualisation.evaluation import (
    MuReflectionSweep,
    SpectralSweep,
    WavefunctionSweep,
)
from kitaev.visualisation.figures import (
    plot_eigenvector_agreement,
    plot_energy_sweep,
    plot_loss_history,
    plot_mu_reflection,
    plot_probe_history,
    plot_wavefunction_grid,
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
    for epoch in range(1, 13):
        history.record("train_loss", 1.0 / epoch)
        history.record("val_loss", 1.1 / epoch)
        history.record("train_fsm", 0.6 / epoch)
        history.record("train_var", 0.4 / epoch)
        history.record("train_pin", 0.3 / epoch)
        history.record("train_pin_wt", 1.0 - epoch / 20)
    for call in range(1, 5):
        history.record("probe_epoch", float(call * 3))
        for key in (
            "probe_e_mae_topological",
            "probe_e_mae_trivial",
            "probe_edge_mae",
            "probe_subspace_infidelity",
            "probe_subspace_infidelity_max",
        ):
            history.record(key, 0.1 / call)
    return history


@pytest.fixture
def spectral_sweep() -> SpectralSweep:
    mu = np.linspace(-4, 4, 40)
    return SpectralSweep(
        mu=mu,
        energy_exact=np.abs(mu),
        energy_pred_signed=np.abs(mu) + 0.01,
        energy_pred=np.abs(mu) + 0.01,
        abs_error=np.full_like(mu, 0.01),
        edge_weight_exact=np.clip(1 - np.abs(mu) / 2, 0, 1),
        edge_weight_pred=np.clip(1 - np.abs(mu) / 2, 0, 1) + 0.01,
        subspace_fidelity=np.full_like(mu, 0.999),
        transition=2.0,
        n_edge_sites=2,
    )


@pytest.fixture
def wavefunction_sweep() -> WavefunctionSweep:
    sites = np.arange(5)
    probe_mus = [-3.0, 1.0, 3.0]
    shape = (len(probe_mus), len(sites))
    density = np.full((len(probe_mus), 2, len(sites)), np.nan)
    density[1] = 0.4  # only the topological column carries a manifold density
    return WavefunctionSweep(
        probe_mus=probe_mus,
        sites=sites,
        particle_exact=np.full(shape, 0.2),
        hole_exact=np.full(shape, 0.2),
        particle_pred=np.full(shape, 0.19),
        hole_pred=np.full(shape, 0.21),
        manifold_density=density,
        branch=["keep", "keep", "Xi-flip"],
    )


@pytest.fixture
def mu_reflection_sweep() -> MuReflectionSweep:
    mu_half = np.linspace(0, 4, 30)
    return MuReflectionSweep(
        mu_half=mu_half,
        energy_pos=np.abs(mu_half),
        energy_neg=np.abs(mu_half) + 0.001,
        max_abs_diff=0.001,
    )


def test_plot_loss_history_with_weight_axis(history, tmp_path: Path) -> None:
    save_path = tmp_path / "loss.png"
    fig = plot_loss_history(
        history,
        component_keys=["fsm", "var", "pin"],
        weight_key="pin_wt",
        split_epoch=6,
        save_path=save_path,
    )
    assert isinstance(fig, Figure)
    # left panel, right panel, and the twinned weight axis.
    assert len(fig.axes) == 3
    assert save_path.exists() and save_path.stat().st_size > 0


def test_plot_loss_history_skips_missing_components(history) -> None:
    fig = plot_loss_history(history, component_keys=["fsm", "does_not_exist"])
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2  # no weight axis when weight_key is None


def test_plot_probe_history_returns_figure(history, tmp_path: Path) -> None:
    save_path = tmp_path / "probe.png"
    fig = plot_probe_history(history, split_epoch=6, save_path=save_path)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert save_path.exists() and save_path.stat().st_size > 0


def test_plot_energy_sweep_returns_figure(spectral_sweep, tmp_path: Path) -> None:
    save_path = tmp_path / "energy.png"
    fig = plot_energy_sweep(
        spectral_sweep, hopping=1.0, model_label="test", save_path=save_path
    )
    assert isinstance(fig, Figure)
    assert save_path.exists() and save_path.stat().st_size > 0


def test_plot_eigenvector_agreement_returns_figure(spectral_sweep) -> None:
    fig = plot_eigenvector_agreement(spectral_sweep, hopping=1.0, model_label="test")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_plot_wavefunction_grid_one_column_per_probe(wavefunction_sweep) -> None:
    fig = plot_wavefunction_grid(wavefunction_sweep, hopping=1.0)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2 * len(wavefunction_sweep.probe_mus)


def test_plot_wavefunction_grid_handles_single_probe() -> None:
    sites = np.arange(4)
    sweep = WavefunctionSweep(
        probe_mus=[0.0],
        sites=sites,
        particle_exact=np.full((1, 4), 0.25),
        hole_exact=np.full((1, 4), 0.25),
        particle_pred=np.full((1, 4), 0.24),
        hole_pred=np.full((1, 4), 0.26),
    )
    fig = plot_wavefunction_grid(sweep, hopping=1.0)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_plot_mu_reflection_title_tracks_structural_flag(mu_reflection_sweep) -> None:
    learned = plot_mu_reflection(
        mu_reflection_sweep, hopping=1.0, structural_fold=False
    )
    structural = plot_mu_reflection(
        mu_reflection_sweep, hopping=1.0, structural_fold=True
    )
    assert "Learned" in learned.axes[0].get_title()
    assert "construction" in structural.axes[0].get_title()
