"""Tests for kitaev.visualisation.figures (the house-style standard figures)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

from kitaev.training.utils import TrainingHistory
from kitaev.visualisation.evaluation import (
    LowSpectrumSweep,
    ModelErrorBand,
    MuReflectionSweep,
    SpectralSweep,
    WavefunctionSweep,
)
from kitaev.visualisation.figures import (
    plot_eigenvector_agreement,
    plot_energy_sweep,
    plot_loss_history,
    plot_model_comparison,
    plot_mu_reflection,
    plot_pair_density_waterfall,
    plot_probe_history,
    plot_spectral_fan,
    plot_wavefunction_grid,
    plot_wavefunction_waterfall,
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


def test_plot_loss_history_draws_floor_line(history) -> None:
    fig = plot_loss_history(history, component_keys=["fsm", "var"], floor_value=0.05)
    labels = [line.get_label() for line in fig.axes[1].get_lines()]
    assert any("analytic floor" in str(label) for label in labels)


def _error_band(label: str, error: float) -> ModelErrorBand:
    mu = np.linspace(-4, 4, 40)
    return ModelErrorBand(
        label=label,
        mu=mu,
        abs_error_median=np.full_like(mu, error),
        abs_error_lo=np.full_like(mu, error * 0.5),
        abs_error_hi=np.full_like(mu, error * 1.5),
        mae_trivial=[error, error * 1.1, error * 0.9],
        mae_topological=[error * 2, error * 2.2, error * 1.8],
        transition=2.0,
        n_seeds=3,
    )


def test_plot_model_comparison_two_panels_and_ticks(tmp_path: Path) -> None:
    bands = [_error_band("chiral", 1e-5), _error_band("nambu", 4e-3)]
    save_path = tmp_path / "comparison.png"

    fig = plot_model_comparison(bands, hopping=1.0, save_path=save_path)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    tick_labels = [t.get_text() for t in fig.axes[1].get_xticklabels()]
    assert tick_labels == ["chiral", "nambu"]
    assert save_path.exists() and save_path.stat().st_size > 0


def test_plot_wavefunction_waterfall_three_panels() -> None:
    mu = np.linspace(-4, 4, 25)
    sites = np.arange(6)
    shape = (mu.size, sites.size)
    sweep = WavefunctionSweep(
        probe_mus=list(mu),
        sites=sites,
        particle_exact=np.full(shape, 0.1),
        hole_exact=np.full(shape, 0.1),
        particle_pred=np.full(shape, 0.09),
        hole_pred=np.full(shape, 0.11),
    )

    fig = plot_wavefunction_waterfall(sweep, hopping=1.0, model_label="chiral")

    assert isinstance(fig, Figure)
    # 2 rows x 3 cols of image axes, one shared density colourbar, one diff bar
    assert len(fig.axes) == 8
    assert "particle" in fig.axes[0].get_ylabel()
    assert "hole" in fig.axes[3].get_ylabel()


def test_plot_pair_density_waterfall_is_gauge_invariant_combination(
    tmp_path: Path,
) -> None:
    mu = np.linspace(-4, 4, 20)
    sites = np.arange(5)
    shape = (mu.size, sites.size)
    rng = np.random.default_rng(0)
    # Two "seeds" that split the same pair density differently between the
    # particle and hole sectors: rho = particle + hole must match.
    particle_a = rng.uniform(0.0, 0.2, shape)
    hole_a = 0.4 - particle_a
    sweep = WavefunctionSweep(
        probe_mus=list(mu),
        sites=sites,
        particle_exact=np.full(shape, 0.2),
        hole_exact=np.full(shape, 0.2),
        particle_pred=particle_a,
        hole_pred=hole_a,
    )
    save_path = tmp_path / "pair.png"

    fig = plot_pair_density_waterfall(
        sweep, hopping=1.0, model_label="chiral", save_path=save_path
    )

    assert isinstance(fig, Figure)
    # model, exact, |diff| image axes + two colourbars
    assert len(fig.axes) == 5
    # rho_pred is uniform 0.4, rho_exact uniform 0.4 -> difference panel is flat
    diff_image = fig.axes[2].get_images()[0]
    assert float(np.abs(diff_image.get_array()).max()) == pytest.approx(0.0, abs=1e-12)
    assert save_path.exists() and save_path.stat().st_size > 0


def test_plot_spectral_fan_with_and_without_predicted(spectral_sweep) -> None:
    mu = spectral_sweep.mu
    low = LowSpectrumSweep(
        mu=mu,
        levels=np.stack([np.abs(mu) * 0.1, np.abs(mu) * 0.5, np.abs(mu) + 1.0], axis=1),
        transition=2.0,
    )

    bare = plot_spectral_fan(low, hopping=1.0)
    overlaid = plot_spectral_fan(
        low, hopping=1.0, predicted=spectral_sweep, model_label="chiral"
    )

    assert isinstance(bare, Figure)
    assert isinstance(overlaid, Figure)
    labels = [line.get_label() for line in overlaid.axes[0].get_lines()]
    assert any("chiral branch" in str(label) for label in labels)
