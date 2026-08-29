"""Tests for kitaev.xai.figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

from kitaev.xai.conditioning import ConditioningSweep
from kitaev.xai.dispersion import SeedDispersion
from kitaev.xai.figures import (
    plot_conditioning,
    plot_residual_field,
    plot_seed_dispersion,
    plot_transparency_axis,
)
from kitaev.xai.internalisation import KITAEV_PROFILES
from kitaev.xai.residual_field import ResidualField

matplotlib.use("Agg")  # headless backend, must be set before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    yield
    plt.close("all")


def _conditioning(basis: str) -> ConditioningSweep:
    mu = np.linspace(-4.0, 4.0, 20)
    return ConditioningSweep(
        mu=mu,
        sigma_min=np.full(20, 1e-3),
        sigma_max=np.full(20, 2.0),
        condition_number=np.full(20, 2_000.0),
        basis=basis,
        transition=2.0,
    )


def test_plot_transparency_axis(tmp_path: Path) -> None:
    profiles = list(KITAEV_PROFILES.values())
    errors = {p.name: {"trivial": 1e-3, "topological": 1e-4} for p in profiles}
    save_path = tmp_path / "transparency.png"

    fig = plot_transparency_axis(profiles, errors, save_path=save_path)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert save_path.exists() and save_path.stat().st_size > 0


def test_plot_conditioning(tmp_path: Path) -> None:
    save_path = tmp_path / "conditioning.png"

    fig = plot_conditioning(
        [_conditioning("nambu"), _conditioning("chiral")],
        hopping=1.0,
        save_path=save_path,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert save_path.exists() and save_path.stat().st_size > 0


def test_plot_seed_dispersion(tmp_path: Path) -> None:
    mu = np.linspace(-4.0, 4.0, 20)
    dispersion = SeedDispersion(
        mu=mu,
        energy_std=np.full(20, 1e-3),
        density_std_mean=np.full(20, 1e-3),
        density_std_max=np.full(20, 2e-3),
        edge_weight_std=np.full(20, 1e-3),
        n_seeds=5,
        transition=2.0,
    )
    save_path = tmp_path / "dispersion.png"

    fig = plot_seed_dispersion(
        dispersion, hopping=1.0, model_label="chiral", save_path=save_path
    )

    assert isinstance(fig, Figure)
    assert save_path.exists() and save_path.stat().st_size > 0


def test_plot_seed_dispersion_honours_shared_ylim() -> None:
    mu = np.linspace(-4.0, 4.0, 20)
    dispersion = SeedDispersion(
        mu=mu,
        energy_std=np.full(20, 1e-3),
        density_std_mean=np.full(20, 1e-3),
        density_std_max=np.full(20, 2e-3),
        edge_weight_std=np.full(20, 1e-3),
        n_seeds=5,
        transition=2.0,
    )

    fig = plot_seed_dispersion(
        dispersion, hopping=1.0, model_label="chiral", ylim=(1e-9, 1e-1)
    )

    assert fig.axes[0].get_ylim() == pytest.approx((1e-9, 1e-1))


def test_plot_residual_field(tmp_path: Path) -> None:
    mu = np.linspace(-4.0, 4.0, 20)
    base = np.abs(mu) + 1e-6
    fields = [
        ResidualField(
            mu=mu,
            residual_median=base,
            residual_min=base * 0.8,
            residual_max=base * 1.2,
            n_seeds=5,
            basis="nambu",
            label="a",
            transition=2.0,
        ),
        ResidualField(
            mu=mu,
            residual_median=base * 0.5,
            residual_min=base * 0.5,
            residual_max=base * 0.5,
            n_seeds=1,
            basis="chiral",
            label="b",
            transition=2.0,
        ),
    ]
    save_path = tmp_path / "residual.png"

    fig = plot_residual_field(fields, hopping=1.0, save_path=save_path)

    assert isinstance(fig, Figure)
    assert save_path.exists() and save_path.stat().st_size > 0
