"""Tests for kitaev.visualisation.animation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation

from kitaev.analytical import KitaevChainHamiltonian
from kitaev.models import ChiralToBdGAdapter, SirenPINNChiral, SirenPINNDualHead
from kitaev.visualisation.animation import (
    MovieModel,
    animate_seed_fan,
    animate_spectrum,
    animate_wavefunction_residual,
    build_wavefunction_movie,
    save_animation,
)
from kitaev.xai.loading import psi_only

matplotlib.use("Agg")  # headless backend, must be set before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402

N_SITES = 6
FRAMES = 5


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    yield
    plt.close("all")


@pytest.fixture
def hamiltonian() -> KitaevChainHamiltonian:
    return KitaevChainHamiltonian(n_sites=N_SITES, hopping=1.0, pairing=0.5)


@pytest.fixture
def mu_frames() -> np.ndarray:
    return np.linspace(-3.0, 3.0, FRAMES)


@pytest.fixture
def movie_models() -> list[MovieModel]:
    dual = SirenPINNDualHead(n_sites=2 * N_SITES, hidden_features=8, hidden_layers=1)
    chiral = SirenPINNChiral(n_sites=N_SITES, hidden_features=8, hidden_layers=1)
    return [
        MovieModel("dual", dual, psi_only(dual), "nambu"),
        MovieModel(
            "chiral",
            ChiralToBdGAdapter(chiral, hopping=1.0, pairing=0.5),
            chiral,
            "chiral",
        ),
    ]


def test_build_wavefunction_movie_shapes(hamiltonian, mu_frames, movie_models) -> None:
    movie = build_wavefunction_movie(movie_models, hamiltonian, mu_frames)

    assert movie.mu_frames.shape == (FRAMES,)
    assert movie.particle_exact.shape == movie.hole_exact.shape == (FRAMES, N_SITES)
    assert set(movie.particle) == set(movie.hole) == {"dual", "chiral"}
    for label in ("dual", "chiral"):
        assert movie.particle[label].shape == (FRAMES, N_SITES)
        assert movie.hole[label].shape == (FRAMES, N_SITES)
        assert movie.residuals[label].shape == (FRAMES,)
        assert np.all(movie.residuals[label] >= 0.0)
    # particle + hole is a genuine probability profile
    total = movie.particle_exact.sum(axis=1) + movie.hole_exact.sum(axis=1)
    assert total == pytest.approx(np.ones(FRAMES), abs=1e-6)
    assert movie.transition == pytest.approx(2.0)


def test_build_wavefunction_movie_rejects_bad_basis(hamiltonian, mu_frames) -> None:
    dual = SirenPINNDualHead(n_sites=2 * N_SITES, hidden_features=8, hidden_layers=1)
    bad = [MovieModel("x", dual, psi_only(dual), "majorana")]
    with pytest.raises(ValueError, match="basis must be one of"):
        build_wavefunction_movie(bad, hamiltonian, mu_frames)


def test_animate_wavefunction_residual_saves_gif(
    hamiltonian, mu_frames, movie_models, tmp_path: Path
) -> None:
    movie = build_wavefunction_movie(movie_models, hamiltonian, mu_frames)
    gif = tmp_path / "wave.gif"

    anim = animate_wavefunction_residual(movie, hopping=1.0, fps=5, save_path=gif)

    assert isinstance(anim, FuncAnimation)
    assert gif.exists() and gif.stat().st_size > 0


def test_animate_seed_fan_runs_with_and_without_exact(
    mu_frames, tmp_path: Path
) -> None:
    rng = np.random.default_rng(0)
    particle = [np.abs(rng.normal(size=(FRAMES, N_SITES))) for _ in range(3)]
    hole = [np.abs(rng.normal(size=(FRAMES, N_SITES))) for _ in range(3)]
    exact = np.full((FRAMES, N_SITES), 1.0 / N_SITES)

    with_exact = animate_seed_fan(
        particle,
        hole,
        mu_frames,
        model_label="chiral",
        transition=2.0,
        particle_exact=exact,
        hole_exact=exact,
        fps=5,
        save_path=tmp_path / "fan_exact.gif",
    )
    without = animate_seed_fan(
        particle,
        hole,
        mu_frames,
        model_label="chiral",
        transition=2.0,
        fps=5,
        save_path=tmp_path / "fan_plain.gif",
    )

    assert isinstance(with_exact, FuncAnimation)
    assert isinstance(without, FuncAnimation)
    assert (tmp_path / "fan_exact.gif").stat().st_size > 0
    assert (tmp_path / "fan_plain.gif").stat().st_size > 0


def test_animate_spectrum_runs_and_saves(
    hamiltonian, mu_frames, tmp_path: Path
) -> None:
    gif = tmp_path / "spectrum.gif"
    anim = animate_spectrum(hamiltonian, mu_frames, n_levels=4, fps=5, save_path=gif)
    assert isinstance(anim, FuncAnimation)
    assert gif.exists() and gif.stat().st_size > 0


def test_save_animation_rejects_non_gif(hamiltonian, mu_frames, tmp_path: Path) -> None:
    anim = animate_spectrum(hamiltonian, mu_frames, n_levels=3)
    with pytest.raises(ValueError, match="must be a .gif"):
        save_animation(anim, tmp_path / "spectrum.mp4")
