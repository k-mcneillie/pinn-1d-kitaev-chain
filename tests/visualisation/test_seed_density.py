"""Tests for the cross-seed density sweep and its publication figures.

These are the static counterparts of the fan animation: they must make the
distinction the animation shows -- raw per-sector density scatters across
seeds inside the topological phase (a gauge degree of freedom), the
gauge-invariant pair density does not.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from kitaev.analytical import KitaevChainHamiltonian
from kitaev.models import SirenPINNDualHead
from kitaev.visualisation.evaluation import SeedDensitySweep, sweep_seed_densities
from kitaev.visualisation.figures import (
    plot_seed_density_dispersion_maps,
    plot_seed_density_slices,
    plot_seed_edge_weight_envelope,
)

matplotlib.use("Agg")  # headless backend, must be set before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402

N_SITES = 6


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    yield
    plt.close("all")


@pytest.fixture
def hamiltonian() -> KitaevChainHamiltonian:
    return KitaevChainHamiltonian(n_sites=N_SITES, hopping=1.0, pairing=0.5)


def _seed_models(n: int) -> list[SirenPINNDualHead]:
    models = []
    for seed in range(n):
        torch.manual_seed(seed)
        models.append(
            SirenPINNDualHead(n_sites=2 * N_SITES, hidden_features=8, hidden_layers=1)
        )
    return models


class _DegeneratePair(torch.nn.Module):
    """A model whose output rotates within a fixed, Xi-invariant 2D span.

    With ``psi_+ = [u; v]`` and ``psi_- = [v; u]`` (``u`` orthogonal to
    ``v``), the particle-hole swap ``Xi`` maps the span onto itself, so
    ``span{psi, Xi psi}`` is the same manifold for every rotation angle.
    The raw per-sector density then depends on ``theta`` while the
    projector diagonal that :func:`predicted_pair_density` returns does
    not -- the property the topological-phase gauge story rests on.
    """

    def __init__(self, theta: float, psi_plus: torch.Tensor, psi_minus: torch.Tensor):
        super().__init__()
        self.theta = theta
        self.register_buffer("psi_plus", psi_plus)
        self.register_buffer("psi_minus", psi_minus)

    def forward(self, mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vec = np.cos(self.theta) * self.psi_plus + np.sin(self.theta) * self.psi_minus
        psi = vec.unsqueeze(0).repeat(mu.shape[0], 1)
        return psi.norm(dim=1, keepdim=True), psi


def test_sweep_seed_densities_shapes_and_normalisation(hamiltonian) -> None:
    mu = np.linspace(-4.0, 4.0, 25)
    sweep = sweep_seed_densities(_seed_models(3), hamiltonian, mu, model_label="siren")

    assert isinstance(sweep, SeedDensitySweep)
    assert sweep.n_seeds == 3
    assert sweep.model_label == "siren"
    assert sweep.raw_particle.shape == (3, 25, N_SITES)
    assert sweep.pair_hole.shape == (3, 25, N_SITES)
    assert sweep.raw_particle_exact.shape == (25, N_SITES)
    # both views are probability densities: the two sectors sum to ~1
    raw_total = (sweep.raw_particle + sweep.raw_hole).sum(axis=2)
    pair_total = (sweep.pair_particle + sweep.pair_hole).sum(axis=2)
    assert np.allclose(raw_total, 1.0, atol=1e-6)
    assert np.allclose(pair_total, 1.0, atol=1e-6)
    assert sweep.raw_density_std().shape == (25, N_SITES)
    assert sweep.pair_density_std().shape == (25, N_SITES)
    assert sweep.edge_weight("raw").shape == (3, 25)
    assert sweep.edge_weight("pair").shape == (3, 25)


def test_sweep_seed_densities_requires_a_seed(hamiltonian) -> None:
    with pytest.raises(ValueError):
        sweep_seed_densities([], hamiltonian, np.array([0.0]))


def test_identical_seeds_have_zero_spread(hamiltonian) -> None:
    torch.manual_seed(0)
    model = SirenPINNDualHead(n_sites=2 * N_SITES, hidden_features=8, hidden_layers=1)
    sweep = sweep_seed_densities(
        [model, model, model], hamiltonian, np.linspace(-4, 4, 9)
    )

    assert np.allclose(sweep.raw_density_std(), 0.0, atol=1e-9)
    assert np.allclose(sweep.pair_density_std(), 0.0, atol=1e-9)


def test_gauge_rotation_moves_raw_density_but_not_the_pair_density(hamiltonian) -> None:
    rng = np.random.default_rng(0)
    u = rng.standard_normal(N_SITES)
    v = rng.standard_normal(N_SITES)
    v = v - (u @ v) / (u @ u) * u  # make v orthogonal to u
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    psi_plus = torch.tensor(np.concatenate([u, v]) / np.sqrt(2.0), dtype=torch.float32)
    psi_minus = torch.tensor(np.concatenate([v, u]) / np.sqrt(2.0), dtype=torch.float32)
    # angles chosen away from pi/4 and 3pi/4, where psi coincides with Xi psi
    models = [
        _DegeneratePair(theta, psi_plus, psi_minus) for theta in (0.0, 0.5, 1.1, 1.9)
    ]

    sweep = sweep_seed_densities(models, hamiltonian, np.linspace(-1.0, 1.0, 5))

    # rotating theta rearranges the raw split ...
    assert sweep.raw_density_std().max() > 1e-2
    # ... but the projector diagonal of the Xi-invariant span does not move.
    assert sweep.pair_density_std().max() < 1e-4


def test_publication_figures_render(hamiltonian) -> None:
    mu = np.linspace(-4.0, 4.0, 40)
    fans = [
        sweep_seed_densities(_seed_models(3), hamiltonian, mu, model_label="model a"),
        sweep_seed_densities(_seed_models(2), hamiltonian, mu, model_label="model b"),
    ]

    assert isinstance(plot_seed_density_dispersion_maps(fans[0], hopping=1.0), Figure)
    assert isinstance(plot_seed_density_slices(fans[0], hopping=1.0), Figure)
    assert isinstance(plot_seed_edge_weight_envelope(fans, hopping=1.0), Figure)


def test_density_slices_honours_requested_mu_positions(hamiltonian) -> None:
    mu = np.linspace(-4.0, 4.0, 41)
    sweep = sweep_seed_densities(_seed_models(2), hamiltonian, mu, model_label="m")

    fig = plot_seed_density_slices(sweep, hopping=1.0, mu_values_in_t=(0.0, 1.0, 3.0))

    # 3 rows (particle, hole, pair) x 3 requested slices
    assert len(fig.axes) >= 9
