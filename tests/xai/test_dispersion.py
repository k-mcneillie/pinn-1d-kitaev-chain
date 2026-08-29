"""Tests for kitaev.xai.dispersion."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kitaev.analytical import KitaevChainHamiltonian
from kitaev.models import ChiralToBdGAdapter, SirenPINNChiral
from kitaev.xai.dispersion import sweep_seed_dispersion

N_SITES = 6


def _adapter(seed: int) -> ChiralToBdGAdapter:
    torch.manual_seed(seed)
    model = SirenPINNChiral(n_sites=N_SITES, hidden_features=8, hidden_layers=1)
    return ChiralToBdGAdapter(model, hopping=1.0, pairing=0.5)


def test_shapes_and_non_negative_spreads() -> None:
    hamiltonian = KitaevChainHamiltonian(n_sites=N_SITES, hopping=1.0, pairing=0.5)
    mu_grid = np.linspace(-3.0, 3.0, 20)

    dispersion = sweep_seed_dispersion(
        [_adapter(0), _adapter(1), _adapter(2)], hamiltonian, mu_grid
    )

    assert dispersion.mu.shape == dispersion.energy_std.shape == (20,)
    assert (
        dispersion.density_std_mean.shape == dispersion.density_std_max.shape == (20,)
    )
    assert dispersion.n_seeds == 3
    assert np.all(dispersion.density_std_mean >= 0.0)
    assert np.all(dispersion.density_std_max >= dispersion.density_std_mean)
    assert dispersion.transition == pytest.approx(2.0)


def test_requires_at_least_two_seeds() -> None:
    hamiltonian = KitaevChainHamiltonian(n_sites=N_SITES, hopping=1.0, pairing=0.5)
    with pytest.raises(ValueError, match="at least two seeds"):
        sweep_seed_dispersion([_adapter(0)], hamiltonian, np.linspace(-3.0, 3.0, 5))
