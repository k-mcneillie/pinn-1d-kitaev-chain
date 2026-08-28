"""Tests for kitaev.visualisation.evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from kitaev.analytical import KitaevChainHamiltonian
from kitaev.models import SirenPINNDualHead
from kitaev.visualisation.evaluation import (
    sweep_energy_and_edge_weight,
    sweep_mu_reflection,
    sweep_spectrum,
    sweep_wavefunction_grid,
    sweep_wavefunctions,
)

N_SITES = 6  # large enough that the default n_edge_sites=2 windows don't overlap


@pytest.fixture
def hamiltonian() -> KitaevChainHamiltonian:
    return KitaevChainHamiltonian(n_sites=N_SITES, hopping=1.0, pairing=0.5)


@pytest.fixture
def dual_head_model() -> SirenPINNDualHead:
    return SirenPINNDualHead(n_sites=2 * N_SITES, hidden_features=8, hidden_layers=1)


def test_sweep_energy_and_edge_weight_shapes_and_metadata(
    dual_head_model, hamiltonian
) -> None:
    mu_sweep = np.linspace(-3, 3, 11)

    sweep = sweep_energy_and_edge_weight(
        dual_head_model, hamiltonian, mu_sweep, n_edge_sites=1
    )

    assert sweep.mu_sweep is mu_sweep
    assert sweep.energy_exact.shape == (11,)
    assert sweep.energy_pred.shape == (11,)
    assert sweep.edge_weight_exact.shape == (11,)
    assert sweep.edge_weight_pred.shape == (11,)
    assert sweep.n_edge_sites == 1
    assert np.all(np.isfinite(sweep.energy_exact))
    assert np.all(np.isfinite(sweep.energy_pred))


def test_sweep_energy_and_edge_weight_exact_matches_manual_diagonalisation(
    dual_head_model, hamiltonian
) -> None:
    mu_sweep = np.array([0.7])

    sweep = sweep_energy_and_edge_weight(dual_head_model, hamiltonian, mu_sweep)

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.build(0.7))
    psi = eigenvectors[:, N_SITES]
    expected_energy = eigenvalues[N_SITES]
    particle_prob = psi[:N_SITES] ** 2
    hole_prob = psi[N_SITES:] ** 2
    edge_sites = np.array([0, 1, N_SITES - 2, N_SITES - 1])  # default n_edge_sites=2
    expected_edge_weight = particle_prob[edge_sites].sum() + hole_prob[edge_sites].sum()

    assert sweep.energy_exact[0] == pytest.approx(expected_energy)
    assert sweep.edge_weight_exact[0] == pytest.approx(expected_edge_weight)


def test_sweep_energy_and_edge_weight_pred_is_bounded_by_normalisation(
    dual_head_model, hamiltonian
) -> None:
    # Combined edge weight from a unit-norm psi_pred can never exceed 1.
    mu_sweep = np.linspace(-3, 3, 20)

    sweep = sweep_energy_and_edge_weight(dual_head_model, hamiltonian, mu_sweep)

    assert np.all(sweep.edge_weight_pred >= 0.0)
    assert np.all(sweep.edge_weight_pred <= 1.0 + 1e-5)
    assert np.all(sweep.energy_pred >= 0.0)  # hard non-negativity constraint


def test_sweep_wavefunctions_shapes_and_probability_normalisation(
    dual_head_model, hamiltonian
) -> None:
    probe_mus = [-2.5, 0.0, 1.5]

    sweep = sweep_wavefunctions(dual_head_model, hamiltonian, probe_mus)

    assert sweep.probe_mus == probe_mus
    assert sweep.sites.shape == (N_SITES,)
    for array in (
        sweep.particle_exact,
        sweep.hole_exact,
        sweep.particle_pred,
        sweep.hole_pred,
    ):
        assert array.shape == (3, N_SITES)

    # Exact eigenvectors are unit-normalised, as are model predictions
    # (SirenPINNDualHead L2-normalises psi_pred), so particle + hole
    # probability must sum to 1 across the full (2N-dim) vector for each row.
    total_exact = sweep.particle_exact.sum(axis=1) + sweep.hole_exact.sum(axis=1)
    total_pred = sweep.particle_pred.sum(axis=1) + sweep.hole_pred.sum(axis=1)
    assert total_exact == pytest.approx(np.ones(3), abs=1e-6)
    assert total_pred == pytest.approx(np.ones(3), abs=1e-5)


def test_sweep_spectrum_shapes_error_and_fidelity(dual_head_model, hamiltonian) -> None:
    mu_grid = np.linspace(-3, 3, 15)

    sweep = sweep_spectrum(dual_head_model, hamiltonian, mu_grid)

    for array in (
        sweep.energy_exact,
        sweep.energy_pred,
        sweep.abs_error,
        sweep.edge_weight_pred,
        sweep.subspace_fidelity,
    ):
        assert array.shape == (15,)
    # energy_exact must match a direct diagonalisation.
    expected = np.array(
        [np.linalg.eigvalsh(hamiltonian.build(float(m)))[N_SITES] for m in mu_grid]
    )
    assert sweep.energy_exact == pytest.approx(expected)
    manual_error = np.abs(sweep.energy_pred - sweep.energy_exact)
    assert sweep.abs_error == pytest.approx(manual_error)
    # ||P psi|| for a unit psi and a rank-2 projector is in [0, 1].
    assert np.all(sweep.subspace_fidelity >= 0.0)
    assert np.all(sweep.subspace_fidelity <= 1.0 + 1e-6)
    assert sweep.transition == pytest.approx(2.0)


def test_sweep_wavefunction_grid_manifold_density_is_topological_only(
    dual_head_model, hamiltonian
) -> None:
    probe_mus = [-3.0, 0.5, 3.0]  # trivial, topological, trivial

    sweep = sweep_wavefunction_grid(dual_head_model, hamiltonian, probe_mus)

    assert sweep.manifold_density.shape == (3, 2, N_SITES)
    assert np.isnan(sweep.manifold_density[0]).all()  # |mu| > 2t
    assert np.isfinite(sweep.manifold_density[1]).all()  # |mu| < 2t
    assert np.isnan(sweep.manifold_density[2]).all()
    # rho sums to 2 over the full 2N vector where it is defined.
    assert sweep.manifold_density[1].sum() == pytest.approx(2.0, abs=1e-6)
    assert sweep.branch is not None and len(sweep.branch) == 3
    assert set(sweep.branch) <= {"keep", "Xi-flip"}


def test_sweep_mu_reflection_shapes_and_non_negative_diff(dual_head_model) -> None:
    sweep = sweep_mu_reflection(dual_head_model, mu_max=4.0, n_points=50)

    assert sweep.mu_half.shape == (50,)
    assert sweep.energy_pos.shape == (50,)
    assert sweep.energy_neg.shape == (50,)
    assert sweep.mu_half[0] == 0.0 and sweep.mu_half[-1] == pytest.approx(4.0)
    assert sweep.max_abs_diff >= 0.0
