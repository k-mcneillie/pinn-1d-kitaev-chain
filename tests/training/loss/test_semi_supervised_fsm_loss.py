"""Tests for kitaev.training.loss.losses.SemiSupervisedFSMLoss."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kitaev.models import SirenPINN
from kitaev.training.loss.losses import SemiSupervisedFSMLoss
from kitaev.training.trainer import _build_kitaev_operators

from .test_nambu_fsm_loss import _ExactLowestEigenvector

N_SITES = 4
HOPPING = 1.0
PAIRING = 0.5
METRIC_KEYS = {"e", "psi", "fsm", "var", "lam_mean", "physics_wt"}


@pytest.fixture
def operators() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _build_kitaev_operators(N_SITES, hopping=HOPPING, pairing=PAIRING)


@pytest.fixture
def model() -> SirenPINN:
    torch.manual_seed(0)
    return SirenPINN(n_sites=2 * N_SITES, hidden_features=8, hidden_layers=1)


@pytest.fixture
def loss_fn() -> SemiSupervisedFSMLoss:
    return SemiSupervisedFSMLoss(total_epochs=100, anneal_duration=10)


@pytest.fixture
def mu_batch() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(8, 1) * 8.0 - 4.0


def _exact_labels(
    mu_values: torch.Tensor, n_sites: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact lowest non-negative ``(E, psi)`` for each mu, matching the helper."""
    H_base, H_mu_diag, _ = _build_kitaev_operators(
        n_sites, hopping=HOPPING, pairing=PAIRING
    )
    energies = []
    vectors = []
    for mu in mu_values.reshape(-1).tolist():
        h = (H_base + mu * H_mu_diag).numpy()
        eigvals, eigvecs = np.linalg.eigh(h)
        energies.append([eigvals[n_sites]])
        vectors.append(eigvecs[:, n_sites])
    dtype = mu_values.dtype
    return (
        torch.tensor(np.array(energies), dtype=dtype),
        torch.tensor(np.array(vectors), dtype=dtype),
    )


def test_returns_scalar_loss_and_all_metric_keys(
    model, loss_fn, mu_batch, operators
) -> None:
    H_base, H_mu_diag, Xi = operators
    total_loss, metrics = loss_fn(model, mu_batch, H_base, H_mu_diag, Xi, epoch=0)

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.shape == ()
    assert torch.isfinite(total_loss)
    assert set(metrics) == METRIC_KEYS
    assert all(math.isfinite(value) for value in metrics.values())


def test_label_free_path_drops_the_data_terms(
    model, loss_fn, mu_batch, operators
) -> None:
    """With no labels, e == psi == 0 and total == physics_wt * (fsm + var)."""
    H_base, H_mu_diag, Xi = operators
    total_loss, metrics = loss_fn(model, mu_batch, H_base, H_mu_diag, Xi, epoch=3)

    assert metrics["e"] == 0.0
    assert metrics["psi"] == 0.0
    expected = metrics["physics_wt"] * (metrics["fsm"] + metrics["var"])
    assert total_loss.item() == pytest.approx(expected)


def test_data_terms_depend_only_on_the_labelled_rows(model, loss_fn, operators) -> None:
    """Perturbing the label-free tail leaves loss_e / loss_psi unchanged."""
    H_base, H_mu_diag, Xi = operators
    torch.manual_seed(2)
    mu_batch = torch.rand(8, 1) * 8.0 - 4.0
    n_labeled = 3
    energy_batch, psi_batch = _exact_labels(mu_batch[:n_labeled], N_SITES)

    _, m_a = loss_fn(
        model,
        mu_batch,
        H_base,
        H_mu_diag,
        Xi,
        epoch=0,
        energy_batch=energy_batch,
        psi_batch=psi_batch,
    )

    perturbed = mu_batch.clone()
    perturbed[n_labeled:] += 0.5
    _, m_b = loss_fn(
        model,
        perturbed,
        H_base,
        H_mu_diag,
        Xi,
        epoch=0,
        energy_batch=energy_batch,
        psi_batch=psi_batch,
    )

    assert m_a["e"] == pytest.approx(m_b["e"])
    assert m_a["psi"] == pytest.approx(m_b["psi"])
    # the physics terms, evaluated on all rows, must react to the change
    assert m_a["fsm"] != pytest.approx(m_b["fsm"])


def test_exact_labels_drive_the_data_terms_to_zero(loss_fn) -> None:
    """Feeding exact (E, psi) from eigh sends loss_e and loss_psi to ~0."""
    H_base, H_mu_diag, Xi = _build_kitaev_operators(
        N_SITES, hopping=HOPPING, pairing=PAIRING
    )
    exact = _ExactLowestEigenvector(N_SITES)
    mu_batch = torch.tensor([[0.4], [1.6], [2.7], [3.3], [-1.2], [-3.1]])
    n_labeled = 4
    energy_batch, psi_batch = _exact_labels(mu_batch[:n_labeled], N_SITES)

    _, metrics = loss_fn(
        exact,
        mu_batch,
        H_base,
        H_mu_diag,
        Xi,
        epoch=0,
        energy_batch=energy_batch,
        psi_batch=psi_batch,
    )

    assert metrics["e"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["psi"] == pytest.approx(0.0, abs=1e-10)


def test_loss_psi_is_invariant_to_the_label_sign(model, operators) -> None:
    """Flipping psi_batch -> -psi_batch leaves loss_psi unchanged (detached align)."""
    loss_fn = SemiSupervisedFSMLoss(total_epochs=100, anneal_duration=10)
    H_base, H_mu_diag, Xi = operators
    torch.manual_seed(3)
    mu_batch = torch.rand(6, 1) * 8.0 - 4.0
    n_labeled = 4
    energy_batch, psi_batch = _exact_labels(mu_batch[:n_labeled], N_SITES)

    _, m_pos = loss_fn(
        model,
        mu_batch,
        H_base,
        H_mu_diag,
        Xi,
        epoch=0,
        energy_batch=energy_batch,
        psi_batch=psi_batch,
    )
    _, m_neg = loss_fn(
        model,
        mu_batch,
        H_base,
        H_mu_diag,
        Xi,
        epoch=0,
        energy_batch=energy_batch,
        psi_batch=-psi_batch,
    )

    assert m_pos["psi"] == pytest.approx(m_neg["psi"])


def test_physics_weight_anneal_schedule(model, mu_batch, operators) -> None:
    H_base, H_mu_diag, Xi = operators
    loss_fn = SemiSupervisedFSMLoss(total_epochs=100, anneal_duration=10)

    _, m0 = loss_fn(model, mu_batch, H_base, H_mu_diag, Xi, epoch=0)
    _, m_mid = loss_fn(model, mu_batch, H_base, H_mu_diag, Xi, epoch=5)
    _, m_end = loss_fn(model, mu_batch, H_base, H_mu_diag, Xi, epoch=10)
    _, m_past = loss_fn(model, mu_batch, H_base, H_mu_diag, Xi, epoch=9999)

    assert m0["physics_wt"] == pytest.approx(0.01)
    assert m_mid["physics_wt"] == pytest.approx(0.01 + 0.99 * 0.5)
    assert m_end["physics_wt"] == pytest.approx(1.0)
    assert m_past["physics_wt"] == pytest.approx(1.0)


def test_exact_lowest_eigenvector_drives_loss_to_gap_scale(loss_fn) -> None:
    """Label-free: the exact lowest eigenvector leaves fsm ~ lambda_1^2, var ~ 0."""
    H_base, H_mu_diag, Xi = _build_kitaev_operators(
        N_SITES, hopping=HOPPING, pairing=PAIRING
    )
    exact = _ExactLowestEigenvector(N_SITES)
    mu_batch = torch.tensor([[0.5], [1.5], [2.5], [3.5]], dtype=torch.float64)

    _, metrics = loss_fn(
        exact,
        mu_batch,
        H_base.double(),
        H_mu_diag.double(),
        Xi.double(),
        epoch=0,
    )

    lambda1_sq = np.mean(
        [
            np.linalg.eigvalsh((H_base + mu * H_mu_diag).numpy())[N_SITES] ** 2
            for mu in mu_batch.reshape(-1).tolist()
        ]
    )
    expected_fsm = lambda1_sq / (2 * N_SITES)

    assert metrics["var"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["fsm"] == pytest.approx(expected_fsm, rel=1e-6)
    assert metrics["lam_mean"] > 0.0


def test_xi_is_ignored(model, loss_fn, mu_batch, operators) -> None:
    """Xi must not affect the result; epoch still drives the anneal."""
    H_base, H_mu_diag, _ = operators
    dim = 2 * N_SITES

    torch.manual_seed(4)
    loss_a, metrics_a = loss_fn(
        model, mu_batch, H_base, H_mu_diag, torch.randn(dim, dim), epoch=0
    )
    torch.manual_seed(4)
    loss_b, metrics_b = loss_fn(
        model, mu_batch, H_base, H_mu_diag, torch.zeros(dim, dim), epoch=0
    )

    assert torch.equal(loss_a, loss_b)
    assert metrics_a == metrics_b


def test_gradients_flow_to_every_model_parameter(model, loss_fn, operators) -> None:
    H_base, H_mu_diag, Xi = operators
    torch.manual_seed(5)
    mu_batch = torch.rand(6, 1) * 8.0 - 4.0
    n_labeled = 4
    energy_batch, psi_batch = _exact_labels(mu_batch[:n_labeled], N_SITES)

    total_loss, _ = loss_fn(
        model,
        mu_batch,
        H_base,
        H_mu_diag,
        Xi,
        epoch=0,
        energy_batch=energy_batch,
        psi_batch=psi_batch,
    )
    total_loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no grad for {name}"
        assert torch.any(param.grad != 0), f"zero grad for {name}"


if __name__ == "__main__":
    pytest.main()
