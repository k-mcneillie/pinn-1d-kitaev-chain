"""Tests for kitaev.training.loss.losses.NambuFSMLoss."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kitaev.models import SirenPINN, SirenPINNNambuFolded
from kitaev.training.loss.losses import NambuFSMLoss
from kitaev.training.trainer import _build_kitaev_operators

N_SITES = 4
HOPPING = 1.0
PAIRING = 0.5
METRIC_KEYS = {"fsm", "var", "lam_mean"}


@pytest.fixture
def operators() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _build_kitaev_operators(N_SITES, hopping=HOPPING, pairing=PAIRING)


@pytest.fixture
def model() -> SirenPINN:
    torch.manual_seed(0)
    return SirenPINN(n_sites=2 * N_SITES, hidden_features=8, hidden_layers=1)


@pytest.fixture
def loss_fn() -> NambuFSMLoss:
    return NambuFSMLoss()


@pytest.fixture
def mu_batch() -> torch.Tensor:
    return torch.rand(8, 1) * 8.0 - 4.0


class _ExactLowestEigenvector(torch.nn.Module):
    """Returns the exact lowest non-negative BdG eigenvector for each mu."""

    def __init__(self, n_sites: int) -> None:
        super().__init__()
        self.n_sites = n_sites

    def forward(self, mu_batch: torch.Tensor) -> torch.Tensor:
        H_base, H_mu_diag, _ = _build_kitaev_operators(
            self.n_sites, hopping=HOPPING, pairing=PAIRING
        )
        vectors = []
        for mu in mu_batch.reshape(-1).tolist():
            h = (H_base + mu * H_mu_diag).numpy()
            _, eigvecs = np.linalg.eigh(h)
            vectors.append(eigvecs[:, self.n_sites])  # first non-negative branch
        return torch.tensor(np.array(vectors), dtype=mu_batch.dtype)


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


def test_total_loss_is_fsm_plus_var(model, loss_fn, mu_batch, operators) -> None:
    H_base, H_mu_diag, Xi = operators
    total_loss, metrics = loss_fn(model, mu_batch, H_base, H_mu_diag, Xi, epoch=0)
    assert total_loss.item() == pytest.approx(metrics["fsm"] + metrics["var"])


def test_gradients_flow_to_model_parameters(
    model, loss_fn, mu_batch, operators
) -> None:
    H_base, H_mu_diag, Xi = operators
    total_loss, _ = loss_fn(model, mu_batch, H_base, H_mu_diag, Xi, epoch=0)
    total_loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert any(torch.any(grad != 0) for grad in grads)


def test_xi_and_epoch_are_ignored(model, loss_fn, mu_batch, operators) -> None:
    """Xi and epoch must not affect the result."""
    H_base, H_mu_diag, _ = operators
    dim = 2 * N_SITES
    junk = torch.randn(dim, dim)

    torch.manual_seed(1)
    loss_a, metrics_a = loss_fn(model, mu_batch, H_base, H_mu_diag, junk, epoch=0)
    torch.manual_seed(1)
    loss_b, metrics_b = loss_fn(
        model, mu_batch, H_base, H_mu_diag, torch.zeros(dim, dim), epoch=9999
    )

    assert torch.equal(loss_a, loss_b)
    assert metrics_a == metrics_b


def test_loss_is_invariant_under_particle_hole_swap(loss_fn, operators) -> None:
    """fsm + var is unchanged by psi -> Xi psi -- the property that lets

    loss_pin be dropped. Xi psi sends E_R -> -E_R but leaves both terms
    numerically identical.
    """
    H_base, H_mu_diag, Xi = operators
    mu_batch = torch.tensor([[0.3], [1.7], [-2.4], [3.5]], dtype=torch.float64)

    torch.manual_seed(2)
    raw = torch.randn(mu_batch.shape[0], 2 * N_SITES, dtype=torch.float64)
    psi = torch.nn.functional.normalize(raw, p=2, dim=1)

    class _Fixed(torch.nn.Module):
        def __init__(self, out: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("out", out)

        def forward(self, _: torch.Tensor) -> torch.Tensor:
            return self.out

    xi64 = Xi.to(torch.float64)
    loss_psi, m_psi = loss_fn(
        _Fixed(psi), mu_batch, H_base.double(), H_mu_diag.double(), Xi, epoch=0
    )
    loss_xi, m_xi = loss_fn(
        _Fixed(psi @ xi64.T),
        mu_batch,
        H_base.double(),
        H_mu_diag.double(),
        Xi,
        epoch=0,
    )

    assert loss_psi.item() == pytest.approx(loss_xi.item(), rel=1e-10, abs=1e-12)
    assert m_psi["fsm"] == pytest.approx(m_xi["fsm"], rel=1e-10, abs=1e-12)
    assert m_psi["var"] == pytest.approx(m_xi["var"], rel=1e-10, abs=1e-12)


def test_exact_lowest_eigenvector_drives_loss_to_gap_scale(loss_fn) -> None:
    """Feeding the exact lowest eigenvector leaves fsm ~ lambda_1^2 and var ~ 0."""
    H_base, H_mu_diag, Xi = _build_kitaev_operators(
        N_SITES, hopping=HOPPING, pairing=PAIRING
    )
    exact = _ExactLowestEigenvector(N_SITES)
    mu_batch = torch.tensor([[0.5], [1.5], [2.5], [3.5]], dtype=torch.float64)

    total_loss, metrics = loss_fn(
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
    # torch.mean(H_psi ** 2) averages over batch * 2N elements, so the exact
    # value is mean_b(lambda_1(mu_b)^2) / (2 N).
    expected_fsm = lambda1_sq / (2 * N_SITES)

    assert metrics["var"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["fsm"] == pytest.approx(expected_fsm, rel=1e-6)
    assert total_loss.item() == pytest.approx(expected_fsm, rel=1e-6)
    assert metrics["lam_mean"] > 0.0


def test_reflection_folded_model_makes_loss_even_in_mu(loss_fn, operators) -> None:
    """With SirenPINNNambuFolded the per-sample loss at +mu equals that at -mu."""
    H_base, H_mu_diag, Xi = operators
    torch.manual_seed(3)
    folded = SirenPINNNambuFolded(
        n_sites=2 * N_SITES, hidden_features=8, hidden_layers=1
    )
    mu = torch.linspace(0.2, 4.0, 6).unsqueeze(-1)

    _, m_pos = loss_fn(folded, mu, H_base, H_mu_diag, Xi, epoch=0)
    _, m_neg = loss_fn(folded, -mu, H_base, H_mu_diag, Xi, epoch=0)

    assert m_pos["fsm"] == pytest.approx(m_neg["fsm"], rel=1e-5, abs=1e-9)
    assert m_pos["var"] == pytest.approx(m_neg["var"], rel=1e-5, abs=1e-9)
    assert m_pos["lam_mean"] == pytest.approx(m_neg["lam_mean"], rel=1e-5, abs=1e-9)


if __name__ == "__main__":
    pytest.main()
