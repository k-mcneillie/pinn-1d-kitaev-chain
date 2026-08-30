"""Tests for kitaev.training.loss.losses.ChiralSVDLoss."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kitaev.analytical import KitaevChainHamiltonian, chiral_block
from kitaev.models import SirenPINNChiralFull
from kitaev.training.loss.losses import ChiralSVDLoss, chiral_svd_pointwise_residual

N_SITES = 6
HOPPING = 1.0
PAIRING = 0.5
METRIC_KEYS = {
    "svd",
    "sigma_min_mean",
    "sigma_max_mean",
    "ortho_u",
    "gauge",
    "gauge_wt",
}


@pytest.fixture
def model() -> SirenPINNChiralFull:
    torch.manual_seed(0)
    return SirenPINNChiralFull(n_sites=N_SITES, hidden_features=8, hidden_layers=1)


@pytest.fixture
def loss_fn() -> ChiralSVDLoss:
    return ChiralSVDLoss(n_sites=N_SITES, hopping=HOPPING, pairing=PAIRING)


@pytest.fixture
def mu_batch() -> torch.Tensor:
    return torch.rand(8, 1) * 3.9 + 0.05


class _ExactSVD(torch.nn.Module):
    """Returns the exact SVD frames (U, sigma, V) of h(mu) for each mu."""

    def __init__(self, n_sites: int) -> None:
        super().__init__()
        self.n_sites = n_sites

    def forward(
        self, mu_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u_list, s_list, v_list = [], [], []
        for mu in mu_batch.reshape(-1).tolist():
            block = chiral_block(mu, self.n_sites, HOPPING, PAIRING)
            left, singular, right_t = np.linalg.svd(block)
            u_list.append(left)
            s_list.append(singular)
            v_list.append(right_t.T)
        dtype = mu_batch.dtype
        return (
            torch.tensor(np.array(u_list), dtype=dtype),
            torch.tensor(np.array(s_list), dtype=dtype),
            torch.tensor(np.array(v_list), dtype=dtype),
        )


class _ConstantFrame(torch.nn.Module):
    """Returns a fixed (U, sigma, V) regardless of mu -- a smooth frame."""

    def __init__(self, n_sites: int) -> None:
        super().__init__()
        self.n_sites = n_sites
        block = chiral_block(1.0, n_sites, HOPPING, PAIRING)
        left, singular, right_t = np.linalg.svd(block)
        self.register_buffer("u_fixed", torch.tensor(left, dtype=torch.float64))
        self.register_buffer("s_fixed", torch.tensor(singular, dtype=torch.float64))
        self.register_buffer("v_fixed", torch.tensor(right_t.T, dtype=torch.float64))

    def forward(
        self, mu_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = mu_batch.shape[0]
        return (
            self.u_fixed.expand(batch, self.n_sites, self.n_sites),
            self.s_fixed.expand(batch, self.n_sites),
            self.v_fixed.expand(batch, self.n_sites, self.n_sites),
        )


def test_returns_scalar_loss_and_all_metric_keys(model, loss_fn, mu_batch) -> None:
    total_loss, metrics = loss_fn(model, mu_batch, None, None, None, epoch=0)

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.shape == ()
    assert torch.isfinite(total_loss)
    assert set(metrics) == METRIC_KEYS
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["gauge_wt"] == 0.0
    assert metrics["gauge"] == 0.0


def test_gradients_flow_to_model_parameters(model, loss_fn, mu_batch) -> None:
    total_loss, _ = loss_fn(model, mu_batch, None, None, None, epoch=0)
    total_loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert any(torch.any(grad != 0) for grad in grads)


def test_hamiltonian_operator_arguments_are_ignored(model, loss_fn, mu_batch) -> None:
    dim = 2 * N_SITES
    zeros = torch.zeros(dim, dim)
    junk = torch.randn(dim, dim)

    loss_a, metrics_a = loss_fn(model, mu_batch, zeros, zeros, zeros, epoch=0)
    loss_b, metrics_b = loss_fn(model, mu_batch, junk, junk, junk, epoch=9999)

    assert torch.equal(loss_a, loss_b)
    assert metrics_a == metrics_b


def test_exact_svd_drives_residual_and_orthogonality_to_zero(loss_fn) -> None:
    exact = _ExactSVD(N_SITES)
    mu_batch = torch.tensor([[0.5], [1.5], [2.5], [3.5]], dtype=torch.float64)

    total_loss, metrics = loss_fn(exact, mu_batch, None, None, None, epoch=0)

    assert metrics["svd"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["ortho_u"] == pytest.approx(0.0, abs=1e-10)
    assert total_loss.item() == pytest.approx(0.0, abs=1e-10)
    assert metrics["sigma_min_mean"] > 0.0


def test_exact_frames_reproduce_the_full_bdg_spectrum() -> None:
    """sort(cat([sigma, -sigma])) from the exact frames matches eigvalsh(H)."""
    exact = _ExactSVD(N_SITES)
    ham = KitaevChainHamiltonian(n_sites=N_SITES, hopping=HOPPING, pairing=PAIRING)

    for mu in (0.5, 1.5, 2.5, 3.5):
        _u, sigma, _v = exact(torch.tensor([[mu]], dtype=torch.float64))
        predicted = torch.sort(torch.cat([sigma, -sigma], dim=1), dim=1).values
        exact_spectrum = np.sort(np.linalg.eigvalsh(ham.build(mu)))
        assert np.allclose(predicted.numpy().ravel(), exact_spectrum, atol=1e-6)


def test_gauge_term_is_active_only_when_weighted(model, mu_batch) -> None:
    weighted = ChiralSVDLoss(
        n_sites=N_SITES, hopping=HOPPING, pairing=PAIRING, gauge_weight=1e-2
    )
    total_loss, metrics = weighted(model, mu_batch, None, None, None, epoch=0)

    assert metrics["gauge_wt"] == 1e-2
    assert metrics["gauge"] > 0.0
    assert total_loss.item() > metrics["svd"]

    # A frame that does not move with mu makes the smoothness term vanish.
    constant = _ConstantFrame(N_SITES)
    mu64 = mu_batch.double()
    _total, const_metrics = weighted(constant, mu64, None, None, None, epoch=0)
    assert const_metrics["gauge"] == pytest.approx(0.0, abs=1e-12)


def test_pointwise_residual_shape_and_exact_frames(loss_fn) -> None:
    del loss_fn
    exact = _ExactSVD(N_SITES)
    mu_batch = torch.tensor([[0.4], [1.9], [3.3]], dtype=torch.float64)

    residual = chiral_svd_pointwise_residual(
        exact, mu_batch, N_SITES, hopping=HOPPING, pairing=PAIRING
    )

    assert residual.shape == (3,)
    assert torch.all(residual >= 0.0)
    assert torch.allclose(residual, torch.zeros(3, dtype=torch.float64), atol=1e-10)


if __name__ == "__main__":
    pytest.main()
