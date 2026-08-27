"""Tests for kitaev.training.loss.losses.ChiralFSMLoss."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kitaev.analytical import chiral_block, chiral_block_batched
from kitaev.models import SirenPINNChiral
from kitaev.training.loss.losses import ChiralFSMLoss

N_SITES = 6
HOPPING = 1.0
PAIRING = 0.5
METRIC_KEYS = {"fsm", "var", "lam_mean"}


@pytest.fixture
def model() -> SirenPINNChiral:
    torch.manual_seed(0)
    return SirenPINNChiral(n_sites=N_SITES, hidden_features=8, hidden_layers=1)


@pytest.fixture
def loss_fn() -> ChiralFSMLoss:
    return ChiralFSMLoss(n_sites=N_SITES, hopping=HOPPING, pairing=PAIRING)


@pytest.fixture
def mu_batch() -> torch.Tensor:
    return torch.rand(8, 1) * 3.9 + 0.05


class _ExactSmallestTriple(torch.nn.Module):
    """Returns the exact smallest singular pair of h(mu) for each mu."""

    def __init__(self, n_sites: int) -> None:
        super().__init__()
        self.n_sites = n_sites

    def forward(self, mu_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left_vectors = []
        right_vectors = []
        for mu in mu_batch.reshape(-1).tolist():
            block = chiral_block(mu, self.n_sites, HOPPING, PAIRING)
            left, singular, right_t = np.linalg.svd(block)
            k = int(np.argmin(singular))
            left_vectors.append(left[:, k])
            right_vectors.append(right_t[k, :])
        return (
            torch.tensor(np.array(left_vectors), dtype=mu_batch.dtype),
            torch.tensor(np.array(right_vectors), dtype=mu_batch.dtype),
        )


def test_chiral_block_batched_matches_numpy_reference() -> None:
    mu_batch = torch.tensor([[0.3], [1.7], [-2.4]])
    batched = chiral_block_batched(mu_batch, N_SITES, HOPPING, PAIRING)

    assert batched.shape == (3, N_SITES, N_SITES)
    for row, mu in enumerate(mu_batch.reshape(-1).tolist()):
        reference = chiral_block(mu, N_SITES, HOPPING, PAIRING)
        assert np.allclose(batched[row].numpy(), reference)


def test_returns_scalar_loss_and_all_metric_keys(model, loss_fn, mu_batch) -> None:
    total_loss, metrics = loss_fn(model, mu_batch, None, None, None, epoch=0)

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.shape == ()
    assert torch.isfinite(total_loss)
    assert set(metrics) == METRIC_KEYS
    assert all(math.isfinite(value) for value in metrics.values())


def test_gradients_flow_to_model_parameters(model, loss_fn, mu_batch) -> None:
    total_loss, _ = loss_fn(model, mu_batch, None, None, None, epoch=0)
    total_loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert any(torch.any(grad != 0) for grad in grads)


def test_hamiltonian_operator_arguments_are_ignored(model, loss_fn, mu_batch) -> None:
    """H_base / H_mu_diag / Xi / epoch must not affect the result."""
    dim = 2 * N_SITES
    zeros = torch.zeros(dim, dim)
    junk = torch.randn(dim, dim)

    torch.manual_seed(1)
    loss_a, metrics_a = loss_fn(model, mu_batch, zeros, zeros, zeros, epoch=0)
    torch.manual_seed(1)
    loss_b, metrics_b = loss_fn(model, mu_batch, junk, junk, junk, epoch=9999)

    assert torch.equal(loss_a, loss_b)
    assert metrics_a == metrics_b


def test_exact_smallest_triple_drives_loss_to_gap_scale(loss_fn) -> None:
    """Feeding the exact smallest singular pair leaves only lambda_min^2."""
    exact = _ExactSmallestTriple(N_SITES)
    mu_batch = torch.tensor([[0.5], [1.5], [2.5], [3.5]], dtype=torch.float64)

    total_loss, metrics = loss_fn(exact, mu_batch, None, None, None, epoch=0)

    lambda_min_sq = np.mean(
        [
            np.linalg.svd(
                chiral_block(mu, N_SITES, HOPPING, PAIRING), compute_uv=False
            ).min()
            ** 2
            for mu in mu_batch.reshape(-1).tolist()
        ]
    )

    # var vanishes for a genuine singular pair. fsm sums two mean-squared
    # terms; for an exact pair each is mean_b(lambda_b^2) / n_sites (the
    # torch.mean is over the batch*n_sites elements), one from ||h v|| and
    # one from ||h^T u||.
    expected_fsm = 2.0 * lambda_min_sq / N_SITES
    assert metrics["var"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["fsm"] == pytest.approx(expected_fsm, rel=1e-6)
    assert total_loss.item() == pytest.approx(expected_fsm, rel=1e-6)
    assert metrics["lam_mean"] > 0.0


if __name__ == "__main__":
    pytest.main()
