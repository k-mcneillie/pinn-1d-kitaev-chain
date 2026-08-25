"""Tests for kitaev.training.loss.losses.PinnedFSMLoss."""

from __future__ import annotations

import math

import pytest
import torch

from kitaev.training.loss.losses import PinnedFSMLoss
from kitaev.training.trainer import _build_kitaev_operators, _ExampleSirenStandIn

N_SITES = 2
HIDDEN_FEATURES = 4
METRIC_KEYS = {"fsm", "var", "pin", "ph", "pin_wt"}


@pytest.fixture
def kitaev_operators_cpu() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _build_kitaev_operators(N_SITES, hopping=1.0, pairing=0.5)


@pytest.fixture
def tiny_model() -> _ExampleSirenStandIn:
    return _ExampleSirenStandIn(N_SITES, HIDDEN_FEATURES)


@pytest.fixture
def mu_batch() -> torch.Tensor:
    return torch.rand(6, 1) * 6.0 - 3.0


def test_pin_weight_before_anneal_duration(
    kitaev_operators_cpu, tiny_model, mu_batch
) -> None:
    H_base, H_mu_diag, Xi = kitaev_operators_cpu
    loss_fn = PinnedFSMLoss(total_epochs=100, anneal_duration=10)

    _, metrics = loss_fn(tiny_model, mu_batch, H_base, H_mu_diag, Xi, epoch=0)

    assert metrics["pin_wt"] == pytest.approx(1.0)


def test_pin_weight_midway_through_anneal(
    kitaev_operators_cpu, tiny_model, mu_batch
) -> None:
    H_base, H_mu_diag, Xi = kitaev_operators_cpu
    loss_fn = PinnedFSMLoss(total_epochs=100, anneal_duration=10)

    _, metrics = loss_fn(tiny_model, mu_batch, H_base, H_mu_diag, Xi, epoch=5)

    assert metrics["pin_wt"] == pytest.approx(1.0 - 0.99 * 0.5)


@pytest.mark.parametrize("epoch", [10, 50])
def test_pin_weight_at_and_after_anneal_duration(
    kitaev_operators_cpu, tiny_model, mu_batch, epoch: int
) -> None:
    H_base, H_mu_diag, Xi = kitaev_operators_cpu
    loss_fn = PinnedFSMLoss(total_epochs=100, anneal_duration=10)

    _, metrics = loss_fn(tiny_model, mu_batch, H_base, H_mu_diag, Xi, epoch=epoch)

    assert metrics["pin_wt"] == pytest.approx(0.01)


def test_returns_finite_total_loss_and_all_metric_keys(
    kitaev_operators_cpu, tiny_model, mu_batch
) -> None:
    H_base, H_mu_diag, Xi = kitaev_operators_cpu
    loss_fn = PinnedFSMLoss(total_epochs=100, anneal_duration=10)

    total_loss, metrics = loss_fn(tiny_model, mu_batch, H_base, H_mu_diag, Xi, epoch=3)

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.shape == ()
    assert torch.isfinite(total_loss)
    assert set(metrics) == METRIC_KEYS
    assert all(math.isfinite(value) for value in metrics.values())


def test_gradients_flow_to_model_parameters(
    kitaev_operators_cpu, tiny_model, mu_batch
) -> None:
    H_base, H_mu_diag, Xi = kitaev_operators_cpu
    loss_fn = PinnedFSMLoss(total_epochs=100, anneal_duration=10)

    total_loss, _ = loss_fn(tiny_model, mu_batch, H_base, H_mu_diag, Xi, epoch=3)
    total_loss.backward()

    grads = [p.grad for p in tiny_model.parameters() if p.grad is not None]
    assert grads
    assert any(torch.any(grad != 0) for grad in grads)
