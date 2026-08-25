"""Tests for kitaev.training.loss.losses."""

from __future__ import annotations

import math

import pytest
import torch

from kitaev.models import SirenPINNDualHead
from kitaev.training.loss.losses import PinnedFSMLoss, SemiSupervisedLoss
from kitaev.training.trainer import _build_kitaev_operators, _ExampleSirenStandIn

N_SITES = 2
HIDDEN_FEATURES = 4
METRIC_KEYS = {"fsm", "var", "pin", "ph", "pin_wt"}
SEMI_SUPERVISED_METRIC_KEYS = {"e", "psi", "res", "ph", "physics_wt"}


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


# ---------------------------------------------------------------------------
# SemiSupervisedLoss
# ---------------------------------------------------------------------------


@pytest.fixture
def dual_head_model() -> SirenPINNDualHead:
    return SirenPINNDualHead(
        n_sites=2 * N_SITES, hidden_features=HIDDEN_FEATURES, hidden_layers=1
    )


@pytest.fixture
def labeled_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = torch.rand(3, 1) * 6.0 - 3.0
    energy = torch.rand(3, 1)
    psi = torch.nn.functional.normalize(torch.rand(3, 2 * N_SITES), dim=1)
    return mu, energy, psi


def test_physics_weight_before_anneal_duration(
    kitaev_operators_cpu, dual_head_model, mu_batch
) -> None:
    H_base, H_mu_diag, Xi = kitaev_operators_cpu
    loss_fn = SemiSupervisedLoss(total_epochs=100, anneal_duration=10)

    _, metrics = loss_fn(dual_head_model, mu_batch, H_base, H_mu_diag, Xi, epoch=0)

    assert metrics["physics_wt"] == pytest.approx(0.01)


def test_physics_weight_midway_through_anneal(
    kitaev_operators_cpu, dual_head_model, mu_batch
) -> None:
    H_base, H_mu_diag, Xi = kitaev_operators_cpu
    loss_fn = SemiSupervisedLoss(total_epochs=100, anneal_duration=10)

    _, metrics = loss_fn(dual_head_model, mu_batch, H_base, H_mu_diag, Xi, epoch=5)

    assert metrics["physics_wt"] == pytest.approx(0.01 + 0.99 * 0.5)


@pytest.mark.parametrize("epoch", [10, 50])
def test_physics_weight_at_and_after_anneal_duration(
    kitaev_operators_cpu, dual_head_model, mu_batch, epoch: int
) -> None:
    H_base, H_mu_diag, Xi = kitaev_operators_cpu
    loss_fn = SemiSupervisedLoss(total_epochs=100, anneal_duration=10)

    _, metrics = loss_fn(dual_head_model, mu_batch, H_base, H_mu_diag, Xi, epoch=epoch)

    assert metrics["physics_wt"] == pytest.approx(1.0)


def test_physics_weight_scales_res_and_ph_in_total_loss(
    kitaev_operators_cpu, dual_head_model, mu_batch
) -> None:
    H_base, H_mu_diag, Xi = kitaev_operators_cpu
    loss_fn = SemiSupervisedLoss(total_epochs=100, anneal_duration=10)

    total_loss, metrics = loss_fn(
        dual_head_model, mu_batch, H_base, H_mu_diag, Xi, epoch=0
    )

    # No labels given, so total_loss should be exactly physics_wt * (res + ph).
    expected = metrics["physics_wt"] * (metrics["res"] + metrics["ph"])
    assert total_loss.item() == pytest.approx(expected, rel=1e-5)


def test_returns_finite_total_loss_and_all_metric_keys_when_labelled(
    kitaev_operators_cpu, dual_head_model, mu_batch, labeled_batch
) -> None:
    H_base, H_mu_diag, Xi = kitaev_operators_cpu
    mu_labeled, energy, psi = labeled_batch
    loss_fn = SemiSupervisedLoss()

    total_loss, metrics = loss_fn(
        dual_head_model,
        torch.cat([mu_labeled, mu_batch], dim=0),
        H_base,
        H_mu_diag,
        Xi,
        epoch=0,
        energy_batch=energy,
        psi_batch=psi,
    )

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.shape == ()
    assert torch.isfinite(total_loss)
    assert set(metrics) == SEMI_SUPERVISED_METRIC_KEYS
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["e"] > 0.0
    assert metrics["psi"] > 0.0


def test_loss_psi_is_invariant_to_psi_batch_sign(
    kitaev_operators_cpu, dual_head_model, mu_batch, labeled_batch
) -> None:
    # An eigenvector's overall sign is arbitrary. loss_psi must therefore
    # report the same value whichever sign convention psi_batch happens to
    # use, since a psi_pred that is the correct eigenstate up to sign is not
    # actually wrong.
    H_base, H_mu_diag, Xi = kitaev_operators_cpu
    mu_labeled, energy, psi = labeled_batch
    mu_full = torch.cat([mu_labeled, mu_batch], dim=0)
    loss_fn = SemiSupervisedLoss()

    _, metrics_pos = loss_fn(
        dual_head_model,
        mu_full,
        H_base,
        H_mu_diag,
        Xi,
        epoch=0,
        energy_batch=energy,
        psi_batch=psi,
    )
    _, metrics_neg = loss_fn(
        dual_head_model,
        mu_full,
        H_base,
        H_mu_diag,
        Xi,
        epoch=0,
        energy_batch=energy,
        psi_batch=-psi,
    )

    assert metrics_pos["psi"] == pytest.approx(metrics_neg["psi"])


def test_data_terms_are_zero_without_labels(
    kitaev_operators_cpu, dual_head_model, mu_batch
) -> None:
    H_base, H_mu_diag, Xi = kitaev_operators_cpu
    loss_fn = SemiSupervisedLoss()

    total_loss, metrics = loss_fn(
        dual_head_model, mu_batch, H_base, H_mu_diag, Xi, epoch=0
    )

    assert torch.isfinite(total_loss)
    assert metrics["e"] == 0.0
    assert metrics["psi"] == 0.0
    assert metrics["res"] > 0.0


def test_semi_supervised_gradients_flow_to_model_parameters(
    kitaev_operators_cpu, dual_head_model, mu_batch, labeled_batch
) -> None:
    H_base, H_mu_diag, Xi = kitaev_operators_cpu
    mu_labeled, energy, psi = labeled_batch
    loss_fn = SemiSupervisedLoss()

    total_loss, _ = loss_fn(
        dual_head_model,
        torch.cat([mu_labeled, mu_batch], dim=0),
        H_base,
        H_mu_diag,
        Xi,
        epoch=0,
        energy_batch=energy,
        psi_batch=psi,
    )
    total_loss.backward()

    grads = [p.grad for p in dual_head_model.parameters() if p.grad is not None]
    assert grads
    assert any(torch.any(grad != 0) for grad in grads)
