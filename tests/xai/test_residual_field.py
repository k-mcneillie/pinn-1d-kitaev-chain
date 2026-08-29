"""Tests for kitaev.xai.residual_field."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kitaev.models import SirenPINN, SirenPINNChiral, SirenPINNDualHead
from kitaev.xai.loading import psi_only
from kitaev.xai.residual_field import sweep_residual_field

N_SITES = 6
MU_GRID = np.linspace(-3.0, 3.0, 15)


def _chiral_models(count: int) -> list[SirenPINNChiral]:
    models = []
    for seed in range(count):
        torch.manual_seed(seed)
        models.append(
            SirenPINNChiral(n_sites=N_SITES, hidden_features=8, hidden_layers=1)
        )
    return models


def test_chiral_basis_residual_over_seeds() -> None:
    field = sweep_residual_field(
        _chiral_models(3),
        basis="chiral",
        label="chiral",
        n_sites=N_SITES,
        hopping=1.0,
        pairing=0.5,
        mu_grid=MU_GRID,
    )

    assert field.residual_median.shape == (15,)
    assert field.n_seeds == 3
    assert np.all(field.residual_min >= 0.0)
    assert np.all(field.residual_min <= field.residual_median)
    assert np.all(field.residual_median <= field.residual_max)
    assert field.basis == "chiral"


def test_single_model_has_a_zero_width_band() -> None:
    field = sweep_residual_field(
        _chiral_models(1),
        basis="chiral",
        label="chiral",
        n_sites=N_SITES,
        hopping=1.0,
        pairing=0.5,
        mu_grid=MU_GRID,
    )

    assert field.n_seeds == 1
    assert field.residual_min == pytest.approx(field.residual_max)
    assert field.residual_min == pytest.approx(field.residual_median)


def test_nambu_basis_accepts_wrapped_dual_heads() -> None:
    models = []
    for seed in range(2):
        torch.manual_seed(seed)
        dual = SirenPINNDualHead(
            n_sites=2 * N_SITES, hidden_features=8, hidden_layers=1
        )
        models.append(psi_only(dual))

    field = sweep_residual_field(
        models,
        basis="nambu",
        label="dual",
        n_sites=N_SITES,
        hopping=1.0,
        pairing=0.5,
        mu_grid=MU_GRID,
    )

    assert field.residual_median.shape == (15,)
    assert field.n_seeds == 2


def test_rejects_unknown_basis_and_empty_models() -> None:
    torch.manual_seed(0)
    model = SirenPINN(n_sites=2 * N_SITES, hidden_features=8, hidden_layers=1)

    with pytest.raises(ValueError, match="basis must be one of"):
        sweep_residual_field(
            [model],
            basis="other",
            label="",
            n_sites=N_SITES,
            hopping=1.0,
            pairing=0.5,
            mu_grid=MU_GRID,
        )
    with pytest.raises(ValueError, match="at least one model"):
        sweep_residual_field(
            [],
            basis="nambu",
            label="",
            n_sites=N_SITES,
            hopping=1.0,
            pairing=0.5,
            mu_grid=MU_GRID,
        )
