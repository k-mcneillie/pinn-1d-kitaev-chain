"""Tests for kitaev.xai.loading."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from kitaev.models import SirenPINN, SirenPINNDualHead
from kitaev.xai.loading import (
    load_seed_checkpoints,
    psi_only,
    read_comparison_errors,
)


def _build() -> SirenPINN:
    return SirenPINN(n_sites=12, hidden_features=8, hidden_layers=1)


def test_read_comparison_errors_averages_over_seeds(tmp_path: Path) -> None:
    csv_path = tmp_path / "comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "seed", "e_mae_topological", "e_mae_trivial"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "chiral",
                "seed": 0,
                "e_mae_topological": 1e-4,
                "e_mae_trivial": 6e-6,
            }
        )
        writer.writerow(
            {
                "model": "chiral",
                "seed": 1,
                "e_mae_topological": 3e-4,
                "e_mae_trivial": 1e-5,
            }
        )

    errors = read_comparison_errors(csv_path)

    assert set(errors) == {"chiral"}
    assert errors["chiral"]["trivial"] == (6e-6 + 1e-5) / 2
    assert errors["chiral"]["topological"] == (1e-4 + 3e-4) / 2


def test_read_comparison_errors_accepts_a_session_directory(tmp_path: Path) -> None:
    with (tmp_path / "four_model_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "seed", "e_mae_topological", "e_mae_trivial"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "chiral",
                "seed": 0,
                "e_mae_topological": 1e-4,
                "e_mae_trivial": 6e-6,
            }
        )

    errors = read_comparison_errors(tmp_path)

    assert errors["chiral"]["trivial"] == 6e-6


def test_load_seed_checkpoints_round_trips(tmp_path: Path) -> None:
    model_dir = tmp_path / "chiral"
    model_dir.mkdir()
    for seed in (0, 1):
        torch.save(_build().state_dict(), model_dir / f"seed_{seed}.pt")

    models = load_seed_checkpoints(_build, model_dir)

    assert len(models) == 2
    assert all(not model.training for model in models)


def test_load_seed_checkpoints_missing_dir_is_empty(tmp_path: Path) -> None:
    assert load_seed_checkpoints(_build, tmp_path / "nothing_here") == []


def test_psi_only_passthrough_and_unwrap() -> None:
    plain = _build()
    assert psi_only(plain) is plain

    dual = SirenPINNDualHead(n_sites=12, hidden_features=8, hidden_layers=1)
    wrapped = psi_only(dual)
    assert wrapped is not dual
    assert wrapped(torch.zeros(3, 1)).shape == (3, 12)
