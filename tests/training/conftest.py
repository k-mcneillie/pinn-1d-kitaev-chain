"""Shared fixtures for the src/kitaev/training/ test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from accelerate import Accelerator
from sesh import Session
from torch.utils.data import DataLoader, TensorDataset

from kitaev.training.trainer import _build_kitaev_operators, _ExampleSirenStandIn


@pytest.fixture
def accelerator() -> Accelerator:
    """A real, freshly constructed Accelerator for a single test."""
    return Accelerator()


@pytest.fixture
def n_sites() -> int:
    """The default lattice size used across trainer tests."""
    return 2


@pytest.fixture
def kitaev_operators(
    accelerator: Accelerator, n_sites: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Real Kitaev-chain operators, placed on the accelerator's device.

    Args:
        accelerator: The test's Accelerator, whose device the operators
            are moved to (mirroring what `main()` does in production).
        n_sites: Lattice size to build the operators for.

    Returns:
        Tuple of ``(H_base, H_mu_diag, Xi)`` on ``accelerator.device``.
    """
    H_base, H_mu_diag, Xi = _build_kitaev_operators(n_sites, hopping=1.0, pairing=0.5)
    return (
        H_base.to(accelerator.device),
        H_mu_diag.to(accelerator.device),
        Xi.to(accelerator.device),
    )


@pytest.fixture
def tiny_model_factory(accelerator: Accelerator):
    """Factory for small, real `_ExampleSirenStandIn` models on-device."""

    def _make(n_sites: int = 2, hidden_features: int = 4) -> _ExampleSirenStandIn:
        return _ExampleSirenStandIn(n_sites, hidden_features).to(accelerator.device)

    return _make


@pytest.fixture
def tiny_loader_factory():
    """Factory for small, real mu-only DataLoaders.

    Loaders are deliberately left on the CPU, unplaced: `fit()` calls
    `accelerator.prepare(loader)` itself, which is how batch placement
    happens in production too (see `main()`).
    """

    def _make(
        n_samples: int = 8,
        batch_size: int = 4,
        shuffle: bool = False,
        low: float = -3.0,
        high: float = 3.0,
    ) -> DataLoader:
        mu = torch.rand(n_samples, 1) * (high - low) + low
        return DataLoader(TensorDataset(mu), batch_size=batch_size, shuffle=shuffle)

    return _make


@pytest.fixture
def make_session(tmp_path: Path):
    """Factory for real, filesystem-isolated `sesh.Session` instances.

    `output_root=tmp_path` keeps every write inside pytest's per-test
    temporary directory instead of the real repository, and
    `enable_mlflow=False` avoids the mlflow filesystem-backend
    exception seen when running the real `main()` in this environment.
    Both are ordinary constructor parameters, not a mock.
    """

    def _make(name: str = "test-session", enable_mlflow: bool = False) -> Session:
        return Session(name=name, output_root=tmp_path, enable_mlflow=enable_mlflow)

    return _make
