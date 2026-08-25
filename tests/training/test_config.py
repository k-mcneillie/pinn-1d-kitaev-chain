"""Tests for kitaev.training.config.TrainerConfig."""

from __future__ import annotations

from kitaev.training.config import TrainerConfig


def test_default_values() -> None:
    config = TrainerConfig()
    assert config.epochs == 3000
    assert config.print_freq == 500
    assert config.patience is None
    assert config.grad_clip_norm == 1.0


def test_custom_values() -> None:
    config = TrainerConfig(epochs=10, print_freq=2, patience=5, grad_clip_norm=None)
    assert config.epochs == 10
    assert config.print_freq == 2
    assert config.patience == 5
    assert config.grad_clip_norm is None


def test_equality_is_value_based() -> None:
    assert TrainerConfig() == TrainerConfig()
    assert TrainerConfig(epochs=1) != TrainerConfig(epochs=2)
