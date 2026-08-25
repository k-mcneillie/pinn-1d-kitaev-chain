"""Tests for kitaev.training.loss.base.BaseLoss."""

from __future__ import annotations

import pytest
import torch

from kitaev.training.loss.base import BaseLoss


class _ConcreteLoss(BaseLoss):
    """Minimal real subclass that forwards to the abstract body."""

    def __call__(
        self,
        model: torch.nn.Module,
        mu_batch: torch.Tensor,
        H_base: torch.Tensor,
        H_mu_diag: torch.Tensor,
        Xi: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return super().__call__(model, mu_batch, H_base, H_mu_diag, Xi, epoch)


def test_call_raises_not_implemented() -> None:
    loss_fn = _ConcreteLoss()
    with pytest.raises(NotImplementedError):
        loss_fn(None, None, None, None, None, epoch=0)


def test_cannot_instantiate_base_loss_directly() -> None:
    with pytest.raises(TypeError):
        BaseLoss()
