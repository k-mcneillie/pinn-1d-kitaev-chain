# src/kitaev/training/callbacks.py
"""Epoch-boundary hooks for :class:`kitaev.training.trainer.UnifiedTrainer`.

A callback is a small, optional object the trainer invokes once at the
start and once at the end of every epoch. Both hook methods default to
no-ops, so a subclass overrides only the one it needs. This is the
extension point used by the streaming / curriculum / adaptive sampling
strategies in :mod:`kitaev.data.streaming` -- each is a callback that
updates what the next epoch samples -- and it keeps that logic out of the
trainer's own epoch loop.
"""

from __future__ import annotations

import torch

from .utils import TrainingHistory


class TrainingCallback:
    """Base class for objects hooked into the trainer's epoch loop.

    Subclasses override :meth:`on_epoch_start` and/or :meth:`on_epoch_end`.
    The trainer calls every callback in the order it was given, passing the
    (accelerate-unwrapped) model and the run's :class:`TrainingHistory` so a
    callback can inspect progress so far.
    """

    def on_epoch_start(
        self,
        epoch: int,
        model: torch.nn.Module,
        history: TrainingHistory,
    ) -> None:
        """Called immediately before an epoch's training pass.

        Args:
            epoch: The 1-based epoch about to run.
            model: The model being trained, with any accelerate wrapping
                removed.
            history: Per-epoch metric history for the run so far.
        """

    def on_epoch_end(
        self,
        epoch: int,
        model: torch.nn.Module,
        history: TrainingHistory,
    ) -> None:
        """Called after an epoch's training and validation passes.

        Args:
            epoch: The 1-based epoch that just finished.
            model: The model being trained, with any accelerate wrapping
                removed.
            history: Per-epoch metric history including the epoch just
                recorded.
        """
