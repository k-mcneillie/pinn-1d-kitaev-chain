from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseLoss(ABC):
    """Abstract base for all PINN physics losses used by :class:`UnifiedTrainer`.

    Subclasses receive the model, a batch of mu values, and the three
    constant Hamiltonian pieces (``H_base``, ``H_mu_diag``, ``Xi``), and
    must return a scalar loss tensor plus a dictionary of scalar metrics
    for logging. This signature matches the existing loss classes
    (``PinnedFSMLoss``, ``ShiftInvertLoss``, etc.) unchanged, so they can
    be used with this trainer without modification.
    """

    @abstractmethod
    def __call__(
        self,
        model: torch.nn.Module,
        mu_batch: torch.Tensor,
        H_base: torch.Tensor,
        H_mu_diag: torch.Tensor,
        Xi: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the loss and metrics for a single batch.

        Args:
            model: The model being trained.
            mu_batch: Batch of mu values.
            H_base: Mu-independent part of the batched Hamiltonian.
            H_mu_diag: Diagonal matrix such that
                ``H_base + mu * H_mu_diag`` gives the full Hamiltonian
                at a given mu.
            Xi: Particle-hole symmetry operator.
            epoch: Current epoch number, for losses whose weighting
                anneals over the course of training.

        Returns:
            Tuple of ``(loss, metrics)``, where ``metrics`` is a
            dictionary of scalar values for logging.
        """
        raise NotImplementedError
