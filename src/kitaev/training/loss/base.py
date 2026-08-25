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

    A subclass that trains semi-supervised (e.g. ``SemiSupervisedLoss``)
    may additionally declare two optional, keyword-only parameters,
    ``energy_batch`` and ``psi_batch``, defaulting to ``None``.
    ``UnifiedTrainer`` forwards a batch's exact labels through these two
    keywords whenever its dataloader supplies them, and omits them
    entirely otherwise — so a subclass that does not declare them (like
    ``PinnedFSMLoss``) is never called with them and needs no changes.
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
