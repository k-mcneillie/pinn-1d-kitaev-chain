from __future__ import annotations

import torch

from . import BaseLoss


class PinnedFSMLoss(BaseLoss):
    """Physics loss combining an FSM residual with a pinned-eigenstate term.

    Penalises deviation from a zero-energy eigenstate of the batched
    Hamiltonian (the FSM residual), while pinning the solution towards a
    non-negative Rayleigh quotient via a softplus penalty whose weight
    anneals from ``1.0`` down to ``0.01`` over ``anneal_duration``
    epochs. A Rayleigh-variance term and a particle-hole symmetry
    residual term are also included.

    Attributes:
        total_epochs: Total number of epochs the associated training
            run is expected to last.
        anneal_duration: Number of epochs over which ``pin_weight``
            decays from ``1.0`` to ``0.01``.
    """

    def __init__(
        self,
        total_epochs: int = 3000,
        anneal_duration: int = 2000,
    ) -> None:
        """Initialises the loss with its annealing schedule.

        Args:
            total_epochs: Total number of epochs the associated
                training run is expected to last.
            anneal_duration: Number of epochs over which ``pin_weight``
                decays from ``1.0`` to ``0.01``.
        """
        self.total_epochs = total_epochs
        self.anneal_duration = anneal_duration

    def __call__(
        self,
        model: torch.nn.Module,
        mu_batch: torch.Tensor,
        H_base: torch.Tensor,
        H_mu_diag: torch.Tensor,
        Xi: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the pinned-FSM loss and its component metrics for a batch.

        Args:
            model: The model being trained.
            mu_batch: Batch of mu values.
            H_base: Mu-independent part of the batched Hamiltonian.
            H_mu_diag: Diagonal matrix such that
                ``H_base + mu * H_mu_diag`` gives the full Hamiltonian
                at a given mu.
            Xi: Particle-hole symmetry operator.
            epoch: Current epoch number, used to anneal ``pin_weight``.

        Returns:
            Tuple of ``(total_loss, metrics)``, where ``metrics``
            contains the individual loss components plus the current
            ``pin_weight``.
        """
        if epoch < self.anneal_duration:
            pin_weight = 1.0 - 0.99 * (epoch / self.anneal_duration)
        else:
            pin_weight = 0.01

        psi_pred = model(mu_batch)
        H_batch = H_base.unsqueeze(0) + mu_batch.unsqueeze(-1) * H_mu_diag.unsqueeze(0)

        H_psi = torch.bmm(H_batch, psi_pred.unsqueeze(-1)).squeeze(-1)
        loss_fsm = torch.mean(H_psi**2)

        E_rayleigh = torch.sum(psi_pred * H_psi, dim=1, keepdim=True)
        loss_pin = torch.mean(torch.nn.functional.softplus(-E_rayleigh, beta=10.0))
        loss_var = torch.mean((H_psi - E_rayleigh * psi_pred) ** 2)

        Xi_psi = torch.matmul(psi_pred, Xi.T)
        H_Xi_psi = torch.bmm(H_batch, Xi_psi.unsqueeze(-1)).squeeze(-1)
        PH_res = H_Xi_psi + E_rayleigh * Xi_psi
        loss_ph = torch.mean(PH_res**2)

        total_loss = loss_fsm + loss_var + 0.1 * loss_ph + pin_weight * loss_pin

        return total_loss, {
            "fsm": loss_fsm.item(),
            "var": loss_var.item(),
            "pin": loss_pin.item(),
            "ph": loss_ph.item(),
            "pin_wt": pin_weight,
        }
