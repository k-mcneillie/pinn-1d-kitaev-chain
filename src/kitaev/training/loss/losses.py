from __future__ import annotations

import torch

from kitaev.analytical import chiral_block_batched

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


class NambuFSMLoss(BaseLoss):
    """Minimal folded-spectrum loss on the Nambu-basis BdG Hamiltonian.

    The two-term, schedule-free physics loss for a single-head model that
    returns a ``2N`` Nambu eigenvector (:class:`kitaev.models.SirenPINN` or
    :class:`kitaev.models.SirenPINNNambuFolded`). It is
    :class:`PinnedFSMLoss` with ``loss_ph`` and ``loss_pin`` removed -- the
    Nambu-basis counterpart of :class:`ChiralFSMLoss`, which is already
    exactly this shape on the ``N x N`` chiral block::

        loss = loss_fsm + loss_var

    with no relative weight and no annealing schedule:

        loss_fsm:
            ``mean(||H(mu) psi||^2)``. The folded-spectrum residual: its
            minimiser over the unit sphere is the eigenvector of the
            eigenvalue nearest zero. Identical to :class:`PinnedFSMLoss`'s
            ``loss_fsm``.
        loss_var:
            ``mean(||H(mu) psi - E_R psi||^2)`` with
            ``E_R = psi^T H(mu) psi`` the Rayleigh quotient. Forces ``psi``
            to be a genuine eigenvector, so ``E_R`` is a meaningful energy.
            Identical to :class:`PinnedFSMLoss`'s ``loss_var``.

    Dropped relative to :class:`PinnedFSMLoss`:

    - ``loss_ph``: ``Xi H Xi = -H`` with ``Xi`` orthogonal makes the
      particle-hole residual numerically equal to ``loss_var`` term by
      term, so it carries no independent gradient (see
      ``docs/markdown/particle-hole-redundancy.md``).
    - ``loss_pin``: ``loss_fsm + loss_var`` is exactly invariant under
      ``psi -> Xi psi`` (which sends ``E_R -> -E_R``), so the ``+-E`` branch
      is an unbroken global gauge. It is resolved at *evaluation* time by
      sign-alignment to the reference, not by a training penalty. The
      reported ``lam_mean`` is therefore ``mean(|E_R|)``.

    ``Xi`` and ``epoch`` are accepted to match the :class:`BaseLoss` call
    contract but are unused: this loss has no ``Xi`` term and no schedule.
    """

    def __init__(self) -> None:
        """Initialise the loss. Takes no arguments: no schedule, no weights."""

    def __call__(
        self,
        model: torch.nn.Module,
        mu_batch: torch.Tensor,
        H_base: torch.Tensor,
        H_mu_diag: torch.Tensor,
        Xi: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the folded-spectrum loss and its components for a batch.

        Args:
            model: The model being trained. Must return a
                ``(batch_size, 2N)`` unit-norm Nambu eigenvector when
                called on ``mu_batch``.
            mu_batch: Batch of mu values, shape ``(batch_size, 1)``.
            H_base: Mu-independent part of the batched Hamiltonian.
            H_mu_diag: Diagonal matrix such that
                ``H_base + mu * H_mu_diag`` gives ``H(mu)``.
            Xi: Unused (kept for the :class:`BaseLoss` contract).
            epoch: Unused (this loss has no annealing schedule).

        Returns:
            Tuple of ``(total_loss, metrics)``, where ``metrics`` holds the
            two loss components and the mean absolute Rayleigh-quotient
            energy.
        """
        del Xi, epoch

        psi_pred = model(mu_batch)
        H_batch = H_base.unsqueeze(0) + mu_batch.unsqueeze(-1) * H_mu_diag.unsqueeze(0)

        H_psi = torch.bmm(H_batch, psi_pred.unsqueeze(-1)).squeeze(-1)
        loss_fsm = torch.mean(H_psi**2)

        E_rayleigh = torch.sum(psi_pred * H_psi, dim=1, keepdim=True)
        loss_var = torch.mean((H_psi - E_rayleigh * psi_pred) ** 2)

        total_loss = loss_fsm + loss_var

        return total_loss, {
            "fsm": loss_fsm.item(),
            "var": loss_var.item(),
            "lam_mean": E_rayleigh.abs().mean().item(),
        }


class SemiSupervisedLoss(BaseLoss):
    """Semi-supervised loss combining exact-label data terms with FSM/PH residuals.

    Intended for a dual-head model (e.g. ``SirenPINNDualHead``) that
    predicts both an energy ``E_pred`` and an eigenvector ``psi_pred``
    for each mu. ``mu_batch`` is expected to be the concatenation of a
    labelled block followed by a label-free block: the first
    ``energy_batch.shape[0]`` rows have known ``(E, psi)`` targets, and
    any remaining rows do not. The physics residual terms are evaluated
    over the *entire* batch, since they require no ground truth, while
    the data terms are restricted to the labelled rows only.

    The four terms, following the semi-supervised formulation this
    project's exploratory notebooks converged on, are:

        loss = loss_e + loss_psi + physics_weight * (loss_res + loss_ph)

    Args:

        loss_e:
            MSE between predicted and exact energy, on labelled rows.
        loss_psi:
            MSE between predicted and exact eigenvector, on labelled
            rows, with ``psi_batch``'s sign aligned to ``psi_pred``'s
            per row before the comparison — an eigenvector is only
            defined up to an overall sign, and nothing else in this
            loss breaks that ambiguity (see below), so this term must.
        loss_res:
            Mean squared Schrodinger residual,
            ``H(mu) psi_pred - E_pred * psi_pred``, over all rows.
        loss_ph:
            Mean squared particle-hole symmetry residual, the same
            residual re-evaluated on ``Xi @ psi_pred`` at energy
            ``-E_pred``, over all rows.

    Early in training, the dual-head model's own ``E_pred``/``psi_pred``
    are close to random, so the physics residual terms (which are
    computed *from those predictions*, not from ground truth) are an
    unreliable, potentially actively misleading signal. ``physics_weight``
    therefore anneals from ``0.01`` up to ``1.0`` over ``anneal_duration``
    epochs — the mirror image of :class:`PinnedFSMLoss`'s ``pin_weight``
    schedule — so the model first locks onto the labelled data before the
    physics constraints are weighted at full strength. ``loss_e``/
    ``loss_psi`` are left unweighted throughout, since they are exact
    ground truth wherever they are available at all.

    Passing ``energy_batch=None``/``psi_batch=None`` drops the two data
    terms, leaving a pure label-free physics loss usable on an
    unsupervised-only batch.

    Attributes:
        total_epochs: Total number of epochs the associated training
            run is expected to last.
        anneal_duration: Number of epochs over which ``physics_weight``
            rises from ``0.01`` to ``1.0``.
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
            anneal_duration: Number of epochs over which
                ``physics_weight`` rises from ``0.01`` to ``1.0``.
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
        energy_batch: torch.Tensor | None = None,
        psi_batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the semi-supervised loss and its component metrics for a batch.

        Args:
            model: The dual-head model being trained. Must return a
                ``(E_pred, psi_pred)`` tuple when called on ``mu_batch``.
            mu_batch: Batch of mu values. Rows ``[:n_labeled]`` must
                correspond, in order, to ``energy_batch``/``psi_batch``
                when those are given.
            H_base: Mu-independent part of the batched Hamiltonian.
            H_mu_diag: Diagonal matrix such that
                ``H_base + mu * H_mu_diag`` gives the full Hamiltonian
                at a given mu.
            Xi: Particle-hole symmetry operator.
            epoch: Current epoch number, used to anneal
                ``physics_weight``.
            energy_batch: Exact energies for the labelled rows of
                ``mu_batch``, shape ``(n_labeled, 1)``. ``None`` disables
                ``loss_e``.
            psi_batch: Exact eigenvectors for the labelled rows of
                ``mu_batch``, shape ``(n_labeled, 2*n_sites)``. ``None``
                disables ``loss_psi``.

        Returns:
            Tuple of ``(total_loss, metrics)``, where ``metrics``
            contains the four individual (unweighted) loss components
            plus the current ``physics_wt``.
        """
        if epoch < self.anneal_duration:
            physics_weight = 0.01 + 0.99 * (epoch / self.anneal_duration)
        else:
            physics_weight = 1.0

        E_pred, psi_pred = model(mu_batch)
        H_batch = H_base.unsqueeze(0) + mu_batch.unsqueeze(-1) * H_mu_diag.unsqueeze(0)

        H_psi = torch.bmm(H_batch, psi_pred.unsqueeze(-1)).squeeze(-1)
        residual = H_psi - E_pred * psi_pred
        loss_res = torch.mean(residual**2)

        Xi_psi = torch.matmul(psi_pred, Xi.T)
        H_Xi_psi = torch.bmm(H_batch, Xi_psi.unsqueeze(-1)).squeeze(-1)
        ph_residual = H_Xi_psi + E_pred * Xi_psi
        loss_ph = torch.mean(ph_residual**2)

        if energy_batch is not None and psi_batch is not None:
            n_labeled = energy_batch.shape[0]
            loss_e = torch.nn.functional.mse_loss(E_pred[:n_labeled], energy_batch)

            # An eigenvector is only defined up to an overall sign, and
            # nothing else in this loss breaks that ambiguity: loss_res and
            # loss_ph are both invariant under psi_pred -> -psi_pred (the
            # residual itself flips sign, but its square does not), so
            # enforcing a consistent sign convention is left entirely to
            # loss_psi. Without this alignment, an MSE against the "wrong"
            # sign of psi_batch would report a large, misleading error
            # (exactly 4x the mean squared amplitude of a unit-norm vector)
            # for a psi_pred that is in fact the correct eigenstate.
            with torch.no_grad():
                sign = torch.sign(
                    torch.sum(psi_pred[:n_labeled] * psi_batch, dim=1, keepdim=True)
                )
                sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            loss_psi = torch.nn.functional.mse_loss(
                psi_pred[:n_labeled], sign * psi_batch
            )
        else:
            loss_e = torch.zeros((), device=mu_batch.device)
            loss_psi = torch.zeros((), device=mu_batch.device)

        total_loss = loss_e + loss_psi + physics_weight * (loss_res + loss_ph)

        return total_loss, {
            "e": loss_e.item(),
            "psi": loss_psi.item(),
            "res": loss_res.item(),
            "ph": loss_ph.item(),
            "physics_wt": physics_weight,
        }


class ChiralFSMLoss(BaseLoss):
    """Folded-spectrum loss on the Majorana-basis chiral block ``h(mu)``.

    Intended for :class:`kitaev.models.SirenPINNChiral`, which returns the
    left/right singular vectors ``(u, v)`` of the smallest singular value of
    the real ``N x N`` bidiagonal operator ``h(mu)`` (see
    :func:`kitaev.analytical.chiral_block`). Because the singular values of
    ``h(mu)`` are exactly the non-negative BdG eigenvalues, and the model
    normalises ``u`` and ``v`` by construction, this loss needs neither a
    normalisation term, an energy-non-negativity term, nor a particle-hole
    residual: those are all structural. What remains is the pair

        loss = loss_fsm + loss_var

    with no relative weight and no annealing schedule:

        loss_fsm:
            ``mean(||h v||^2) + mean(||h^T u||^2)``. The folded-spectrum
            residual: driving both to zero pushes ``(u, v)`` onto the
            directions of the *smallest* singular value of ``h(mu)`` -- the
            near-null space in the topological phase, the smallest bulk gap
            otherwise. Directly analogous to
            :class:`PinnedFSMLoss`'s ``mean(||H psi||^2)``.
        loss_var:
            ``mean(||h v - lambda_R u||^2) + mean(||h^T u - lambda_R v||^2)``
            with ``lambda_R = u^T h v`` the Rayleigh-quotient singular value.
            Forces ``(u, v)`` to be a genuine matched singular pair, so that
            ``lambda_R`` is a meaningful energy. Analogous to
            :class:`PinnedFSMLoss`'s ``loss_var``.

    The reported ``lam_mean`` metric is the batch mean of ``|lambda_R|`` --
    the model's energy estimate. The absolute value is taken because the
    loss is invariant under ``(u, v) -> (u, -v)``, which negates
    ``lambda_R``; the sign carries no information and
    :func:`kitaev.analytical.resolve_singular_branch` fixes the branch at
    reconstruction time.

    ``H_base``, ``H_mu_diag``, ``Xi`` and ``epoch`` are accepted to match the
    :class:`BaseLoss` call contract but are unused: this loss builds its own
    ``h(mu)`` from ``mu_batch``.

    Attributes:
        n_sites: Number of physical lattice sites, ``N``.
        hopping: Nearest-neighbour hopping amplitude, ``t``.
        pairing: P-wave pairing amplitude, ``delta``.
    """

    def __init__(
        self,
        n_sites: int,
        *,
        hopping: float = 1.0,
        pairing: float = 0.5,
    ) -> None:
        """Initialise the loss with the chain's physical parameters.

        Args:
            n_sites: Number of physical lattice sites, ``N``.
            hopping: Nearest-neighbour hopping amplitude, ``t``.
            pairing: P-wave pairing amplitude, ``delta``. Both ``hopping``
                and ``pairing`` are in the same energy units as ``mu``.
        """
        self.n_sites = n_sites
        self.hopping = hopping
        self.pairing = pairing

    def __call__(
        self,
        model: torch.nn.Module,
        mu_batch: torch.Tensor,
        H_base: torch.Tensor,
        H_mu_diag: torch.Tensor,
        Xi: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the chiral folded-spectrum loss for a batch.

        Args:
            model: The model being trained. Must return a ``(u, v)`` tuple
                of unit-norm ``(batch_size, n_sites)`` tensors when called
                on ``mu_batch``.
            mu_batch: Batch of mu values, shape ``(batch_size, 1)``.
            H_base: Unused (kept for the :class:`BaseLoss` contract).
            H_mu_diag: Unused.
            Xi: Unused.
            epoch: Unused (this loss has no annealing schedule).

        Returns:
            Tuple of ``(total_loss, metrics)``, where ``metrics`` holds the
            two loss components and the mean Rayleigh-quotient energy.
        """
        del H_base, H_mu_diag, Xi, epoch

        u, v = model(mu_batch)
        h_batch = chiral_block_batched(
            mu_batch, self.n_sites, self.hopping, self.pairing
        )

        h_v = torch.bmm(h_batch, v.unsqueeze(-1)).squeeze(-1)
        ht_u = torch.bmm(h_batch.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)

        loss_fsm = torch.mean(h_v**2) + torch.mean(ht_u**2)

        lam_rayleigh = torch.sum(u * h_v, dim=1, keepdim=True)

        loss_var = torch.mean((h_v - lam_rayleigh * u) ** 2) + torch.mean(
            (ht_u - lam_rayleigh * v) ** 2
        )

        total_loss = loss_fsm + loss_var

        return total_loss, {
            "fsm": loss_fsm.item(),
            "var": loss_var.item(),
            "lam_mean": lam_rayleigh.abs().mean().item(),
        }


def chiral_pointwise_residual(
    model: torch.nn.Module,
    mu_batch: torch.Tensor,
    n_sites: int,
    *,
    hopping: float = 1.0,
    pairing: float = 0.5,
) -> torch.Tensor:
    """Per-sample chiral physics residual for residual-adaptive sampling.

    This is the same folded-spectrum + eigenvector-consistency quantity
    :class:`ChiralFSMLoss` sums into a scalar, but kept **per mu**:

        residual(mu) = ||h(mu) v||^2 + ||h(mu)^T u||^2
                     + ||h(mu) v - lambda_R u||^2
                     + ||h(mu)^T u - lambda_R v||^2

    with ``(u, v)`` the model output at ``mu`` and
    ``lambda_R = u^T h(mu) v``. Used by
    :class:`kitaev.data.streaming.AdaptiveSampling` to find the mu values
    where the model is currently doing worst.

    Args:
        model: A chiral model returning ``(u, v)`` for a mu batch.
        mu_batch: Chemical-potential values, shape ``(batch_size, 1)``.
        n_sites: Number of physical lattice sites, ``N``.
        hopping: Nearest-neighbour hopping amplitude, ``t``.
        pairing: P-wave pairing amplitude, ``delta``.

    Returns:
        A tensor of shape ``(batch_size,)`` of non-negative residuals.
    """
    u, v = model(mu_batch)
    h_batch = chiral_block_batched(mu_batch, n_sites, hopping, pairing)

    h_v = torch.bmm(h_batch, v.unsqueeze(-1)).squeeze(-1)
    ht_u = torch.bmm(h_batch.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
    lam = torch.sum(u * h_v, dim=1, keepdim=True)

    fsm = (h_v**2).sum(dim=1) + (ht_u**2).sum(dim=1)
    var = ((h_v - lam * u) ** 2).sum(dim=1) + ((ht_u - lam * v) ** 2).sum(dim=1)
    residual: torch.Tensor = fsm + var
    return residual
