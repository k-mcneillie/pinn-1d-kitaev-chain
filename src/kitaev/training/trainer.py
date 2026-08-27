from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from sesh import Session
from torch.utils.data import DataLoader

from kitaev.data.generators.unsupervised import UnsupervisedMuGenerator
from kitaev.data.mu_sampler import MuSampler
from kitaev.data.sampling_region import TRANSITION_FOCUSED_REGIONS

from .callbacks import TrainingCallback
from .config import TrainerConfig, TwoPhaseConfig
from .loss import BaseLoss, PinnedFSMLoss
from .utils import EarlyStopping, EpochAccumulator, TrainingHistory


class UnifiedTrainer:
    """Trains a PINN model against a physics loss, with device handling
    delegated to accelerate.

    This class never constructs, queries, or stores a ``torch.device``.
    All placement (CPU/CUDA/MPS/multi-GPU/TPU, and optionally mixed
    precision) is the responsibility of the ``Accelerator`` instance
    passed in, which is created once by the caller — see the
    ``if __name__ == "__main__"`` block at the bottom of this module for
    where that instantiation belongs in a full pipeline. Model,
    optimiser, and (if given) scheduler are handed to
    ``accelerator.prepare`` in the constructor; dataloaders are prepared
    lazily inside :meth:`fit`, since they are not always known at
    construction time.

    This is the single training entrypoint for the project regardless of
    whether ``loss_fn`` is purely label-free (``PinnedFSMLoss``, trained
    from ``train_loader`` alone) or semi-supervised (``SemiSupervisedLoss``,
    trained from a small labelled ``train_loader`` plus a much larger
    ``unlabeled_loader`` passed to :meth:`fit`) — swapping between the two
    is a choice of ``loss_fn`` and dataloaders, not a different training
    loop, which is what keeps ablations across losses comparable.

    Attributes:
        accelerator: The shared ``Accelerator`` instance handling device
            placement and (optionally) distributed training.
        loss_fn: The physics loss to optimise against.
        config: Training hyperparameters, see :class:`TrainerConfig`.
        history: Full per-epoch metric history for the run.
        model: The (accelerate-wrapped) model being trained.
        optimiser: The (accelerate-wrapped) optimiser.
        scheduler: The (accelerate-wrapped) learning-rate scheduler, or
            ``None`` if none was supplied.
    """

    def __init__(
        self,
        session: Session,
        accelerator: Accelerator,
        model: nn.Module,
        loss_fn: BaseLoss,
        optimiser: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler | None = None,
        config: TrainerConfig | None = None,
        callbacks: Sequence[TrainingCallback] = (),
    ) -> None:
        """Wires together the trainer's collaborators and prepares them via accelerate.

        Args:
            accelerator: A single, shared ``Accelerator`` instance. Must
                be created exactly once per process, upstream of this
                trainer.
            model: The model to train. Its parameters must already have
                been used to construct ``optimiser`` before this call.
            loss_fn: The physics loss to optimise.
            optimiser: A fully constructed optimiser over
                ``model.parameters()``. Constructing it is left to the
                caller so any optimiser/hyperparameter choice is
                possible without extending this class.
            scheduler: An optional, fully constructed learning-rate
                scheduler wrapping ``optimiser``.
            config: Training hyperparameters. Defaults to
                ``TrainerConfig()`` if omitted.
            callbacks: Optional :class:`~kitaev.training.callbacks.TrainingCallback`
                objects, invoked in order at the start and end of every
                epoch. Used by the streaming sampling strategies in
                :mod:`kitaev.training.sampling` to update what the next
                epoch samples; empty (no hooks) by default.
        """
        self.session = session
        self.accelerator = accelerator
        self.loss_fn = loss_fn
        self.config = config or TrainerConfig()
        self.callbacks = tuple(callbacks)
        self.history = TrainingHistory()
        self._early_stopping = EarlyStopping(self.config.patience)

        # Recorded before accelerator.prepare() wraps ``optimiser`` in an
        # AcceleratedOptimizer, since isinstance checks against the wrapped
        # object would no longer see the underlying LBFGS type.
        self._is_lbfgs = isinstance(optimiser, optim.LBFGS)

        if scheduler is not None:
            self.model, self.optimiser, self.scheduler = self.accelerator.prepare(
                model, optimiser, scheduler
            )
        else:
            self.model, self.optimiser = self.accelerator.prepare(model, optimiser)
            self.scheduler = None

    def fit(
        self,
        train_loader: DataLoader,
        H_base: torch.Tensor,
        H_mu_diag: torch.Tensor,
        Xi: torch.Tensor,
        val_loader: DataLoader | None = None,
        unlabeled_loader: DataLoader | None = None,
    ) -> nn.Module:
        """Runs the full training loop and returns the best (or final) model.

        Args:
            train_loader: Yields batches whose first element is a
                ``mu`` tensor. A batch may optionally carry exact
                labels as its second and third elements, ``energy``
                and ``psi`` (as produced by, e.g.,
                ``SupervisedKitaevDataset.dataloader``) — when present,
                these are forwarded to ``loss_fn`` as the
                ``energy_batch``/``psi_batch`` keyword arguments, for
                losses that declare them (e.g. ``SemiSupervisedLoss``).
                Prepared internally via ``accelerator.prepare``, so it
                should not be pre-prepared by the caller.
            H_base: Mu-independent part of the batched Hamiltonian.
                Should already be on ``self.accelerator.device``.
            H_mu_diag: Diagonal matrix such that
                ``H_base + mu * H_mu_diag`` gives the full Hamiltonian
                at a given mu. Should already be on
                ``self.accelerator.device``.
            Xi: Particle-hole symmetry operator. Should already be on
                ``self.accelerator.device``.
            val_loader: Optional validation dataloader, same shape
                convention as ``train_loader``. When given, enables
                checkpointing of the best model and (if configured)
                early stopping.
            unlabeled_loader: Optional dataloader of label-free
                ``mu`` batches (e.g. from
                ``UnsupervisedMuGenerator.dataloader``), for
                semi-supervised training. When given, each step draws
                one batch from it, cycling once exhausted, and
                concatenates its ``mu`` values onto ``train_loader``'s
                (and, separately, ``val_loader``'s) ``mu`` batch before
                calling ``loss_fn`` — letting a single loss combine a
                small labelled batch with a much larger label-free one,
                as ``SemiSupervisedLoss`` does. Has no effect on a
                ``train_loader``/``val_loader`` batch that carries no
                labels of its own beyond widening the physics-only
                batch.

        Returns:
            The trained model, unwrapped from any accelerate wrapping.
            If validation was used, this is the best checkpoint seen;
            otherwise it is the model's final-epoch state.
        """

        self.session.info(
            f"--- Starting training ({self.loss_fn.__class__.__name__}) ---"
        )
        start_time = time.time()

        train_loader = self.accelerator.prepare(train_loader)
        if val_loader is not None:
            val_loader = self.accelerator.prepare(val_loader)
        if unlabeled_loader is not None:
            unlabeled_loader = self.accelerator.prepare(unlabeled_loader)

        # Independent cycles so draining one during training doesn't
        # advance the other's position through val_loader's epochs.
        train_unlabeled_cycle = (
            self._cycle(unlabeled_loader) if unlabeled_loader is not None else None
        )
        val_unlabeled_cycle = (
            self._cycle(unlabeled_loader) if unlabeled_loader is not None else None
        )

        for epoch in range(1, self.config.epochs + 1):
            self._run_callbacks("on_epoch_start", epoch)

            train_metrics = self._run_epoch(
                train_loader,
                H_base,
                H_mu_diag,
                Xi,
                train=True,
                epoch=epoch,
                unlabeled_cycle=train_unlabeled_cycle,
            )
            self._record("train", train_metrics)

            if self.scheduler is not None:
                self.scheduler.step()

            val_metrics = None
            should_stop = False
            if val_loader is not None:
                val_metrics = self._run_epoch(
                    val_loader,
                    H_base,
                    H_mu_diag,
                    Xi,
                    train=False,
                    epoch=epoch,
                    unlabeled_cycle=val_unlabeled_cycle,
                )
                self._record("val", val_metrics)
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                should_stop = self._early_stopping.step(
                    val_metrics["loss"], epoch, unwrapped_model
                )

            self._run_callbacks("on_epoch_end", epoch)

            if epoch % self.config.print_freq == 0 or epoch == self.config.epochs:
                self._log_epoch(epoch, train_metrics, val_metrics)

            if should_stop:
                self.session.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no improvement for {self.config.patience} epochs)."
                )
                break

        self.session.info(f"Training complete in {time.time() - start_time:.2f}s.")
        return self._finalise_model()

    def _run_epoch(
        self,
        data_loader: DataLoader,
        H_base: torch.Tensor,
        H_mu_diag: torch.Tensor,
        Xi: torch.Tensor,
        *,
        train: bool,
        epoch: int,
        unlabeled_cycle: Iterator[tuple[torch.Tensor, ...]] | None = None,
    ) -> dict[str, float]:
        """Runs one full pass over ``data_loader``, either training or evaluating.

        Args:
            data_loader: Prepared dataloader to iterate over. Each
                batch's first element is treated as ``mu``; a second
                and third element, if present, are treated as exact
                ``(energy, psi)`` labels for that batch and forwarded
                to ``loss_fn``.
            H_base: See :meth:`fit`.
            H_mu_diag: See :meth:`fit`.
            Xi: See :meth:`fit`.
            train: If ``True``, runs optimiser steps and puts the model
                in train mode; if ``False``, runs under
                ``torch.no_grad()`` in eval mode.
            epoch: Current epoch number.
            unlabeled_cycle: Optional never-ending iterator over a
                label-free dataloader (see :meth:`fit`). One batch is
                drawn per step and its ``mu`` values are concatenated
                onto ``data_loader``'s own ``mu`` batch.

        Returns:
            Averaged metrics for the epoch, including a ``"loss"`` key.
        """
        self.model.train(train)
        accumulator = EpochAccumulator()

        with torch.enable_grad() if train else torch.no_grad():
            for batch in data_loader:
                # Already on the correct device via accelerate.
                mu_batch = batch[0]
                energy_batch = batch[1] if len(batch) > 1 else None
                psi_batch = batch[2] if len(batch) > 2 else None

                if unlabeled_cycle is not None:
                    unlabeled_mu = next(unlabeled_cycle)[0]
                    mu_batch = torch.cat([mu_batch, unlabeled_mu], dim=0)

                if train:
                    loss_value, metrics = self._train_step(
                        mu_batch,
                        H_base,
                        H_mu_diag,
                        Xi,
                        epoch=epoch,
                        energy_batch=energy_batch,
                        psi_batch=psi_batch,
                    )
                else:
                    loss, metrics = self._call_loss(
                        mu_batch,
                        H_base,
                        H_mu_diag,
                        Xi,
                        epoch,
                        energy_batch=energy_batch,
                        psi_batch=psi_batch,
                    )
                    loss_value = loss.item()
                accumulator.update({**metrics, "loss": loss_value})

        return accumulator.averages()

    def _call_loss(
        self,
        mu_batch: torch.Tensor,
        H_base: torch.Tensor,
        H_mu_diag: torch.Tensor,
        Xi: torch.Tensor,
        epoch: int,
        energy_batch: torch.Tensor | None = None,
        psi_batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Invokes ``loss_fn``, forwarding labels only when they are present.

        ``energy_batch``/``psi_batch`` are omitted from the call
        entirely when ``None``, rather than passed as ``None``, so that
        losses which (like ``PinnedFSMLoss``) do not declare these
        keyword arguments at all keep working unmodified.
        """
        extra_kwargs: dict[str, torch.Tensor] = {}
        if energy_batch is not None:
            extra_kwargs["energy_batch"] = energy_batch
        if psi_batch is not None:
            extra_kwargs["psi_batch"] = psi_batch
        return self.loss_fn(
            self.model, mu_batch, H_base, H_mu_diag, Xi, epoch=epoch, **extra_kwargs
        )

    def _train_step(
        self,
        mu_batch: torch.Tensor,
        H_base: torch.Tensor,
        H_mu_diag: torch.Tensor,
        Xi: torch.Tensor,
        epoch: int,
        energy_batch: torch.Tensor | None = None,
        psi_batch: torch.Tensor | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Dispatches a single optimisation step to the correct routine.

        L-BFGS requires a closure-based step; every other optimiser uses
        the standard zero-grad/backward/step sequence. Both routines use
        ``accelerator.backward`` rather than ``loss.backward()``
        directly, since accelerate must intercept the backward pass to
        correctly handle mixed precision or distributed gradient
        synchronisation if either is enabled.

        Args:
            mu_batch: Batch of mu values.
            H_base: See :meth:`fit`.
            H_mu_diag: See :meth:`fit`.
            Xi: See :meth:`fit`.
            epoch: Current epoch number.
            energy_batch: Optional exact energies for (a leading subset
                of) ``mu_batch``, forwarded to ``loss_fn`` when given.
            psi_batch: Optional exact eigenvectors for (a leading
                subset of) ``mu_batch``, forwarded to ``loss_fn`` when
                given.

        Returns:
            Tuple of ``(loss_value, metrics)`` for this batch.
        """
        if self._is_lbfgs:
            return self._lbfgs_step(
                mu_batch,
                H_base,
                H_mu_diag,
                Xi,
                epoch=epoch,
                energy_batch=energy_batch,
                psi_batch=psi_batch,
            )
        return self._standard_step(
            mu_batch,
            H_base,
            H_mu_diag,
            Xi,
            epoch=epoch,
            energy_batch=energy_batch,
            psi_batch=psi_batch,
        )

    def _standard_step(
        self,
        mu_batch: torch.Tensor,
        H_base: torch.Tensor,
        H_mu_diag: torch.Tensor,
        Xi: torch.Tensor,
        epoch: int,
        energy_batch: torch.Tensor | None = None,
        psi_batch: torch.Tensor | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Runs a single zero-grad/backward/step optimisation step."""
        self.optimiser.zero_grad()
        loss, metrics = self._call_loss(
            mu_batch,
            H_base,
            H_mu_diag,
            Xi,
            epoch,
            energy_batch=energy_batch,
            psi_batch=psi_batch,
        )
        self.accelerator.backward(loss)
        if self.config.grad_clip_norm is not None:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )
        self.optimiser.step()
        return loss.item(), metrics

    def _lbfgs_step(
        self,
        mu_batch: torch.Tensor,
        H_base: torch.Tensor,
        H_mu_diag: torch.Tensor,
        Xi: torch.Tensor,
        epoch: int,
        energy_batch: torch.Tensor | None = None,
        psi_batch: torch.Tensor | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Runs a single L-BFGS step via its closure-based interface.

        Both the loss and the metrics dict are captured from inside the
        closure via mutable containers, since ``optimiser.step(closure)``
        may invoke the closure multiple times internally, and
        accelerate's ``AcceleratedOptimizer.step`` discards the
        closure's return value rather than passing it back.
        ``energy_batch``/``psi_batch`` are drawn once, outside the
        closure, so every internal re-evaluation of the closure sees
        the same labels rather than a freshly cycled batch.
        """
        captured_metrics: dict[str, float] = {}
        captured_loss = float("nan")

        def closure() -> torch.Tensor:
            nonlocal captured_loss
            self.optimiser.zero_grad()
            closure_loss, closure_metrics = self._call_loss(
                mu_batch,
                H_base,
                H_mu_diag,
                Xi,
                epoch,
                energy_batch=energy_batch,
                psi_batch=psi_batch,
            )
            self.accelerator.backward(closure_loss)
            captured_loss = closure_loss.item()
            captured_metrics.clear()
            captured_metrics.update(closure_metrics)
            return closure_loss

        self.optimiser.step(closure)
        return captured_loss, captured_metrics

    @staticmethod
    def _cycle(
        loader: DataLoader,
    ) -> Iterator[tuple[torch.Tensor, ...]]:
        """Yields batches from ``loader`` forever, re-iterating (and thus
        re-shuffling, if configured) once each pass is exhausted.

        Used instead of ``itertools.cycle`` so batches are not cached in
        memory to be replayed; each pass is a fresh call to
        ``iter(loader)``.
        """
        while True:
            yield from loader

    def _run_callbacks(self, hook: str, epoch: int) -> None:
        """Invokes ``hook`` on every registered callback, in order.

        Args:
            hook: Either ``"on_epoch_start"`` or ``"on_epoch_end"``.
            epoch: The 1-based epoch number to pass through.
        """
        if not self.callbacks:
            return
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        for callback in self.callbacks:
            getattr(callback, hook)(epoch, unwrapped_model, self.history)

    def _record(self, split: str, metrics: dict[str, float]) -> None:
        """Appends a split's averaged metrics into the training history.

        Args:
            split: Either ``"train"`` or ``"val"``, used as the metric
                key prefix.
            metrics: Averaged metrics for the epoch.
        """
        for key, value in metrics.items():
            self.history.record(f"{split}_{key}", value)

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        """Logs a single formatted progress line for the given epoch.

        Metrics whose key ends in ``_wt`` (e.g. ``pin_wt``, ``physics_wt``)
        are annealing-schedule weights rather than loss components: they
        are pulled out of the main ``.6f``-formatted list and appended at
        the end at 3 decimal places instead, so any :class:`BaseLoss`
        subclass that reports one gets consistent log formatting without
        this method needing to know its specific name.
        """

        def is_weight_key(key: str) -> bool:
            return key.endswith("_wt")

        train_parts = [
            f"{k}: {v:.6f}" for k, v in train_metrics.items() if not is_weight_key(k)
        ]
        message = f"Epoch {epoch:04d} | " + " | ".join(train_parts)

        if val_metrics is not None:
            val_parts = [
                f"val_{k}: {v:.6f}"
                for k, v in val_metrics.items()
                if not is_weight_key(k)
            ]
            message += " || " + " | ".join(val_parts)

        weight_parts = [
            f"{k}: {v:.3f}" for k, v in train_metrics.items() if is_weight_key(k)
        ]
        if weight_parts:
            message += " | " + " | ".join(weight_parts)

        self.session.info(message)

    def _finalise_model(self) -> nn.Module:
        """Loads the best checkpoint (if any) and returns the unwrapped model.

        Returns:
            The unwrapped model, restored to its best validation-loss
            state if validation was used, otherwise left at its final
            training state.
        """
        unwrapped_model: nn.Module = self.accelerator.unwrap_model(self.model)
        if self._early_stopping.best_state is not None:
            unwrapped_model.load_state_dict(self._early_stopping.best_state)
            self.session.info(
                f"Loaded best model state from validation "
                f"(epoch {self._early_stopping.best_epoch}, "
                f"val loss: {self._early_stopping.best_loss:.6f})."
            )
        return unwrapped_model


def _concat_histories(*histories: TrainingHistory) -> TrainingHistory:
    """Concatenates several run histories end to end into one.

    Args:
        *histories: Populated :class:`TrainingHistory` objects, in the
            order they ran.

    Returns:
        A new :class:`TrainingHistory` whose every series is the
        corresponding series from each input appended in turn.
    """
    merged = TrainingHistory()
    for history in histories:
        for key, series in history.as_dict().items():
            for value in series:
                merged.record(key, value)
    return merged


def run_two_phase(
    session: Session,
    accelerator: Accelerator,
    model: nn.Module,
    loss_fn: BaseLoss,
    train_loader: DataLoader,
    H_base: torch.Tensor,
    H_mu_diag: torch.Tensor,
    Xi: torch.Tensor,
    *,
    two_phase: TwoPhaseConfig | None = None,
    base_config: TrainerConfig | None = None,
    callbacks: Sequence[TrainingCallback] = (),
    val_loader: DataLoader | None = None,
    unlabeled_loader: DataLoader | None = None,
    lbfgs_train_loader: DataLoader | None = None,
    lbfgs_callbacks: Sequence[TrainingCallback] | None = None,
) -> tuple[nn.Module, TrainingHistory]:
    """Runs an AdamW warm-up then an L-BFGS fine-tuning phase on one model.

    Builds and runs two :class:`UnifiedTrainer` instances back to back over
    the *same* model: first AdamW (with a cosine schedule) for
    ``two_phase.adam_epochs``, then L-BFGS for ``two_phase.lbfgs_epochs``
    starting from the AdamW result. This is the standard PINN optimiser
    recipe — see :class:`~kitaev.training.config.TwoPhaseConfig`.

    Args:
        session: The run's :class:`sesh.Session`.
        accelerator: The shared :class:`accelerate.Accelerator`.
        model: The model to train. Must not have been passed to
            ``accelerator.prepare`` yet.
        loss_fn: The physics loss.
        train_loader: The label-free training loader (frozen or streaming).
        H_base: Mu-independent part of the batched Hamiltonian, on device.
        H_mu_diag: Diagonal mu operator, on device.
        Xi: Particle-hole operator, on device.
        two_phase: The two-phase schedule. Defaults to
            :class:`TwoPhaseConfig`.
        base_config: Template :class:`TrainerConfig` for both phases; each
            phase overrides ``epochs`` (and the L-BFGS phase disables
            gradient clipping). Defaults to :class:`TrainerConfig`.
        callbacks: Trainer callbacks forwarded to the AdamW phase (e.g. a
            sampling strategy). They are forwarded to the L-BFGS phase too
            *unless* ``lbfgs_train_loader`` is given (a fixed objective is the
            whole point of passing a dedicated loader) or ``lbfgs_callbacks``
            is set, either of which replaces them for that phase.
        val_loader: Optional validation loader, used in both phases.
        unlabeled_loader: Optional label-free loader concatenated onto the
            train batch, for semi-supervised losses.
        lbfgs_train_loader: Optional loader for the L-BFGS phase only.
            L-BFGS builds its inverse-Hessian estimate from gradient
            differences across steps and is destabilised when each step sees
            a freshly drawn batch (as ``mode="infinite"`` and the other
            streaming samplers produce). Pass a fixed, large-batch
            ``mode="frozen"`` loader here so the second phase optimises a
            stationary objective. Defaults to ``train_loader``.
        lbfgs_callbacks: Explicit callback list for the L-BFGS phase. Use it
            to keep a diagnostic callback (e.g.
            :class:`~kitaev.training.probes.ChiralEvaluationProbe`) running in
            the second phase while dropping the streaming sampler that came
            with ``callbacks``. When ``None`` (the default) the L-BFGS phase
            uses ``callbacks``, or ``()`` if ``lbfgs_train_loader`` is set.

    Returns:
        ``(model, history)`` — the fine-tuned model and the two phases'
        histories concatenated end to end.
    """
    two_phase = two_phase or TwoPhaseConfig()
    base_config = base_config or TrainerConfig()

    adam = optim.AdamW(
        model.parameters(),
        lr=two_phase.adam_lr,
        weight_decay=two_phase.adam_weight_decay,
    )
    adam_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        adam, T_max=max(1, two_phase.adam_epochs)
    )
    adam_config = replace(base_config, epochs=two_phase.adam_epochs)
    phase_one = UnifiedTrainer(
        session=session,
        accelerator=accelerator,
        model=model,
        loss_fn=loss_fn,
        optimiser=adam,
        scheduler=adam_scheduler,
        config=adam_config,
        callbacks=callbacks,
    )
    session.info("--- Two-phase: AdamW warm-up ---")
    model = phase_one.fit(
        train_loader,
        H_base,
        H_mu_diag,
        Xi,
        val_loader=val_loader,
        unlabeled_loader=unlabeled_loader,
    )

    if two_phase.lbfgs_epochs <= 0:
        return model, phase_one.history

    lbfgs = optim.LBFGS(
        model.parameters(),
        lr=two_phase.lbfgs_lr,
        max_iter=two_phase.lbfgs_max_iter,
        history_size=two_phase.lbfgs_history_size,
        line_search_fn=two_phase.lbfgs_line_search_fn,
    )
    lbfgs_config = replace(
        base_config, epochs=two_phase.lbfgs_epochs, grad_clip_norm=None
    )
    lbfgs_loader = (
        lbfgs_train_loader if lbfgs_train_loader is not None else train_loader
    )
    if lbfgs_callbacks is not None:
        phase_two_callbacks: Sequence[TrainingCallback] = lbfgs_callbacks
    elif lbfgs_train_loader is not None:
        phase_two_callbacks = ()
    else:
        phase_two_callbacks = callbacks
    phase_two = UnifiedTrainer(
        session=session,
        accelerator=accelerator,
        model=model,
        loss_fn=loss_fn,
        optimiser=lbfgs,
        scheduler=None,
        config=lbfgs_config,
        callbacks=phase_two_callbacks,
    )
    session.info("--- Two-phase: L-BFGS fine-tuning ---")
    model = phase_two.fit(
        lbfgs_loader,
        H_base,
        H_mu_diag,
        Xi,
        val_loader=val_loader,
        unlabeled_loader=unlabeled_loader,
    )

    return model, _concat_histories(phase_one.history, phase_two.history)


# ======================================================================
# 4. Example usage
# ======================================================================


def _build_kitaev_operators(
    n_sites: int, hopping: float, pairing: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds the batched-Hamiltonian pieces (H_base, H_mu_diag, Xi) on the CPU.

    These are moved onto ``accelerator.device`` once, by the caller, in
    the example below — building them here on the CPU keeps this helper
    itself free of any device decision, consistent with the rest of the
    module.

    Args:
        n_sites: Number of physical lattice sites, N.
        hopping: Hopping amplitude, t.
        pairing: Pairing amplitude, delta.

    Returns:
        Tuple of ``(H_base, H_mu_diag, Xi)``, each of shape
        ``(2*n_sites, 2*n_sites)``.
    """
    dim = 2 * n_sites
    H_base = torch.zeros((dim, dim), dtype=torch.float32)
    for i in range(n_sites - 1):
        H_base[i, i + 1] = H_base[i + 1, i] = -hopping
        H_base[n_sites + i, n_sites + i + 1] = H_base[n_sites + i + 1, n_sites + i] = (
            hopping
        )
        H_base[i, n_sites + i + 1] = pairing
        H_base[n_sites + i + 1, i] = pairing
        H_base[i + 1, n_sites + i] = -pairing
        H_base[n_sites + i, i + 1] = -pairing

    mu_diag = torch.zeros(dim, dtype=torch.float32)
    mu_diag[:n_sites] = -1.0
    mu_diag[n_sites:] = 1.0
    H_mu_diag = torch.diag(mu_diag)

    Xi = torch.zeros((dim, dim), dtype=torch.float32)
    Xi[:n_sites, n_sites:] = torch.eye(n_sites)
    Xi[n_sites:, :n_sites] = torch.eye(n_sites)

    return H_base, H_mu_diag, Xi


class _ExampleSirenStandIn(nn.Module):
    """Minimal placeholder model, standing in for the project's real SirenPINN.

    Replace with the actual SIREN-based model (built from ``SineLayer``)
    when wiring this trainer into the real pipeline — this stand-in
    exists purely so the example below is runnable end to end without
    depending on that module.
    """

    def __init__(self, n_sites: int, hidden_features: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, 2 * n_sites),
        )

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        psi = self.net(mu)
        normalised: torch.Tensor = psi / torch.norm(psi, dim=-1, keepdim=True)
        return normalised


def main() -> None:  # pragma: no cover
    """End-to-end example: build operators, model, and trainer, then fit.

    This is the "step 3, right before training" orchestration point
    referred to earlier: the single place a ``torch.device`` decision
    (via ``Accelerator``) is made, with every downstream object either
    constructed already on that device or handed to
    ``accelerator.prepare`` so it becomes so.
    """

    repo_root = Path(__file__).resolve().parents[3]
    session = Session(
        name="Test",
        output_root=repo_root / "results" / "logs",
    )

    accelerator = Accelerator()

    n_sites = 20
    H_base, H_mu_diag, Xi = _build_kitaev_operators(n_sites, hopping=1.0, pairing=0.5)
    H_base = H_base.to(accelerator.device)
    H_mu_diag = H_mu_diag.to(accelerator.device)
    Xi = Xi.to(accelerator.device)

    sampler = MuSampler(TRANSITION_FOCUSED_REGIONS)
    generator = UnsupervisedMuGenerator(sampler=sampler)
    train_loader = generator.dataloader(total_samples=4096, batch_size=1024)
    val_loader = generator.dataloader(total_samples=512, batch_size=1024)

    model = _ExampleSirenStandIn(n_sites=n_sites)
    loss_fn = PinnedFSMLoss(total_epochs=3000, anneal_duration=2000)
    optimiser = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=3000)
    config = TrainerConfig(
        epochs=3000, print_freq=500, patience=300, grad_clip_norm=1.0
    )

    trainer = UnifiedTrainer(
        session=session,
        accelerator=accelerator,
        model=model,
        loss_fn=loss_fn,
        optimiser=optimiser,
        scheduler=scheduler,
        config=config,
    )

    trained_model = trainer.fit(
        train_loader, H_base, H_mu_diag, Xi, val_loader=val_loader
    )

    print(trained_model)
    print({key: values[-1] for key, values in trainer.history.as_dict().items()})


if __name__ == "__main__":  # pragma: no cover
    main()
