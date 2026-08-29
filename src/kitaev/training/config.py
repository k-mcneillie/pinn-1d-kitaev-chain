from dataclasses import dataclass


@dataclass
class TrainerConfig:
    """Plain configuration for :class:`UnifiedTrainer`.

    Bundling these as a dataclass rather than loose constructor
    arguments keeps ``UnifiedTrainer.__init__`` focused on wiring
    together its collaborators (model, optimiser, accelerator), with the
    numeric training knobs kept separately and easy to snapshot,
    log, or sweep over independently of the trainer's other state.

    Attributes:
        epochs: Maximum number of training epochs to run.
        print_freq: How often (in epochs) to log a progress line.
        patience: Number of epochs without validation improvement
            before early stopping triggers. ``None`` disables early
            stopping entirely.
        grad_clip_norm: Maximum gradient norm for clipping on standard
            (non-L-BFGS) optimiser steps. ``None`` disables clipping.
        restore_best: When ``True`` (the default) and a validation loader
            is used, the model returned by ``fit`` is rolled back to the
            state that had the lowest validation loss. When ``False`` the
            final-epoch state is returned instead, and the best epoch is
            only recorded, not restored. Set ``False`` for fixed-budget
            training where a plateaued validation loss (the folded
            spectrum loss has a floor of the energy squared) would
            otherwise freeze the model while the eigenvector is still
            sharpening.
    """

    epochs: int = 3000
    print_freq: int = 500
    patience: int | None = None
    grad_clip_norm: float | None = 1.0
    restore_best: bool = True


@dataclass
class TwoPhaseConfig:
    """Config for the Adam(W)-then-L-BFGS two-phase optimiser schedule.

    The established PINN recipe: a first-order method (AdamW here) to get
    the parameters into a good basin quickly and robustly, then a
    quasi-Newton method (L-BFGS with a strong-Wolfe line search) to drive
    the residual down the last few orders of magnitude, which L-BFGS does
    far more efficiently than Adam once the loss surface is locally
    well-behaved. Consumed by
    :func:`kitaev.training.trainer.run_two_phase`.

    Attributes:
        adam_epochs: Epochs of the AdamW warm-up phase. A cosine schedule
            anneals its learning rate to zero over this many epochs.
        adam_lr: AdamW initial learning rate.
        adam_weight_decay: AdamW weight decay.
        lbfgs_epochs: Epochs of the L-BFGS fine-tuning phase. ``0`` skips
            it (AdamW only).
        lbfgs_lr: L-BFGS learning rate. With ``line_search_fn`` set, the
            line search chooses the step and this is effectively an upper
            bound, so ``1.0`` is the usual choice.
        lbfgs_max_iter: L-BFGS iterations per optimiser step (per batch).
        lbfgs_history_size: Number of past updates L-BFGS keeps for its
            inverse-Hessian approximation.
        lbfgs_line_search_fn: Line search passed to ``torch.optim.LBFGS``;
            ``"strong_wolfe"`` is strongly recommended for PINNs. ``None``
            disables the line search.
    """

    adam_epochs: int = 2000
    adam_lr: float = 8e-4
    adam_weight_decay: float = 1e-6
    lbfgs_epochs: int = 300
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 20
    lbfgs_history_size: int = 100
    lbfgs_line_search_fn: str | None = "strong_wolfe"
