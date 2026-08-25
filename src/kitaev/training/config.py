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
    """

    epochs: int = 3000
    print_freq: int = 500
    patience: int | None = None
    grad_clip_norm: float | None = 1.0
