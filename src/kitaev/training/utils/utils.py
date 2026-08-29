import copy

import torch


class EpochAccumulator:
    """Accumulates per-batch metric dictionaries and averages them at epoch end.

    Replaces the manual ``epoch_metrics.get(k, 0.0) + v`` pattern with a
    small, independently testable object, so the epoch loop itself only
    has to call :meth:`update` once per batch.
    """

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._count = 0

    def update(self, metrics: dict[str, float]) -> None:
        """Adds one batch's metrics into the running totals.

        Args:
            metrics: Mapping of metric name to scalar value for a single
                batch.
        """
        for key, value in metrics.items():
            self._sums[key] = self._sums.get(key, 0.0) + value
        self._count += 1

    def averages(self) -> dict[str, float]:
        """Returns the mean of every accumulated metric over all updates.

        Returns:
            Mapping of metric name to its average value. Empty if
            :meth:`update` was never called.
        """
        if self._count == 0:
            return {}
        return {key: total / self._count for key, total in self._sums.items()}


class TrainingHistory:
    """Stores per-epoch averaged metrics across the full run, keyed by name.

    A thin wrapper around ``dict[str, list[float]]`` rather than the
    original's bare dictionary-of-lists inline in the trainer, so the
    "append, creating the list on first use" pattern lives in one place.
    """

    def __init__(self) -> None:
        self._series: dict[str, list[float]] = {}

    def record(self, key: str, value: float) -> None:
        """Appends a value to the named metric's series.

        Args:
            key: Metric name, typically prefixed ``train_`` or ``val_``.
            value: Value for the current epoch.
        """
        self._series.setdefault(key, []).append(value)

    def __getitem__(self, key: str) -> list[float]:
        return self._series[key]

    def __contains__(self, key: str) -> bool:
        return key in self._series

    def get(self, key: str, default: list[float] | None = None) -> list[float]:
        """Returns the named series, or ``default`` (else ``[]``) if absent."""
        return self._series.get(key, [] if default is None else default)

    def as_dict(self) -> dict[str, list[float]]:
        """Returns a plain dict copy of the full history, e.g. for plotting."""
        return dict(self._series)


class EarlyStopping:
    """Tracks the best validation loss and snapshots the corresponding model state.

    Separated out from the trainer so the "is this the best epoch so
    far, and should we stop" decision is independently readable and
    testable, rather than interleaved with logging and the epoch loop.

    Attributes:
        patience: Epochs without improvement before stopping is
            signalled. ``None`` disables stopping.
        track_state: When ``True`` the best epoch's model state dict is
            deep-copied into ``best_state`` so the trainer can roll back
            to it. When ``False`` only ``best_loss`` and ``best_epoch``
            are tracked (for logging and provenance) and ``best_state``
            stays ``None``, avoiding the per-improvement copy when the
            caller has opted out of best-checkpoint restoration.
        best_loss: Lowest validation loss observed so far.
        best_epoch: Epoch at which ``best_loss`` was recorded.
        best_state: Deep-copied, unwrapped model state dict at
            ``best_epoch``, or ``None`` when ``track_state`` is ``False``.
    """

    def __init__(self, patience: int | None, *, track_state: bool = True) -> None:
        self.patience = patience
        self.track_state = track_state
        self.best_loss: float = float("inf")
        self.best_epoch: int | None = None
        self.best_state: dict[str, torch.Tensor] | None = None
        self._epochs_without_improvement = 0

    def step(
        self, val_loss: float, epoch: int, unwrapped_model: torch.nn.Module
    ) -> bool:
        """Registers one epoch's validation loss and updates the best checkpoint.

        Args:
            val_loss: Validation loss for the current epoch.
            epoch: The current epoch number.
            unwrapped_model: The model with any accelerate wrapping
                removed (via ``accelerator.unwrap_model``), so the
                snapshotted state dict has plain, portable keys.

        Returns:
            ``True`` if training should stop now (patience exhausted),
            ``False`` otherwise.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            if self.track_state:
                self.best_state = copy.deepcopy(unwrapped_model.state_dict())
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        return (
            self.patience is not None
            and self._epochs_without_improvement >= self.patience
        )
