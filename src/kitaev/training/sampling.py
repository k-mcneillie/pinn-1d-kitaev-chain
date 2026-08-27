# src/kitaev/training/sampling.py
"""Streaming (infinite) collocation sampling for label-free PINN training.

The default label-free pipeline (:meth:`UnsupervisedMuGenerator.dataloader`)
pre-generates a **fixed** pool of ``mu`` points and re-shuffles it every
epoch, so the physics residual is only ever evaluated on that frozen grid.
This module adds the alternative: a :class:`torch.utils.data.IterableDataset`
that draws **fresh** ``mu`` values every step, plus three
:class:`~kitaev.training.callbacks.TrainingCallback` sampling *strategies*
that decide, epoch by epoch, where those fresh points come from:

- :class:`FixedRegionSampling` -- plain infinite sampling from a static
  region mixture (the streaming analogue of the frozen pool).
- :class:`CurriculumSampling` -- the active region mixture changes at
  configured epoch boundaries (e.g. topological interior first, then admit
  the trivial phase).
- :class:`AdaptiveSampling` -- residual-adaptive refinement (RAR/RAD):
  every few epochs the current model's per-point residual is evaluated on a
  large candidate pool and the worst points are folded back into the
  sampling distribution.

All three are opt-in and configured through :class:`SamplingConfig`; the
frozen pool remains the default.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from kitaev.data.mu_sampler import MuSampler
from kitaev.data.sampling_region import SamplingRegion

from .callbacks import TrainingCallback
from .loss import chiral_pointwise_residual
from .utils import TrainingHistory


class SamplingStrategy(TrainingCallback):
    """A per-epoch source of fresh ``mu`` batches for streaming training.

    Subclasses implement :meth:`draw` and, if they need to change what they
    sample as training progresses, override the
    :class:`~kitaev.training.callbacks.TrainingCallback` hooks. The same
    object is passed both to :class:`StreamingMuDataset` (which calls
    :meth:`draw`) and to ``UnifiedTrainer(callbacks=...)`` (which calls the
    hooks), so its epoch-to-epoch state stays consistent.
    """

    def draw(self, batch_size: int) -> Tensor:
        """Draws one fresh batch of ``mu`` values.

        Args:
            batch_size: Number of ``mu`` values to return.

        Returns:
            A tensor of shape ``(batch_size, 1)``.
        """
        raise NotImplementedError  # pragma: no cover - abstract


class FixedRegionSampling(SamplingStrategy):
    """Infinite sampling from a single, unchanging region mixture.

    The streaming counterpart of the frozen pool: every step is a fresh
    draw from the same :class:`~kitaev.data.mu_sampler.MuSampler`, so the
    model never sees the same collocation grid twice.

    Attributes:
        sampler: The sampler every batch is drawn from.
    """

    def __init__(self, sampler: MuSampler) -> None:
        """Initialises the strategy with a fixed sampler.

        Args:
            sampler: The :class:`~kitaev.data.mu_sampler.MuSampler` to draw
                every batch from.
        """
        self.sampler = sampler

    @classmethod
    def from_regions(cls, regions: Sequence[SamplingRegion]) -> FixedRegionSampling:
        """Builds the strategy directly from a region mixture.

        Args:
            regions: Weighted intervals to sample from.

        Returns:
            A configured :class:`FixedRegionSampling`.
        """
        return cls(MuSampler(regions))

    def draw(self, batch_size: int) -> Tensor:
        """Draws a fresh batch from the fixed sampler."""
        return self.sampler.sample(batch_size)


class CurriculumSampling(SamplingStrategy):
    """Region mixture that switches at configured epoch boundaries.

    ``stages`` is a list of ``(start_epoch, regions)`` pairs. Whenever the
    trainer starts an epoch ``>= start_epoch``, that stage's region mixture
    becomes the one :meth:`draw` samples from. Stages are applied in
    ascending ``start_epoch`` order; the first stage should start at epoch
    ``1`` so a mixture is always active.

    A typical schedule for the Kitaev chain trains the topological interior
    first (where the near-zero mode converges fastest and the ``(u, v)``
    gauge settles cleanly) before admitting the dense trivial-phase
    spectrum.

    Attributes:
        stages: The ``(start_epoch, MuSampler)`` schedule, sorted by
            ``start_epoch``.
    """

    def __init__(
        self,
        stages: Sequence[tuple[int, Sequence[SamplingRegion]]],
    ) -> None:
        """Initialises the curriculum from a stage schedule.

        Args:
            stages: ``(start_epoch, regions)`` pairs. Need not be
                pre-sorted. The earliest ``start_epoch`` should be ``1``.

        Raises:
            ValueError: If ``stages`` is empty.
        """
        if not stages:
            raise ValueError("CurriculumSampling requires at least one stage.")
        ordered = sorted(stages, key=lambda item: item[0])
        self.stages: list[tuple[int, MuSampler]] = [
            (start, MuSampler(regions)) for start, regions in ordered
        ]
        self._active = self.stages[0][1]

    def on_epoch_start(
        self,
        epoch: int,
        model: torch.nn.Module,
        history: TrainingHistory,
    ) -> None:
        """Selects the stage whose ``start_epoch`` this epoch has reached."""
        del model, history
        for start, sampler in self.stages:
            if epoch >= start:
                self._active = sampler
            else:
                break

    def draw(self, batch_size: int) -> Tensor:
        """Draws a fresh batch from the currently active stage."""
        return self._active.sample(batch_size)


class AdaptiveSampling(SamplingStrategy):
    """Residual-adaptive refinement (RAR/RAD) of the sampling distribution.

    Every ``refine_every`` epochs the current model's per-point residual
    (:func:`kitaev.training.loss.chiral_pointwise_residual`) is evaluated on
    a fresh candidate pool of ``candidate_pool`` points drawn from
    ``base``. The worst ``keep_fraction`` of them are retained as "hot"
    points. Between refinements, a fraction ``adaptive_fraction`` of every
    batch is drawn as ``hot point + Gaussian jitter`` (clamped to
    ``domain``) and the remainder from ``base``.

    Before the first refinement the strategy behaves exactly like
    :class:`FixedRegionSampling` over ``base``.

    Attributes:
        base: The background sampler; also the candidate-pool source.
        n_sites: Physical site count ``N`` for the residual operator.
        hopping: Hopping amplitude ``t`` for the residual operator.
        pairing: Pairing amplitude ``delta`` for the residual operator.
        refine_every: Epoch interval between refinements.
        candidate_pool: Number of candidate points scored per refinement.
        keep_fraction: Fraction of the candidate pool retained as hot
            points.
        adaptive_fraction: Fraction of each drawn batch taken from the hot
            points.
        jitter: Standard deviation of the Gaussian noise added to a hot
            point when it is resampled.
        domain: ``(low, high)`` clamp applied to jittered hot points.
    """

    def __init__(
        self,
        base: MuSampler,
        *,
        n_sites: int,
        hopping: float = 1.0,
        pairing: float = 0.5,
        refine_every: int = 200,
        candidate_pool: int = 4096,
        keep_fraction: float = 0.25,
        adaptive_fraction: float = 0.5,
        jitter: float = 0.05,
        domain: tuple[float, float] = (0.05, 4.0),
    ) -> None:
        """Initialises the adaptive strategy. See the class docstring for args."""
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError("keep_fraction must be in (0, 1].")
        if not 0.0 <= adaptive_fraction <= 1.0:
            raise ValueError("adaptive_fraction must be in [0, 1].")
        self.base = base
        self.n_sites = n_sites
        self.hopping = hopping
        self.pairing = pairing
        self.refine_every = refine_every
        self.candidate_pool = candidate_pool
        self.keep_fraction = keep_fraction
        self.adaptive_fraction = adaptive_fraction
        self.jitter = jitter
        self.domain = domain
        self._hot: Tensor | None = None

    def draw(self, batch_size: int) -> Tensor:
        """Draws a batch, mixing hot-point resamples with background samples."""
        if self._hot is None or self.adaptive_fraction == 0.0:
            return self.base.sample(batch_size)

        n_adaptive = int(round(batch_size * self.adaptive_fraction))
        n_base = batch_size - n_adaptive

        picks = torch.randint(0, self._hot.shape[0], (n_adaptive,))
        hot = self._hot[picks].unsqueeze(-1)
        hot = hot + self.jitter * torch.randn(n_adaptive, 1)
        hot = hot.clamp(self.domain[0], self.domain[1])

        parts = [hot]
        if n_base > 0:
            parts.append(self.base.sample(n_base))
        batch = torch.cat(parts, dim=0)
        return batch[torch.randperm(batch.shape[0])]

    def on_epoch_end(
        self,
        epoch: int,
        model: torch.nn.Module,
        history: TrainingHistory,
    ) -> None:
        """Rescores a candidate pool and refreshes the hot points."""
        del history
        if epoch % self.refine_every != 0:
            return

        try:
            device = next(model.parameters()).device
        except StopIteration:  # pragma: no cover - models always have params
            device = torch.device("cpu")

        candidates = self.base.sample(self.candidate_pool).to(device)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            residual = chiral_pointwise_residual(
                model,
                candidates,
                self.n_sites,
                hopping=self.hopping,
                pairing=self.pairing,
            )
        model.train(was_training)

        keep = max(1, int(self.candidate_pool * self.keep_fraction))
        worst = torch.topk(residual, keep).indices
        self._hot = candidates[worst].reshape(-1).detach().cpu()


class StreamingMuDataset(IterableDataset):
    """Yields a fixed number of freshly-sampled ``mu`` batches per epoch.

    Each ``__iter__`` (one per trainer epoch) produces ``steps_per_epoch``
    batches, each a fresh :meth:`SamplingStrategy.draw`. Wrap it in a
    ``DataLoader(dataset, batch_size=None)`` -- the dataset has already done
    the batching -- which is what :func:`streaming_dataloader` returns.

    Attributes:
        strategy: The sampling strategy queried for every batch.
        batch_size: Number of ``mu`` values per batch.
        steps_per_epoch: Number of batches yielded per epoch.
    """

    def __init__(
        self,
        strategy: SamplingStrategy,
        *,
        batch_size: int,
        steps_per_epoch: int,
    ) -> None:
        """Initialises the dataset.

        Args:
            strategy: The :class:`SamplingStrategy` to draw batches from.
            batch_size: Number of ``mu`` values per batch.
            steps_per_epoch: Number of batches to yield per epoch.
        """
        super().__init__()
        self.strategy = strategy
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

    def __iter__(self) -> Iterator[tuple[Tensor]]:
        """Yields ``steps_per_epoch`` single-element ``(mu,)`` tuples."""
        for _ in range(self.steps_per_epoch):
            yield (self.strategy.draw(self.batch_size),)


def streaming_dataloader(
    strategy: SamplingStrategy,
    *,
    batch_size: int,
    steps_per_epoch: int,
) -> DataLoader:
    """Builds a ``DataLoader`` over a :class:`StreamingMuDataset`.

    Args:
        strategy: The sampling strategy to stream batches from.
        batch_size: Number of ``mu`` values per batch.
        steps_per_epoch: Number of batches per epoch.

    Returns:
        A ``DataLoader`` yielding ``(mu,)`` batches of shape
        ``(batch_size, 1)``; ``batch_size=None`` because the dataset
        batches internally.
    """
    dataset = StreamingMuDataset(
        strategy, batch_size=batch_size, steps_per_epoch=steps_per_epoch
    )
    return DataLoader(dataset, batch_size=None)


@dataclass
class SamplingConfig:
    """Selects and parameterises the label-free ``mu`` sampling scheme.

    ``mode`` picks the strategy; the remaining fields configure it. Only
    the fields relevant to the chosen ``mode`` are read.

    Attributes:
        mode: One of ``"frozen"`` (default; the pre-generated pool),
            ``"infinite"`` (:class:`FixedRegionSampling`),
            ``"curriculum"`` (:class:`CurriculumSampling`), or
            ``"adaptive"`` (:class:`AdaptiveSampling`).
        batch_size: ``mu`` values per batch, all modes.
        steps_per_epoch: Batches per epoch for the streaming modes
            (``infinite`` / ``curriculum`` / ``adaptive``). One trainer
            "epoch" is this many optimiser steps.
        total_samples: Frozen-pool size (``frozen`` mode only).
        curriculum_stages: ``(start_epoch, regions)`` schedule
            (``curriculum`` mode only). The earliest ``start_epoch`` should
            be ``1``.
        n_sites: Physical site count ``N`` for the adaptive residual
            operator (``adaptive`` mode only).
        hopping: Hopping ``t`` for the adaptive residual operator.
        pairing: Pairing ``delta`` for the adaptive residual operator.
        refine_every: Epoch interval between adaptive refinements.
        candidate_pool: Candidate points scored per adaptive refinement.
        keep_fraction: Fraction of the candidate pool kept as hot points.
        adaptive_fraction: Fraction of each batch drawn near hot points.
        jitter: Gaussian noise standard deviation for hot-point resamples.
        domain: ``(low, high)`` clamp for jittered hot points.
    """

    mode: str = "frozen"
    batch_size: int = 1024
    steps_per_epoch: int = 8
    total_samples: int = 8192
    curriculum_stages: Sequence[tuple[int, Sequence[SamplingRegion]]] | None = None
    n_sites: int = 20
    hopping: float = 1.0
    pairing: float = 0.5
    refine_every: int = 200
    candidate_pool: int = 4096
    keep_fraction: float = 0.25
    adaptive_fraction: float = 0.5
    jitter: float = 0.05
    domain: tuple[float, float] = (0.05, 4.0)
    _valid_modes: tuple[str, ...] = field(
        default=("frozen", "infinite", "curriculum", "adaptive"),
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validates ``mode``."""
        if self.mode not in self._valid_modes:
            raise ValueError(
                f"Unknown sampling mode {self.mode!r}; expected one of "
                f"{self._valid_modes}."
            )


def build_sampling(
    config: SamplingConfig,
    regions: Sequence[SamplingRegion],
) -> tuple[DataLoader, list[TrainingCallback]]:
    """Turns a :class:`SamplingConfig` into a loader and its trainer callbacks.

    Args:
        config: The sampling configuration.
        regions: The base region mixture. Used directly for ``frozen`` /
            ``infinite`` / ``adaptive`` modes and as the fallback if a
            ``curriculum`` schedule is not supplied.

    Returns:
        ``(train_loader, callbacks)``. Pass ``train_loader`` to
        :meth:`UnifiedTrainer.fit` and ``callbacks`` to
        ``UnifiedTrainer(callbacks=...)``. ``callbacks`` is empty for
        ``frozen`` and ``infinite`` modes.
    """
    if config.mode == "frozen":
        from kitaev.data.generators.unsupervised import UnsupervisedMuGenerator

        generator = UnsupervisedMuGenerator(sampler=MuSampler(regions))
        loader = generator.dataloader(
            total_samples=config.total_samples, batch_size=config.batch_size
        )
        return loader, []

    strategy: SamplingStrategy
    if config.mode == "infinite":
        strategy = FixedRegionSampling(MuSampler(regions))
    elif config.mode == "curriculum":
        stages = config.curriculum_stages or ((1, tuple(regions)),)
        strategy = CurriculumSampling(stages)
    else:  # "adaptive"
        strategy = AdaptiveSampling(
            MuSampler(regions),
            n_sites=config.n_sites,
            hopping=config.hopping,
            pairing=config.pairing,
            refine_every=config.refine_every,
            candidate_pool=config.candidate_pool,
            keep_fraction=config.keep_fraction,
            adaptive_fraction=config.adaptive_fraction,
            jitter=config.jitter,
            domain=config.domain,
        )

    loader = streaming_dataloader(
        strategy,
        batch_size=config.batch_size,
        steps_per_epoch=config.steps_per_epoch,
    )
    return loader, [strategy]
