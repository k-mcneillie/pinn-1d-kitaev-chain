from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from .sampling_region import SamplingRegion

DEFAULT_KITAEV_REGIONS: tuple[SamplingRegion, ...] = (
    SamplingRegion(low=-3.0, high=3.0, weight=1),
)


class MuSampler:
    """Draws batches of chemical-potential values from a weighted mixture of intervals.

    This class owns exactly one responsibility: producing mu values
    according to a configurable sampling scheme. It knows nothing about
    Hamiltonians, labels, or PyTorch datasets, which is what allows it
    to be shared, unmodified, by both the unsupervised generator and the
    supervised dataset below — the sampling strategy and the consumer of
    the samples are two separate concerns.

    Weights are relative, not fractions that must sum to 1.0: each
    region's share of a sampled batch is its own weight divided by the
    sum of every region's weight in the scheme. This is what makes a
    scheme composable — adding, removing, or reweighting one region
    changes only that region's own share and everyone else's relative
    *proportions* stay fixed; nothing has to be manually recomputed to
    keep a fixed total, the way it would if weights had to sum to
    exactly 1.0.

    Attributes:
        regions: The weighted intervals making up the sampling scheme.
        device: The torch device on which sampled tensors are created.
    """

    def __init__(
        self,
        regions: Sequence[SamplingRegion],
    ) -> None:
        """Initialises the sampler with a given region scheme.

        Args:
            regions: Weighted intervals to sample from. Weights are
                relative (see the class docstring) and need not sum to
                any particular value, so long as their total is
                positive.

        Raises:
            ValueError: If ``regions`` is empty, or if the total weight
                across all regions is not strictly positive.
        """
        total_weight = sum(region.weight for region in regions)
        if not regions or total_weight <= 0:
            raise ValueError(
                "At least one region with a positive total weight is "
                f"required, got {len(regions)} region(s) with total "
                f"weight {total_weight!r}."
            )
        self.regions = tuple(regions)
        self._total_weight = total_weight

    def sample(self, batch_size: int) -> Tensor:
        """Draws a shuffled batch of mu values from the configured mixture.

        Each region contributes a number of samples proportional to its
        weight, with any rounding remainder assigned to the final region
        so the returned batch is always exactly ``batch_size`` long. The
        per-region samples are concatenated and then shuffled so that
        downstream consumers see a mixed batch rather than one ordered
        by region.

        Args:
            batch_size: Total number of mu values to draw.

        Returns:
            A tensor of shape ``(batch_size, 1)`` on ``self.device``.
        """
        counts = self._allocate_counts(batch_size)
        chunks = [
            self._sample_region(region, count)
            for region, count in zip(self.regions, counts, strict=True)
        ]
        mu_batch = torch.cat(chunks, dim=0)
        shuffle_idx = torch.randperm(batch_size)
        return mu_batch[shuffle_idx]

    def _allocate_counts(self, batch_size: int) -> list[int]:
        """Converts region weights into exact per-region sample counts.

        Args:
            batch_size: Total number of samples to allocate across
                regions.

        Returns:
            A list of integer counts, one per region, summing exactly
            to ``batch_size``.
        """
        counts = [
            int(batch_size * region.weight / self._total_weight)
            for region in self.regions
        ]
        counts[-1] += batch_size - sum(counts)
        return counts

    def _sample_region(self, region: SamplingRegion, count: int) -> Tensor:
        """Draws ``count`` uniform samples from a single region.

        Args:
            region: The interval to sample from.
            count: Number of samples to draw.

        Returns:
            A tensor of shape ``(count, 1)`` on ``self.device``.
        """
        span = region.high - region.low
        return torch.rand(count, 1) * span + region.low
