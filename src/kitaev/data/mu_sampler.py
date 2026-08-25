from __future__ import annotations
from .sampling_region import SamplingRegion
from collections.abc import Sequence
import torch
import numpy as np
from torch import Tensor

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
            regions: Weighted intervals to sample from. Weights must sum
                to 1.0 (within floating-point tolerance).
 
        Raises:
            ValueError: If the region weights do not sum to 1.0.
        """
        total_weight = sum(region.weight for region in regions)
        if not np.isclose(total_weight, 1.0):
            raise ValueError(
                f"Region weights must sum to 1.0, got {total_weight!r} "
                f"from {len(regions)} region(s)."
            )
        self.regions = tuple(regions)
 
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
        counts = [int(batch_size * region.weight) for region in self.regions]
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
 
 