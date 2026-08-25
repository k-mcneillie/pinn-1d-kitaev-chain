from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingRegion:
    """A single weighted interval used to build a mixture distribution over mu.

    Rather than hard-coding a fixed number of named regions inside the
    sampler itself, the sampling scheme is expressed as a list of these
    objects. This is what makes the breakdown configurable: adding,
    removing, or reweighting regions (e.g. widening the transition
    window, or dropping the RHS oversampling entirely) is a change to
    the region list passed in, not a change to the sampler's code.

    Attributes:
        low: Lower bound of the interval (inclusive).
        high: Upper bound of the interval (exclusive).
        weight: Relative weight of this interval within its scheme.
            A region's actual share of a sampled batch is its weight
            divided by the sum of every region's weight in the same
            :class:`~kitaev.data.mu_sampler.MuSampler` — weights need
            not sum to 1.0 (or any other particular value), which is
            what lets a scheme be extended by adding a region without
            having to rebalance every other region's weight by hand.
    """

    low: float
    high: float
    weight: float


#: Region breakdown concentrating samples near the topological transition:
#: a quarter of each batch drawn uniformly across the full domain, a
#: quarter concentrated near each side of the transition (mu = -2,
#: mu = +2), and a quarter drawn from the trivial-phase side.
TRANSITION_FOCUSED_REGIONS: tuple[SamplingRegion, ...] = (
    SamplingRegion(low=-3.0, high=3.0, weight=0.25),
    SamplingRegion(low=-2.2, high=-1.8, weight=0.25),
    SamplingRegion(low=1.8, high=2.2, weight=0.25),
    SamplingRegion(low=2.0, high=3.0, weight=0.25),
)
