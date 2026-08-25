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
        weight: Fraction of a batch drawn from this interval. The
            weights across all regions in a scheme must sum to 1.0.
    """
 
    low: float
    high: float
    weight: float

 
if __name__ == "__main__":
    #: Default region breakdown, equivalent to the original hard-coded scheme:
    #: a quarter of each batch drawn uniformly across the full domain, a
    #: quarter concentrated near each side of the topological transition
    #: (mu = -2, mu = +2), and a quarter drawn from the trivial-phase side.
    KITAEV_REGIONS = (
        SamplingRegion(low=-3.0, high=3.0, weight=0.25),
        SamplingRegion(low=-2.2, high=-1.8, weight=0.25),
        SamplingRegion(low=1.8, high=2.2, weight=0.25),
        SamplingRegion(low=2.0, high=3.0, weight=0.25),
    )
    