# tests/data/test_sampling_region.py

import numpy as np
import pytest

from kitaev.data.mu_sampler import DEFAULT_KITAEV_REGIONS
from kitaev.data.sampling_region import SamplingRegion


@pytest.fixture
def valid_regions():
    return [
        SamplingRegion(low=-3.0, high=3.0, weight=0.25),
        SamplingRegion(low=-2.2, high=-1.8, weight=0.25),
        SamplingRegion(low=1.8, high=2.2, weight=0.25),
        SamplingRegion(low=2.0, high=3.0, weight=0.25),
    ]


def test_init_sampling_region():
    region = SamplingRegion(low=-3.0, high=3.0, weight=0.25)
    assert region.low == -3.0
    assert region.high == 3.0
    assert region.weight == 0.25


def test_dataclass_immutability():
    region = SamplingRegion(low=-3.0, high=3.0, weight=0.25)
    with pytest.raises(AttributeError):
        region.low = 0.0  # type: ignore


def test_region_equality_and_hash():
    region1 = SamplingRegion(low=-3.0, high=3.0, weight=0.25)
    region2 = SamplingRegion(low=-3.0, high=3.0, weight=0.25)
    assert region1 == region2
    assert hash(region1) == hash(region2)


def test_main_block():
    regions = DEFAULT_KITAEV_REGIONS
    assert len(regions) == 1
    assert all(isinstance(r, SamplingRegion) for r in regions)
    assert np.isclose(sum(r.weight for r in regions), 1.0)
