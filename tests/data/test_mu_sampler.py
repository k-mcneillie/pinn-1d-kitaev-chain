# tests/data/test_mu_sampler.py

import pytest

from kitaev.data.mu_sampler import DEFAULT_KITAEV_REGIONS, MuSampler
from kitaev.data.sampling_region import SamplingRegion


@pytest.fixture
def valid_regions():
    return [
        SamplingRegion(low=-3.0, high=3.0, weight=1.0),
    ]


@pytest.fixture
def mu_sampler(valid_regions):
    return MuSampler(regions=valid_regions)


def test_init_default_regions():
    assert len(DEFAULT_KITAEV_REGIONS) == 1
    assert DEFAULT_KITAEV_REGIONS[0].low == -3.0
    assert DEFAULT_KITAEV_REGIONS[0].high == 3.0
    assert DEFAULT_KITAEV_REGIONS[0].weight == 1.0


def test_invalid_regions():
    with pytest.raises(ValueError):
        MuSampler(regions=[])
    with pytest.raises(ValueError):
        MuSampler(regions=[SamplingRegion(low=-3.0, high=3.0, weight=0.5)])


def test_allocate_counts(mu_sampler):
    batch_size = 10
    counts = mu_sampler._allocate_counts(batch_size)
    assert sum(counts) == batch_size
    batch_size = 7
    counts = mu_sampler._allocate_counts(batch_size)
    assert sum(counts) == batch_size


def test_sample_region(mu_sampler):
    region = DEFAULT_KITAEV_REGIONS[0]
    count = 100
    samples = mu_sampler._sample_region(region, count)
    assert samples.shape == (count, 1)
    assert (samples >= region.low).all()
    assert (samples < region.high).all()


def test_sample(mu_sampler):
    batch_size = 50
    samples = mu_sampler.sample(batch_size)
    assert samples.shape == (batch_size, 1)
    assert (samples >= -3.0).all()
    assert (samples < 3.0).all()
