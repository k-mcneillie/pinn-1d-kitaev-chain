# tests/data/generators/test_unsupervised.py
from itertools import islice

import pytest
from torch.utils.data import DataLoader

from kitaev.data.generators.unsupervised import UnsupervisedMuGenerator
from kitaev.data.mu_sampler import DEFAULT_KITAEV_REGIONS, MuSampler


@pytest.fixture
def unsupervised_generator():
    sampler = MuSampler(DEFAULT_KITAEV_REGIONS)
    return UnsupervisedMuGenerator(sampler=sampler)


def test_infinite_batches(unsupervised_generator):
    batches = islice(unsupervised_generator.infinite_batches(batch_size=32), 5)
    for batch in batches:
        assert batch.shape == (32, 1)


def test_dataloader(unsupervised_generator):
    dataloader = unsupervised_generator.dataloader(total_samples=100, batch_size=32)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == 32
    batches = list(dataloader)
    assert all(len(batch) == 1 for batch in batches)
    assert batches[0][0].shape == (32, 1)
    assert batches[-1][0].shape == (4, 1)
    assert sum(b[0].shape[0] for b in batches) == 100
