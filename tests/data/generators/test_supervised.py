# tests/data/generators/test_supervised.py

import numpy as np
import pytest
import torch

from kitaev.analytical import KitaevChainHamiltonian
from kitaev.data.generators.supervised import SupervisedKitaevDataset
from kitaev.data.mu_sampler import MuSampler
from kitaev.data.sampling_region import SamplingRegion


@pytest.fixture
def setup_supervised_dataset():
    n_sites = 2
    total_samples = 10
    hamiltonian = KitaevChainHamiltonian(n_sites=n_sites)
    sampler = MuSampler(regions=[SamplingRegion(low=-3.0, high=3.0, weight=1.0)])
    return SupervisedKitaevDataset(
        sampler=sampler, total_samples=total_samples, hamiltonian=hamiltonian
    )


def test_init_supervised_dataset(setup_supervised_dataset):
    dataset = setup_supervised_dataset
    assert len(dataset) == 10
    assert dataset.dim == 4  # 2n_sites


def test_getitem(setup_supervised_dataset):
    dataset = setup_supervised_dataset
    index = 0
    mu, energy, psi = dataset[index]
    assert mu.shape == (1,)
    assert energy.shape == (1,)
    assert psi.shape == (4,)


def test_dataloader(setup_supervised_dataset):
    dataset = setup_supervised_dataset
    dataloader = dataset.dataloader(batch_size=4, shuffle=True)
    assert isinstance(dataloader, torch.utils.data.DataLoader)
    assert dataloader.batch_size == 4


def test_compute_gauge_fixed_labels(setup_supervised_dataset):
    mu_values = np.random.rand(5)
    energy, psi = setup_supervised_dataset._compute_gauge_fixed_labels(mu_values)
    assert energy.shape == (5,)
    assert psi.shape == (5, 4)


@pytest.mark.parametrize("mu_value", [-3.0, 0.0, 3.0])
def test_edge_cases(mu_value):
    # Test edge cases for mu_sampler.sample
    pass
