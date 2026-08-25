import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from kitaev.analytical import KitaevChainHamiltonian
from kitaev.data.mu_sampler import MuSampler


class SupervisedKitaevDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """A labelled (mu, E, psi) dataset built by exact diagonalisation.

    Unlike the unsupervised generator, this class carries real numerical
    responsibility: for every sampled mu it must diagonalise the BdG
    Hamiltonian and apply sign-continuity gauge fixing across
    neighbouring mu values. Gauge fixing is required because the
    eigensolver returns an eigenvector defined only up to an overall
    sign; without fixing that sign consistently, psi would flip
    unpredictably between adjacent mu samples, which would corrupt any
    downstream loss (e.g. a psi-continuity penalty) that assumes psi
    varies smoothly with mu.

    Gauge fixing requires the eigenvectors to be labelled in mu-sorted
    order — sign continuity is only meaningful relative to the
    *previous mu*, not relative to an arbitrary batch order. Samples are
    therefore sorted before labelling and restored to their original
    order afterwards, which is precisely the subtlety that a
    single-shared-method design (branching on a `supervised` flag) makes
    easy to break: it is not obvious from a shared method's signature
    that its unsupervised branch has no such ordering constraint while
    its supervised branch does.

    Attributes:
        hamiltonian: The :class:`KitaevChainHamiltonian` used to
            generate ground-truth labels.
        mu: Chemical-potential values, shape ``(n_samples, 1)``, in
            original (pre-sort) order.
        energy: Lowest non-negative eigenvalues, shape
            ``(n_samples, 1)``, matching ``mu``'s order.
        psi: Gauge-fixed eigenvectors, shape ``(n_samples, 2*N)``,
            matching ``mu``'s order.
    """

    def __init__(
        self,
        sampler: MuSampler,
        total_samples: int,
        hamiltonian: KitaevChainHamiltonian,
    ) -> None:
        """Samples mu and computes gauge-fixed exact labels for each value.

        Args:
            sampler: The :class:`MuSampler` used to draw the mu values
                to be labelled.
            total_samples: Number of (mu, E, psi) triples to generate.
            hamiltonian: The Hamiltonian builder used to diagonalise
                each sampled mu.
        """
        self.sampler = sampler
        self.total_samples = total_samples
        self.hamiltonian = hamiltonian
        self.dim = self.hamiltonian.dim

        mu_flat = self.sampler.sample(self.total_samples).cpu().numpy().flatten()
        energy, psi = self._compute_gauge_fixed_labels(mu_flat)

        self.mu = torch.tensor(mu_flat[:, None], dtype=torch.float32)
        self.energy = torch.tensor(energy[:, None], dtype=torch.float32)
        self.psi = torch.tensor(psi, dtype=torch.float32)

    def _compute_gauge_fixed_labels(
        self, mu: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Diagonalises the Hamiltonian at each mu with sign-continuity gauge fixing.

        Args:
            mu: 1D array of chemical-potential values, in arbitrary
                (unsorted) order.

        Returns:
            A tuple ``(energy, psi)`` where ``energy`` has shape
            ``(len(mu),)`` and ``psi`` has shape
            ``(len(mu), hamiltonian.dim)``, both restored to match the
            original order of ``mu``.
        """
        split_index = self.hamiltonian.n_sites

        sort_idx = np.argsort(mu)
        mu_sorted = mu[sort_idx]

        energy_sorted = np.zeros_like(mu_sorted)
        psi_sorted = np.zeros((len(mu_sorted), self.hamiltonian.dim))

        last_psi: npt.NDArray[np.float64] | None = None
        for i, mu_value in enumerate(mu_sorted):
            H = self.hamiltonian.build(mu_value)
            eigenvalues, eigenvectors = np.linalg.eigh(H)

            energy_sorted[i] = eigenvalues[split_index]
            psi = eigenvectors[:, split_index]

            if last_psi is not None and np.dot(psi, last_psi) < 0:
                psi = -psi
            last_psi = psi

            psi_sorted[i] = psi

        unsort_idx = np.argsort(sort_idx)
        return energy_sorted[unsort_idx], psi_sorted[unsort_idx]

    def __len__(self) -> int:
        """Returns the number of (mu, E, psi) triples in the dataset."""
        return self.total_samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        """Returns the (mu, E, psi) triple at the given index.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            A tuple ``(mu, energy, psi)`` for the sample at ``index``.
        """
        return self.mu[index], self.energy[index], self.psi[index]

    def dataloader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Wraps this dataset in a ``DataLoader``.

        Args:
            batch_size: Batch size used by the returned ``DataLoader``.
            shuffle: Whether to shuffle samples each epoch.

        Returns:
            A ``DataLoader`` yielding ``(mu, energy, psi)`` batches.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
