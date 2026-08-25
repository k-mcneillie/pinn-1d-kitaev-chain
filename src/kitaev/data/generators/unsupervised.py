from collections.abc import Iterator

from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from kitaev.data.mu_sampler import MuSampler


class UnsupervisedMuGenerator:
    """Produces mu-only batches for physics-residual (label-free) training.

    This generator has no knowledge of the Hamiltonian or of exact
    eigenstates — it exists purely to turn a :class:`MuSampler` into the
    two consumption patterns training typically needs: an infinite
    on-device stream, or a fixed-size, shuffled, epoch-based
    ``DataLoader``.

    Attributes:
        sampler: The :class:`MuSampler` used to draw batches.
    """

    def __init__(self, sampler: MuSampler) -> None:
        """Initialises the generator with a given sampler.

        Args:
            sampler: The :class:`MuSampler` used to draw batches.
        """
        self.sampler = sampler

    def infinite_batches(self, batch_size: int) -> Iterator[Tensor]:
        """Yields an unending stream of mu batches, sampled on demand.

        Suited to continuous unsupervised PINN training where each
        optimisation step consumes a fresh batch rather than iterating
        over a fixed, pre-generated dataset.

        Args:
            batch_size: Number of mu values per yielded batch.

        Yields:
            Tensors of shape ``(batch_size, 1)``.
        """
        while True:
            yield self.sampler.sample(batch_size)

    def dataloader(self, total_samples: int, batch_size: int) -> DataLoader:
        """Builds a fixed-size, shuffled ``DataLoader`` of mu-only batches.

        Args:
            total_samples: Total number of mu values to pre-generate.
            batch_size: Batch size used by the returned ``DataLoader``.

        Returns:
            A ``DataLoader`` yielding batches of shape
            ``(batch_size, 1)``.
        """
        mu = self.sampler.sample(total_samples).cpu()
        dataset = TensorDataset(mu)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
