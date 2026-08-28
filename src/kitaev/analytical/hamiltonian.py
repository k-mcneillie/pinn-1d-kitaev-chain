# kitaev/analytical/hamiltonian.py
#
# ==========================
# Import Packages
# ==========================
from __future__ import annotations

import numpy as np
import torch


class KitaevChainHamiltonian:
    """Build the Bogoliubov-de Gennes (BdG) Hamiltonian for a 1D Kitaev chain.

    The Kitaev chain is a toy model of a one-dimensional p-wave
    superconductor made of N spinless fermions. Physically, it is the
    simplest system that hosts a topological phase transition and, in its
    non-trivial phase, a pair of Majorana bound states (MBS) localised at
    the two ends of the chain. It plays the same structural role for this
    project as the 2D magnetic-impurity/superconductor lattice from the
    dissertation work, but at a scale small enough that exact
    diagonalisation is trivial, making it a convenient testbed for the
    PINN surrogate.

    The second-quantised Hamiltonian is

        H = -mu * sum_n c_n^dagger c_n
            + sum_n [
                -t * c_{n+1}^dagger c_n
                + delta * c_n c_{n+1}
                + h.c.
            ]

    where c_n and c_n^dagger annihilate and create a spinless fermion on
    site n, respectively. The parameters are the chemical potential mu,
    nearest-neighbour hopping amplitude t, and p-wave pairing amplitude
    delta.

    The pairing term, delta * c_n c_{n+1}, is what makes this a
    superconducting rather than a purely tight-binding problem. It creates
    and annihilates pairs of particles rather than conserving particle
    number, so the Hamiltonian cannot be diagonalised in the ordinary
    particle-number basis.

    Why a doubled (Nambu/BdG) basis is needed
    ------------------------------------------
    Because the pairing term mixes creation and annihilation operators,
    the natural basis for diagonalisation is the Nambu spinor basis, which
    treats particle and hole degrees of freedom on an equal footing:

        Psi = (
            c_1, ..., c_N,
            c_1^dagger, ..., c_N^dagger
        )^T

    Writing H in this doubled basis produces a 2N x 2N matrix with a
    built-in redundancy. For every eigenvalue +E there is a partner
    eigenvalue -E due to particle-hole symmetry, and the corresponding
    eigenvectors are related by swapping the particle and hole blocks.

    This redundancy is a mathematical consequence of doubling the basis,
    rather than additional physical degrees of freedom. However, it is
    precisely this structure that allows zero-energy (E = 0) solutions to
    be interpreted as Majorana modes: a state pinned to E = 0 is, in this
    basis, its own particle-hole partner.

    Matrix structure
    ----------------
    In the particle-block/hole-block layout used below, the 2N x 2N
    matrix is built from four N x N blocks:

        H = [[ A,   B ],
             [ B^T, -A ]]

    The blocks have the following physical interpretation:

        A:
            The particle-sector block. Its diagonal encodes the on-site
            chemical potential (-mu per site), while its off-diagonal
            elements encode ordinary nearest-neighbour hopping (-t).

        -A:
            The hole-sector block. It is the particle-hole-symmetric
            counterpart of the particle block, with the corresponding
            signs reversed.

        B:
            The off-diagonal particle-hole coupling block. It encodes the
            p-wave pairing amplitude delta and couples particle and hole
            degrees of freedom on neighbouring sites.

    The pairing structure contains opposite signs for the two orientations
    of each nearest-neighbour bond. This antisymmetry is required by the
    fermionic relation

        c_n c_{n+1} = -c_{n+1} c_n.

    The pairing term is therefore what couples the particle and hole
    sectors together and is ultimately responsible for the topological
    phase transition and the emergence of Majorana edge modes as mu is
    tuned through the phase boundary.

    The topological phase transition occurs at

        |mu| = 2 * |t|.

    For |mu| < 2 * |t|, the chain is in the topologically non-trivial phase
    and, for open boundary conditions, hosts a pair of near-zero-energy
    Majorana edge modes. For |mu| > 2 * |t|, the chain is topologically
    trivial.

    This provides a simple one-dimensional setting in which the
    topological superconductivity relevant to the dissertation work can
    be studied. The resulting Hamiltonian is small enough for exact
    diagonalisation, making it suitable as a ground-truth operator for
    the PINN surrogate.

    Attributes:
        n_sites: Number of physical lattice sites in the chain.
        hopping: Nearest-neighbour hopping amplitude.
        pairing: P-wave pairing amplitude.
        dim: Dimension of the BdG matrix, equal to ``2 * n_sites``.
    """

    def __init__(
        self,
        n_sites: int = 20,
        hopping: float = 1.0,
        pairing: float = 0.5,
    ) -> None:
        """Initialise the Kitaev-chain Hamiltonian.

        Args:
            n_sites: Number of physical lattice sites in the chain. The
                resulting BdG Hamiltonian has dimension ``2 * n_sites``.
            hopping: Nearest-neighbour hopping amplitude, t.
            pairing: P-wave pairing amplitude, delta.
        """
        self.n_sites = n_sites
        self.hopping = hopping
        self.pairing = pairing
        self.dim = 2 * n_sites

    def build(self, mu: float) -> np.ndarray:
        """Assemble the 2N x 2N BdG Hamiltonian at chemical potential mu.

        For each site n, the on-site chemical potential contributes -mu to
        the particle sector and +mu to the hole sector. The opposite signs
        reflect the particle-hole structure of the BdG representation.

        For each nearest-neighbour bond (n, n + 1):

        - The particle sector receives a hopping matrix element -t.
        - The hole sector receives the corresponding +t term.
        - The pairing amplitude delta couples a particle on one site to a
          hole on the neighbouring site.
        - The opposite pairing signs on the two orientations of the bond
          enforce the antisymmetry required by fermionic pairing.

        The resulting matrix is real and symmetric for the parameterisation
        used here and can therefore be diagonalised directly using
        ``numpy.linalg.eigh``.

        The lowest non-negative eigenvalue and its corresponding
        eigenvector provide exact ground-truth quantities for the PINN
        surrogate. In particular, the dependence of these quantities on
        ``mu`` provides the spectral structure that the surrogate is
        required to learn.

        Args:
            mu: Chemical potential at which to evaluate the Hamiltonian.
                This is the control parameter for the topological phase
                transition and the input variable for the PINN surrogate.

        Returns:
            A real symmetric NumPy array of shape
            ``(2 * n_sites, 2 * n_sites)`` representing the BdG Hamiltonian
            at the specified chemical potential.
        """
        n_sites = self.n_sites
        hopping = self.hopping
        pairing = self.pairing

        hamiltonian = np.zeros(
            (self.dim, self.dim),
            dtype=float,
        )

        # On-site chemical potential: opposite signs in the particle
        # and hole sectors due to particle-hole symmetry.
        for site in range(n_sites):
            hamiltonian[site, site] = -mu
            hamiltonian[n_sites + site, n_sites + site] = mu

        # Nearest-neighbour hopping and p-wave pairing.
        for site in range(n_sites - 1):
            next_site = site + 1

            # Ordinary hopping:
            # particle sector (-t), hole-sector mirror (+t).
            hamiltonian[site, next_site] = -hopping
            hamiltonian[next_site, site] = -hopping

            hamiltonian[
                n_sites + site,
                n_sites + next_site,
            ] = hopping
            hamiltonian[
                n_sites + next_site,
                n_sites + site,
            ] = hopping

            # P-wave pairing:
            # couples a particle on one site to a hole on the neighbouring
            # site, with the antisymmetric sign pattern required by
            # fermionic statistics.
            hamiltonian[site, n_sites + next_site] = pairing
            hamiltonian[n_sites + next_site, site] = pairing

            hamiltonian[next_site, n_sites + site] = -pairing
            hamiltonian[n_sites + site, next_site] = -pairing

        return hamiltonian

    def __call__(self, mu: float) -> np.ndarray:
        """Build the Hamiltonian at the specified chemical potential.

        This provides shorthand access to :meth:`build`, allowing a
        ``KitaevChainHamiltonian`` instance to be called directly.

        Args:
            mu: Chemical potential at which to evaluate the Hamiltonian.

        Returns:
            The BdG Hamiltonian evaluated at ``mu``.
        """
        return self.build(mu)


def bdg_block_batched(
    mu_batch: torch.Tensor,
    n_sites: int,
    hopping: float = 1.0,
    pairing: float = 0.5,
) -> torch.Tensor:
    """Assemble the ``2N x 2N`` BdG Hamiltonian ``H(mu)`` for a batch of ``mu``.

    The Torch, batch-vectorised counterpart of
    :meth:`KitaevChainHamiltonian.build`, and the Nambu-basis analogue of
    :func:`kitaev.analytical.chiral_block_batched`. Only the diagonal
    depends on ``mu_batch`` (``-mu`` in the particle sector, ``+mu`` in the
    hole sector); the hopping and pairing blocks are constant. The result
    matches ``KitaevChainHamiltonian(n_sites, hopping, pairing).build(mu)``
    to machine precision.

    Used to attach a Rayleigh-quotient energy to a bare-eigenvector model
    via :class:`kitaev.models.RayleighEnergyAdapter`; it does not replace
    the ``(H_base, H_mu_diag, Xi)`` triple that ``UnifiedTrainer`` and the
    Nambu-basis losses build for themselves.

    Args:
        mu_batch: Chemical-potential values, shape ``(batch_size, 1)`` or
            ``(batch_size,)``.
        n_sites: Number of physical lattice sites, ``N``. The returned
            matrix is ``2N x 2N``.
        hopping: Nearest-neighbour hopping amplitude, ``t``.
        pairing: P-wave pairing amplitude, ``delta``.

    Returns:
        A tensor of shape ``(batch_size, 2 * n_sites, 2 * n_sites)``, on the
        same device and dtype as ``mu_batch``.
    """
    mu = mu_batch.reshape(-1)
    dim = 2 * n_sites
    device = mu.device
    block = mu.new_zeros(mu.shape[0], dim, dim)

    site = torch.arange(n_sites - 1, device=device)
    # Ordinary hopping: particle sector (-t), hole-sector mirror (+t).
    block[:, site, site + 1] = -hopping
    block[:, site + 1, site] = -hopping
    block[:, n_sites + site, n_sites + site + 1] = hopping
    block[:, n_sites + site + 1, n_sites + site] = hopping
    # P-wave pairing: antisymmetric across the two orientations of each bond.
    block[:, site, n_sites + site + 1] = pairing
    block[:, n_sites + site + 1, site] = pairing
    block[:, site + 1, n_sites + site] = -pairing
    block[:, n_sites + site, site + 1] = -pairing

    # On-site chemical potential.
    diag = torch.arange(n_sites, device=device)
    block[:, diag, diag] = -mu.unsqueeze(-1)
    block[:, n_sites + diag, n_sites + diag] = mu.unsqueeze(-1)

    return block
