# kitaev/analytical/hamiltonian.py
#
# ==========================
# Import Packages
# ==========================
import numpy as np


# ==========================
# BdG Hamiltonian
# ==========================
class KitaevChainHamiltonian:
    """Builds the Bogoliubov-de Gennes (BdG) Hamiltonian for a 1D Kitaev chain.

    Conceptual overview
    --------------------
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
            + sum_n [ -t * c_{n+1}^dagger c_n + delta * c_n c_{n+1} + h.c. ]

    where c_n / c_n^dagger annihilate/create a spinless fermion on site n,
    mu is the chemical potential, t is the nearest-neighbour hopping
    amplitude, and delta is the p-wave pairing amplitude. The pairing term
    (delta * c_n c_{n+1}) is what makes this a superconducting rather than
    a purely tight-binding problem: it creates and annihilates pairs of
    particles rather than conserving particle number, so the Hamiltonian
    cannot be diagonalised in the ordinary particle-number basis.

    Why a doubled (Nambu/BdG) basis is needed
    ------------------------------------------
    Because the pairing term mixes creation and annihilation operators,
    the natural basis for diagonalisation is the Nambu spinor basis, which
    treats particle and hole degrees of freedom on an equal footing:

        Psi = (c_1, ..., c_N, c_1^dagger, ..., c_N^dagger)^T

    Writing H in this doubled basis produces a 2N x 2N matrix with a
    built-in redundancy: for every eigenvalue +E there is a partner
    eigenvalue -E (particle-hole symmetry), and the corresponding
    eigenvectors are related by swapping the particle and hole blocks.
    This redundancy is a mathematical artefact of doubling the basis, not
    extra physics, but it is precisely this structure that allows
    zero-energy (E = 0) solutions to be interpreted as Majorana modes:
    a state pinned to E = 0 is, in this basis, "its own particle-hole
    partner".

    Matrix structure
    -----------------
    In the (particle-block, hole-block) layout used below, the 2N x 2N
    matrix is built from four N x N blocks:

        H = [[ A,  B ],
             [ B^T, -A ]]

    - **A (top-left, sites [0:N])**: the particle-sector block. Its
      diagonal encodes the on-site chemical potential (-mu per site) and
      its off-diagonal encodes ordinary nearest-neighbour hopping (-t).
    - **-A (bottom-right, sites [N:2N])**: the hole-sector block. It is
      the negative of the particle block, reflecting particle-hole
      symmetry: a hole moving forward is equivalent to a particle moving
      backward with opposite energy.
    - **B (off-diagonal blocks, particle-hole coupling)**: encodes the
      p-wave pairing amplitude, delta. This is what couples the particle
      and hole sectors together and is ultimately responsible for the
      topological phase transition and the emergence of Majorana edge
      modes as mu is tuned through +/-2t.

    Parameters
    ----------
    n_sites : int, default 20
        Number of physical lattice sites, N, in the chain. The resulting
        Hamiltonian has shape (2*N, 2*N) due to the doubled BdG basis.
    hopping : float, default 1.0
        Nearest-neighbour hopping amplitude, t, in the particle sector
        (and correspondingly -t in the hole sector).
    pairing : float, default 0.5
        p-wave pairing amplitude, delta, coupling neighbouring sites in
        the particle-hole off-diagonal blocks.

    Attributes
    ----------
    n_sites, hopping, pairing : as above.
    dim : int
        Dimension of the BdG matrix, equal to 2 * n_sites.

    Notes
    -----
    The topological phase transition of this model occurs at
    |mu| = 2 * hopping. For |mu| < 2t the chain is in the topologically
    non-trivial phase and (for open boundary conditions) hosts a pair of
    near-zero-energy Majorana edge modes; for |mu| > 2t it is trivial.
    This mirrors, in a much simpler 1D setting, the topological
    superconductivity discussed for the 2D magnetic-impurity lattice in
    the dissertation work this project builds on.
    """

    def __init__(self, n_sites: int = 20, hopping: float = 1.0, pairing: float = 0.5):
        self.n_sites = n_sites
        self.hopping = hopping
        self.pairing = pairing
        self.dim = 2 * n_sites

    def build(self, mu: float) -> np.ndarray:
        """Assemble the 2N x 2N BdG Hamiltonian matrix at chemical potential mu.

        For each site n, the on-site chemical potential contributes -mu to
        the particle sector and +mu to the hole sector (the sign flip is
        the particle-hole symmetry mentioned above). For each nearest-
        neighbour bond (n, n+1):

        - the particle sector gets a hopping matrix element -t (and its
          hole-sector mirror +t);
        - the pairing amplitude delta contributes off-diagonal terms
          coupling a particle on one site to a hole on the neighbouring
          site, with a sign structure ( +delta / -delta on the two
          off-diagonal blocks) that enforces the antisymmetry required of
          a fermionic pairing term (c_n c_{n+1} = -c_{n+1} c_n).

        Parameters
        ----------
        mu : float
            Chemical potential at which to evaluate the Hamiltonian. This
            is the control parameter the PINN surrogate is trained to map
            to the lowest non-negative eigenvalue and eigenvector.

        Returns
        -------
        numpy.ndarray
            Real, symmetric array of shape (2*n_sites, 2*n_sites)
            representing the BdG Hamiltonian at the given mu, ready for
            exact diagonalisation (e.g. via numpy.linalg.eigh) or for use
            as the ground-truth operator in a physics-residual loss.
        """
        N = self.n_sites  # noqa: N806
        t = self.hopping
        delta = self.pairing

        H = np.zeros((self.dim, self.dim))  # noqa: N806

        # On-site chemical potential: opposite sign in hole sector
        # (particle-hole symmetry).
        for i in range(N):
            H[i, i] = -mu
            H[N + i, N + i] = +mu

        # Nearest-neighbour hopping and p-wave pairing.
        for i in range(N - 1):
            # Ordinary hopping, particle sector (-t) and its hole-sector
            # mirror (+t).
            H[i, i + 1] = H[i + 1, i] = -t
            H[N + i, N + i + 1] = H[N + i + 1, N + i] = +t

            # p-wave pairing: couples a particle on site i to a hole on
            # site i+1 and vice versa, with the antisymmetric sign
            # pattern required of fermionic pairing.
            H[i, N + i + 1] = +delta
            H[N + i + 1, i] = +delta
            H[i + 1, N + i] = -delta
            H[N + i, i + 1] = -delta

        return H

    def __call__(self, mu: float) -> np.ndarray:
        """Shorthand for :meth:`build`, so the instance can be called directly."""
        return self.build(mu)
