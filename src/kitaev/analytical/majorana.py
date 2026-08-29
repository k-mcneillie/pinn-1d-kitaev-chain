# kitaev/analytical/majorana.py
"""Majorana-basis (chiral) reduction of the Kitaev-chain BdG Hamiltonian.

The ``2N x 2N`` Bogoliubov-de Gennes (BdG) Hamiltonian built by
:class:`kitaev.analytical.KitaevChainHamiltonian` is real symmetric and, in
the Nambu basis ``Psi = (c_1..c_N, c_1^dagger..c_N^dagger)^T``, has the
particle-hole block structure ``[[A, B], [B^T, -A]]``. That representation
exposes only one of the chain's symmetries (particle-hole,
``Xi H Xi = -H``).

Rotating to Majorana operators

    a_n = c_n + c_n^dagger,        b_n = -i (c_n - c_n^dagger),

ordered by sublattice as ``gamma = (a_1..a_N, b_1..b_N)^T``, block-
diagonalises the problem. With the fixed unitary basis change

    Omega = (1 / sqrt(2)) * [[  I,  I ],
                             [ -iI, iI ]]

the transformed Hamiltonian is purely imaginary and antisymmetric, and its
sublattice blocks vanish:

    Omega H(mu) Omega^dagger = i * [[ 0,       h(mu) ],
                                    [ -h(mu)^T, 0     ]]

where ``h(mu)`` is a real ``N x N`` bidiagonal matrix carrying the entire
content of the ``2N``-dimensional spectral problem. This is a consequence of
the chiral symmetry of symmetry class BDI, to which the real Kitaev chain
belongs.

Consequences used elsewhere in the project:

- The singular values of ``h(mu)`` are exactly the non-negative eigenvalues
  of the full BdG Hamiltonian, so the ``+-E`` spectral pairing is structural
  rather than something a loss must learn.
- For a singular pair ``h v = lambda u``, ``h^T u = lambda v`` with unit
  ``u``, ``v``, the corresponding BdG eigenvector in the original ``c``-basis
  is ``psi = ((u + v) / 2, (u - v) / 2)`` (particle block, hole block), which
  is automatically unit-norm; its ``-E`` partner is the particle/hole block
  swap ``Xi psi``.
- A Majorana zero mode is a zero singular value of ``h(mu)``. Seeking a null
  vector ``v_n ~ z^n`` gives ``(t + delta) z^2 + mu z + (t - delta) = 0``; a
  root crosses ``|z| = 1`` exactly at ``|mu| = 2 t``, the topological
  transition, independent of ``delta``.
- ``h(-mu) = -D h(mu) D`` with ``D = diag((-1)^n)``, so the spectrum is even
  in ``mu`` and only ``mu >= 0`` needs to be solved.
"""

from __future__ import annotations

import numpy as np
import torch


def chiral_block(
    mu: float,
    n_sites: int = 20,
    hopping: float = 1.0,
    pairing: float = 0.5,
) -> np.ndarray:
    """Assemble the ``N x N`` chiral block ``h(mu)`` of the Kitaev chain.

    ``h(mu)`` is the off-diagonal sublattice block of the Majorana-basis
    Hamiltonian ``Omega H(mu) Omega^dagger = i [[0, h], [-h^T, 0]]`` (see the
    module docstring). It is bidiagonal:

    - diagonal:        ``h[n, n]   = -mu``
    - super-diagonal:  ``h[n, n+1] = -(hopping + pairing)``
    - sub-diagonal:    ``h[n+1, n] = -(hopping - pairing)``

    All entries scale linearly with the energy unit; the natural choice is
    ``hopping = t = 1`` with ``mu`` and ``pairing`` expressed in units of
    ``t``.

    Args:
        mu: Chemical potential at which to evaluate the block.
        n_sites: Number of physical lattice sites, ``N``. The returned
            matrix is ``N x N`` (not ``2N x 2N``).
        hopping: Nearest-neighbour hopping amplitude, ``t``.
        pairing: P-wave pairing amplitude, ``delta``.

    Returns:
        A real NumPy array of shape ``(n_sites, n_sites)``.
    """
    block = np.zeros((n_sites, n_sites), dtype=float)

    diagonal = np.arange(n_sites)
    block[diagonal, diagonal] = -mu

    upper = np.arange(n_sites - 1)
    block[upper, upper + 1] = -(hopping + pairing)
    block[upper + 1, upper] = -(hopping - pairing)

    return block


def chiral_block_batched(
    mu_batch: torch.Tensor,
    n_sites: int,
    hopping: float = 1.0,
    pairing: float = 0.5,
) -> torch.Tensor:
    """Assemble the chiral block ``h(mu)`` for a batch of ``mu`` values.

    The Torch, batch-vectorised counterpart of :func:`chiral_block`, used by
    the physics residual in :class:`kitaev.training.loss.ChiralFSMLoss` and
    by :class:`kitaev.models.ChiralToBdGAdapter`. Only the ``-mu`` diagonal
    depends on ``mu_batch``; the bidiagonal off-diagonals are constant.

    Args:
        mu_batch: Chemical-potential values, shape ``(batch_size, 1)`` or
            ``(batch_size,)``.
        n_sites: Number of physical lattice sites, ``N``.
        hopping: Nearest-neighbour hopping amplitude, ``t``.
        pairing: P-wave pairing amplitude, ``delta``.

    Returns:
        A tensor of shape ``(batch_size, n_sites, n_sites)``, on the same
        device and dtype as ``mu_batch``.
    """
    mu = mu_batch.reshape(-1)
    block = mu.new_zeros(mu.shape[0], n_sites, n_sites)

    diagonal = torch.arange(n_sites, device=mu.device)
    block[:, diagonal, diagonal] = -mu.unsqueeze(-1)

    upper = torch.arange(n_sites - 1, device=mu.device)
    block[:, upper, upper + 1] = -(hopping + pairing)
    block[:, upper + 1, upper] = -(hopping - pairing)

    return block


def chiral_block_matvec(
    mu_batch: torch.Tensor,
    vec: torch.Tensor,
    *,
    hopping: float = 1.0,
    pairing: float = 0.5,
    adjoint: bool = False,
) -> torch.Tensor:
    """Apply the chiral block ``h(mu)`` (or ``h(mu)^T``) without forming it.

    ``h(mu)`` is bidiagonal (see :func:`chiral_block`), so ``h(mu) @ vec`` is
    a three-term recurrence rather than a dense matrix-vector product. This
    is the training hot-path counterpart of :func:`chiral_block_batched`: it
    returns exactly

        torch.bmm(chiral_block_batched(mu_batch, N, hopping, pairing),
                  vec.unsqueeze(-1)).squeeze(-1)

    (or the same with ``.transpose(1, 2)`` when ``adjoint=True``), but at
    ``O(batch * N)`` cost and with no ``(batch, N, N)`` allocation or
    ``bmm``. On accelerators where many small batched matmul / scatter
    kernels dominate, that difference is most of the per-step cost of a
    :class:`kitaev.training.loss.ChiralFSMLoss` step. The dense
    :func:`chiral_block_batched` is kept for probes, analytics and tests.

    With diagonal ``-mu``, super-diagonal ``-(hopping + pairing)`` and
    sub-diagonal ``-(hopping - pairing)``::

        (h @ x)[n]   = -mu x[n] - (t + d) x[n+1] - (t - d) x[n-1]
        (h^T @ x)[n] = -mu x[n] - (t - d) x[n+1] - (t + d) x[n-1]

    with out-of-range terms dropped at the chain ends.

    Args:
        mu_batch: Chemical-potential values, shape ``(batch_size, 1)`` or
            ``(batch_size,)``.
        vec: Vectors to apply ``h`` to, shape ``(batch_size, n_sites)``.
        hopping: Nearest-neighbour hopping amplitude, ``t``.
        pairing: P-wave pairing amplitude, ``delta``.
        adjoint: Apply ``h(mu)^T`` instead of ``h(mu)``.

    Returns:
        ``h(mu) @ vec`` (or ``h(mu)^T @ vec``), shape
        ``(batch_size, n_sites)``, on the device and dtype of ``vec``.
    """
    mu = mu_batch.reshape(-1, 1)

    zero = vec.new_zeros(vec.shape[0], 1)
    shift_up = torch.cat([vec[:, 1:], zero], dim=1)  # position n <- vec[n + 1]
    shift_down = torch.cat([zero, vec[:, :-1]], dim=1)  # position n <- vec[n - 1]

    super_coeff = -(hopping + pairing)
    sub_coeff = -(hopping - pairing)
    if adjoint:
        super_coeff, sub_coeff = sub_coeff, super_coeff

    return -mu * vec + super_coeff * shift_up + sub_coeff * shift_down


def resolve_singular_branch(
    u: torch.Tensor,
    v: torch.Tensor,
    h_batch: torch.Tensor,
    *,
    tol: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Canonicalise a predicted singular pair onto the ``+lambda`` branch.

    The chiral folded-spectrum objective
    (:class:`kitaev.training.loss.ChiralFSMLoss`) is invariant under
    ``(u, v) -> (u, -v)``. That flip negates the Rayleigh quotient
    ``lambda_R = u^T h v`` and swaps the reconstructed BdG eigenvector
    ``psi = ((u + v) / 2, (u - v) / 2)`` with its particle-hole partner
    ``Xi psi`` (the particle and hole blocks exchange). A network trained on
    that loss is therefore free to pick either branch, and nothing stops the
    choice from flipping as a function of ``mu`` -- which shows up as a
    spurious particle/hole swap in the trivial phase, where ``lambda_R`` is
    safely non-zero.

    This routine removes that freedom: wherever ``lambda_R < -tol`` it flips
    ``v`` (and hence ``h v`` and ``lambda_R``) so that the returned
    ``lambda_R`` is non-negative. Points with ``|lambda_R| <= tol`` -- the
    near-zero modes of the topological phase, where ``psi`` and ``Xi psi``
    span the same degenerate eigenspace and the branch is physically
    ambiguous -- are left untouched, so the network's own smooth gauge is
    preserved there. The flip sign is detached from the graph, so applying
    this inside a loss does not change its gradient.

    Args:
        u: Left singular vectors, shape ``(batch_size, N)``.
        v: Right singular vectors, shape ``(batch_size, N)``.
        h_batch: The chiral blocks ``h(mu)``, shape
            ``(batch_size, N, N)`` (see :func:`chiral_block_batched`).
        tol: Half-width of the neighbourhood of zero within which the
            branch is treated as ambiguous and left as-is.

    Returns:
        ``(u, v, h_v, lambda_R)`` with ``v``, ``h_v`` and ``lambda_R``
        flipped as needed; ``lambda_R`` has shape ``(batch_size, 1)`` and is
        ``>= -tol`` (non-negative wherever the branch was well defined).
    """
    h_v = torch.bmm(h_batch, v.unsqueeze(-1)).squeeze(-1)
    lam = torch.sum(u * h_v, dim=1, keepdim=True)

    flip = torch.where(
        lam < -tol, torch.full_like(lam, -1.0), torch.ones_like(lam)
    ).detach()

    v = v * flip
    h_v = h_v * flip
    lam = lam * flip
    return u, v, h_v, lam


def majorana_basis_change(n_sites: int = 20) -> np.ndarray:
    """Build the fixed unitary ``Omega`` mapping the Nambu basis to Majoranas.

    ``Omega`` sends ``Psi = (c_1..c_N, c_1^dagger..c_N^dagger)^T`` to the
    sublattice-ordered Majorana vector ``gamma = (a_1..a_N, b_1..b_N)^T``
    with ``a_n = c_n + c_n^dagger`` and ``b_n = -i (c_n - c_n^dagger)``:

        Omega = (1 / sqrt(2)) * [[  I,  I ],
                                 [ -iI, iI ]]

    It is unitary (``Omega Omega^dagger = I``) and satisfies
    ``Omega H(mu) Omega^dagger = i [[0, h(mu)], [-h(mu)^T, 0]]`` with
    ``h(mu)`` from :func:`chiral_block`.

    Args:
        n_sites: Number of physical lattice sites, ``N``. The returned
            matrix is ``2N x 2N``.

    Returns:
        A complex NumPy array of shape ``(2 * n_sites, 2 * n_sites)``.
    """
    identity = np.eye(n_sites)
    top = np.hstack([identity, identity])
    bottom = np.hstack([-1j * identity, 1j * identity])
    omega: np.ndarray = np.vstack([top, bottom]) / np.sqrt(2.0)
    return omega


def reconstruct_bdg_eigenvector(
    u: np.ndarray,
    v: np.ndarray,
    *,
    sign: int = 1,
) -> np.ndarray:
    """Map a singular pair of ``h(mu)`` back to a BdG eigenvector.

    Given left/right singular vectors ``u``, ``v`` of :func:`chiral_block`
    (``h v = lambda u``, ``h^T u = lambda v``) with ``||u|| = ||v|| = 1``,
    the eigenvector of the full BdG Hamiltonian in the Nambu ``c``-basis is,
    for the ``+lambda`` branch,

        psi = ( (u + v) / 2 ,  (u - v) / 2 )

    laid out as (particle block, hole block). It is automatically unit-norm.
    The ``-lambda`` partner is the particle/hole block swap, obtained here
    with ``sign = -1``.

    Args:
        u: Left singular vector, shape ``(N,)``.
        v: Right singular vector, shape ``(N,)``.
        sign: ``+1`` for the ``+lambda`` eigenvector, ``-1`` for its
            particle-hole (``-lambda``) partner.

    Returns:
        A real NumPy array of shape ``(2 * N,)``.
    """
    particle = (u + sign * v) / 2.0
    hole = (u - sign * v) / 2.0
    return np.concatenate([particle, hole])
