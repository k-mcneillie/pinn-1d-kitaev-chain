"""Conditioning of the eigen-residual, as a function of the chemical potential.

The label-free losses pin the solution through a residual whose small
singular values are physical spectral gaps. When such a gap collapses the
residual stops distinguishing the wanted state from nearby ones, the
optimiser loses its signal, and the eigenvector becomes under-determined.
This module measures that directly.

At a fixed ``mu`` the eigen-residual is linear in the state, so its
Jacobian is the residual operator itself and its spectrum can be read off
by exact diagonalisation rather than autodiff. No trained model is needed.
Two bases are compared.

- ``"nambu"`` uses the full ``2N x 2N`` BdG Hamiltonian ``H(mu)``. The
  residual operator is ``H(mu) - E I`` with ``E`` the target eigenvalue, so
  its singular values are ``|E_k - E|`` over the whole spectrum.
- ``"chiral"`` uses the ``N x N`` bidiagonal block ``h(mu)``. The relevant
  gaps are ``|sigma_k - sigma_1|`` between the singular values, with
  ``sigma_1`` the smallest.

In the topological phase the Nambu gap includes the exponentially small
Majorana splitting, whereas the chiral gap to the next singular value stays
of order ``t``. The condition number ``sigma_max / sigma_min`` makes the
contrast visible across the transition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from kitaev.analytical import KitaevChainHamiltonian, chiral_block

_VALID_BASES = ("nambu", "chiral")


@dataclass
class ConditioningSweep:
    """Residual-operator spectrum across a ``mu`` grid, for one basis.

    Attributes:
        mu: The chemical-potential grid swept over, shape ``(n_points,)``.
        sigma_min: The smallest non-zero gap in the residual operator at
            each ``mu`` (the spectral gap the loss relies on).
        sigma_max: The largest gap at each ``mu``.
        condition_number: ``sigma_max / sigma_min`` at each ``mu``.
        basis: Either ``"nambu"`` or ``"chiral"``.
        transition: The topological transition, ``2 * hopping``.
    """

    mu: npt.NDArray[np.float64]
    sigma_min: npt.NDArray[np.float64]
    sigma_max: npt.NDArray[np.float64]
    condition_number: npt.NDArray[np.float64]
    basis: str
    transition: float


def _nambu_gaps(
    mu: float, n_sites: int, hopping: float, pairing: float
) -> npt.NDArray[np.float64]:
    """Absolute distances ``|E_k - E|`` from every BdG level to the target."""
    hamiltonian = KitaevChainHamiltonian(
        n_sites=n_sites, hopping=hopping, pairing=pairing
    )
    eigenvalues = np.linalg.eigvalsh(hamiltonian.build(mu))
    target = eigenvalues[n_sites]  # lowest non-negative eigenvalue
    gaps: npt.NDArray[np.float64] = np.abs(eigenvalues - target)
    return gaps


def _chiral_gaps(
    mu: float, n_sites: int, hopping: float, pairing: float
) -> npt.NDArray[np.float64]:
    """Absolute distances ``|sigma_k - sigma_1|`` between the singular values."""
    singular_values = np.linalg.svd(
        chiral_block(mu, n_sites, hopping, pairing), compute_uv=False
    )
    gaps: npt.NDArray[np.float64] = np.abs(singular_values - singular_values.min())
    return gaps


def sweep_conditioning(
    *,
    basis: str,
    n_sites: int,
    hopping: float,
    pairing: float,
    mu_grid: npt.NDArray[np.float64],
    tol: float = 1e-9,
) -> ConditioningSweep:
    """Sweep the residual-operator spectrum over ``mu_grid`` for one basis.

    Args:
        basis: ``"nambu"`` for the full ``2N x 2N`` eigen-residual or
            ``"chiral"`` for the ``N x N`` singular-value residual.
        n_sites: Number of physical lattice sites, ``N``.
        hopping: Nearest-neighbour hopping amplitude, ``t``.
        pairing: P-wave pairing amplitude, ``delta``.
        mu_grid: 1D array of chemical-potential values.
        tol: Gaps below this are treated as the residual's own null
            direction and excluded from ``sigma_min``.

    Returns:
        The populated :class:`ConditioningSweep`.

    Raises:
        ValueError: If ``basis`` is not one of ``"nambu"`` or ``"chiral"``.
    """
    if basis not in _VALID_BASES:
        raise ValueError(f"basis must be one of {_VALID_BASES}, got {basis!r}")

    mu_grid = np.asarray(mu_grid, dtype=float)
    gap_fn = _nambu_gaps if basis == "nambu" else _chiral_gaps

    sigma_min = np.empty_like(mu_grid)
    sigma_max = np.empty_like(mu_grid)
    for i, mu in enumerate(mu_grid):
        gaps = gap_fn(float(mu), n_sites, hopping, pairing)
        non_zero = gaps[gaps > tol]
        sigma_min[i] = non_zero.min() if non_zero.size else float("nan")
        sigma_max[i] = gaps.max()

    return ConditioningSweep(
        mu=mu_grid,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        condition_number=sigma_max / sigma_min,
        basis=basis,
        transition=2.0 * hopping,
    )
