"""Cross-seed dispersion of a model's predictions, as a function of ``mu``.

When the training objective is flat over a subspace of solutions, as the
folded-spectrum loss is over the two-dimensional near-zero Majorana
manifold in the topological phase, the particular solution a run lands on
is set by the random seed rather than by the physics. Repeating a model
over several seeds and measuring how much its output moves therefore maps
out exactly where its predictions are under-determined.

No cross-seed alignment is required: the predicted energy is compared
through its magnitude, invariant to the sign branch, and the eigenvector
through its per-site probability density, invariant to the overall sign.
Note that inside the topological phase the per-site density of a single
representative is *not* invariant to rotation within the degenerate
``+-lambda_1`` pair -- so the ``density_std`` a Nambu-basis model shows
there is a genuine measure of under-determination, but one confined to
that gauge degree of freedom rather than to the physical near-zero
subspace. The gauge-invariant companion -- the pair density ``rho_n/2``
stacked over seeds, which collapses for every model that found the right
subspace -- is
:func:`kitaev.visualisation.evaluation.sweep_seed_densities` and its
figures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from kitaev.analytical import KitaevChainHamiltonian


@dataclass
class SeedDispersion:
    """Standard deviation of a model's predictions across seeds, vs ``mu``.

    The energy and edge-weight spreads are gauge-invariant; the raw
    per-site ``density_std`` is not, inside the topological phase (see the
    module docstring).

    Attributes:
        mu: The chemical-potential grid, shape ``(n_points,)``.
        energy_std: Std across seeds of ``|E_pred(mu)|`` at each ``mu``.
        density_std_mean: Std across seeds of the raw per-site probability
            density, averaged over sites, at each ``mu``. Gauge-dependent
            for ``|mu| < 2t``.
        density_std_max: The same std, taken at the site where it is
            largest, at each ``mu``.
        edge_weight_std: Std across seeds of the combined edge weight at
            each ``mu``.
        n_seeds: Number of seeds the statistics were taken over.
        transition: The topological transition, ``2 * hopping``.
    """

    mu: npt.NDArray[np.float64]
    energy_std: npt.NDArray[np.float64]
    density_std_mean: npt.NDArray[np.float64]
    density_std_max: npt.NDArray[np.float64]
    edge_weight_std: npt.NDArray[np.float64]
    n_seeds: int
    transition: float


def _edge_sites(n_sites: int, n_edge_sites: int) -> npt.NDArray[np.int64]:
    """Site indices counted as edge sites at both ends of the chain."""
    return np.concatenate(
        [np.arange(n_edge_sites), np.arange(n_sites - n_edge_sites, n_sites)]
    )


def sweep_seed_dispersion(
    adapters: Sequence[torch.nn.Module],
    hamiltonian: KitaevChainHamiltonian,
    mu_grid: npt.NDArray[np.float64],
    *,
    device: torch.device | str = "cpu",
    n_edge_sites: int = 2,
) -> SeedDispersion:
    """Measure how far one model's predictions move across seeds.

    Each adapter is evaluated once over ``mu_grid``. The gauge-invariant
    observables are stacked over seeds and their per-``mu`` standard
    deviation is returned.

    Args:
        adapters: One trained ``(E, psi)`` model (or adapter) per seed,
            all of the same architecture. At least two are needed for a
            meaningful spread.
        hamiltonian: The Hamiltonian the models were trained against, used
            only for its site count and hopping.
        mu_grid: 1D array of chemical-potential values.
        device: Device for the forward passes.
        n_edge_sites: Sites counted at each end of the chain for the edge
            weight.

    Returns:
        The populated :class:`SeedDispersion`.

    Raises:
        ValueError: If fewer than two adapters are supplied.
    """
    if len(adapters) < 2:
        raise ValueError("need at least two seeds to measure dispersion")

    mu_grid = np.asarray(mu_grid, dtype=float)
    n_sites = hamiltonian.n_sites
    edge = _edge_sites(n_sites, n_edge_sites)
    mu_tensor = torch.tensor(mu_grid[:, None], dtype=torch.float32, device=device)

    energies = []
    densities = []
    edge_weights = []
    for adapter in adapters:
        adapter.to(device).eval()
        with torch.no_grad():
            e_pred, psi_pred = adapter(mu_tensor)
        density = (psi_pred.detach().cpu().numpy()) ** 2  # (n_mu, 2N)
        energies.append(np.abs(e_pred.detach().cpu().numpy().reshape(-1)))
        densities.append(density)
        edge_weights.append(
            density[:, :n_sites][:, edge].sum(axis=1)
            + density[:, n_sites:][:, edge].sum(axis=1)
        )

    energy_stack = np.stack(energies)  # (n_seed, n_mu)
    density_stack = np.stack(densities)  # (n_seed, n_mu, 2N)
    edge_stack = np.stack(edge_weights)  # (n_seed, n_mu)

    density_std = density_stack.std(axis=0)  # (n_mu, 2N)

    return SeedDispersion(
        mu=mu_grid,
        energy_std=energy_stack.std(axis=0),
        density_std_mean=density_std.mean(axis=1),
        density_std_max=density_std.max(axis=1),
        edge_weight_std=edge_stack.std(axis=0),
        n_seeds=len(adapters),
        transition=2.0 * hamiltonian.hopping,
    )
