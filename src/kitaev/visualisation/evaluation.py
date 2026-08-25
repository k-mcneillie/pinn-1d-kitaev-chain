"""Post-training evaluation sweeps: comparing a trained dual-head model against
exact diagonalisation of the same Hamiltonian.

Kept separate from `plots.py` so the numerical comparison (this module) can be
tested, reused, and reasoned about independently of how it is visualised.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from kitaev.analytical import KitaevChainHamiltonian


@dataclass
class EnergyEdgeWeightSweep:
    """Exact-vs-model energy and combined edge weight across a mu sweep.

    Attributes:
        mu_sweep: The mu values swept over, shape ``(n_points,)``.
        energy_exact: Exact lowest non-negative eigenvalue at each mu.
        energy_pred: Model-predicted energy at each mu.
        edge_weight_exact: Exact combined (particle + hole) edge weight
            at each mu.
        edge_weight_pred: Model-predicted combined edge weight at each
            mu.
        n_edge_sites: Number of sites counted at each end of the chain
            when computing edge weight.
    """

    mu_sweep: npt.NDArray[np.float64]
    energy_exact: npt.NDArray[np.float64]
    energy_pred: npt.NDArray[np.float64]
    edge_weight_exact: npt.NDArray[np.float64]
    edge_weight_pred: npt.NDArray[np.float64]
    n_edge_sites: int


@dataclass
class WavefunctionSweep:
    """Exact-vs-model particle/hole probability density profiles at chosen mu values.

    Attributes:
        probe_mus: The mu values probed.
        sites: Physical site indices, shape ``(n_sites,)``.
        particle_exact: Exact particle-sector ``|psi_n|^2``, shape
            ``(len(probe_mus), n_sites)``.
        hole_exact: Exact hole-sector ``|psi_n|^2``, same shape.
        particle_pred: Model particle-sector ``|psi_n|^2``, same shape.
        hole_pred: Model hole-sector ``|psi_n|^2``, same shape.
    """

    probe_mus: Sequence[float]
    sites: npt.NDArray[np.int64]
    particle_exact: npt.NDArray[np.float64]
    hole_exact: npt.NDArray[np.float64]
    particle_pred: npt.NDArray[np.float64]
    hole_pred: npt.NDArray[np.float64]


def _edge_sites(n_sites: int, n_edge_sites: int) -> npt.NDArray[np.int64]:
    """Site indices counted as "edge" at both ends of the chain."""
    return np.concatenate(
        [np.arange(n_edge_sites), np.arange(n_sites - n_edge_sites, n_sites)]
    )


def sweep_energy_and_edge_weight(
    model: torch.nn.Module,
    hamiltonian: KitaevChainHamiltonian,
    mu_sweep: npt.NDArray[np.float64],
    *,
    device: torch.device | str = "cpu",
    n_edge_sites: int = 2,
) -> EnergyEdgeWeightSweep:
    """Sweeps a dual-head model and exact diagonalisation over the same mu values.

    For each mu, exact diagonalisation gives the lowest non-negative
    eigenvalue and its eigenvector, split into particle/hole sectors and
    summed over ``edge_sites`` in each sector to give the combined edge
    weight (see :class:`EnergyEdgeWeightSweep`). The model is evaluated
    once, batched, over the full sweep.

    Args:
        model: A trained dual-head model (e.g. ``SirenPINNDualHead``)
            that returns a ``(E_pred, psi_pred)`` tuple when called.
            Switched to eval mode and run under ``torch.no_grad()``.
        hamiltonian: The exact Hamiltonian to diagonalise at each mu.
        mu_sweep: 1D array of mu values to evaluate at.
        device: Device to run the model's forward pass on.
        n_edge_sites: Number of sites counted at each end of the chain
            for edge weight.

    Returns:
        The populated sweep result.
    """
    n_sites = hamiltonian.n_sites
    split_index = n_sites
    edge_sites = _edge_sites(n_sites, n_edge_sites)

    energy_exact = np.zeros_like(mu_sweep)
    edge_weight_exact = np.zeros_like(mu_sweep)
    for i, mu in enumerate(mu_sweep):
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.build(float(mu)))
        energy_exact[i] = eigenvalues[split_index]
        psi = eigenvectors[:, split_index]
        particle_prob = psi[:n_sites] ** 2
        hole_prob = psi[n_sites:] ** 2
        edge_weight_exact[i] = (
            particle_prob[edge_sites].sum() + hole_prob[edge_sites].sum()
        )

    model.eval()
    with torch.no_grad():
        mu_tensor = torch.tensor(mu_sweep[:, None], dtype=torch.float32, device=device)
        energy_pred_t, psi_pred_t = model(mu_tensor)
        energy_pred = energy_pred_t.cpu().numpy().flatten()
        prob_pred = (psi_pred_t**2).cpu().numpy()

    particle_prob_pred = prob_pred[:, :n_sites]
    hole_prob_pred = prob_pred[:, n_sites:]
    edge_weight_pred = particle_prob_pred[:, edge_sites].sum(axis=1) + hole_prob_pred[
        :, edge_sites
    ].sum(axis=1)

    return EnergyEdgeWeightSweep(
        mu_sweep=mu_sweep,
        energy_exact=energy_exact,
        energy_pred=energy_pred,
        edge_weight_exact=edge_weight_exact,
        edge_weight_pred=edge_weight_pred,
        n_edge_sites=n_edge_sites,
    )


def sweep_wavefunctions(
    model: torch.nn.Module,
    hamiltonian: KitaevChainHamiltonian,
    probe_mus: Sequence[float],
    *,
    device: torch.device | str = "cpu",
) -> WavefunctionSweep:
    """Compares particle/hole probability density profiles at chosen mu values.

    Args:
        model: A trained dual-head model, as in
            :func:`sweep_energy_and_edge_weight`.
        hamiltonian: The exact Hamiltonian to diagonalise at each probe
            mu.
        probe_mus: The mu values to probe.
        device: Device to run the model's forward pass on.

    Returns:
        The populated sweep result.
    """
    n_sites = hamiltonian.n_sites
    split_index = n_sites
    sites = np.arange(n_sites)

    particle_exact = np.zeros((len(probe_mus), n_sites))
    hole_exact = np.zeros((len(probe_mus), n_sites))
    for i, mu in enumerate(probe_mus):
        _, eigenvectors = np.linalg.eigh(hamiltonian.build(float(mu)))
        psi = eigenvectors[:, split_index]
        particle_exact[i] = psi[:n_sites] ** 2
        hole_exact[i] = psi[n_sites:] ** 2

    model.eval()
    with torch.no_grad():
        mu_tensor = torch.tensor(
            [[mu] for mu in probe_mus], dtype=torch.float32, device=device
        )
        _, psi_pred_t = model(mu_tensor)
        prob_pred = (psi_pred_t**2).cpu().numpy()

    return WavefunctionSweep(
        probe_mus=list(probe_mus),
        sites=sites,
        particle_exact=particle_exact,
        hole_exact=hole_exact,
        particle_pred=prob_pred[:, :n_sites],
        hole_pred=prob_pred[:, n_sites:],
    )
