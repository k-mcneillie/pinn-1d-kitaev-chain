"""A model's own physics residual, resolved per chemical potential.

Averaged into a scalar the residual is just the training loss. Kept per
``mu`` it is a difficulty map that shows where a trained model still
violates the physics, which is the interpretable artefact a self-adaptive
weighting scheme would also produce, obtained here without changing the
training loop.

Because the particular solution a run lands on is partly set by the seed,
especially in the topological phase, the map is taken over every available
seed and reported as a median with the full observed inter-seed range,
matching how :mod:`kitaev.xai.dispersion` treats the same models.

The residual is the folded-spectrum plus eigenvector-consistency quantity
that :class:`kitaev.training.loss.NambuFSMLoss` and
:class:`kitaev.training.loss.ChiralFSMLoss` sum, evaluated through
:func:`kitaev.training.loss.nambu_pointwise_residual` or
:func:`kitaev.training.loss.chiral_pointwise_residual` depending on the
basis the model works in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from kitaev.training.loss import chiral_pointwise_residual, nambu_pointwise_residual

_VALID_BASES = ("nambu", "chiral")


@dataclass
class ResidualField:
    """One model's per-``mu`` physics residual, summarised over seeds.

    Attributes:
        mu: The chemical-potential grid, shape ``(n_points,)``.
        residual_median: Median non-negative residual across seeds at each
            ``mu``.
        residual_min: Smallest residual across seeds at each ``mu``.
        residual_max: Largest residual across seeds at each ``mu``.
        n_seeds: Number of seeds the summary was taken over.
        basis: ``"nambu"`` or ``"chiral"``, the residual form used.
        label: A short human-readable name for the model.
        transition: The topological transition, ``2 * hopping``.
    """

    mu: npt.NDArray[np.float64]
    residual_median: npt.NDArray[np.float64]
    residual_min: npt.NDArray[np.float64]
    residual_max: npt.NDArray[np.float64]
    n_seeds: int
    basis: str
    label: str
    transition: float


def sweep_residual_field(
    models: Sequence[torch.nn.Module],
    *,
    basis: str,
    label: str,
    n_sites: int,
    hopping: float,
    pairing: float,
    mu_grid: npt.NDArray[np.float64],
    device: torch.device | str = "cpu",
) -> ResidualField:
    """Evaluate every seed's pointwise residual across ``mu_grid`` and summarise.

    Args:
        models: One trained model per seed, all of the same architecture.
            For ``basis="chiral"`` each must return the ``(u, v)`` singular
            pair. For ``basis="nambu"`` each must return the ``2N``
            eigenvector directly, so a dual-head model has to be wrapped
            (see :func:`kitaev.xai.loading.psi_only`). A single model may
            be passed as a one-element sequence, in which case the band
            has zero width.
        basis: ``"nambu"`` or ``"chiral"``.
        label: A short name for the model, used in figures.
        n_sites: Number of physical lattice sites, ``N``.
        hopping: Nearest-neighbour hopping amplitude, ``t``.
        pairing: P-wave pairing amplitude, ``delta``.
        mu_grid: 1D array of chemical-potential values.
        device: Device for the forward passes.

    Returns:
        The populated :class:`ResidualField`.

    Raises:
        ValueError: If ``basis`` is not one of ``"nambu"`` or ``"chiral"``,
            or if ``models`` is empty.
    """
    if basis not in _VALID_BASES:
        raise ValueError(f"basis must be one of {_VALID_BASES}, got {basis!r}")
    if not models:
        raise ValueError("need at least one model")

    mu_grid = np.asarray(mu_grid, dtype=float)
    mu_tensor = torch.tensor(mu_grid[:, None], dtype=torch.float32, device=device)
    residual_fn = (
        chiral_pointwise_residual if basis == "chiral" else nambu_pointwise_residual
    )

    curves = []
    for model in models:
        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():
            residual = residual_fn(
                model, mu_tensor, n_sites, hopping=hopping, pairing=pairing
            )
        curves.append(residual.detach().cpu().numpy())

    stack = np.stack(curves)  # (n_seeds, n_mu)
    return ResidualField(
        mu=mu_grid,
        residual_median=np.median(stack, axis=0),
        residual_min=stack.min(axis=0),
        residual_max=stack.max(axis=0),
        n_seeds=len(models),
        basis=basis,
        label=label,
        transition=2.0 * hopping,
    )
