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
        particle_exact: Reference particle-sector density, shape
            ``(len(probe_mus), n_sites)``. Trivial phase (``|mu| >= 2t``):
            ``|psi_n|^2`` of the single lowest non-negative ``eigh``
            eigenvector. Topological phase (``|mu| < 2t``): the
            gauge-invariant pair density ``rho/2``, since a single
            eigenvector of the near-degenerate ``+-lambda_1`` pair is an
            arbitrary member of the doublet.
        hole_exact: Reference hole-sector density, same convention.
        particle_pred: Model particle-sector density, same convention (the
            projector diagonal of ``span{psi, Xi psi}`` halved, in the
            topological phase).
        hole_pred: Model hole-sector density, same convention.
        manifold_density: Optional gauge-invariant near-zero manifold
            density ``rho(n) = |psi_+(n)|^2 + |psi_-(n)|^2`` (un-halved)
            split into ``[particle, hole]`` halves, shape
            ``(len(probe_mus), 2, n_sites)``. Rows for trivial-phase mu
            values (where the ``+-lambda_1`` pair is not degenerate) are
            ``np.nan``. ``None`` when the sweep did not compute it (see
            :func:`sweep_wavefunction_grid`).
        branch: Optional per-mu record of whether the raw model eigenvector
            was kept (``"keep"``) or particle/hole-swapped (``"Xi-flip"``)
            to align its branch with the reference, one entry per probe mu.
            ``None`` when no branch alignment was applied.
    """

    probe_mus: Sequence[float]
    sites: npt.NDArray[np.int64]
    particle_exact: npt.NDArray[np.float64]
    hole_exact: npt.NDArray[np.float64]
    particle_pred: npt.NDArray[np.float64]
    hole_pred: npt.NDArray[np.float64]
    manifold_density: npt.NDArray[np.float64] | None = None
    branch: list[str] | None = None


@dataclass
class SpectralSweep:
    """Exact-vs-model spectrum, edge weight and eigenvector fidelity over a mu grid.

    The data behind the project's standard "energy sweep" and "eigenvector
    agreement" figures: one exact-diagonalisation pass and one batched
    model forward over the same grid.

    Attributes:
        mu: The mu grid swept over, shape ``(n_points,)``.
        energy_exact: Exact lowest non-negative eigenvalue at each mu.
        energy_pred_signed: Model energy at each mu, sign as returned by the
            adapter (a model without a resolved ``+-E`` branch may return
            either sign).
        energy_pred: ``|energy_pred_signed|`` -- ``E`` and ``-E`` are the
            same physical state, so this is what the error is measured on.
        abs_error: ``|energy_pred - energy_exact|`` at each mu.
        edge_weight_exact: Exact combined (particle + hole) weight on the
            outermost ``n_edge_sites`` sites of each end.
        edge_weight_pred: Model combined edge weight, from the unit-norm
            predicted eigenvector.
        subspace_fidelity: ``||P psi_pred||`` at each mu, where ``P``
            projects onto the span of the two exact eigenvectors of
            smallest ``|E|`` -- the eigenvector-accuracy measure that stays
            well defined where the ``+-lambda_1`` pair is degenerate.
        transition: The topological transition, ``2 * hopping``.
        n_edge_sites: Sites counted at each end for the edge weight.
    """

    mu: npt.NDArray[np.float64]
    energy_exact: npt.NDArray[np.float64]
    energy_pred_signed: npt.NDArray[np.float64]
    energy_pred: npt.NDArray[np.float64]
    abs_error: npt.NDArray[np.float64]
    edge_weight_exact: npt.NDArray[np.float64]
    edge_weight_pred: npt.NDArray[np.float64]
    subspace_fidelity: npt.NDArray[np.float64]
    transition: float
    n_edge_sites: int


@dataclass
class LowSpectrumSweep:
    """The lowest few distinct exact levels ``sigma_k(mu)`` across a mu grid.

    The BdG spectrum is ``{+- sigma_k}``, so the ``|E_k|`` come in equal
    ``+-`` pairs. ``levels`` keeps only the distinct non-negative half:
    ``sigma_1`` collapses towards zero for ``|mu| < 2t`` (the near-zero
    edge mode) and ``sigma_2, sigma_3, ...`` are the bulk levels whose gap
    closes at ``|mu| = 2t``. This is the context for why the topological
    eigenvector is under-determined.

    Attributes:
        mu: The mu grid swept over, shape ``(n_points,)``.
        levels: The ``n_levels`` smallest ``sigma_k`` at each mu,
            ascending, shape ``(n_points, n_levels)``.
        transition: The topological transition, ``2 * hopping``.
    """

    mu: npt.NDArray[np.float64]
    levels: npt.NDArray[np.float64]
    transition: float


@dataclass
class ModelErrorBand:
    """One model's ``|E_pred - E_exact|(mu)`` reduced over its seeds.

    The per-mu curve is summarised as a median with an inter-quartile
    band. The full inter-seed range is dominated by single-seed,
    single-mu outliers on a log axis, so the 25th-to-75th percentile band
    is what the comparison figure shades. The scalar trivial / topological
    MAEs are kept per seed so a plot can scatter them over the summary
    bars.

    Attributes:
        label: Short model name for legends and bar ticks.
        mu: The shared mu grid, shape ``(n_points,)``.
        abs_error_median: Median over seeds of ``|E_pred - E_exact|`` at
            each mu.
        abs_error_lo: 25th percentile over seeds at each mu.
        abs_error_hi: 75th percentile over seeds at each mu.
        mae_trivial: Per-seed mean absolute energy error over ``|mu| >=
            2t``.
        mae_topological: Per-seed mean absolute energy error over ``|mu| <
            2t``.
        transition: The topological transition, ``2 * hopping``.
        n_seeds: Number of seeds the band was built from.
    """

    label: str
    mu: npt.NDArray[np.float64]
    abs_error_median: npt.NDArray[np.float64]
    abs_error_lo: npt.NDArray[np.float64]
    abs_error_hi: npt.NDArray[np.float64]
    mae_trivial: list[float]
    mae_topological: list[float]
    transition: float
    n_seeds: int


@dataclass
class MuReflectionSweep:
    """Model spectrum at ``+mu`` vs ``-mu``, to show evenness in ``mu``.

    Attributes:
        mu_half: The non-negative mu values probed, shape ``(n_points,)``.
        energy_pos: ``|E_pred(+mu)|`` at each ``mu_half`` value.
        energy_neg: ``|E_pred(-mu)|`` at each ``mu_half`` value.
        max_abs_diff: ``max |energy_pos - energy_neg|`` -- zero when
            evenness in ``mu`` is structural, small but non-zero when it is
            only learned.
    """

    mu_half: npt.NDArray[np.float64]
    energy_pos: npt.NDArray[np.float64]
    energy_neg: npt.NDArray[np.float64]
    max_abs_diff: float


@dataclass
class SeedDensitySweep:
    """Per-seed site densities of one model over a dense ``mu`` grid, two ways.

    Built from every seed's checkpoint of a single model. It carries two
    parallel views of the same forward passes so a figure can show, side
    by side, what is a gauge artefact and what is physical:

    - the **raw** per-sector density ``|psi^p_n|^2`` / ``|psi^h_n|^2`` of
      each seed's unit-normalised prediction. Inside the topological phase
      (``|mu| < 2t``) the ``+-lambda_1`` pair is degenerate and the
      folded-spectrum objective is flat over it, so which representative a
      seed lands on -- and hence this raw split -- is a gauge choice. The
      seeds fan apart here even when every one of them has solved the
      physics.
    - the **gauge-invariant** pair density ``rho_n / 2`` of each seed
      (:func:`predicted_pair_density`), the diagonal of the projector onto
      ``span{psi, Xi psi}``. Invariant to any rotation within the
      degenerate pair, so it collapses onto the exact curve for every
      model that has found the right near-zero subspace, Nambu-basis
      models included.

    Both views are normalised the same way: each sector array and its
    reference sum to ``~1`` when the two sectors are added.

    Attributes:
        mu: The chemical-potential grid, shape ``(n_mu,)``.
        sites: Physical site indices, shape ``(n_sites,)``.
        transition: The topological transition, ``2 * hopping``.
        model_label: Name of the model these seeds belong to.
        n_seeds: Number of seeds stacked.
        raw_particle: Raw particle-sector density per seed, shape
            ``(n_seeds, n_mu, n_sites)``.
        raw_hole: Raw hole-sector counterpart, same shape.
        raw_particle_exact: Reference raw particle density, shape
            ``(n_mu, n_sites)`` -- the single lowest non-negative ``eigh``
            eigenvector's density (an arbitrary member of the pair inside
            the topological phase; shown only to make the gauge spread
            legible, not as a target there).
        raw_hole_exact: Hole counterpart, same shape.
        pair_particle: Gauge-invariant ``rho^p_n / 2`` per seed, shape
            ``(n_seeds, n_mu, n_sites)``.
        pair_hole: Hole counterpart, same shape.
        pair_particle_exact: Exact ``rho^p_n / 2``
            (:func:`near_zero_pair_density`), shape ``(n_mu, n_sites)``,
            well defined at every ``mu``.
        pair_hole_exact: Hole counterpart, same shape.
    """

    mu: npt.NDArray[np.float64]
    sites: npt.NDArray[np.int64]
    transition: float
    model_label: str
    n_seeds: int
    raw_particle: npt.NDArray[np.float64]
    raw_hole: npt.NDArray[np.float64]
    raw_particle_exact: npt.NDArray[np.float64]
    raw_hole_exact: npt.NDArray[np.float64]
    pair_particle: npt.NDArray[np.float64]
    pair_hole: npt.NDArray[np.float64]
    pair_particle_exact: npt.NDArray[np.float64]
    pair_hole_exact: npt.NDArray[np.float64]

    def raw_density_std(self) -> npt.NDArray[np.float64]:
        """Inter-seed std of the raw total site density, shape ``(n_mu, n_sites)``."""
        return (self.raw_particle + self.raw_hole).std(axis=0)

    def pair_density_std(self) -> npt.NDArray[np.float64]:
        """Inter-seed std of the gauge-invariant site density, ``(n_mu, n_sites)``."""
        return (self.pair_particle + self.pair_hole).std(axis=0)

    def edge_weight(
        self, which: str, *, n_edge_sites: int = 2, end: str = "both"
    ) -> npt.NDArray[np.float64]:
        """Per-seed edge weight vs ``mu``, shape ``(n_seeds, n_mu)``.

        ``which`` is ``"raw"`` or ``"pair"``; ``end`` is ``"both"``,
        ``"left"`` or ``"right"``. The weight is the total density on the
        outermost ``n_edge_sites`` sites of the chosen end(s). The
        ``"both"`` sum is nearly gauge-invariant on its own (a rotation
        within the Majorana pair moves weight from one end to the other,
        leaving the total put), so the ``"left"`` weight is the scalar
        that actually fans across seeds inside the topological phase.
        """
        n_sites = len(self.sites)
        if end == "both":
            cols = _edge_sites(n_sites, n_edge_sites)
        elif end == "left":
            cols = np.arange(n_edge_sites)
        elif end == "right":
            cols = np.arange(n_sites - n_edge_sites, n_sites)
        else:  # pragma: no cover - guard
            raise ValueError(f"end must be 'both', 'left' or 'right', got {end!r}")
        if which == "raw":
            p, h = self.raw_particle, self.raw_hole
        elif which == "pair":
            p, h = self.pair_particle, self.pair_hole
        else:  # pragma: no cover - guard
            raise ValueError(f"which must be 'raw' or 'pair', got {which!r}")
        return p[:, :, cols].sum(axis=2) + h[:, :, cols].sum(axis=2)


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


def near_zero_pair_density(
    eigenvalues: npt.NDArray[np.float64],
    eigenvectors: npt.NDArray[np.float64],
    n_sites: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Particle/hole projector diagonals of the 2D near-zero eigenspace.

    ``rho_sector(n) = sum_{k in {-lambda_1, +lambda_1}} |psi_k(n)|^2`` for
    that sector, from the two smallest-``|E|`` exact eigenvectors. Unlike
    either eigenvector on its own this is invariant to any rotation within
    the (near-)degenerate pair, so it is well defined even where ``eigh``
    hands back an arbitrary basis of the doublet -- which it routinely
    does inside the topological phase, where the ``+-lambda_1`` splitting
    falls below machine precision and a single column comes out an
    arbitrary, often one-sided, Majorana combination. Each returned array
    sums to ~1; the two together sum to ~2.

    This is the exact reference for the gauge-invariant pair density; the
    model counterpart is :func:`predicted_pair_density`. Both are reused
    by :mod:`kitaev.xai.density_fan` and by the four-model comparison, so
    there is a single implementation.
    """
    near = eigenvectors[:, np.argsort(np.abs(eigenvalues))[:2]]
    return (near[:n_sites, :] ** 2).sum(axis=1), (near[n_sites:, :] ** 2).sum(axis=1)


def predicted_pair_density(
    psi_pred: npt.NDArray[np.float64], n_sites: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Model counterpart of :func:`near_zero_pair_density`.

    The projector diagonal of ``span{psi, Xi psi}`` (``Xi`` the
    particle-hole swap), Gram-Schmidt orthonormalised. ``{psi, Xi psi}``
    spans the model's near-zero 2D subspace -- exactly for the chiral
    model where ``Xi psi`` is structural, to within its branch error for
    the Nambu-basis models -- so it is gauge-invariant in the same sense.
    Each returned array sums to ~1; the two together sum to ~2.
    """
    u1 = psi_pred / (np.linalg.norm(psi_pred) + 1e-30)
    u2 = np.concatenate([u1[n_sites:], u1[:n_sites]])
    u2 = u2 - (u1 @ u2) * u1
    norm = np.linalg.norm(u2)
    u2 = u2 / norm if norm > 1e-9 else np.zeros_like(u2)
    return u1[:n_sites] ** 2 + u2[:n_sites] ** 2, u1[n_sites:] ** 2 + u2[n_sites:] ** 2


def sweep_wavefunctions(
    model: torch.nn.Module,
    hamiltonian: KitaevChainHamiltonian,
    probe_mus: Sequence[float],
    *,
    device: torch.device | str = "cpu",
) -> WavefunctionSweep:
    """Compares particle/hole probability density profiles at chosen mu values.

    Outside the topological phase the reference is the single lowest
    non-negative ``eigh`` eigenvector and the prediction is its raw
    density. Inside it (``|mu| < 2t``) the ``+-lambda_1`` pair is
    (near-)degenerate, so a single eigenvector is an arbitrary member of
    the doublet; both sides are then the gauge-invariant pair density
    ``rho/2`` -- :func:`near_zero_pair_density` for the reference,
    :func:`predicted_pair_density` for the model -- which is the density
    a balanced energy eigenstate carries and is comparable between models.

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
    transition = 2.0 * hamiltonian.hopping

    model.eval()
    with torch.no_grad():
        mu_tensor = torch.tensor(
            [[mu] for mu in probe_mus], dtype=torch.float32, device=device
        )
        psi_pred = model(mu_tensor)[1].detach().cpu().numpy()

    particle_exact = np.zeros((len(probe_mus), n_sites))
    hole_exact = np.zeros((len(probe_mus), n_sites))
    particle_pred = np.zeros((len(probe_mus), n_sites))
    hole_pred = np.zeros((len(probe_mus), n_sites))
    for i, mu in enumerate(probe_mus):
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.build(float(mu)))
        if abs(mu) < transition:
            rho_p, rho_h = near_zero_pair_density(eigenvalues, eigenvectors, n_sites)
            particle_exact[i] = rho_p / 2.0
            hole_exact[i] = rho_h / 2.0
            pred_p, pred_h = predicted_pair_density(psi_pred[i], n_sites)
            particle_pred[i] = pred_p / 2.0
            hole_pred[i] = pred_h / 2.0
        else:
            psi = eigenvectors[:, split_index]
            particle_exact[i] = psi[:n_sites] ** 2
            hole_exact[i] = psi[n_sites:] ** 2
            particle_pred[i] = psi_pred[i, :n_sites] ** 2
            hole_pred[i] = psi_pred[i, n_sites:] ** 2

    return WavefunctionSweep(
        probe_mus=list(probe_mus),
        sites=sites,
        particle_exact=particle_exact,
        hole_exact=hole_exact,
        particle_pred=particle_pred,
        hole_pred=hole_pred,
    )


def sweep_seed_densities(
    models: Sequence[torch.nn.Module],
    hamiltonian: KitaevChainHamiltonian,
    mu_grid: npt.NDArray[np.float64],
    *,
    model_label: str = "model",
    device: torch.device | str = "cpu",
) -> SeedDensitySweep:
    """Stack one model's site densities over its seeds, raw and gauge-invariant.

    Each element of ``models`` is a trained ``(E, psi)`` model or adapter
    for the same architecture at a different seed. Every one is evaluated
    once over ``mu_grid``; the exact reference is diagonalised once. The
    result feeds the publication figures
    :func:`kitaev.visualisation.figures.plot_seed_density_dispersion_maps`,
    :func:`kitaev.visualisation.figures.plot_seed_density_slices` and
    :func:`kitaev.visualisation.figures.plot_seed_edge_weight_envelope` --
    the static counterparts of the cross-seed fan animation.

    Args:
        models: One trained model (or adapter) per seed, same
            architecture. At least one; two or more for a meaningful
            spread.
        hamiltonian: The Hamiltonian the models were trained against.
        mu_grid: 1D array of chemical-potential values.
        model_label: Name of the model, kept on the result for titles.
        device: Device for the forward passes.

    Returns:
        The populated :class:`SeedDensitySweep`.

    Raises:
        ValueError: If ``models`` is empty.
    """
    if len(models) == 0:
        raise ValueError("need at least one seed model")

    mu_grid = np.asarray(mu_grid, dtype=float)
    n_sites = hamiltonian.n_sites
    n_mu = mu_grid.size
    sites = np.arange(n_sites)
    transition = 2.0 * hamiltonian.hopping

    raw_particle_exact = np.zeros((n_mu, n_sites))
    raw_hole_exact = np.zeros((n_mu, n_sites))
    pair_particle_exact = np.zeros((n_mu, n_sites))
    pair_hole_exact = np.zeros((n_mu, n_sites))
    for i, mu in enumerate(mu_grid):
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.build(float(mu)))
        psi = eigenvectors[:, n_sites]
        raw_particle_exact[i] = psi[:n_sites] ** 2
        raw_hole_exact[i] = psi[n_sites:] ** 2
        rho_p, rho_h = near_zero_pair_density(eigenvalues, eigenvectors, n_sites)
        pair_particle_exact[i] = rho_p / 2.0
        pair_hole_exact[i] = rho_h / 2.0

    n_seeds = len(models)
    raw_particle = np.zeros((n_seeds, n_mu, n_sites))
    raw_hole = np.zeros((n_seeds, n_mu, n_sites))
    pair_particle = np.zeros((n_seeds, n_mu, n_sites))
    pair_hole = np.zeros((n_seeds, n_mu, n_sites))
    mu_tensor = torch.tensor(mu_grid[:, None], dtype=torch.float32, device=device)
    for s, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            psi_pred = model(mu_tensor)[1].detach().cpu().numpy()
        psi_pred = psi_pred / np.clip(
            np.linalg.norm(psi_pred, axis=1, keepdims=True), 1e-12, None
        )
        raw_particle[s] = psi_pred[:, :n_sites] ** 2
        raw_hole[s] = psi_pred[:, n_sites:] ** 2
        for i in range(n_mu):
            pred_p, pred_h = predicted_pair_density(psi_pred[i], n_sites)
            pair_particle[s, i] = pred_p / 2.0
            pair_hole[s, i] = pred_h / 2.0

    return SeedDensitySweep(
        mu=mu_grid,
        sites=sites,
        transition=transition,
        model_label=model_label,
        n_seeds=n_seeds,
        raw_particle=raw_particle,
        raw_hole=raw_hole,
        raw_particle_exact=raw_particle_exact,
        raw_hole_exact=raw_hole_exact,
        pair_particle=pair_particle,
        pair_hole=pair_hole,
        pair_particle_exact=pair_particle_exact,
        pair_hole_exact=pair_hole_exact,
    )


def sweep_spectrum(
    model: torch.nn.Module,
    hamiltonian: KitaevChainHamiltonian,
    mu_grid: npt.NDArray[np.float64],
    *,
    device: torch.device | str = "cpu",
    n_edge_sites: int = 2,
) -> SpectralSweep:
    """Sweep energy, edge weight and eigenvector fidelity over a mu grid.

    One exact-diagonalisation pass caches, per mu, the lowest non-negative
    eigenvalue, its edge weight, and the two eigenvectors of smallest
    ``|E|``; the model is then evaluated once, batched, over the whole
    grid. This is the data behind
    :func:`kitaev.visualisation.figures.plot_energy_sweep` and
    :func:`kitaev.visualisation.figures.plot_eigenvector_agreement`.

    Args:
        model: A trained model (or adapter) that returns an
            ``(E_pred, psi_pred)`` tuple, with ``psi_pred`` a
            ``(batch, 2N)`` Nambu-basis vector. Switched to eval mode and
            run under ``torch.no_grad()``.
        hamiltonian: The exact Hamiltonian to diagonalise at each mu.
        mu_grid: 1D array of mu values to evaluate at.
        device: Device for the model forward pass.
        n_edge_sites: Sites counted at each end of the chain for the edge
            weight.

    Returns:
        The populated :class:`SpectralSweep`.
    """
    mu_grid = np.asarray(mu_grid, dtype=float)
    n_sites = hamiltonian.n_sites
    split_index = n_sites
    edge_sites = _edge_sites(n_sites, n_edge_sites)

    energy_exact = np.zeros_like(mu_grid)
    edge_weight_exact = np.zeros_like(mu_grid)
    near_zero = np.zeros((mu_grid.size, 2 * n_sites, 2))
    for i, mu in enumerate(mu_grid):
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.build(float(mu)))
        energy_exact[i] = eigenvalues[split_index]
        psi = eigenvectors[:, split_index]
        edge_weight_exact[i] = (psi[:n_sites][edge_sites] ** 2).sum() + (
            psi[n_sites:][edge_sites] ** 2
        ).sum()
        near_zero[i] = eigenvectors[:, np.argsort(np.abs(eigenvalues))[:2]]

    model.eval()
    with torch.no_grad():
        mu_tensor = torch.tensor(mu_grid[:, None], dtype=torch.float32, device=device)
        e_pred_t, psi_pred_t = model(mu_tensor)
    energy_pred_signed = e_pred_t.detach().cpu().numpy().reshape(-1)
    psi_pred = psi_pred_t.detach().cpu().numpy()

    psi_norm = np.linalg.norm(psi_pred, axis=1, keepdims=True)
    psi_unit = psi_pred / np.clip(psi_norm, 1e-12, None)

    energy_pred = np.abs(energy_pred_signed)
    abs_error = np.abs(energy_pred - energy_exact)

    edge_weight_pred = (psi_unit[:, :n_sites][:, edge_sites] ** 2).sum(axis=1) + (
        psi_unit[:, n_sites:][:, edge_sites] ** 2
    ).sum(axis=1)

    projection = np.einsum("gij,gi->gj", near_zero, psi_unit)
    subspace_fidelity = np.linalg.norm(projection, axis=1)

    return SpectralSweep(
        mu=mu_grid,
        energy_exact=energy_exact,
        energy_pred_signed=energy_pred_signed,
        energy_pred=energy_pred,
        abs_error=abs_error,
        edge_weight_exact=edge_weight_exact,
        edge_weight_pred=edge_weight_pred,
        subspace_fidelity=subspace_fidelity,
        transition=2.0 * hamiltonian.hopping,
        n_edge_sites=n_edge_sites,
    )


def sweep_wavefunction_grid(
    model: torch.nn.Module,
    hamiltonian: KitaevChainHamiltonian,
    probe_mus: Sequence[float],
    *,
    device: torch.device | str = "cpu",
    branch_align: bool = True,
) -> WavefunctionSweep:
    """Probe wavefunctions at chosen mu values, with branch alignment and rho.

    Extends :func:`sweep_wavefunctions` with the record the standard
    density figure needs beyond a raw exact-vs-model overlay:

    - **Trivial phase (``|mu| >= 2t``).** Reference is the single lowest
      non-negative ``eigh`` eigenvector. A model whose ``+-E`` branch is
      only a gauge (no ``loss_pin``, no structural resolver) can settle on
      ``Xi psi`` here -- particle and hole sectors swapped relative to
      ``eigh``'s column -- so whichever of ``{psi, Xi psi}`` better
      overlaps the reference is kept before the split into sectors, and
      ``branch`` records which.
    - **Topological phase (``|mu| < 2t``).** The ``+-lambda_1`` pair is
      (near-)degenerate, so a single eigenvector is an arbitrary,
      frequently one-sided, member of the doublet. Both ``*_exact`` and
      ``*_pred`` are the gauge-invariant pair density ``rho/2`` instead
      (:func:`near_zero_pair_density` / :func:`predicted_pair_density`)
      -- the density a balanced energy eigenstate carries. ``branch`` is
      ``"keep"`` for these columns (no flip is applied). ``manifold_density``
      additionally carries the un-halved ``rho(n) = |psi_+(n)|^2 +
      |psi_-(n)|^2`` split into ``[particle, hole]`` halves.

    Args:
        model: A trained model (or adapter) returning ``(E_pred,
            psi_pred)``.
        hamiltonian: The exact Hamiltonian to diagonalise at each probe mu.
        probe_mus: The mu values to probe.
        device: Device for the model forward pass.
        branch_align: When ``True`` (default) apply the trivial-phase
            branch alignment described above; when ``False`` every column
            is left raw and ``branch`` is all ``"keep"``.

    Returns:
        A :class:`WavefunctionSweep` with ``manifold_density`` and
        ``branch`` populated.
    """
    n_sites = hamiltonian.n_sites
    split_index = n_sites
    sites = np.arange(n_sites)
    transition = 2.0 * hamiltonian.hopping

    particle_exact = np.zeros((len(probe_mus), n_sites))
    hole_exact = np.zeros((len(probe_mus), n_sites))
    particle_pred = np.zeros((len(probe_mus), n_sites))
    hole_pred = np.zeros((len(probe_mus), n_sites))
    manifold_density = np.full((len(probe_mus), 2, n_sites), np.nan)
    branch: list[str] = []

    model.eval()
    with torch.no_grad():
        mu_tensor = torch.tensor(
            [[mu] for mu in probe_mus], dtype=torch.float32, device=device
        )
        psi_pred_all = model(mu_tensor)[1].detach().cpu().numpy()

    for col, mu in enumerate(probe_mus):
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.build(float(mu)))
        psi = psi_pred_all[col]

        if abs(mu) < transition:
            rho_p, rho_h = near_zero_pair_density(eigenvalues, eigenvectors, n_sites)
            manifold_density[col, 0] = rho_p
            manifold_density[col, 1] = rho_h
            particle_exact[col] = rho_p / 2.0
            hole_exact[col] = rho_h / 2.0
            pred_p, pred_h = predicted_pair_density(psi, n_sites)
            particle_pred[col] = pred_p / 2.0
            hole_pred[col] = pred_h / 2.0
            branch.append("keep")
            continue

        psi_ref = eigenvectors[:, split_index]
        particle_exact[col] = psi_ref[:n_sites] ** 2
        hole_exact[col] = psi_ref[n_sites:] ** 2
        psi_xi = np.concatenate([psi[n_sites:], psi[:n_sites]])
        if branch_align and abs(psi_xi @ psi_ref) > abs(psi @ psi_ref):
            psi = psi_xi
            branch.append("Xi-flip")
        else:
            branch.append("keep")
        particle_pred[col] = psi[:n_sites] ** 2
        hole_pred[col] = psi[n_sites:] ** 2

    return WavefunctionSweep(
        probe_mus=list(probe_mus),
        sites=sites,
        particle_exact=particle_exact,
        hole_exact=hole_exact,
        particle_pred=particle_pred,
        hole_pred=hole_pred,
        manifold_density=manifold_density,
        branch=branch,
    )


def sweep_low_spectrum(
    hamiltonian: KitaevChainHamiltonian,
    mu_grid: npt.NDArray[np.float64],
    *,
    n_levels: int = 3,
) -> LowSpectrumSweep:
    """Exact-diagonalise over ``mu_grid`` and keep the smallest ``sigma_k``.

    Purely analytical, no model involved. The ``+-`` pairing of the BdG
    spectrum means the raw ``|E_k|`` are duplicated, so this returns the
    distinct non-negative half.

    Args:
        hamiltonian: The exact Hamiltonian to diagonalise at each mu.
        mu_grid: 1D array of mu values.
        n_levels: How many of the smallest distinct levels to keep.

    Returns:
        The populated :class:`LowSpectrumSweep`.
    """
    mu_grid = np.asarray(mu_grid, dtype=float)
    split = hamiltonian.n_sites
    levels = np.zeros((mu_grid.size, n_levels))
    for i, mu in enumerate(mu_grid):
        eigenvalues = np.linalg.eigvalsh(hamiltonian.build(float(mu)))
        levels[i] = eigenvalues[split : split + n_levels]
    return LowSpectrumSweep(
        mu=mu_grid, levels=levels, transition=2.0 * hamiltonian.hopping
    )


def build_model_error_band(
    label: str,
    sweeps: Sequence[SpectralSweep],
) -> ModelErrorBand:
    """Reduce one model's per-seed :class:`SpectralSweep` set to a band.

    Args:
        label: Short model name.
        sweeps: One :class:`SpectralSweep` per seed, all on the same mu
            grid. Must be non-empty.

    Returns:
        The populated :class:`ModelErrorBand`.

    Raises:
        ValueError: If ``sweeps`` is empty.
    """
    if not sweeps:
        raise ValueError("need at least one SpectralSweep to build a band")
    mu = sweeps[0].mu
    stack = np.stack([s.abs_error for s in sweeps])  # (n_seeds, n_mu)
    topological = np.abs(mu) < sweeps[0].transition
    lo, hi = np.percentile(stack, [25, 75], axis=0)
    return ModelErrorBand(
        label=label,
        mu=mu,
        abs_error_median=np.median(stack, axis=0),
        abs_error_lo=lo,
        abs_error_hi=hi,
        mae_trivial=[float(s.abs_error[~topological].mean()) for s in sweeps],
        mae_topological=[float(s.abs_error[topological].mean()) for s in sweeps],
        transition=sweeps[0].transition,
        n_seeds=len(sweeps),
    )


def fsm_convergence_floor(
    hamiltonian: KitaevChainHamiltonian,
    mu_samples: npt.NDArray[np.float64],
    *,
    factor: float = 1.0,
) -> float:
    """Analytic lower bound the folded-spectrum loss term approaches.

    ``||H psi||^2`` is minimised over the unit sphere by the lowest state,
    giving ``E_1(mu)^2``; averaged over the training mu distribution this
    is the value the ``fsm`` component flattens onto once the energy is
    right. The chiral loss sums ``||h v||^2 + ||h^T u||^2``, each with the
    same ``sigma_1(mu)^2 = E_1(mu)^2`` floor, so pass ``factor=2`` there.

    Args:
        hamiltonian: The exact Hamiltonian.
        mu_samples: A representative sample of the training mu
            distribution.
        factor: Multiplier on ``<E_1(mu)^2>`` (``2`` for the chiral loss,
            ``1`` for the Nambu folded-spectrum losses).

    Returns:
        ``factor * mean(E_1(mu)^2)`` over ``mu_samples``.
    """
    split_index = hamiltonian.n_sites
    e1 = np.array(
        [
            np.linalg.eigvalsh(hamiltonian.build(float(mu)))[split_index]
            for mu in np.asarray(mu_samples, dtype=float)
        ]
    )
    return float(factor * np.mean(e1**2))


def sweep_mu_reflection(
    model: torch.nn.Module,
    *,
    device: torch.device | str = "cpu",
    mu_max: float = 4.0,
    n_points: int = 300,
) -> MuReflectionSweep:
    """Compare the model's ``|E(+mu)|`` and ``|E(-mu)|`` over ``[0, mu_max]``.

    Args:
        model: A trained model (or adapter) returning ``(E_pred,
            psi_pred)``.
        device: Device for the model forward pass.
        mu_max: Upper end of the non-negative mu range probed.
        n_points: Number of points in ``[0, mu_max]``.

    Returns:
        The populated :class:`MuReflectionSweep`.
    """
    mu_half = np.linspace(0.0, mu_max, n_points)
    model.eval()
    with torch.no_grad():
        pos = torch.tensor(mu_half[:, None], dtype=torch.float32, device=device)
        energy_pos = np.abs(model(pos)[0].detach().cpu().numpy().reshape(-1))
        energy_neg = np.abs(model(-pos)[0].detach().cpu().numpy().reshape(-1))
    return MuReflectionSweep(
        mu_half=mu_half,
        energy_pos=energy_pos,
        energy_neg=energy_neg,
        max_abs_diff=float(np.abs(energy_pos - energy_neg).max()),
    )
