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
        manifold_density: Optional gauge-invariant near-zero manifold
            density ``rho(n) = |psi_+(n)|^2 + |psi_-(n)|^2`` split into
            ``[particle, hole]`` halves, shape ``(len(probe_mus), 2,
            n_sites)``. Rows for trivial-phase mu values (where the
            ``+-lambda_1`` pair is not degenerate) are ``np.nan``. ``None``
            when the sweep did not compute it (see
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

    Extends :func:`sweep_wavefunctions` with the two pieces the standard
    density figure needs beyond a raw exact-vs-model overlay:

    - **Branch alignment.** A model whose ``+-E`` branch is only a gauge
      (no ``loss_pin``, no structural resolver) can settle on ``Xi psi``
      in the trivial phase -- particle and hole sectors swapped relative
      to ``eigh``'s column. For each trivial-phase mu, whichever of
      ``{psi, Xi psi}`` better overlaps the reference is kept before the
      split into sectors. Topological columns are left raw (a ``Xi`` flip
      is nearly a no-op on their density, and they only ever illustrate one
      arbitrary section of the degenerate manifold).
    - **Manifold density.** For topological mu values, the gauge-invariant
      ``rho(n) = |psi_+(n)|^2 + |psi_-(n)|^2`` from the two smallest-``|E|``
      eigenvectors, so the figure can show ``rho / 2`` -- the density a
      balanced energy eigenstate would have.

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
        psi_ref = eigenvectors[:, split_index]
        particle_exact[col] = psi_ref[:n_sites] ** 2
        hole_exact[col] = psi_ref[n_sites:] ** 2

        psi = psi_pred_all[col]
        psi_xi = np.concatenate([psi[n_sites:], psi[:n_sites]])
        if (
            branch_align
            and abs(mu) >= transition
            and abs(psi_xi @ psi_ref) > abs(psi @ psi_ref)
        ):
            psi = psi_xi
            branch.append("Xi-flip")
        else:
            branch.append("keep")
        particle_pred[col] = psi[:n_sites] ** 2
        hole_pred[col] = psi[n_sites:] ** 2

        if abs(mu) < transition:
            near = eigenvectors[:, np.argsort(np.abs(eigenvalues))[:2]]
            manifold_density[col, 0] = (near[:n_sites, :] ** 2).sum(axis=1)
            manifold_density[col, 1] = (near[n_sites:, :] ** 2).sum(axis=1)

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
