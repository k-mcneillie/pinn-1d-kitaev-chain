"""One-call rendering of the full standard figure set for a finished run.

:func:`save_run_figures` is what a training script calls once a model is
trained: give it the ``(E, psi)`` adapter, the run history and the exact
Hamiltonian, and it builds the sweeps and writes all six standard figures
(see :mod:`kitaev.visualisation.figures`) into a directory, returning the
path of each.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch

from kitaev.analytical import KitaevChainHamiltonian
from kitaev.training.utils import TrainingHistory

from .evaluation import (
    sweep_mu_reflection,
    sweep_spectrum,
    sweep_wavefunction_grid,
)
from .figures import (
    plot_eigenvector_agreement,
    plot_energy_sweep,
    plot_loss_history,
    plot_mu_reflection,
    plot_probe_history,
    plot_wavefunction_grid,
)

#: Default probe-mu columns for the density grid, chosen to straddle the
#: transition; the two-sided set mirrors across ``mu = 0``.
_TWO_SIDED_PROBE_MUS = (-3.4, -1.0, 0.0, 1.0, 3.4)
_FOLDED_PROBE_MUS = (0.3, 1.0, 1.9, 2.6, 3.4)


def save_run_figures(
    *,
    adapter: torch.nn.Module,
    history: TrainingHistory,
    hamiltonian: KitaevChainHamiltonian,
    mu_grid: npt.NDArray[np.float64],
    out_dir: str | Path,
    model_label: str,
    component_keys: Sequence[str],
    weight_key: str | None = None,
    split_epoch: int | None = None,
    floor_value: float | None = None,
    structural_fold: bool = False,
    probe_mus: Sequence[float] | None = None,
    device: torch.device | str = "cpu",
    dpi: int = 300,
) -> dict[str, Path]:
    """Render and save the six standard figures for one trained model.

    Args:
        adapter: A model or adapter returning ``(E_pred, psi_pred)`` with
            ``psi_pred`` a ``(batch, 2N)`` Nambu-basis vector -- i.e. a
            dual-head model directly, or a chiral / bare-eigenvector model
            wrapped in its adapter. Moved to ``device`` internally.
        history: The run's :class:`~kitaev.training.utils.TrainingHistory`
            (loss series plus the ``probe_*`` series if a
            :class:`~kitaev.training.probes.BdGEvaluationProbe` ran).
        hamiltonian: The exact Hamiltonian the model was trained against.
        mu_grid: The ``mu`` grid for the energy / eigenvector / reflection
            sweeps. A grid crossing ``0`` is treated as two-sided (both
            transitions shaded); a non-negative grid as the folded
            half-domain.
        out_dir: Directory to write the PNGs into; created if absent.
        model_label: Legend label for the predicted curves.
        component_keys: Loss-component names for
            :func:`~kitaev.visualisation.figures.plot_loss_history`.
        weight_key: Optional annealing-weight series name (e.g.
            ``"pin_wt"``), drawn on a secondary axis of the loss figure.
        split_epoch: AdamW -> L-BFGS hand-over epoch, marked on the loss
            and probe figures when given.
        floor_value: Analytic folded-spectrum floor for the loss figure's
            components panel (see
            :func:`kitaev.visualisation.evaluation.fsm_convergence_floor`);
            omitted for losses without such a floor.
        structural_fold: Whether the model enforces evenness in ``mu`` by
            construction (changes only the reflection figure's title).
        probe_mus: Override the density-grid ``mu`` columns; defaults to a
            transition-straddling set matched to the grid's sidedness.
        device: Device for the model forward passes.
        dpi: Resolution used when saving.

    Returns:
        A mapping from figure name to the path it was written to.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mu_grid = np.asarray(mu_grid, dtype=float)
    two_sided = bool(mu_grid.min() < 0)
    hopping = hamiltonian.hopping
    adapter = adapter.to(device)

    if probe_mus is None:
        probe_mus = _TWO_SIDED_PROBE_MUS if two_sided else _FOLDED_PROBE_MUS

    spectrum = sweep_spectrum(adapter, hamiltonian, mu_grid, device=device)
    wavefunctions = sweep_wavefunction_grid(
        adapter, hamiltonian, probe_mus, device=device
    )
    reflection = sweep_mu_reflection(
        adapter, device=device, mu_max=float(np.abs(mu_grid).max())
    )

    paths = {
        "loss_history": out_dir / "loss_history.png",
        "probe_history": out_dir / "probe_history.png",
        "energy_sweep": out_dir / "energy_sweep.png",
        "eigenvector_agreement": out_dir / "eigenvector_agreement.png",
        "wavefunctions": out_dir / "wavefunctions.png",
        "mu_reflection": out_dir / "mu_reflection.png",
    }

    plot_loss_history(
        history,
        component_keys=component_keys,
        weight_key=weight_key,
        split_epoch=split_epoch,
        floor_value=floor_value,
        save_path=paths["loss_history"],
        dpi=dpi,
    )
    if "probe_epoch" in history:
        plot_probe_history(
            history, split_epoch=split_epoch, save_path=paths["probe_history"], dpi=dpi
        )
    else:
        paths.pop("probe_history")
    plot_energy_sweep(
        spectrum,
        hopping=hopping,
        model_label=model_label,
        two_sided=two_sided,
        save_path=paths["energy_sweep"],
        dpi=dpi,
    )
    plot_eigenvector_agreement(
        spectrum,
        hopping=hopping,
        model_label=model_label,
        two_sided=two_sided,
        save_path=paths["eigenvector_agreement"],
        dpi=dpi,
    )
    plot_wavefunction_grid(
        wavefunctions, hopping=hopping, save_path=paths["wavefunctions"], dpi=dpi
    )
    plot_mu_reflection(
        reflection,
        hopping=hopping,
        structural_fold=structural_fold,
        save_path=paths["mu_reflection"],
        dpi=dpi,
    )

    plt.close("all")
    return paths
