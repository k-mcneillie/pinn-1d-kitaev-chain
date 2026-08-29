"""Tests for kitaev.visualisation.report.save_run_figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from kitaev.analytical import KitaevChainHamiltonian
from kitaev.models import ChiralToBdGAdapter, SirenPINNChiral
from kitaev.training.utils import TrainingHistory
from kitaev.visualisation.report import rerender_wavefunctions, save_run_figures

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_SITES = 6


def _history(*, with_probe: bool) -> TrainingHistory:
    history = TrainingHistory()
    for epoch in range(1, 9):
        history.record("train_loss", 1.0 / epoch)
        history.record("val_loss", 1.1 / epoch)
        history.record("train_fsm", 0.6 / epoch)
        history.record("train_var", 0.4 / epoch)
    if with_probe:
        for call in range(1, 4):
            history.record("probe_epoch", float(call * 2))
            for key in (
                "probe_e_mae_topological",
                "probe_e_mae_trivial",
                "probe_edge_mae",
                "probe_subspace_infidelity",
                "probe_subspace_infidelity_max",
            ):
                history.record(key, 0.1 / call)
    return history


def _adapter() -> ChiralToBdGAdapter:
    model = SirenPINNChiral(n_sites=N_SITES, hidden_features=8, hidden_layers=1)
    return ChiralToBdGAdapter(model, hopping=1.0, pairing=0.5)


def test_save_run_figures_writes_the_full_set(tmp_path: Path) -> None:
    hamiltonian = KitaevChainHamiltonian(n_sites=N_SITES, hopping=1.0, pairing=0.5)

    paths = save_run_figures(
        adapter=_adapter(),
        history=_history(with_probe=True),
        hamiltonian=hamiltonian,
        mu_grid=np.linspace(-4, 4, 24),
        out_dir=tmp_path / "figs",
        model_label="chiral PINN",
        component_keys=("fsm", "var"),
        split_epoch=4,
        structural_fold=True,
    )

    assert set(paths) == {
        "loss_history",
        "probe_history",
        "energy_sweep",
        "eigenvector_agreement",
        "wavefunctions",
        "mu_reflection",
    }
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0
    plt.close("all")


def test_save_run_figures_skips_probe_history_without_probe_series(
    tmp_path: Path,
) -> None:
    hamiltonian = KitaevChainHamiltonian(n_sites=N_SITES, hopping=1.0, pairing=0.5)

    paths = save_run_figures(
        adapter=_adapter(),
        history=_history(with_probe=False),
        hamiltonian=hamiltonian,
        mu_grid=np.linspace(0.05, 4, 24),  # folded half-domain grid
        out_dir=tmp_path / "figs",
        model_label="chiral PINN",
        component_keys=("fsm", "var"),
    )

    assert "probe_history" not in paths
    assert (tmp_path / "figs" / "energy_sweep.png").exists()
    plt.close("all")


def test_rerender_wavefunctions_writes_only_the_density_figure(tmp_path: Path) -> None:
    hamiltonian = KitaevChainHamiltonian(n_sites=N_SITES, hopping=1.0, pairing=0.5)

    path = rerender_wavefunctions(
        adapter=_adapter(),
        hamiltonian=hamiltonian,
        out_dir=tmp_path / "seed_0",
        two_sided=True,
    )

    assert path == tmp_path / "seed_0" / "wavefunctions.png"
    assert path.exists() and path.stat().st_size > 0
    assert list((tmp_path / "seed_0").iterdir()) == [path]
    plt.close("all")
