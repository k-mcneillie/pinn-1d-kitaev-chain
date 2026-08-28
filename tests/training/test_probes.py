"""Tests for the BdGEvaluationProbe physical-error callback."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kitaev.analytical import (
    KitaevChainHamiltonian,
    chiral_block,
    reconstruct_bdg_eigenvector,
)
from kitaev.models import (
    ChiralToBdGAdapter,
    RayleighEnergyAdapter,
    SirenPINNChiral,
)
from kitaev.training.probes import BdGEvaluationProbe
from kitaev.training.utils import TrainingHistory

N_SITES = 8
HOPPING = 1.0
PAIRING = 0.5

PROBE_KEYS = {
    "probe_epoch",
    "probe_e_mae",
    "probe_e_mae_topological",
    "probe_e_mae_trivial",
    "probe_edge_mae",
    "probe_subspace_infidelity",
    "probe_subspace_infidelity_max",
    "probe_psi_norm",
}


def _chiral_adapter(model: torch.nn.Module) -> ChiralToBdGAdapter:
    return ChiralToBdGAdapter(model, hopping=HOPPING, pairing=PAIRING)


class _ExactChiralModel(torch.nn.Module):
    """Returns the exact smallest singular pair ``(u, v)`` of ``h(mu)``."""

    def __init__(self, n_sites: int, hopping: float, pairing: float) -> None:
        super().__init__()
        self.n_sites = n_sites
        self._hopping = hopping
        self._pairing = pairing
        self.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left_vectors, right_vectors = [], []
        for mu in x.reshape(-1).tolist():
            h = chiral_block(mu, self.n_sites, self._hopping, self._pairing)
            left, singular, right_t = np.linalg.svd(h)
            k = int(np.argmin(singular))
            left_vectors.append(left[:, k])
            right_vectors.append(right_t[k, :])
        u = torch.tensor(np.array(left_vectors), dtype=x.dtype, device=x.device)
        v = torch.tensor(np.array(right_vectors), dtype=x.dtype, device=x.device)
        return u, v


class _ExactEPsiModel(torch.nn.Module):
    """A perfect dual-head-style model: returns ``(E, psi)`` directly."""

    def __init__(
        self, n_sites: int, hopping: float, pairing: float, *, scale: float = 1.0
    ) -> None:
        super().__init__()
        self.n_sites = n_sites
        self._hopping = hopping
        self._pairing = pairing
        self._scale = scale
        self.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        energies, states = [], []
        for mu in x.reshape(-1).tolist():
            h = chiral_block(mu, self.n_sites, self._hopping, self._pairing)
            left, singular, right_t = np.linalg.svd(h)
            k = int(np.argmin(singular))
            energies.append(singular[k])
            states.append(
                reconstruct_bdg_eigenvector(left[:, k], right_t[k, :], sign=1)
            )
        e = torch.tensor(np.array(energies), dtype=x.dtype, device=x.device)
        psi = self._scale * torch.tensor(
            np.array(states), dtype=x.dtype, device=x.device
        )
        return e.unsqueeze(-1), psi


class _ExactPsiOnlyModel(torch.nn.Module):
    """A bare-eigenvector model: returns only the exact lowest ``psi``."""

    def __init__(self, n_sites: int, hopping: float, pairing: float) -> None:
        super().__init__()
        self.n_sites = n_sites
        self._hopping = hopping
        self._pairing = pairing
        self.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = []
        for mu in x.reshape(-1).tolist():
            h = chiral_block(mu, self.n_sites, self._hopping, self._pairing)
            left, singular, right_t = np.linalg.svd(h)
            k = int(np.argmin(singular))
            states.append(
                reconstruct_bdg_eigenvector(left[:, k], right_t[k, :], sign=1)
            )
        return torch.tensor(np.array(states), dtype=x.dtype, device=x.device)


def _rayleigh_adapter(model: torch.nn.Module) -> RayleighEnergyAdapter:
    return RayleighEnergyAdapter(
        model, n_sites=N_SITES, hopping=HOPPING, pairing=PAIRING
    )


class _RecordingSession:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


def _probe(**overrides) -> BdGEvaluationProbe:
    kwargs = {
        "n_sites": N_SITES,
        "hopping": HOPPING,
        "pairing": PAIRING,
        "mu_grid": np.linspace(0.1, 4.0, 40),
        "every": 3,
    }
    kwargs.update(overrides)
    return BdGEvaluationProbe(**kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_probe_default_grid_spans_the_valid_half_domain() -> None:
    probe = BdGEvaluationProbe(n_sites=N_SITES, hopping=HOPPING, pairing=PAIRING)
    assert probe.mu_grid.shape == (200,)
    assert probe.mu_grid[0] == pytest.approx(0.05)
    assert probe.mu_grid[-1] == pytest.approx(4.0 * HOPPING)


def test_probe_caches_exact_references_at_construction() -> None:
    grid = np.linspace(0.1, 4.0, 25)
    probe = _probe(mu_grid=grid)
    ham = KitaevChainHamiltonian(n_sites=N_SITES, hopping=HOPPING, pairing=PAIRING)

    assert probe._e_exact.shape == (25,)
    assert probe._edge_exact.shape == (25,)
    assert probe._near_zero.shape == (25, 2 * N_SITES, 2)
    expected_e = np.linalg.eigvalsh(ham.build(float(grid[10])))[N_SITES]
    assert probe._e_exact[10] == pytest.approx(expected_e)


# ---------------------------------------------------------------------------
# Firing schedule
# ---------------------------------------------------------------------------


def test_probe_fires_on_first_epoch_then_every_interval() -> None:
    probe = _probe(every=3, adapt=_chiral_adapter)
    model = SirenPINNChiral(n_sites=N_SITES, hidden_features=8, hidden_layers=1)
    history = TrainingHistory()

    for epoch in range(1, 8):
        probe.on_epoch_end(epoch, model, history)

    # calls 1 (first), 3, 6 fire; 2, 4, 5, 7 are skipped.
    assert history["probe_epoch"] == [1.0, 3.0, 6.0]


def test_probe_records_every_metric_key_each_evaluation() -> None:
    probe = _probe(every=1, adapt=_chiral_adapter)
    model = SirenPINNChiral(n_sites=N_SITES, hidden_features=8, hidden_layers=1)
    history = TrainingHistory()

    probe.on_epoch_end(1, model, history)
    probe.on_epoch_end(2, model, history)

    for key in PROBE_KEYS:
        assert key in history
        assert len(history[key]) == 2


# ---------------------------------------------------------------------------
# Model-agnostic evaluation
# ---------------------------------------------------------------------------


def test_probe_scores_a_chiral_model_through_the_adapter() -> None:
    probe = _probe(mu_grid=np.linspace(2.3, 3.8, 15), every=1, adapt=_chiral_adapter)
    model = _ExactChiralModel(N_SITES, HOPPING, PAIRING)
    history = TrainingHistory()

    probe.on_epoch_end(1, model, history)

    assert history["probe_e_mae"][-1] == pytest.approx(0.0, abs=1e-5)
    assert history["probe_edge_mae"][-1] == pytest.approx(0.0, abs=1e-5)
    assert history["probe_subspace_infidelity"][-1] == pytest.approx(0.0, abs=1e-5)
    assert history["probe_psi_norm"][-1] == pytest.approx(1.0, abs=1e-5)


def test_probe_scores_a_direct_e_psi_model_without_an_adapter() -> None:
    probe = _probe(mu_grid=np.linspace(2.3, 3.8, 15), every=1)  # adapt=None
    model = _ExactEPsiModel(N_SITES, HOPPING, PAIRING)
    history = TrainingHistory()

    probe.on_epoch_end(1, model, history)

    assert history["probe_e_mae"][-1] == pytest.approx(0.0, abs=1e-5)
    assert history["probe_edge_mae"][-1] == pytest.approx(0.0, abs=1e-5)
    assert history["probe_subspace_infidelity"][-1] == pytest.approx(0.0, abs=1e-5)


def test_probe_scores_a_bare_psi_model_through_the_rayleigh_adapter() -> None:
    probe = _probe(mu_grid=np.linspace(2.3, 3.8, 15), every=1, adapt=_rayleigh_adapter)
    model = _ExactPsiOnlyModel(N_SITES, HOPPING, PAIRING)
    history = TrainingHistory()

    probe.on_epoch_end(1, model, history)

    assert history["probe_e_mae"][-1] == pytest.approx(0.0, abs=1e-5)
    assert history["probe_edge_mae"][-1] == pytest.approx(0.0, abs=1e-5)
    assert history["probe_subspace_infidelity"][-1] == pytest.approx(0.0, abs=1e-5)
    assert history["probe_psi_norm"][-1] == pytest.approx(1.0, abs=1e-5)


def test_probe_normalises_psi_but_reports_the_raw_norm() -> None:
    probe = _probe(mu_grid=np.linspace(2.3, 3.8, 12), every=1)
    model = _ExactEPsiModel(N_SITES, HOPPING, PAIRING, scale=2.0)
    history = TrainingHistory()

    probe.on_epoch_end(1, model, history)

    # Raw norm is 2, but the density / subspace metrics use the normalised
    # direction and are still essentially exact.
    assert history["probe_psi_norm"][-1] == pytest.approx(2.0, abs=1e-5)
    assert history["probe_edge_mae"][-1] == pytest.approx(0.0, abs=1e-5)
    assert history["probe_subspace_infidelity"][-1] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Side effects
# ---------------------------------------------------------------------------


def test_probe_writes_to_session_when_supplied() -> None:
    session = _RecordingSession()
    probe = _probe(every=1, session=session, adapt=_chiral_adapter)
    model = SirenPINNChiral(n_sites=N_SITES, hidden_features=8, hidden_layers=1)

    probe.on_epoch_end(1, model, TrainingHistory())

    assert len(session.messages) == 1
    assert "probe" in session.messages[0]
    assert "E MAE" in session.messages[0]


def test_probe_restores_the_models_training_flag() -> None:
    # adapt wraps the model in a fresh module whose .eval() also touches the
    # shared submodule; the probe must restore the raw model's flag.
    probe = _probe(every=1, adapt=_chiral_adapter)
    model = SirenPINNChiral(n_sites=N_SITES, hidden_features=8, hidden_layers=1)

    model.train()
    probe.on_epoch_end(1, model, TrainingHistory())
    assert model.training is True

    model.eval()
    probe.on_epoch_end(2, model, TrainingHistory())
    assert model.training is False


def test_probe_reports_nan_when_a_phase_has_no_grid_points() -> None:
    probe = _probe(mu_grid=np.linspace(2.5, 4.0, 20), every=1, adapt=_chiral_adapter)
    model = SirenPINNChiral(n_sites=N_SITES, hidden_features=8, hidden_layers=1)
    history = TrainingHistory()

    probe.on_epoch_end(1, model, history)

    assert math.isnan(history["probe_e_mae_topological"][-1])
    assert not math.isnan(history["probe_e_mae_trivial"][-1])


def test_probe_phase_split_uses_absolute_mu() -> None:
    # A wholly negative, wholly trivial grid (|mu| in [2.5, 4.0]): the split
    # is on |mu| < 2t, so every point is trivial and the topological mean is
    # NaN. A naive ``mu < 2t`` split would misclassify all of these as
    # topological.
    probe = _probe(mu_grid=np.linspace(-4.0, -2.5, 20), every=1)
    model = _ExactEPsiModel(N_SITES, HOPPING, PAIRING)
    history = TrainingHistory()

    probe.on_epoch_end(1, model, history)

    assert math.isnan(history["probe_e_mae_topological"][-1])
    assert not math.isnan(history["probe_e_mae_trivial"][-1])
    assert history["probe_e_mae_trivial"][-1] == pytest.approx(0.0, abs=1e-5)


if __name__ == "__main__":
    pytest.main()
