"""Tests for kitaev.xai.report."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from kitaev.xai.conditioning import ConditioningSweep
from kitaev.xai.dispersion import SeedDispersion
from kitaev.xai.internalisation import KITAEV_PROFILES
from kitaev.xai.report import (
    XaiAnalysis,
    save_xai_report,
    shared_dispersion_ylim,
)
from kitaev.xai.residual_field import ResidualField

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _analysis() -> XaiAnalysis:
    mu = np.linspace(-4.0, 4.0, 16)
    profiles = list(KITAEV_PROFILES.values())
    return XaiAnalysis(
        profiles=profiles,
        errors={p.name: {"trivial": 1e-3, "topological": 1e-4} for p in profiles},
        dispersion={
            "chiral": SeedDispersion(
                mu=mu,
                energy_std=np.full(16, 1e-3),
                density_std_mean=np.full(16, 1e-3),
                density_std_max=np.full(16, 2e-3),
                edge_weight_std=np.full(16, 1e-3),
                n_seeds=5,
                transition=2.0,
            )
        },
        conditioning={
            "nambu": ConditioningSweep(
                mu=mu,
                sigma_min=np.full(16, 1e-3),
                sigma_max=np.full(16, 2.0),
                condition_number=np.full(16, 2_000.0),
                basis="nambu",
                transition=2.0,
            )
        },
        residual_fields={
            "chiral": ResidualField(
                mu=mu,
                residual_median=np.abs(mu) + 1e-6,
                residual_min=np.abs(mu) * 0.8 + 1e-6,
                residual_max=np.abs(mu) * 1.2 + 1e-6,
                n_seeds=5,
                basis="chiral",
                label="chiral PINN",
                transition=2.0,
            )
        },
    )


def test_save_xai_report_writes_the_expected_set(tmp_path: Path) -> None:
    paths = save_xai_report(_analysis(), tmp_path / "xai", hopping=1.0)

    assert set(paths) == {
        "transparency_axis",
        "conditioning",
        "residual_field",
        "dispersion_chiral",
    }
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0
    plt.close("all")


def test_save_xai_report_skips_absent_inputs(tmp_path: Path) -> None:
    paths = save_xai_report(
        XaiAnalysis(profiles=[], errors={}), tmp_path / "xai", hopping=1.0
    )

    assert paths == {}
    plt.close("all")


def test_shared_dispersion_ylim_spans_all_series_with_padding() -> None:
    mu = np.linspace(-4.0, 4.0, 8)

    def _disp(scale: float) -> SeedDispersion:
        return SeedDispersion(
            mu=mu,
            energy_std=np.full(8, scale),
            density_std_mean=np.full(8, scale),
            density_std_max=np.full(8, scale * 2.0),
            edge_weight_std=np.full(8, scale),
            n_seeds=3,
            transition=2.0,
        )

    lo, hi = shared_dispersion_ylim([_disp(1e-6), _disp(1e-2)])

    assert lo < 1e-6  # padded below the smallest positive value
    assert hi > 2e-2  # padded above the largest value


def test_shared_dispersion_ylim_none_when_all_zero() -> None:
    mu = np.linspace(-4.0, 4.0, 8)
    flat = SeedDispersion(
        mu=mu,
        energy_std=np.zeros(8),
        density_std_mean=np.zeros(8),
        density_std_max=np.zeros(8),
        edge_weight_std=np.zeros(8),
        n_seeds=3,
        transition=2.0,
    )

    assert shared_dispersion_ylim([flat]) is None
    assert shared_dispersion_ylim([]) is None
