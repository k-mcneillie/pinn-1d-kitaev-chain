"""Tests for kitaev.xai.conditioning."""

from __future__ import annotations

import numpy as np
import pytest

from kitaev.xai.conditioning import sweep_conditioning

MU_GRID = np.linspace(-4.0, 4.0, 41)


@pytest.mark.parametrize("basis", ["nambu", "chiral"])
def test_shapes_and_condition_number_definition(basis: str) -> None:
    sweep = sweep_conditioning(
        basis=basis, n_sites=8, hopping=1.0, pairing=0.5, mu_grid=MU_GRID
    )

    assert sweep.mu.shape == sweep.sigma_min.shape == (41,)
    assert sweep.sigma_max.shape == sweep.condition_number.shape == (41,)
    assert np.all(sweep.sigma_min > 0.0)
    assert sweep.condition_number == pytest.approx(sweep.sigma_max / sweep.sigma_min)
    assert sweep.transition == pytest.approx(2.0)


def test_chiral_gap_survives_where_the_nambu_gap_collapses() -> None:
    # At mu = 0 the Nambu residual gap is the exponentially small Majorana
    # splitting, whereas the chiral gap to the next singular value stays of
    # order t.
    at_zero = np.array([0.0])
    nambu = sweep_conditioning(
        basis="nambu", n_sites=16, hopping=1.0, pairing=0.5, mu_grid=at_zero
    )
    chiral = sweep_conditioning(
        basis="chiral", n_sites=16, hopping=1.0, pairing=0.5, mu_grid=at_zero
    )

    assert nambu.sigma_min[0] < 1e-2
    assert chiral.sigma_min[0] > 0.1
    assert chiral.sigma_min[0] > 10.0 * nambu.sigma_min[0]


def test_rejects_unknown_basis() -> None:
    with pytest.raises(ValueError, match="basis must be one of"):
        sweep_conditioning(
            basis="majorana", n_sites=6, hopping=1.0, pairing=0.5, mu_grid=MU_GRID
        )
