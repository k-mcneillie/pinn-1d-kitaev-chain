"""Tests for kitaev.xai.internalisation."""

from __future__ import annotations

import dataclasses

import pytest

from kitaev.xai.internalisation import KITAEV_PROFILES


def test_all_four_profiles_present() -> None:
    assert set(KITAEV_PROFILES) == {
        "semi_supervised",
        "nambu_baseline",
        "structural_nambu",
        "chiral",
    }


def test_chiral_internalises_the_most_and_the_nambu_pair_the_least() -> None:
    by_name = {name: p.n_structural for name, p in KITAEV_PROFILES.items()}
    # chiral is the unique maximum.
    assert by_name["chiral"] == max(by_name.values())
    assert by_name["chiral"] > by_name["structural_nambu"]
    # semi_supervised and nambu_baseline share the minimum: only the unit
    # norm is structural for either (semi_supervised's energy is a signed
    # Rayleigh quotient, not a softplus head), so they differ by supervision
    # alone. structural_nambu sits one above, having folded evenness in mu in.
    least = min(by_name.values())
    assert by_name["semi_supervised"] == least
    assert by_name["nambu_baseline"] == least
    assert by_name["structural_nambu"] == least + 1


def test_derived_counts_are_consistent() -> None:
    for profile in KITAEV_PROFILES.values():
        assert profile.n_structural == len(profile.structural_guarantees)
        assert profile.loss_workload == (
            profile.n_loss_terms + profile.n_tunable_weights
        )


def test_profile_is_frozen() -> None:
    profile = KITAEV_PROFILES["chiral"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.name = "changed"  # type: ignore[misc]
