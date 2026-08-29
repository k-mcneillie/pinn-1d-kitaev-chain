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


def test_chiral_internalises_the_most_and_baseline_the_least() -> None:
    ordered = sorted(KITAEV_PROFILES.values(), key=lambda profile: profile.n_structural)
    assert ordered[-1].name == "chiral"
    assert ordered[0].name == "nambu_baseline"


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
