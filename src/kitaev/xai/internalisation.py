"""Accounting for how much known structure each model has internalised.

The project's four models solve the same spectral problem but differ in
*where* the physics lives. Some properties (unit norm, energy sign, the
plus-minus pairing, the particle-hole partner relation, evenness in the
chemical potential) can either be guaranteed by the architecture or merely
encouraged by a loss term. This module turns that qualitative distinction
into a small set of counts, so the four-model progression can be plotted as
a measured trade-off between what the network is asked to learn and how
accurate it is (see :func:`kitaev.xai.figures.plot_transparency_axis`).

Nothing here runs a model. A :class:`InternalisationProfile` is a static
description assembled by hand from a model and its loss, and
:data:`KITAEV_PROFILES` holds the four profiles for the current study.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InternalisationProfile:
    """A static description of what one model guarantees versus penalises.

    Attributes:
        name: The model identifier, matching the comparison harness recipe
            name.
        structural_guarantees: Physical properties that hold exactly by
            construction, as short human-readable labels.
        penalised_constraints: Physical properties that are only encouraged
            by a loss term rather than guaranteed.
        n_loss_terms: Number of additive terms in the training loss.
        n_tunable_weights: Number of relative weights or annealing
            schedules the optimiser has to balance across those terms.
        raw_output_dim: Width of the network's output head.
        effective_target: One line describing the object the network must
            actually fit once the structural guarantees are removed.
    """

    name: str
    structural_guarantees: tuple[str, ...]
    penalised_constraints: tuple[str, ...]
    n_loss_terms: int
    n_tunable_weights: int
    raw_output_dim: int
    effective_target: str

    @property
    def n_structural(self) -> int:
        """Number of physical properties guaranteed by construction."""
        return len(self.structural_guarantees)

    @property
    def loss_workload(self) -> int:
        """Loss terms plus tunable weights the optimiser has to balance.

        A compact proxy for how much bookkeeping has been left to the
        loss rather than moved into the architecture or the
        representation.
        """
        return self.n_loss_terms + self.n_tunable_weights


#: The four profiles for the current Kitaev-chain study, keyed by the
#: comparison harness recipe name. Counts follow the loss and model cards
#: in ``experiments/four_model_comparison.py``.
KITAEV_PROFILES: dict[str, InternalisationProfile] = {
    "semi_supervised": InternalisationProfile(
        name="semi_supervised",
        # Single-head SirenPINN + SemiSupervisedFSMLoss: only the unit norm
        # is structural. The energy is the signed Rayleigh quotient
        # psi^T H psi, pinned to the +E branch by the label on the 700
        # labelled rows and by evaluation-time alignment elsewhere -- not a
        # softplus head, so "energy non-negative" is no longer structural.
        # Same structural content as nambu_baseline; the two differ only in
        # labels vs the annealed pin.
        structural_guarantees=("unit norm",),
        penalised_constraints=(
            "eigen-equation",
            "energy non-negative",
            "particle-hole partner",
            "evenness in mu",
        ),
        n_loss_terms=4,  # data E, data psi, fsm, var
        n_tunable_weights=1,  # physics-weight anneal
        raw_output_dim=40,
        effective_target=(
            "a 2N eigenvector with a Rayleigh-quotient energy, anchored by a "
            "few exact labels"
        ),
    ),
    "nambu_baseline": InternalisationProfile(
        name="nambu_baseline",
        structural_guarantees=("unit norm",),
        penalised_constraints=(
            "eigen-equation",
            "energy non-negative",
            "particle-hole partner",
            "evenness in mu",
        ),
        n_loss_terms=4,  # fsm, var, particle-hole, pin
        n_tunable_weights=2,  # pin-weight anneal, fixed 0.1 on the ph term
        raw_output_dim=40,
        effective_target="a 2N eigenvector from a pure physics residual",
    ),
    "structural_nambu": InternalisationProfile(
        name="structural_nambu",
        structural_guarantees=("unit norm", "evenness in mu"),
        penalised_constraints=("eigen-equation", "energy-branch selection"),
        n_loss_terms=2,  # fsm, var
        n_tunable_weights=0,
        raw_output_dim=40,
        effective_target="a 2N eigenvector on the folded half-domain",
    ),
    "chiral": InternalisationProfile(
        name="chiral",
        structural_guarantees=(
            "unit norm",
            "plus-minus pairing",
            "particle-hole partner",
            "energy non-negative",
            "evenness in mu",
        ),
        penalised_constraints=("singular-pair consistency",),
        n_loss_terms=2,  # fsm, var
        n_tunable_weights=0,
        raw_output_dim=40,
        effective_target="one singular pair of a known N x N bidiagonal operator",
    ),
}
