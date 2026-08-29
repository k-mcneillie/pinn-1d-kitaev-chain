"""Matched-conditions, multi-seed comparison of the four Kitaev-chain models.

Every model shares the evaluation harness -- identical ``SamplingRegion``
mixture (mirrored for the two-sided models), identical ``mu`` grid,
identical two-phase *structure* (AdamW warm-up then L-BFGS fine-tune), and
the shared :class:`BdGEvaluationProbe` metric set plus a mu-reflection
residual.

The AdamW epoch budget is set *per model* (``Recipe.adam_epochs``), each to
its own convergence plateau from the ``--budget-sweep`` pilot. A single
shared budget would guard against "model X only won because it trained
longer"; that objection does not apply here, because every model reaches
its floor on the gauge-invariant metrics (energy, subspace) well inside any
of these budgets, and the Nambu-basis models' per-site density error in the
topological phase is under-determined by the objective and so does not
improve with more epochs (see
``docs/markdown/under-determination-and-n-scaling.md``). Training each model
to its own plateau is therefore the honest choice; the L-BFGS tail is fixed
across models.

All configuration is recorded exhaustively: per-region sample counts (per
batch / per epoch / over the AdamW phase), the frozen validation and L-BFGS
pools, the evaluation grid, the full optimiser spec, and per-model
architecture (layer list, parameter counts, buffers) go out as ``sesh``
model / dataset cards; per-run outcomes go via ``log_metrics``; the six
standard figures per (model, seed) are rendered by
:func:`kitaev.visualisation.save_run_figures`; a tidy CSV and a per-model
median / IQR / worst-seed / pass-rate summary are also written. The CSV lives
in the session
directory, so a run never overwrites an earlier one and the
interpretability notebook can load it from the same place as the
checkpoints.

Each label-free model runs its full fixed budget and the final-epoch state
is the one scored, checkpointed, and figured (``restore_best=False``).
Best-validation selection is avoided because the label-free loss plateaus
at its energy-squared floor while the eigenvector is still sharpening.
``semi_supervised`` is the exception: it consumes exact labels, its
validation loss bottoms out well before the budget ends, and it keeps its
best-validation state (``restore_best=True``) so the comparison measures
the approach rather than a hobbled version of it. Every CSV row carries
provenance for the state it describes -- ``completed``,
``final_epoch``, ``best_val_epoch`` / ``best_val_loss``,
``state_dict_sha256`` -- and a convergence read-out: ``var_tail_decades``
and ``infidelity_tail_decades`` (base-10 decades of descent over the final
quarter of training) with a ``converged`` flag that is true when both are
below 0.3.

An optional ``--budget-sweep`` mode replaces the comparison with an AdamW
epoch-count ablation (fixed L-BFGS tail) to show the trivial-phase gap is
stable across training budgets. It writes the same per-run figures and
checkpoint as the main comparison, keyed by budget under
``figures/<model>/adam_<budget>/seed_<seed>/``.

``sesh`` currently writes every model card to
``<sub_folder>/model_card.md`` and every dataset card to
``<sub_folder>/dataset_card.md`` -- a fixed filename -- so two cards sharing
a ``sub_folder`` overwrite each other. Every card here is therefore given
its own ``sub_folder`` (``model_artifacts/<model>``,
``data_artifacts/<mixture>``).

This exists to settle the trivial-phase accuracy finding (structural Nambu
~= 4.6e-3 vs chiral ~= 6.5e-6): the two earlier runs differed in BOTH the
sampling regions AND the L-BFGS budget (nb 4 ran 100 L-BFGS epochs, nb 3 ran
300). Here both are held fixed so the comparison is clean.

After the seed loop, cross-model figures are rendered from the checkpoints
into ``<session>/figures/comparison/`` -- the headline energy-error panel,
a spectral-fan context figure and a per-model density waterfall.
``--figures-only <session dir>`` re-renders those against an existing run
without retraining, and also refreshes every per-seed
``figures/<model>/**/seed_*/wavefunctions.png`` from its checkpoint (the
only per-seed figure that does not need the training history).

Usage:
    python experiments/four_model_comparison.py --smoke
    python experiments/four_model_comparison.py --seeds 0 1 2 3 4
    python experiments/four_model_comparison.py --budget-sweep \\
        --models chiral structural_nambu --seeds 0 1
    python experiments/four_model_comparison.py --figures-only \\
        results/logs/<session>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import platform
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from sesh import Session
from torch.utils.data import DataLoader

from kitaev.analytical import KitaevChainHamiltonian
from kitaev.data import SupervisedKitaevDataset, UnsupervisedMuGenerator
from kitaev.data.mu_sampler import MuSampler
from kitaev.data.sampling_region import SamplingRegion
from kitaev.models import (
    ChiralToBdGAdapter,
    RayleighEnergyAdapter,
    SirenPINN,
    SirenPINNChiral,
    SirenPINNDualHead,
    SirenPINNNambuFolded,
)
from kitaev.training import BdGEvaluationProbe, TwoPhaseConfig, run_two_phase
from kitaev.training.config import TrainerConfig
from kitaev.training.loss import (
    ChiralFSMLoss,
    NambuFSMLoss,
    PinnedFSMLoss,
    SemiSupervisedLoss,
)
from kitaev.training.sampling import SamplingConfig, build_sampling
from kitaev.training.trainer import _build_kitaev_operators
from kitaev.visualisation import (
    build_model_error_band,
    fsm_convergence_floor,
    plot_model_comparison,
    plot_pair_density_waterfall,
    plot_spectral_fan,
    plot_wavefunction_waterfall,
    rerender_wavefunctions,
    save_run_figures,
    sweep_low_spectrum,
    sweep_spectrum,
    sweep_wavefunctions,
)

# --------------------------------------------------------------------------
# Shared physical parameters
# --------------------------------------------------------------------------
N = 20
T = 1.0
DELTA = 0.5
TRANSITION = 2.0 * T
MU_GRID = np.linspace(-4.0, 4.0, 240)

# --------------------------------------------------------------------------
# Shared, matched training configuration
# --------------------------------------------------------------------------
# adam_epochs here is only the fallback for a recipe that does not set its
# own Recipe.adam_epochs, and the base the --budget-sweep overrides. The
# converged run-of-record budgets are per model (see the RECIPES table),
# each taken from the 20260829 pilot sweep: structural_nambu plateaus by
# ~9k, chiral settles by ~12k, semi_supervised converges early (label
# driven, restore_best), nambu_baseline follows structural_nambu.
TWO_PHASE = TwoPhaseConfig(
    adam_epochs=9000,
    adam_lr=8e-4,
    adam_weight_decay=1e-6,
    lbfgs_epochs=300,
    lbfgs_max_iter=25,
    lbfgs_history_size=20,
    lbfgs_line_search_fn="strong_wolfe",
)
SMOKE_TWO_PHASE = TwoPhaseConfig(
    adam_epochs=6,
    adam_lr=8e-4,
    adam_weight_decay=1e-6,
    lbfgs_epochs=4,
    lbfgs_max_iter=5,
    lbfgs_history_size=10,
    lbfgs_line_search_fn="strong_wolfe",
)
# restore_best=False is the default: the label-free models run their full
# fixed budget and the final-epoch state is the one scored and
# checkpointed. Best-validation selection is unwanted for them because the
# label-free loss plateaus at its energy-squared floor long before the
# eigenvector stops sharpening, so a rollback would freeze a worse
# wavefunction. The best epoch is still recorded per run (see the
# provenance columns in run_one). Individual recipes override this via
# Recipe.restore_best -- semi_supervised does, being label-driven.
TWO_PHASE_BASE = TrainerConfig(
    epochs=1, print_freq=1000, patience=None, grad_clip_norm=1.0, restore_best=False
)
# A seed "passes" if it clears both bars at once: energy MAE over the full
# domain, and worst-mu subspace infidelity. The pass-rate over seeds is the
# single most legible reliability number and it exposes the models that are
# bimodal over seeds (a median alone hides those). See summarise().
PASS_E_MAE_MAX = 1e-4
PASS_INFIDELITY_MAX = 1e-2

# A run is called converged if both its eigenvector-consistency loss and
# its subspace infidelity fell by less than this many base-10 decades over
# the final quarter of training. More than this means the budget is short.
CONVERGENCE_DECADE_TOL = 0.3

# One region mixture, dense on the transition shoulders and the deep
# interior. HALF is for the folded models (train on [0, 4t]); FULL is its
# mirror for the two-sided models, at 2x the batch to match density.
#
# The lower edge is 0.01t, not 0. Right at mu = 0 the smallest singular
# value sigma_1 -> 0, so the folded-spectrum objective has no gradient
# there and the folded models cannot be constrained in |mu| <~ 0.01t; the
# small residual bump seen at mu = 0 is that domain-edge degeneracy, not a
# model failure. The fold itself (psi(-mu) from psi(mu)) is exact, so the
# lower edge can sit as close to 0 as we like without breaking anything.
HALF_REGIONS = (
    SamplingRegion(low=0.01, high=4.0, weight=1.0),
    SamplingRegion(low=1.7, high=2.6, weight=1.5),
    SamplingRegion(low=2.0, high=4.0, weight=0.5),
)
FULL_REGIONS = (
    SamplingRegion(low=-4.0, high=4.0, weight=1.0),
    SamplingRegion(low=-2.6, high=-1.7, weight=1.5),
    SamplingRegion(low=1.7, high=2.6, weight=1.5),
    SamplingRegion(low=-4.0, high=-2.0, weight=0.5),
    SamplingRegion(low=2.0, high=4.0, weight=0.5),
)
HALF_BATCH = 1024
FULL_BATCH = 2048
STEPS_PER_EPOCH = 8
SMOKE_STEPS_PER_EPOCH = 3

# semi_supervised is the one method that also consumes exact labels. The
# label-free collocation budget is matched to the other two-sided models
# (FULL_REGIONS, FULL_BATCH); only this small labelled set is extra, and it
# is drawn from the same region mixture.
N_LABELLED_TRAIN = 700
N_LABELLED_VAL = 140
LABELLED_BATCH = 70
SMOKE_N_LABELLED_TRAIN = 28
SMOKE_N_LABELLED_VAL = 14
SMOKE_LABELLED_BATCH = 14

# AdamW-epoch budgets for the optional --budget-sweep ablation. Only the
# AdamW count varies; the L-BFGS tail stays fixed so the point is "does
# more first-stage optimisation help", not "does more of everything help".
BUDGET_SWEEP_DEFAULT = (1000, 2000, 3000, 6000)
SMOKE_BUDGET_SWEEP = (4, 8)


def _two_phase(smoke: bool) -> TwoPhaseConfig:
    return SMOKE_TWO_PHASE if smoke else TWO_PHASE


def _steps_per_epoch(smoke: bool) -> int:
    return SMOKE_STEPS_PER_EPOCH if smoke else STEPS_PER_EPOCH


# --------------------------------------------------------------------------
# Metadata helpers -- everything the cards need, computed, not hand-typed
# --------------------------------------------------------------------------
def _allocate_counts(regions: tuple[SamplingRegion, ...], batch_size: int) -> list[int]:
    """Mirror of ``MuSampler._allocate_counts``: floor split by weight,

    remainder assigned to the final region. This is exactly how many mu
    values each region contributes to every drawn batch.
    """
    total = sum(r.weight for r in regions)
    counts = [int(batch_size * r.weight / total) for r in regions]
    counts[-1] += batch_size - sum(counts)
    return counts


def _phase_of(low: float, high: float) -> str:
    """Classify an interval relative to the transition at ``|mu| = 2t``."""
    if max(abs(low), abs(high)) < TRANSITION:
        return "topological"
    if min(abs(low), abs(high)) >= TRANSITION and low * high >= 0:
        return "trivial"
    return "transition/mixed"


def sampling_metadata(
    regions: tuple[SamplingRegion, ...],
    *,
    batch_size: int,
    steps_per_epoch: int,
    adam_epochs: int,
) -> dict[str, Any]:
    """Full per-region + aggregate breakdown of the streaming train sampler."""
    total_w = sum(r.weight for r in regions)
    counts = _allocate_counts(regions, batch_size)
    per_region = []
    for r, c in zip(regions, counts, strict=True):
        width = r.high - r.low
        per_region.append(
            {
                "interval_t": f"[{r.low:g}, {r.high:g}]",
                "low": r.low,
                "high": r.high,
                "width": round(width, 6),
                "weight": r.weight,
                "weight_share": round(r.weight / total_w, 6),
                "phase": _phase_of(r.low, r.high),
                "samples_per_batch": c,
                "samples_per_epoch": c * steps_per_epoch,
                "samples_adam_phase": c * steps_per_epoch * adam_epochs,
                "samples_per_unit_mu_per_batch": round(c / width, 3),
            }
        )
    return {
        "sampler_class": "MuSampler (via build_sampling)",
        "mode": "infinite (streaming: a fresh batch is drawn every optimiser step)",
        "allocation_rule": (
            "floor(batch_size * weight / total_weight) per region, remainder "
            "to the last region, then the whole batch is shuffled"
        ),
        "within_region_draw": "uniform(low, high) via torch.rand",
        "n_regions": len(regions),
        "total_weight": total_w,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "adam_epochs": adam_epochs,
        "points_per_batch": batch_size,
        "points_per_epoch": batch_size * steps_per_epoch,
        "points_adam_phase": batch_size * steps_per_epoch * adam_epochs,
        "regions": per_region,
    }


def frozen_pool_metadata(
    regions: tuple[SamplingRegion, ...], pool_size: int, *, purpose: str
) -> dict[str, Any]:
    """Per-region composition of a frozen (drawn-once, reused) pool."""
    counts = _allocate_counts(regions, pool_size)
    return {
        "purpose": purpose,
        "mode": "frozen (drawn once; identical batch reused every epoch)",
        "pool_size": pool_size,
        "effective_unique_points": pool_size,
        "per_region_counts": {
            f"[{r.low:g},{r.high:g}]": c for r, c in zip(regions, counts, strict=True)
        },
    }


def eval_grid_metadata() -> dict[str, Any]:
    """The fixed grid the probe scores on -- independent of the training sampler."""
    topo = int((np.abs(MU_GRID) < TRANSITION).sum())
    return {
        "grid": "np.linspace(-4, 4, 240)",
        "n_points": int(MU_GRID.size),
        "spacing_mu": round(float(MU_GRID[1] - MU_GRID[0]), 6),
        "span_t": [float(MU_GRID[0]), float(MU_GRID[-1])],
        "n_topological": topo,
        "n_trivial": int(MU_GRID.size) - topo,
        "reference": (
            "numpy.linalg.eigh(KitaevChainHamiltonian.build(mu)); the exact "
            "references are diagonalised once and cached at probe construction"
        ),
        "independent_of_training_sampler": True,
    }


def _layer_specs(model: torch.nn.Module) -> list[str]:
    specs = []
    for sine in model.net:  # Sequential of SineLayer
        lin = sine.linear
        specs.append(
            f"SineLayer({lin.in_features}->{lin.out_features}, "
            f"omega_0={sine.omega_0}, is_first={sine.is_first})"
        )
    head = model.psi_head
    specs.append(
        f"Linear({head.in_features}->{head.out_features}, "
        f"bias={head.bias is not None}) [psi head]"
    )
    if hasattr(model, "energy_head"):
        energy = model.energy_head
        specs.append(
            f"Linear({energy.in_features}->{energy.out_features}, "
            f"bias={energy.bias is not None}) [energy head -> softplus]"
        )
    return specs


def model_metadata(model: torch.nn.Module) -> dict[str, Any]:
    """Architecture facts read straight off an instantiated model."""
    return {
        "class": type(model).__name__,
        "param_count_total": sum(p.numel() for p in model.parameters()),
        "param_count_trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "param_count_per_child": {
            name: sum(p.numel() for p in child.parameters())
            for name, child in model.named_children()
        },
        "buffers": {name: list(buf.shape) for name, buf in model.named_buffers()},
        "layers": _layer_specs(model),
        "float_dtype": str(next(model.parameters()).dtype),
        "normalisation": "F.normalize(p=2, dim=1, eps=1e-12) on the head output",
        "head_init": "uniform(-b, b), b = sqrt(6 / hidden_features) / hidden_omega_0",
    }


def optimiser_metadata(smoke: bool, adam_epochs: int | None = None) -> dict[str, Any]:
    """The complete two-phase optimiser spec (see run_two_phase).

    ``adam_epochs`` overrides the AdamW phase length for this card, so a
    per-model :attr:`Recipe.adam_epochs` is reflected accurately. It is
    ignored under ``smoke``.
    """
    tp = _two_phase(smoke)
    if adam_epochs is not None and not smoke:
        tp = replace(tp, adam_epochs=adam_epochs)
    return {
        "strategy": (
            "two-phase (run_two_phase): AdamW warm-up, then L-BFGS fine-tune "
            "on the same model"
        ),
        "phase1_adamw": {
            "optimiser": "torch.optim.AdamW",
            "epochs": tp.adam_epochs,
            "lr": tp.adam_lr,
            "weight_decay": tp.adam_weight_decay,
            "betas": "(0.9, 0.999) [torch default]",
            "eps": "1e-8 [torch default]",
            "lr_scheduler": f"CosineAnnealingLR(T_max={max(1, tp.adam_epochs)})",
            "grad_clip_norm": TWO_PHASE_BASE.grad_clip_norm,
            "train_loader": "streaming (mode=infinite): fresh batch every step",
        },
        "phase2_lbfgs": {
            "optimiser": "torch.optim.LBFGS",
            "epochs": tp.lbfgs_epochs,
            "lr": tp.lbfgs_lr,
            "max_iter": tp.lbfgs_max_iter,
            "history_size": tp.lbfgs_history_size,
            "line_search_fn": tp.lbfgs_line_search_fn,
            "grad_clip": "disabled in the L-BFGS phase",
            "train_loader": "frozen pool: one fixed batch reused every epoch",
            "start_epoch": (
                "adam_epochs + 1 (continuous epoch axis; these losses have no "
                "epoch-keyed schedule, so it is a no-op here)"
            ),
        },
        "early_stopping": "disabled (patience=None); each phase runs its full budget",
        "checkpoint_selection": (
            "final-epoch state (restore_best=False); the best validation epoch "
            "is recorded per run but not restored"
        ),
    }


# --------------------------------------------------------------------------
# Model recipes + their static model-card content
# --------------------------------------------------------------------------
@dataclass
class Recipe:
    """Everything model-specific for one entry in the comparison."""

    name: str
    two_sided: bool  # False -> trains on [0, 4t] with a structural fold
    build_model: Any  # () -> nn.Module
    build_loss: Any  # (TwoPhaseConfig) -> BaseLoss
    build_adapt: Any  # (nn.Module) -> (E, psi) model
    build_loaders: Any  # (smoke: bool) -> LoaderBundle
    loss_class: str
    plot_label: str  # legend label for the predicted curves in the figures
    component_keys: tuple[str, ...]  # loss-component series for plot_loss_history
    structural_fold: bool  # evenness in mu is by construction (figure titles)
    card_architecture: str
    card_static_parameters: dict[str, Any]
    card_intended_use: list[str]
    card_limitations: list[str]
    card_loss: dict[str, Any]
    card_description: str
    weight_key: str | None = None  # annealing-weight series, if the loss has one
    basis: str = "nambu"  # residual basis for the XAI pipeline: "nambu" | "chiral"
    # train-history key for the eigenvector-consistency term, whose tail
    # slope says whether the wavefunction was still sharpening at the budget
    # cut-off (see the convergence columns in run_one).
    residual_key: str = "var"
    # multiplier on <E_1(mu)^2> for the analytic floor line on the loss
    # figure. 1.0 for the Nambu folded-spectrum losses, 2.0 for the chiral
    # loss (it sums two residual terms), None for losses with no such floor
    # (semi_supervised's supervised residual bottoms out at zero).
    fsm_floor_factor: float | None = 1.0
    # Per-recipe override of the shared restore_best=False protocol. The
    # label-free structural models all run their full fixed budget and keep
    # the final-epoch state. semi_supervised is the exception: it is the
    # only model that consumes exact labels, its folded-spectrum-dominated
    # validation loss bottoms out long before the budget ends, and holding
    # it to restore_best=False measures a hobbled version of it rather than
    # the approach itself. It therefore keeps its best-validation state.
    restore_best: bool = False
    # Per-model AdamW epoch budget for the converged run of record, each set
    # to the model's own plateau from the --budget-sweep pilot. None falls
    # back to TWO_PHASE.adam_epochs. Ignored under --smoke and overridden by
    # --budget-sweep. See the module docstring for why the budget is not
    # shared across models.
    adam_epochs: int | None = None
    run_kwargs: dict = field(default_factory=dict)


_SIREN_STATIC = {
    "backbone": "SIREN (sinusoidal representation network)",
    "hidden_features": 64,
    "hidden_layers": 2,
    "first_layer_omega_0": 30.0,
    "hidden_omega_0": 1.0,
    "input_scale": 4.0,
    "in_features": 1,
}

RECIPES: dict[str, Recipe] = {
    "structural_nambu": Recipe(
        name="structural_nambu",
        adam_epochs=9000,  # pilot: flat 9k -> 12k
        two_sided=False,
        build_model=lambda: SirenPINNNambuFolded(
            n_sites=2 * N, hidden_features=64, hidden_layers=2, input_scale=4.0
        ),
        build_loss=lambda _tp: NambuFSMLoss(),
        build_adapt=lambda m: RayleighEnergyAdapter(
            m, n_sites=N, hopping=T, pairing=DELTA
        ),
        build_loaders=lambda smoke: _unsupervised_loaders(two_sided=False, smoke=smoke),
        loss_class="NambuFSMLoss",
        plot_label="structural Nambu",
        component_keys=("fsm", "var"),
        structural_fold=True,
        card_architecture=(
            "SIREN coordinate backbone; one linear head -> 2N Nambu-basis BdG "
            "eigenvector. The mu -> -mu reflection is folded into the forward "
            "pass: psi(mu) = normalise(0.5 [g(mu) + Gamma g(-mu)]), "
            "Gamma = -(tau_x (x) D), D = diag((-1)^n). Energy is the Rayleigh "
            "quotient psi^T H(mu) psi, evaluated by the loss, not a head output."
        ),
        card_static_parameters={
            **_SIREN_STATIC,
            "n_sites_arg": 2 * N,
            "output_dim": 2 * N,
            "fold": "structural (Gamma = -(tau_x (x) D)); trains on [0, 4t] only",
            "structural_guarantees": ["||psi|| = 1", "E(-mu) = E(mu)"],
        },
        card_intended_use=[
            "Label-free continuous surrogate for the lowest-|E| BdG eigenpair "
            "of the 1D Kitaev chain vs mu over -4t < mu < 4t.",
            "Gauge-invariant spectral quantities: energy, gap, combined edge "
            "weight, near-zero subspace fidelity.",
            "The 'symmetry in the architecture, not the loss' rung of the "
            "four-model comparison; baseline for the chiral-basis model.",
        ],
        card_limitations=[
            "Topological phase |mu| < 2t: fsm + var is exactly flat over the "
            "2D near-zero Majorana manifold (tie-breaking term ~ lambda_1^2 ~ "
            "e^{-2N/xi}), so the per-mu eigenvector is undetermined; only "
            "subspace-level quantities are meaningful there "
            "(docs/markdown/topological-eigenvector-ambiguity.md, "
            "manifold-density-rho.md).",
            "The +-E branch is a gauge (loss_pin dropped); resolved at "
            "evaluation by sign/Xi-alignment to the reference, not "
            "structurally.",
            "Trivial phase |mu| > 2t: energy MAE ~1e-3, markedly worse than "
            "the chiral model (~1e-6) -- the dense 2N x 2N eigen-residual is "
            "less well conditioned than the N x N bidiagonal SVD residual "
            "(this script exists to confirm that under matched conditions).",
            "Real Kitaev chain (class BDI) only.",
        ],
        card_loss={
            "class": "NambuFSMLoss",
            "terms": {
                "fsm": "mean(||H(mu) psi||^2)",
                "var": "mean(||H(mu) psi - E_R psi||^2), E_R = psi^T H(mu) psi",
            },
            "total": "fsm + var",
            "schedule": None,
            "relative_weights": None,
            "dropped_vs_PinnedFSMLoss": [
                "0.1 * ph  (== var identically; Xi H Xi = -H, Xi orthogonal)",
                "w_pin(epoch) * pin  (branch gauge; resolved at evaluation)",
            ],
        },
        card_description=(
            "SirenPINNNambuFolded + NambuFSMLoss. Two loss terms (fsm + var), "
            "no schedule, no weights; norm and mu-parity structural."
        ),
    ),
    "chiral": Recipe(
        name="chiral",
        # pilot: settles by ~12k (E MAE ~3e-6, two orders under the pass bar)
        adam_epochs=12000,
        two_sided=False,
        build_model=lambda: SirenPINNChiral(
            n_sites=N, hidden_features=64, hidden_layers=2, input_scale=4.0
        ),
        build_loss=lambda _tp: ChiralFSMLoss(n_sites=N, hopping=T, pairing=DELTA),
        build_adapt=lambda m: ChiralToBdGAdapter(m, hopping=T, pairing=DELTA),
        build_loaders=lambda smoke: _unsupervised_loaders(two_sided=False, smoke=smoke),
        loss_class="ChiralFSMLoss",
        plot_label="chiral PINN",
        component_keys=("fsm", "var", "lam_mean"),
        structural_fold=True,
        basis="chiral",
        fsm_floor_factor=2.0,  # loss_fsm = mean||hv||^2 + mean||h^T u||^2
        card_architecture=(
            "SIREN coordinate backbone; one head -> (u, v), the left/right "
            "singular vectors of the smallest singular value of the real "
            "N x N bidiagonal chiral block h(mu) (Majorana/BDI reduction, "
            "Omega H Omega^dag = i [[0, h], [-h^T, 0]]). Energy = "
            "Rayleigh-quotient singular value u^T h(mu) v. mu -> -mu folded "
            "via h(-mu) = -D h(mu) D; +-E branch fixed by "
            "resolve_singular_branch at reconstruction."
        ),
        card_static_parameters={
            **_SIREN_STATIC,
            "n_sites_arg": N,
            "output_dim": 2 * N,
            "reduced_block": (
                "h(mu): real N x N bidiagonal, diag -mu, super -(t+d), sub -(t-d); "
                "its singular values are the non-negative BdG spectrum"
            ),
            "fold": "structural (h(-mu) = -D h(mu) D); trains on [0.05, 4t]",
            "structural_guarantees": [
                "||u|| = ||v|| = 1",
                "+-E spectral pairing (SVD)",
                "Xi-partner = block swap",
                "E >= 0 (singular value)",
                "E(-mu) = E(mu)",
            ],
        },
        card_intended_use=[
            "Label-free continuous surrogate for the lowest-|E| BdG eigenpair "
            "of the 1D Kitaev chain vs mu over -4t < mu < 4t.",
            "A determined near-zero eigenvector: sigma_1(h) is simple (O(t) "
            "gap), so the reconstruction is the balanced energy eigenstate "
            "(site density rho/2).",
            "The 'symmetry in the representation' rung of the four-model "
            "comparison; the basis where +-E pairing and E >= 0 are structural.",
        ],
        card_limitations=[
            "mu = 0: the -D factor makes u sign-discontinuous unless u(0) is "
            "odd-supported -- a measure-zero gauge artefact; sampling excludes "
            "mu in [0, 0.05 t).",
            "Reconstructs the energy eigenstates {psi_+, psi_-}, not the "
            "end-localised Majoranas {gamma_L, gamma_R} (a further 45-degree "
            "rotation, or the delta-W construction).",
            "Single-triple parametrisation: the smallest singular pair only, "
            "not the full spectrum -- per-mode wavefunctions across the "
            "degenerate phase need the O(N) SVD extension.",
            "Class BDI only: complex Delta, NNN hopping that spoils the "
            "bipartite lattice, or a Zeeman term removes the "
            "block-off-diagonal form.",
        ],
        card_loss={
            "class": "ChiralFSMLoss",
            "terms": {
                "fsm": "mean(||h v||^2) + mean(||h^T u||^2)",
                "var": (
                    "mean(||h v - sigma_R u||^2) + mean(||h^T u - sigma_R v||^2), "
                    "sigma_R = u^T h v"
                ),
            },
            "total": "fsm + var",
            "schedule": None,
            "relative_weights": None,
            "notes": "Nambu-basis loss_ph / loss_pin are structural here, so absent.",
        },
        card_description=(
            "SirenPINNChiral + ChiralFSMLoss. Two loss terms (fsm + var), no "
            "schedule, no weights; norm, +-E pairing, Xi-partner, E >= 0 and "
            "mu-parity all structural."
        ),
    ),
    "nambu_baseline": Recipe(
        name="nambu_baseline",
        # no direct pilot; follows structural_nambu (same basis, softer loss)
        adam_epochs=9000,
        two_sided=True,
        build_model=lambda: SirenPINN(
            n_sites=2 * N, hidden_features=64, hidden_layers=2, input_scale=4.0
        ),
        build_loss=lambda tp: PinnedFSMLoss(
            total_epochs=tp.adam_epochs,
            anneal_duration=max(1, int(tp.adam_epochs * 2 / 3)),
        ),
        build_adapt=lambda m: RayleighEnergyAdapter(
            m, n_sites=N, hopping=T, pairing=DELTA
        ),
        build_loaders=lambda smoke: _unsupervised_loaders(two_sided=True, smoke=smoke),
        loss_class="PinnedFSMLoss",
        plot_label="FSM baseline",
        component_keys=("fsm", "var", "ph", "pin"),
        weight_key="pin_wt",
        structural_fold=False,
        card_architecture=(
            "SIREN coordinate backbone; one linear head -> 2N Nambu-basis BdG "
            "eigenvector, L2-normalised. No mu-fold: trained on the full "
            "[-4t, 4t] domain. Energy is the Rayleigh quotient psi^T H(mu) psi "
            "(RayleighEnergyAdapter), returned signed -- the E >= 0 branch is "
            "only softly selected, by the annealed loss_pin term."
        ),
        card_static_parameters={
            **_SIREN_STATIC,
            "n_sites_arg": 2 * N,
            "output_dim": 2 * N,
            "fold": "none (full-domain training over [-4t, 4t])",
            "structural_guarantees": ["||psi|| = 1"],
        },
        card_intended_use=[
            "Label-free continuous surrogate for the lowest-|E| BdG eigenpair "
            "of the 1D Kitaev chain vs mu over -4t < mu < 4t.",
            "The soft-constraint status quo: the 'symmetry in the loss' rung "
            "of the four-model comparison, the baseline the folded, chiral and "
            "semi-supervised models are all measured against.",
            "Gauge-invariant spectral quantities only (energy, gap, edge "
            "weight, near-zero subspace fidelity).",
        ],
        card_limitations=[
            "Topological phase |mu| < 2t: fsm + var is flat over the 2D "
            "near-zero Majorana manifold, so the per-mu eigenvector is "
            "undetermined; only subspace-level quantities are meaningful "
            "there (docs/markdown/manifold-density-rho.md).",
            "The +-E branch and evenness in mu are both only learned / softly "
            "enforced, never structural; a trivial-phase sign flip or "
            "particle/hole swap is a genuine property of this model.",
            "loss_ph is provably redundant (Xi H Xi = -H, Xi orthogonal makes "
            "it equal to loss_var term by term) yet is still computed and "
            "weighted -- four terms and one anneal schedule to tune.",
            "Real Kitaev chain (class BDI) only.",
        ],
        card_loss={
            "class": "PinnedFSMLoss",
            "terms": {
                "fsm": "mean(||H(mu) psi||^2)",
                "var": "mean(||H(mu) psi - E_R psi||^2), E_R = psi^T H(mu) psi",
                "ph": "mean(||H(mu) (Xi psi) + E_R (Xi psi)||^2)  [== var]",
                "pin": "mean(softplus(-E_R, beta=10))  [selects E_R >= 0]",
            },
            "total": "fsm + var + 0.1 * ph + w_pin(epoch) * pin",
            "schedule": (
                "w_pin anneals 1.0 -> 0.01 linearly over anneal_duration "
                "(= floor(2/3 * adam_epochs)), then holds at 0.01"
            ),
            "relative_weights": "ph fixed at 0.1; fsm, var at 1.0",
        },
        card_description=(
            "SirenPINN + PinnedFSMLoss. The original soft-constraint recipe: "
            "four terms, one annealed weight; only ||psi|| = 1 is structural, "
            "everything else is penalised."
        ),
    ),
    "semi_supervised": Recipe(
        name="semi_supervised",
        adam_epochs=4000,  # label-driven + restore_best; converges early
        two_sided=True,
        build_model=lambda: SirenPINNDualHead(
            n_sites=2 * N, hidden_features=64, hidden_layers=3, input_scale=4.0
        ),
        build_loss=lambda tp: SemiSupervisedLoss(
            total_epochs=tp.adam_epochs,
            anneal_duration=max(1, tp.adam_epochs // 3),
        ),
        build_adapt=lambda m: m,  # the dual head already returns (E, psi)
        build_loaders=lambda smoke: _semisupervised_loaders(smoke),
        loss_class="SemiSupervisedLoss",
        plot_label="dual-head PINN (v1)",
        component_keys=("e", "psi", "res", "ph"),
        weight_key="physics_wt",
        residual_key="res",
        fsm_floor_factor=None,  # supervised residual has no E_1^2 floor
        # The original approach and the only label-consuming model; kept in
        # the comparison to motivate the move to label-free structural
        # methods, and evaluated under its intended protocol (best-val).
        restore_best=True,
        structural_fold=False,
        card_architecture=(
            "SIREN coordinate backbone with two linear heads: a scalar energy "
            "head through softplus (E >= 0 structural) and a 2N eigenvector "
            "head, L2-normalised. No mu-fold. Trained on the full [-4t, 4t] "
            "domain against a small exact-label set plus a large label-free "
            "collocation stream."
        ),
        card_static_parameters={
            **_SIREN_STATIC,
            "hidden_layers": 3,
            "hidden_omega_0": 2.0,  # SirenPINNDualHead's default, unlike the others
            "n_sites_arg": 2 * N,
            "output_dim": 2 * N,
            "energy_head": "Linear(hidden -> 1) then softplus(beta=10)",
            "energy_softplus_beta": 10.0,
            "fold": "none (full-domain training over [-4t, 4t])",
            "structural_guarantees": ["||psi|| = 1", "E >= 0 (softplus head)"],
        },
        card_intended_use=[
            "Semi-supervised continuous surrogate for the lowest-|E| BdG "
            "eigenpair of the 1D Kitaev chain vs mu over -4t < mu < 4t.",
            "The labelled-anchor rung of the four-model journey (semi-sup -> "
            "unsupervised -> basis + loss reduction -> operator): what a "
            "handful of exact eigh labels buys over a pure physics residual.",
            "Energy, gap, edge weight, near-zero subspace fidelity; per-mu psi "
            "in the trivial phase (the labels fix its sign there).",
        ],
        card_limitations=[
            "Needs exact labels: N_LABELLED_TRAIN exact diagonalisations, "
            "gauge-fixed for sign continuity, are computed up front.",
            "The psi label MSE only breaks the overall sign of psi; it does "
            "not resolve the 2D topological-phase manifold ambiguity, so "
            "per-mu topological eigenvectors remain under-determined.",
            "physics_weight anneals 0.01 -> 1.0 (over 1/3 of the AdamW "
            "epochs) -- another schedule to tune; loss_ph is carried despite "
            "being redundant with loss_res.",
            "+-E spectral pairing and evenness in mu are only learned; class "
            "BDI (real Kitaev chain) only.",
        ],
        card_loss={
            "class": "SemiSupervisedLoss",
            "terms": {
                "e": "mse(E_pred, E_exact) on labelled rows",
                "psi": "mse(psi_pred, sign-aligned psi_exact) on labelled rows",
                "res": "mean(||H psi_pred - E_pred psi_pred||^2), all rows",
                "ph": "mean(||H (Xi psi_pred) + E_pred (Xi psi_pred)||^2)  [== res]",
            },
            "total": "e + psi + w_phys(epoch) * (res + ph)",
            "schedule": (
                "w_phys anneals 0.01 -> 1.0 linearly over anneal_duration "
                "(= adam_epochs // 3), then holds at 1.0"
            ),
            "relative_weights": "e, psi unweighted; res + ph share w_phys",
            "data": {
                "labelled_train": N_LABELLED_TRAIN,
                "labelled_val": N_LABELLED_VAL,
                "labelled_batch": LABELLED_BATCH,
                "label_free_per_step": FULL_BATCH,
                "labels": "exact eigh (mu, E, psi), sign-continuity gauge-fixed",
                "region_mixture": "FULL_REGIONS (same as the other two-sided models)",
            },
        },
        card_description=(
            "SirenPINNDualHead + SemiSupervisedLoss. A few exact labels plus a "
            "label-free physics residual; E >= 0 and ||psi|| = 1 structural, "
            "the rest penalised with one annealed weight."
        ),
    ),
}

# The narrative order of the study: constraints migrating loss -> architecture
# -> representation. Used as the default --models list and the summary order.
MODEL_ORDER = ("semi_supervised", "nambu_baseline", "structural_nambu", "chiral")


# --------------------------------------------------------------------------
# sesh cards
# --------------------------------------------------------------------------
def log_dataset_cards(session: Session, *, smoke: bool) -> None:
    """One exhaustive card per data source used, each in its own sub_folder.

    ``sesh`` writes every dataset card to ``<sub_folder>/dataset_card.md``,
    so each card here gets a distinct ``data_artifacts/<slug>`` sub_folder
    or they would overwrite one another.
    """
    tp = _two_phase(smoke)
    steps = _steps_per_epoch(smoke)
    shared = {
        "domain_transition": "|mu| = 2t (independent of Delta)",
        "units": "energies and mu in units of the hopping t",
        "labels": "none (label-free collocation); only mu values are drawn",
        "rng": (
            "torch.manual_seed(seed) + np.random.seed(seed) per run; MuSampler "
            "draws via torch.rand / torch.randperm, so batches are "
            "seed-dependent"
        ),
        "evaluation_grid": eval_grid_metadata(),
    }
    for slug, regions, batch, domain, blurb in (
        (
            "kitaev-collocation-half-domain",
            HALF_REGIONS,
            HALF_BATCH,
            "[0.05, 4] t  (folded models; mu < 0 is the exact structural image)",
            "SirenPINNChiral and SirenPINNNambuFolded train on the folded half-domain.",
        ),
        (
            "kitaev-collocation-full-domain",
            FULL_REGIONS,
            FULL_BATCH,
            "[-4, 4] t  (two-sided models with no mu-fold)",
            "Mirror of the half-domain mixture for the two-sided models "
            "(SirenPINN, and SirenPINNDualHead's label-free stream), which "
            "have no structural fold; batch doubled to match collocation "
            "density.",
        ),
    ):
        session.log_dataset_card(
            name=slug,
            sub_folder=f"data_artifacts/{slug}",
            parameters={
                **shared,
                "domain": domain,
                "train_sampler": sampling_metadata(
                    regions,
                    batch_size=batch,
                    steps_per_epoch=steps,
                    adam_epochs=tp.adam_epochs,
                ),
                "validation_pool": frozen_pool_metadata(
                    regions,
                    batch,
                    purpose="validation-loss curve / checkpoint selection",
                ),
                "lbfgs_pool": frozen_pool_metadata(
                    regions,
                    2 * batch,
                    purpose="stationary objective for the L-BFGS phase",
                ),
            },
            description=(
                f"{blurb} A uniform full-domain slice (weight 1.0), the "
                f"transition shoulder [1.7, 2.6] t oversampled (weight 1.5), "
                f"and the trivial phase [2, 4] t (weight 0.5). Every "
                f"per-region sample count is in the train_sampler.regions "
                f"block."
            ),
        )

    # The one labelled source: consumed only by semi_supervised, drawn from
    # the same FULL_REGIONS mixture as the two-sided collocation card.
    n_train = SMOKE_N_LABELLED_TRAIN if smoke else N_LABELLED_TRAIN
    n_val = SMOKE_N_LABELLED_VAL if smoke else N_LABELLED_VAL
    label_batch = SMOKE_LABELLED_BATCH if smoke else LABELLED_BATCH
    session.log_dataset_card(
        name="kitaev-labelled-full-domain",
        sub_folder="data_artifacts/kitaev-labelled-full-domain",
        parameters={
            "domain_transition": "|mu| = 2t (independent of Delta)",
            "units": "energies and mu in units of the hopping t",
            "labels": (
                "exact numpy.linalg.eigh (mu, E, psi) triples; E is the "
                "lowest non-negative eigenvalue, psi its 2N Nambu eigenvector, "
                "sign-continuity gauge-fixed in mu-sorted order"
            ),
            "region_mixture": "FULL_REGIONS (same as the two-sided collocation card)",
            "consumed_by": "semi_supervised only",
            "train_samples": n_train,
            "val_samples": n_val,
            "train_batch_size": label_batch,
            "val_batch_size": n_val,
            "lbfgs_labelled_batch": "single frozen batch of all train_samples",
            "diagonalisations_up_front": n_train + n_val,
            "evaluation_grid": eval_grid_metadata(),
        },
        description=(
            "The small exact-label anchor set for the semi-supervised model. "
            "Every mu is diagonalised once at construction; the label-free "
            "physics residual still uses the matched full-domain collocation "
            "stream. Kept deliberately small -- the point of the "
            "semi-supervised rung is how little supervision is needed."
        ),
    )


_STANDARD_FIGURES = (
    "loss_history.png -- total train/val loss + components (+ anneal weight)",
    "probe_history.png -- energy / edge-weight MAE + subspace infidelity vs epoch",
    "energy_sweep.png -- exact vs model E(mu) with a log-scale error inset",
    "eigenvector_agreement.png -- near-zero subspace fidelity + combined edge weight",
    "wavefunctions.png -- particle/hole density at 5 mu, model vs exact, with rho/2",
    "mu_reflection.png -- |E(+mu)| vs |E(-mu)|",
)


def log_model_card(
    session: Session,
    recipe: Recipe,
    seeds: list[int],
    *,
    smoke: bool,
    adam_epochs: int | None = None,
) -> None:
    """One exhaustive card per model, each in its own sub_folder.

    ``sesh`` writes every model card to ``<sub_folder>/model_card.md``, so
    each card here gets a distinct ``model_artifacts/<model>`` sub_folder
    or they would overwrite one another.
    """
    model = recipe.build_model()
    if recipe.name == "semi_supervised":
        regime = "semi-supervised (few exact eigh labels + label-free physics residual)"
        dataset_card = [
            "kitaev-labelled-full-domain",
            "kitaev-collocation-full-domain (label-free stream)",
        ]
    else:
        regime = "label-free / unsupervised"
        dataset_card = (
            "kitaev-collocation-full-domain"
            if recipe.two_sided
            else "kitaev-collocation-half-domain"
        )
    session.log_model_card(
        name=recipe.name,
        sub_folder=f"model_artifacts/{recipe.name}",
        architecture=recipe.card_architecture,
        parameters={**recipe.card_static_parameters, **model_metadata(model)},
        intended_use=recipe.card_intended_use,
        limitations=recipe.card_limitations,
        description=recipe.card_description,
        training_metadata={
            "regime": regime,
            "loss": recipe.card_loss,
            "optimiser": optimiser_metadata(smoke, adam_epochs),
            "dataset_card": dataset_card,
            "evaluation": {
                "probe": "BdGEvaluationProbe on linspace(-4, 4, 240), every 50 epochs",
                "adapter": type(recipe.build_adapt(model)).__name__,
                "metrics": [
                    "probe_e_mae (+ topological / trivial split)",
                    "probe_edge_mae (combined particle+hole weight, 2 outer sites/end)",
                    "probe_subspace_infidelity (mean / max): 1 - ||P_M psi||",
                    "probe_psi_norm",
                    "reflection_residual: max_mu | |E(mu)| - |E(-mu)| |",
                ],
                "figures_per_seed": list(_STANDARD_FIGURES),
            },
            "seeds": seeds,
            "n_seeds": len(seeds),
            "smoke": smoke,
            "environment": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "device": str(Accelerator().device),
            },
        },
    )


# --------------------------------------------------------------------------
# Training + scoring
# --------------------------------------------------------------------------
@dataclass
class LoaderBundle:
    """The full set of loaders one recipe hands to :func:`run_two_phase`.

    ``unlabeled`` / ``lbfgs_unlabeled`` are only populated for a
    semi-supervised recipe, where ``train`` / ``val`` / ``lbfgs_train``
    carry the small labelled set and the large label-free collocation
    stream is separate; for a label-free recipe they are ``None`` and
    ``train`` is itself the (streaming) collocation loader.
    """

    train: DataLoader
    sampling_callbacks: list
    val: DataLoader
    lbfgs_train: DataLoader
    unlabeled: DataLoader | None = None
    lbfgs_unlabeled: DataLoader | None = None


def _unsupervised_loaders(two_sided: bool, smoke: bool) -> LoaderBundle:
    """Streaming collocation loader + frozen val + frozen L-BFGS pool, matched."""
    regions = FULL_REGIONS if two_sided else HALF_REGIONS
    batch = FULL_BATCH if two_sided else HALF_BATCH
    steps = _steps_per_epoch(smoke)
    train_loader, sampling_callbacks = build_sampling(
        SamplingConfig(mode="infinite", batch_size=batch, steps_per_epoch=steps),
        regions,
    )
    val_loader, _ = build_sampling(
        SamplingConfig(mode="frozen", batch_size=batch, total_samples=batch),
        regions,
    )
    lbfgs_loader, _ = build_sampling(
        SamplingConfig(mode="frozen", batch_size=2 * batch, total_samples=2 * batch),
        regions,
    )
    return LoaderBundle(
        train=train_loader,
        sampling_callbacks=list(sampling_callbacks),
        val=val_loader,
        lbfgs_train=lbfgs_loader,
    )


def _semisupervised_loaders(smoke: bool) -> LoaderBundle:
    """Small labelled set + matched label-free collocation stream.

    Labels come from exact diagonalisation of the same ``FULL_REGIONS``
    mixture the other two-sided models sample; the label-free budget
    (``FULL_BATCH`` per step) matches them too. The L-BFGS phase gets both
    contributions as a single frozen batch each, so the quasi-Newton
    curvature estimate sees a stationary objective.
    """
    n_train = SMOKE_N_LABELLED_TRAIN if smoke else N_LABELLED_TRAIN
    n_val = SMOKE_N_LABELLED_VAL if smoke else N_LABELLED_VAL
    label_batch = SMOKE_LABELLED_BATCH if smoke else LABELLED_BATCH
    steps = _steps_per_epoch(smoke)

    hamiltonian = KitaevChainHamiltonian(n_sites=N, hopping=T, pairing=DELTA)
    train_dataset = SupervisedKitaevDataset(
        sampler=MuSampler(FULL_REGIONS),
        total_samples=n_train,
        hamiltonian=hamiltonian,
    )
    val_dataset = SupervisedKitaevDataset(
        sampler=MuSampler(FULL_REGIONS),
        total_samples=n_val,
        hamiltonian=hamiltonian,
    )
    unlabeled = UnsupervisedMuGenerator(sampler=MuSampler(FULL_REGIONS)).dataloader(
        total_samples=FULL_BATCH * steps, batch_size=FULL_BATCH
    )
    lbfgs_unlabeled = UnsupervisedMuGenerator(
        sampler=MuSampler(FULL_REGIONS)
    ).dataloader(total_samples=2 * FULL_BATCH, batch_size=2 * FULL_BATCH)
    return LoaderBundle(
        train=train_dataset.dataloader(batch_size=label_batch, shuffle=True),
        sampling_callbacks=[],
        val=val_dataset.dataloader(batch_size=n_val, shuffle=False),
        lbfgs_train=train_dataset.dataloader(batch_size=n_train, shuffle=False),
        unlabeled=unlabeled,
        lbfgs_unlabeled=lbfgs_unlabeled,
    )


def _state_dict_sha256(model: torch.nn.Module) -> str:
    """Deterministic SHA-256 over a model's parameters and buffers.

    Hashes each tensor's raw bytes in sorted-key order, so the digest
    identifies exactly the weights that were scored and checkpointed and
    can be matched against the ``seed_*.pt`` file later.
    """
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _tail_decades(series: list[float], frac: float = 0.25) -> float:
    """Base-10 decades a positive series fell across its final ``frac``.

    A small positive number means the series had flattened by the budget
    cut-off. A large one means it was still descending and the budget is
    too short. Endpoints are averaged over a few samples to damp the
    L-BFGS tail's jumpiness. Returns 0.0 if the series is too short or
    non-positive.
    """
    if len(series) < 8:
        return 0.0
    tail = series[-max(4, int(len(series) * frac)) :]
    window = max(1, len(tail) // 4)
    start = float(np.mean(tail[:window]))
    end = float(np.mean(tail[-window:]))
    if start <= 0.0 or end <= 0.0:
        return 0.0
    return math.log10(start) - math.log10(end)


def _reflection_residual(adapter, device) -> float:
    """max_mu | |E(mu)| - |E(-mu)| | over [0, 4t] -- 0 iff mu-parity is exact."""
    adapter.eval()
    half = np.linspace(0.0, 4.0, 200)[:, None]
    with torch.no_grad():
        pos = torch.tensor(half, dtype=torch.float32, device=device)
        e_pos = np.abs(adapter(pos)[0].detach().cpu().numpy().ravel())
        e_neg = np.abs(adapter(-pos)[0].detach().cpu().numpy().ravel())
    return float(np.abs(e_pos - e_neg).max())


def _topo_density_maes(adapter: Any, device: str) -> tuple[float, float]:
    """Topological-phase site-density error: gauge-invariant vs raw.

    Both are means of ``|predicted - exact|`` over ``|mu| < 2t`` on
    ``MU_GRID``, with each state unit-normalised so its density has total
    mass 1.

    Returns ``(pair_density_mae, raw_density_mae)``, both MAEs over the
    ``|mu| < 2t`` grid of ``MU_GRID`` with each state unit-normalised:

    - ``pair_density_mae`` -- error of the gauge-invariant site density
      ``P_nn``, the particle-block diagonal of the projector onto the
      near-zero 2D subspace. For the model this is the diagonal of the
      projector onto ``span{psi, Xi psi}`` (Gram-Schmidt orthonormalised);
      for the reference it is the sum of squares of the two smallest-|E|
      exact eigenvectors. Basis-independent, so it is invariant to any
      rotation within the degenerate manifold; bounded by the subspace
      infidelity and expected to fall with the training budget for every
      model.
    - ``raw_density_mae`` -- per-site ``|Delta p_n| + |Delta h_n|`` of the
      raw sector densities ``|psi_k|^2`` against a single ``eigh``
      representative (``vecs[:, N]``). Gauge-*dependent* in the
      topological phase: neither the folded-spectrum loss nor ``eigh``
      pins a representative within the near-degenerate ``+-sigma_1`` pair.
      Expected to plateau for the Nambu models (the flat direction is
      never resolved) and to fall for the chiral model (its ``N x N`` SVD
      residual has no flat direction).

    ``raw_density_mae - pair_density_mae`` is, to leading order, the part
    of the density error confined to the gauge degree of freedom.
    """
    adapter.eval()
    hamiltonian = KitaevChainHamiltonian(n_sites=N, hopping=T, pairing=DELTA)
    mu = MU_GRID[np.abs(MU_GRID) < TRANSITION]
    with torch.no_grad():
        psi = (
            adapter(torch.tensor(mu[:, None], dtype=torch.float32, device=device))[1]
            .detach()
            .cpu()
            .numpy()
        )
    psi = psi / np.linalg.norm(psi, axis=1, keepdims=True)

    raw_gap = np.empty((mu.size, N))
    pair_gap = np.empty((mu.size, N))
    for i, m in enumerate(mu):
        w, vecs = np.linalg.eigh(hamiltonian.build(float(m)))
        near = vecs[:, np.argsort(np.abs(w))[:2]]  # exact 2D near-zero basis
        p_exact = (near[:N, :] ** 2).sum(axis=1)  # projector diagonal, particle
        ref = vecs[:, N] ** 2  # one eigh representative, all 2N sectors

        u1 = psi[i]
        u2 = np.concatenate([psi[i, N:], psi[i, :N]])  # Xi @ psi
        u2 = u2 - (u1 @ u2) * u1
        nrm = np.linalg.norm(u2)
        u2 = u2 / nrm if nrm > 1e-9 else u2
        p_pred = u1[:N] ** 2 + u2[:N] ** 2

        pair_gap[i] = np.abs(p_pred - p_exact)
        raw_gap[i] = np.abs(psi[i, :N] ** 2 - ref[:N]) + np.abs(
            psi[i, N:] ** 2 - ref[N:]
        )
    return float(np.mean(pair_gap)), float(np.mean(raw_gap))


def run_one(
    recipe: Recipe,
    seed: int,
    *,
    smoke: bool,
    session: Session,
    two_phase: TwoPhaseConfig | None = None,
    light: bool = False,
    artefact_tag: str | None = None,
) -> dict:
    """Train one (model, seed) under the shared config; return a metric row.

    Args:
        recipe: The model recipe to run.
        seed: Random seed for this run.
        smoke: Use the tiny wiring-check budget.
        session: The run's :class:`sesh.Session`.
        two_phase: Override the two-phase schedule. Defaults to the smoke
            or full schedule per ``smoke``. Used by the budget sweep to
            vary only the AdamW epoch count.
        light: Skip the per-run figures and checkpoint entirely. Off by
            default; nothing in this script sets it, but it is kept as an
            escape hatch for callers that only want the metric row.
        artefact_tag: Extra path segment between the model name and
            ``seed_<seed>`` for the checkpoint and figure directories, so
            repeated (model, seed) runs at different settings do not
            collide. The budget sweep passes ``"adam_<budget>"``; the
            default (``None``) keeps the plain ``<model>/seed_<seed>``
            layout the interpretability notebook expects.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    accelerator = Accelerator()
    device = accelerator.device

    H_base, H_mu_diag, Xi = (
        x.to(device) for x in _build_kitaev_operators(N, hopping=T, pairing=DELTA)
    )

    two_phase = two_phase or _two_phase(smoke)
    base_config = replace(TWO_PHASE_BASE, restore_best=recipe.restore_best)
    model = recipe.build_model()
    loss_fn = recipe.build_loss(two_phase)
    adapt = recipe.build_adapt
    loaders = recipe.build_loaders(smoke)

    probe = BdGEvaluationProbe(
        n_sites=N,
        hopping=T,
        pairing=DELTA,
        mu_grid=MU_GRID,
        every=2 if smoke else 50,
        session=session,
        adapt=adapt,  # identity for the dual head; a real wrapper otherwise
    )

    t0 = time.perf_counter()
    trained, history = run_two_phase(
        session=session,
        accelerator=accelerator,
        model=model,
        loss_fn=loss_fn,
        train_loader=loaders.train,
        H_base=H_base,
        H_mu_diag=H_mu_diag,
        Xi=Xi,
        two_phase=two_phase,
        base_config=base_config,
        callbacks=[*loaders.sampling_callbacks, probe],
        val_loader=loaders.val,
        unlabeled_loader=loaders.unlabeled,
        lbfgs_train_loader=loaders.lbfgs_train,
        lbfgs_unlabeled_loader=loaders.lbfgs_unlabeled,
        lbfgs_callbacks=[probe],
        **recipe.run_kwargs,
    )
    wall = time.perf_counter() - t0

    adapter = adapt(trained).to(device)

    # Provenance: the returned model is the final-epoch state for every
    # recipe except semi_supervised (restore_best=True -> best-validation
    # state); these columns pin down which state that was and whether the
    # budget was long enough. Derived from the concatenated two-phase
    # history, so no extra trainer plumbing.
    expected_epochs = two_phase.adam_epochs + max(0, two_phase.lbfgs_epochs)
    final_epoch = len(history["train_loss"])
    val_loss_series = history.get("val_loss", [])
    best_val_epoch = (
        min(range(len(val_loss_series)), key=val_loss_series.__getitem__) + 1
        if val_loss_series
        else None
    )
    pair_density_mae_topo, raw_density_mae_topo = _topo_density_maes(adapter, device)

    resid_series = history.get(f"train_{recipe.residual_key}", history["train_loss"])
    var_tail_decades = _tail_decades(resid_series)
    infidelity_tail_decades = _tail_decades(history["probe_subspace_infidelity"])
    converged = (
        var_tail_decades < CONVERGENCE_DECADE_TOL
        and infidelity_tail_decades < CONVERGENCE_DECADE_TOL
    )

    if not light:
        # Persist the trained weights so the interpretability notebook can
        # rebuild this model without retraining. One file per (model, seed)
        # -- or per (model, tag, seed) when artefact_tag is set (the budget
        # sweep). session.path() makes and returns a directory, so name the
        # file under it.
        artefact_seg = (
            (recipe.name,) if artefact_tag is None else (recipe.name, artefact_tag)
        )
        checkpoint_dir = session.path("checkpoints", *artefact_seg)
        torch.save(trained.state_dict(), checkpoint_dir / f"seed_{seed}.pt")

        hamiltonian = KitaevChainHamiltonian(n_sites=N, hopping=T, pairing=DELTA)
        floor_value = (
            fsm_convergence_floor(hamiltonian, MU_GRID, factor=recipe.fsm_floor_factor)
            if recipe.fsm_floor_factor is not None
            else None
        )
        save_run_figures(
            adapter=adapter,
            history=history,
            hamiltonian=hamiltonian,
            mu_grid=MU_GRID,
            out_dir=session.path("figures", *artefact_seg, f"seed_{seed}"),
            model_label=recipe.plot_label,
            component_keys=recipe.component_keys,
            weight_key=recipe.weight_key,
            split_epoch=two_phase.adam_epochs,
            floor_value=floor_value,
            structural_fold=recipe.structural_fold,
            device=device,
        )

    row = {
        "model": recipe.name,
        "seed": seed,
        "e_mae_full": history["probe_e_mae"][-1],
        "e_mae_topological": history["probe_e_mae_topological"][-1],
        "e_mae_trivial": history["probe_e_mae_trivial"][-1],
        "subspace_infidelity_mean": history["probe_subspace_infidelity"][-1],
        "subspace_infidelity_max": history["probe_subspace_infidelity_max"][-1],
        "edge_mae": history["probe_edge_mae"][-1],
        # topological-phase density error: gauge-invariant (rho_n) vs raw
        # (2N sectors). The gap raw - pair is the gauge-confined part; it is
        # expected to persist for the Nambu models across budgets and
        # vanish for the chiral model. See _topo_density_maes.
        "pair_density_mae_topo": pair_density_mae_topo,
        "raw_density_mae_topo": raw_density_mae_topo,
        "psi_norm": history["probe_psi_norm"][-1],
        "reflection_residual": _reflection_residual(adapter, device),
        "train_loss_final": history["train_loss"][-1],
        "epochs": final_epoch,
        "wall_seconds": round(wall, 1),
        # provenance / convergence
        "restore_best": recipe.restore_best,
        "completed": final_epoch >= expected_epochs,
        "final_epoch": final_epoch,
        "best_val_epoch": best_val_epoch,
        "best_val_loss": min(val_loss_series) if val_loss_series else None,
        "state_dict_sha256": _state_dict_sha256(trained),
        "var_tail_decades": round(var_tail_decades, 3),
        "infidelity_tail_decades": round(infidelity_tail_decades, 3),
        "converged": converged,
    }

    session.log_metrics(
        {k: float(v) for k, v in row.items() if isinstance(v, (int, float))},
        step=seed,
    )
    session.info(
        f"[{recipe.name} seed {seed}] "
        f"E MAE topo {row['e_mae_topological']:.3e} / triv "
        f"{row['e_mae_trivial']:.3e} | infid {row['subspace_infidelity_mean']:.3e} "
        f"| edge {row['edge_mae']:.3e} | refl {row['reflection_residual']:.2e} "
        f"| converged={row['converged']} "
        f"(var {row['var_tail_decades']:+.2f} dec, "
        f"infid {row['infidelity_tail_decades']:+.2f} dec)"
    )
    return row


def run_budget_sweep(
    session: Session,
    *,
    models: list[str],
    seeds: list[int],
    budgets: tuple[int, ...],
    smoke: bool,
) -> list[dict]:
    """Train each (model, budget, seed) with a fixed L-BFGS tail.

    Only the AdamW epoch count varies. The cosine schedule's ``T_max``
    tracks it (see :func:`run_two_phase`), so every point is a complete
    schedule rather than a truncated one. Each (model, budget, seed) run
    writes the same per-run figures and checkpoint as the main comparison,
    under ``figures/<model>/adam_<budget>/seed_<seed>/`` (and the matching
    ``checkpoints/`` path), so the loss curve and probe history at every
    budget can be inspected directly.

    Args:
        session: The run's :class:`sesh.Session`.
        models: Model recipe names to sweep.
        seeds: Seeds per (model, budget).
        budgets: AdamW epoch counts to try.
        smoke: Use the tiny wiring-check L-BFGS tail and batch sizes.

    Returns:
        One metric row per (model, budget, seed), each tagged with
        ``adam_epochs``.
    """
    base = _two_phase(smoke)
    rows: list[dict] = []
    for model_name in models:
        recipe = RECIPES[model_name]
        for budget in budgets:
            two_phase = replace(base, adam_epochs=budget)
            for seed in seeds:
                row = run_one(
                    recipe,
                    seed,
                    smoke=smoke,
                    session=session,
                    two_phase=two_phase,
                    artefact_tag=f"adam_{budget}",
                )
                row["adam_epochs"] = budget
                rows.append(row)
                session.info(
                    f"[sweep {recipe.name} budget {budget} seed {seed}] "
                    f"E MAE triv {row['e_mae_trivial']:.3e} | "
                    f"infid {row['subspace_infidelity_mean']:.3e}"
                )
    return rows


def rerender_seed_wavefunctions(
    session_dir: Path,
    *,
    models: list[str],
    device: str = "cpu",
) -> list[Path]:
    """Refresh every per-seed ``wavefunctions.png`` under a finished run.

    Walks each ``figures/<model>/**/seed_*`` directory (the plain layout
    and the budget sweep's ``adam_<budget>/seed_*`` one), reloads the
    matching ``checkpoints/<...>/seed_<s>.pt`` and rewrites only the
    density figure -- the one that consumes ``sweep_wavefunction_grid`` --
    in place. The rest of the per-seed set needs the training history,
    which is not checkpointed, so it is left untouched.
    """
    hamiltonian = KitaevChainHamiltonian(n_sites=N, hopping=T, pairing=DELTA)
    two_sided = bool(MU_GRID.min() < 0)
    written: list[Path] = []
    for name in models:
        recipe = RECIPES[name]
        model_figs = session_dir / "figures" / name
        if not model_figs.is_dir():
            continue
        for seed_dir in sorted(p for p in model_figs.glob("**/seed_*") if p.is_dir()):
            rel = seed_dir.relative_to(session_dir / "figures")
            checkpoint = (session_dir / "checkpoints" / rel).with_suffix(".pt")
            if not checkpoint.exists():
                continue
            model = recipe.build_model()
            model.load_state_dict(
                torch.load(checkpoint, map_location=device, weights_only=True)
            )
            model.eval()
            written.append(
                rerender_wavefunctions(
                    adapter=recipe.build_adapt(model).to(device),
                    hamiltonian=hamiltonian,
                    out_dir=seed_dir,
                    two_sided=two_sided,
                    device=device,
                )
            )
    return written


def render_comparison_figures(
    session_dir: Path,
    *,
    models: list[str],
    device: str = "cpu",
) -> dict[str, Path]:
    """Cross-model figures from a finished run's checkpoints.

    Reloads every ``seed_*.pt`` under ``<session_dir>/checkpoints/`` and
    writes the headline comparison panel, a spectral-fan context figure
    and a per-model density waterfall into
    ``<session_dir>/figures/comparison/``. No training, so it doubles as
    the body of ``--figures-only``.

    Args:
        session_dir: A comparison run's session directory.
        models: Recipe names to include, in plot order. Every seed
            checkpoint found on disk for each is used.
        device: Device for the model forward passes.

    Returns:
        A mapping from figure name to the path it was written to.
    """
    from kitaev.xai.loading import load_seed_checkpoints

    hamiltonian = KitaevChainHamiltonian(n_sites=N, hopping=T, pairing=DELTA)
    out_dir = session_dir / "figures" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    bands = []
    spectra_repr: dict[str, object] = {}
    repr_checkpoint: dict[str, tuple[object, int]] = {}
    for name in models:
        recipe = RECIPES[name]
        checkpoints = load_seed_checkpoints(
            recipe.build_model, session_dir / "checkpoints" / name, device=device
        )
        if not checkpoints:
            print(f"skip {name}: no checkpoints under {session_dir}")
            continue
        sweeps = [
            sweep_spectrum(
                recipe.build_adapt(model).to(device),
                hamiltonian,
                MU_GRID,
                device=device,
            )
            for model in checkpoints
        ]
        bands.append(build_model_error_band(recipe.plot_label, sweeps))
        # The waterfall shows one seed; pick the one whose energy error is
        # nearest the seed-wise median so it is not an unlucky outlier.
        seed_error = np.array([float(np.mean(s.abs_error)) for s in sweeps])
        idx = int(np.argmin(np.abs(seed_error - np.median(seed_error))))
        spectra_repr[name] = sweeps[idx]
        repr_checkpoint[name] = (checkpoints[idx], idx)

    paths: dict[str, Path] = {}
    if bands:
        paths["comparison"] = out_dir / "energy_error.png"
        plot_model_comparison(bands, hopping=T, save_path=paths["comparison"])

        fan_model = "chiral" if "chiral" in spectra_repr else next(iter(spectra_repr))
        paths["spectral_fan"] = out_dir / "spectral_fan.png"
        plot_spectral_fan(
            sweep_low_spectrum(hamiltonian, MU_GRID, n_levels=4),
            hopping=T,
            predicted=spectra_repr[fan_model],  # type: ignore[arg-type]
            model_label=RECIPES[fan_model].plot_label,
            save_path=paths["spectral_fan"],
        )

    dense_mu = np.linspace(float(MU_GRID.min()), float(MU_GRID.max()), 160)
    for name in models:
        recipe = RECIPES[name]
        if name not in repr_checkpoint:
            continue
        model, idx = repr_checkpoint[name]
        wf = sweep_wavefunctions(
            recipe.build_adapt(model).to(device),
            hamiltonian,
            list(dense_mu),
            device=device,
        )
        label = f"{recipe.plot_label} (seed {idx})"
        key = f"waterfall_{name}"
        paths[key] = out_dir / f"{key}.png"
        plot_wavefunction_waterfall(
            wf, hopping=T, model_label=label, save_path=paths[key]
        )
        # The fair cross-model view: the raw particle/hole split above is
        # gauge-dependent inside the topological phase, the pair density is
        # not (see plot_pair_density_waterfall).
        pair_key = f"pair_density_{name}"
        paths[pair_key] = out_dir / f"{pair_key}.png"
        plot_pair_density_waterfall(
            wf, hopping=T, model_label=label, save_path=paths[pair_key]
        )

    return paths


def _seed_agg(rows: list[dict], key: str) -> str:
    """``median [q25, q75] max`` of ``key`` over ``rows``, for the summaries.

    Missing keys (older CSVs, budget-sweep rows) aggregate to ``nan``
    rather than raising.
    """
    a = np.asarray([r.get(key, float("nan")) for r in rows], dtype=float)
    lo, hi = (float(x) for x in np.percentile(a, [25, 75]))
    return f"{float(np.median(a)):.2e} [{lo:.2e}, {hi:.2e}] {a.max():.2e}"


_SUMMARY_COL = 40


def summarise(rows: list[dict]) -> str:
    """Per-model seed summary: median [q25, q75] max, plus a pass-rate.

    Central tendency is the median with the inter-quartile range; the
    worst seed (max) sits alongside because several models are bimodal
    over seeds and a median alone hides that. ``pass`` counts the seeds
    clearing a fixed reliability bar (``PASS_E_MAE_MAX`` on full-domain
    energy MAE and ``PASS_INFIDELITY_MAX`` on worst-mu subspace
    infidelity).
    """
    metrics = [
        "e_mae_full",
        "e_mae_topological",
        "subspace_infidelity_max",
        "pair_density_mae_topo",
        "raw_density_mae_topo",
    ]
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    sub = "median [q25, q75] max"
    lines = [
        f"{'model':<18} " + "  ".join(f"{m:>{_SUMMARY_COL}}" for m in metrics),
        f"{'':<18} " + "  ".join(f"{sub:>{_SUMMARY_COL}}" for _ in metrics),
    ]
    for model, model_rows in by_model.items():
        cells = [_seed_agg(model_rows, m) for m in metrics]
        lines.append(f"{model:<18} " + "  ".join(f"{c:>{_SUMMARY_COL}}" for c in cells))

    lines.append("")
    lines.append(
        f"{'model':<18} {'completed':>10} {'converged':>10} {'pass':>8}   "
        "worst infidelity-tail"
    )
    for model, model_rows in by_model.items():
        n = len(model_rows)
        done = sum(bool(r["completed"]) for r in model_rows)
        conv = sum(bool(r["converged"]) for r in model_rows)
        passed = sum(
            r["e_mae_full"] < PASS_E_MAE_MAX
            and r["subspace_infidelity_max"] < PASS_INFIDELITY_MAX
            for r in model_rows
        )
        worst = max(r["infidelity_tail_decades"] for r in model_rows)
        lines.append(
            f"{model:<18} {f'{done}/{n}':>10} {f'{conv}/{n}':>10} "
            f"{f'{passed}/{n}':>8}   {worst:+.2f} decades"
        )
    return "\n".join(lines)


def summarise_budget_sweep(rows: list[dict]) -> str:
    """Per (model, adam_epochs): median [q25, q75] max over seeds.

    The convergence read-outs -- ``var_tail_decades`` and the
    ``converged`` count -- are the point of the sweep: they say where a
    fixed budget can stop.
    """
    metrics = [
        "e_mae_topological",
        "subspace_infidelity_mean",
        "raw_density_mae_topo",
        "var_tail_decades",
    ]
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["adam_epochs"]), []).append(r)

    cols = "  ".join(f"{m:>{_SUMMARY_COL}}" for m in metrics)
    lines = [f"{'model':<18} {'adam_epochs':>11} {cols}   converged"]
    for (model, budget), group in sorted(groups.items()):
        cells = [_seed_agg(group, m) for m in metrics]
        conv = sum(bool(r["converged"]) for r in group)
        row = "  ".join(f"{c:>{_SUMMARY_COL}}" for c in cells)
        lines.append(f"{model:<18} {budget:>11} {row}   {conv}/{len(group)}")
    return "\n".join(lines)


def _write_csv(rows: list[dict], path: Path) -> None:
    """Write ``rows`` as a CSV at ``path`` (parent created if absent)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> list[dict]:
    """Run the comparison (default) or the AdamW-budget sweep; return the rows.

    Returns:
        One metric row per run. The CSV is written into the session
        directory by default, so successive runs never overwrite one
        another.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_ORDER),
        choices=sorted(RECIPES),
        help="subset / reorder the models; default is the full journey order",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--smoke", action="store_true", help="tiny-budget wiring check")
    parser.add_argument(
        "--budget-sweep",
        nargs="*",
        type=int,
        default=None,
        metavar="EPOCHS",
        help=(
            "run the AdamW-budget ablation instead of the main comparison. The "
            f"bare flag uses {BUDGET_SWEEP_DEFAULT}; pass integers to override. "
            "Multiplies the run count by the number of budgets, so pair it with "
            "a reduced --models / --seeds."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "CSV path; default is <session dir>/<name>.csv, so a run never "
            "overwrites an earlier one"
        ),
    )
    parser.add_argument(
        "--figures-only",
        type=Path,
        default=None,
        metavar="SESSION_DIR",
        help=(
            "skip training; re-render the cross-model comparison figures "
            "from the checkpoints already in this session directory"
        ),
    )
    args = parser.parse_args()

    if args.figures_only is not None:
        for path in rerender_seed_wavefunctions(args.figures_only, models=args.models):
            print(f"{'wavefunctions':<24} {path}")
        paths = render_comparison_figures(args.figures_only, models=args.models)
        for name, path in paths.items():
            print(f"{name:<24} {path}")
        return []

    sweep = args.budget_sweep is not None
    session = Session(
        name="four-model-comparison-budget-sweep" if sweep else "four-model-comparison",
        output_root=Path(__file__).resolve().parents[1] / "results" / "logs",
        enable_mlflow=False,
    )
    session_dir = session.path()  # session root -- the CSV lives here by default

    if sweep:
        budgets = tuple(args.budget_sweep) or (
            SMOKE_BUDGET_SWEEP if args.smoke else BUDGET_SWEEP_DEFAULT
        )
        session.info(
            f"budget sweep: models={args.models} seeds={args.seeds} "
            f"budgets={budgets} smoke={args.smoke}"
        )
        session.log_params(
            {
                "N": N,
                "t": T,
                "delta": DELTA,
                "budget_sweep": list(budgets),
                "lbfgs_epochs": _two_phase(args.smoke).lbfgs_epochs,
                "restore_best": TWO_PHASE_BASE.restore_best,
                "seeds": args.seeds,
                "smoke": args.smoke,
            }
        )
        rows = run_budget_sweep(
            session,
            models=args.models,
            seeds=args.seeds,
            budgets=budgets,
            smoke=args.smoke,
        )
        out = args.out or (session_dir / "four_model_comparison_budget_sweep.csv")
        _write_csv(rows, out)
        session.info(f"wrote {out}")
        print("\n" + summarise_budget_sweep(rows) + "\n")
        print(f"budget-sweep rows: {out}")
        return rows

    base_tp = _two_phase(args.smoke)

    def _model_tp(recipe: Recipe) -> TwoPhaseConfig:
        """Two-phase config for one model: its own AdamW budget, shared rest."""
        if args.smoke or recipe.adam_epochs is None:
            return base_tp
        return replace(base_tp, adam_epochs=recipe.adam_epochs)

    budget_by_model = {n: _model_tp(RECIPES[n]).adam_epochs for n in args.models}
    session.info(
        f"models={args.models} seeds={args.seeds} smoke={args.smoke} "
        f"(adam {budget_by_model}, lbfgs {base_tp.lbfgs_epochs})"
    )
    session.log_params(
        {
            "N": N,
            "t": T,
            "delta": DELTA,
            "adam_epochs": base_tp.adam_epochs,
            "adam_epochs_per_model": ", ".join(
                f"{k}={v}" for k, v in budget_by_model.items()
            ),
            "lbfgs_epochs": base_tp.lbfgs_epochs,
            "restore_best": TWO_PHASE_BASE.restore_best,
            "seeds": args.seeds,
            "smoke": args.smoke,
        }
    )
    log_dataset_cards(session, smoke=args.smoke)

    rows = []
    for model_name in args.models:
        recipe = RECIPES[model_name]
        model_tp = _model_tp(recipe)
        log_model_card(
            session,
            recipe,
            args.seeds,
            smoke=args.smoke,
            adam_epochs=model_tp.adam_epochs,
        )
        for seed in args.seeds:
            rows.append(
                run_one(
                    recipe,
                    seed,
                    smoke=args.smoke,
                    session=session,
                    two_phase=model_tp,
                )
            )

    out = args.out or (session_dir / "four_model_comparison.csv")
    _write_csv(rows, out)
    session.info(f"wrote {out}")

    figure_paths = render_comparison_figures(session_dir, models=args.models)
    for name, path in figure_paths.items():
        session.info(f"comparison figure {name}: {path}")

    print("\n" + summarise(rows) + "\n")
    print(f"per-run rows: {out}")
    return rows


if __name__ == "__main__":
    main()
