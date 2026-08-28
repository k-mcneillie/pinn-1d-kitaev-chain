"""Matched-conditions, multi-seed comparison of the four Kitaev-chain models.

Every model is trained under ONE shared configuration -- identical
``SamplingRegion`` mixture (mirrored for the two-sided models), identical
two-phase budget (AdamW + L-BFGS), identical ``mu`` grid -- and scored with
the shared :class:`BdGEvaluationProbe` metric set plus a mu-reflection
residual.

All configuration is recorded exhaustively: per-region sample counts (per
batch / per epoch / over the AdamW phase), the frozen validation and L-BFGS
pools, the evaluation grid, the full optimiser spec, and per-model
architecture (layer list, parameter counts, buffers) go out as ``sesh``
model / dataset cards; per-run outcomes go via ``log_metrics``; the six
standard figures per (model, seed) are rendered by
:func:`kitaev.visualisation.save_run_figures`; a tidy CSV and a per-model
mean +/- std summary are also written.

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

Usage:
    python experiments/four_model_comparison.py --smoke
    python experiments/four_model_comparison.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
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
from kitaev.visualisation import save_run_figures

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
TWO_PHASE = TwoPhaseConfig(
    adam_epochs=3000,
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
TWO_PHASE_BASE = TrainerConfig(
    epochs=1, print_freq=1000, patience=None, grad_clip_norm=1.0
)

# One region mixture, dense on the transition shoulders and the deep
# interior. HALF is for the folded models (train on [0, 4t]); FULL is its
# mirror for the two-sided models, at 2x the batch to match density.
HALF_REGIONS = (
    SamplingRegion(low=0.05, high=4.0, weight=1.0),
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


def optimiser_metadata(smoke: bool) -> dict[str, Any]:
    """The complete two-phase optimiser spec (see run_two_phase)."""
    tp = _two_phase(smoke)
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
        "checkpoint_selection": "lowest validation loss over the run (frozen val pool)",
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
        plot_label="dual-head PINN",
        component_keys=("e", "psi", "res", "ph"),
        weight_key="physics_wt",
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
    session: Session, recipe: Recipe, seeds: list[int], *, smoke: bool
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
            "optimiser": optimiser_metadata(smoke),
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


def _reflection_residual(adapter, device) -> float:
    """max_mu | |E(mu)| - |E(-mu)| | over [0, 4t] -- 0 iff mu-parity is exact."""
    adapter.eval()
    half = np.linspace(0.0, 4.0, 200)[:, None]
    with torch.no_grad():
        pos = torch.tensor(half, dtype=torch.float32, device=device)
        e_pos = np.abs(adapter(pos)[0].detach().cpu().numpy().ravel())
        e_neg = np.abs(adapter(-pos)[0].detach().cpu().numpy().ravel())
    return float(np.abs(e_pos - e_neg).max())


def run_one(recipe: Recipe, seed: int, *, smoke: bool, session: Session) -> dict:
    """Train one (model, seed) under the shared config; return a metric row."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    accelerator = Accelerator()
    device = accelerator.device

    H_base, H_mu_diag, Xi = (
        x.to(device) for x in _build_kitaev_operators(N, hopping=T, pairing=DELTA)
    )

    two_phase = _two_phase(smoke)
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
        base_config=TWO_PHASE_BASE,
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

    save_run_figures(
        adapter=adapter,
        history=history,
        hamiltonian=KitaevChainHamiltonian(n_sites=N, hopping=T, pairing=DELTA),
        mu_grid=MU_GRID,
        out_dir=session.path("figures", recipe.name, f"seed_{seed}"),
        model_label=recipe.plot_label,
        component_keys=recipe.component_keys,
        weight_key=recipe.weight_key,
        split_epoch=two_phase.adam_epochs,
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
        "psi_norm": history["probe_psi_norm"][-1],
        "reflection_residual": _reflection_residual(adapter, device),
        "train_loss_final": history["train_loss"][-1],
        "epochs": len(history["train_loss"]),
        "wall_seconds": round(wall, 1),
    }

    session.log_metrics(
        {k: float(v) for k, v in row.items() if isinstance(v, (int, float))},
        step=seed,
    )
    session.info(
        f"[{recipe.name} seed {seed}] "
        f"E MAE topo {row['e_mae_topological']:.3e} / triv "
        f"{row['e_mae_trivial']:.3e} | infid {row['subspace_infidelity_mean']:.3e} "
        f"| edge {row['edge_mae']:.3e} | refl {row['reflection_residual']:.2e}"
    )
    return row


def summarise(rows: list[dict]) -> str:
    """Per-model mean +/- population std over seeds, as an aligned table."""
    metrics = [
        "e_mae_topological",
        "e_mae_trivial",
        "e_mae_full",
        "subspace_infidelity_mean",
        "subspace_infidelity_max",
        "edge_mae",
        "reflection_residual",
    ]
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    lines = [f"{'model':<18} " + "  ".join(f"{m:>26}" for m in metrics)]
    for model, model_rows in by_model.items():
        cells = []
        for m in metrics:
            vals = [r[m] for r in model_rows]
            mu = mean(vals)
            sd = pstdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{mu:.3e}+/-{sd:.1e}")
        lines.append(f"{model:<18} " + "  ".join(f"{c:>26}" for c in cells))
    return "\n".join(lines)


def main() -> None:
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
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "four_model_comparison.csv",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    session = Session(
        name="four-model-comparison",
        output_root=Path(__file__).resolve().parents[1] / "results" / "logs",
        enable_mlflow=False,
    )
    tp = _two_phase(args.smoke)
    session.info(
        f"models={args.models} seeds={args.seeds} smoke={args.smoke} "
        f"(adam {tp.adam_epochs}, lbfgs {tp.lbfgs_epochs})"
    )
    session.log_params(
        {
            "N": N,
            "t": T,
            "delta": DELTA,
            "adam_epochs": tp.adam_epochs,
            "lbfgs_epochs": tp.lbfgs_epochs,
            "seeds": args.seeds,
            "smoke": args.smoke,
        }
    )
    log_dataset_cards(session, smoke=args.smoke)

    rows: list[dict] = []
    for model_name in args.models:
        recipe = RECIPES[model_name]
        log_model_card(session, recipe, args.seeds, smoke=args.smoke)
        for seed in args.seeds:
            rows.append(run_one(recipe, seed, smoke=args.smoke, session=session))

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    session.info(f"wrote {args.out}")
    print("\n" + summarise(rows) + "\n")
    print(f"per-run rows: {args.out}")


if __name__ == "__main__":
    main()
