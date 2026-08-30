"""System-size ladder: the gauge-confined density error vs chain length N.

Trains ``chiral`` (SirenPINNChiral + ChiralFSMLoss) and ``structural_nambu``
(SirenPINNNambuFolded + NambuFSMLoss) at several ``N`` and reports, per
(N, model, seed):

- ``e_mae_topological``            energy MAE over ``|mu| < 2t`` vs eigh
- ``subspace_infidelity_{mean,max}`` 1 - ||P_M psi_pred|| over the near-zero pair
- ``pair_density_mae_topo``        error of the gauge-invariant projector
                                   diagonal ``(P_M)_{nn}`` (basis-free)
- ``raw_density_mae_topo``         error of a single representative's per-site
                                   density vs one eigh eigenvector
- ``gauge_gap = raw - pair``       the part of the density error confined to
                                   the internal gauge of the degenerate
                                   subspace

The prediction (``docs/markdown/derivations/under-determination-and-n-scaling.md``):
``gauge_gap`` stays ~0 for ``chiral`` at every ``N`` (the smallest singular
value of ``h`` is simple, ``O(t)`` gap, N-independent) and stays large / does
not shrink for ``structural_nambu`` (the selecting signal is ``O(lambda_1^2)
~ exp(-2N/xi)``). Budgets are per model, matched to the plateau each showed
in ``four_model_comparison.py --budget-sweep``; the point is the trend in
``gauge_gap``, not ultimate precision, so they are deliberately modest.

    python experiments/chiral_n_sweep.py
    python experiments/chiral_n_sweep.py --n-values 10 20 40 --seeds 0 1 2
    python experiments/chiral_n_sweep.py --figures-only results/logs/<session>
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from sesh import Session

from kitaev.analytical import KitaevChainHamiltonian
from kitaev.data.sampling_region import SamplingRegion
from kitaev.models import (
    ChiralToBdGAdapter,
    RayleighEnergyAdapter,
    SirenPINNChiral,
    SirenPINNNambuFolded,
)
from kitaev.training import BdGEvaluationProbe, TwoPhaseConfig, run_two_phase
from kitaev.training.config import TrainerConfig
from kitaev.training.loss import ChiralFSMLoss, NambuFSMLoss
from kitaev.training.sampling import SamplingConfig, build_sampling
from kitaev.training.trainer import _build_kitaev_operators
from kitaev.visualisation import (
    fsm_convergence_floor,
    rerender_wavefunctions,
    save_run_figures,
)

T = 1.0
DELTA = 0.5
TRANSITION = 2.0 * T
MU_GRID = np.linspace(-4.0, 4.0, 240)

# The half-domain mixture from four_model_comparison.py; N-independent (mu
# ranges only), so it is reused verbatim for every rung of the ladder.
HALF_REGIONS = (
    SamplingRegion(low=0.01, high=4.0, weight=1.0),
    SamplingRegion(low=1.7, high=2.6, weight=1.5),
    SamplingRegion(low=2.0, high=4.0, weight=0.5),
)
HALF_BATCH = 1024
STEPS_PER_EPOCH = 8

# Per-model plot settings for save_run_figures, matching the recipes in
# four_model_comparison.py (both models here train on the folded half
# domain, so structural_fold is True for both).
PLOT_PARAMS: dict[str, dict[str, Any]] = {
    "chiral": {
        "model_label": "chiral PINN",
        "component_keys": ("fsm", "var", "lam_mean"),
        "fsm_floor_factor": 2.0,  # loss_fsm = mean||hv||^2 + mean||h^T u||^2
    },
    "structural_nambu": {
        "model_label": "structural Nambu",
        "component_keys": ("fsm", "var"),
        "fsm_floor_factor": 1.0,
    },
}


def evaluate(
    adapter: Any, n_sites: int, device: str
) -> tuple[float, float, float, float, float]:
    """One eigh sweep -> (e_mae_topo, infid_mean, infid_max, pair_mae, raw_mae).

    ``pair_mae`` uses the projector diagonal of the 2-D near-zero subspace
    (gauge-invariant); ``raw_mae`` compares a single predicted vector's
    per-site density against one eigh eigenvector (gauge-dependent). Both
    are averaged over ``|mu| < 2t``.
    """
    adapter.eval()
    ham = KitaevChainHamiltonian(n_sites=n_sites, hopping=T, pairing=DELTA)
    with torch.no_grad():
        e_t, psi_t = adapter(
            torch.tensor(MU_GRID[:, None], dtype=torch.float32, device=device)
        )
    e_pred = np.abs(e_t.detach().cpu().numpy().reshape(-1))
    psi = psi_t.detach().cpu().numpy()
    psi = psi / np.linalg.norm(psi, axis=1, keepdims=True)

    topo = np.abs(MU_GRID) < TRANSITION
    e_exact = np.empty(MU_GRID.size)
    infid = np.empty(MU_GRID.size)
    pair_err: list[float] = []
    raw_err: list[float] = []
    for i, mu in enumerate(MU_GRID):
        w, vecs = np.linalg.eigh(ham.build(float(mu)))
        order = np.argsort(np.abs(w))
        e_exact[i] = np.abs(w[order[0]])
        near = vecs[:, order[:2]]
        infid[i] = 1.0 - np.linalg.norm(near.T @ psi[i])
        if not topo[i]:
            continue
        p_exact = (near[:n_sites, :] ** 2).sum(axis=1)
        ref = vecs[:, n_sites] ** 2
        u1 = psi[i]
        u2 = np.concatenate([psi[i, n_sites:], psi[i, :n_sites]])  # Xi @ psi
        u2 = u2 - (u1 @ u2) * u1
        nrm = np.linalg.norm(u2)
        u2 = u2 / nrm if nrm > 1e-9 else u2
        p_pred = u1[:n_sites] ** 2 + u2[:n_sites] ** 2
        pair_err.append(float(np.abs(p_pred - p_exact).mean()))
        raw_err.append(
            float(
                (
                    np.abs(psi[i, :n_sites] ** 2 - ref[:n_sites])
                    + np.abs(psi[i, n_sites:] ** 2 - ref[n_sites:])
                ).mean()
            )
        )
    return (
        float(np.abs(e_pred - e_exact)[topo].mean()),
        float(infid.mean()),
        float(infid.max()),
        float(np.mean(pair_err)),
        float(np.mean(raw_err)),
    )


def _loaders() -> tuple[Any, Any, Any]:
    train, _ = build_sampling(
        SamplingConfig(
            mode="infinite", batch_size=HALF_BATCH, steps_per_epoch=STEPS_PER_EPOCH
        ),
        HALF_REGIONS,
    )
    val, _ = build_sampling(
        SamplingConfig(mode="frozen", batch_size=HALF_BATCH, total_samples=HALF_BATCH),
        HALF_REGIONS,
    )
    lbfgs, _ = build_sampling(
        SamplingConfig(
            mode="frozen", batch_size=2 * HALF_BATCH, total_samples=2 * HALF_BATCH
        ),
        HALF_REGIONS,
    )
    return train, val, lbfgs


def train_one(
    kind: str, n_sites: int, seed: int, budget: int, session: Session
) -> dict[str, Any]:
    """Train one (model, N, seed) under a two-phase budget; return a metric row."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    acc = Accelerator()
    device = acc.device
    H_base, H_mu_diag, Xi = (
        x.to(device) for x in _build_kitaev_operators(n_sites, hopping=T, pairing=DELTA)
    )

    if kind == "chiral":
        model: torch.nn.Module = SirenPINNChiral(
            n_sites=n_sites, hidden_features=64, hidden_layers=2, input_scale=4.0
        )
        loss_fn: Any = ChiralFSMLoss(n_sites=n_sites, hopping=T, pairing=DELTA)

        def adapt(m: torch.nn.Module) -> Any:
            return ChiralToBdGAdapter(m, hopping=T, pairing=DELTA)
    else:
        model = SirenPINNNambuFolded(
            n_sites=2 * n_sites, hidden_features=64, hidden_layers=2, input_scale=4.0
        )
        loss_fn = NambuFSMLoss()

        def adapt(m: torch.nn.Module) -> Any:
            return RayleighEnergyAdapter(m, n_sites=n_sites, hopping=T, pairing=DELTA)

    train_loader, val_loader, lbfgs_loader = _loaders()
    two_phase = TwoPhaseConfig(
        adam_epochs=budget,
        adam_lr=8e-4,
        adam_weight_decay=1e-6,
        lbfgs_epochs=300,
        lbfgs_max_iter=25,
        lbfgs_history_size=20,
        lbfgs_line_search_fn="strong_wolfe",
    )
    base_config = TrainerConfig(
        epochs=1, print_freq=2000, patience=None, grad_clip_norm=1.0
    )
    probe = BdGEvaluationProbe(
        n_sites=n_sites,
        hopping=T,
        pairing=DELTA,
        mu_grid=MU_GRID,
        every=50,
        session=session,
        adapt=adapt,
    )

    session.info(f"--- {kind} N={n_sites} seed={seed} budget={budget} ---")
    t0 = time.perf_counter()
    trained, history = run_two_phase(
        session=session,
        accelerator=acc,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        H_base=H_base,
        H_mu_diag=H_mu_diag,
        Xi=Xi,
        two_phase=two_phase,
        base_config=base_config,
        callbacks=[probe],
        val_loader=val_loader,
        lbfgs_train_loader=lbfgs_loader,
        lbfgs_callbacks=[probe],
    )
    wall = time.perf_counter() - t0

    adapter = adapt(trained).to(device)
    e_mae, infid_mean, infid_max, pair_mae, raw_mae = evaluate(adapter, n_sites, device)

    # Same per-run artefacts as four_model_comparison: the trained weights
    # plus the standard figure set (loss_history, probe_history,
    # energy_sweep, eigenvector_agreement, wavefunctions, mu_reflection),
    # keyed by N so nothing collides across the ladder.
    plot = PLOT_PARAMS[kind]
    hamiltonian = KitaevChainHamiltonian(n_sites=n_sites, hopping=T, pairing=DELTA)
    ckpt_dir = session.path("checkpoints", kind, f"N{n_sites}")
    torch.save(trained.state_dict(), ckpt_dir / f"seed_{seed}.pt")
    save_run_figures(
        adapter=adapter,
        history=history,
        hamiltonian=hamiltonian,
        mu_grid=MU_GRID,
        out_dir=session.path("figures", kind, f"N{n_sites}", f"seed_{seed}"),
        model_label=f"{plot['model_label']} (N={n_sites})",
        component_keys=plot["component_keys"],
        weight_key=None,
        split_epoch=two_phase.adam_epochs,
        floor_value=fsm_convergence_floor(
            hamiltonian, MU_GRID, factor=plot["fsm_floor_factor"]
        ),
        structural_fold=True,
        device=device,
    )
    row = {
        "n_sites": n_sites,
        "model": kind,
        "seed": seed,
        "adam_epochs": budget,
        "e_mae_topological": e_mae,
        "subspace_infidelity_mean": infid_mean,
        "subspace_infidelity_max": infid_max,
        "pair_density_mae_topo": pair_mae,
        "raw_density_mae_topo": raw_mae,
        "gauge_gap": raw_mae - pair_mae,
        "wall_seconds": round(wall, 1),
    }
    session.info(
        f"[{kind} N={n_sites} seed={seed}] E MAE topo {e_mae:.3e} | "
        f"infid {infid_mean:.3e} | pair {pair_mae:.3e} | raw {raw_mae:.3e} | "
        f"gap {raw_mae - pair_mae:.3e}"
    )
    return row


def summarise(rows: list[dict[str, Any]]) -> str:
    """Median over seeds of the headline columns, per (model, N)."""
    keys = (
        "e_mae_topological",
        "subspace_infidelity_max",
        "pair_density_mae_topo",
        "raw_density_mae_topo",
        "gauge_gap",
    )
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["n_sites"]), []).append(r)

    head = f"{'model':<18}{'N':>5}  " + "  ".join(f"{k:>24}" for k in keys)
    lines = [head, "-" * len(head)]
    for (model, n_sites), grp in sorted(groups.items()):
        cells = [f"{statistics.median(g[k] for g in grp):.3e}" for k in keys]
        lines.append(
            f"{model:<18}{n_sites:>5}  " + "  ".join(f"{c:>24}" for c in cells)
        )
    return "\n".join(lines)


def _rebuild_adapter(kind: str, n_sites: int, checkpoint: Path) -> Any:
    """Load one ``seed_*.pt`` for ``(kind, n_sites)`` into its (E, psi) adapter."""
    if kind == "chiral":
        model: torch.nn.Module = SirenPINNChiral(
            n_sites=n_sites, hidden_features=64, hidden_layers=2, input_scale=4.0
        )
    else:
        model = SirenPINNNambuFolded(
            n_sites=2 * n_sites, hidden_features=64, hidden_layers=2, input_scale=4.0
        )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    if kind == "chiral":
        return ChiralToBdGAdapter(model, hopping=T, pairing=DELTA)
    return RayleighEnergyAdapter(model, n_sites=n_sites, hopping=T, pairing=DELTA)


def render_figures_only(session_dir: Path) -> list[Path]:
    """Re-render the per-seed ``wavefunctions.png`` from a finished sweep's checkpoints.

    Only the density figure is refreshed: it is the one that consumes
    ``sweep_wavefunction_grid``, and the training history the rest of the
    standard set needs is not checkpointed. Every
    ``figures/<kind>/N<n>/seed_<s>/`` directory with a matching checkpoint
    is rewritten in place.
    """
    two_sided = bool(MU_GRID.min() < 0)
    written: list[Path] = []
    for kind in ("chiral", "structural_nambu"):
        for ckpt in sorted((session_dir / "checkpoints" / kind).glob("N*/seed_*.pt")):
            n_sites = int(ckpt.parent.name[1:])
            out_dir = session_dir / "figures" / kind / ckpt.parent.name / ckpt.stem
            if not out_dir.is_dir():
                continue
            adapter = _rebuild_adapter(kind, n_sites, ckpt)
            hamiltonian = KitaevChainHamiltonian(
                n_sites=n_sites, hopping=T, pairing=DELTA
            )
            written.append(
                rerender_wavefunctions(
                    adapter=adapter,
                    hamiltonian=hamiltonian,
                    out_dir=out_dir,
                    two_sided=two_sided,
                )
            )
    return written


def main() -> list[dict[str, Any]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-only",
        type=Path,
        default=None,
        metavar="SESSION_DIR",
        help=(
            "skip training; re-render just the per-seed wavefunctions.png in "
            "this session directory from its checkpoints"
        ),
    )
    parser.add_argument("--n-values", nargs="+", type=int, default=[10, 20, 40])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--chiral-epochs",
        type=int,
        default=9000,
        help="AdamW budget for the chiral model (pilot plateau).",
    )
    parser.add_argument(
        "--nambu-epochs",
        type=int,
        default=6000,
        help="AdamW budget for structural_nambu (energy/subspace converge early).",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.figures_only is not None:
        written = render_figures_only(args.figures_only)
        for path in written:
            print(f"wavefunctions  {path}")
        print(f"re-rendered {len(written)} density figures")
        return []

    session = Session(
        name="chiral-n-sweep",
        output_root=Path(__file__).resolve().parents[1] / "results" / "logs",
        enable_mlflow=False,
    )
    session.info(
        f"N values={args.n_values} seeds={args.seeds} "
        f"chiral_epochs={args.chiral_epochs} nambu_epochs={args.nambu_epochs}"
    )
    session.log_params(
        {
            "t": T,
            "delta": DELTA,
            "n_values": args.n_values,
            "seeds": args.seeds,
            "chiral_epochs": args.chiral_epochs,
            "nambu_epochs": args.nambu_epochs,
            "lbfgs_epochs": 300,
        }
    )

    budget = {"chiral": args.chiral_epochs, "structural_nambu": args.nambu_epochs}
    rows: list[dict[str, Any]] = []
    for n_sites in args.n_values:
        for kind in ("chiral", "structural_nambu"):
            for seed in args.seeds:
                rows.append(train_one(kind, n_sites, seed, budget[kind], session))

    out = args.out or (session.path() / "chiral_n_sweep.csv")
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    session.info(f"wrote {out}")

    print("\n" + summarise(rows) + "\n")
    print(f"n-sweep rows: {out}")
    return rows


if __name__ == "__main__":
    main()
