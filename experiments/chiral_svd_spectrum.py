"""Full-SVD chiral PINN: the whole BdG spectrum and the individual Majoranas.

Trains ``SirenPINNChiralFull`` + ``ChiralSVDLoss`` on the 1-D Kitaev chain
(``N = 20``) and reports, per seed:

- the shared lowest-pair columns of ``four_model_comparison.csv``
  (``e_mae_{full,topological,trivial}``, ``subspace_infidelity_{mean,max}``,
  ``edge_mae``, ``reflection_residual``, ``psi_norm``), computed through
  ``ChiralFullToBdGAdapter`` with the same ``BdGEvaluationProbe``,
  ``MU_GRID`` and Hamiltonian, so a row drops straight into that table; and
- whole-spectrum columns only this model can produce:
  ``spectrum_mae``, ``spectrum_mae_bulk``, ``spectrum_mae_nearzero``,
  ``spectrum_max_abs`` and ``det_sign_agreement`` (fraction of the grid on
  which the frame's ``sign(det h)`` matches the analytic value).

Per-seed figures: the ``2N`` predicted spectrum over exact diagonalisation,
the reconstructed left/right Majorana densities across the topological
phase, and the ``sign(det h)`` invariant with ``sigma_min(mu)``. With two or
more seeds a frame-reproducibility panel contrasts the gauge-invariant
per-triple density dispersion (expected ~0) with the gauge-dependent frame
dispersion.

    python experiments/chiral_svd_spectrum.py
    python experiments/chiral_svd_spectrum.py --seeds 0 1 2 --epochs 3000
    python experiments/chiral_svd_spectrum.py --seeds 0 --gauge-weight 1e-3
    python experiments/chiral_svd_spectrum.py --smoke
    python experiments/chiral_svd_spectrum.py --figures-only results/logs/<session>
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from sesh import Session

from kitaev.analytical import (
    KitaevChainHamiltonian,
    chiral_block,
    reconstruct_bdg_eigenvectors,
    resolve_svd_sign,
)
from kitaev.data.sampling_region import SamplingRegion
from kitaev.models import ChiralFullToBdGAdapter, SirenPINNChiralFull
from kitaev.training import (
    BdGEvaluationProbe,
    SpectrumEvaluationProbe,
    TwoPhaseConfig,
    run_two_phase,
)
from kitaev.training.config import TrainerConfig
from kitaev.training.loss import ChiralSVDLoss
from kitaev.training.sampling import SamplingConfig, build_sampling
from kitaev.training.trainer import _build_kitaev_operators
from kitaev.visualisation import mark_transition, save_run_figures, use_house_style

N_SITES = 20
T = 1.0
DELTA = 0.5
TRANSITION = 2.0 * T
MU_GRID = np.linspace(-4.0, 4.0, 240)

# The half-domain collocation mixture from four_model_comparison.py.
HALF_REGIONS = (
    SamplingRegion(low=0.05, high=4.0, weight=1.0),
    SamplingRegion(low=1.7, high=2.6, weight=1.5),
    SamplingRegion(low=2.0, high=4.0, weight=0.5),
)
HALF_BATCH = 1024
STEPS_PER_EPOCH = 8

# mu values (topological phase) for the Majorana-density figure.
MAJORANA_MUS = (0.3, 0.9, 1.5, 1.9)


def build_chiral_svd_model(n_sites: int = N_SITES) -> SirenPINNChiralFull:
    """The single-place model factory, reused by training and reload."""
    return SirenPINNChiralFull(
        n_sites=n_sites,
        hidden_features=64,
        hidden_layers=2,
        input_scale=4.0,
        hopping=T,
        pairing=DELTA,
    )


def _adapter(model: torch.nn.Module) -> ChiralFullToBdGAdapter:
    return ChiralFullToBdGAdapter(model, hopping=T, pairing=DELTA)


def _spectrum_of(model: torch.nn.Module, mu_tensor: torch.Tensor) -> torch.Tensor:
    return _adapter(model).full_spectrum(mu_tensor)


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


def _reflection_residual(adapter: torch.nn.Module, device: str) -> float:
    """max_mu | |E(mu)| - |E(-mu)| | over [0, 4t] -- 0 iff mu-parity is exact."""
    adapter.eval()
    half = np.linspace(0.0, 4.0, 200)[:, None]
    with torch.no_grad():
        pos = torch.tensor(half, dtype=torch.float32, device=device)
        e_pos = np.abs(adapter(pos)[0].detach().cpu().numpy().ravel())
        e_neg = np.abs(adapter(-pos)[0].detach().cpu().numpy().ravel())
    return float(np.abs(e_pos - e_neg).max())


def _lowest_pair_metrics(history: Any, adapter: torch.nn.Module, device: str) -> dict:
    """Shared columns, from the BdGEvaluationProbe history plus the reflection."""
    return {
        "e_mae_full": history["probe_e_mae"][-1],
        "e_mae_topological": history["probe_e_mae_topological"][-1],
        "e_mae_trivial": history["probe_e_mae_trivial"][-1],
        "subspace_infidelity_mean": history["probe_subspace_infidelity"][-1],
        "subspace_infidelity_max": history["probe_subspace_infidelity_max"][-1],
        "edge_mae": history["probe_edge_mae"][-1],
        "reflection_residual": _reflection_residual(adapter, device),
        "psi_norm": history["probe_psi_norm"][-1],
    }


def _spectrum_metrics(history: Any, adapter: ChiralFullToBdGAdapter) -> dict:
    """Whole-spectrum columns, from the SpectrumEvaluationProbe history."""
    x = torch.tensor(MU_GRID[:, None], dtype=torch.float32)
    with torch.no_grad():
        det_pred = adapter.det_sign(x).cpu().numpy()
    det_exact = np.array(
        [np.sign(np.linalg.det(chiral_block(mu, N_SITES, T, DELTA))) for mu in MU_GRID]
    )
    return {
        "spectrum_mae": history["probe_spectrum_mae"][-1],
        "spectrum_mae_bulk": history["probe_spectrum_mae_bulk"][-1],
        "spectrum_mae_nearzero": history["probe_spectrum_mae_nearzero"][-1],
        "spectrum_max_abs": history["probe_spectrum_max_abs"][-1],
        "det_sign_agreement": float(np.mean(det_pred == det_exact)),
    }


def _plot_full_spectrum(adapter: ChiralFullToBdGAdapter, out_path: Path) -> None:
    """Predicted +-sigma_k(mu) over exact diagonalisation, with a residual panel."""
    ham = KitaevChainHamiltonian(n_sites=N_SITES, hopping=T, pairing=DELTA)
    exact = np.array(
        [np.sort(np.linalg.eigvalsh(ham.build(float(mu)))) for mu in MU_GRID]
    )
    with torch.no_grad():
        pred = (
            adapter.full_spectrum(torch.tensor(MU_GRID[:, None], dtype=torch.float32))
            .cpu()
            .numpy()
        )

    use_house_style()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 6), height_ratios=(3, 1), sharex=True
    )
    for k in range(2 * N_SITES):
        ax_top.plot(MU_GRID, exact[:, k], color="0.7", lw=2.0)
        ax_top.plot(MU_GRID, pred[:, k], color="#e4572e", lw=0.9)
    ax_top.set_ylabel(r"BdG eigenvalue $E$")
    ax_top.set_title(
        "Full spectrum: predicted (coral) over exact diagonalisation (grey)"
    )
    mark_transition(ax_top, hopping=T, mu_max=4.0, two_sided=True)

    ax_bot.plot(MU_GRID, np.abs(pred - exact).max(axis=1), color="#1b2a41", lw=1.2)
    ax_bot.set_yscale("log")
    ax_bot.set_ylabel("max level error")
    ax_bot.set_xlabel(r"chemical potential $\mu / t$")
    mark_transition(ax_bot, hopping=T, mu_max=4.0, two_sided=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_majorana_densities(adapter: ChiralFullToBdGAdapter, out_path: Path) -> None:
    """Reconstructed left/right end-mode site densities across the topological phase."""
    ham = KitaevChainHamiltonian(n_sites=N_SITES, hopping=T, pairing=DELTA)
    sites = np.arange(N_SITES)
    model = adapter.model

    use_house_style()
    fig, axes = plt.subplots(
        1, len(MAJORANA_MUS), figsize=(3.2 * len(MAJORANA_MUS), 3.0), sharey=True
    )
    for ax, mu in zip(axes, MAJORANA_MUS, strict=True):
        x = torch.tensor([[mu]], dtype=torch.float32)
        with torch.no_grad():
            u_mat, sigma, v_mat = model(x)
        order = torch.argsort(sigma[0])[:2]
        u_pair = u_mat[0, :, order].T  # (2, N)
        v_pair = v_mat[0, :, order].T
        u_pair, v_pair = resolve_svd_sign(u_pair.unsqueeze(-1), v_pair.unsqueeze(-1))
        psi = reconstruct_bdg_eigenvectors(
            u_pair.squeeze(-1), v_pair.squeeze(-1)
        ).numpy()
        left = (psi[0] + psi[1]) / np.sqrt(2.0)
        right = (psi[0] - psi[1]) / np.sqrt(2.0)
        dens_left = left[:N_SITES] ** 2 + left[N_SITES:] ** 2
        dens_right = right[:N_SITES] ** 2 + right[N_SITES:] ** 2

        w, vecs = np.linalg.eigh(ham.build(float(mu)))
        near = vecs[:, np.argsort(np.abs(w))[:2]]
        gl = (near[:, 0] + near[:, 1]) / np.sqrt(2.0)
        gr = (near[:, 0] - near[:, 1]) / np.sqrt(2.0)
        ex_left = gl[:N_SITES] ** 2 + gl[N_SITES:] ** 2
        ex_right = gr[:N_SITES] ** 2 + gr[N_SITES:] ** 2

        ax.fill_between(sites, ex_left, color="#2a9d8f", alpha=0.25, lw=0)
        ax.fill_between(sites, ex_right, color="#e9c46a", alpha=0.25, lw=0)
        ax.plot(sites, dens_left, color="#2a9d8f", lw=1.4, label="left mode")
        ax.plot(sites, dens_right, color="#e9c46a", lw=1.4, label="right mode")
        ax.set_title(rf"$\mu = {mu:.2f}\,t$")
        ax.set_xlabel("site $n$")
    axes[0].set_ylabel("site density (predicted lines, exact shaded)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_det_sign(adapter: ChiralFullToBdGAdapter, out_path: Path) -> None:
    """sign(det h) from the frame vs analytic, plus sigma_min(mu) on a log axis."""
    x = torch.tensor(MU_GRID[:, None], dtype=torch.float32)
    with torch.no_grad():
        det_pred = adapter.det_sign(x).cpu().numpy()
        _u, sigma, _v = adapter.model(x)
    sigma_min = sigma.min(dim=1).values.cpu().numpy()
    det_exact = np.array(
        [np.sign(np.linalg.det(chiral_block(mu, N_SITES, T, DELTA))) for mu in MU_GRID]
    )
    ham = KitaevChainHamiltonian(n_sites=N_SITES, hopping=T, pairing=DELTA)
    sigma_exact = np.array(
        [np.abs(np.linalg.eigvalsh(ham.build(float(mu)))).min() for mu in MU_GRID]
    )

    use_house_style()
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax_top.step(MU_GRID, det_exact, color="0.7", lw=2.5, where="mid", label="analytic")
    ax_top.step(MU_GRID, det_pred, color="#e4572e", lw=1.1, where="mid", label="frame")
    ax_top.set_ylabel(r"$\mathrm{sign}\,\det h(\mu)$")
    ax_top.set_yticks((-1, 1))
    ax_top.legend(frameon=False, fontsize=8)
    ax_top.set_title(r"$\mathbb{Z}_2$ datum from the learned frame")
    mark_transition(ax_top, hopping=T, mu_max=4.0, two_sided=True)

    ax_bot.plot(MU_GRID, sigma_exact, color="0.7", lw=2.5, label="exact")
    ax_bot.plot(MU_GRID, sigma_min, color="#1b2a41", lw=1.1, label="predicted")
    ax_bot.set_yscale("log")
    ax_bot.set_ylabel(r"$\sigma_{\min}(\mu)$")
    ax_bot.set_xlabel(r"chemical potential $\mu / t$")
    ax_bot.legend(frameon=False, fontsize=8)
    mark_transition(ax_bot, hopping=T, mu_max=4.0, two_sided=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _frame_dispersions(models: list[torch.nn.Module]) -> tuple[float, float]:
    """(gauge-invariant density dispersion, gauge-dependent frame dispersion).

    ``density`` = std across seeds of every triple's per-site density
    ``u_k^2 + v_k^2``, then meaned -- gauge-invariant, expected ~0.
    ``frame`` = after resolve_svd_sign and sigma-nearest column matching,
    mean over mu of ``||U_a - U_b||_F`` across seed pairs -- gauge-dependent.
    """
    x = torch.tensor(MU_GRID[:, None], dtype=torch.float32)
    densities, us = [], []
    for model in models:
        with torch.no_grad():
            u_mat, sigma, v_mat = model(x)
            u_mat, v_mat = resolve_svd_sign(u_mat, v_mat)
        order = torch.argsort(sigma, dim=1)
        u_sorted = torch.gather(u_mat, 2, order.unsqueeze(1).expand(-1, N_SITES, -1))
        v_sorted = torch.gather(v_mat, 2, order.unsqueeze(1).expand(-1, N_SITES, -1))
        densities.append((u_sorted**2 + v_sorted**2).cpu().numpy())
        us.append(u_sorted.cpu().numpy())

    density_std = float(np.mean(np.std(np.stack(densities), axis=0)))
    frame = []
    for a in range(len(us)):
        for b in range(a + 1, len(us)):
            frame.append(float(np.mean(np.linalg.norm(us[a] - us[b], axis=(1, 2)))))
    return density_std, float(np.mean(frame)) if frame else 0.0


def _plot_frame_reproducibility(
    models: list[torch.nn.Module], out_png: Path, out_csv: Path
) -> None:
    density_std, frame_std = _frame_dispersions(models)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerow(["frame_density_dispersion", density_std])
        writer.writerow(["frame_vector_dispersion", frame_std])
        writer.writerow(["gauge_gap", frame_std - density_std])

    use_house_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(
        ["density\n(gauge-invariant)", "frame\n(gauge-dependent)"],
        [density_std, frame_std],
        color=["#2a9d8f", "#e4572e"],
    )
    ax.set_ylabel("inter-seed dispersion")
    ax.set_title(f"Frame reproducibility ({len(models)} seeds)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def train_one(
    seed: int, epochs: int, gauge_weight: float, smoke: bool, session: Session
) -> dict[str, Any]:
    """Train one seed of the full-SVD chiral model; return a metric row."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # torch.matrix_exp has no MPS kernel, so fall back to CPU on Apple
    # Silicon; CUDA is unaffected.
    force_cpu = torch.backends.mps.is_available() and not torch.cuda.is_available()
    acc = Accelerator(cpu=force_cpu)
    if force_cpu:
        session.info("matrix_exp unsupported on MPS; running on CPU")
    device = acc.device
    H_base, H_mu_diag, Xi = (
        x.to(device) for x in _build_kitaev_operators(N_SITES, hopping=T, pairing=DELTA)
    )

    model = build_chiral_svd_model()
    loss_fn = ChiralSVDLoss(
        n_sites=N_SITES, hopping=T, pairing=DELTA, gauge_weight=gauge_weight
    )
    train_loader, val_loader, lbfgs_loader = _loaders()
    two_phase = TwoPhaseConfig(
        adam_epochs=epochs,
        adam_lr=8e-4,
        adam_weight_decay=1e-6,
        lbfgs_epochs=5 if smoke else 300,
        lbfgs_max_iter=25,
        lbfgs_history_size=20,
        lbfgs_line_search_fn="strong_wolfe",
    )
    base_config = TrainerConfig(
        epochs=1, print_freq=2000, patience=None, grad_clip_norm=1.0
    )
    every = 2 if smoke else 50
    probe = BdGEvaluationProbe(
        n_sites=N_SITES,
        hopping=T,
        pairing=DELTA,
        mu_grid=MU_GRID,
        every=every,
        session=session,
        adapt=_adapter,
    )
    spectrum_probe = SpectrumEvaluationProbe(
        n_sites=N_SITES,
        spectrum=_spectrum_of,
        hopping=T,
        pairing=DELTA,
        mu_grid=MU_GRID,
        every=every,
        session=session,
    )

    session.info(f"--- chiral_svd seed={seed} epochs={epochs} gauge={gauge_weight} ---")
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
        callbacks=[probe, spectrum_probe],
        val_loader=val_loader,
        lbfgs_train_loader=lbfgs_loader,
        lbfgs_callbacks=[probe, spectrum_probe],
    )
    wall = time.perf_counter() - t0

    adapter = _adapter(trained).to(device)
    hamiltonian = KitaevChainHamiltonian(n_sites=N_SITES, hopping=T, pairing=DELTA)
    ckpt_dir = session.path("checkpoints", "chiral_svd")
    torch.save(trained.state_dict(), ckpt_dir / f"seed_{seed}.pt")

    fig_dir = session.path("figures", "chiral_svd", f"seed_{seed}")
    save_run_figures(
        adapter=adapter,
        history=history,
        hamiltonian=hamiltonian,
        mu_grid=MU_GRID,
        out_dir=fig_dir,
        model_label="chiral full SVD",
        component_keys=("svd",),
        weight_key=None,
        split_epoch=two_phase.adam_epochs,
        floor_value=0.0,
        structural_fold=True,
        device=device,
    )
    _plot_full_spectrum(adapter, fig_dir / "full_spectrum_vs_eigh.png")
    _plot_majorana_densities(adapter, fig_dir / "majorana_left_right_densities.png")
    _plot_det_sign(adapter, fig_dir / "det_sign_invariant.png")

    row: dict[str, Any] = {
        "model": "chiral_svd",
        "seed": seed,
        "adam_epochs": epochs,
        "gauge_weight": gauge_weight,
    }
    row.update(_lowest_pair_metrics(history, adapter, str(device)))
    row.update(_spectrum_metrics(history, adapter))
    row["wall_seconds"] = round(wall, 1)
    session.info(
        f"[chiral_svd seed={seed}] E MAE full {row['e_mae_full']:.3e} | "
        f"spectrum MAE {row['spectrum_mae']:.3e} "
        f"(bulk {row['spectrum_mae_bulk']:.3e}) | "
        f"det-sign agree {row['det_sign_agreement']:.3f}"
    )
    return row


def _rebuild_adapter(checkpoint: Path) -> ChiralFullToBdGAdapter:
    model = build_chiral_svd_model()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    return ChiralFullToBdGAdapter(model, hopping=T, pairing=DELTA)


def render_figures_only(session_dir: Path) -> list[Path]:
    """Re-render the three custom per-seed figures from a finished run's checkpoints."""
    written: list[Path] = []
    checkpoints = sorted((session_dir / "checkpoints" / "chiral_svd").glob("seed_*.pt"))
    models: list[torch.nn.Module] = []
    for ckpt in checkpoints:
        out_dir = session_dir / "figures" / "chiral_svd" / ckpt.stem
        if not out_dir.is_dir():
            continue
        adapter = _rebuild_adapter(ckpt)
        models.append(adapter.model)
        _plot_full_spectrum(adapter, out_dir / "full_spectrum_vs_eigh.png")
        _plot_majorana_densities(adapter, out_dir / "majorana_left_right_densities.png")
        _plot_det_sign(adapter, out_dir / "det_sign_invariant.png")
        written.extend(
            [
                out_dir / "full_spectrum_vs_eigh.png",
                out_dir / "majorana_left_right_densities.png",
                out_dir / "det_sign_invariant.png",
            ]
        )
    if len(models) >= 2:
        repro_dir = session_dir / "figures" / "chiral_svd"
        _plot_frame_reproducibility(
            models,
            repro_dir / "frame_reproducibility.png",
            repro_dir / "frame_reproducibility.csv",
        )
        written.append(repro_dir / "frame_reproducibility.png")
    return written


def summarise(rows: list[dict[str, Any]]) -> str:
    """Median over seeds of the headline columns."""
    keys = (
        "e_mae_full",
        "subspace_infidelity_max",
        "spectrum_mae",
        "spectrum_mae_bulk",
        "det_sign_agreement",
    )
    head = "  ".join(f"{k:>24}" for k in keys)
    cells = [f"{statistics.median(r[k] for r in rows):.3e}" for k in keys]
    return head + "\n" + "  ".join(f"{c:>24}" for c in cells)


def main() -> list[dict[str, Any]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-only",
        type=Path,
        default=None,
        metavar="SESSION_DIR",
        help="skip training; re-render the custom figures from this session",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--gauge-weight", type=float, default=0.0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one seed, tiny budget, for an end-to-end shakedown",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.figures_only is not None:
        written = render_figures_only(args.figures_only)
        for path in written:
            print(f"figure  {path}")
        print(f"re-rendered {len(written)} figures")
        return []

    seeds = [0] if args.smoke else args.seeds
    epochs = 40 if args.smoke else args.epochs

    session = Session(
        name="chiral-svd-spectrum",
        output_root=Path(__file__).resolve().parents[1] / "results" / "logs",
        enable_mlflow=False,
    )
    session.info(f"seeds={seeds} epochs={epochs} gauge_weight={args.gauge_weight}")
    session.log_params(
        {
            "t": T,
            "delta": DELTA,
            "n_sites": N_SITES,
            "seeds": seeds,
            "epochs": epochs,
            "gauge_weight": args.gauge_weight,
        }
    )

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows.append(train_one(seed, epochs, args.gauge_weight, args.smoke, session))

    out = args.out or (session.path() / "chiral_svd_spectrum.csv")
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    session.info(f"wrote {out}")

    if len(rows) >= 2:
        models = [
            _rebuild_adapter(p).model
            for p in sorted(
                (session.path("checkpoints", "chiral_svd")).glob("seed_*.pt")
            )
        ]
        repro_dir = session.path("figures", "chiral_svd")
        _plot_frame_reproducibility(
            models,
            repro_dir / "frame_reproducibility.png",
            repro_dir / "frame_reproducibility.csv",
        )

    print("\n" + summarise(rows) + "\n")
    print(f"chiral-svd rows: {out}")
    return rows


if __name__ == "__main__":
    main()
