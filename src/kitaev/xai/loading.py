"""Turn a completed comparison run's on-disk artefacts back into objects.

A ``four_model_comparison.py`` run leaves a checkpoint per (model, seed)
under ``<session>/checkpoints/`` and a CSV of per-run errors. These helpers
reload those so the rest of :mod:`kitaev.xai` can consume them. They take
plain callables and paths and never import anything from ``experiments/``.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from statistics import mean

import torch


def read_comparison_errors(csv_path: str | Path) -> dict[str, dict[str, float]]:
    """Mean topological / trivial energy MAE per model from a comparison CSV.

    Args:
        csv_path: Path to the ``four_model_comparison.csv`` a comparison
            run wrote, or the run's session directory (in which case the
            file of that name inside it is used).

    Returns:
        A mapping from model name to ``{"topological": mae, "trivial":
        mae}``, each averaged over that model's seeds.
    """
    path = Path(csv_path)
    if path.is_dir():
        path = path / "four_model_comparison.csv"

    collected: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"topological": [], "trivial": []}
    )
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            collected[row["model"]]["topological"].append(
                float(row["e_mae_topological"])
            )
            collected[row["model"]]["trivial"].append(float(row["e_mae_trivial"]))
    return {
        model: {phase: mean(values) for phase, values in phases.items()}
        for model, phases in collected.items()
    }


def load_seed_checkpoints(
    build_model: Callable[[], torch.nn.Module],
    checkpoint_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> list[torch.nn.Module]:
    """Rebuild every ``seed_*.pt`` state dict in a directory.

    Args:
        build_model: Zero-argument factory returning a fresh model of the
            architecture the checkpoints were saved from.
        checkpoint_dir: Directory holding the ``seed_*.pt`` files.
        device: Device to place the rebuilt models on.

    Returns:
        One eval-mode model per checkpoint, in seed order. Empty if the
        directory holds no matching files.
    """
    models: list[torch.nn.Module] = []
    for checkpoint in sorted(Path(checkpoint_dir).glob("seed_*.pt")):
        model = build_model()
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        model.to(device).eval()
        models.append(model)
    return models


class _PsiOnly(torch.nn.Module):
    """Wraps a dual-head model so ``forward`` returns only the eigenvector."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eigenvector: torch.Tensor = self.model(x)[1]
        return eigenvector


def psi_only(
    model: torch.nn.Module, *, device: torch.device | str = "cpu"
) -> torch.nn.Module:
    """Return a model giving just ``psi``, unwrapping a dual-head ``(E, psi)``.

    Args:
        model: A model whose ``forward`` returns either ``psi`` or an
            ``(E, psi)`` tuple.
        device: Device the model is on, for the one-row probe forward pass.

    Returns:
        ``model`` unchanged when it already returns ``psi``, otherwise a
        thin :class:`~torch.nn.Module` wrapper that selects the
        eigenvector.
    """
    with torch.no_grad():
        probe = model(torch.zeros(1, 1, device=device))
    if isinstance(probe, tuple):
        return _PsiOnly(model)
    return model
