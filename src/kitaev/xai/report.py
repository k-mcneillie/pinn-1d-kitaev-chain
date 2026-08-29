"""One-call rendering of the interpretability figure set.

:class:`XaiAnalysis` collects everything the pipeline computes for a
completed model comparison, and :func:`save_xai_report` turns it into the
figure set on disk. The heavy lifting, loading checkpoints and running
forward passes, belongs to the caller (see
``notebooks/xai/four-model-interpretability.ipynb``); this module only
renders.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .conditioning import ConditioningSweep
from .dispersion import SeedDispersion
from .figures import (
    plot_conditioning,
    plot_residual_field,
    plot_seed_dispersion,
    plot_transparency_axis,
)
from .internalisation import InternalisationProfile
from .residual_field import ResidualField


@dataclass
class XaiAnalysis:
    """Everything the interpretability pipeline computes for one comparison.

    Attributes:
        profiles: One
            :class:`~kitaev.xai.internalisation.InternalisationProfile` per
            model.
        errors: Model name to a mapping with keys ``"topological"`` and
            ``"trivial"``, each a mean energy MAE over seeds.
        dispersion: Model name to its
            :class:`~kitaev.xai.dispersion.SeedDispersion`.
        conditioning: Basis name (``"nambu"`` or ``"chiral"``) to its
            :class:`~kitaev.xai.conditioning.ConditioningSweep`.
        residual_fields: Model name to its
            :class:`~kitaev.xai.residual_field.ResidualField`.
    """

    profiles: list[InternalisationProfile]
    errors: dict[str, dict[str, float]]
    dispersion: dict[str, SeedDispersion] = field(default_factory=dict)
    conditioning: dict[str, ConditioningSweep] = field(default_factory=dict)
    residual_fields: dict[str, ResidualField] = field(default_factory=dict)


def shared_dispersion_ylim(
    dispersions: Iterable[SeedDispersion],
) -> tuple[float, float] | None:
    """A common ``(low, high)`` log-axis range across several dispersions.

    The cross-seed section compares models by the magnitude of their
    spread, so every panel needs the same y-axis. This spans the smallest
    positive and largest value over all four plotted series of every
    dispersion, padded by half a decade each side. Returns ``None`` if
    nothing positive is present (a degenerate all-zero case).
    """
    highs: list[float] = []
    lows: list[float] = []
    for d in dispersions:
        stacked = np.concatenate(
            [
                d.density_std_max,
                d.density_std_mean,
                d.edge_weight_std,
                d.energy_std,
            ]
        )
        positive = stacked[np.isfinite(stacked) & (stacked > 0.0)]
        if positive.size:
            highs.append(float(positive.max()))
            lows.append(float(positive.min()))
    if not highs:
        return None
    return min(lows) / np.sqrt(10.0), max(highs) * np.sqrt(10.0)


def save_xai_report(
    analysis: XaiAnalysis,
    out_dir: str | Path,
    *,
    hopping: float,
    dpi: int = 300,
) -> dict[str, Path]:
    """Render and save every interpretability figure the analysis supports.

    Figures whose inputs are absent from ``analysis`` are skipped rather
    than raising, so a partial pipeline still produces a partial report.

    Args:
        analysis: The computed :class:`XaiAnalysis`.
        out_dir: Directory to write the PNGs into, created if absent.
        hopping: The hopping amplitude ``t``, for the transition markers.
        dpi: Resolution used when saving.

    Returns:
        A mapping from figure name to the path it was written to.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if analysis.profiles and analysis.errors:
        paths["transparency_axis"] = out_dir / "transparency_axis.png"
        plot_transparency_axis(
            analysis.profiles,
            analysis.errors,
            save_path=paths["transparency_axis"],
            dpi=dpi,
        )

    if analysis.conditioning:
        paths["conditioning"] = out_dir / "conditioning.png"
        plot_conditioning(
            list(analysis.conditioning.values()),
            hopping=hopping,
            save_path=paths["conditioning"],
            dpi=dpi,
        )

    if analysis.residual_fields:
        paths["residual_field"] = out_dir / "residual_field.png"
        plot_residual_field(
            list(analysis.residual_fields.values()),
            hopping=hopping,
            save_path=paths["residual_field"],
            dpi=dpi,
        )

    dispersion_ylim = shared_dispersion_ylim(analysis.dispersion.values())
    for name, dispersion in analysis.dispersion.items():
        key = f"dispersion_{name}"
        paths[key] = out_dir / f"{key}.png"
        plot_seed_dispersion(
            dispersion,
            hopping=hopping,
            model_label=name,
            ylim=dispersion_ylim,
            save_path=paths[key],
            dpi=dpi,
        )

    plt.close("all")
    return paths
