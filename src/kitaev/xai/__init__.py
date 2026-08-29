"""Interpretability pipeline for the Kitaev-chain PINN study.

The four models solve one problem but differ in how much known structure
they carry in the architecture rather than the loss. This package measures
the consequences of that choice.

- :mod:`~kitaev.xai.internalisation` counts what each model guarantees by
  construction versus penalises.
- :mod:`~kitaev.xai.conditioning` measures the residual-operator spectral
  gap across the transition, per basis, analytically.
- :mod:`~kitaev.xai.dispersion` measures how far a model's predictions move
  across seeds, which maps where they are under-determined.
- :mod:`~kitaev.xai.residual_field` resolves a trained model's own physics
  residual per chemical potential, a difficulty map.
- :mod:`~kitaev.xai.figures` and :mod:`~kitaev.xai.report` render the set.

``notebooks/xai/four-model-interpretability.ipynb`` wires these onto a
completed ``four_model_comparison.py`` run.
"""

from .conditioning import ConditioningSweep, sweep_conditioning
from .dispersion import SeedDispersion, sweep_seed_dispersion
from .figures import (
    plot_conditioning,
    plot_residual_field,
    plot_seed_dispersion,
    plot_transparency_axis,
)
from .internalisation import KITAEV_PROFILES, InternalisationProfile
from .loading import load_seed_checkpoints, psi_only, read_comparison_errors
from .report import XaiAnalysis, save_xai_report, shared_dispersion_ylim
from .residual_field import ResidualField, sweep_residual_field

__all__ = [
    "ConditioningSweep",
    "sweep_conditioning",
    "SeedDispersion",
    "sweep_seed_dispersion",
    "ResidualField",
    "sweep_residual_field",
    "InternalisationProfile",
    "KITAEV_PROFILES",
    "read_comparison_errors",
    "load_seed_checkpoints",
    "psi_only",
    "plot_transparency_axis",
    "plot_conditioning",
    "plot_seed_dispersion",
    "plot_residual_field",
    "XaiAnalysis",
    "save_xai_report",
    "shared_dispersion_ylim",
]
