"""Animated, time-scrubbed views of the Kitaev-chain models.

These builders return a :class:`~matplotlib.animation.FuncAnimation` over a
grid of chemical-potential frames. Render one inline in a notebook with
``HTML(anim.to_jshtml())`` (needs only matplotlib, no ffmpeg) or write a
gif with :func:`save_animation`.

The numerics reuse the same sweeps and pointwise-residual functions the
static figures and the interpretability pipeline already use, so an
animation never says anything a still figure could not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure

from kitaev.analytical import KitaevChainHamiltonian
from kitaev.training.loss import chiral_pointwise_residual, nambu_pointwise_residual

from .style import (
    CORAL,
    INK,
    LEVEL_COLOURS,
    SLATE,
    TEAL,
    mark_transition,
    use_house_style,
)

_LEVEL_COLOURS = LEVEL_COLOURS

_SERIES_COLOURS = (INK, CORAL, TEAL, SLATE)
_VALID_BASES = ("nambu", "chiral")


@dataclass(frozen=True)
class MovieModel:
    """One model's two handles for the wavefunction movie.

    Attributes:
        label: Short name for the legend.
        adapter: A model or adapter returning ``(E, psi)`` with ``psi`` a
            ``(batch, 2N)`` Nambu-basis vector -- drives the density
            panel.
        residual_model: The raw model in its own basis (``psi`` for a
            Nambu model, ``(u, v)`` for a chiral one) -- drives the
            pointwise residual. For a dual-head model wrap it so
            ``forward`` returns only ``psi`` (see
            :func:`kitaev.xai.loading.psi_only`).
        basis: ``"nambu"`` or ``"chiral"``, selects the residual function.
    """

    label: str
    adapter: torch.nn.Module
    residual_model: torch.nn.Module
    basis: str


@dataclass
class WavefunctionMovie:
    """Precomputed frames for :func:`animate_wavefunction_residual`.

    The particle and hole sectors are kept apart -- their balance is
    physical (a good Majorana end mode has ``|psi^p_n|^2 = |psi^h_n|^2``
    site by site), so the animations show both rather than the sum.

    Attributes:
        mu_frames: The chemical-potential values, one per animation frame,
            shape ``(n_frames,)``.
        particle_exact: Exact particle-sector density ``|psi^p_n(mu)|^2``
            of the lowest non-negative state, shape ``(n_frames,
            n_sites)``.
        hole_exact: Exact hole-sector density ``|psi^h_n(mu)|^2``, same
            shape.
        particle: Per-model predicted particle-sector density, same shape
            each.
        hole: Per-model predicted hole-sector density, same shape each.
        residuals: Per-model pointwise physics residual at each frame,
            shape ``(n_frames,)`` each.
        transition: The topological transition, ``2 * hopping``.
    """

    mu_frames: npt.NDArray[np.float64]
    particle_exact: npt.NDArray[np.float64]
    hole_exact: npt.NDArray[np.float64]
    particle: dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)
    hole: dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)
    residuals: dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)
    transition: float = 4.0


def _sector_densities(
    psi: npt.NDArray[np.float64], n_sites: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """``(particle, hole)`` site densities of a unit-normalised ``(B, 2N)`` psi."""
    norm = np.linalg.norm(psi, axis=1, keepdims=True)
    unit = psi / np.clip(norm, 1e-12, None)
    return unit[:, :n_sites] ** 2, unit[:, n_sites:] ** 2


def _styled_figure(
    nrows: int, ncols: int, *, figsize: tuple[float, float], dpi: int | None
) -> tuple[Figure, Any]:
    """House-styled ``plt.subplots``; a lower ``dpi`` keeps embedded frames small.

    ``to_jshtml`` stores one PNG per frame, so dropping the figure dpi to
    72 or so roughly halves a long animation's embedded size without
    hurting a screen-resolution view.
    """
    use_house_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if dpi is not None:
        fig.set_dpi(dpi)
    return fig, axes


def build_wavefunction_movie(
    models: Sequence[MovieModel],
    hamiltonian: KitaevChainHamiltonian,
    mu_frames: npt.NDArray[np.float64],
    *,
    device: torch.device | str = "cpu",
) -> WavefunctionMovie:
    """Evaluate every model and exact diagonalisation over the frame grid.

    Args:
        models: The models to animate.
        hamiltonian: The exact Hamiltonian to diagonalise at each frame.
        mu_frames: 1D array of chemical-potential values, one per frame.
        device: Device for the model forward passes.

    Returns:
        The populated :class:`WavefunctionMovie`.

    Raises:
        ValueError: If any model carries an unknown ``basis``.
    """
    for model in models:
        if model.basis not in _VALID_BASES:
            raise ValueError(
                f"{model.label!r} basis must be one of {_VALID_BASES}, "
                f"got {model.basis!r}"
            )

    mu_frames = np.asarray(mu_frames, dtype=float)
    n_sites = hamiltonian.n_sites
    hopping, pairing = hamiltonian.hopping, hamiltonian.pairing
    split = n_sites

    particle_exact = np.zeros((mu_frames.size, n_sites))
    hole_exact = np.zeros((mu_frames.size, n_sites))
    for i, mu in enumerate(mu_frames):
        _, eigenvectors = np.linalg.eigh(hamiltonian.build(float(mu)))
        psi = eigenvectors[:, split]
        particle_exact[i] = psi[:n_sites] ** 2
        hole_exact[i] = psi[n_sites:] ** 2

    mu_tensor = torch.tensor(mu_frames[:, None], dtype=torch.float32, device=device)
    particle: dict[str, npt.NDArray[np.float64]] = {}
    hole: dict[str, npt.NDArray[np.float64]] = {}
    residuals: dict[str, npt.NDArray[np.float64]] = {}
    for model in models:
        model.adapter.to(device).eval()
        model.residual_model.to(device).eval()
        with torch.no_grad():
            psi_pred = model.adapter(mu_tensor)[1].detach().cpu().numpy()
            residual_fn = (
                chiral_pointwise_residual
                if model.basis == "chiral"
                else nambu_pointwise_residual
            )
            residual = residual_fn(
                model.residual_model,
                mu_tensor,
                n_sites,
                hopping=hopping,
                pairing=pairing,
            )
        p_density, h_density = _sector_densities(psi_pred, n_sites)
        particle[model.label] = p_density
        hole[model.label] = h_density
        residuals[model.label] = residual.detach().cpu().numpy()

    return WavefunctionMovie(
        mu_frames=mu_frames,
        particle_exact=particle_exact,
        hole_exact=hole_exact,
        particle=particle,
        hole=hole,
        residuals=residuals,
        transition=2.0 * hopping,
    )


def animate_wavefunction_residual(
    movie: WavefunctionMovie,
    *,
    hopping: float,
    fps: int = 15,
    dpi: int | None = None,
    save_path: str | Path | None = None,
) -> FuncAnimation:
    """Animate the particle and hole densities with the residual and a playhead.

    Three panels: particle-sector density, hole-sector density, and the
    per-model pointwise residual with a playhead at the current ``mu``. A
    good Majorana end mode has the particle and hole panels agreeing site
    by site; a model that only gets the combined density right will not.

    Args:
        movie: The result of :func:`build_wavefunction_movie`.
        hopping: The hopping amplitude ``t``, for the transition markers.
        fps: Frames per second, used when saving.
        dpi: Figure dpi. Leave ``None`` for the house default; pass ~72
            to shrink an inline ``to_jshtml`` embed.
        save_path: If given, a gif is written here (see
            :func:`save_animation`).

    Returns:
        The :class:`~matplotlib.animation.FuncAnimation`. Keep a reference
        to it alive until it is rendered or saved.
    """
    fig, (ax_p, ax_h, ax_r) = _styled_figure(1, 3, figsize=(16, 4.8), dpi=dpi)
    fig.subplots_adjust(top=0.82, bottom=0.14, wspace=0.28)
    labels = list(movie.particle)
    colours = _SERIES_COLOURS
    sites = np.arange(movie.particle_exact.shape[1])

    ceiling = max(
        float(movie.particle_exact.max()),
        float(movie.hole_exact.max()),
        *(float(a.max()) for a in (*movie.particle.values(), *movie.hole.values())),
    )
    sectors = (
        (ax_p, movie.particle_exact, movie.particle, r"particle $|\psi^p_n|^2$"),
        (ax_h, movie.hole_exact, movie.hole, r"hole $|\psi^h_n|^2$"),
    )
    exact_lines: list[Any] = []
    model_line_sets: list[list[Any]] = []
    fills: list[Any] = []
    for ax, exact_arr, model_map, sector_title in sectors:
        fills.append(ax.fill_between(sites, exact_arr[0], color=INK, alpha=0.12, lw=0))
        (exact_line,) = ax.plot(sites, exact_arr[0], color=INK, lw=1.8, label="exact")
        exact_lines.append(exact_line)
        model_line_sets.append(
            [
                ax.plot(
                    sites,
                    model_map[label][0],
                    color=colours[i % len(colours)],
                    lw=1.4,
                    ls=(0, (3, 2)),
                    label=label,
                )[0]
                for i, label in enumerate(labels)
            ]
        )
        ax.set_ylim(0.0, ceiling * 1.15 + 1e-9)
        ax.set_xlabel("site $n$")
        ax.set_title(sector_title)
    ax_p.set_ylabel(r"$|\psi_n(\mu)|^2$")
    ax_p.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)

    mu_max = float(np.abs(movie.mu_frames).max())
    mark_transition(ax_r, hopping=hopping, mu_max=mu_max, two_sided=True)
    for i, label in enumerate(labels):
        ax_r.plot(
            movie.mu_frames,
            movie.residuals[label],
            color=colours[i % len(colours)],
            lw=1.6,
            label=label,
        )
    ax_r.set_yscale("log")
    ax_r.set_xlabel(r"$\mu / t$")
    ax_r.set_ylabel("pointwise residual")
    ax_r.set_title("physics residual")
    ax_r.legend(fontsize=8)
    playhead = ax_r.axvline(movie.mu_frames[0], color=SLATE, lw=1.4)

    title = fig.suptitle("", y=0.98, fontsize=13, weight="bold")

    def update(frame: int) -> None:
        mu = movie.mu_frames[frame]
        for j, (ax, exact_arr, model_map, _t) in enumerate(sectors):
            exact_lines[j].set_ydata(exact_arr[frame])
            fills[j].remove()
            fills[j] = ax.fill_between(
                sites, exact_arr[frame], color=INK, alpha=0.12, lw=0
            )
            for line, label in zip(model_line_sets[j], labels, strict=True):
                line.set_ydata(model_map[label][frame])
        playhead.set_xdata([mu, mu])
        phase = "topological" if abs(mu) < movie.transition else "trivial"
        title.set_text(rf"$\mu = {mu:+.2f}\,t$  ({phase})")

    anim = FuncAnimation(
        fig, update, frames=movie.mu_frames.size, interval=1000 / fps, blit=False
    )
    _finish(anim, save_path, fps)
    return anim


def _seed_colours(n_seeds: int) -> list[Any]:
    """Distinct per-seed colours from viridis, so a fan reads as a fan."""
    if n_seeds <= 1:
        return [CORAL]
    ramp = plt.get_cmap("viridis")
    return [ramp(x) for x in np.linspace(0.05, 0.9, n_seeds)]


def animate_seed_fan(
    particle_by_seed: Sequence[npt.NDArray[np.float64]],
    hole_by_seed: Sequence[npt.NDArray[np.float64]],
    mu_frames: npt.NDArray[np.float64],
    *,
    model_label: str,
    transition: float,
    particle_exact: npt.NDArray[np.float64] | None = None,
    hole_exact: npt.NDArray[np.float64] | None = None,
    fps: int = 15,
    dpi: int | None = None,
    save_path: str | Path | None = None,
) -> FuncAnimation:
    """Animate every seed's particle and hole densities for one model.

    Two panels, particle and hole. Where the model is under-determined the
    seed curves fan apart and jitter between frames; where it is pinned
    they collapse onto one line. Each seed has its own colour.

    Args:
        particle_by_seed: One ``(n_frames, n_sites)`` particle-sector
            density per seed.
        hole_by_seed: The hole-sector counterpart, same shapes.
        mu_frames: 1D array of chemical-potential values, one per frame.
        model_label: Name for the title.
        transition: The topological transition, ``2 * hopping``.
        particle_exact: Optional ``(n_frames, n_sites)`` exact
            particle-sector density drawn as a filled reference.
        hole_exact: The hole-sector exact reference.
        fps: Frames per second, used when saving.
        dpi: Figure dpi; pass ~72 to shrink an inline ``to_jshtml`` embed.
        save_path: If given, a gif is written here.

    Returns:
        The :class:`~matplotlib.animation.FuncAnimation`.
    """
    fig, axes = _styled_figure(1, 2, figsize=(13, 4.9), dpi=dpi)
    fig.subplots_adjust(top=0.80, bottom=0.13, wspace=0.22)
    mu_frames = np.asarray(mu_frames, dtype=float)
    p_stack = np.stack([np.asarray(d, dtype=float) for d in particle_by_seed])
    h_stack = np.stack([np.asarray(d, dtype=float) for d in hole_by_seed])
    n_seeds = p_stack.shape[0]
    sites = np.arange(p_stack.shape[2])
    seed_colours = _seed_colours(n_seeds)

    sectors = (
        (axes[0], p_stack, particle_exact, r"particle $|\psi^p_n|^2$"),
        (axes[1], h_stack, hole_exact, r"hole $|\psi^h_n|^2$"),
    )
    exacts: list[npt.NDArray[np.float64] | None] = []
    fills: list[Any] = []
    seed_line_sets: list[list[Any]] = []
    for ax, stack, exact_arr, sector_title in sectors:
        exact_arr = None if exact_arr is None else np.asarray(exact_arr, dtype=float)
        exacts.append(exact_arr)
        fills.append(
            None
            if exact_arr is None
            else ax.fill_between(
                sites, exact_arr[0], color=INK, alpha=0.14, lw=0, label="exact"
            )
        )
        seed_line_sets.append(
            [
                ax.plot(
                    sites, stack[s, 0], color=seed_colours[s], lw=1.2, label=f"seed {s}"
                )[0]
                for s in range(n_seeds)
            ]
        )
        # Robust ceiling: a single pathological gauge spike (e.g. right at
        # mu = 0) must not flatten the rest of the sweep to the axis floor.
        ceiling = float(np.percentile(stack, 99.5))
        if exact_arr is not None:
            ceiling = max(ceiling, float(exact_arr.max()))
        ax.set_ylim(0.0, ceiling * 1.3 + 1e-9)
        ax.set_xlabel("site $n$")
        ax.set_title(sector_title)
    axes[0].set_ylabel(r"$|\psi_n(\mu)|^2$")
    axes[0].legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)

    title = fig.suptitle("", y=0.98, fontsize=13, weight="bold")

    def update(frame: int) -> None:
        mu = mu_frames[frame]
        for j, (ax, stack, _exact, _t) in enumerate(sectors):
            for s, line in enumerate(seed_line_sets[j]):
                line.set_ydata(stack[s, frame])
            exact_arr = exacts[j]
            if exact_arr is not None:
                fills[j].remove()
                fills[j] = ax.fill_between(
                    sites, exact_arr[frame], color=INK, alpha=0.14, lw=0
                )
        phase = "topological" if abs(mu) < transition else "trivial"
        title.set_text(
            rf"{model_label}: {n_seeds} seeds, $\mu = {mu:+.2f}\,t$  ({phase})"
        )

    anim = FuncAnimation(
        fig, update, frames=mu_frames.size, interval=1000 / fps, blit=False
    )
    _finish(anim, save_path, fps)
    return anim


def animate_spectrum(
    hamiltonian: KitaevChainHamiltonian,
    mu_frames: npt.NDArray[np.float64],
    *,
    n_levels: int = 4,
    fps: int = 15,
    dpi: int | None = None,
    save_path: str | Path | None = None,
) -> FuncAnimation:
    """Draw the lowest distinct exact levels ``sigma_k(mu)`` out as ``mu`` sweeps.

    ``sigma_1`` collapsing towards zero for ``|mu| < 2t`` and the bulk gap
    closing at ``|mu| = 2t`` become something you watch happen. The
    ``+-`` pairing of the BdG spectrum is folded out, so each ``sigma_k``
    is a distinct curve.

    Args:
        hamiltonian: The exact Hamiltonian to diagonalise at each frame.
        mu_frames: 1D array of chemical-potential values, one per frame.
        n_levels: How many of the smallest distinct levels to track.
        fps: Frames per second, used when saving.
        dpi: Figure dpi; pass ~72 to shrink an inline ``to_jshtml`` embed.
        save_path: If given, a gif is written here.

    Returns:
        The :class:`~matplotlib.animation.FuncAnimation`.
    """
    mu_frames = np.asarray(mu_frames, dtype=float)
    split = hamiltonian.n_sites
    levels = np.zeros((mu_frames.size, n_levels))
    for i, mu in enumerate(mu_frames):
        eigenvalues = np.linalg.eigvalsh(hamiltonian.build(float(mu)))
        levels[i] = eigenvalues[split : split + n_levels]

    fig, ax = _styled_figure(1, 1, figsize=(9, 4.6), dpi=dpi)
    transition = 2.0 * hamiltonian.hopping
    mark_transition(
        ax,
        hopping=hamiltonian.hopping,
        mu_max=float(np.abs(mu_frames).max()),
        two_sided=True,
    )
    trails = [
        ax.plot(
            [],
            [],
            color=_LEVEL_COLOURS[k % len(_LEVEL_COLOURS)],
            lw=1.8 if k == 0 else 1.1,
            label=rf"$\sigma_{{{k + 1}}}$",
        )[0]
        for k in range(n_levels)
    ]
    (head,) = ax.plot([], [], "o", color=CORAL, ms=5, zorder=5)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.set_yscale("log")
    ax.set_xlim(float(mu_frames.min()), float(mu_frames.max()))
    ax.set_ylim(max(levels.min() * 0.5, 1e-6), levels.max() * 1.5)
    ax.set_xlabel(r"$\mu / t$")
    ax.set_ylabel(r"$\sigma_k$")
    title = ax.set_title("")

    def update(frame: int) -> None:
        for k, line in enumerate(trails):
            line.set_data(mu_frames[: frame + 1], levels[: frame + 1, k])
        head.set_data([mu_frames[frame]] * n_levels, levels[frame])
        mu = mu_frames[frame]
        phase = "topological" if abs(mu) < transition else "trivial"
        title.set_text(rf"lowest $\sigma_k$,  $\mu = {mu:+.2f}\,t$  ({phase})")

    anim = FuncAnimation(
        fig, update, frames=mu_frames.size, interval=1000 / fps, blit=False
    )
    _finish(anim, save_path, fps)
    return anim


def save_animation(
    anim: FuncAnimation, save_path: str | Path, *, fps: int = 15
) -> Path:
    """Write ``anim`` to ``save_path`` as a gif.

    Args:
        anim: A built animation.
        save_path: Destination path. Must end ``.gif`` -- mp4 needs ffmpeg,
            which is not a dependency.
        fps: Frames per second.

    Returns:
        The path written.

    Raises:
        ValueError: If ``save_path`` does not end ``.gif``.
    """
    save_path = Path(save_path)
    if save_path.suffix.lower() != ".gif":
        raise ValueError(f"save_path must be a .gif, got {save_path.name!r}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(save_path), writer=PillowWriter(fps=fps))
    return save_path


def _finish(anim: FuncAnimation, save_path: str | Path | None, fps: int) -> None:
    if save_path is not None:
        save_animation(anim, save_path, fps=fps)
