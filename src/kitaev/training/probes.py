# src/kitaev/training/probes.py
"""Physical-error probe for BdG spectral PINNs, hooked into the epoch loop.

The label-free training losses in this project (e.g.
:class:`kitaev.training.loss.ChiralFSMLoss`) are, after a few hundred
epochs, numerically dominated by a folded-spectrum floor
``<lambda_1(mu)^2>``. That floor is a property of the exact spectrum -- it
is nonzero because the trivial-phase gap is of order ``t`` -- and its
flatness is convergence, not stagnation. To make that visible,
:class:`BdGEvaluationProbe` compares the model against exact
diagonalisation on a fixed ``mu`` grid every few epochs and records
interpretable errors into the run's :class:`TrainingHistory`:

- ``probe_e_mae`` (and its topological / trivial split): mean absolute
  error of the predicted lowest non-negative eigenvalue;
- ``probe_edge_mae``: mean absolute error of the combined particle + hole
  probability weight on the outermost ``n_edge_sites`` sites at each end;
- ``probe_subspace_infidelity`` (mean and max): ``1 - ||P psi_pred||``,
  where ``P`` projects onto the span of the two exact eigenvectors of
  smallest ``|E|``. This is the well-defined eigenvector-accuracy measure
  even where the ``+-lambda_1`` pair is degenerate (the topological phase);
- ``probe_psi_norm``: mean Euclidean norm of ``psi_pred`` (should be ~1),
  a sanity check when comparing architectures.

The probe is model-agnostic: it expects a callable returning
``(E_pred, psi_pred)`` with ``psi_pred`` a ``2N`` Nambu-basis vector. A
dual-head model satisfies this directly; a model that returns something
else (e.g. the chiral ``(u, v)`` pair) is bridged with the ``adapt``
argument. Every architecture in the project can therefore be scored on the
same grid with the same metrics.

The exact references are computed once, at construction, in a single
diagonalisation sweep; each probe call is then only a forward pass.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch

from kitaev.analytical import KitaevChainHamiltonian

from .callbacks import TrainingCallback
from .utils import TrainingHistory

if TYPE_CHECKING:
    from sesh import Session

#: A callable turning the model under training into one whose ``forward``
#: returns ``(E_pred, psi_pred)``.
ModelAdapter = Callable[[torch.nn.Module], torch.nn.Module]


class BdGEvaluationProbe(TrainingCallback):
    """Records energy / edge-weight / eigenvector errors against ``eigh``.

    A :class:`~kitaev.training.callbacks.TrainingCallback` that evaluates the
    model on a fixed ``mu`` grid every ``every`` epochs and appends the
    resulting scalar errors to the run history under the ``probe_*`` keys
    (see the module docstring). The cumulative epoch index of each
    evaluation is stored as ``probe_epoch`` so the series can be plotted
    against the loss curves; the counter is kept on the instance, so a
    single probe passed to both stages of :func:`run_two_phase` produces a
    continuous ``probe_epoch`` axis across the AdamW and L-BFGS phases.

    Attributes:
        mu_grid: The chemical-potential grid the probe evaluates on.
        every: Evaluate once every this many epochs (plus the first).

    Args:
        n_sites: Number of physical lattice sites, ``N``.
        hopping: Nearest-neighbour hopping amplitude, ``t``.
        pairing: P-wave pairing amplitude, ``delta``.
        mu_grid: Grid to evaluate on. Defaults to 200 points spanning
            ``[0.05, 4 t]``.
        every: Epoch interval between evaluations. The first epoch is
            always evaluated so there is a baseline point.
        n_edge_sites: Sites counted at each end of the chain for the edge
            weight.
        session: Optional :class:`sesh.Session`; when given, each
            evaluation is also written to the run log.
        adapt: Optional callable mapping the model under training to one
            whose ``forward`` returns ``(E_pred, psi_pred)`` with
            ``psi_pred`` a ``(batch, 2 * n_sites)`` Nambu-basis tensor and
            ``E_pred`` broadcastable to ``(batch,)``. Pass ``None`` (the
            default) when the model already returns that pair, e.g. a
            dual-head model. For the chiral model pass
            ``lambda m: ChiralToBdGAdapter(m, hopping=t, pairing=delta)``.
            The adapted object must be an ``nn.Module`` sharing the model's
            parameters.
    """

    def __init__(
        self,
        *,
        n_sites: int,
        hopping: float = 1.0,
        pairing: float = 0.5,
        mu_grid: npt.NDArray[np.float64] | None = None,
        every: int = 100,
        n_edge_sites: int = 2,
        session: Session | None = None,
        adapt: ModelAdapter | None = None,
    ) -> None:
        """Diagonalise the exact references once and cache them."""
        self.n_sites = n_sites
        self.hopping = hopping
        self.pairing = pairing
        self.every = every
        self.n_edge_sites = n_edge_sites
        self.session = session
        self.adapt = adapt

        if mu_grid is None:
            mu_grid = np.linspace(0.05, 4.0 * hopping, 200)
        self.mu_grid = np.asarray(mu_grid, dtype=float)
        self._transition = 2.0 * hopping

        edge = np.concatenate(
            [
                np.arange(n_edge_sites),
                np.arange(n_sites - n_edge_sites, n_sites),
            ]
        )
        self._edge_sites = edge

        hamiltonian = KitaevChainHamiltonian(
            n_sites=n_sites, hopping=hopping, pairing=pairing
        )
        grid_size = self.mu_grid.shape[0]
        self._e_exact = np.zeros(grid_size)
        self._edge_exact = np.zeros(grid_size)
        self._near_zero = np.zeros((grid_size, 2 * n_sites, 2))
        for i, mu in enumerate(self.mu_grid):
            eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.build(float(mu)))
            self._e_exact[i] = eigenvalues[n_sites]
            psi = eigenvectors[:, n_sites]
            self._edge_exact[i] = (psi[:n_sites][edge] ** 2).sum() + (
                psi[n_sites:][edge] ** 2
            ).sum()
            self._near_zero[i] = eigenvectors[:, np.argsort(np.abs(eigenvalues))[:2]]

        self._calls = 0

    def on_epoch_end(
        self,
        epoch: int,
        model: torch.nn.Module,
        history: TrainingHistory,
    ) -> None:
        """Evaluate and record on the first epoch and every ``every`` after."""
        del epoch
        self._calls += 1
        if self._calls != 1 and self._calls % self.every != 0:
            return
        self._evaluate(model, history)

    def _evaluate(
        self,
        model: torch.nn.Module,
        history: TrainingHistory,
    ) -> None:
        """Run one forward sweep and append the ``probe_*`` metrics."""
        eval_model = self.adapt(model) if self.adapt is not None else model

        was_training = model.training
        eval_model.eval()
        param = next(model.parameters())
        with torch.no_grad():
            mu_tensor = torch.tensor(
                self.mu_grid[:, None], dtype=param.dtype, device=param.device
            )
            e_pred_t, psi_pred_t = eval_model(mu_tensor)
        if was_training:
            model.train()

        # |E|: E and -E are the same physical state; models that do not
        # branch-resolve their Rayleigh quotient may return either sign.
        e_pred = np.abs(e_pred_t.detach().cpu().numpy().reshape(-1))
        psi_pred = psi_pred_t.detach().cpu().numpy()

        # Normalise psi for the density / subspace metrics; report the raw
        # norm separately so a non-normalising model is still visible.
        psi_norm = np.linalg.norm(psi_pred, axis=1)
        psi_unit = psi_pred / np.clip(psi_norm, 1e-12, None)[:, None]

        abs_err = np.abs(e_pred - self._e_exact)
        # The transition sits at |mu| = 2t, so the phase split is on the
        # magnitude: this stays correct for a two-sided grid (e.g. the
        # dual-head model's [-3, 3] domain), where mu < -2t is trivial too.
        topological = np.abs(self.mu_grid) < self._transition
        trivial = ~topological

        n_sites = self.n_sites
        edge = self._edge_sites
        edge_pred = (psi_unit[:, :n_sites][:, edge] ** 2).sum(axis=1) + (
            psi_unit[:, n_sites:][:, edge] ** 2
        ).sum(axis=1)
        edge_err = np.abs(edge_pred - self._edge_exact)

        projection = np.einsum("gij,gi->gj", self._near_zero, psi_unit)
        infidelity = 1.0 - np.linalg.norm(projection, axis=1)

        metrics = {
            "probe_epoch": float(self._calls),
            "probe_e_mae": float(abs_err.mean()),
            "probe_e_mae_topological": _safe_mean(abs_err, topological),
            "probe_e_mae_trivial": _safe_mean(abs_err, trivial),
            "probe_edge_mae": float(edge_err.mean()),
            "probe_subspace_infidelity": float(infidelity.mean()),
            "probe_subspace_infidelity_max": float(infidelity.max()),
            "probe_psi_norm": float(psi_norm.mean()),
        }
        for key, value in metrics.items():
            history.record(key, value)

        if self.session is not None:
            self.session.info(
                f"probe | step {self._calls:>5d} | "
                f"E MAE {metrics['probe_e_mae']:.3e} "
                f"(topo {metrics['probe_e_mae_topological']:.3e}, "
                f"triv {metrics['probe_e_mae_trivial']:.3e}) | "
                f"edge MAE {metrics['probe_edge_mae']:.3e} | "
                f"infidelity {metrics['probe_subspace_infidelity']:.3e} "
                f"(max {metrics['probe_subspace_infidelity_max']:.3e})"
            )


def _safe_mean(values: npt.NDArray[np.float64], mask: npt.NDArray[np.bool_]) -> float:
    """Mean of ``values[mask]``, or NaN if the mask selects nothing."""
    if not mask.any():
        return float("nan")
    return float(values[mask].mean())
