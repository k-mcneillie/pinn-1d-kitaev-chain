# src/kitaev/training/probes.py
"""Physical-error probe for the chiral PINN, hooked into the epoch loop.

After a few hundred epochs the training loss of
:class:`kitaev.training.loss.ChiralFSMLoss` is numerically dominated by the
folded-spectrum floor ``<lambda_1(mu)^2>``. That floor is a property of the
exact spectrum -- it is nonzero because the trivial-phase gap is of order
``t`` -- and its flatness is convergence, not stagnation. To make that
visible, :class:`ChiralEvaluationProbe` compares the model against exact
diagonalisation on a fixed ``mu`` grid every few epochs and records
interpretable errors into the run's :class:`TrainingHistory`:

- ``probe_e_mae`` (and its topological / trivial split): mean absolute
  error of the predicted lowest non-negative eigenvalue;
- ``probe_edge_mae``: mean absolute error of the combined particle + hole
  probability weight on the outermost ``n_edge_sites`` sites at each end;
- ``probe_subspace_infidelity`` (mean and max): ``1 - ||P psi_pred||``,
  where ``P`` projects onto the span of the two exact eigenvectors of
  smallest ``|E|``. This is the well-defined eigenvector-accuracy measure
  even where the ``+-lambda_1`` pair is degenerate (the topological phase).

All three fall steadily while the loss is flat, which is the evidence that
training is still improving the physical solution.

The exact references are computed once, at construction, in a single
diagonalisation sweep; each probe call is then only a forward pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import torch

from kitaev.analytical import KitaevChainHamiltonian
from kitaev.models.siren_chiral import ChiralToBdGAdapter, SirenPINNChiral

from .callbacks import TrainingCallback
from .utils import TrainingHistory

if TYPE_CHECKING:
    from sesh import Session


class ChiralEvaluationProbe(TrainingCallback):
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
    ) -> None:
        """Diagonalise the exact references once and cache them."""
        self.n_sites = n_sites
        self.hopping = hopping
        self.pairing = pairing
        self.every = every
        self.n_edge_sites = n_edge_sites
        self.session = session

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
        device = next(model.parameters()).device
        adapter = ChiralToBdGAdapter(
            cast(SirenPINNChiral, model),
            hopping=self.hopping,
            pairing=self.pairing,
        ).to(device)

        was_training = model.training
        adapter.eval()
        with torch.no_grad():
            mu_tensor = torch.tensor(
                self.mu_grid[:, None], dtype=torch.float32, device=device
            )
            e_pred_t, psi_pred_t = adapter(mu_tensor)
        if was_training:
            model.train()

        e_pred = e_pred_t.cpu().numpy().ravel()
        psi_pred = psi_pred_t.cpu().numpy()

        abs_err = np.abs(e_pred - self._e_exact)
        topological = self.mu_grid < self._transition
        trivial = ~topological

        n_sites = self.n_sites
        edge = self._edge_sites
        edge_pred = (psi_pred[:, :n_sites][:, edge] ** 2).sum(axis=1) + (
            psi_pred[:, n_sites:][:, edge] ** 2
        ).sum(axis=1)
        edge_err = np.abs(edge_pred - self._edge_exact)

        projection = np.einsum("gij,gi->gj", self._near_zero, psi_pred)
        infidelity = 1.0 - np.linalg.norm(projection, axis=1)

        metrics = {
            "probe_epoch": float(self._calls),
            "probe_e_mae": float(abs_err.mean()),
            "probe_e_mae_topological": _safe_mean(abs_err, topological),
            "probe_e_mae_trivial": _safe_mean(abs_err, trivial),
            "probe_edge_mae": float(edge_err.mean()),
            "probe_subspace_infidelity": float(infidelity.mean()),
            "probe_subspace_infidelity_max": float(infidelity.max()),
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
