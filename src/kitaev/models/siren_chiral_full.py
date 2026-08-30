# kitaev/models/siren_chiral_full.py
"""SIREN PINN emitting the full SVD of the chiral block ``h(mu)``.

:class:`kitaev.models.SirenPINNChiral` learns only the *smallest* singular
triple of the ``N x N`` bidiagonal chiral block ``h(mu)`` (see
:func:`kitaev.analytical.chiral_block`). Inside the topological phase the two
near-zero singular values are split by only ``lambda_1 ~ e^(-N/xi)``, so that
model pins the near-zero subspace and the balanced energy eigenstate but not
the individual end-localised Majorana modes, their signed per-mode
wavefunctions, or the rest of the spectrum.

:class:`SirenPINNChiralFull` instead learns the entire decomposition

    h(mu) = U(mu) Sigma(mu) V(mu)^T

with ``U`` in ``SO(N)`` and ``V`` in ``O(N)`` by construction (matrix
exponentials of skew generators, plus a determinant-fixing last-column
reflection on ``V``) and ``Sigma >= 0`` by construction (``softplus``). Paired
with :class:`kitaev.training.loss.ChiralSVDLoss` -- a single Frobenius
residual with no folded-spectrum floor -- it yields the whole ``2N`` BdG
spectrum in one pass, the individual Majoranas as smooth signed curves, and
the ``Z2`` datum ``sign(det h(mu))`` from the frame.

The determinant reflection
-------------------------
``det h(mu) = det U * prod sigma_k * det V``. With ``U, V`` both in ``SO(N)``
and ``sigma_k >= 0`` this forces ``det h >= 0``, but the bidiagonal
determinant follows a Chebyshev recurrence and changes sign roughly ``N / 2``
times across ``|mu| < 2 sqrt(t^2 - delta^2)`` -- most of the topological
phase. The model therefore multiplies the last column of ``V`` by the
analytic ``s(mu) = sign(det h(mu))`` (from
:func:`kitaev.analytical.chiral_block_det_sign`, ``O(N)`` and detached), so
``V`` reaches ``O(N)`` and every ``mu`` is representable. The only
discontinuity is a sign flip of that column where ``sigma_min(mu) = 0``
(a gap closing), where the corresponding singular vector is a null vector of
``h`` defined only up to sign -- energies, densities, the spectrum and
subspace fidelity are untouched. ``N`` is assumed even (physical chains), so
``s(-mu) = s(mu)`` and the ``mu``-fold bookkeeping is clean.

Device note
-----------
``torch.matrix_exp`` has no MPS kernel, so this model runs on CPU or CUDA
only; on Apple Silicon either force CPU or set
``PYTORCH_ENABLE_MPS_FALLBACK=1``.

Not built here (documented for later)
-------------------------------------
- A truncated ``r``-triple Stiefel / Householder frame for ``N`` in the
  hundreds; unnecessary at ``N = 20`` where ``matrix_exp`` is free, and it
  reintroduces a folded-spectrum pull term.
- A rank curriculum that grows the number of retained triples; add only if
  the Frobenius residual plateaus above float precision.

Chemical-potential reflection
-----------------------------
``h(-mu) = -D h(mu) D`` with ``D = diag((-1)^n)``, so
``U(-mu) = -D U(mu)``, ``V(-mu) = D V(mu)`` (both column-wise / row-scaled)
and ``Sigma(-mu) = Sigma(mu)``. Only ``|mu|`` is fed to the backbone and the
fixed transform is applied for ``mu < 0``; training only needs ``mu >= 0``.
As in :class:`SirenPINNChiral`, the ``-D`` factor makes ``U`` row-sign
discontinuous at ``mu = 0`` (a measure-zero gauge artefact that does not
affect ``E`` or ``|psi|^2``); sample ``mu`` in ``[0.05 t, 4 t]``.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from kitaev.analytical import (
    chiral_block_batched,
    chiral_block_det_sign,
    fill_skew,
    resolve_singular_branch,
)

from .layers import SineLayer


class SirenPINNChiralFull(nn.Module):
    """SIREN surrogate for the full SVD ``h(mu) = U Sigma V^T``.

    The network maps the chemical potential to two orthogonal frames and a
    non-negative singular spectrum:

        mu -> SIREN backbone -> features -> (skew_u, skew_v, s_raw)
            -> U = matrix_exp(fill_skew(skew_u))            in SO(N)
               V = matrix_exp(fill_skew(skew_v)) . R(mu)    in O(N)
               Sigma = softplus(s_raw)                      >= 0

    where ``R(mu) = diag(1, ..., 1, sign(det h(mu)))`` is the analytic
    last-column reflection (see the module docstring). ``Sigma`` is returned
    unsorted -- sorting would break smoothness in ``mu`` and the column
    correspondence with ``U``, ``V``, and the Frobenius loss is
    permutation-agnostic. Consumers that want the smallest triple take
    ``sigma.argmin(dim=1)`` (:class:`ChiralFullToBdGAdapter` does).

    Attributes:
        net: Shared SIREN feature-extraction network.
        head_u: Linear head for the ``N (N - 1) / 2`` free parameters of the
            skew generator of ``U``.
        head_v: Linear head for the skew generator of ``V``.
        head_s: Linear head for the pre-``softplus`` singular values.
        D: Buffer holding ``(-1)^n``, used for the ``mu -> -mu`` fold.

    Args:
        n_sites: Number of physical lattice sites, ``N`` (assumed even).
        in_features: Number of input features, normally one (``mu``).
        hidden_features: Width of the shared SIREN representation.
        hidden_layers: Number of hidden SIREN layers after the first.
        hidden_omega_0: Frequency parameter for every hidden SIREN layer.
        input_scale: Divides ``|mu|`` before the first SIREN layer. The
            default of ``4.0`` matches the ``mu`` in ``[0, 4 t]`` domain.
        hopping: Nearest-neighbour hopping amplitude, ``t``. Used only for
            the analytic ``sign(det h(mu))`` reflection.
        pairing: P-wave pairing amplitude, ``delta``. Likewise.
    """

    D: torch.Tensor

    def __init__(
        self,
        n_sites: int,
        *,
        in_features: int = 1,
        hidden_features: int = 32,
        hidden_layers: int = 2,
        hidden_omega_0: float = 1.0,
        input_scale: float = 4.0,
        hopping: float = 1.0,
        pairing: float = 0.5,
    ) -> None:
        """Initialise the full-SVD chiral SIREN PINN.

        Args:
            n_sites: Number of physical lattice sites, ``N`` (assumed even).
            in_features: Number of input features.
            hidden_features: Width of the shared SIREN representation.
            hidden_layers: Number of hidden SIREN layers following the first.
            hidden_omega_0: Frequency parameter used by every hidden SIREN
                layer (the first layer always uses ``30.0``).
            input_scale: Divides ``|mu|`` before the first SIREN layer.
            hopping: Hopping amplitude ``t`` for the ``sign(det h(mu))``
                reflection of ``V``.
            pairing: Pairing amplitude ``delta`` for the same.
        """
        super().__init__()

        self.n_sites = n_sites
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.hidden_omega_0 = hidden_omega_0
        self.input_scale = input_scale
        self.hopping = hopping
        self.pairing = pairing
        self.skew_dim = n_sites * (n_sites - 1) // 2

        layers = [
            SineLayer(
                self.in_features,
                self.hidden_features,
                is_first=True,
                omega_0=30.0,
            )
        ]
        for _ in range(hidden_layers):
            layers.append(
                SineLayer(
                    self.hidden_features,
                    self.hidden_features,
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                )
            )
        self.net = nn.Sequential(*layers)

        self.head_u = nn.Linear(self.hidden_features, self.skew_dim)
        self.head_v = nn.Linear(self.hidden_features, self.skew_dim)
        self.head_s = nn.Linear(self.hidden_features, self.n_sites)

        bound = math.sqrt(6.0 / self.hidden_features) / self.hidden_omega_0
        with torch.no_grad():
            self.head_u.weight.uniform_(-bound, bound)
            self.head_v.weight.uniform_(-bound, bound)
            self.head_s.weight.uniform_(-bound, bound)

        signs = torch.tensor(
            [(-1.0) ** n for n in range(self.n_sites)],
            dtype=torch.get_default_dtype(),
        )
        self.register_buffer("D", signs)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the full SVD frames of ``h(mu)``.

        Args:
            x: Chemical-potential values, shape ``(batch_size, in_features)``.
                Values may be negative.

        Returns:
            ``(U, sigma, V)`` with ``U`` shape ``(batch_size, N, N)`` in
            ``SO(N)``, ``V`` shape ``(batch_size, N, N)`` in ``O(N)`` with
            ``det V = sign(det h(mu))``, and ``sigma`` shape
            ``(batch_size, N)`` non-negative and unsorted.
        """
        magnitude = x.abs()
        features = self.net(magnitude / self.input_scale)
        batch = features.shape[0]

        u_mat = torch.matrix_exp(fill_skew(self.head_u(features), self.n_sites))
        v_so = torch.matrix_exp(fill_skew(self.head_v(features), self.n_sites))
        sigma = F.softplus(self.head_s(features))

        # Last-column reflection so det V = sign(det h(mu)); s is analytic
        # and detached, contributing no gradient.
        s = chiral_block_det_sign(x, self.n_sites, self.hopping, self.pairing).detach()
        reflect = torch.cat(
            [s.new_ones(batch, self.n_sites - 1), s.unsqueeze(1)], dim=1
        )
        v_mat = v_so * reflect.unsqueeze(1)

        # mu -> -mu fold: h(-mu) = -D h(mu) D = (-D U) Sigma (D V)^T.
        is_negative = (x < 0).reshape(batch, 1, 1)
        d_row = self.D.reshape(1, self.n_sites, 1)
        u_mat = torch.where(is_negative, -(d_row * u_mat), u_mat)
        v_mat = torch.where(is_negative, d_row * v_mat, v_mat)

        return u_mat, sigma, v_mat


class ChiralFullToBdGAdapter(nn.Module):
    """Presents :class:`SirenPINNChiralFull` as a dual-head ``(E, psi)`` model.

    ``forward`` selects the smallest singular triple ``argmin(sigma)`` and
    reconstructs the same ``(E_pred, psi_pred)`` pair as
    :class:`kitaev.models.ChiralToBdGAdapter`, so
    :class:`kitaev.training.BdGEvaluationProbe`, the visualisation sweeps and
    the four-model comparison metrics all run unchanged. The branch / sign of
    the selected pair is fixed by
    :func:`kitaev.analytical.resolve_singular_branch`, identically to the
    single-triple adapter.

    Two extra methods expose the capability the full frame adds:

    - :meth:`full_spectrum` -- all ``2N`` BdG eigenvalues ``{+-sigma_k}``,
      sorted, from one forward pass.
    - :meth:`det_sign` -- ``sign(det h(mu))`` read off the learned frame
      (``det U`` times ``det V``); equal by construction to the analytic
      value the model was fed, so it verifies the frame carries the ``Z2``
      datum consistently rather than discovering it.

    Args:
        model: The wrapped :class:`SirenPINNChiralFull` instance.
        hopping: Hopping amplitude ``t`` used to rebuild ``h(mu)``. Must
            match the value the model was trained against.
        pairing: Pairing amplitude ``delta``. Must likewise match.
    """

    def __init__(
        self,
        model: SirenPINNChiralFull,
        *,
        hopping: float = 1.0,
        pairing: float = 0.5,
    ) -> None:
        """Wrap a full-SVD chiral model in the dual-head interface.

        Args:
            model: The :class:`SirenPINNChiralFull` to wrap.
            hopping: Hopping amplitude used to rebuild ``h(mu)``.
            pairing: Pairing amplitude used to rebuild ``h(mu)``.
        """
        super().__init__()
        self.model = model
        self.hopping = hopping
        self.pairing = pairing

    def _smallest_pair(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Left/right singular vectors of the smallest singular value."""
        u_mat, sigma, v_mat = self.model(x)
        k = torch.argmin(sigma, dim=1)
        index = k.reshape(-1, 1, 1).expand(-1, self.model.n_sites, 1)
        u = torch.gather(u_mat, 2, index).squeeze(-1)
        v = torch.gather(v_mat, 2, index).squeeze(-1)
        return u, v

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(E_pred, psi_pred)`` for the smallest singular triple.

        Args:
            x: Chemical-potential values, shape ``(batch_size, in_features)``.

        Returns:
            ``(E_pred, psi_pred)`` of shapes ``(batch_size, 1)`` and
            ``(batch_size, 2 * n_sites)``.
        """
        u, v = self._smallest_pair(x)
        h_batch = chiral_block_batched(
            x, self.model.n_sites, self.hopping, self.pairing
        )
        u, v, _h_v, energy_pred = resolve_singular_branch(u, v, h_batch)
        psi_pred = torch.cat([(u + v) / 2.0, (u - v) / 2.0], dim=1)
        return energy_pred, psi_pred

    def full_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """All ``2N`` BdG eigenvalues ``{+-sigma_k}``, sorted ascending.

        Args:
            x: Chemical-potential values, shape ``(batch_size, in_features)``.

        Returns:
            A tensor of shape ``(batch_size, 2 * n_sites)``.
        """
        _u, sigma, _v = self.model(x)
        return torch.sort(torch.cat([sigma, -sigma], dim=1), dim=1).values

    def det_sign(self, x: torch.Tensor) -> torch.Tensor:
        """``sign(det h(mu))`` from the learned frame, shape ``(batch_size,)``.

        Args:
            x: Chemical-potential values, shape ``(batch_size, in_features)``.

        Returns:
            A detached tensor of shape ``(batch_size,)`` with values in
            ``{-1.0, +1.0}``. Detached because the sign is a discrete
            diagnostic with no meaningful gradient.
        """
        u_mat, _sigma, v_mat = self.model(x)
        sign_u = torch.linalg.slogdet(u_mat).sign
        sign_v = torch.linalg.slogdet(v_mat).sign
        sign: torch.Tensor = (sign_u * sign_v).detach()
        return sign
