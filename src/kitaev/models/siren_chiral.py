# kitaev/models/siren_chiral.py
"""SIREN-based PINN in the Majorana (chiral) basis for the Kitaev chain."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from kitaev.analytical import chiral_block_batched, resolve_singular_branch

from .layers import SineLayer


class SirenPINNChiral(nn.Module):
    """SIREN surrogate for the smallest singular triple of the chiral block.

    This model works in the Majorana basis, where the chiral (BDI) symmetry
    of the real Kitaev chain block-diagonalises the ``2N``-dimensional
    Bogoliubov-de Gennes (BdG) problem into a single real ``N x N``
    bidiagonal operator ``h(mu)`` (see
    :func:`kitaev.analytical.chiral_block`). The BdG single-particle
    spectrum is exactly ``{+-lambda_k}``, the singular values of ``h(mu)``,
    so the ``+-E`` pairing, the particle-hole partner relation, and unit
    normalisation are all structural rather than penalised.

    The network maps the chemical potential to the left/right singular
    vectors of the *smallest* singular value of ``h(mu)``:

        mu -> SIREN backbone -> shared features -> psi head -> (u(mu), v(mu))

    with ``u, v`` in ``R^N``. The associated energy is not an output: it is
    the Rayleigh quotient ``lambda_R(mu) = u(mu)^T h(mu) v(mu)``, evaluated
    by the loss, exactly as :class:`kitaev.training.loss.PinnedFSMLoss`
    computes a Rayleigh quotient in place of an energy head.

    Structural guarantees
    ---------------------
    - ``||u|| = ||v|| = 1`` (explicit L2 normalisation of each head half), so
      the reconstructed BdG eigenvector ``psi = ((u + v) / 2, (u - v) / 2)``
      is automatically unit-norm and its ``-E`` partner is the particle/hole
      block swap ``Xi psi``.
    - The energy is a singular value, the Rayleigh quotient
      ``lambda_R = u^T h(mu) v``, not a network output -- so no softplus and
      no pinning penalty are needed. The training loss is invariant under
      ``(u, v) -> (u, -v)``, which negates ``lambda_R`` and swaps ``psi``
      with its ``-E`` partner, so the raw model may sit on either branch;
      :func:`kitaev.analytical.resolve_singular_branch` (applied by
      :class:`ChiralToBdGAdapter`) selects the ``E >= 0`` branch at
      reconstruction time.

    Chemical-potential reflection
    -----------------------------
    The chiral block obeys ``h(-mu) = -D h(mu) D`` with
    ``D = diag((-1)^n)``, so the spectrum is even in ``mu`` and the singular
    vectors transform as ``u(-mu) = -D u(mu)``, ``v(-mu) = D v(mu)``. The
    forward pass exploits this: only ``|mu|`` is fed to the backbone and the
    fixed transform is applied for ``mu < 0``. Training therefore only needs
    to cover ``mu >= 0``.

    One caveat: the ``-D`` factor on ``u`` makes ``u`` sign-discontinuous at
    ``mu = 0`` unless ``u(0)`` is supported on odd sites. This is a
    measure-zero global-gauge artefact that does not affect ``E(mu)`` or
    ``|psi(mu)|^2``; training should simply avoid an epsilon-neighbourhood of
    ``mu = 0`` (e.g. sample ``mu`` in ``[0.05 t, 4 t]``).

    SIREN backbone
    --------------
    Sinusoidal representation layers ``y = sin(omega_0 (W x + b))`` provide
    smooth derivatives of the network output, which matters because the
    physics residual differentiates the solution with respect to ``mu``. The
    first layer uses ``omega_0 = 30`` (standard SIREN); hidden layers use
    ``hidden_omega_0``.

    Input scaling
    -------------
    The SIREN initialisation is calibrated for roughly unit-scale inputs.
    The absolute chemical potential ``|mu|`` is divided by ``input_scale``
    (default ``4.0``, matching the ``mu`` in ``[0, 4 t]`` training domain)
    before it reaches the first SIREN layer.

    Attributes:
        net: Shared SIREN feature-extraction network.
        psi_head: Linear projection producing the concatenated
            ``(u, v)`` prediction of width ``2 * n_sites``.
        D: Buffer holding ``(-1)^n``, used for the ``mu -> -mu`` fold.

    Args:
        n_sites: Number of physical lattice sites, ``N``. Note this is ``N``,
            not ``2N``: unlike :class:`kitaev.models.SirenPINN`, whose
            ``n_sites`` is the full BdG dimension, here the heads produce two
            ``N``-vectors.
        in_features: Number of input features, normally one (``mu``).
        hidden_features: Width of the shared SIREN representation.
        hidden_layers: Number of hidden SIREN layers after the first.
        hidden_omega_0: Frequency parameter for every hidden SIREN layer.
        input_scale: Divides ``|mu|`` before the first SIREN layer.
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
    ) -> None:
        """Initialise the chiral-basis SIREN PINN.

        Args:
            n_sites: Number of physical lattice sites, ``N`` (not ``2N``).
            in_features: Number of input features.
            hidden_features: Width of the shared SIREN representation.
            hidden_layers: Number of hidden SIREN layers following the
                first layer.
            hidden_omega_0: Frequency parameter used by every hidden SIREN
                layer (the first layer always uses ``30.0``). Exposed as a
                constructor argument so it can be swept in an ablation
                study.
            input_scale: Divides ``|mu|`` before it reaches the first SIREN
                layer, bringing the input to the roughly unit scale the
                SIREN initialisation expects. The default of ``4.0`` matches
                the ``mu`` in ``[0, 4 t]`` training domain.
        """
        super().__init__()

        self.n_sites = n_sites
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.hidden_omega_0 = hidden_omega_0
        self.input_scale = input_scale

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

        # Single head producing the concatenated (u, v) singular pair.
        self.psi_head = nn.Linear(self.hidden_features, 2 * self.n_sites)

        # Use the SIREN-style scaling associated with the hidden-layer
        # frequency for the linear prediction head.
        bound = math.sqrt(6.0 / self.hidden_features) / self.hidden_omega_0

        with torch.no_grad():
            self.psi_head.weight.uniform_(-bound, bound)

        signs = torch.tensor(
            [(-1.0) ** n for n in range(self.n_sites)],
            dtype=torch.get_default_dtype(),
        )
        self.register_buffer("D", signs)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the predicted singular pair ``(u, v)`` of ``h(mu)``.

        Only ``|mu|`` is passed through the SIREN backbone; the exact
        ``mu -> -mu`` transform ``u -> -D u``, ``v -> D v`` is applied
        row-wise wherever ``mu < 0`` (see the class docstring). Each head
        half is L2-normalised, so ``||u|| = ||v|| = 1`` by construction.

        Args:
            x: Input tensor of chemical-potential values, shape
                ``(batch_size, in_features)``. Values may be negative.

        Returns:
            A tuple ``(u, v)``, each of shape ``(batch_size, n_sites)`` and
            unit L2 norm along ``dim=1``.
        """
        magnitude = x.abs()
        is_negative = x < 0

        features = self.net(magnitude / self.input_scale)
        raw = self.psi_head(features)

        u = F.normalize(raw[:, : self.n_sites], p=2, dim=1, eps=1e-12)
        v = F.normalize(raw[:, self.n_sites :], p=2, dim=1, eps=1e-12)

        # Chemical-potential reflection: h(-mu) = -D h(mu) D, hence
        # u(-mu) = -D u(mu) and v(-mu) = D v(mu).
        u = torch.where(is_negative, -self.D * u, u)
        v = torch.where(is_negative, self.D * v, v)

        return u, v


class ChiralToBdGAdapter(nn.Module):
    """Presents a :class:`SirenPINNChiral` as a dual-head ``(E, psi)`` model.

    :class:`SirenPINNChiral` returns a singular pair ``(u, v)`` of the
    ``N x N`` chiral block ``h(mu)``. The project's evaluation and plotting
    utilities (:mod:`kitaev.visualisation`) instead expect a model that
    returns ``(E_pred, psi_pred)`` with ``psi_pred`` a ``2N`` BdG
    eigenvector in the Nambu basis, as :class:`kitaev.models.SirenPINNDualHead`
    does. This wrapper bridges the two so the existing sweeps work
    unchanged:

    - ``E_pred`` is the Rayleigh-quotient singular value
      ``lambda_R = u^T h(mu) v``, canonicalised onto the ``+lambda`` branch
      by :func:`kitaev.analytical.resolve_singular_branch` so that it is
      non-negative wherever the branch is well defined. This also fixes the
      particle/hole assignment of ``psi_pred``, which is otherwise free to
      flip with ``mu`` because the training loss is invariant under
      ``(u, v) -> (u, -v)``.
    - ``psi_pred = ((u + v) / 2, (u - v) / 2)`` (particle block, hole
      block), which is unit-norm by construction (see
      :func:`kitaev.analytical.reconstruct_bdg_eigenvector`).

    Args:
        model: The wrapped :class:`SirenPINNChiral` instance.
        hopping: Nearest-neighbour hopping amplitude, ``t``. Must match the
            value the model was trained against.
        pairing: P-wave pairing amplitude, ``delta``. Must likewise match.
    """

    def __init__(
        self,
        model: SirenPINNChiral,
        *,
        hopping: float = 1.0,
        pairing: float = 0.5,
    ) -> None:
        """Wrap a chiral model in the dual-head interface.

        Args:
            model: The :class:`SirenPINNChiral` to wrap.
            hopping: Hopping amplitude used to rebuild ``h(mu)``.
            pairing: Pairing amplitude used to rebuild ``h(mu)``.
        """
        super().__init__()
        self.model = model
        self.hopping = hopping
        self.pairing = pairing

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(E_pred, psi_pred)`` for the wrapped chiral model.

        Args:
            x: Chemical-potential values, shape ``(batch_size, in_features)``.

        Returns:
            A tuple ``(E_pred, psi_pred)`` of shapes ``(batch_size, 1)`` and
            ``(batch_size, 2 * n_sites)``.
        """
        u, v = self.model(x)
        h_batch = chiral_block_batched(
            x, self.model.n_sites, self.hopping, self.pairing
        )
        u, v, _h_v, energy_pred = resolve_singular_branch(u, v, h_batch)
        psi_pred = torch.cat([(u + v) / 2.0, (u - v) / 2.0], dim=1)
        return energy_pred, psi_pred
