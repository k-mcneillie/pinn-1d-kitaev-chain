# kitaev/models/siren_folded.py
"""SIREN-based PINN in the Nambu basis with the mu -> -mu fold built in."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from .layers import SineLayer


class SirenPINNNambuFolded(nn.Module):
    """SIREN surrogate for the near-zero BdG eigenvector, with a structural fold.

    This model works in the full ``2N`` Nambu basis, exactly like
    :class:`kitaev.models.SirenPINN`, but bakes the Kitaev chain's
    ``mu -> -mu`` reflection symmetry into the forward pass so that the
    predicted spectrum is even in ``mu`` by construction rather than by
    training. It is the middle rung of the project's unsupervised study,
    between the soft-constraint baseline (:class:`SirenPINN` +
    :class:`~kitaev.training.loss.PinnedFSMLoss`) and the chiral reduction
    (:class:`SirenPINNChiral` + :class:`~kitaev.training.loss.ChiralFSMLoss`).

        mu -> SIREN backbone -> shared features -> psi head -> psi(mu) in R^2N

    The reflection operator
    ---------------------------
    Let ``D = diag((-1)^n)`` act on an ``N``-vector and, for a Nambu vector
    ``psi = (p, h)`` (particle block ``p``, hole block ``h``, each ``R^N``),
    define

        Gamma psi = (-D h, -D p).

    ``Gamma = -(tau_x (x) D)`` is real, symmetric, orthogonal and involutory
    (``Gamma^2 = I``), and it implements the exact BDI reflection

        Gamma H(mu) Gamma = H(-mu)

    of the batched BdG Hamiltonian (the Nambu-basis form of the chiral
    block's ``h(-mu) = -D h(mu) D``). It depends on neither ``t`` nor
    ``Delta``, so this model takes no ``hopping`` / ``pairing`` arguments.

    The symmetrised backbone
    ------------------------
    With ``g`` the raw backbone-plus-head map, the forward pass returns

        psi(mu) = normalise( 0.5 * [ g(mu) + Gamma g(-mu) ] ).

    Because ``Gamma^2 = I`` and ``F.normalize`` commutes with the orthogonal
    ``Gamma``, this satisfies ``psi(-mu) = Gamma psi(mu)`` **exactly**, and
    hence

        E_R(-mu) = psi(-mu)^T H(-mu) psi(-mu)
                 = psi(mu)^T Gamma H(-mu) Gamma psi(mu)
                 = psi(mu)^T H(mu) psi(mu) = E_R(mu)

    for the Rayleigh-quotient energy. Training therefore only needs to cover
    ``mu >= 0``; the negative half is a fixed linear image. The cost is two
    backbone evaluations per forward pass, negligible at ``N = 20``.

    Structural guarantees
    ---------------------
    - ``||psi|| = 1`` (explicit L2 normalisation of the symmetrised output).
    - ``psi(-mu) = Gamma psi(mu)`` and hence ``E_R(-mu) = E_R(mu)``, exactly.

    Left to the loss (:class:`~kitaev.training.loss.NambuFSMLoss`): that
    ``psi`` is an eigenvector at all; the ``+-E`` branch (a global gauge
    under ``psi -> Xi psi``, fixed at evaluation by sign-alignment to the
    reference, not by a penalty); and, in the topological phase, which
    vector of the two-dimensional near-zero Majorana manifold is returned.

    The ``mu = 0`` point
    --------------------
    Unlike :class:`SirenPINNChiral`, whose ``-D`` factor makes a single
    vector sign-discontinuous at ``mu = 0``, the fold here forces
    ``psi(0) = normalise( 0.5 (I + Gamma) g(0) )``, the projection onto the
    ``+1`` eigenspace of the orthogonal ``Gamma`` -- well defined and
    continuous through ``mu = 0``. Sampling ``mu`` down to ``0.0`` is
    therefore valid; no epsilon exclusion is needed. The only degenerate
    case is the measure-zero event that ``g(0)`` lands entirely in
    ``ker(Gamma + I)``, guarded by the ``eps`` in :func:`F.normalize`.

    SIREN backbone
    --------------
    Sinusoidal representation layers ``y = sin(omega_0 (W x + b))`` give the
    smooth ``mu``-derivatives the physics residual needs. The first layer
    uses ``omega_0 = 30`` (standard SIREN); hidden layers use
    ``hidden_omega_0``.

    Input scaling
    -------------
    ``x`` is divided by ``input_scale`` (default ``4.0``, matching
    ``|mu| <= 4 t``) before the first SIREN layer, bringing it to the
    roughly unit scale the SIREN initialisation expects.

    Attributes:
        net: Shared SIREN feature-extraction network.
        psi_head: Linear projection producing the raw ``2N`` Nambu vector.
        D: Buffer holding ``(-1)^n`` of length ``N``, used to apply
            ``Gamma``.

    Args:
        n_sites: Output dimensionality of the eigenvector prediction, the
            full BdG dimension ``2N``. This is the **same** convention as
            :class:`SirenPINN` and deliberately unlike
            :class:`SirenPINNChiral` (whose ``n_sites`` is ``N``). Must be
            even.
        in_features: Number of input features, normally one (``mu``).
        hidden_features: Width of the shared SIREN representation.
        hidden_layers: Number of hidden SIREN layers after the first.
        hidden_omega_0: Frequency parameter for every hidden SIREN layer.
        input_scale: Divides the raw input before the first SIREN layer.
    """

    D: torch.Tensor

    def __init__(
        self,
        n_sites: int,
        *,
        in_features: int = 1,
        hidden_features: int = 64,
        hidden_layers: int = 2,
        hidden_omega_0: float = 1.0,
        input_scale: float = 4.0,
    ) -> None:
        """Initialise the folded Nambu-basis SIREN PINN.

        Args:
            n_sites: Full BdG output dimension ``2N`` (must be even), the
                same convention as :class:`SirenPINN`.
            in_features: Number of input features.
            hidden_features: Width of the shared SIREN representation.
            hidden_layers: Number of hidden SIREN layers following the
                first layer.
            hidden_omega_0: Frequency parameter used by every hidden SIREN
                layer (the first layer always uses ``30.0``). Exposed as a
                constructor argument so it can be swept in an ablation
                study.
            input_scale: Divides the raw input before it reaches the first
                SIREN layer. The default of ``4.0`` matches the
                ``|mu| <= 4 t`` domain.

        Raises:
            ValueError: If ``n_sites`` is not even.
        """
        super().__init__()

        if n_sites % 2 != 0:
            raise ValueError(
                f"n_sites must be even (the full BdG dimension 2N); got {n_sites}"
            )

        self.n_sites = n_sites
        self.n_phys = n_sites // 2
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

        # Single head producing the raw 2N Nambu vector.
        self.psi_head = nn.Linear(self.hidden_features, self.n_sites)

        # Use the SIREN-style scaling associated with the hidden-layer
        # frequency for the linear prediction head.
        bound = math.sqrt(6.0 / self.hidden_features) / self.hidden_omega_0

        with torch.no_grad():
            self.psi_head.weight.uniform_(-bound, bound)

        signs = torch.tensor(
            [(-1.0) ** n for n in range(self.n_phys)],
            dtype=torch.get_default_dtype(),
        )
        self.register_buffer("D", signs)

    def _fold(self, psi: torch.Tensor) -> torch.Tensor:
        """Apply the reflection operator ``Gamma psi = (-D h, -D p)``.

        Args:
            psi: Nambu vectors, shape ``(batch_size, 2N)``, split into a
                particle block ``psi[:, :N]`` and a hole block
                ``psi[:, N:]``.

        Returns:
            ``Gamma psi`` of the same shape: the particle/hole blocks
            swapped, each multiplied by ``-D``.
        """
        particle = psi[:, : self.n_phys]
        hole = psi[:, self.n_phys :]
        return torch.cat((-self.D * hole, -self.D * particle), dim=1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the normalised, reflection-folded Nambu eigenvector.

        The backbone is evaluated at both ``+mu`` and ``-mu``; the two raw
        predictions are combined as ``0.5 * [ g(mu) + Gamma g(-mu) ]`` and
        L2-normalised. The result satisfies ``psi(-mu) = Gamma psi(mu)``
        exactly (see the class docstring). ``x`` is divided by
        ``input_scale`` before the first SIREN layer.

        Args:
            x: Input tensor of chemical-potential values, shape
                ``(batch_size, in_features)``. Values may be negative.

        Returns:
            L2-normalised predicted eigenvector, shape
            ``(batch_size, 2N)``.
        """
        raw_pos = self.psi_head(self.net(x / self.input_scale))
        raw_neg = self.psi_head(self.net(-x / self.input_scale))

        symmetrised = 0.5 * (raw_pos + self._fold(raw_neg))

        return F.normalize(symmetrised, p=2, dim=1, eps=1e-12)
