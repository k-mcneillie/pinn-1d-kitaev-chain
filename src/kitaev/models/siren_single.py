# kitaev/pinn/model.py
"""SIREN-based PINN architecture for the Kitaev-chain surrogate."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from kitaev.analytical import bdg_block_batched

from .layers import SineLayer


class SirenPINN(nn.Module):
    """SIREN-based neural network with an eigenvector prediction head.

    This model represents a continuous surrogate for the eigenvector
    structure of the Kitaev-chain Bogoliubov-de Gennes (BdG) Hamiltonian.
    Rather than diagonalising the Hamiltonian explicitly for every value
    of the chemical potential, the network learns a mapping from the
    scalar control parameter ``mu`` to the corresponding BdG eigenvector.

    The architecture consists of a shared SIREN feature extractor followed
    by a single prediction head:

        mu -> SIREN backbone -> shared features -> eigenvector head -> Psi(mu)

    The SIREN backbone learns a continuous latent representation of the
    underlying spectral problem, while the eigenvector head maps this
    representation directly into the BdG vector space.

    SIREN backbone
    --------------
    The backbone uses sinusoidal representation layers rather than the
    conventional ReLU or tanh activations. A SIREN layer has the form

        y = sin(omega_0 * (W x + b)),

    where ``omega_0`` controls the frequency of the sinusoidal activation.

    Periodic activations are particularly useful for physics-informed
    problems because they provide smooth derivatives of the network output.
    This is important for PINNs because derivatives of the neural-network
    solution can appear explicitly in the physics residual.

    The first layer uses ``omega_0 = 30``. This follows the standard SIREN
    construction, where a relatively large first-layer frequency allows the
    network to represent high-frequency structure in the input-to-feature
    mapping.

    The subsequent hidden layers use ``omega_0 = hidden_omega_0``, a
    constructor argument defaulting to ``1.0``. This is deliberately
    different from the first layer's fixed ``30.0`` and, being an
    ordinary argument rather than a hardcoded constant, can be swept in
    an ablation study.

    Input scaling
    -------------
    The standard SIREN initialisation (see :class:`SineLayer`) is
    calibrated for roughly unit-scale inputs: the first layer's weights
    are drawn from ``[-1/in_features, 1/in_features]``, which combined
    with ``omega_0=30`` assumes the pre-activation ``Wx`` stays of
    order 1. This project's ``mu`` domain is ``[-3, 3]`` rather than
    ``[-1, 1]``, so ``x`` is divided by ``input_scale`` (default
    ``3.0``) before it reaches the first SIREN layer, bringing it back
    into the scale the initialisation expects.

    Eigenvector prediction head
    ----------------------------
    After the shared SIREN backbone produces a feature vector of dimension
    ``hidden_features``, a single linear projection maps this representation
    into the BdG eigenvector space.

    The eigenvector head maps

        R^hidden_features -> R^n_sites,

    producing the predicted BdG eigenvector ``Psi_pred``.

    The value of ``n_sites`` determines the output dimensionality supplied
    to the model. For the default Kitaev-chain configuration of N = 20
    sites, the BdG Hamiltonian has dimension 2N = 40, so the model should
    be instantiated with ``n_sites=40`` when predicting the full BdG
    eigenvector.

    Eigenvector normalisation
    -------------------------
    Eigenvectors have an arbitrary overall scale, so a predicted
    eigenvector must be constrained before it can be compared with the
    normalised eigenvectors obtained from exact diagonalisation.

    The model therefore applies an explicit L2 normalisation:

        Psi_pred -> Psi_pred / ||Psi_pred||_2.

    This constrains the predicted eigenvector to unit Euclidean norm and
    removes the arbitrary amplitude degree of freedom. It does not,
    however, remove the sign ambiguity of an eigenvector: both ``Psi`` and
    ``-Psi`` represent the same physical eigenstate.

    The normalisation is therefore a constraint on the representation of
    the eigenvector rather than an additional physical assumption.

    Initialisation
    --------------
    The SIREN backbone is assumed to perform its own layer-specific
    initialisation following the SIREN prescription.

    The output head is an ordinary linear layer. Its weights are
    initialised using

        bound = sqrt(6 / hidden_features) / hidden_omega_0,

    i.e. the same frequency used by the hidden SIREN layers. This
    preserves the SIREN-style scaling associated with the hidden-layer
    frequency, while the head itself remains a linear prediction layer.

    Attributes:
        net: Shared SIREN feature-extraction network.
        psi_head: Linear projection producing the BdG eigenvector
            prediction.

    Args:
        n_sites: Output dimensionality of the eigenvector prediction.
            For a Kitaev chain with ``N`` physical sites and a full BdG
            representation, this is normally ``2N``.
        in_features: Number of input features. For the Kitaev-chain
            surrogate this is normally one, corresponding to the chemical
            potential ``mu``.
        hidden_features: Number of features produced by each SIREN layer.
        hidden_layers: Number of hidden SIREN layers following the first
            SIREN layer.
        hidden_omega_0: Frequency parameter used by every hidden SIREN
            layer.
        input_scale: Divides the raw input before the first SIREN
            layer; see "Input scaling" above.
    """

    def __init__(
        self,
        n_sites: int,
        *,
        in_features: int = 1,
        hidden_features: int = 32,
        hidden_layers: int = 2,
        hidden_omega_0: float = 1.0,
        input_scale: float = 3.0,
    ) -> None:
        """Initialise the single-head SIREN PINN.

        Args:
            n_sites: Output dimensionality of the eigenvector prediction.
            in_features: Number of input features.
            hidden_features: Width of the shared SIREN representation.
            hidden_layers: Number of hidden SIREN layers following the
                first layer.
            hidden_omega_0: Frequency parameter used by every hidden
                SIREN layer (the first layer always uses ``30.0``,
                following the standard SIREN construction). Exposed as
                a constructor argument, rather than hardcoded, so it
                can be swept in an ablation study.
            input_scale: Divides the raw input before it reaches the
                first SIREN layer. The standard SIREN initialisation
                (see :class:`SineLayer`) is calibrated for roughly
                unit-scale inputs; without this rescaling, an input
                domain of e.g. ``mu`` in ``[-3, 3]`` combined with the
                first layer's ``omega_0=30`` drives the first sine
                activation through dozens of oscillation cycles at
                initialisation, well outside the regime the SIREN
                scheme was designed for. The default of ``3.0`` matches
                this project's standard ``mu`` sampling domain,
                ``[-3, 3]``; override it if a different physical domain
                is used.
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

        # Single eigenvector prediction head.
        self.psi_head = nn.Linear(self.hidden_features, self.n_sites)

        # Use the SIREN-style scaling associated with the hidden-layer
        # frequency for the linear prediction head.
        bound = math.sqrt(6.0 / self.hidden_features) / self.hidden_omega_0

        with torch.no_grad():
            self.psi_head.weight.uniform_(-bound, bound)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the predicted and normalised eigenvector.

        The input is first transformed by the shared SIREN backbone. The
        resulting feature representation is then passed to the eigenvector
        prediction head.

        The eigenvector head produces an ``n_sites``-dimensional prediction
        corresponding to the target BdG eigenvector.

        The eigenvector prediction is subsequently normalised to unit L2
        norm:

            Psi_pred = Psi_pred / ||Psi_pred||_2.

        This ensures that the predicted eigenvector has the same norm
        convention as a conventionally normalised eigenvector obtained
        from exact diagonalisation.

        Before reaching the SIREN backbone, ``x`` is divided by
        ``self.input_scale`` to bring it to the roughly unit scale the
        SIREN initialisation is calibrated for (see ``input_scale`` in
        :meth:`__init__`).

        Args:
            x: Input tensor containing the chemical potential values.
                Expected shape is ``(batch_size, in_features)``.

        Returns:
            L2-normalised predicted eigenvector with shape
            ``(batch_size, n_sites)``.
        """
        features = self.net(x / self.input_scale)

        psi_pred = self.psi_head(features)

        # Eigenvector normalisation
        # -------------------------
        # The psi head predicts an eigenvector in the BdG vector space:
        #
        #     Psi_pred in R^(2N)
        #
        # However, an eigenvector has an arbitrary overall scale. If Psi
        # is an eigenvector of H, then c * Psi is also an eigenvector for
        # any non-zero scalar c.
        #
        # Exact diagonalisation conventionally returns normalised
        # eigenvectors satisfying:
        #
        #     ||Psi||^2 = 1
        #
        # We therefore enforce this known mathematical constraint directly
        # in the architecture rather than asking the PINN loss to learn it.
        #
        # The normalisation maps the unconstrained prediction onto the
        # unit sphere in the BdG vector space:
        #
        #     Psi_hat = Psi_pred / ||Psi_pred||_2
        #
        # For N = 20 sites, the BdG space has dimension 2N = 40, so the
        # predicted eigenvector lies in R^40 and the normalised prediction
        # lies on the 39-dimensional unit sphere:
        #
        #     S^39 = {Psi in R^40 : ||Psi||_2 = 1}.
        #
        # This is an architectural constraint on the representation of
        # the eigenvector. It does NOT constrain the entire eigenvector
        # space to a single state, nor does it determine which eigenvector
        # the network should predict.
        #
        # The physical eigenvalue/eigenvector relationship is still
        # enforced separately through the PINN physics residual:
        #
        #     H(mu) Psi_hat(mu) - E_pred(mu) Psi_hat(mu) = 0.
        #
        # One remaining ambiguity is the global sign: Psi and -Psi
        # represent the same eigenstate. L2 normalisation removes the
        # arbitrary magnitude but not this sign ambiguity.
        #
        # Normalisation is therefore an eigenvector constraint arising
        # from the spectral problem, not a requirement of the SIREN
        # architecture itself.
        # ================================================================
        psi_pred = F.normalize(
            psi_pred,
            p=2,
            dim=1,
            eps=1e-12,
        )

        return psi_pred


class RayleighEnergyAdapter(nn.Module):
    """Attach a Rayleigh-quotient energy to a model that returns only ``psi``.

    :class:`SirenPINN` returns a single ``2N``-component Nambu-basis
    eigenvector; the project's evaluation and plotting utilities
    (:mod:`kitaev.visualisation`) and :class:`kitaev.training.BdGEvaluationProbe`
    expect a model that returns ``(E_pred, psi_pred)``, as
    :class:`SirenPINNDualHead` does. This wrapper bridges the two, the
    Nambu-basis analogue of :class:`kitaev.models.ChiralToBdGAdapter`:

    - ``psi_pred`` is ``model(mu)`` unchanged -- already unit norm.
    - ``E_pred`` is the Rayleigh quotient
      ``E_R = psi^T H(mu) psi`` with ``H(mu)`` from
      :func:`kitaev.analytical.bdg_block_batched`, the same energy estimate
      :class:`kitaev.training.loss.PinnedFSMLoss` forms internally.

    Unlike :class:`ChiralToBdGAdapter`, ``E_R`` is returned **signed** and
    no branch canonicalisation is applied: :class:`SirenPINN` +
    :class:`~kitaev.training.loss.PinnedFSMLoss` has only the soft
    ``loss_pin`` term selecting ``E_R >= 0``, not an architectural
    guarantee, so a sign flip or a particle/hole swap in the trivial phase
    is a genuine property of the baseline model rather than a gauge to be
    removed here. Consumers that want a non-negative energy (the probe, the
    energy sweep) take the absolute value themselves.

    Args:
        model: The wrapped model, returning a ``(batch, 2 * n_sites)``
            unit-norm eigenvector for a ``mu`` batch.
        n_sites: Number of physical lattice sites, ``N`` (so the BdG
            matrix is ``2N x 2N``). This is the physical site count, not
            ``model.n_sites``, which for :class:`SirenPINN` is the output
            width ``2N``.
        hopping: Nearest-neighbour hopping amplitude, ``t``. Must match the
            value the model was trained against.
        pairing: P-wave pairing amplitude, ``delta``. Must likewise match.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        n_sites: int,
        hopping: float = 1.0,
        pairing: float = 0.5,
    ) -> None:
        """Wrap a bare-eigenvector model in the dual-head ``(E, psi)`` interface."""
        super().__init__()
        self.model = model
        self.n_sites = n_sites
        self.hopping = hopping
        self.pairing = pairing

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(E_pred, psi_pred)`` for the wrapped model.

        Args:
            x: Chemical-potential values, shape ``(batch_size, in_features)``.

        Returns:
            A tuple ``(E_pred, psi_pred)`` with ``E_pred`` of shape
            ``(batch_size, 1)`` (the signed Rayleigh quotient) and
            ``psi_pred`` of shape ``(batch_size, 2 * n_sites)``.
        """
        psi_pred = self.model(x)
        h_batch = bdg_block_batched(x, self.n_sites, self.hopping, self.pairing)
        e_pred = torch.einsum("bi,bij,bj->b", psi_pred, h_batch, psi_pred)
        return e_pred.unsqueeze(-1), psi_pred
