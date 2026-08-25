# kitaev/pinn/model.py
"""SIREN-based PINN architecture for the Kitaev-chain surrogate."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from .layers import SineLayer


class SirenPINNDualHead(nn.Module):
    """SIREN-based neural network with separate energy and eigenvector heads.

    This model represents a continuous surrogate for the spectral properties
    of the Kitaev-chain Bogoliubov-de Gennes (BdG) Hamiltonian. Rather than
    diagonalising the Hamiltonian explicitly for every value of the chemical
    potential, the network learns a mapping from the scalar control parameter
    ``mu`` to the corresponding spectral quantities.

    The architecture consists of a shared SIREN feature extractor followed
    by two independent prediction heads:

        mu -> SIREN backbone -> shared features
                              |-> energy head -> E(mu)
                              |-> eigenvector head -> Psi(mu)

    The shared backbone allows the two predictions to be represented using
    a common latent representation of the underlying spectral problem,
    while the separate heads allow the scalar energy and vector-valued
    eigenstate to be learned as distinct quantities.

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
    constructor argument defaulting to ``2.0``. This is deliberately
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

    Dual prediction heads
    ---------------------
    After the shared SIREN backbone produces a feature vector of dimension
    ``hidden_features``, two independent linear projections produce the
    required spectral quantities.

    The energy head maps

        R^hidden_features -> R,

    producing a scalar prediction ``E_pred`` for the target energy.

    The eigenvector head maps

        R^hidden_features -> R^(2N),

    producing the predicted BdG eigenvector ``Psi_pred``. For the default
    Kitaev-chain configuration of N = 20 sites, the BdG Hamiltonian has
    dimension 2N = 40, which is why the current implementation uses an
    output dimension of 40.

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

    Energy non-negativity
    ----------------------
    The target quantity, the lowest non-negative eigenvalue of the BdG
    Hamiltonian, is by construction non-negative (see
    :meth:`kitaev.analytical.KitaevChainHamiltonian.build`). Unlike
    ``PinnedFSMLoss``, which only *softly* encourages a non-negative
    Rayleigh quotient via an annealed penalty, this model enforces
    non-negativity architecturally: the raw energy-head output is
    passed through a softplus,

        E_pred = softplus(raw_output, beta=energy_softplus_beta),

    so ``E_pred`` can never be negative regardless of what the network
    has learned so far. Softplus rather than ReLU specifically, because
    a large fraction of this domain (the entire topological phase) has
    a true target of ``E == 0``: ReLU's derivative is exactly zero for
    every negative pre-activation, so once the raw output drifts
    negative in that region — which is most of the time, since zero is
    the target — gradient descent would have no signal to correct it.
    Softplus remains smooth and keeps a non-vanishing gradient
    (``sigmoid(raw_output)``) throughout, consistent with the rest of
    this architecture's preference for smooth, differentiable
    constraints over ones with dead zones or kinks.

    Initialisation
    --------------
    The SIREN backbone is assumed to perform its own layer-specific
    initialisation following the SIREN prescription.

    The two output heads are ordinary linear layers. Their weights are
    initialised using

        bound = sqrt(6 / hidden_features) / hidden_omega_0,

    i.e. the same frequency used by the hidden SIREN layers. This
    preserves the SIREN-style scaling associated with the hidden-layer
    frequency, while the heads themselves remain linear prediction layers.

    Attributes:
        net: Shared SIREN feature-extraction network.
        energy_head: Linear projection producing the scalar energy
            prediction.
        psi_head: Linear projection producing the BdG eigenvector
            prediction.

    Args:
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
        energy_softplus_beta: Softplus sharpness for the energy head;
            see "Energy non-negativity" above.
    """

    def __init__(
        self,
        n_sites: int,
        *,
        in_features: int = 1,
        hidden_features: int = 32,
        hidden_layers: int = 2,
        hidden_omega_0: float = 2.0,
        input_scale: float = 3.0,
        energy_softplus_beta: float = 10.0,
    ) -> None:
        """Initialise the dual-head SIREN PINN.

        Args:
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
            energy_softplus_beta: Sharpness of the softplus applied to
                the energy head's raw output (see "Energy
                non-negativity" above). Higher values approach ReLU
                more closely (a sharper, more ReLU-like transition
                near zero) while remaining smooth everywhere.
        """
        super().__init__()

        self.n_sites = n_sites
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.hidden_omega_0 = hidden_omega_0
        self.input_scale = input_scale
        self.energy_softplus_beta = energy_softplus_beta

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

        # Separate prediction heads.
        self.energy_head = nn.Linear(self.hidden_features, 1)
        self.psi_head = nn.Linear(self.hidden_features, self.n_sites)

        # Use the SIREN-style scaling associated with the hidden-layer
        # frequency for the linear prediction heads.
        bound = math.sqrt(6.0 / self.hidden_features) / self.hidden_omega_0

        with torch.no_grad():
            self.energy_head.weight.uniform_(-bound, bound)
            self.psi_head.weight.uniform_(-bound, bound)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute predicted energy and normalised eigenvector.

        The input is first transformed by the shared SIREN backbone. The
        resulting feature representation is then passed independently to
        the energy and eigenvector heads.

        The energy head produces a scalar prediction for each input sample,
        while the eigenvector head produces a 40-dimensional prediction
        corresponding to the 40-dimensional BdG eigenvector of the default
        20-site Kitaev chain.

        The eigenvector prediction is subsequently normalised to unit L2
        norm:

            Psi_pred = Psi_pred / ||Psi_pred||_2.

        This ensures that the predicted eigenvector has the same norm
        convention as a conventionally normalised eigenvector obtained
        from exact diagonalisation.

        The energy prediction is passed through a softplus so it can
        never be negative (see "Energy non-negativity" in the class
        docstring), and, before reaching the SIREN backbone, ``x`` is
        divided by ``self.input_scale`` to bring it to the roughly
        unit scale the SIREN initialisation is calibrated for (see
        ``input_scale`` in :meth:`__init__`).

        Args:
            x: Input tensor containing the chemical potential values.
                Expected shape is ``(batch_size, in_features)``.

        Returns:
            A tuple containing:

                E_pred:
                    Predicted energy with shape ``(batch_size, 1)``,
                    guaranteed non-negative.

                psi_pred:
                    L2-normalised predicted eigenvector with shape
                    ``(batch_size, 40)``.
        """
        features = self.net(x / self.input_scale)

        energy_pred = F.softplus(
            self.energy_head(features), beta=self.energy_softplus_beta
        )
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

        return energy_pred, psi_pred
