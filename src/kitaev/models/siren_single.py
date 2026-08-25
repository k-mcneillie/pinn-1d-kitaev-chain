# kitaev/pinn/model.py
"""SIREN-based PINN architecture for the Kitaev-chain surrogate."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

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

    The subsequent hidden layers use ``omega_0 = 2`` in this architecture.
    The frequency parameter is therefore deliberately different from the
    first layer and is treated as an architectural hyperparameter.

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

        bound = sqrt(6 / hidden_features) / omega_0,

    with ``omega_0 = 2`` corresponding to the frequency used by the
    hidden SIREN layers. Thus the implementation uses

        bound = sqrt(6 / hidden_features) / 2.

    This preserves the SIREN-style scaling associated with the hidden-layer
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
    """

    def __init__(
        self,
        n_sites: int,
        *,
        in_features: int = 1,
        hidden_features: int = 32,
        hidden_layers: int = 2,
    ) -> None:
        """Initialise the single-head SIREN PINN.

        Args:
            n_sites: Output dimensionality of the eigenvector prediction.
            in_features: Number of input features.
            hidden_features: Width of the shared SIREN representation.
            hidden_layers: Number of hidden SIREN layers following the
                first layer.
        """
        super().__init__()

        self.n_sites = n_sites
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers

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
                    omega_0=1.0,
                )
            )

        self.net = nn.Sequential(*layers)

        # Single eigenvector prediction head.
        self.psi_head = nn.Linear(self.hidden_features, self.n_sites)

        # Use the SIREN-style scaling associated with the hidden-layer
        # frequency omega_0 = 2.0 for the linear prediction head.
        bound = math.sqrt(6.0 / self.hidden_features) / 2.0

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

        Args:
            x: Input tensor containing the chemical potential values.
                Expected shape is ``(batch_size, in_features)``.

        Returns:
            L2-normalised predicted eigenvector with shape
            ``(batch_size, n_sites)``.
        """
        features = self.net(x)

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
