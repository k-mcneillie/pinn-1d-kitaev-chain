# kitaev/models/layers/sine_layer.py
#
# ==========================
# Import Packages
# ==========================
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# ==========================
# Sine Layer
# ==========================


class SineLayer(nn.Module):
    """A single sinusoidal-activation layer for a SIREN network.

    A SIREN (SInusoidal REpresentation Network) replaces the usual
    ReLU/tanh-style activation with a sine, ``sin(omega_0 * (Wx + b))``.
    Unlike ReLU networks, whose derivatives are piecewise constant, a
    sine activation is infinitely differentiable and every derivative is
    itself a sinusoid. This matters directly for this project, since the
    physics-residual losses (``||H psi - E psi||^2`` and friends)
    require differentiating the network's output with respect to its
    input (mu) to enforce the eigenvalue equation. A ReLU-based network
    would have discontinuous or zero higher-order derivatives almost
    everywhere, making such residual losses far less well behaved.

    The ``omega_0`` frequency parameter scales the pre-activation before
    the sine is applied. It controls how rapidly the layer's output can
    oscillate as a function of its input, which in turn controls how
    sharp a feature (e.g. the steep, near-discontinuous change in
    edge-mode localisation near the Kitaev chain's topological
    transition) the network can represent.

    Standard SIREN design (Sitzmann et al., 2020) recommends using a
    *high* omega_0 (commonly 30.0) on the first layer only, with a lower
    omega_0 (commonly close to 1.0) on subsequent hidden layers. The
    first-layer value sets the frequency spectrum available to the whole
    network, since a low first-layer omega_0 restricts the range of
    frequencies later layers can compose into sharper features.

    The initialisation scheme differs for the first layer versus later
    layers precisely because of this omega_0 scaling:

    * First layer (``is_first=True``): weights are drawn uniformly from
      ``[-1/in_features, 1/in_features]``. This layer sees the raw,
      un-rescaled input directly, so its initialisation does not need to
      account for omega_0.
    * Subsequent layers (``is_first=False``): weights are drawn
      uniformly from
      ``[-sqrt(6/in_features)/omega_0, sqrt(6/in_features)/omega_0]``, a
      Kaiming/Glorot-style bound rescaled by ``1/omega_0``. This
      compensates for omega_0 multiplying the pre-activation before the
      sine is applied, keeping the distribution of activations stable
      across layers and preventing the sine's output from being driven
      into its saturating regions at initialisation.

    Attributes:
        omega_0: Frequency scaling factor applied before the sine
            activation.
        is_first: Whether this is the first layer of the SIREN network.
        in_features: Number of input features to this layer.
        linear: The underlying affine transform applied before the sine.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        """Initialises the layer and its weights.

        Args:
            in_features: Number of input features to this layer.
            out_features: Number of output features produced by this
                layer.
            bias: Whether the underlying linear transform includes a
                learnable bias term.
            is_first: Whether this is the first layer of the SIREN
                network. Governs which of the two initialisation
                schemes described in the class docstring is used.
            omega_0: Frequency scaling factor applied before the sine
                activation. Conventionally kept high (around 30.0) on
                the first layer only and lower on hidden layers.
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear = nn.Linear(self.in_features, self.out_features, bias=self.bias)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialises the layer's weights according to the SIREN scheme.

        Uses the wider, omega_0-independent bound for the first layer
        and the narrower, omega_0-rescaled bound for all subsequent
        layers, as described in the class docstring. Runs under
        ``torch.no_grad()`` since this is a direct in-place weight
        initialisation, not part of the autograd-tracked forward pass.
        """
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = np.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """Applies the affine transform followed by the scaled sine activation.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``, equal to
            ``sin(omega_0 * linear(x))``.
        """
        return torch.sin(self.omega_0 * self.linear(x))
