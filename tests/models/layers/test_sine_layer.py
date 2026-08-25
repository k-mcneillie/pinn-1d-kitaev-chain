# tests/models/test_sine_layer.py
from __future__ import annotations

from unittest import TestCase

import numpy as np
import torch
from torch.nn import Linear

from kitaev.models.layers import SineLayer


class TestSineLayer(TestCase):
    """Comprehensive test suite for the SineLayer class."""

    def test_defaults(self):
        """Test default initialization parameters."""
        layer = SineLayer(1, 1)
        self.assertEqual(layer.in_features, 1)
        self.assertEqual(layer.out_features, 1)
        self.assertEqual(layer.omega_0, 30.0)
        self.assertFalse(layer.is_first)
        self.assertIsInstance(layer.linear, Linear)
        self.assertTrue(hasattr(layer.linear, "bias"))

    def test_init_first_layer(self):
        """Test weight initialization for first layer."""
        layer = SineLayer(2, 4, is_first=True)
        bound = 1.0 / layer.in_features
        with torch.no_grad():
            self.assertTrue(torch.all(torch.abs(layer.linear.weight) <= bound * 1.0001))

    def test_init_hidden_layer(self):
        """Test weight initialization for hidden layers."""
        layer = SineLayer(2, 4, is_first=False, omega_0=2.0)
        bound = np.sqrt(6.0 / layer.in_features) / layer.omega_0
        with torch.no_grad():
            self.assertTrue(torch.all(torch.abs(layer.linear.weight) <= bound * 1.0001))

    def test_bias_false(self):
        """Test bias=False configuration."""
        layer = SineLayer(1, 1, bias=False)
        self.assertIsNone(layer.linear.bias)

    def test_forward_shape_and_value(self):
        """Test forward pass shape and value."""
        layer = SineLayer(1, 1, omega_0=1.0)
        x = torch.randn(3, 1)
        output = layer(x)
        self.assertEqual(output.shape, (3, 1))
        self.assertTrue(
            torch.allclose(
                output,
                torch.sin(layer.linear(x)),
                rtol=1e-3,
                atol=1e-3,
            )
        )

    def test_init_weights_callable(self):
        """Test init_weights() can be called directly."""
        layer = SineLayer(2, 4)
        layer.init_weights()


if __name__ == "__main__":
    import pytest

    pytest.main()
