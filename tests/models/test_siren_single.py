# tests/models/test_siren_single.py
from __future__ import annotations

from unittest import TestCase

import numpy as np
import torch

from kitaev.models.siren_single import SirenPINN


class TestSirenPINN(TestCase):
    """Comprehensive test suite for the SirenPINN class."""

    def test_construction(self):
        """Test instance initialization and basic properties."""
        model = SirenPINN(n_sites=2)
        self.assertEqual(model.n_sites, 2)
        self.assertEqual(len(model.net), 3)  # 1 input layer + 2 hidden
        self.assertEqual(model.net[0].omega_0, 30.0)
        self.assertEqual(model.net[1].omega_0, 1.0)
        self.assertEqual(model.net[2].omega_0, 1.0)
        self.assertEqual(model.psi_head.out_features, 2)

    def test_head_weight_initialisation(self):
        """Test psi_head weight initialization."""
        model = SirenPINN(n_sites=2)
        hidden_features = model.hidden_features
        bound = np.sqrt(6.0 / hidden_features) / model.hidden_omega_0
        with torch.no_grad():
            self.assertTrue(
                torch.all(torch.abs(model.psi_head.weight) <= bound * 1.0001)
            )

    def test_forward_shape(self):
        """Test forward pass shape."""
        model = SirenPINN(n_sites=2)
        x = torch.randn(3, 1)
        output = model(x)
        self.assertEqual(output.shape, (3, 2))

    def test_output_normalised(self):
        """Test eigenvector output normalisation."""
        model = SirenPINN(n_sites=2)
        x = torch.randn(3, 1)
        output = model(x)
        norms = torch.norm(output, p=2, dim=1)
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), rtol=1e-3, atol=1e-3)
        )

    def test_custom_params(self):
        """Test model with non-default parameters."""
        model = SirenPINN(
            n_sites=3,
            in_features=2,
            hidden_features=16,
            hidden_layers=4,
        )
        self.assertEqual(model.n_sites, 3)
        self.assertEqual(len(model.net), 5)  # 1 input + 4 hidden
        self.assertEqual(model.net[0].omega_0, 30.0)
        self.assertEqual(model.net[1].omega_0, 1.0)
        self.assertEqual(model.net[-1].omega_0, 1.0)
        self.assertEqual(model.psi_head.out_features, 3)

    def test_differentiability(self):
        """Test output differentiability."""
        model = SirenPINN(n_sites=2)
        x = torch.randn(1, 1, requires_grad=True)

        output = model(x)

        self.assertTrue(output.requires_grad)

        x.grad = None
        output.sum().backward()

        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertFalse(torch.isnan(x.grad).any())

    def test_custom_hidden_omega_0_is_used_by_hidden_layers_and_head(self):
        """hidden_omega_0 must be a real, swept-able constructor argument."""
        model = SirenPINN(n_sites=2, hidden_omega_0=5.0)
        self.assertEqual(model.net[0].omega_0, 30.0)  # first layer unaffected
        self.assertEqual(model.net[1].omega_0, 5.0)
        self.assertEqual(model.net[2].omega_0, 5.0)

        bound = np.sqrt(6.0 / model.hidden_features) / 5.0
        with torch.no_grad():
            self.assertTrue(
                torch.all(torch.abs(model.psi_head.weight) <= bound * 1.0001)
            )

    def test_input_scale_divides_input_before_backbone(self):
        """The raw input must be divided by input_scale before the SIREN net."""
        model = SirenPINN(n_sites=2, input_scale=5.0)
        x = torch.tensor([[10.0], [-7.5]])

        with torch.no_grad():
            expected_features = model.net(x / 5.0)
            expected_psi = torch.nn.functional.normalize(
                model.psi_head(expected_features), p=2, dim=1, eps=1e-12
            )

        psi_pred = model(x)
        self.assertTrue(torch.allclose(psi_pred, expected_psi))


if __name__ == "__main__":
    import pytest

    pytest.main()
