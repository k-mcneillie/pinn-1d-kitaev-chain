# tests/models/test_siren_dual.py
from __future__ import annotations

from unittest import TestCase

import numpy as np
import torch

from kitaev.models import SirenPINNDualHead


class TestSirenPINNDualHead(TestCase):
    """Comprehensive test suite for the SirenPINNDualHead class."""

    def test_construction(self):
        """Test instance initialization and basic properties."""
        model = SirenPINNDualHead(n_sites=2)
        self.assertEqual(model.n_sites, 2)
        self.assertEqual(len(model.net), 3)  # 1 input layer + 2 hidden
        self.assertEqual(model.net[0].omega_0, 30.0)
        self.assertEqual(model.net[1].omega_0, 2.0)
        self.assertEqual(model.net[2].omega_0, 2.0)
        self.assertEqual(model.energy_head.out_features, 1)
        self.assertEqual(model.psi_head.out_features, 2)

    def test_head_weight_initialisation(self):
        """Test head weight initializations."""
        model = SirenPINNDualHead(n_sites=2)
        hidden_features = model.hidden_features
        bound = np.sqrt(6.0 / hidden_features) / model.hidden_omega_0
        with torch.no_grad():
            self.assertTrue(
                torch.all(torch.abs(model.energy_head.weight) <= bound * 1.0001)
            )
            self.assertTrue(
                torch.all(torch.abs(model.psi_head.weight) <= bound * 1.0001)
            )

    def test_forward_shapes_and_norm(self):
        """Test forward pass shapes and output normalisation."""
        model = SirenPINNDualHead(n_sites=2)
        x = torch.randn(3, 1)
        energy_pred, psi_pred = model(x)
        self.assertEqual(energy_pred.shape, (3, 1))
        self.assertEqual(psi_pred.shape, (3, 2))
        norms = torch.norm(psi_pred, p=2, dim=1)
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), rtol=1e-3, atol=1e-3)
        )

    def test_custom_hidden_omega_0_is_used_by_hidden_layers_and_heads(self):
        """hidden_omega_0 must be a real, swept-able constructor argument."""
        model = SirenPINNDualHead(n_sites=2, hidden_omega_0=5.0)
        self.assertEqual(model.net[0].omega_0, 30.0)  # first layer unaffected
        self.assertEqual(model.net[1].omega_0, 5.0)
        self.assertEqual(model.net[2].omega_0, 5.0)

        bound = np.sqrt(6.0 / model.hidden_features) / 5.0
        with torch.no_grad():
            self.assertTrue(
                torch.all(torch.abs(model.energy_head.weight) <= bound * 1.0001)
            )

    def test_input_scale_divides_input_before_backbone(self):
        """The raw input must be divided by input_scale before the SIREN net."""
        model = SirenPINNDualHead(n_sites=2, input_scale=5.0)
        x = torch.tensor([[10.0], [-7.5]])

        with torch.no_grad():
            expected_features = model.net(x / 5.0)
            expected_energy = torch.nn.functional.softplus(
                model.energy_head(expected_features),
                beta=model.energy_softplus_beta,
            )

        energy_pred, _ = model(x)
        self.assertTrue(torch.allclose(energy_pred, expected_energy))

    def test_energy_head_output_is_always_non_negative(self):
        """E_pred must stay non-negative even from a strongly negative raw output."""
        model = SirenPINNDualHead(n_sites=2)
        with torch.no_grad():
            model.energy_head.weight.fill_(-100.0)
            model.energy_head.bias.fill_(-100.0)

        x = torch.randn(20, 1) * 5.0
        energy_pred, _ = model(x)

        self.assertTrue(torch.all(energy_pred >= 0.0))


if __name__ == "__main__":
    import pytest

    pytest.main()
