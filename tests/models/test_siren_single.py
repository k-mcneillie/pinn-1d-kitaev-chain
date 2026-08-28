# tests/models/test_siren_single.py
from __future__ import annotations

from unittest import TestCase

import numpy as np
import torch

from kitaev.analytical import KitaevChainHamiltonian, bdg_block_batched
from kitaev.models.siren_single import RayleighEnergyAdapter, SirenPINN


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


class _ConstEigenvectorModel(torch.nn.Module):
    """Returns a fixed eigenvector for every ``mu`` in the batch."""

    def __init__(self, psi: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_psi", psi)
        self.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._psi.unsqueeze(0).expand(x.shape[0], -1)


class TestRayleighEnergyAdapter(TestCase):
    """Test suite for ``RayleighEnergyAdapter``."""

    N_SITES = 5
    HOPPING = 1.0
    PAIRING = 0.5

    def _adapter(self, model: torch.nn.Module) -> RayleighEnergyAdapter:
        return RayleighEnergyAdapter(
            model,
            n_sites=self.N_SITES,
            hopping=self.HOPPING,
            pairing=self.PAIRING,
        )

    def test_forward_shapes(self):
        """(E, psi) with E of shape (B, 1) and psi of shape (B, 2N)."""
        model = SirenPINN(n_sites=2 * self.N_SITES)
        e_pred, psi_pred = self._adapter(model)(torch.randn(4, 1))
        self.assertEqual(e_pred.shape, (4, 1))
        self.assertEqual(psi_pred.shape, (4, 2 * self.N_SITES))

    def test_psi_is_passed_through_unchanged(self):
        """psi_pred is exactly model(mu), still unit norm."""
        model = SirenPINN(n_sites=2 * self.N_SITES)
        x = torch.randn(6, 1)
        _, psi_pred = self._adapter(model)(x)
        self.assertTrue(torch.equal(psi_pred, model(x)))
        norms = torch.norm(psi_pred, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_energy_is_the_rayleigh_quotient(self):
        """E_pred equals psi^T H(mu) psi recomputed independently."""
        model = SirenPINN(n_sites=2 * self.N_SITES)
        x = torch.linspace(-4.0, 4.0, 9).unsqueeze(-1)
        e_pred, psi_pred = self._adapter(model)(x)

        h_batch = bdg_block_batched(x, self.N_SITES, self.HOPPING, self.PAIRING)
        expected = torch.einsum("bi,bij,bj->b", psi_pred, h_batch, psi_pred)
        self.assertTrue(torch.allclose(e_pred.squeeze(-1), expected, atol=1e-5))

    def test_exact_eigenvector_recovers_the_eigenvalue(self):
        """Feeding an exact eigenvector makes E_pred that eigenvalue."""
        ham = KitaevChainHamiltonian(
            n_sites=self.N_SITES, hopping=self.HOPPING, pairing=self.PAIRING
        )
        mu = 0.7
        eigvals, eigvecs = np.linalg.eigh(ham.build(mu))
        psi = torch.tensor(eigvecs[:, self.N_SITES], dtype=torch.float32)

        adapter = self._adapter(_ConstEigenvectorModel(psi))
        e_pred, _ = adapter(torch.full((3, 1), mu))

        self.assertTrue(
            torch.allclose(
                e_pred.squeeze(-1),
                torch.full((3,), float(eigvals[self.N_SITES])),
                atol=1e-5,
            )
        )

    def test_gradient_reaches_the_wrapped_model(self):
        """A gradient from E_pred flows into the wrapped model's parameters."""
        model = SirenPINN(n_sites=2 * self.N_SITES)
        adapter = self._adapter(model)

        e_pred, _ = adapter(torch.randn(4, 1))
        e_pred.sum().backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertFalse(any(torch.isnan(g).any() for g in grads))

    def test_shares_parameters_with_the_wrapped_model(self):
        """The adapter exposes the wrapped model's parameters, not copies."""
        model = SirenPINN(n_sites=2 * self.N_SITES)
        adapter = self._adapter(model)
        model_params = {id(p) for p in model.parameters()}
        self.assertTrue(model_params <= {id(p) for p in adapter.parameters()})


if __name__ == "__main__":
    import pytest

    pytest.main()
