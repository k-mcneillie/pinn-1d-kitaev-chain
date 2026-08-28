# tests/models/test_siren_folded.py
from __future__ import annotations

from unittest import TestCase

import numpy as np
import torch

from kitaev.analytical import bdg_block_batched
from kitaev.models import RayleighEnergyAdapter, SirenPINNNambuFolded


def _gamma_matrix(n_phys: int) -> np.ndarray:
    """The reflection operator ``Gamma = -(tau_x (x) diag((-1)^n))`` as 2N x 2N."""
    d = np.diag([(-1.0) ** n for n in range(n_phys)])
    zero = np.zeros((n_phys, n_phys))
    return np.block([[zero, -d], [-d, zero]])


class TestSirenPINNNambuFolded(TestCase):
    """Test suite for the folded Nambu-basis SIREN PINN."""

    N_PHYS = 5
    N_SITES = 2 * N_PHYS
    HOPPING = 1.0
    PAIRING = 0.5

    def _model(self, **kwargs) -> SirenPINNNambuFolded:
        torch.manual_seed(0)
        params = {
            "n_sites": self.N_SITES,
            "hidden_features": 16,
            "hidden_layers": 2,
        }
        params.update(kwargs)
        return SirenPINNNambuFolded(**params)

    def test_construction(self):
        """Instance properties: layer count, head width, D buffer."""
        model = self._model()
        self.assertEqual(model.n_sites, self.N_SITES)
        self.assertEqual(model.n_phys, self.N_PHYS)
        self.assertEqual(len(model.net), 3)  # 1 input + 2 hidden
        self.assertEqual(model.net[0].omega_0, 30.0)
        self.assertEqual(model.net[1].omega_0, 1.0)
        self.assertEqual(model.psi_head.out_features, self.N_SITES)
        expected_d = torch.tensor([(-1.0) ** n for n in range(self.N_PHYS)])
        self.assertTrue(torch.equal(model.D, expected_d))

    def test_odd_n_sites_is_rejected(self):
        """n_sites is the full BdG dimension 2N and must be even."""
        with self.assertRaises(ValueError):
            SirenPINNNambuFolded(n_sites=7)

    def test_head_weight_initialisation(self):
        """psi_head weights respect the SIREN-scaled bound."""
        model = self._model()
        bound = np.sqrt(6.0 / model.hidden_features) / model.hidden_omega_0
        with torch.no_grad():
            self.assertTrue(
                torch.all(torch.abs(model.psi_head.weight) <= bound * 1.0001)
            )

    def test_forward_shape(self):
        """Output is (B, 2N)."""
        model = self._model()
        out = model(torch.randn(4, 1))
        self.assertEqual(out.shape, (4, self.N_SITES))

    def test_output_normalised(self):
        """psi is unit L2 norm along dim=1."""
        model = self._model()
        out = model(torch.randn(7, 1))
        norms = torch.norm(out, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_exact_mu_reflection(self):
        """model(-mu) equals Gamma @ model(mu) to machine precision."""
        model = self._model()
        gamma = torch.tensor(_gamma_matrix(self.N_PHYS), dtype=torch.float32)
        mu = torch.linspace(-4.0, 4.0, 17).unsqueeze(-1)

        with torch.no_grad():
            psi_pos = model(mu)
            psi_neg = model(-mu)

        self.assertTrue(
            torch.allclose(psi_neg, psi_pos @ gamma.T, atol=1e-6),
            msg=f"max dev {(psi_neg - psi_pos @ gamma.T).abs().max().item():.2e}",
        )

    def test_reflection_holds_near_mu_zero(self):
        """The fold is exact at mu -> 0 too (no gauge kink there)."""
        model = self._model()
        gamma = torch.tensor(_gamma_matrix(self.N_PHYS), dtype=torch.float32)
        mu = torch.tensor([[1e-5], [1e-3], [1e-1]])
        with torch.no_grad():
            dev = (model(-mu) - model(mu) @ gamma.T).abs().max().item()
        self.assertLess(dev, 1e-6)

    def test_mu_zero_is_finite_and_continuous(self):
        """psi(0) is finite and psi is continuous across mu = 0."""
        model = self._model()
        with torch.no_grad():
            psi_zero = model(torch.zeros(1, 1))
            jump = model(torch.full((1, 1), 1e-4)) - model(torch.full((1, 1), -1e-4))
        self.assertFalse(torch.isnan(psi_zero).any())
        self.assertTrue(torch.isfinite(psi_zero).all())
        self.assertLess(jump.abs().max().item(), 1e-3)

    def test_rayleigh_energy_is_even_in_mu(self):
        """E_R(-mu) = E_R(mu) exactly, via RayleighEnergyAdapter."""
        model = self._model()
        adapter = RayleighEnergyAdapter(
            model, n_sites=self.N_PHYS, hopping=self.HOPPING, pairing=self.PAIRING
        )
        mu = torch.linspace(0.0, 4.0, 21).unsqueeze(-1)
        with torch.no_grad():
            e_pos, _ = adapter(mu)
            e_neg, _ = adapter(-mu)
        self.assertTrue(torch.allclose(e_pos, e_neg, atol=1e-6))

    def test_reflection_operator_realises_h_of_minus_mu(self):
        """Sanity check on Gamma itself: Gamma H(mu) Gamma == H(-mu)."""
        gamma = _gamma_matrix(self.N_PHYS)
        mu = np.array([[0.4], [1.9], [3.1]])
        h_batch = bdg_block_batched(
            torch.tensor(mu), self.N_PHYS, self.HOPPING, self.PAIRING
        ).numpy()
        h_neg = bdg_block_batched(
            torch.tensor(-mu), self.N_PHYS, self.HOPPING, self.PAIRING
        ).numpy()
        for k in range(mu.shape[0]):
            self.assertTrue(
                np.allclose(gamma @ h_batch[k] @ gamma, h_neg[k], atol=1e-12)
            )

    def test_differentiability(self):
        """Output is differentiable and grads reach every parameter."""
        model = self._model()
        x = torch.randn(3, 1, requires_grad=True)
        out = model(x)
        self.assertTrue(out.requires_grad)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertFalse(any(torch.isnan(g).any() for g in grads))

    def test_input_scale_divides_input_before_backbone(self):
        """The symmetrised construction uses x / input_scale at both +-mu."""
        model = self._model(input_scale=5.0)
        x = torch.tensor([[10.0], [-7.5], [0.0]])
        with torch.no_grad():
            raw_pos = model.psi_head(model.net(x / 5.0))
            raw_neg = model.psi_head(model.net(-x / 5.0))
            expected = torch.nn.functional.normalize(
                0.5 * (raw_pos + model._fold(raw_neg)), p=2, dim=1, eps=1e-12
            )
            got = model(x)
        self.assertTrue(torch.allclose(got, expected))

    def test_custom_hidden_omega_0_is_used_by_hidden_layers_and_head(self):
        """hidden_omega_0 is a real, swept-able constructor argument."""
        model = self._model(hidden_omega_0=5.0)
        self.assertEqual(model.net[0].omega_0, 30.0)
        self.assertEqual(model.net[1].omega_0, 5.0)
        self.assertEqual(model.net[2].omega_0, 5.0)
        bound = np.sqrt(6.0 / model.hidden_features) / 5.0
        with torch.no_grad():
            self.assertTrue(
                torch.all(torch.abs(model.psi_head.weight) <= bound * 1.0001)
            )


if __name__ == "__main__":
    import pytest

    pytest.main()
