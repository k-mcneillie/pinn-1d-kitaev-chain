# tests/models/test_siren_chiral.py
from __future__ import annotations

from unittest import TestCase

import numpy as np
import torch

from kitaev.analytical import (
    chiral_block_batched,
    reconstruct_bdg_eigenvector,
    resolve_singular_branch,
)
from kitaev.models.siren_chiral import ChiralToBdGAdapter, SirenPINNChiral


class TestSirenPINNChiral(TestCase):
    """Comprehensive test suite for the SirenPINNChiral class."""

    def test_construction(self):
        """n_sites is N; the single head has width 2N."""
        model = SirenPINNChiral(n_sites=5)
        self.assertEqual(model.n_sites, 5)
        self.assertEqual(len(model.net), 3)  # 1 input layer + 2 hidden
        self.assertEqual(model.net[0].omega_0, 30.0)
        self.assertEqual(model.net[1].omega_0, 1.0)
        self.assertEqual(model.psi_head.out_features, 10)

    def test_head_weight_initialisation(self):
        """psi_head weights use the SIREN hidden-frequency scaling."""
        model = SirenPINNChiral(n_sites=4)
        bound = np.sqrt(6.0 / model.hidden_features) / model.hidden_omega_0
        with torch.no_grad():
            self.assertTrue(
                torch.all(torch.abs(model.psi_head.weight) <= bound * 1.0001)
            )

    def test_forward_shapes(self):
        """forward returns (u, v), each (batch, n_sites)."""
        model = SirenPINNChiral(n_sites=6)
        x = torch.randn(4, 1)
        u, v = model(x)
        self.assertEqual(u.shape, (4, 6))
        self.assertEqual(v.shape, (4, 6))

    def test_outputs_unit_norm(self):
        """u and v are L2-normalised along dim=1."""
        model = SirenPINNChiral(n_sites=6)
        x = torch.randn(5, 1) * 4.0
        u, v = model(x)
        ones = torch.ones(5)
        self.assertTrue(torch.allclose(u.norm(p=2, dim=1), ones, atol=1e-5))
        self.assertTrue(torch.allclose(v.norm(p=2, dim=1), ones, atol=1e-5))

    def test_chemical_potential_reflection_equivariance(self):
        """model(-mu) == (-D u, D v) evaluated at |mu|, away from mu = 0."""
        model = SirenPINNChiral(n_sites=8)
        mu = torch.tensor([[0.2], [1.1], [2.0], [3.7]])

        u_pos, v_pos = model(mu)
        u_neg, v_neg = model(-mu)

        self.assertTrue(torch.allclose(u_neg, -model.D * u_pos, atol=1e-6))
        self.assertTrue(torch.allclose(v_neg, model.D * v_pos, atol=1e-6))

    def test_spectrum_even_in_mu_via_reflection(self):
        """The reflection keeps the Rayleigh quotient u^T h v even in mu."""
        from kitaev.analytical import chiral_block_batched

        model = SirenPINNChiral(n_sites=10)
        mu = torch.tensor([[0.6], [1.8], [3.1]])

        u_pos, v_pos = model(mu)
        u_neg, v_neg = model(-mu)
        h_pos = chiral_block_batched(mu, 10, 1.0, 0.5)
        h_neg = chiral_block_batched(-mu, 10, 1.0, 0.5)

        lam_pos = torch.einsum("bi,bij,bj->b", u_pos, h_pos, v_pos)
        lam_neg = torch.einsum("bi,bij,bj->b", u_neg, h_neg, v_neg)
        self.assertTrue(torch.allclose(lam_pos, lam_neg, atol=1e-5))

    def test_differentiability(self):
        """Gradients reach the input and every parameter, without NaNs."""
        model = SirenPINNChiral(n_sites=4)
        x = torch.randn(2, 1, requires_grad=True)

        u, v = model(x)
        self.assertTrue(u.requires_grad)

        (u.sum() + v.sum()).backward()

        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertFalse(torch.isnan(x.grad).any())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertFalse(torch.isnan(parameter.grad).any())

    def test_input_scale_divides_absolute_input(self):
        """|mu| / input_scale reaches the backbone; sign drives the fold."""
        model = SirenPINNChiral(n_sites=3, input_scale=5.0)
        x = torch.tensor([[10.0], [7.5]])

        with torch.no_grad():
            features = model.net(x.abs() / 5.0)
            raw = model.psi_head(features)
            expected_u = torch.nn.functional.normalize(
                raw[:, :3], p=2, dim=1, eps=1e-12
            )

        u, _ = model(x)
        self.assertTrue(torch.allclose(u, expected_u))


class TestChiralToBdGAdapter(TestCase):
    """The adapter exposes SirenPINNChiral as a dual-head (E, psi) model."""

    def test_forward_shapes(self):
        model = SirenPINNChiral(n_sites=7)
        adapter = ChiralToBdGAdapter(model, hopping=1.0, pairing=0.5)
        x = torch.randn(4, 1)

        energy_pred, psi_pred = adapter(x)

        self.assertEqual(energy_pred.shape, (4, 1))
        self.assertEqual(psi_pred.shape, (4, 14))

    def test_psi_is_unit_norm_and_reconstruction_consistent(self):
        model = SirenPINNChiral(n_sites=7)
        adapter = ChiralToBdGAdapter(model, hopping=1.0, pairing=0.5)
        x = torch.tensor([[0.4], [1.6], [3.2]])

        u, v = model(x)
        h_batch = chiral_block_batched(x, 7, 1.0, 0.5)
        u_res, v_res, _, _ = resolve_singular_branch(u, v, h_batch)
        _, psi_pred = adapter(x)

        norms = psi_pred.norm(p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones(3), atol=1e-5))

        # psi_pred is reconstructed from the branch-resolved pair.
        for row in range(3):
            expected = reconstruct_bdg_eigenvector(
                u_res[row].detach().numpy(), v_res[row].detach().numpy(), sign=1
            )
            self.assertTrue(
                np.allclose(psi_pred[row].detach().numpy(), expected, atol=1e-6)
            )

    def test_energy_is_branch_resolved_rayleigh_quotient(self):
        model = SirenPINNChiral(n_sites=9)
        adapter = ChiralToBdGAdapter(model, hopping=1.0, pairing=0.5)
        x = torch.tensor([[0.7], [2.4]])

        u, v = model(x)
        h_batch = chiral_block_batched(x, 9, 1.0, 0.5)
        _, _, _, expected = resolve_singular_branch(u, v, h_batch)

        energy_pred, _ = adapter(x)
        self.assertTrue(torch.allclose(energy_pred, expected, atol=1e-6))

    def test_energy_is_non_negative_across_a_sweep(self):
        """The branch fix leaves E_pred >= 0 wherever lambda_R is resolved."""
        torch.manual_seed(0)
        model = SirenPINNChiral(n_sites=12)
        adapter = ChiralToBdGAdapter(model, hopping=1.0, pairing=0.5)
        x = torch.linspace(0.05, 4.0, 200).unsqueeze(-1)

        energy_pred, _ = adapter(x)

        self.assertGreaterEqual(energy_pred.min().item(), -1e-3)


if __name__ == "__main__":
    import pytest

    pytest.main()
