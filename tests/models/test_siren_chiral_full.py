# tests/models/test_siren_chiral_full.py
from __future__ import annotations

from unittest import TestCase

import numpy as np
import torch

from kitaev.analytical import (
    chiral_block,
    chiral_block_batched,
    chiral_block_det_sign,
    resolve_singular_branch,
)
from kitaev.models.siren_chiral_full import (
    ChiralFullToBdGAdapter,
    SirenPINNChiralFull,
)

T = 1.0
DELTA = 0.5


class TestSirenPINNChiralFull(TestCase):
    """The full-SVD chiral model: U, sigma, V frames of h(mu)."""

    def test_construction(self):
        """Head widths are N(N-1)/2, N(N-1)/2 and N; backbone is 1 + hidden."""
        model = SirenPINNChiralFull(n_sites=6, hidden_layers=2)
        self.assertEqual(model.skew_dim, 15)
        self.assertEqual(model.head_u.out_features, 15)
        self.assertEqual(model.head_v.out_features, 15)
        self.assertEqual(model.head_s.out_features, 6)
        self.assertEqual(len(model.net), 3)
        self.assertEqual(model.net[0].omega_0, 30.0)
        self.assertEqual(model.net[1].omega_0, 1.0)

    def test_head_weight_initialisation(self):
        """Every head uses the SIREN hidden-frequency scaling bound."""
        model = SirenPINNChiralFull(n_sites=5)
        bound = np.sqrt(6.0 / model.hidden_features) / model.hidden_omega_0
        for head in (model.head_u, model.head_v, model.head_s):
            with torch.no_grad():
                self.assertTrue(torch.all(head.weight.abs() <= bound * 1.0001))

    def test_forward_shapes(self):
        """forward returns (U, sigma, V) of shapes (B,N,N), (B,N), (B,N,N)."""
        model = SirenPINNChiralFull(n_sites=8)
        u_mat, sigma, v_mat = model(torch.randn(4, 1))
        self.assertEqual(u_mat.shape, (4, 8, 8))
        self.assertEqual(sigma.shape, (4, 8))
        self.assertEqual(v_mat.shape, (4, 8, 8))

    def test_sigma_is_non_negative(self):
        model = SirenPINNChiralFull(n_sites=6)
        _u, sigma, _v = model(torch.linspace(0.05, 4.0, 40).unsqueeze(-1))
        self.assertGreaterEqual(sigma.min().item(), 0.0)

    def test_frames_are_orthogonal(self):
        """U^T U = V^T V = I to double precision across a mu sweep."""
        model = SirenPINNChiralFull(n_sites=8).double()
        x = torch.linspace(-4.0, 4.0, 41, dtype=torch.float64).unsqueeze(-1)
        u_mat, _sigma, v_mat = model(x)
        eye = torch.eye(8, dtype=torch.float64)
        self.assertTrue(
            torch.allclose(
                u_mat.transpose(-2, -1) @ u_mat, eye.expand_as(u_mat), atol=1e-10
            )
        )
        self.assertTrue(
            torch.allclose(
                v_mat.transpose(-2, -1) @ v_mat, eye.expand_as(v_mat), atol=1e-10
            )
        )

    def test_determinant_class_matches_analytic_det_sign(self):
        """det U ~ +1; det V ~ sign(det h(mu)) away from gap closings."""
        model = SirenPINNChiralFull(n_sites=8, hopping=T, pairing=DELTA).double()
        mu_values = [0.3, 1.1, 2.5, 3.7]
        x = torch.tensor(mu_values, dtype=torch.float64).unsqueeze(-1)
        u_mat, _sigma, v_mat = model(x)

        self.assertTrue(
            torch.allclose(
                torch.linalg.det(u_mat),
                torch.ones(len(mu_values), dtype=torch.float64),
                atol=1e-9,
            )
        )
        want = torch.tensor(
            [np.sign(np.linalg.det(chiral_block(m, 8, T, DELTA))) for m in mu_values],
            dtype=torch.float64,
        )
        self.assertTrue(torch.allclose(torch.linalg.det(v_mat).sign(), want))

    def test_reconstruction_convention_lock(self):
        """h_recon = U diag(sigma) V^T obeys bmm(h_recon, V) == U * sigma."""
        model = SirenPINNChiralFull(n_sites=6).double()
        x = torch.tensor([[0.5], [1.7], [3.1]], dtype=torch.float64)
        u_mat, sigma, v_mat = model(x)
        h_recon = u_mat @ torch.diag_embed(sigma) @ v_mat.transpose(-2, -1)
        self.assertTrue(
            torch.allclose(
                torch.bmm(h_recon, v_mat), u_mat * sigma.unsqueeze(1), atol=1e-9
            )
        )

    def test_chemical_potential_reflection_equivariance(self):
        """model(-mu) = (-D U, sigma, D V) from model(|mu|), row-wise."""
        model = SirenPINNChiralFull(n_sites=8).double()
        x = torch.tensor([[0.4], [1.3], [2.6], [3.5]], dtype=torch.float64)
        u_pos, s_pos, v_pos = model(x)
        u_neg, s_neg, v_neg = model(-x)

        d_row = model.D.reshape(1, 8, 1)
        self.assertTrue(torch.allclose(u_neg, -(d_row * u_pos), atol=1e-9))
        self.assertTrue(torch.allclose(v_neg, d_row * v_pos, atol=1e-9))
        self.assertTrue(torch.allclose(s_neg, s_pos, atol=1e-9))

    def test_spectrum_even_in_mu(self):
        """The Rayleigh spectrum of the frames is even in mu."""
        model = SirenPINNChiralFull(n_sites=8, hopping=T, pairing=DELTA).double()
        x = torch.linspace(0.1, 3.8, 20, dtype=torch.float64).unsqueeze(-1)
        h_pos = chiral_block_batched(x, 8, T, DELTA)
        h_neg = chiral_block_batched(-x, 8, T, DELTA)
        u_p, _s, v_p = model(x)
        u_n, _s2, v_n = model(-x)
        rayleigh_pos = torch.einsum("bik,bij,bjk->bk", u_p, h_pos, v_p)
        rayleigh_neg = torch.einsum("bik,bij,bjk->bk", u_n, h_neg, v_n)
        self.assertTrue(
            torch.allclose(rayleigh_pos.abs(), rayleigh_neg.abs(), atol=1e-8)
        )

    def test_differentiability(self):
        """Gradients reach the input and every parameter, without NaNs."""
        model = SirenPINNChiralFull(n_sites=4)
        x = torch.randn(3, 1, requires_grad=True)
        u_mat, sigma, v_mat = model(x)
        (u_mat.sum() + sigma.sum() + v_mat.sum()).backward()

        assert x.grad is not None
        self.assertFalse(torch.isnan(x.grad).any())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertFalse(torch.isnan(parameter.grad).any())

    def test_det_sign_is_detached_from_the_graph(self):
        """The analytic reflection contributes no gradient path."""
        model = SirenPINNChiralFull(n_sites=5)
        x = torch.tensor([[0.6], [1.2]], requires_grad=True)
        _u, _sigma, v_mat = model(x)
        v_mat.sum().backward()
        assert x.grad is not None
        self.assertFalse(torch.isnan(x.grad).any())


class TestChiralFullToBdGAdapter(TestCase):
    """The adapter exposes the smallest triple as a dual-head (E, psi) model."""

    def test_forward_shapes(self):
        model = SirenPINNChiralFull(n_sites=7)
        adapter = ChiralFullToBdGAdapter(model, hopping=T, pairing=DELTA)
        energy_pred, psi_pred = adapter(torch.randn(4, 1))
        self.assertEqual(energy_pred.shape, (4, 1))
        self.assertEqual(psi_pred.shape, (4, 14))

    def test_psi_unit_norm(self):
        model = SirenPINNChiralFull(n_sites=7)
        adapter = ChiralFullToBdGAdapter(model, hopping=T, pairing=DELTA)
        _e, psi_pred = adapter(torch.tensor([[0.4], [1.6], [3.2]]))
        self.assertTrue(
            torch.allclose(psi_pred.norm(p=2, dim=1), torch.ones(3), atol=1e-5)
        )

    def test_energy_is_branch_resolved_rayleigh_of_smallest_triple(self):
        model = SirenPINNChiralFull(n_sites=9)
        adapter = ChiralFullToBdGAdapter(model, hopping=T, pairing=DELTA)
        x = torch.tensor([[0.7], [2.4]])

        u_mat, sigma, v_mat = model(x)
        k = torch.argmin(sigma, dim=1)
        index = k.reshape(-1, 1, 1).expand(-1, 9, 1)
        u = torch.gather(u_mat, 2, index).squeeze(-1)
        v = torch.gather(v_mat, 2, index).squeeze(-1)
        h_batch = chiral_block_batched(x, 9, T, DELTA)
        _, _, _, expected = resolve_singular_branch(u, v, h_batch)

        energy_pred, _ = adapter(x)
        self.assertTrue(torch.allclose(energy_pred, expected, atol=1e-6))

    def test_energy_non_negative_across_a_sweep(self):
        torch.manual_seed(0)
        model = SirenPINNChiralFull(n_sites=12)
        adapter = ChiralFullToBdGAdapter(model, hopping=T, pairing=DELTA)
        energy_pred, _ = adapter(torch.linspace(0.05, 4.0, 200).unsqueeze(-1))
        self.assertGreaterEqual(energy_pred.min().item(), -1e-3)

    def test_full_spectrum_shape_and_symmetry(self):
        """full_spectrum is (B, 2N), sorted, and equal to sort(cat([sigma, -sigma]))."""
        model = SirenPINNChiralFull(n_sites=6)
        adapter = ChiralFullToBdGAdapter(model, hopping=T, pairing=DELTA)
        x = torch.linspace(0.05, 4.0, 30).unsqueeze(-1)
        spectrum = adapter.full_spectrum(x)
        _u, sigma, _v = model(x)

        self.assertEqual(spectrum.shape, (30, 12))
        self.assertTrue(torch.all(spectrum[:, 1:] >= spectrum[:, :-1] - 1e-6))
        want = torch.sort(torch.cat([sigma, -sigma], dim=1), dim=1).values
        self.assertTrue(torch.allclose(spectrum, want, atol=1e-6))
        # Particle-hole symmetric about zero.
        self.assertTrue(torch.allclose(spectrum, -spectrum.flip(-1), atol=1e-5))

    def test_det_sign_matches_numpy_determinant(self):
        model = SirenPINNChiralFull(n_sites=8, hopping=T, pairing=DELTA).double()
        adapter = ChiralFullToBdGAdapter(model, hopping=T, pairing=DELTA)
        mu_grid = np.linspace(0.1, 3.9, 50)
        got = adapter.det_sign(torch.tensor(mu_grid, dtype=torch.float64).unsqueeze(-1))
        want = np.array(
            [np.sign(np.linalg.det(chiral_block(m, 8, T, DELTA))) for m in mu_grid]
        )
        self.assertTrue(np.array_equal(got.numpy(), want))

    def test_det_sign_agrees_with_analytic_recurrence(self):
        model = SirenPINNChiralFull(n_sites=10, hopping=T, pairing=DELTA).double()
        adapter = ChiralFullToBdGAdapter(model, hopping=T, pairing=DELTA)
        x = torch.linspace(0.1, 3.9, 60, dtype=torch.float64).unsqueeze(-1)
        analytic = chiral_block_det_sign(x, 10, T, DELTA)
        self.assertTrue(torch.equal(adapter.det_sign(x), analytic))


if __name__ == "__main__":
    import pytest

    pytest.main()
