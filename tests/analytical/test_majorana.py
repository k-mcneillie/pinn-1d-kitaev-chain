# tests/analytical/test_majorana.py
from __future__ import annotations

from unittest import TestCase

import numpy as np
import torch

from kitaev.analytical import (
    KitaevChainHamiltonian,
    chiral_block,
    majorana_basis_change,
    reconstruct_bdg_eigenvector,
    resolve_singular_branch,
)

T = 1.0
DELTA = 0.5


def _staggered_signs(n_sites: int) -> np.ndarray:
    return np.diag([(-1.0) ** n for n in range(n_sites)])


class TestChiralBlock(TestCase):
    """Tests for the reduced chiral block h(mu)."""

    def test_bidiagonal_entries(self):
        """h(mu) is bidiagonal with the documented entries."""
        n_sites = 5
        mu = 0.7
        h = chiral_block(mu, n_sites, T, DELTA)

        self.assertEqual(h.shape, (n_sites, n_sites))
        for n in range(n_sites):
            self.assertAlmostEqual(h[n, n], -mu)
        for n in range(n_sites - 1):
            self.assertAlmostEqual(h[n, n + 1], -(T + DELTA))
            self.assertAlmostEqual(h[n + 1, n], -(T - DELTA))

        # Everything outside the three diagonals is zero.
        band = np.zeros_like(h, dtype=bool)
        idx = np.arange(n_sites)
        band[idx, idx] = True
        band[idx[:-1], idx[1:]] = True
        band[idx[1:], idx[:-1]] = True
        self.assertTrue(np.all(h[~band] == 0.0))

    def test_scales_linearly_with_hopping_unit(self):
        """All entries scale with the energy unit t (mu, delta in units of t)."""
        n_sites = 4
        h_unit = chiral_block(1.5, n_sites, 1.0, 0.5)
        h_scaled = chiral_block(3.0, n_sites, 2.0, 1.0)
        self.assertTrue(np.allclose(h_scaled, 2.0 * h_unit))

    def test_chemical_potential_reflection(self):
        """h(-mu) = -D h(mu) D with D = diag((-1)^n)."""
        n_sites = 8
        d_mat = _staggered_signs(n_sites)
        for mu in (0.3, 1.7, 2.0, 3.4):
            h_plus = chiral_block(mu, n_sites, T, DELTA)
            h_minus = chiral_block(-mu, n_sites, T, DELTA)
            self.assertTrue(np.allclose(h_minus, -d_mat @ h_plus @ d_mat))


class TestMajoranaBasisChange(TestCase):
    """Tests for the unitary Omega and the block-diagonalisation identity."""

    def test_omega_is_unitary(self):
        omega = majorana_basis_change(6)
        identity = np.eye(12)
        self.assertTrue(np.allclose(omega @ omega.conj().T, identity))

    def test_block_diagonalises_bdg_hamiltonian(self):
        """Omega H(mu) Omega^dagger == i [[0, h], [-h^T, 0]] for a mu sweep."""
        n_sites = 8
        omega = majorana_basis_change(n_sites)
        ham = KitaevChainHamiltonian(n_sites=n_sites, hopping=T, pairing=DELTA)
        zero = np.zeros((n_sites, n_sites))

        for mu in np.linspace(-4.0, 4.0, 33):
            transformed = omega @ ham.build(mu) @ omega.conj().T
            h = chiral_block(mu, n_sites, T, DELTA)
            target = 1j * np.block([[zero, h], [-h.T, zero]])
            self.assertTrue(np.allclose(transformed, target, atol=1e-12))


class TestSpectralCorrespondence(TestCase):
    """The singular values of h(mu) are the non-negative BdG eigenvalues."""

    def test_singular_values_match_eigh(self):
        n_sites = 12
        ham = KitaevChainHamiltonian(n_sites=n_sites, hopping=T, pairing=DELTA)
        for mu in np.linspace(-4.0, 4.0, 21):
            singular = np.sort(
                np.linalg.svd(chiral_block(mu, n_sites, T, DELTA), compute_uv=False)
            )
            eigenvalues = np.linalg.eigvalsh(ham.build(mu))
            non_negative = np.sort(eigenvalues[eigenvalues >= -1e-12])
            self.assertTrue(np.allclose(singular, non_negative, atol=1e-9))

    def test_gap_closes_at_transition(self):
        """lambda_min -> 0 approaching |mu| = 2t; ~ |mu| - 2t beyond it."""
        n_sites = 60  # large enough that the finite-size splitting is tiny
        deep = np.linalg.svd(
            chiral_block(0.5, n_sites, T, DELTA), compute_uv=False
        ).min()
        near = np.linalg.svd(
            chiral_block(1.9, n_sites, T, DELTA), compute_uv=False
        ).min()
        trivial = np.linalg.svd(
            chiral_block(3.0, n_sites, T, DELTA), compute_uv=False
        ).min()

        self.assertLess(deep, 1e-6)
        self.assertLess(near, deep + 0.2)
        self.assertAlmostEqual(trivial, 3.0 - 2.0 * T, delta=0.05)

    def test_characteristic_equation_root_on_unit_circle_at_transition(self):
        """(t + delta) z^2 + mu z + (t - delta) = 0 has |z| = 1 at |mu| = 2t."""
        for mu in (-2.0 * T, 2.0 * T):
            roots = np.roots([T + DELTA, mu, T - DELTA])
            self.assertTrue(np.any(np.isclose(np.abs(roots), 1.0)))


class TestReconstruction(TestCase):
    """Mapping a singular pair back to a BdG eigenvector."""

    def test_reconstructed_vector_matches_eigh(self):
        n_sites = 10
        ham = KitaevChainHamiltonian(n_sites=n_sites, hopping=T, pairing=DELTA)
        xi = np.block(
            [
                [np.zeros((n_sites, n_sites)), np.eye(n_sites)],
                [np.eye(n_sites), np.zeros((n_sites, n_sites))],
            ]
        )

        for mu in (0.4, 1.3, 1.95, 2.6, 3.5):
            h = chiral_block(mu, n_sites, T, DELTA)
            left, singular, right_t = np.linalg.svd(h)
            k = int(np.argmin(singular))
            u_k, v_k, lam = left[:, k], right_t[k, :], singular[k]

            psi_plus = reconstruct_bdg_eigenvector(u_k, v_k, sign=1)
            psi_minus = reconstruct_bdg_eigenvector(u_k, v_k, sign=-1)

            self.assertAlmostEqual(np.linalg.norm(psi_plus), 1.0, places=10)

            eigenvalues, eigenvectors = np.linalg.eigh(ham.build(mu))
            j_plus = int(np.argmin(np.abs(eigenvalues - lam)))
            j_minus = int(np.argmin(np.abs(eigenvalues + lam)))

            self.assertAlmostEqual(
                abs(psi_plus @ eigenvectors[:, j_plus]), 1.0, places=8
            )
            self.assertAlmostEqual(
                abs(psi_minus @ eigenvectors[:, j_minus]), 1.0, places=8
            )
            # The -E partner is the particle/hole block swap of the +E vector.
            self.assertTrue(np.allclose(psi_minus, xi @ psi_plus, atol=1e-12))


class TestResolveSingularBranch(TestCase):
    """Canonicalising a predicted singular pair onto the +lambda branch."""

    def test_negative_lambda_is_flipped_to_positive(self):
        """lambda_R < -tol => v is negated and lambda_R becomes positive."""
        n_sites = 6
        h = chiral_block(3.0, n_sites, T, DELTA)  # trivial phase, lam ~ 1
        left, singular, right_t = np.linalg.svd(h)
        k = int(np.argmin(singular))
        # Deliberately take the wrong branch: negate u so u^T h v < 0.
        u = torch.tensor(-left[:, k], dtype=torch.float64).unsqueeze(0)
        v = torch.tensor(right_t[k, :], dtype=torch.float64).unsqueeze(0)
        h_batch = torch.tensor(h, dtype=torch.float64).unsqueeze(0)

        u_res, v_res, h_v_res, lam_res = resolve_singular_branch(u, v, h_batch)

        self.assertGreater(lam_res.item(), 0.0)
        self.assertAlmostEqual(lam_res.item(), singular[k], places=10)
        self.assertTrue(torch.allclose(v_res, -v))
        self.assertTrue(torch.allclose(u_res, u))  # u is never touched
        self.assertTrue(
            torch.allclose(h_v_res, torch.bmm(h_batch, v_res.unsqueeze(-1)).squeeze(-1))
        )

    def test_near_zero_lambda_is_left_untouched(self):
        """|lambda_R| <= tol (topological near-zero mode) => pair unchanged."""
        n_sites = 40  # finite-size splitting well below tol
        h = chiral_block(0.5, n_sites, T, DELTA)
        left, singular, right_t = np.linalg.svd(h)
        k = int(np.argmin(singular))
        self.assertLess(singular[k], 1e-3)
        # Wrong-looking branch, but lambda_R ~ 0 so it must not be flipped.
        u = torch.tensor(-left[:, k], dtype=torch.float64).unsqueeze(0)
        v = torch.tensor(right_t[k, :], dtype=torch.float64).unsqueeze(0)
        h_batch = torch.tensor(h, dtype=torch.float64).unsqueeze(0)

        u_res, v_res, _, _ = resolve_singular_branch(u, v, h_batch, tol=1e-3)

        self.assertTrue(torch.equal(u_res, u))
        self.assertTrue(torch.equal(v_res, v))

    def test_flip_swaps_reconstructed_vector_with_its_partner(self):
        """Flipping v maps psi = ((u+v)/2, (u-v)/2) to its Xi (block-swap)."""
        n_sites = 8
        h = chiral_block(2.7, n_sites, T, DELTA)
        left, singular, right_t = np.linalg.svd(h)
        k = int(np.argmin(singular))
        u = torch.tensor(-left[:, k], dtype=torch.float64).unsqueeze(0)
        v = torch.tensor(right_t[k, :], dtype=torch.float64).unsqueeze(0)
        h_batch = torch.tensor(h, dtype=torch.float64).unsqueeze(0)

        psi_raw = reconstruct_bdg_eigenvector(u[0].numpy(), v[0].numpy(), sign=1)
        u_res, v_res, _, _ = resolve_singular_branch(u, v, h_batch)
        psi_res = reconstruct_bdg_eigenvector(
            u_res[0].numpy(), v_res[0].numpy(), sign=1
        )

        xi = np.block(
            [
                [np.zeros((n_sites, n_sites)), np.eye(n_sites)],
                [np.eye(n_sites), np.zeros((n_sites, n_sites))],
            ]
        )
        self.assertTrue(np.allclose(psi_res, xi @ psi_raw, atol=1e-12))

    def test_flip_sign_does_not_carry_gradient(self):
        """The branch decision is detached: backward still works cleanly."""
        n_sites = 5
        h_batch = torch.tensor(
            chiral_block(3.0, n_sites, T, DELTA), dtype=torch.float64
        ).unsqueeze(0)
        u = torch.randn(1, n_sites, dtype=torch.float64, requires_grad=True)
        v = torch.randn(1, n_sites, dtype=torch.float64, requires_grad=True)

        _, _, _, lam_res = resolve_singular_branch(u, v, h_batch)
        lam_res.sum().backward()

        self.assertIsNotNone(u.grad)
        self.assertIsNotNone(v.grad)
        self.assertFalse(torch.isnan(u.grad).any())
        self.assertFalse(torch.isnan(v.grad).any())


if __name__ == "__main__":
    import pytest

    pytest.main()
