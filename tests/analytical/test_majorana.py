# tests/analytical/test_majorana.py
from __future__ import annotations

from unittest import TestCase

import numpy as np
import torch

from kitaev.analytical import (
    KitaevChainHamiltonian,
    chiral_block,
    chiral_block_batched,
    chiral_block_det_sign,
    chiral_block_matvec,
    fill_skew,
    majorana_basis_change,
    reconstruct_bdg_eigenvector,
    reconstruct_bdg_eigenvectors,
    resolve_singular_branch,
    resolve_svd_sign,
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


class TestChiralBlockMatvec(TestCase):
    """chiral_block_matvec is the matrix-free equivalent of the dense bmm."""

    def _dense_apply(
        self, mu_batch: torch.Tensor, vec: torch.Tensor, *, adjoint: bool
    ) -> torch.Tensor:
        n_sites = vec.shape[1]
        h = chiral_block_batched(mu_batch, n_sites, T, DELTA)
        if adjoint:
            h = h.transpose(1, 2)
        return torch.bmm(h, vec.unsqueeze(-1)).squeeze(-1)

    def test_matches_dense_bmm(self):
        """h(mu) @ v and h(mu)^T @ u match the dense product across shapes."""
        torch.manual_seed(0)
        for n_sites in (1, 2, 5, 20):
            mu_batch = (torch.rand(17, 1, dtype=torch.float64) * 8.0) - 4.0
            for adjoint in (False, True):
                vec = torch.randn(17, n_sites, dtype=torch.float64)
                got = chiral_block_matvec(
                    mu_batch, vec, hopping=T, pairing=DELTA, adjoint=adjoint
                )
                want = self._dense_apply(mu_batch, vec, adjoint=adjoint)
                self.assertTrue(torch.allclose(got, want, atol=1e-12))

    def test_accepts_flat_mu(self):
        """A 1-D mu batch is handled the same as a column vector."""
        mu_flat = torch.linspace(-4.0, 4.0, 9, dtype=torch.float64)
        vec = torch.randn(9, 6, dtype=torch.float64)
        col = chiral_block_matvec(mu_flat[:, None], vec, hopping=T, pairing=DELTA)
        flat = chiral_block_matvec(mu_flat, vec, hopping=T, pairing=DELTA)
        self.assertTrue(torch.allclose(col, flat, atol=1e-12))

    def test_gradient_matches_dense(self):
        """d/dmu of the matvec agrees with the dense path (autograd parity)."""
        mu_dense = torch.tensor([[0.3], [1.7], [2.4]], requires_grad=True)
        mu_free = mu_dense.detach().clone().requires_grad_(True)
        vec = torch.randn(3, 8)

        dense = self._dense_apply(mu_dense, vec, adjoint=False)
        free = chiral_block_matvec(mu_free, vec, hopping=T, pairing=DELTA)
        dense.pow(2).sum().backward()
        free.pow(2).sum().backward()

        assert mu_dense.grad is not None and mu_free.grad is not None
        self.assertTrue(torch.allclose(mu_dense.grad, mu_free.grad, atol=1e-6))


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


class TestFillSkew(TestCase):
    """Mapping free parameters to real skew-symmetric generators."""

    def test_shape_and_antisymmetry(self):
        """Output is (B, N, N) and exactly antisymmetric."""
        n_sites = 6
        vec = torch.randn(4, n_sites * (n_sites - 1) // 2, dtype=torch.float64)
        skew = fill_skew(vec, n_sites)

        self.assertEqual(skew.shape, (4, n_sites, n_sites))
        self.assertTrue(torch.allclose(skew, -skew.transpose(-2, -1), atol=0.0))
        diag = torch.diagonal(skew, dim1=-2, dim2=-1)
        self.assertTrue(torch.all(diag == 0.0))

    def test_reproduces_the_upper_triangle(self):
        """The N(N-1)/2 parameters land in the strict upper triangle in order."""
        n_sites = 5
        expected = n_sites * (n_sites - 1) // 2
        vec = torch.arange(1, expected + 1, dtype=torch.float64).unsqueeze(0)
        skew = fill_skew(vec, n_sites)[0]

        rows, cols = torch.triu_indices(n_sites, n_sites, offset=1)
        self.assertTrue(torch.equal(skew[rows, cols], vec[0]))
        self.assertTrue(torch.equal(skew[cols, rows], -vec[0]))

    def test_exp_of_generator_is_orthogonal(self):
        """matrix_exp(fill_skew(...)) is in SO(N)."""
        n_sites = 7
        vec = torch.randn(3, n_sites * (n_sites - 1) // 2, dtype=torch.float64)
        q = torch.matrix_exp(fill_skew(vec, n_sites))
        eye = torch.eye(n_sites, dtype=torch.float64).expand_as(q)

        self.assertTrue(torch.allclose(q.transpose(-2, -1) @ q, eye, atol=1e-10))
        self.assertTrue(
            torch.allclose(
                torch.linalg.det(q), torch.ones(3, dtype=torch.float64), atol=1e-10
            )
        )

    def test_leading_batch_dims_and_device_follow_input(self):
        """Extra leading axes are preserved; dtype/device follow vec."""
        n_sites = 4
        vec = torch.randn(2, 3, n_sites * (n_sites - 1) // 2, dtype=torch.float32)
        skew = fill_skew(vec, n_sites)
        self.assertEqual(skew.shape, (2, 3, n_sites, n_sites))
        self.assertEqual(skew.dtype, torch.float32)
        self.assertEqual(skew.device, vec.device)

    def test_differentiable_in_vec(self):
        """Gradients flow back to vec without NaNs."""
        n_sites = 5
        vec = torch.randn(
            2, n_sites * (n_sites - 1) // 2, dtype=torch.float64, requires_grad=True
        )
        fill_skew(vec, n_sites).pow(2).sum().backward()
        self.assertIsNotNone(vec.grad)
        self.assertFalse(torch.isnan(vec.grad).any())
        self.assertGreater(vec.grad.abs().sum().item(), 0.0)

    def test_wrong_width_raises(self):
        vec = torch.zeros(1, 5)
        with self.assertRaises(ValueError):
            fill_skew(vec, 6)


class TestChiralBlockDetSign(TestCase):
    """Sign of det h(mu) from the O(N) continuant recurrence."""

    def test_matches_numpy_determinant_across_the_domain(self):
        """Agrees with sign(det(chiral_block)) at every mu, including det < 0."""
        n_sites = 20
        mu_grid = np.linspace(0.05, 4.0, 241)
        mu = torch.tensor(mu_grid, dtype=torch.float64)
        got = chiral_block_det_sign(mu, n_sites, T, DELTA).numpy()

        want = np.array(
            [
                np.sign(np.linalg.det(chiral_block(m, n_sites, T, DELTA)))
                for m in mu_grid
            ]
        )
        self.assertTrue(np.array_equal(got, want))
        # The topological phase genuinely contains both signs.
        topological = mu_grid < 2.0
        self.assertIn(-1.0, set(got[topological]))
        self.assertIn(1.0, set(got[topological]))

    def test_values_are_pm_one_and_shape_is_flat(self):
        mu = torch.tensor([[0.3], [1.1], [2.4], [3.9]], dtype=torch.float64)
        got = chiral_block_det_sign(mu, 12, T, DELTA)
        self.assertEqual(got.shape, (4,))
        self.assertTrue(set(got.tolist()).issubset({-1.0, 1.0}))

    def test_even_in_mu_for_even_chain(self):
        """det h is even in mu for N even, so the sign is too."""
        n_sites = 20
        mu = torch.linspace(0.1, 3.7, 30, dtype=torch.float64)
        pos = chiral_block_det_sign(mu, n_sites, T, DELTA)
        neg = chiral_block_det_sign(-mu, n_sites, T, DELTA)
        self.assertTrue(torch.equal(pos, neg))


class TestReconstructBdgEigenvectorsBatched(TestCase):
    """Batched Torch reconstruction of BdG eigenvectors from a frame."""

    def test_matches_numpy_single_vector_version(self):
        n_sites = 8
        h = chiral_block(1.3, n_sites, T, DELTA)
        left, _singular, right_t = np.linalg.svd(h)
        u = torch.tensor(left.T, dtype=torch.float64)  # row k = u_k
        v = torch.tensor(right_t, dtype=torch.float64)

        for sign in (1, -1):
            got = reconstruct_bdg_eigenvectors(u, v, sign=sign).numpy()
            want = np.stack(
                [
                    reconstruct_bdg_eigenvector(left[:, k], right_t[k, :], sign=sign)
                    for k in range(n_sites)
                ]
            )
            self.assertTrue(np.allclose(got, want, atol=1e-12))

    def test_preserves_leading_axes_and_unit_norm(self):
        n_sites = 6
        u = torch.randn(3, 4, n_sites, dtype=torch.float64)
        u = u / u.norm(dim=-1, keepdim=True)
        # Make v share u's norm but be independent, then orthonormalise pairwise.
        v = torch.randn(3, 4, n_sites, dtype=torch.float64)
        v = v / v.norm(dim=-1, keepdim=True)
        psi = reconstruct_bdg_eigenvectors(u, v, sign=1)
        self.assertEqual(psi.shape, (3, 4, 2 * n_sites))
        # ||psi||^2 = (||u||^2 + ||v||^2) / 2 = 1 when u, v are unit and the
        # cross term u.v cancels between the particle and hole blocks.
        expected = 0.5 * (u.pow(2).sum(-1) + v.pow(2).sum(-1))
        self.assertTrue(torch.allclose(psi.pow(2).sum(-1), expected, atol=1e-12))


class TestResolveSvdSign(TestCase):
    """Reproducible per-column sign canonicalisation of an SVD frame."""

    def _frame(self):
        n_sites = 6
        h = chiral_block(2.7, n_sites, T, DELTA)
        left, _s, right_t = np.linalg.svd(h)
        u = torch.tensor(left, dtype=torch.float64).unsqueeze(0)
        v = torch.tensor(right_t.T, dtype=torch.float64).unsqueeze(0)
        return u, v

    def test_leading_entry_is_made_positive(self):
        u, v = self._frame()
        u_res, _v_res = resolve_svd_sign(u * -1.0, v * -1.0)
        # First above-tol entry of every column is now positive.
        lead = torch.where(
            u_res.abs() > 1e-6, u_res, torch.full_like(u_res, float("nan"))
        )
        first = lead[0].transpose(0, 1)  # column-major
        for col in first:
            nonnan = col[~torch.isnan(col)]
            self.assertGreater(nonnan[0].item(), 0.0)

    def test_idempotent_and_deterministic(self):
        u, v = self._frame()
        once = resolve_svd_sign(u, v)
        twice = resolve_svd_sign(*once)
        self.assertTrue(torch.equal(once[0], twice[0]))
        self.assertTrue(torch.equal(once[1], twice[1]))

    def test_matched_pair_flips_together(self):
        u, v = self._frame()
        u_res, v_res = resolve_svd_sign(u, v)
        # Column-wise, (u_res, v_res) is either (u, v) or (-u, -v).
        ratio_u = (u_res / u)[0]
        ratio_v = (v_res / v)[0]
        self.assertTrue(torch.allclose(ratio_u, ratio_v, atol=1e-9))

    def test_all_near_zero_column_is_left_alone(self):
        u = torch.zeros(1, 4, 1, dtype=torch.float64)
        v = torch.ones(1, 4, 1, dtype=torch.float64)
        u_res, v_res = resolve_svd_sign(u, v, tol=1e-6)
        self.assertTrue(torch.equal(u_res, u))
        self.assertTrue(torch.equal(v_res, v))

    def test_sign_is_detached(self):
        u, v = self._frame()
        u = u.clone().requires_grad_(True)
        v = v.clone().requires_grad_(True)
        u_res, v_res = resolve_svd_sign(u, v)
        (u_res.sum() + v_res.sum()).backward()
        self.assertFalse(torch.isnan(u.grad).any())
        self.assertFalse(torch.isnan(v.grad).any())


if __name__ == "__main__":
    import pytest

    pytest.main()
