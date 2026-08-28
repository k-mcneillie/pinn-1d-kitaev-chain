# tests/test_hamiltonian.py
from __future__ import annotations

from unittest import TestCase

import numpy as np
import torch
from numpy.linalg import eigvalsh

from kitaev.analytical import KitaevChainHamiltonian, bdg_block_batched


class TestKitaevChainHamiltonian(TestCase):  # noqa: N806
    """
    Comprehensive test suite for the KitaevChainHamiltonian class.
    """

    def test_construction(self):
        """
        Test instance initialization and basic properties.
        """
        # Test defaults
        h = KitaevChainHamiltonian()
        self.assertEqual(h.n_sites, 20)
        self.assertEqual(h.hopping, 1.0)
        self.assertEqual(h.pairing, 0.5)
        self.assertEqual(h.dim, 40)

        # Test custom parameters
        h = KitaevChainHamiltonian(n_sites=10, hopping=0.5, pairing=0.2)
        self.assertEqual(h.n_sites, 10)
        self.assertEqual(h.hopping, 0.5)
        self.assertEqual(h.pairing, 0.2)
        self.assertEqual(h.dim, 20)

    def test_matrix_shape(self):
        """
        Test matrix dimensions and symmetry.
        """
        h = KitaevChainHamiltonian()
        H = h.build(0.5)
        self.assertEqual(H.shape, (40, 40))
        self.assertTrue(np.isrealobj(H))
        self.assertTrue(np.allclose(H, H.T))

    def test_explicit_entries(self):
        """
        Test specific matrix entries for correctness.
        """
        h = KitaevChainHamiltonian(n_sites=2)
        H = h.build(0.5)
        N = 2

        # Test diagonal entries (particle and hole sectors)
        self.assertEqual(H[0, 0], -0.5)
        self.assertEqual(H[N + 0, N + 0], 0.5)
        self.assertEqual(H[1, 1], -0.5)
        self.assertEqual(H[N + 1, N + 1], 0.5)

        # Test hopping entries
        self.assertEqual(H[0, 1], -1.0)
        self.assertEqual(H[1, 0], -1.0)
        self.assertEqual(H[N, N + 1], 1.0)
        self.assertEqual(H[N + 1, N], 1.0)

        # Test pairing entries
        self.assertEqual(H[0, N + 1], 0.5)
        self.assertEqual(H[N + 1, 0], 0.5)
        self.assertEqual(H[1, N], -0.5)
        self.assertEqual(H[N, 1], -0.5)

    def test_block_structure(self):
        """
        Test matrix block structure (H = [[A, B], [B^T, -A]]).
        """
        h = KitaevChainHamiltonian(n_sites=2)
        H = h.build(0.5)
        N = 2

        A = H[:N, :N]
        B = H[:N, N:]
        B_t = B.T

        # Test block relations
        self.assertTrue(np.allclose(H[N:, :N], B_t))
        self.assertTrue(np.allclose(H[N:, N:], -A))

    def test_particle_hole_symmetry(self):
        """
        Test particle-hole symmetry in eigenvalues.
        """
        h = KitaevChainHamiltonian()
        H = h.build(0.5)
        w = eigvalsh(H)
        w_sorted = np.sort(w)

        # Check symmetry
        self.assertTrue(np.allclose(w_sorted, -w_sorted[::-1]))

    def test_edge_cases(self):
        """
        Test edge cases for n_sites and parameters.
        """
        # Test n_sites=1
        h = KitaevChainHamiltonian(n_sites=1)
        H = h.build(0.5)
        self.assertEqual(H.shape, (2, 2))
        self.assertTrue(np.allclose(H.diagonal(), [-0.5, 0.5]))

        # Test n_sites=0
        h = KitaevChainHamiltonian(n_sites=0)
        H = h.build(0.5)
        self.assertEqual(H.shape, (0, 0))

        # Test hopping=0.0: particle-sector hopping should vanish,
        # but pairing (delta) and on-site mu terms remain untouched.
        h = KitaevChainHamiltonian(hopping=0.0)
        H = h.build(0.5)
        N = h.n_sites

        # on-site terms: particle sector is -mu, hole sector is +mu
        self.assertTrue(np.isclose(H[0, 0], -0.5))
        self.assertTrue(np.isclose(H[N, N], 0.5))

        # hopping off-diagonals should be exactly zero
        self.assertTrue(np.isclose(H[0, 1], 0.0))
        self.assertTrue(np.isclose(H[N, N + 1], 0.0))

        # pairing terms should be unaffected by hopping=0
        self.assertTrue(np.isclose(H[0, N + 1], h.pairing))

    def test_topological_phase(self):
        """
        Test topological phase transition.
        """
        h = KitaevChainHamiltonian(n_sites=20, hopping=1.0)

        # Topological phase (mu well inside |mu| < 2t, away from the
        # transition where finite-size splitting grows) — near mu=0,
        # the splitting is smallest and most robust to N.
        w = eigvalsh(h.build(0.5))
        self.assertLess(np.min(np.abs(w)), 1e-4)

        # Trivial phase (mu > 2t)
        w = eigvalsh(h.build(3.0))
        self.assertGreater(np.min(np.abs(w)), 1.0)

    def test_callable(self):
        """
        Test __call__ method.
        """
        h = KitaevChainHamiltonian()
        H1 = h.build(0.5)
        H2 = h(0.5)
        self.assertTrue(np.array_equal(H1, H2))


class TestBdgBlockBatched(TestCase):  # noqa: N806
    """Test suite for the batched Torch BdG builder ``bdg_block_batched``."""

    def test_matches_kitaev_chain_hamiltonian_build(self):
        """Each slice reproduces ``KitaevChainHamiltonian.build`` exactly."""
        n_sites, t, d = 6, 1.0, 0.5
        ham = KitaevChainHamiltonian(n_sites=n_sites, hopping=t, pairing=d)
        mu = torch.linspace(-4.0, 4.0, 17, dtype=torch.float64)

        batched = bdg_block_batched(mu, n_sites, hopping=t, pairing=d)

        self.assertEqual(batched.shape, (17, 2 * n_sites, 2 * n_sites))
        for k, mu_k in enumerate(mu.tolist()):
            expected = torch.tensor(ham.build(mu_k), dtype=torch.float64)
            self.assertTrue(torch.allclose(batched[k], expected, atol=1e-12))

    def test_each_slice_is_real_symmetric(self):
        """Every batch element is a real symmetric matrix."""
        mu = torch.tensor([[-2.7], [0.0], [3.1]])
        batched = bdg_block_batched(mu, 5, hopping=1.0, pairing=0.3)
        for slab in batched:
            self.assertTrue(torch.allclose(slab, slab.T, atol=1e-6))

    def test_linear_in_mu(self):
        """Only the diagonal moves with mu; the off-diagonal blocks are fixed."""
        n_sites = 5
        mu = torch.tensor([-1.5, 0.4, 2.9])
        batched = bdg_block_batched(mu, n_sites, hopping=1.0, pairing=0.5)

        off_diag_mask = ~torch.eye(2 * n_sites, dtype=torch.bool)
        first_off = batched[0][off_diag_mask]
        for slab in batched[1:]:
            self.assertTrue(torch.allclose(slab[off_diag_mask], first_off))

        # Diagonal: -mu on the particle sector, +mu on the hole sector.
        for k, mu_k in enumerate(mu.tolist()):
            diag = torch.diagonal(batched[k])
            particle = torch.full((n_sites,), -mu_k)
            hole = torch.full((n_sites,), mu_k)
            self.assertTrue(torch.allclose(diag[:n_sites], particle))
            self.assertTrue(torch.allclose(diag[n_sites:], hole))

    def test_accepts_1d_and_2d_mu_identically(self):
        """``(B,)`` and ``(B, 1)`` inputs give the same result."""
        flat = torch.tensor([-3.0, 1.0, 2.5])
        column = flat.unsqueeze(-1)
        self.assertTrue(
            torch.allclose(
                bdg_block_batched(flat, 4),
                bdg_block_batched(column, 4),
            )
        )

    def test_dtype_follows_input(self):
        """The output dtype and device track ``mu_batch``."""
        mu = torch.tensor([0.5, -0.5], dtype=torch.float64)
        out = bdg_block_batched(mu, 3)
        self.assertEqual(out.dtype, torch.float64)
        self.assertEqual(out.device, mu.device)


if __name__ == "__main__":
    import pytest

    pytest.main()
