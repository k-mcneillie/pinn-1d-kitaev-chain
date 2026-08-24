# tests/test_hamiltonian.py
from __future__ import annotations

from unittest import TestCase

import numpy as np
from numpy.linalg import eigvalsh

from kitaev.analytical import KitaevChainHamiltonian


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


if __name__ == "__main__":
    import pytest

    pytest.main()
