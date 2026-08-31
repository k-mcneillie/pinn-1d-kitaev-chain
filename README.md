# Physics-Informed Neural Eigensolvers for the 1-D Kitaev Chain

This repository studies how symmetry constraints, carried in the loss, in the
architecture, or in the choice of representation, govern whether a
physics-informed neural network (PINN) can solve a parametric spectral problem
with a symmetry-protected degeneracy. The test system is the one-dimensional
Kitaev chain [1], a minimal model of a topological superconductor. A network is
trained to map the chemical potential $\mu$ to the lowest non-negative
Bogoliubov–de Gennes (BdG) eigenpair across the topological transition at
$|\mu| = 2t$, using label-free physics residuals (one variant also uses a small
quantity of exact-diagonalisation data). The chain is exactly solvable, so the
work is methodological: every prediction is checked against exact
diagonalisation at every $\mu$.

Across the transition the lowest eigenvector changes character, from a
delocalised bulk state to a pair of end-localised Majorana modes, while the
energy $E(\mu)$ stays continuous. The eigenvector is therefore the
discriminating target, and recovering it at each $\mu$ is where the models
separate. Four models solve the same problem with the physics constraints moved
progressively from the loss into the architecture and then into the
representation.

The Kitaev Hamiltonian for $N$ sites with open boundaries is

$$
H = -\mu \sum_{n=1}^{N} c_n^\dagger c_n
  - \sum_{n=1}^{N-1}\!\left( t\, c_{n+1}^\dagger c_n
  + \Delta\, c_n c_{n+1} + \mathrm{h.c.} \right),
$$

with $t$ the nearest-neighbour hopping and $\Delta$ the $p$-wave pairing
amplitude. Throughout the repository $N = 20$, $t = 1$, $\Delta = 0.5\,t$ unless
stated otherwise, so $H(\mu)$ in the Bogoliubov–de Gennes form is a
$40 \times 40$ real symmetric matrix, affine in the input $\mu$.

## Key results

- A particle–hole penalty commonly added to BdG PINN losses is algebraically
  equal, in value and in gradient, to the eigenvector-consistency residual it
  accompanies, so it adds nothing to the optimisation.
- In the topological phase the folded-spectrum objective is flat across the
  two-dimensional near-zero eigenspace. The term that would select the energy
  eigenstate has depth of order $e^{-2N/\xi}$, below optimiser tolerance, and it
  shrinks as the chain lengthens. More sampling or a larger network does not
  resolve this; a symmetry-adapted basis does.
- A fixed rotation to the chiral basis of symmetry class BDI reduces the
  $2N$-dimensional eigenproblem to the singular value decomposition of an
  $N \times N$ bidiagonal matrix. The near-zero degeneracy becomes a simple
  singular value, the per-site eigenvector becomes determined, and the loss goes
  from four scheduled terms to two unweighted ones with no loss of accuracy on
  any gauge-invariant quantity.

Full derivations, the loss definitions, and the two supporting propositions are
in the manuscript (`docs/write-up/main.tex`). The complete quantitative record
is in `FINDINGS.md`.

![Predicted particle and hole site density of the lowest BdG mode versus mu, for five independently seeded runs of the chiral model, overlaid on exact diagonalisation](https://github.com/user-attachments/assets/da0dde8e-2331-4866-a426-8c4e05977a27)

**Figure 1.** Predicted particle and hole site density of the lowest BdG mode as
a function of $\mu$, for five independently seeded runs of the chiral model
(`SirenPINNChiral` with `ChiralFSMLoss`), overlaid on exact diagonalisation. The
seeds are indistinguishable from the reference in both phases, including through
the transition. The same plot for the Nambu-basis models is a spread of mutually
inconsistent one-sided profiles in the topological phase.

## Installation

```bash
conda env create -f environment.yml
conda activate pinn-ml-py3.12
pip install -e .
```

The repository uses the `just` command runner:

```bash
just              # list recipes
just test         # pytest with coverage
just check-all    # lint, format-check, type-check, test
```

## Usage

```bash
python experiments/four_model_comparison.py                            # five-seed, four-model reference run
python experiments/four_model_comparison.py --figures-only <run-dir>   # regenerate figures from a run
python experiments/chiral_n_sweep.py                                   # system-size sweep
```

`notebooks/1d-kitaev-analytical.ipynb` presents the exact solution and phase
diagram. The per-model notebooks under `notebooks/unsupervised/` and
`notebooks/semi-supervised/` each walk through one model on a shared skeleton.

## Models

All models share an 11k-parameter sinusoidal representation network (SIREN) [3]
backbone with a per-model head. The energy is never a network output; it is the
Rayleigh quotient of the predicted state, formed by the loss or the evaluation
adapter. A model-agnostic probe scores $(E, \psi)$ against `numpy.linalg.eigh`
on a fixed $\mu$ grid during training.

| Model | Network | Loss | Notebook | Role |
|---|---|---|---|---|
| 1 · semi-supervised | `SirenPINN` | `SemiSupervisedFSMLoss` | `semi-supervised/semi-supervised-fsm.ipynb` | A few exact labels alongside the physics residuals. |
| 2 · Nambu baseline | `SirenPINN` | `PinnedFSMLoss` | `unsupervised/nambu-fsm-baseline.ipynb` | Label-free, four-term soft-penalty loss in the Nambu basis. |
| 3 · structural Nambu | `SirenPINNNambuFolded` | `NambuFSMLoss` | `unsupervised/structural-nambu.ipynb` | Redundant and pinned terms removed by symmetry argument; the $\mu$-parity folded into the architecture; still the Nambu basis. |
| 4 · chiral | `SirenPINNChiral` | `ChiralFSMLoss` | `unsupervised/chiral-pinn-experiment.ipynb` | Chiral (BDI) basis; SVD of the $N \times N$ bidiagonal block; two-term unweighted loss. |

## Comparison

Reference run `results/logs/20260830_canonical_four-model-comparison/`, five
seeds per model, final-epoch state. Entries are the median over seeds with the
worst seed in parentheses. Gauge-invariant quantities are recovered label-free
by every model; the models separate on the single-vector topological density.
See `FINDINGS.md` for the full metric set.

| Metric (topological phase) | Model 1 | Model 2 | Model 3 | Model 4 |
|---|---|---|---|---|
| Energy MAE | $2.3\times10^{-4}\ (2.6\times10^{-4})$ | $3.4\times10^{-5}\ (1.1\times10^{-4})$ | $1.0\times10^{-5}\ (5.9\times10^{-5})$ | $2.9\times10^{-6}\ (1.2\times10^{-5})$ |
| Subspace infidelity $1 - \lVert P_M\psi\rVert$ | $2.7\times10^{-4}\ (4.5\times10^{-4})$ | $2.6\times10^{-5}\ (3.1\times10^{-4})$ | $4.4\times10^{-6}\ (3.5\times10^{-5})$ | $4.5\times10^{-6}\ (2.1\times10^{-5})$ |
| Single-vector density MAE | $1.6\times10^{-2}\ (2.0\times10^{-2})$ | $2.6\times10^{-2}\ (3.0\times10^{-2})$ | $2.7\times10^{-2}$ (seed range $2.4\times10^{-5}$ to $4.3\times10^{-2}$) | $1.5\times10^{-4}\ (3.2\times10^{-4})$ |

## Repository layout

```text
src/kitaev/
  analytical/     Exact Hamiltonian and diagonalisation (KitaevChainHamiltonian);
                  the Majorana/chiral reduction (the unitary Omega, the N x N
                  bidiagonal block h(mu), eigenvector reconstruction).
  data/           mu sampling and labelled / label-free dataset generators.
  models/         SIREN architectures: SirenPINN, SirenPINNNambuFolded,
                  SirenPINNChiral, and the sine layer.
  training/       Loss functions (PinnedFSMLoss, NambuFSMLoss,
                  SemiSupervisedFSMLoss, ChiralFSMLoss),
                  the two-phase trainer, samplers, probes, callbacks.
  visualisation/  Evaluation and plotting against exact diagonalisation.
  xai/            Post-hoc interpretability on a completed comparison run.
experiments/
  four_model_comparison.py   Five-seed, four-model reference run.
  chiral_n_sweep.py          System-size sweep (chiral vs structural Nambu).
notebooks/
  1d-kitaev-analytical.ipynb        Exact solution and phase diagram.
  unsupervised/, semi-supervised/   One notebook per model on a shared skeleton.
  xai/                              Interpretability and animations.
docs/write-up/    Manuscript (LaTeX): derivations, loss definitions, proofs, protocol.
FINDINGS.md       Canonical, growing record of quantitative results.
tests/            Mirrors src/kitaev/; coverage enforced in CI.
```

## Citation

```text
McNeillie, K. (2026). Physics-Informed Neural Eigensolvers for the 1-D
Kitaev Chain. GitHub repository.
https://github.com/k-mcneillie/pinn-1d-kitaev-chain
```

## References

[1] A. Y. Kitaev, "Unpaired Majorana fermions in quantum wires," Phys.-Usp. **44**, 131 (2001).

[2] K. McNeillie, "Numerical investigation of topological states near magnetic structures on a superconductor," dissertation, University of St Andrews, supervised by B. Braunecker (2023).

[3] V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G. Wetzstein, "Implicit Neural Representations with Periodic Activation Functions," Adv. Neural Inf. Process. Syst. **33** (2020).

[4] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," J. Comput. Phys. **378**, 686 (2019).

[5] H. Jin, M. Mattheakis, and P. Protopapas, "Physics-Informed Neural Networks for Quantum Eigenvalue Problems," arXiv:2203.00451 (2022).
