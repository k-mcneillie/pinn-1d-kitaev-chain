# PINN 1D Kitaev Chain

This repository implements physics-informed neural network (PINN)
surrogates for the spectral problem of the one-dimensional Kitaev chain, a
minimal lattice model of a topological superconductor. A neural network is
trained to reproduce the lowest non-negative eigenvalue and the
corresponding eigenvector of the chain's Bogoliubov-de Gennes (BdG)
Hamiltonian as a continuous function of the chemical potential, including
across the model's topological phase transition, using label-free physics
residuals, exact-diagonalization labels, or a combination of both.

## 1. Introduction

The system under study is a parametric family of Hermitian eigenvalue
problems: a Hamiltonian matrix $H(\mu)$ is defined for each value of a
scalar control parameter $\mu$, the chemical potential. The quantity of
interest is the continuous map

$$
\mu \;\longmapsto\; \big(E(\mu),\ \psi(\mu)\big),
$$

from $\mu$ to the lowest non-negative eigenvalue and its eigenvector,
approximated here by a neural network rather than obtained by repeated
numerical diagonalization. This problem is distinguished from a generic
parametric regression task by the presence of a topological phase
transition: at a critical value of $\mu$, the family of matrices $H(\mu)$
undergoes a qualitative change in eigenvector structure — from a
delocalized bulk state to a pair of states sharply localized at the two
ends of the chain — while the associated eigenvalue remains continuous. A
surrogate constructed by naive interpolation is not guaranteed to
represent this transition correctly, since the transition is not visible
in $E(\mu)$ alone. This provides a controlled test of whether
physics-informed training, which embeds the governing eigenvalue equation
directly in the training objective, recovers structure that supervised
regression on $E(\mu)$ and $\psi(\mu)$ separately would not.

The physical system of interest belongs to the study of topological
superconductivity and Majorana fermions, introduced in Sec. 2. The
methodological objective of this work is largely independent of that
physical context: to determine whether a PINN, trained with little or no
labeled data, can learn a parametric eigenvalue problem whose solution
changes qualitatively at a phase transition, and to compare the result
against a semi-supervised variant in which a small quantity of exact
diagonalization data is available.

## 2. Physical background

### 2.1 Topological phase transitions

A conventional phase transition — the loss of magnetic order, for example
— is characterized by a local order parameter that varies continuously or
discontinuously with a control parameter. A topological phase transition
is distinguished from this by the nature of the invariant that
characterizes it: a global, integer-valued topological invariant of the
system's band structure, rather than any local observable. Because this
invariant is discrete, it cannot vary continuously; a change in its value
requires the closing of the system's energy gap. This is the source of
the practical interest in topological phases: an observable protected by
a topological invariant is insensitive to smooth, local perturbations —
disorder, noise, small parameter variations — that would ordinarily
degrade a phase distinguished by local order.

### 2.2 Majorana fermions and Majorana bound states

A Majorana fermion is a fermion identical to its own antiparticle, a
possibility proposed by Majorana for elementary particles [1] but not
observed as such; it is, however, realizable as an emergent quasiparticle
excitation in certain condensed matter systems. In a topological
superconductor, single-particle excitations reorganize, in the
topological phase, into pairs of spatially separated Majorana bound
states (MBS): zero-energy modes localized at the two ends of a finite
topological superconducting wire. A single MBS does not by itself
constitute a usable qubit. A pair of spatially separated MBS jointly
encodes one ordinary fermionic degree of freedom non-locally, and because
the encoded information is distributed between two spatially separated
points, no local measurement or local noise process can access or
corrupt it in isolation. Combined with the non-Abelian exchange
statistics predicted for MBS, this non-local encoding is the physical
basis for a leading proposal for fault-tolerant, topologically protected
quantum computation [2, 3].

Two obstacles are established in the literature as the primary
limitations on realizing this protection experimentally: disorder-induced
in-gap states that reproduce the transport signatures of MBS without the
associated topology [4], and quasiparticle poisoning, in which
parity-nonconserving tunneling from an external reservoir corrupts the
encoded state [5]. Neither is treated computationally in this repository;
the present work concerns the equilibrium spectral problem only.

### 2.3 The Kitaev chain

More than one physical route to Majorana bound states exists. One is
intrinsic: a one-dimensional chain of spinless fermions with genuine
p-wave pairing, the Kitaev chain [6]. A second, physically distinct route
is induced: classical magnetic impurities placed in proximity to an
ordinary s-wave superconductor split Cooper pairs and generate localized
in-gap Yu-Shiba-Rusinov (YSR) states; a chain of closely spaced
impurities hybridizes these into YSR bands, and for a suitable spin
texture (helical order, or ferromagnetic order combined with spin-orbit
coupling), the same gap-closing-and-reopening transition occurs, hosting
MBS at the chain's ends by the same topological mechanism, despite the
underlying Hamiltonian being a two-dimensional, spinful tight-binding
lattice rather than a one-dimensional, spinless, intrinsically paired
one. This second route is the subject of prior dissertation research by
the present author [7], which studies the induced mechanism numerically
across a range of impurity-chain geometries and spin textures.

That dissertation's own methodology treats the one-dimensional Kitaev
chain as the field's standard reference model for identifying the
signatures of a topological phase transition, noting explicitly that
"although the Kitaev toy model differs from the model used in this paper,
it clearly outlines the signatures of a topological phase transition" and
that it "should act as a guide" [7, Sec. 3.1.1]. The present work adopts
that same reference model and poses a different question of it: not
whether it exhibits the correct physics, which is established, but
whether a neural network, trained under physics constraints derived from
it, can learn that physics directly. The Kitaev chain's minimality —
spinless, one-dimensional, nearest-neighbor coupling only — is what makes
it the appropriate computational starting point for that question: small
enough to diagonalize exactly at every parameter value, so that a
surrogate's predictions can be validated against ground truth everywhere,
while still exhibiting the full topological transition and hosting
genuine MBS. `KitaevChainHamiltonian` in this repository implements this
model; `notebooks/1d-kitaev-analytical.ipynb` presents its exact solution
and phase diagram.

## 3. Methodology: the Hamiltonian

The second-quantized Kitaev chain Hamiltonian, for $N$ sites with open
boundary conditions, is

$$
H \;=\; -\mu \sum_{n=1}^{N} c_n^\dagger c_n \;-\; \sum_{n=1}^{N-1}
\Big( t\, c_{n+1}^\dagger c_n + \Delta\, c_n c_{n+1} + \text{h.c.} \Big),
$$

where $c_n$ and $c_n^\dagger$ are fermionic annihilation and creation
operators on site $n$, $t$ is the nearest-neighbor hopping amplitude,
$\Delta$ is the (real, p-wave) pairing amplitude, and $\mu$ is the
chemical potential, treated in this work as the neural surrogate's input.
The pairing term $\Delta\, c_n c_{n+1}$ creates and annihilates pairs of
particles rather than conserving particle number, so $H$ cannot be
diagonalized in the ordinary particle-number (Fock) basis.

**Bogoliubov-de Gennes construction.** Because pairing mixes creation and
annihilation operators, the natural basis is the doubled Nambu spinor
basis $\Psi = (c_1, \dots, c_N,\ c_1^\dagger, \dots,
c_N^\dagger)^{\mathsf T}$, in which $H$ takes the form of a $2N \times 2N$
real symmetric matrix, the BdG Hamiltonian $H_{\mathrm{BdG}}$, with block
structure

$$
H_{\mathrm{BdG}} =
\begin{pmatrix} A & B \\ B^{\mathsf T} & -A \end{pmatrix},
$$

where $A$ (the particle sector) carries $-\mu$ on its diagonal and $-t$
hopping off-diagonal, $-A$ (the hole sector) is its particle-hole mirror,
and $B$ carries the pairing amplitude $\Delta$ with the antisymmetric
sign pattern required by fermionic statistics, $c_n c_{n+1} = -c_{n+1}
c_n$. This doubling introduces a redundancy — the BdG spectrum is
symmetric under $E \to -E$ — that follows from the enlarged basis rather
than from additional physics; it is, however, what allows a state pinned
to $E = 0$ to be interpreted as its own particle-hole conjugate, i.e., a
Majorana mode. Formally, this particle-hole symmetry is the statement
that $\Xi H_{\mathrm{BdG}}(\mu)\, \Xi = -H_{\mathrm{BdG}}(\mu)$ for the
particle-hole operator $\Xi$, an identity verified to hold exactly, to
floating-point precision, in this codebase (`PinnedFSMLoss` and
`SemiSupervisedLoss`'s `loss_ph` term is derived directly from it).

**Phase transition.** The infinite-chain (periodic) bulk dispersion is

$$
E(k) = \pm\sqrt{\big(\mu + 2t\cos k\big)^2 + 4\Delta^2 \sin^2 k},
$$

whose gap closes — the condition for a topological phase transition —
exactly at $\mu = \pm 2t$. For $|\mu| < 2t$, the finite, open chain is
topologically non-trivial and hosts a pair of exponentially localized
MBS, one at each end, split from exact zero energy only by a finite-size
correction $E \sim e^{-L/\xi}$ that vanishes as chain length $L \to
\infty$. For $|\mu| > 2t$, the chain is topologically trivial: the lowest
excitation is an extended bulk state with no edge localization. This is
the qualitative structural change referred to in Sec. 1, which a PINN
surrogate for $\mu \mapsto (E, \psi)$ must resolve rather than
interpolate through.

## 4. Methodology: physics-informed neural network training

Rather than diagonalizing $H_{\mathrm{BdG}}(\mu)$ at every $\mu$ of
interest, a SIREN-based neural network [8] — sinusoidal activations,
selected because their derivatives remain smooth sinusoids, which is
required once the physics residual involves differentiating the network
with respect to its own input and output — is trained to represent $\mu
\mapsto \psi(\mu)$, and, in the dual-head variant, $E(\mu)$ jointly. Two
training regimes are implemented and compared.

**Unsupervised, physics-residual-only training**
(`kitaev.training.loss.PinnedFSMLoss`, paired with the single-head
`SirenPINN`) does not use exact diagonalization labels. The network is
trained by penalizing the Schrodinger residual $\lVert H(\mu)\psi -
E_{\text{Rayleigh}}\psi \rVert^2$, where $E_{\text{Rayleigh}}$ is the
model's own Rayleigh quotient rather than a ground-truth label, together
with an annealed soft constraint toward a non-negative Rayleigh quotient
and architectural hard constraints such as eigenvector normalization
enforced in the network's forward pass. This regime constitutes the
methodological contribution the present work considers most significant:
the network is required to recover the correct, topologically
non-trivial spectrum from the governing equation alone.

**Semi-supervised training** (`kitaev.training.loss.SemiSupervisedLoss`,
paired with the dual-head `SirenPINNDualHead`) supplements the same
physics residual with a small set of exact $(\mu, E, \psi)$ labels that
anchor the model's absolute energy scale. This regime is deliberately
regarded as the less significant contribution of the two: it substitutes
a small quantity of exact supervision unavailable to the unsupervised
approach, and its principal function in this repository is validation of
the surrounding training framework (`UnifiedTrainer` and the shared
loss/data/model abstractions) rather than a claim of methodological
novelty in its own right.

## 5. Relation to prior literature and positioning

Physics-informed neural networks for parametric quantum eigenvalue
problems are an established line of work following the original PINN
formulation of Raissi et al. [9], including unsupervised PINN treatments
of the finite square well and the hydrogen atom [10]. Machine learning has
separately been applied to Kitaev chains directly, though in a different
mode: convolutional networks trained to autonomously tune physical
minimal-Kitaev-chain devices toward a Majorana "sweet spot" using
experimental conductance data [11, 12]. This is a device-control problem
rather than a spectral-surrogate problem.

A literature search conducted in the course of this work did not
identify prior application of a physics-informed neural surrogate to the
Kitaev chain's own BdG spectral problem — that is, treating $\mu \mapsto
(E, \psi)$ across a topological phase transition as the PINN's target,
as distinct from device tuning or a non-topological quantum eigenvalue
problem. Within the specific research program this work extends — subgap
states induced by magnetic impurity chains on two-dimensional
superconductors [7, 13, 14, 15] — no machine-learning treatment was
identified either. This absence is stated as the basis for treating the
approach as a plausible candidate for methodological novelty, to be
confirmed by a systematic literature review rather than asserted as an
established result.

## 6. Relation to prior and ongoing research

The Kitaev chain is used here specifically because it can be diagonalized
exactly at every training and evaluation point, which allows a PINN
methodology to be validated against ground truth throughout training. It
is not intended as an end target. The training framework developed
here — the loss formulations, the hard-constraint architectural
patterns, and the comparison between unsupervised and semi-supervised
regimes — is intended to generalize to the higher-dimensional, less
tractable lattice models studied in the author's prior dissertation
research [7]: two-dimensional magnetic-impurity chains on superconducting
substrates, for which exact diagonalization is substantially more
computationally expensive and a trained PINN surrogate would be of
direct practical value rather than solely a methodological
demonstration. A manuscript scaffold exists at `docs/write_up/main.tex`
(template content at the time of writing), pending confirmation of the
novelty hypothesis stated in Sec. 5.

## 7. Repository organization

```text
src/kitaev/
  analytical/       Exact Hamiltonian construction and diagonalisation
                     (KitaevChainHamiltonian).
  data/              mu sampling schemes (MuSampler, SamplingRegion) and
                     dataset/generator classes for labelled and label-free
                     training data.
  models/            SIREN-based network architectures: SirenPINN
                     (single-head, unsupervised) and SirenPINNDualHead
                     (dual-head, semi-supervised).
  training/          Loss functions (PinnedFSMLoss, SemiSupervisedLoss),
                     UnifiedTrainer (the single training entrypoint for
                     both regimes), and supporting utilities.
  visualisation/     Reusable, tested evaluation and plotting functions
                     for comparing a trained model against exact
                     diagonalisation.
notebooks/
  1d-kitaev-analytical.ipynb        Exact solution and phase diagram.
  unsupervised/                     Unsupervised, physics-residual-only
                                     training (PinnedFSMLoss) and hard-
                                     constraint variations; the primary
                                     locus of this work's methodological
                                     contribution (Sec. 4-5).
  semi-supervised/                  Semi-supervised training walkthrough.
  legacy/                           Early exploratory work.
docs/
  papers/            Reference literature, including the author's prior
                      dissertation and literature review.
  write_up/          Manuscript scaffold (LaTeX).
tests/               Mirrors src/kitaev/; 100% coverage enforced in CI.
```

## 8. Installation and usage

### Environment setup

```bash
conda env create -f environment.yml
conda activate pinn-ml-py3.12
pip install -e .
```

### Tests and quality gates

This repository uses the `just` command runner:

```bash
just            # list all available commands
just test       # run the pytest suite with coverage
just check-all  # lint, format-check, type-check, and test
```

### Notebooks

`notebooks/1d-kitaev-analytical.ipynb` presents the exact solution and
phase diagram. `notebooks/semi-supervised/dual-head-experiement.ipynb`
presents a complete semi-supervised training run, including methodology
and results.

## 9. Citation

```text
McNeillie, K. (2026). PINN 1D Kitaev Chain. GitHub Repository.
https://github.com/k-mcneillie/pinn-1d-kitaev-chain
```

## References

[1] E. Majorana, "Teoria simmetrica dell'elettrone e del positrone," Nuovo Cim. 14, 171 (1937).

[2] A. Kitaev, "Fault-tolerant quantum computation by anyons," Ann. Phys. 303, 2 (2003).

[3] V. Lahtinen and J. K. Pachos, "A Short Introduction to Topological Quantum Computation," SciPost Phys. 3, 021 (2017).

[4] H. Pan and S. Das Sarma, "Physical mechanisms for zero-bias conductance peaks in Majorana nanowires," Phys. Rev. Res. 2, 013377 (2020).

[5] S. Das Sarma, M. Freedman, and C. Nayak, "Majorana zero modes and topological quantum computation," npj Quantum Inf. 1, 15001 (2015).

[6] A. Y. Kitaev, "Unpaired Majorana fermions in quantum wires," Phys.-Usp. 44, 131 (2001).

[7] K. McNeillie, "Numerical investigation of topological states near magnetic structures on a superconductor," dissertation, University of Edinburgh, supervised by B. Braunecker (2023).

[8] V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G. Wetzstein, "Implicit Neural Representations with Periodic Activation Functions," Advances in Neural Information Processing Systems 33 (2020).

[9] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," J. Comput. Phys. 378, 686 (2019).

[10] "Physics-Informed Neural Networks for Quantum Eigenvalue Problems," arXiv:2203.00451 (2022).

[11] "Majorana Qubits and Non-Abelian Physics in Quantum Dot-Based Minimal Kitaev Chains," PRX Quantum 5, 010323 (2024).

[12] "Cross-Platform Autonomous Control of Minimal Kitaev Chains," PRX Intelligence, DOI:10.1103/ryll-qb42.

[13] C. Mier, D.-J. Choi, and N. Lorente, "Calculations of in-gap states of ferromagnetic spin chains on s-wave wide-band superconductors," Phys. Rev. B 104, 245415 (2021).

[14] C. J. F. Carroll and B. Braunecker, "Subgap states at ferromagnetic and spiral-ordered magnetic chains in two-dimensional superconductors. I. Continuum description," Phys. Rev. B 104, 245133 (2021).

[15] C. J. F. Carroll and B. Braunecker, "Subgap states at ferromagnetic and spiral-ordered magnetic chains in two-dimensional superconductors. II. Topological classification," Phys. Rev. B 104, 245134 (2021).
