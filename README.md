# Preconditioning Proximal Galerkin

This repository contains a package that implements preconditioners for the proximal Galerkin algorithm applied to solving variational inequalities. In particular
    - obstacle problem
    - Signorini problem
    - Strain-constrained elasticity problem

The preconditioners can be grouped into two classes: monolithic multigrid with Vanka-type relaxation and a block preconditioning approach with preconditioned CG for the ''top-left'' block and geometric multigrid for an approximation of the Schur complement.

We also include self-contained scripts that do not require installation of the ```proximalgalerkin``` package and run on the current Firedrake release branch.

Further details can be found in the paper `Preconditioning proximal Galerkin: mesh, degree, and parameter robust solvers for variational inequalities', I. P. A. Papadopoulos, P. D. Brubeck (2026).

## proximalgalerkin package

### Installation

The scripts currently only run on the Firedrake branch ```pbrubeck/slate-robust-multigrid```. To switch to this branch, you require a developer install of Firedrake.


Then switch to the current branch and run make.


Finally, you can install the ```proximalgalerkin``` package.


### Tables and Figures