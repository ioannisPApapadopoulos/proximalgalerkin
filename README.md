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

The scripts currently only run on the Firedrake branch ```pbrubeck/slate-robust-multigrid```. To switch to this branch, you require a developer install of Firedrake (https://www.firedrakeproject.org/install.html#developer-install).

```
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/release/scripts/firedrake-configure
git clone https://gitlab.com/petsc/petsc.git
cd petsc
python3 ../firedrake-configure --show-petsc-configure-options | xargs -L1 ./configure
make PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-firedrake-default all
make check
cd ..
git clone git@github.com:firedrakeproject/firedrake.git --branch main
export $(python3 firedrake-configure --show-env)
python3 -m venv venv-firedrake
. venv-firedrake/bin/activate
pip cache purge
pip install $PETSC_DIR/src/binding/petsc4py
pip install -r ./firedrake/requirements-build.txt
pip install --no-build-isolation --no-binary h5py --editable './firedrake[check,docs,vtk,netgen]'
```

Then switch to the current branch and run make.

```
cd firedrake
git fetch
git checkout pbrubeck/slate-robust-multigrid
make
cd ..
```


Finally, you clone the ```proximalgalerkin``` package and pip install.

```
git clone git@github.com:ioannisPApapadopoulos/proximalgalerkin.git
cd proximalgalerkin
pip install .
```


### Tables and Figures