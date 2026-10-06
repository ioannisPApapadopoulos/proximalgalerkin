# Preconditioning Proximal Galerkin

This repository contains a package that implements preconditioners for the proximal Galerkin algorithm applied to solving variational inequalities. In particular

- obstacle problems
- Signorini problems
- Strain-constrained elasticity problems

The main preconditioner is a block preconditioning approach via an elimination of the latent block. We use preconditioned CG for the latent block and geometric multigrid for an approximation of the Schur complement. This Schur approximation is constructed either via a discretization of a specific PDE or algebraically via Slate. This package also supports monolithic multigrid with Vanka-type relaxation.

The scripts in [examples](https://github.com/ioannisPApapadopoulos/proximalgalerkin/tree/main/examples) folder require an installation of the ```proximalgalerkin``` package.

We also include [self-contained scripts](https://github.com/ioannisPApapadopoulos/proximalgalerkin/tree/main/self-contained-scripts) that do not require installation of the ```proximalgalerkin``` package and run on the current Firedrake release branch.

Further details about the preconditioners can be found in the paper 'Preconditioning proximal Galerkin: mesh, degree, and parameter robust solvers for variational inequalities', I. P. A. Papadopoulos, P. D. Brubeck (2026).

## Self-contained scripts

In the [self-contained-scripts](https://github.com/ioannisPApapadopoulos/proximalgalerkin/tree/main/self-contained-scripts) directory we include 3 scripts for the obstacle, Signorini, and strain-constrained elasticity problems, respectively, which are "self-contained". They do not require installation of this package and run on the release branch of Firedrake.

They implement the block preconditioner where the Schur complement is approximated by a PDE and the
PDE is inverted by geometric multigrid with Chebyshev relaxation and preconditioned by a Jacobi iteration.

### Installation

These scripts use Firedrake and VTK (to plot the solutions).

Follow the instructions here for installing Firedrake (https://www.firedrakeproject.org/install.html#installing-firedrake). To also install netgen and VTK, please run:

```
pip install --no-binary h5py 'firedrake[check,vtk]'
```

## proximalgalerkin package

This package is for running the scripts required to generate the Figures and Tables in the manuscript.

### Installation

The scripts in examples/ currently only run on the Firedrake branch ```pbrubeck/slate-robust-multigrid```. To switch to this branch, you require a developer install of Firedrake (https://www.firedrakeproject.org/install.html#developer-install).

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

Then switch to the correct branch and run make.

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

Scripts to generate the Tables and Figures found in the manuscript.

|Figure|File: examples/|
|:-:|:-:|
|6a,b|[run_obstacle_2d.py](https://github.com/ioannisPApapadopoulos/proximalgalerkin/tree/main/examples/01_obstacle/run_obstacle_2d.py)|
|6c|[run_obstacle_3d.py](https://github.com/ioannisPApapadopoulos/proximalgalerkin/tree/main/examples/01_obstacle/run_obstacle_3d.py)|
|7a|[run_signorini_2d.py](https://github.com/ioannisPApapadopoulos/proximalgalerkin/tree/main/examples/02_signorini/run_signorini_2d.py)|
|7b|[run_signorini_3d.py](https://github.com/ioannisPApapadopoulos/proximalgalerkin/tree/main/examples/02_signorini/run_signorini_3d.py)|
|8a|[run_strain_2d.py](https://github.com/ioannisPApapadopoulos/proximalgalerkin/tree/main/examples/03_strain/run_strain_2d.py)|
|8b|[run_strain_3d.py](https://github.com/ioannisPApapadopoulos/proximalgalerkin/tree/main/examples/03_strain/run_strain_3d.py)|


|Table|File: examples/|
|:-:|:-:|
|2,3|[run_obstacle_2d.py](https://github.com/ioannisPApapadopoulos/proximalgalerkin/tree/main/examples/01_obstacle/run_obstacle_2d.py)|
|4|[run_obstacle_3d.py](https://github.com/ioannisPApapadopoulos/proximalgalerkin/tree/main/examples/01_obstacle/run_obstacle_3d.py)|
|5,6|[run_signorini_2d.py](https://github.com/ioannisPApapadopoulos/proximalgalerkin/tree/main/examples/02_signorini/run_signorini_2d.py)|
|7|[run_signorini_3d.py](https://github.com/ioannisPApapadopoulos/proximalgalerkin/tree/main/examples/02_signorini/run_signorini_3d.py)|
|8|[run_strain_2d.py](https://github.com/ioannisPApapadopoulos/proximalgalerkin/tree/main/examples/03_strain/run_strain_2d.py)|
|8|[run_strain_3d.py](https://github.com/ioannisPApapadopoulos/proximalgalerkin/tree/main/examples/03_strain/run_strain_3d.py)|
