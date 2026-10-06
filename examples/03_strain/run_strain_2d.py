from firedrake import *
from strain_2d import *

epsilon=1e-3
lmbda = 1e2
smoothing_its=5
for n in [40]:
    for refinements in [1,2]:
        for degree in [1]:
            problem =StrainProblem(
                n=n,
                alpha0=1e-2,
                alpha_max=1e1,
                model_parameter=lmbda,
                preconditioner="lu",
                max_pg_steps=40,
                refinements=refinements,
                degree=degree,
                save_pvd=False,
                snes_atol=1e-6,
                pg_rtol=1e-4,
            )
            solve_and_append(problem, "results/03_strain_results.csv")
            
            problem = OperatorPrecon(
                n=n,
                alpha0=1e-2,
                alpha_max=1e1,
                model_parameter=lmbda,
                preconditioner="block_cg_bjacobi_gmres_bjacobi",
                smoothing_its=smoothing_its,
                max_pg_steps=40,
                refinements=refinements,
                degree=degree,
                epsilon=epsilon,
                save_pvd=False,
                snes_atol=1e-6,
                pg_rtol=1e-4,
            )
            solve_and_append(problem, "results/03_strain_results.csv")

            problem = Slate(
                n=n,
                alpha0=1e-2,
                alpha_max=1e1,
                model_parameter=lmbda,
                preconditioner="block_cg_bjacobi_gmres_bjacobi",
                smoothing_its=smoothing_its,
                max_pg_steps=40,
                refinements=refinements,
                degree=degree,
                epsilon=epsilon,
                save_pvd=False,
                snes_atol=1e-6,
                pg_rtol=1e-4,
            )
            solve_and_append(problem, "results/03_strain_results.csv")