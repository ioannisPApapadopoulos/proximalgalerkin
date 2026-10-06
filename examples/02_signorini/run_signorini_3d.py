from firedrake import *
from signorini_3d import *

n = 5
for refinements in [1,2]:
    for lmbda in [1e1, 1e2, 1e3, 1e4, 1e5]:
        problem = MTW(
            n=n,
            alpha0=1e-1,
            alpha_max=1e1,
            snes_atol=1e-5,
            model_parameter=lmbda,
            preconditioner="block_cg_bjacobi_cg_star",
            smoothing_its=8,
            max_pg_steps=40,
            pg_rtol=1e-3,
            refinements=refinements,
            degree=1,
            epsilon=1e-4,
            save_pvd=False,
        )
        solve_and_append(problem, "results/02_signorini_3d_results.csv")

        problem = Alfeld(
            n=n,
            alpha0=1e-1,
            alpha_max=1e1,
            snes_atol=1e-5,
            model_parameter=lmbda,
            preconditioner="block_cg_bjacobi_cg_star",
            smoothing_its=8,
            max_pg_steps=40,
            pg_rtol=1e-3,
            refinements=refinements,
            degree=3,
            epsilon=1e-4,
            save_pvd=False,
        )
        solve_and_append(problem, "results/02_signorini_3d_results.csv")

n = 20
for refinements in [2]:
    for lmbda in [1e1,1e2,1e3,1e4,1e4,1e5]:
        problem = OperatorPrecon(
            n=n,
            alpha0=1e-1,
            alpha_max=1e1,
            snes_atol=1e-5,
            model_parameter=lmbda,
            preconditioner="block_cg_bjacobi_cg_star",
            smoothing_its=8,
            max_pg_steps=40,
            pg_rtol=1e-3,
            refinements=refinements,
            degree=1,
            epsilon=1e-4,
            save_pvd=False,
        )
        solve_and_append(problem, "results/02_signorini_3d_results.csv")
