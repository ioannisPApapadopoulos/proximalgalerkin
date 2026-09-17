from firedrake import *
from obstacle_3d import *

for n in [20]:
    for refinements in [1,2]:
        for degree in [1,2]:
            problem = ObstacleProblem(
                    n=n,
                    alpha0=1e-4,
                    preconditioner="monolithic_vanka",
                    max_pg_steps=40,
                    pg_rtol=1e-3,
                    alpha_max=1e-1,
                    refinements=refinements,
                    degree=degree,
                    snes_atol=1e-6,
                    smoothing_its=5,
                    epsilon=0.0,
                    save_pvd=False,
                    overlap_no=2,
                )
            solve_and_append(problem, "results/01_obstacle_3d_results.csv")

for n in [20]:
    for refinements in [1,2]:
        for degree in [1,2]:
            problem = OperatorPrecon(
                    n=n,
                    alpha0=1e-4,
                    preconditioner="block_cg_bjacobi_chebyshev_jacobi",
                    max_pg_steps=40,
                    pg_rtol=1e-3,
                    alpha_max=1e-1,
                    refinements=refinements,
                    degree=degree,
                    snes_atol=1e-6,
                    smoothing_its=5,
                    epsilon=1e-4,
                    save_pvd=False,
                )
            solve_and_append(problem, "results/01_obstacle_3d_results.csv")

for n in [20]:
    for refinements in [1,2]:
        for degree in [1,2]:
            problem = OperatorPrecon(
                    n=n,
                    alpha0=1e-4,
                    preconditioner="block_cg_bjacobi_star",
                    max_pg_steps=40,
                    pg_rtol=1e-3,
                    alpha_max=1e-1,
                    refinements=refinements,
                    degree=degree,
                    snes_atol=1e-6,
                    smoothing_its=5,
                    epsilon=1e-4,
                    save_pvd=False,
                )
            solve_and_append(problem, "results/01_obstacle_3d_results.csv")


for n in [20]:
    for refinements in [1]:
        for degree in [1,2]:
            problem = Slate_Jacobi(
                    n=n,
                    alpha0=1e-4,
                    preconditioner="block_cg_bjacobi_chebyshev_jacobi",
                    max_pg_steps=40,
                    pg_rtol=1e-3,
                    alpha_max=1e-1,
                    refinements=refinements,
                    degree=degree,
                    snes_atol=1e-6,
                    smoothing_its=5,
                    epsilon=1e-4,
                    save_pvd=False,
                )
            solve_and_append(problem, "results/01_obstacle_slate_3d_results.csv")


            problem = Slate_Star(
                    n=n,
                    alpha0=1e-4,
                    preconditioner="block_cg_bjacobi_star",
                    max_pg_steps=40,
                    pg_rtol=1e-3,
                    alpha_max=1e-1,
                    refinements=refinements,
                    degree=degree,
                    snes_atol=1e-6,
                    smoothing_its=5,
                    epsilon=1e-4,
                    save_pvd=False,
                )
            solve_and_append(problem, "results/01_obstacle_slate_3d_results.csv")