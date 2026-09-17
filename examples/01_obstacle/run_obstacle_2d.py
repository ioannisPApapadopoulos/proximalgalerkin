from firedrake import *
from obstacle_2d import *

for n in [32,64]:
    for refinements in [1,2]:
        for degree in [1,2,3]:
            if n == 32 and refinements == 2:
                break
            problem = ObstacleProblem(
                    n=n,
                    alpha0=1e-1,
                    preconditioner="lu",
                    max_pg_steps=40,
                    pg_rtol=1e-3,
                    alpha_max=30.0,
                    refinements=refinements,
                    degree=degree,
                    snes_atol=1e-5,
                    smoothing_its=5,
                    epsilon=0.0,
                    save_pvd=False,
                )
            solve_and_append(problem, "results/01_obstacle_results.csv")

for n in [32,64]:
    for refinements in [1,2]:
        for degree in [1,2,3]:
            problem = ObstacleProblem(
                    n=n,
                    alpha0=1e-1,
                    preconditioner="monolithic_vanka",
                    max_pg_steps=40,
                    pg_rtol=1e-3,
                    alpha_max=30.0,
                    refinements=refinements,
                    degree=degree,
                    snes_atol=1e-5,
                    smoothing_its=5,
                    epsilon=0.0,
                    save_pvd=False,
                    overlap_no=2,
                )
            solve_and_append(problem, "results/01_obstacle_results.csv")

for n in [32,64]:
    for refinements in [1,2]:
        for degree in [1,2,3]:
            problem = OperatorPrecon(
                    n=n,
                    alpha0=1e-1,
                    preconditioner="block_cg_bjacobi_chebyshev_jacobi",
                    max_pg_steps=40,
                    pg_rtol=1e-3,
                    alpha_max=30.0,
                    refinements=refinements,
                    degree=degree,
                    snes_atol=1e-5,
                    smoothing_its=5,
                    epsilon=1e-5,
                    save_pvd=False,
                )
            solve_and_append(problem, "results/01_obstacle_results.csv")

for n in [32,64]:
    for refinements in [1,2]:
        for degree in [1,2,3]:
            problem = OperatorPrecon(
                    n=n,
                    alpha0=1e-1,
                    preconditioner="block_cg_bjacobi_star",
                    max_pg_steps=40,
                    pg_rtol=1e-3,
                    alpha_max=30.0,
                    refinements=refinements,
                    degree=degree,
                    snes_atol=1e-5,
                    smoothing_its=5,
                    epsilon=1e-5,
                    save_pvd=False,
                )
            solve_and_append(problem, "results/01_obstacle_results.csv")

for n in [32,64]:
    for refinements in [1,2]:
        for degree in [1,2,3]:
            problem = Slate_Jacobi(
                    n=n,
                    alpha0=1e-1,
                    preconditioner="block_cg_bjacobi_chebyshev_jacobi",
                    max_pg_steps=40,
                    pg_rtol=1e-3,
                    alpha_max=30.0,
                    refinements=refinements,
                    degree=degree,
                    snes_atol=1e-5,
                    smoothing_its=5,
                    epsilon=1e-5,
                    save_pvd=False,
                )
            solve_and_append(problem, "results/01_obstacle_slate_results.csv")


            problem = Slate_Star(
                    n=n,
                    alpha0=1e-1,
                    preconditioner="block_cg_bjacobi_star",
                    max_pg_steps=40,
                    pg_rtol=1e-3,
                    alpha_max=30.0,
                    refinements=refinements,
                    degree=degree,
                    snes_atol=1e-5,
                    smoothing_its=5,
                    epsilon=1e-5,
                    save_pvd=False,
                )
            solve_and_append(problem, "results/01_obstacle_slate_results.csv")