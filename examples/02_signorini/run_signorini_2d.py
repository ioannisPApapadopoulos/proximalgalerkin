from firedrake import *
from signorini_2d import *

lmbda = 1e1
for n in [30]:
    for refinements in [1,2]:
        for degree in [1,2]:
            problem = SignoriniProblem(
                    n=n,
                    alpha0=1e0,
                    alpha_max=1e2,
                    snes_atol=1e-5,
                    model_parameter=lmbda,
                    preconditioner="lu",
                    smoothing_its=5,
                    max_pg_steps=40,
                    pg_rtol=1e-3,
                    refinements=refinements,
                    degree=degree,
                    epsilon=1e-4,
                    save_pvd=False,
                )
            solve_and_append(problem, "results/02_signorini_results.csv")

            problem = OperatorPrecon(
                n=n,
                alpha0=1e0,
                alpha_max=1e2,
                snes_atol=1e-5,
                model_parameter=lmbda,
                preconditioner="block_cg_bjacobi_chebyshev_jacobi",
                smoothing_its=5,
                max_pg_steps=40,
                pg_rtol=1e-3,
                refinements=refinements,
                degree=degree,
                epsilon=1e-4,
                save_pvd=False,
            )
            solve_and_append(problem, "results/02_signorini_results.csv")

            problem = OperatorPrecon(
                n=n,
                alpha0=1e0,
                alpha_max=1e2,
                snes_atol=1e-5,
                model_parameter=lmbda,
                preconditioner="block_cg_bjacobi_star",
                smoothing_its=5,
                max_pg_steps=40,
                pg_rtol=1e-3,
                refinements=refinements,
                degree=degree,
                epsilon=1e-4,
                save_pvd=False,
            )
            solve_and_append(problem, "results/02_signorini_results.csv")

            if degree < 2:
                problem = MTW(
                    n=n,
                    alpha0=1e0,
                    alpha_max=1e2,
                    snes_atol=1e-5,
                    model_parameter=lmbda,
                    preconditioner="block_cg_bjacobi_star",
                    smoothing_its=5,
                    max_pg_steps=40,
                    pg_rtol=1e-3,
                    refinements=refinements,
                    degree=degree,
                    epsilon=1e-4,
                    save_pvd=False,
                )
                solve_and_append(problem, "results/02_signorini_results.csv")

            
            if degree > 1:
                problem = Alfeld(
                    n=n,
                    alpha0=1e0,
                    alpha_max=1e2,
                    snes_atol=1e-5,
                    model_parameter=lmbda,
                    preconditioner="block_cg_bjacobi_star",
                    smoothing_its=5,
                    max_pg_steps=40,
                    pg_rtol=1e-3,
                    refinements=refinements,
                    degree=degree,
                    epsilon=1e-4,
                    save_pvd=False,
                )
                solve_and_append(problem, "results/02_signorini_results.csv")


# Lambda-robustness
n = 30
refinements = 1
for lmbda in [1e1, 1e2, 1e3, 1e4, 1e5]:
    problem = MTW(
        n=n,
        alpha0=1e0,
        alpha_max=1e2,
        snes_atol=1e-5,
        model_parameter=lmbda,
        preconditioner="block_cg_bjacobi_star",
        smoothing_its=5,
        max_pg_steps=40,
        pg_rtol=1e-3,
        refinements=refinements,
        degree=1,
        epsilon=1e-4,
        save_pvd=False,
    )
    solve_and_append(problem, "results/02_signorini_results.csv")

    problem = Alfeld(
        n=n,
        alpha0=1e0,
        alpha_max=1e2,
        snes_atol=1e-5,
        model_parameter=lmbda,
        preconditioner="block_cg_bjacobi_star",
        smoothing_its=5,
        max_pg_steps=40,
        pg_rtol=1e-3,
        refinements=refinements,
        degree=2,
        epsilon=1e-4,
        save_pvd=False,
    )
    solve_and_append(problem, "results/02_signorini_results.csv")


for n in [60]:
    for lmbda in [1e1, 1e2, 1e3, 1e4]:
        problem = OperatorPrecon(
            n=n,
            alpha0=1e0,
            alpha_max=1e2,
            snes_atol=1e-5,
            model_parameter=lmbda,
            preconditioner="block_cg_bjacobi_star",
            smoothing_its=5,
            max_pg_steps=40,
            pg_rtol=1e-3,
            refinements=refinements,
            degree=1,
            epsilon=1e-4,
            save_pvd=False,
        )
        solve_and_append(problem, "results/02_signorini_results.csv")
