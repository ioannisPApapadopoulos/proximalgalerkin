from firedrake import *
from signorini_2d import *

class OperatorPrecon(SignoriniProblem):
    def jacobian_p(self, z, z_test, z_trial):
        u, psi = split(z)
        u_trial, psi_trial = split(z_trial)
        v, q = split(z_test)
        J = self.jacobian(z, z_test, z_trial)
        eps = Constant(self.epsilon)
        ds_2 = self.ds_2
        n_def = self.n_def
        return J + inner(1.0/(exp(-psi)+eps)*dot(u_trial,n_def),dot(v,n_def))*ds_2 - inner(eps*psi_trial, q)*ds_2
    
class MTW(OperatorPrecon):
    def function_space(self, mesh):
        V = FunctionSpace(mesh, "MTW", self.degree)
        W = FunctionSpace(self.contact_boundary, "DG", self.degree-1)
        return V*W

class Alfeld(OperatorPrecon):
    def function_space(self, mesh):
        V = VectorFunctionSpace(mesh, "CG", self.degree, variant="alfeld")
        W = FunctionSpace(self.contact_boundary, "CG", self.degree)
        return V*W
    def transfer_manager(self):
        return CoarsePatchTransferManager()

if __name__ == "__main__":
    # problem = SignoriniProblem(n=20, alpha0=1e0,alpha_max=1e2, model_parameter=20.0, save_pvd=True)
    # problem.pg_solve()

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

    n = 30
    refinements = 1
    for lmbda in [1e2, 1e3, 1e4, 1e5]:
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
        for lmbda in [1e2, 1e3, 1e4]:
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
