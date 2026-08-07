from firedrake import *
from signorini_2d import *

class SignoriniProblemDG(SignoriniProblem):
    def function_space(self, mesh):
        V = VectorFunctionSpace(mesh, "CG", self.degree)
        W = FunctionSpace(self.contact_boundary, "DG", self.degree-1)
        return V*W

class OperatorPreconDG(SignoriniProblemDG):
    def jacobian_p(self, z, z_test, z_trial):
        u, psi = split(z)
        u_trial, psi_trial = split(z_trial)
        v, q = split(z_test)
        J = self.jacobian(z, z_test, z_trial)
        eps = Constant(self.epsilon)
        ds_2 = self.ds_2
        n_def = self.n_def
        return J + inner(1.0/(exp(-psi)+eps)*dot(u_trial,n_def),dot(v,n_def))*ds_2 - inner(eps*psi_trial, q)*ds_2
    
class O_MTW(OperatorPreconDG):
    def function_space(self, mesh):
        V = FunctionSpace(mesh, "MTW", self.degree)
        W = FunctionSpace(self.contact_boundary, "DG", self.degree-1)
        return V*W

class O_Alfeld(OperatorPreconDG):
    def function_space(self, mesh):
        V = VectorFunctionSpace(mesh, "CG", self.degree, variant="alfeld")
        W = FunctionSpace(self.contact_boundary, "CG", self.degree)
        return V*W
    def transfer_manager(self):
        return CoarsePatchTransferManager()

class Slate(SignoriniProblemDG):
    def jacobian_p(self, z, z_test, z_trial):
        Z = z.function_space()
        P = Z.sub(1)
        u, psi = split(z)
        u_trial, psi_trial = split(z_trial)
        v, q = split(z_test)

        dx_1, ds_2 = self.dx_1, self.ds_2
        n_def = self.n_def

        alpha = self._alpha
        eps = Constant(self.epsilon)
        A = Tensor(inner(alpha*self.sigma(u_trial), self.symgrad(v)) * dx_1)
        D = Tensor(inner(psi_trial * (exp(-psi) + eps), q) * ds_2)
        B = Tensor(inner(psi_trial, dot(v, n_def)) * ds_2)

        x0 = TestFunction(P)
        x1 = TrialFunction(P)
        Daux = Tensor(inner(x1 * (exp(-psi) + eps), x0) * ds_2)
        Baux = Tensor(inner(x1, dot(v,n_def)) * ds_2)
        S = A + Baux * Inverse(Daux) * Baux.T
        Jp = S - D + B + B.T
        return Jp

    def solver_parameters(self):
        sp = {
            "mat_type": "matfree",
            "pmat_type": "aij",
            "snes_monitor": None,
            "snes_converged_reason": None,
            "snes_stol": 0,
            "snes_atol": self.snes_atol,
            "ksp_type": "fgmres",
            "ksp_converged_reason": None,
            "ksp_monitor_true_residual": None,
            "ksp_max_it": 200,
            "ksp_atol": self.snes_atol/1e1,
            "ksp_rtol": self.snes_atol/1e1,
            "pc_use_amat": False,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_factorization_type": "full",
            "pc_fieldsplit_0_fields": "1",
            "pc_fieldsplit_1_fields": "0",
            "fieldsplit_1": {
                "ksp_type": "preonly",
                "pc_use_amat": False,
                "pc_type": "mg",
                "mg_levels_ksp_convergence_test": "skip",
                "mg_levels_ksp_type": "chebyshev",
                "mg_levels_ksp_max_it": self.smoothing_its,
                "mg_levels_pc_python_type": "firedrake.ASMStarPC",
                "mg_levels_pc_star_use_coloring": True,
                "mg_coarse_pc_type": "lu",
                "mg_coarse_pc_factor_mat_solver_type": "mumps",
            },
            "fieldsplit_0": {
                "ksp_type": "cg",
                "pc_use_amat": False,
                "pc_type": "bjacobi",
                "ksp_converged_reason": None,
            },
        }
        return sp

if __name__ == "__main__":
    problem = Slate(
        n=30,
        alpha0=1e0,
        alpha_max=1e2,
        snes_atol=1e-5,
        model_parameter=1e1,
        preconditioner="",
        smoothing_its=5,
        max_pg_steps=40,
        pg_rtol=1e-3,
        refinements=1,
        degree=1,
        epsilon=1e-4,
        save_pvd=False,
    )    
    problem.pg_solve()


    # ads
    # lmbda = 1e1
    # for n in [30]:
    #     for refinements in [1,2]:
    #         for degree in [1,2]:
    #             problem = SignoriniProblem(
    #                     n=n,
    #                     alpha0=1e0,
    #                     alpha_max=1e2,
    #                     snes_atol=1e-5,
    #                     model_parameter=lmbda,
    #                     preconditioner="lu",
    #                     smoothing_its=5,
    #                     max_pg_steps=40,
    #                     pg_rtol=1e-3,
    #                     refinements=refinements,
    #                     degree=degree,
    #                     epsilon=1e-4,
    #                     save_pvd=False,
    #                 )
    #             solve_and_append(problem, "results/02_signorini_results.csv")

    #             problem = OperatorPrecon(
    #                 n=n,
    #                 alpha0=1e0,
    #                 alpha_max=1e2,
    #                 snes_atol=1e-5,
    #                 model_parameter=lmbda,
    #                 preconditioner="block_cg_bjacobi_chebyshev_jacobi",
    #                 smoothing_its=5,
    #                 max_pg_steps=40,
    #                 pg_rtol=1e-3,
    #                 refinements=refinements,
    #                 degree=degree,
    #                 epsilon=1e-4,
    #                 save_pvd=False,
    #             )
    #             solve_and_append(problem, "results/02_signorini_results.csv")

    #             problem = OperatorPrecon(
    #                 n=n,
    #                 alpha0=1e0,
    #                 alpha_max=1e2,
    #                 snes_atol=1e-5,
    #                 model_parameter=lmbda,
    #                 preconditioner="block_cg_bjacobi_star",
    #                 smoothing_its=5,
    #                 max_pg_steps=40,
    #                 pg_rtol=1e-3,
    #                 refinements=refinements,
    #                 degree=degree,
    #                 epsilon=1e-4,
    #                 save_pvd=False,
    #             )
    #             solve_and_append(problem, "results/02_signorini_results.csv")

    #             if degree < 2:
    #                 problem = MTW(
    #                     n=n,
    #                     alpha0=1e0,
    #                     alpha_max=1e2,
    #                     snes_atol=1e-5,
    #                     model_parameter=lmbda,
    #                     preconditioner="block_cg_bjacobi_star",
    #                     smoothing_its=5,
    #                     max_pg_steps=40,
    #                     pg_rtol=1e-3,
    #                     refinements=refinements,
    #                     degree=degree,
    #                     epsilon=1e-4,
    #                     save_pvd=False,
    #                 )
    #                 solve_and_append(problem, "results/02_signorini_results.csv")

                
    #             if degree > 1:
    #                 problem = Alfeld(
    #                     n=n,
    #                     alpha0=1e0,
    #                     alpha_max=1e2,
    #                     snes_atol=1e-5,
    #                     model_parameter=lmbda,
    #                     preconditioner="block_cg_bjacobi_star",
    #                     smoothing_its=5,
    #                     max_pg_steps=40,
    #                     pg_rtol=1e-3,
    #                     refinements=refinements,
    #                     degree=degree,
    #                     epsilon=1e-4,
    #                     save_pvd=False,
    #                 )
    #                 solve_and_append(problem, "results/02_signorini_results.csv")

    # n = 30
    # refinements = 1
    # for lmbda in [1e2, 1e3, 1e4, 1e5]:
    #     problem = MTW(
    #         n=n,
    #         alpha0=1e0,
    #         alpha_max=1e2,
    #         snes_atol=1e-5,
    #         model_parameter=lmbda,
    #         preconditioner="block_cg_bjacobi_star",
    #         smoothing_its=5,
    #         max_pg_steps=40,
    #         pg_rtol=1e-3,
    #         refinements=refinements,
    #         degree=1,
    #         epsilon=1e-4,
    #         save_pvd=False,
    #     )
    #     solve_and_append(problem, "results/02_signorini_results.csv")

    #     problem = Alfeld(
    #         n=n,
    #         alpha0=1e0,
    #         alpha_max=1e2,
    #         snes_atol=1e-5,
    #         model_parameter=lmbda,
    #         preconditioner="block_cg_bjacobi_star",
    #         smoothing_its=5,
    #         max_pg_steps=40,
    #         pg_rtol=1e-3,
    #         refinements=refinements,
    #         degree=2,
    #         epsilon=1e-4,
    #         save_pvd=False,
    #     )
    #     solve_and_append(problem, "results/02_signorini_results.csv")


    # for n in [60]:
    #     for lmbda in [1e2, 1e3, 1e4]:
    #         problem = OperatorPrecon(
    #             n=n,
    #             alpha0=1e0,
    #             alpha_max=1e2,
    #             snes_atol=1e-5,
    #             model_parameter=lmbda,
    #             preconditioner="block_cg_bjacobi_star",
    #             smoothing_its=5,
    #             max_pg_steps=40,
    #             pg_rtol=1e-3,
    #             refinements=refinements,
    #             degree=1,
    #             epsilon=1e-4,
    #             save_pvd=False,
    #         )
    #         solve_and_append(problem, "results/02_signorini_results.csv")
