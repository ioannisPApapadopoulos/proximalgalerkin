from firedrake import *
from obstacle_3d import *

"""
Currently needs Firedrake branch: pbrubeck/slate-fieldsplit
Run make in firedrake after git pull

Then:
python -m pip uninstall -y petsctools
python -m pip install --upgrade   git+https://github.com/firedrakeproject/petsctools.git@main

python -m pip uninstall -y firedrake-rtree
python -m pip install --upgrade   git+https://github.com/firedrakeproject/firedrake-rtree.git@main

git checkout -b jp/slate-fieldsplit
git merge main


"""
class Slate_Jacobi(ObstacleProblem):

    def jacobian_p(self, z, z_test, z_trial):
        Z = z.function_space()
        P = Z.sub(1)
        u, psi = split(z)
        u_trial, psi_trial = split(z_trial)
        v, q = split(z_test)

        alpha = self._alpha
        eps = Constant(self.epsilon)
        A = Tensor(inner(alpha*grad(u_trial), grad(v)) * dx)
        D = Tensor(inner(psi_trial * (exp(-psi) + eps), q) * dx)
        B = Tensor(inner(psi_trial, v) * dx)

        x0 = TestFunction(P)
        x1 = TrialFunction(P)
        Daux = Tensor(inner(x1 * (exp(-psi) + eps), x0) * dx)
        Baux = Tensor(inner(x1, v) * dx)
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
                "mg_levels_pc_type": "jacobi",
            },
            "fieldsplit_0": {
                "ksp_type": "cg",
                "pc_use_amat": False,
                "pc_type": "bjacobi",
                "ksp_converged_reason": None,
            },
        }
        return sp

class Slate_Star(Slate_Jacobi):
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
    for n in [20]:
        for refinements in [1,2]:
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