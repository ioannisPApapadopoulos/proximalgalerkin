from firedrake import *
from obstacle_3d import *

class OperatorPrecon(ObstacleProblem):
    def jacobian_p(self, z, z_test, z_trial):
        u, psi = split(z)
        u_trial, psi_trial = split(z_trial)
        v, q = split(z_test)
        J = self.jacobian(z, z_test, z_trial)
        eps = Constant(self.epsilon)
        return J + inner(1.0/(exp(-psi)+eps)*u_trial,v)*dx-inner(eps*psi_trial,q)*dx

    
if __name__ == "__main__":

    for n in [10]:
        for refinements in [1,2,3]:
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
    
    for n in [10]:
        for refinements in [1,2,3]:
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

    for n in [10]:
        for refinements in [1,2,3]:
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