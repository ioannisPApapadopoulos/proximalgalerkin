from firedrake import *
from proximalgalerkin import *


class ObstacleProblem(ProximalGalerkin):

    def mesh(self):
        distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
        base_mesh = UnitSquareMesh(self.n, self.n, distribution_parameters=distribution_parameters)
        hierarchy = MeshHierarchy(base_mesh, self.refinements)
        return hierarchy[-1]

    def function_space(self, mesh):
        U = FunctionSpace(mesh, "CG", self.degree)
        P = FunctionSpace(mesh, "CG", self.degree)
        return U * P

    def residual(self, z):
        u, psi = split(z)
        v, q = split(TestFunction(z.function_space()))

        energy = 0.5 * inner(grad(u), grad(u)) * dx - inner(Constant(20.0), u) * dx
        F = self._alpha * derivative(energy, u, v)
        F += inner(psi - self._psi_old, v) * dx
        F += inner(u + exp(-psi) - Constant(1.0), q) * dx
        return F

    def jacobian(self, z, z_trial):
        return derivative(self.residual(z), z, z_trial)

    def jacobian_p(self, z, z_test, z_trial):
        u, psi = split(z)
        u_trial, psi_trial = split(z_trial)
        v, q = split(z_test)
        J = self.jacobian(z, z_trial)
        eps = Constant(self.epsilon)
        return J + inner(u_trial / (exp(-psi) + eps), v) * dx - inner(eps * psi_trial, q) * dx

    def boundary_conditions(self, Z):
        return DirichletBC(Z.sub(0), 0, "on_boundary")

    def update_alpha(self, alpha):
        return 2 * alpha


if __name__ == "__main__":
    problem = ObstacleProblem(
        alpha0=1e-3,
        preconditioner="block_cg_jacobi_chebyshev",
        max_pg_steps=40,
        pg_rtol=1e-3,
        alpha_max=30.0,
        n=32,
        refinements=1,
        degree=1,
        epsilon=1e-5,
        smoothing_its=2,
    )

    solve_and_append(problem, "01_obstacle_results.csv")
