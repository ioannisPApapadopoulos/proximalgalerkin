from firedrake import *
from netgen.occ import *
from proximalgalerkin import *


class ObstacleProblem(ProximalGalerkin):

    def mesh(self):
        distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
        maxh = 1.0/self.n
        wp = WorkPlane()
        L_shape = wp.Rectangle(2,2).Face() - wp.Rectangle(1,1).Face().Move((1,1,0))
        geo = OCCGeometry(L_shape, dim=2)
        ngmesh = geo.GenerateMesh(maxh=maxh)

        base_mesh = Mesh(ngmesh, distribution_parameters=distribution_parameters)
        (x, y) = SpatialCoordinate(base_mesh)
        r_squared = (x - 1)**2 + (y - 1)**2

        for r in [0.1,0.05]:
            should_refine = conditional(lt(r_squared, r), 1, 0)
            DG0 = FunctionSpace(base_mesh, "DG", 0)
            markers = Function(DG0)
            markers.interpolate(should_refine)
            base_mesh = base_mesh.refine_marked_elements(markers)

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

    def boundary_conditions(self, Z):
        return DirichletBC(Z.sub(0), 0, "on_boundary")

    def update_alpha(self, alpha):
        return 2 * alpha
    
    def save_solutions(self, u, psi):
        out = VTKFile("out/obstacle_L_shape_pg.pvd")
        u.rename("u")
        out.write(u)


class OperatorPrecon(ObstacleProblem):
    def jacobian_p(self, z, z_test, z_trial):
        u, psi = split(z)
        u_trial, psi_trial = split(z_trial)
        v, q = split(z_test)
        J = self.jacobian(z, z_test, z_trial)
        eps = Constant(self.epsilon)
        return J + inner(1.0/(exp(-psi)+eps)*u_trial,v)*dx-inner(eps*psi_trial,q)*dx

if __name__ == "__main__":
    problem = ObstacleProblem(n=32, alpha0=1e-3)
    # problem.pg_solve()

    problem = OperatorPrecon(
        n=10,
        alpha0=1e-3,
        preconditioner="block_cg_jacobi_chebyshev",
        max_pg_steps=40,
        pg_rtol=1e-3,
        alpha_max=30.0,
        refinements=1,
        degree=1,
        epsilon=1e-5,
        snes_atol=1e-5,
        save_pvd=True,
    )
    problem.pg_solve()
    # solve_and_append(problem, "01_obstacle_results.csv")
