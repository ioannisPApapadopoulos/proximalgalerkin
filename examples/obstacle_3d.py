from firedrake import *
from netgen.occ import *
from proximalgalerkin import *


class ObstacleProblem(ProximalGalerkin):

    def mesh(self):
        distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.VERTEX, self.overlap_no)}
        maxh = 1.0/self.n

        shape = Box(Pnt(0,0,0), Pnt(2,2,2)) - Box(Pnt(1,1,1), Pnt(2,2,2))
        geo = OCCGeometry(shape, dim=3)
        ngmesh = geo.GenerateMesh(maxh=maxh)

        base_mesh = Mesh(ngmesh, distribution_parameters=distribution_parameters)

        (x, y, z) = SpatialCoordinate(base_mesh)
        r_squared = (x - 1)**2 + (y - 1)**2 + (z - 1)**2

        for r in [0.05]:
            should_refine = conditional(lt(r_squared, r), 1, 0)
            DG0 = FunctionSpace(base_mesh, "DG", 0)
            markers = Function(DG0)
            markers.interpolate(should_refine)
            base_mesh = base_mesh.refine_marked_elements(markers)

        mh = MeshHierarchy(base_mesh, self.refinements)
        return mh[-1]

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

    def jacobian(self, z, z_test, z_trial):
        u_trial, psi_trial = split(z_trial)
        v, q = split(z_test)
        eps = Constant(self.epsilon)
        return derivative(self.residual(z), z, z_trial)