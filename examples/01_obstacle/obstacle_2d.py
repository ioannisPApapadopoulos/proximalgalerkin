from firedrake import *
from netgen.occ import *
from proximalgalerkin import *


class ObstacleProblem(ProximalGalerkin):

    def mesh(self):
        distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.VERTEX, self.overlap_no)}
        maxh = 1.0/self.n
        wp = WorkPlane()
        L_shape = wp.Rectangle(2,2).Face() - wp.Rectangle(1,1).Face().Move((1,1,0))
        geo = OCCGeometry(L_shape, dim=2)
        ngmesh = geo.GenerateMesh(maxh=maxh)

        base_mesh = Mesh(ngmesh, distribution_parameters=distribution_parameters)
        (x, y) = SpatialCoordinate(base_mesh)
        r_squared = (x - 1)**2 + (y - 1)**2

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
        return sqrt(2) * alpha
    
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
