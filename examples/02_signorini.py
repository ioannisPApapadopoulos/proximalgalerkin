from firedrake import *
from netgen.occ import *
from proximalgalerkin import *



class SignoriniProblem(ProximalGalerkin):

    def mesh(self):
        maxh = 1.0/self.n
        disk = WorkPlane(Axes((0,0,0), n=Z, h=X)).Circle(1).Face()
        geo = OCCGeometry(disk, dim=2)
        ngmesh = geo.GenerateMesh(maxh=maxh)
        degree = self.degree
        distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.NONE, 1),}
        if degree > 1:
            base_mesh = Mesh(Mesh(ngmesh).curve_field(degree), distribution_parameters=distribution_parameters)
        else:
            base_mesh = Mesh(ngmesh, distribution_parameters=distribution_parameters)
        mh = MeshHierarchy(base_mesh, self.refinements)
        mh_contact = SubmeshHierarchy(mh, subdomain_id="on_boundary")
        mesh = mh[-1]
        self.mesh = mesh
        self.contact_boundary = mh_contact[-1]
        return mesh

    def function_space(self, mesh):
        V = VectorFunctionSpace(mesh, "CG", self.degree)
        W = FunctionSpace(self.contact_boundary, "CG", self.degree)
        return V*W

    def symgrad(self, u):
        return sym(grad(u))

    def sigma(self, u):
        return self.model_parameter*div(u)*self.Id + 2*self.mu*self.symgrad(u)

    def obstacle_v(self, x):
        return as_vector([0, -1+sqrt(1-x**2)])

    def residual(self, z):
        u, psi = split(z)
        v, q = split(TestFunction(z.function_space()))
        alpha = self._alpha
        psi_old = self._psi_old

        mesh = self.mesh
        degree = self.degree
        contact_boundary = self.contact_boundary
        dx_1 = Measure("dx", mesh, intersect_measures=(Measure("dx", mesh),Measure("ds", contact_boundary)),metadata={'quadrature_degree': 4*degree})
        ds_2 = Measure("dx", contact_boundary, intersect_measures=(Measure("ds", mesh),),metadata={'quadrature_degree': 4*degree})

        self.dx_1 = dx_1
        self.ds_2 = ds_2

        n = FacetNormal(mesh)
        n_def = -Constant((0,-1))*sign(n[1])
        self.n_def = n_def
        x, y = SpatialCoordinate(mesh)
        Id = Identity(2)
        self.Id = Id

        mu = Constant(5)
        self.mu = mu
    
        F = inner(alpha*self.sigma(u), self.symgrad(v))*dx_1
        F -= inner(alpha*Constant((0,-1)), v)*dx_1
        F += inner(psi-psi_old, dot(v, n_def))*ds_2
        F += inner(dot(u, n_def) + exp(-psi), q)*ds_2
        F -= inner(dot(self.obstacle_v(x), Constant((0,-1))), q)*ds_2
        return F

    def boundary_conditions(self, Z):
        return None

    def update_alpha(self, alpha):
        return 2 * alpha
    
    def free_body_motion(self, Z, mesh):
        # (Near) nullspaces due to the fact that
        # the disk has not been fixed anywhere
        x, y = SpatialCoordinate(mesh)

        tx = Function(Z)
        tx_u, _ = tx.subfunctions
        tx_u.interpolate(Constant((1.0, 0.0)))

        ty = Function(Z)
        ty_u, _ = ty.subfunctions
        ty_u.interpolate(Constant((0.0, 1.0)))

        rot = Function(Z)
        rot_u, _ = rot.subfunctions
        rot_u.interpolate(as_vector((-y, x)))

        horizontal = VectorSpaceBasis([tx_u])
        horizontal.orthonormalize()

        rigid_like = VectorSpaceBasis([tx_u, ty_u, rot_u])
        rigid_like.orthonormalize()

        return (
            MixedVectorSpaceBasis(Z, [horizontal, Z.sub(1)]),
            MixedVectorSpaceBasis(Z, [rigid_like, Z.sub(1)]),
        )
    
    def nullspace(self, Z):
        exact_nullspace, near_nullspace = self.free_body_motion(Z, self.mesh)
        return (exact_nullspace, exact_nullspace, near_nullspace)

    def save_solutions(self, u, psi):
        out = VTKFile("out/Incompressible_bouncy_ball.pvd")
        Q = TensorFunctionSpace(self.mesh, "DG", self.degree-1)
        stress = Function(Q)
        stress.rename("Stress")
        u.rename("Displacement")
        stress.project(self.sigma(u))
        out.write(u,stress)

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


if __name__ == "__main__":
    problem = SignoriniProblem(n=20, alpha0=1e0, model_parameter=20.0, save_pvd=True)
    problem.pg_solve()

    problem = OperatorPrecon(
        n=20,
        alpha0=1e0,
        model_parameter=20.0,
        preconditioner="block_lu",
        max_pg_steps=40,
        pg_rtol=1e-3,
        alpha_max=1e2,
        refinements=2,
        degree=1,
        epsilon=1e-5,
        save_pvd=True,
    )
    problem.pg_solve()
    # solve_and_append(problem, "02_signorini_results.csv")
