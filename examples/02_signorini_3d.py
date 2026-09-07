from firedrake import *
from netgen.occ import *
from proximalgalerkin import *



class SignoriniProblem(ProximalGalerkin):

    def mesh(self):
        maxh = 1.0/self.n
        sphere = Sphere(Pnt(0,0,0), 1.0)
        box = Box(Pnt(-0.5,-0.5,0.5), Pnt(0.5,0.5,2.0))
        half_sphere = sphere - Box(Pnt(-2,-2,0),Pnt(2,2,4))
        shape = Glue([half_sphere,sphere, box])
        shape.faces.Max(Z).name="top"
        shape.faces.Min(Z).name="bottom"
        geo = OCCGeometry(shape, dim=3)
        ngmesh = geo.GenerateMesh(maxh=maxh)
        distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1),}

        base_mesh = Mesh(ngmesh, distribution_parameters=distribution_parameters)
        mh = MeshHierarchy(base_mesh, self.refinements, coarse_facet_label=1000)
        self.contact_label = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name in ["bottom"]]
        mh_contact = SubmeshHierarchy(mh, subdim=2, subdomain_id=self.contact_label)
        mesh = mh[-1]
        self.bc_label = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name in ["top"]]
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

    def obstacle_v(self, x, y):
        return as_vector([0, 0, -1+sqrt(1-(x**2+y**2))])

    def residual(self, z):
        u, psi = split(z)
        v, q = split(TestFunction(z.function_space()))
        alpha = self._alpha
        psi_old = self._psi_old

        mesh = self.mesh
        degree = self.degree
        contact_boundary = self.contact_boundary
        dx_1 = Measure("dx", mesh, intersect_measures=(Measure("dx", mesh),Measure("ds", contact_boundary)),metadata={'max_quadrature_degree': 4*degree})
        ds_2 = Measure("dx", contact_boundary, intersect_measures=(Measure("ds", mesh),),metadata={'max_quadrature_degree': 4*degree})

        self.dx_1 = dx_1
        self.ds_2 = ds_2

        n = FacetNormal(mesh)
        n_def = -Constant((0,0,-1))*sign(n[2])
        self.n_def = n_def
        x1,x2,x3 = SpatialCoordinate(mesh)
        Id = Identity(3)
        self.Id = Id

        mu = Constant(2e2)
        self.mu = mu
    
        F = inner(alpha*self.sigma(u), self.symgrad(v))*dx_1
        F -= inner(alpha*Constant((0,0,-40)), v)*dx_1
        F += inner(psi-psi_old, dot(v, n_def))*ds_2
        F += inner(dot(u, n_def) + exp(-psi), q)*ds_2
        F -= inner(dot(self.obstacle_v(x1,x2), Constant((0,0,-1))), q)*ds_2
        return F

    def boundary_conditions(self, Z):
        return DirichletBC(Z.sub(0), Constant((0,0,-0.3)), [self.bc_label])

    def update_alpha(self, alpha):
        return sqrt(2)  * alpha

    def save_solutions(self, u, psi):
        out = VTKFile("out/Signorini_3D.pvd")
        Q = TensorFunctionSpace(self.mesh, "DG", self.degree-1)
        stress = Function(Q)
        stress.rename("Stress")
        u.rename("Displacement")
        stress.project(self.sigma(u))
        out.write(u,stress)

    def jacobian(self, z, z_test, z_trial):
        u_trial, psi_trial = split(z_trial)
        v, q = split(z_test)
        eps = Constant(self.epsilon)
        ds_2 = self.ds_2
        return derivative(self.residual(z), z, z_trial) #- inner(eps*psi_trial, q)*ds_2

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

    n = 5
    for refinements in [1,2]:
        for lmbda in [1e1, 1e2, 1e3, 1e4, 1e5]:
            problem = MTW(
                n=n,
                alpha0=1e-1,
                alpha_max=1e1,
                snes_atol=1e-5,
                model_parameter=lmbda,
                preconditioner="block_cg_bjacobi_cg_star",
                smoothing_its=8,
                max_pg_steps=40,
                pg_rtol=1e-3,
                refinements=refinements,
                degree=1,
                epsilon=1e-4,
                save_pvd=False,
            )
            solve_and_append(problem, "results/02_signorini_3d_results.csv")

            problem = Alfeld(
                n=n,
                alpha0=1e-1,
                alpha_max=1e1,
                snes_atol=1e-5,
                model_parameter=lmbda,
                preconditioner="block_cg_bjacobi_cg_star",
                smoothing_its=8,
                max_pg_steps=40,
                pg_rtol=1e-3,
                refinements=refinements,
                degree=3,
                epsilon=1e-4,
                save_pvd=False,
            )
            solve_and_append(problem, "results/02_signorini_3d_results.csv")

    n = 20
    for refinements in [2]:
        for lmbda in [1e1,1e2,1e3,1e4,1e4,1e5]:
            problem = OperatorPrecon(
                n=n,
                alpha0=1e-1,
                alpha_max=1e1,
                snes_atol=1e-5,
                model_parameter=lmbda,
                preconditioner="block_cg_bjacobi_cg_star",
                smoothing_its=8,
                max_pg_steps=40,
                pg_rtol=1e-3,
                refinements=refinements,
                degree=1,
                epsilon=1e-4,
                save_pvd=False,
            )
            solve_and_append(problem, "results/02_signorini_3d_results.csv")
