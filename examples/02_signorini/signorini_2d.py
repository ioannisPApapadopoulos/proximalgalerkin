from firedrake import *
from netgen.occ import *
from proximalgalerkin import *

class SignoriniProblem(ProximalGalerkin):

    def mesh(self):
        maxh = 1.0/self.n
        disk = WorkPlane(Axes((0,0,0), n=Z, h=X)).Circle(1).Face()
        rect = WorkPlane().MoveTo(-0.5, 0.5).Rectangle(1.0, 1.5).Face()
        half_disk = disk - WorkPlane().MoveTo(-2, 0.).Rectangle(4.0, 2).Face()
        shape = Glue([half_disk, disk, rect])
        shape.edges.Max(Y).name="top"
        shape.edges.Min(Y).name="bottom"
        geo = OCCGeometry(shape, dim=2)
        ngmesh = geo.GenerateMesh(maxh=maxh)
        distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.VERTEX, self.overlap_no),}

        base_mesh = Mesh(ngmesh, distribution_parameters=distribution_parameters)
        mh = MeshHierarchy(base_mesh, self.refinements, coarse_facet_label=1000)
        self.contact_label = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name in ["bottom"]]
        mh_contact = SubmeshHierarchy(mh, subdim=1, subdomain_id=self.contact_label)
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
        dx_1 = Measure("dx", mesh, intersect_measures=(Measure("dx", mesh),Measure("ds", contact_boundary)),metadata={'max_quadrature_degree': 4*degree})
        ds_2 = Measure("dx", contact_boundary, intersect_measures=(Measure("ds", mesh),),metadata={'max_quadrature_degree': 4*degree})

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
        F += inner(psi-psi_old, dot(v, n_def))*ds_2
        F += inner(dot(u, n_def) + exp(-psi), q)*ds_2
        F -= inner(dot(self.obstacle_v(x), Constant((0,-1))), q)*ds_2
        return F

    def boundary_conditions(self, Z):
        return DirichletBC(Z.sub(0), Constant((0.,-0.3)), [self.bc_label])

    def update_alpha(self, alpha):
        return 2 * alpha

    def save_solutions(self, u, psi):
        out = VTKFile("out/Signorini_2D.pvd")
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
