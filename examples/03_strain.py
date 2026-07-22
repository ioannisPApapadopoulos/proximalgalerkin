from firedrake import *
from proximalgalerkin import *



class StrainProblem(ProximalGalerkin):

    def mesh(self):
        n = self.n
        distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1),}
        base_mesh = RectangleMesh(10*n, n, 1, 0.1, distribution_parameters=distribution_parameters)
        mh = MeshHierarchy(base_mesh, self.refinements)
        mesh = mh[-1]
        self.mesh = mesh
        return mesh

    def function_space(self, mesh):
        degree = self.degree
        V = VectorFunctionSpace(mesh, "CG", degree)
        W = TensorFunctionSpace(mesh, "DG", degree-1)
        return V*W

    def symgrad(self, u):
        return sym(grad(u))

    def sigma(self, u):
        return self.model_parameter*div(u)*self.Id + 2*self.mu*self.symgrad(u)

    def R(self,phi,psi):
        return phi*psi/sqrt(Constant(1)+inner(psi,psi))

    def residual(self, z):
        u, psi = split(z)
        v, q = split(TestFunction(z.function_space()))
        alpha = self._alpha
        psi_old = self._psi_old

        f = Constant((0,-2e-1))
        phi = Constant(0.01)
        self.phi = phi
        Id = Identity(2)
        self.Id = Id
        mu = Constant(70)
        self.mu = mu

        degree = self.degree

        F = inner(alpha*self.sigma(u), self.symgrad(v))*dx
        F -= inner(alpha*f, v)*dx
        F += inner(psi-psi_old, self.symgrad(v))*dx
        F += inner(self.symgrad(u), q)*dx
        F += inner(- self.R(phi,psi), q)*dx(degree=10*degree)
        return F

    def jacobian(self, z, z_test, z_trial):
        eps = Constant(self.epsilon)
        _, psi_trial = split(z_trial)
        _, q = split(z_test)
        J = derivative(self.residual(z), z, z_trial) - inner(eps*psi_trial, q)*dx
        return J

    def boundary_conditions(self, Z):
        return DirichletBC(Z.sub(0), 0, [1])

    def update_alpha(self, alpha):
        return sqrt(2) * alpha
 
    def save_solutions(self, u, psi):
        out = VTKFile("out/Strain_beam.pvd")
        Q = TensorFunctionSpace(self.mesh, "DG", self.degree-1)
        strain = Function(Q)
        strain.rename("strain")
        strain_obs = Function(Q)
        strain_obs.rename("observed strain")
        u.rename("Displacement")
        strain.project(self.symgrad(u))
        strain_obs.interpolate(self.R(self.phi,psi))
        out.write(u,strain,strain_obs)

class OperatorPrecon(StrainProblem):

    def inverse_dR(self, psi, phi, eps, X, Y):
        s = sqrt(1.0 + inner(psi,psi))
        return s/(eps*s+phi)*(inner(X,Y) + phi/(eps*s**3+phi) * inner(psi, Y) * inner(psi, X))

    def jacobian_p(self, z, z_test, z_trial):
        u, psi = split(z)
        u_trial, psi_trial = split(z_trial)
        v, q = split(z_test)
        J = self.jacobian(z, z_test, z_trial)
        eps = Constant(self.epsilon)
        J = derivative(self.residual(z), z, z_trial)
        Jp = J + self.inverse_dR(psi,self.phi,eps,self.symgrad(u_trial),self.symgrad(v))*dx(degree=10*self.degree) - inner(eps*psi_trial, q)*dx
        return Jp

class LmbdaRobust(OperatorPrecon):
    def function_space(self, mesh):
        degree = self.degree
        V = VectorFunctionSpace(mesh, "CG", degree)
        W = TensorFunctionSpace(mesh, "DG", degree-1)
        return V*W

if __name__ == "__main__":
    problem = StrainProblem(n=10, alpha0=1e-2, model_parameter=20.0, save_pvd=True)
    # problem.pg_solve()

    # problem = OperatorPrecon(
    #     n=20,
    #     alpha0=1e-2,
    #     model_parameter=300.0,
    #     preconditioner="block_lu",
    #     max_pg_steps=40,
    #     refinements=1,
    #     degree=2,
    #     epsilon=1e-4,
    #     save_pvd=True,
    #     snes_atol=1e-7,
    # )

    problem = LmbdaRobust(
        n=10,
        alpha0=1e-2,
        model_parameter=300.0,
        preconditioner="block_cg_jacobi_chebyshev",
        max_pg_steps=40,
        refinements=1,
        degree=4,
        epsilon=1e-4,
        save_pvd=True,
        snes_atol=1e-7,
    )
    problem.pg_solve()
    # solve_and_append(problem, "02_signorini_results.csv")
