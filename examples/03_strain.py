from firedrake import *
from proximalgalerkin import *



class StrainProblem(ProximalGalerkin):

    def mesh(self):
        n = self.n
        distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.VERTEX, self.overlap_no),}
        base_mesh = RectangleMesh(10*n, n, 1, 0.1, distribution_parameters=distribution_parameters)
        mh = MeshHierarchy(base_mesh, self.refinements, coarse_facet_label=1000)
        mesh = mh[-1]
        self.mesh = mesh
        return mesh

    def function_space(self, mesh):
        degree = self.degree
        V = VectorFunctionSpace(mesh, "CG", degree)
        W = TensorFunctionSpace(mesh, "DG", degree-1, symmetry=True)
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

        f = Constant((0,-1e1))
        phi = Constant(0.4) # 0.4
        self.phi = phi
        Id = Identity(2)
        self.Id = Id
        mu = Constant(70)
        self.mu = mu

        degree = self.degree
        self.scale = Constant(1e3)

        F = inner(alpha*self.sigma(u), self.symgrad(v))*dx
        F -= inner(alpha*f, v)*dx
        F += inner(psi-psi_old, self.symgrad(v))*dx
        F += self.scale*inner(self.symgrad(u)- self.R(phi,psi), q)*dx(degree=2*degree)
        return F

    def jacobian(self, z, z_test, z_trial):
        eps = Constant(self.epsilon)
        _, psi_trial = split(z_trial)
        _, q = split(z_test)
        J = derivative(self.residual(z), z, z_trial) #- inner(self.scale*eps*psi_trial, q)*dx
        return J

    def boundary_conditions(self, Z):
        return [DirichletBC(Z.sub(0), 0, [1]),
                DirichletBC(Z.sub(0), Constant((-0.2,0)), [2])]

    def update_alpha(self, alpha):
        return sqrt(2) * alpha
 
    def save_solutions(self, u, psi):
        out = VTKFile("out/Strain_2d.pvd")
        Q = TensorFunctionSpace(self.mesh, "CG", self.degree)
        strain = Function(Q)
        strain.rename("Strain")
        strain_obs = Function(Q)
        strain_obs.rename("Observed Strain")
        u.rename("Displacement")
        strain.interpolate(self.symgrad(u))
        strain_obs.interpolate(self.R(self.phi,psi))
        out.write(u,strain,strain_obs)
        # mesh = RectangleMesh(1,1,0.05,0.5)
        # U = VectorFunctionSpace(mesh, "CG", 1)
        # v = Function(U)
        # v.assign(Constant((-0.05,-0.3)))
        # VTKFile("out/Strain_2D_wall.pvd").write(v)

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
        Jp = J + self.inverse_dR(psi,self.phi,eps,self.symgrad(u_trial),self.symgrad(v))*dx(degree=2*self.degree) - inner(self.scale*eps*psi_trial, q)*dx
        return Jp

class MTW(OperatorPrecon):
    def function_space(self, mesh):
        V = FunctionSpace(mesh, "MTW", self.degree)
        W = TensorFunctionSpace(mesh, "DG", self.degree-1, symmetry=True)
        return V*W

class Alfeld(OperatorPrecon):
    def function_space(self, mesh):
        V = VectorFunctionSpace(mesh, "CG", self.degree, variant="alfeld")
        W = TensorFunctionSpace(mesh, "DG", self.degree, variant="alfeld", symmetry=True)
        return V*W
    def transfer_manager(self):
        return CoarsePatchTransferManager()


class Slate(StrainProblem):

    def jacobian_p(self, z, z_test, z_trial):
        Z = z.function_space()
        P = Z.sub(1)
        u, psi = z.subfunctions
        u_trial, psi_trial = split(z_trial)
        v, q = split(z_test)

        alpha = self._alpha
        eps = Constant(self.epsilon)

        phi = self.phi

        degree = self.degree
        scale = self.scale

        A = Tensor(inner(alpha*self.sigma(u_trial), self.symgrad(v))*dx)
        D = Tensor(derivative(scale*inner(-self.R(phi,psi), q)*dx(degree=2*degree), psi, psi_trial)) - scale*inner(eps*psi_trial, q)*dx
        B = Tensor(inner(psi_trial, self.symgrad(v)) * dx)
        Bt = Tensor(scale*inner(q, self.symgrad(u_trial)) * dx)

        x0 = TestFunction(P)
        x1 = TrialFunction(P)
        Daux = Tensor(derivative(inner(-self.R(phi,psi), x0)*dx(degree=2*degree), psi, x1)) - inner(eps*x1, x0)*dx
        Baux = Tensor(inner(x1, self.symgrad(v)) * dx)
        S = A - Baux * Inverse(Daux) * Baux.T
        Jp = S + D + B + Bt

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
                "mg_levels_ksp_type": "gmres",
                "mg_levels_ksp_max_it": self.smoothing_its,
                "mg_levels_pc_type": "bjacobi",
                # "mg_levels_pc_type": "python",
                # "mg_levels_pc_python_type": "firedrake.ASMStarPC",
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
    problem = StrainProblem(n=20, alpha0=1e-4, refinements=1, model_parameter=1e2, snes_atol=1e-6, pg_rtol=1e-4, save_pvd=True)
    #problem.pg_solve()


    problem = OperatorPrecon(
        n=20,
        alpha0=1e-2,
        alpha_max=1e1,
        model_parameter=1e2,
        preconditioner="block_cg_bjacobi_gmres_bjacobi",
        smoothing_its=5,
        max_pg_steps=40,
        refinements=1,
        degree=1,
        epsilon=1e-5,
        save_pvd=True,
        snes_atol=1e-6,
        pg_rtol=1e-4,
    )
    problem.pg_solve()

    problem = Slate(
        n=20,
        alpha0=1e-2,
        alpha_max=1e1,
        model_parameter=1e2,
        preconditioner="",
        smoothing_its=5,
        max_pg_steps=40,
        refinements=1,
        degree=1,
        epsilon=1e-5,
        save_pvd=True,
        snes_atol=1e-6,
        pg_rtol=1e-4,
    )
    # problem.pg_solve()


    # lmbda = 1e1
    # for n in [30]:
    #     for refinements in [1,2]:
    #         for degree in [1,2]:
    #             problem = StrainProblem(
    #                     n=n,
    #                     alpha0=1e-2,
    #                     alpha_max=1e1,
    #                     snes_atol=1e-6,
    #                     model_parameter=lmbda,
    #                     preconditioner="lu",
    #                     smoothing_its=5,
    #                     max_pg_steps=40,
    #                     pg_rtol=1e-4,
    #                     refinements=refinements,
    #                     degree=degree,
    #                     epsilon=1e-3,
    #                     save_pvd=False,
    #                 )
    #             solve_and_append(problem, "results/03_strain_results.csv")

    #             problem = OperatorPrecon(
    #                 n=n,
    #                 alpha0=1e-2,
    #                 alpha_max=1e1,
    #                 snes_atol=1e-6,
    #                 model_parameter=lmbda,
    #                 preconditioner="block_cg_bjacobi_chebyshev_jacobi",
    #                 smoothing_its=5,
    #                 max_pg_steps=40,
    #                 pg_rtol=1e-4,
    #                 refinements=refinements,
    #                 degree=degree,
    #                 epsilon=1e-3,
    #                 save_pvd=False,
    #             )
    #             solve_and_append(problem, "results/03_strain_results.csv")

    #             problem = OperatorPrecon(
    #                 n=n,
    #                 alpha0=1e-2,
    #                 alpha_max=1e1,
    #                 snes_atol=1e-6,
    #                 model_parameter=lmbda,
    #                 preconditioner="block_cg_bjacobi_star",
    #                 smoothing_its=5,
    #                 max_pg_steps=40,
    #                 pg_rtol=1e-4,
    #                 refinements=refinements,
    #                 degree=degree,
    #                 epsilon=1e-3,
    #                 save_pvd=False,
    #             )
    #             solve_and_append(problem, "results/03_strain_results.csv")

    #             if degree < 2:
    #                 problem = MTW(
    #                     n=n,
    #                     alpha0=1e-2,
    #                     alpha_max=1e1,
    #                     snes_atol=1e-6,
    #                     model_parameter=lmbda,
    #                     preconditioner="block_cg_bjacobi_star",
    #                     smoothing_its=5,
    #                     max_pg_steps=40,
    #                     pg_rtol=1e-4,
    #                     refinements=refinements,
    #                     degree=degree,
    #                     epsilon=1e-3,
    #                     save_pvd=False,
    #                 )
    #                 solve_and_append(problem, "results/03_strain_results.csv")

                
    #             if degree > 1:
    #                 problem = Alfeld(
    #                     n=n,
    #                     alpha0=1e-2,
    #                     alpha_max=1e1,
    #                     snes_atol=1e-6,
    #                     model_parameter=lmbda,
    #                     preconditioner="block_cg_bjacobi_star",
    #                     smoothing_its=5,
    #                     max_pg_steps=40,
    #                     pg_rtol=1e-4,
    #                     refinements=refinements,
    #                     degree=degree,
    #                     epsilon=1e-3,
    #                     save_pvd=False,
    #                 )
    #                 solve_and_append(problem, "results/03_strain_results.csv")

    n = 15
    refinements = 1
    for lmbda in [1e2]: #, 1e3, 1e4, 1e5
        problem = MTW(
            n=n,
            alpha0=1e-2,
            alpha_max=1e1,
            snes_atol=1e-6,
            model_parameter=lmbda,
            preconditioner="block_cg_bjacobi_star",
            smoothing_its=5,
            max_pg_steps=40,
            pg_rtol=1e-4,
            refinements=refinements,
            degree=1,
            epsilon=1e-7,
            save_pvd=False,
        )
        # solve_and_append(problem, "results/03_strain_results.csv")
        # problem.pg_solve()

        problem = Alfeld(
            n=n,
            alpha0=1e-2,
            alpha_max=1e1,
            snes_atol=1e-6,
            model_parameter=lmbda,
            preconditioner="block_cg_bjacobi_star",
            smoothing_its=5,
            max_pg_steps=40,
            pg_rtol=1e-4,
            refinements=refinements,
            degree=2,
            epsilon=1e-7,
            save_pvd=True,
        )
        # solve_and_append(problem, "results/03_strain_results.csv")
        # problem.pg_solve()


    # for n in [30]:
    #     for lmbda in [1e2, 1e3, 1e4]:
    #         problem = OperatorPrecon(
    #             n=n,
    #             alpha0=1e-2,
    #             alpha_max=1e1,
    #             snes_atol=1e-6,
    #             model_parameter=lmbda,
    #             preconditioner="block_cg_bjacobi_star",
    #             smoothing_its=5,
    #             max_pg_steps=40,
    #             pg_rtol=1e-4,
    #             refinements=refinements,
    #             degree=1,
    #             epsilon=1e-3,
    #             save_pvd=False,
    #         )
    #         solve_and_append(problem, "results/03_strain_results.csv")


