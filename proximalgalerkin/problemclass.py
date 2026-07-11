from firedrake import *
from .solver_options import *
from .logging import *
import time

class ProximalGalerkin(object):

    def __init__(self,
                n,
                preconditioner="lu",
                alpha0=1.0,
                max_pg_steps=40,
                pg_rtol=1e-3,
                alpha_max=1e2,
                epsilon=0,
                degree=1,
                refinements=1,
                smoothing_its=2,
                model_parameter=None,
                snes_atol=1e-6,
                save_pvd=False,
                ):
        self.alpha0 = alpha0
        self.preconditioner = preconditioner
        self.max_pg_steps = max_pg_steps
        self.pg_rtol = pg_rtol
        self.alpha_max = alpha_max
        self.n = n
        self.refinements = refinements
        self.degree = degree
        self.epsilon = epsilon
        self.smoothing_its = smoothing_its
        self.model_parameter=model_parameter
        self.snes_atol=snes_atol
        self.save_pvd = save_pvd

    def mesh(self):
        raise NotImplementedError

    def function_space(self, mesh):
        raise NotImplementedError

    def residual(self, z):
        raise NotImplementedError

    def jacobian(self, z, z_test, z_trial):
        return derivative(self.residual(z), z, z_trial)

    def jacobian_p(self, z, z_test, z_trial):
        return None
    
    def boundary_conditions(self, Z):
        return None
    
    def nullspace(self, Z):
        return (None, None, None)
    
    def solver_parameters(self):
        preconditioner = self.preconditioner
        if preconditioner in BLOCK_VARIANTS:
            variant = BLOCK_VARIANTS[preconditioner]
            return block_parameters(variant["top_left"], variant["bottom"], self.smoothing_its, self.snes_atol)
        if preconditioner == "monolithic_vanka":
            return monolithic_vanka_parameters(self.smoothing_its, self.snes_atol)
        if preconditioner == "lu":
            return lu_parameters(self.snes_atol)
        if preconditioner == "block_lu":
            return block_lu_parameters(self.snes_atol)
        raise ValueError(f"Unknown preconditioner {preconditioner!r}")

    def update_alpha(self):
        raise NotImplementedError

    def pg_setup(self):
        mesh = self.mesh()
        Z = self.function_space(mesh)

        z = Function(Z)
        u, psi = split(z)
        z_test = TestFunction(Z)
        z_trial = TrialFunction(Z)

        u_old = Function(Z.sub(0))
        psi_old = Function(Z.sub(1))
        alpha = Constant(self.alpha0)
        self._mesh = mesh
        self._Z = Z
        self._u_old = u_old
        self._psi_old = psi_old
        self._alpha = alpha

        F = self.residual(z)
        J = self.jacobian(z, z_test, z_trial)
        Jp = self.jacobian_p(z, z_test, z_trial)
        bcs = self.boundary_conditions(Z)

        nsp, t_nsp, n_nsp = self.nullspace(Z)

        nvp = NonlinearVariationalProblem(F, z, J=J, Jp=Jp, bcs=bcs)
        sp = self.solver_parameters()
        nvs = NonlinearVariationalSolver(
            nvp, solver_parameters=sp, 
            nullspace=nsp, transpose_nullspace=t_nsp, near_nullspace=n_nsp)

        return nvs, z, u_old, psi_old, alpha

    def save_solutions(self):
        return None

    def pg_solve(self):

        nvs, z, u_old, psi_old, alpha = self.pg_setup()

        u, psi = z.subfunctions

        newton_steps = 0
        outer_iterations = 0
        final_error = float("nan")
        start = time.perf_counter()

        for proximal_step in range(1, self.max_pg_steps + 1):
            nvs.solve()
            final_error = norm(u - u_old, "H1")
            step_newton = nvs.snes.its
            step_outer = nvs.snes.getLinearSolveIterations()

            newton_steps += step_newton
            outer_iterations += step_outer
            info_g(
                f"PG iteration {proximal_step}, alpha = {float(alpha):.2e}, "
                f"Cauchy Error = {final_error:.2e}",
                flush=True,
            )

            u_old.assign(u)
            psi_old.assign(psi)

            if final_error < self.pg_rtol:
                break
            if float(alpha) < self.alpha_max:
                alpha.assign(self.update_alpha(alpha))

        elapsed = time.perf_counter() - start
        info_r(f"PG Steps: {proximal_step}, Newton iterations: {newton_steps}, Avg KSP its: {outer_iterations/newton_steps}\n")
        
        if self.save_pvd:
            self.save_solutions(u, psi)

        return {
            "preconditioner": self.preconditioner,
            "mesh": self.n,
            "refinements": self.refinements,
            "cells_per_side": self.n * 2 ** self.refinements,
            "degree": self.degree,
            "epsilon": self.epsilon,
            "proximal_steps": proximal_step,
            "newton_steps": newton_steps,
            "outer_fgmres_iterations": outer_iterations,
            "avg_outer_fgmres_per_newton": (
                outer_iterations / newton_steps if newton_steps else 0.0
            ),
            "final_cauchy_error": float(final_error),
            "elapsed_seconds": elapsed,
            "converged_pg": bool(final_error < self.pg_rtol),
            "model_parameter": self.model_parameter
        }


