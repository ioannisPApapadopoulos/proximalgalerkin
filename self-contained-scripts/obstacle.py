from firedrake import *

"""
This script implements a block preconditioner for the Newton linear systems of proximal Galerkin applied to an obstacle problem.

Let Ω be the unit square. The obstacle problem implemented here seeks to minimize
   ∫_Ω |grad u|^2/2 - f * u dx
subject to u <= 1 almost everywhere.

We invert each Jacobian by applying an outer FGMRES method. We then construct a preconditioner by approximating the inverse of the
Jacobian by a Schur complement factorization. The Schur factorization is done with respect to the D-block 
(the block corresponding to the linearization of the exponential).

This leads to two inverses of D, and one of the Schur complement A + B * D^{-1} * B.T.

However D becomes increasingly singular in later proximal steps. Hence we slightly perturb this block in the preconditioner
by the mass matrix. We then invert this perturbed-D by the conjugate-gradient method preconditioned by a Jacobi iteration.

The Schur complement is approximated by a PDE (analogous to operator preconditioning). We invert this PDE approximation by
geometric multigrid with Chebyshev relaxation + a Jacobi preconditioner.

Only the preconditioner is perturbed. The true Jacobian is left unpurturbed and therefore this is still a true
a true Newton method (rather than a quasi-Newton method).

"""

distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.NONE, 1),}
base = UnitSquareMesh(32, 32, distribution_parameters=distribution_parameters)
mh = MeshHierarchy(base, 1)
mesh = mh[-1]

U = FunctionSpace(mesh, "CG", 1)
P = FunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([U, P])

z = Function(Z)
u,psi = split(z)
z_test = TestFunction(Z)
v,q = split(z_test)

u_old = Function(U)
psi_old = Function(P)

alpha = Constant(1)

f = Constant(20)
phi = Constant(1)

E = 0.5*inner(grad(u), grad(u))*dx - inner(f, u)*dx
F = alpha * derivative(E, z, z_test) 
F += inner(psi-psi_old, v)*dx
F += inner(u + exp(-psi) - phi, q)*dx

z_trial = TrialFunction(Z)

epsilon = Constant(1e-5)

u_trial, psi_trial =  split(z_trial)
J = derivative(F, z, z_trial)

Jp = J + inner(1.0/(exp(-psi)+epsilon)*u_trial,v)*dx - inner(epsilon*psi_trial, q)*dx

atol = 1e-5
sp_mg = {
    "mat_type": "nest",
    "snes_monitor": None,
    "snes_converged_reason": None,
    "snes_atol": atol,
    "ksp_type": "fgmres",
    "ksp_converged_reason": None,
    "ksp_monitor_true_residual": None,
    "ksp_max_it": 50,
    "ksp_atol": atol/10,
    "ksp_rtol": atol/10,
    "pc_use_amat": False,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_factorization_type": "full",
    "pc_fieldsplit_0_fields": "1",
    "pc_fieldsplit_1_fields": "0",
    "fieldsplit_0": {
        "ksp_converged_reason": None,
        "ksp_type": "cg",
        "pc_use_amat": False,
        "pc_type": "jacobi",
    },
    "fieldsplit_1": {
        "ksp_type": "preonly",
        "pc_use_amat": False,
        "pc_type": "mg",
        "pc_mg_type": "full",
        "mg_coarse_mat_type": "aij",
        "mg_coarse_pc_type": "lu",
        "mg_coarse_pc_factor_mat_solver_type": "mumps",
        "mg_coarse_mat_mumps_icntl_14": 1000,
        "mg_levels": {
            "ksp_convergence_test": "skip",
            "ksp_max_it": 5,
            "ksp_type": "chebyshev",
            "pc_type": "jacobi",
        }
    }
}

bcs = DirichletBC(Z.sub(0), 0, "on_boundary")

nvp = NonlinearVariationalProblem(F, z, J=J, Jp=Jp, bcs=bcs)
nvs = NonlinearVariationalSolver(nvp, solver_parameters=sp_mg)

u, psi = z.subfunctions
u.rename("u")

out = VTKFile("out/Obstacle.pvd")

alpha.assign(1e-3)
newton_its = 0
ksp_its = 0

for i in range(40):
    nvs.solve()

    nrm = norm(u-u_old, "H1")

    newton_its += nvs.snes.its
    ksp_its += nvs.snes.getLinearSolveIterations()

    print(f"PG iteration {i+1}, alpha = {float(alpha):.2e}, Cauchy Error = {nrm:.2e}")

    u_old.assign(u)
    psi_old.assign(psi)

    if nrm < 1e-3:
        break
    if float(alpha) < 30:
        alpha.assign(2*alpha)

out.write(u)

print(f"\nProximal Steps: {i+1}, Total Newton iterations: {newton_its}, Avg FGMRES its: {ksp_its/newton_its}")
