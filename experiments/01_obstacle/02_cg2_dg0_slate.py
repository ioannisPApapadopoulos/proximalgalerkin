from firedrake import *
from collections import defaultdict
"""
Block preconditioner for proximal Galerkin applied to the obstacle problem.

Top left block is the -D_ij = -(eta_i exp(-p), eta_j) where eta_i are the
basis functions for p. -D is SPD but becomes singular as the
algorithm progresses. We handle this by approximating it via a shifted D^epsilon

D^epsilon = D + epsilon (eta_i, eta_j) for user-chosen epsilon. This remains invertible.

Schur complement is S = A + B.T D.inv B

B = (eta_i, v_j) is the Gram matrix between u and p
A = alpha*(grad(v_i), grad(v_j)) is a scaled Laplacian.

We approximate S with the assembled shifted D via slate

 S^epsilon =  A + B.T D^epsilon.inv B

We invert S^epsilon with geometric MG with vertex-star patch relaxation.


CG2-DG0 discretization for (u,p)
"""
nx = 10
distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
base = UnitSquareMesh(nx, nx, distribution_parameters=distribution_parameters)
mh = MeshHierarchy(base, 1)
mesh = mh[-1]

V = FunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "DG", 0)
Z = V * Q

z = Function(Z)
u, p = split(z)
z_test = TestFunction(Z)
v, q = split(z_test)
z_trial = TrialFunction(Z)
u_trial, p_trial = split(z_trial)

u_old = Function(V)
p_old = Function(Q)

alpha = Constant(1)

f = Constant(20)
phi = Constant(1)

E = 0.5*inner(grad(u), grad(u))*dx - inner(f, u)*dx

F = alpha * derivative(E, u, v)
F += inner(p - p_old, v)*dx
F += inner(u + exp(-p) - phi, q)*dx

epsilon = Constant(1e-10)


v0, q0 = TestFunctions(Z)
v1, q1 = TrialFunctions(Z)
A = Tensor(inner(alpha*grad(v1), grad(v0)) * dx)
D = Tensor(inner(q1 * (exp(-p) + epsilon), q0) * dx)
B = Tensor(inner(q1, v0) * dx)

x0 = TestFunction(Q)
x1 = TrialFunction(Q)
Daux = Tensor(inner(x1 * (exp(-p) + epsilon), x0) * dx)
Baux = Tensor(inner(x1, v0) * dx)
S = A + Baux * Inverse(Daux) * Baux.T

Jp = S - D + B + B.T

# assemble(Jp).petscmat.view()

sp_krylov = {
    "mat_type": "matfree",
    "pmat_type": "aij",
    "snes_monitor": None,
    "snes_converged_reason": None,
    "snes_stol": 0,
    "snes_atol": 1e-6,
    "ksp_type": "gmres",
    "ksp_converged_reason": None,
    "ksp_monitor_true_residual": None,
    "ksp_max_it": 200,
    "ksp_atol": 1e-7,
    "ksp_rtol": 1e-7,
    "pc_use_amat": False,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_factorization_type": "full",
    # "pc_fieldsplit_schur_precondition": "selfp",
    "pc_fieldsplit_0_fields": "1",
    "pc_fieldsplit_1_fields": "0",
    "fieldsplit_ksp_type": "preonly",
    "fieldsplit_1": {
        # "pc_use_amat": False,
        # "pc_type": "python",
        # "pc_python_type": "firedrake.AssembledPC",
        # "assembled" : {
            "pc_use_amat": False,
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    # },
    "fieldsplit_0": {
        "ksp_converged_reason": None,
        "pc_use_amat": False,
        # "ksp_type": "cg",
        "pc_type": "jacobi",
    },
}


sp_mg = {
    "mat_type": "matfree",
    "pmat_type": "aij",
    "snes_monitor": None,
    "snes_converged_reason": None,
    "snes_stol": 0,
    "snes_atol": 1e-6,
    "ksp_type": "fgmres",
    "ksp_converged_reason": None,
    "ksp_monitor_true_residual": None,
    "ksp_max_it": 200,
    "ksp_atol": 1e-7,
    "ksp_rtol": 1e-7,
    "pc_use_amat": False,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_factorization_type": "full",
    # "pc_fieldsplit_schur_precondition": "selfp",
    "pc_fieldsplit_0_fields": "1",
    "pc_fieldsplit_1_fields": "0",
    "fieldsplit_ksp_type": "preonly",
    "fieldsplit_1": {
        "pc_use_amat": False,
        "pc_type": "mg",
        "pc_mg_type": "full",
        "mg_coarse_mat_type": "aij",
        "mg_coarse_pc_type": "lu",
        # "mg_coarse_pc_use_amat": False,
        "mg_coarse_pc_factor_mat_solver_type": "mumps",
        "mg_coarse_mat_mumps_icntl_14": 1000,
        "mg_levels": {
            "ksp_convergence_test": "skip",
            "ksp_max_it": 5,
            "ksp_type": "chebyshev",
            "pc_type": "jacobi",
            # "pc_use_amat": False,
        },
    },
    "fieldsplit_0": {
        "ksp_converged_reason": None,
        "pc_use_amat": False,
        # "ksp_type": "cg",
        "pc_type": "jacobi",
    },
}

bcs = DirichletBC(Z.sub(0), 0, "on_boundary")

nvp = NonlinearVariationalProblem(F, z, Jp=Jp, bcs=bcs)
nvs = NonlinearVariationalSolver(nvp, solver_parameters=sp_mg, pre_apply_bcs=False)

u, p = z.subfunctions
u.rename("u")

out = VTKFile("out/obstacle_pg.pvd")

alpha.assign(1e-3)
history = defaultdict(list)
history["newton_its"] = 0
history["ksp_its"] = 0
history["max_ksp_its"] = 0

for i in range(40):
    nvs.solve()

    nrm = norm(u-u_old, "H1")

    history["newton_its"] += nvs.snes.its
    history["ksp_its"] += nvs.snes.getLinearSolveIterations()
    history["max_ksp_its"] = max(history["max_ksp_its"], nvs.snes.getLinearSolveIterations()/nvs.snes.its)

    print(f"PG iteration {i+1}, alpha = {float(alpha):.2e}, Cauchy Error = {nrm:.2e}")

    u_old.assign(u)
    p_old.assign(p)

    if nrm < 1e-3:
        break
    if float(alpha) < 30:
        alpha.assign(2*alpha)

out.write(u)

print(f"\nPG Steps: {i+1}, Newton iterations: {history["newton_its"]}, Avg KSP its: {history["ksp_its"]/history["newton_its"]}, Max KSP its: {history["max_ksp_its"]}")
