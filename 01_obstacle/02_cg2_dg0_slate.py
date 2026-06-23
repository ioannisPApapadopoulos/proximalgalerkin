from firedrake import *
from collections import defaultdict
"""
Block preconditioner for proximal Galerkin applied to the obstacle problem.

Top left block is the -D_ij = -(eta_i exp(-psi), eta_j) where eta_i are the
basis functions for psi. -D is SPD but becomes singular as the 
algorithm progresses. We handle this by approximating it via a shifted D^gamma

D^gamma = D + gamma (eta_i, eta_j) for user-chosen gamma. This remains invertible.

Schur complement is S = A + B.T D.inv B

B = (eta_i, v_j) is the Gram matrix between u and psi
A = alpha*(grad(v_i), grad(v_j)) is a scaled Laplacian.

We approximate S with the assembled shifted D via slate

 S^gamma =  A + B.T D^gamma.inv B

We invert S^gamma with geometric MG with vertex-star patch relaxation.


CG2-DG0 discretization for (u,psi)
"""
distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
base = UnitSquareMesh(16, 16, distribution_parameters=distribution_parameters)
mh = MeshHierarchy(base, 1)
mesh = mh[-1]

U = FunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "DG", 0)
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

gamma = Constant(1e-5)


u_trial, psi_trial =  split(z_trial)
J = derivative(F, z, z_trial)


A = Tensor(inner(alpha*grad(u_trial), grad(v))*dx)
B = Tensor(inner(u_trial, q)*dx)
D = Tensor(inner(exp(-psi)*psi_trial + gamma*psi_trial, q)*dx)

Jp = A + B.T * Inverse(D) * B - D


sp_krylov = {
    "mat_type": "aij",
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
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_factorization_type": "full",
    # "pc_fieldsplit_schur_precondition": "selfp",
    "pc_fieldsplit_0_fields": "1",
    "pc_fieldsplit_1_fields": "0",
    "fieldsplit_ksp_type": "preonly",
    "fieldsplit_0": {
        "ksp_type": "preonly",
        # "pc_use_amat": False,
        # "pc_type": "python",
        # "pc_python_type": "firedrake.AssembledPC",
        # "assembled" : {
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        # }
    },
    "fieldsplit_1": {
        "ksp_type": "preonly",
        # "pc_use_amat": False,
        # "pc_type": "python",
        # "pc_python_type": "firedrake.AssembledPC",
        # "assembled" : {
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        # }
    },
}

bcs = DirichletBC(Z.sub(0), 0, "on_boundary")

nvp = NonlinearVariationalProblem(F, z, J=J, Jp=Jp, bcs=bcs)
nvs = NonlinearVariationalSolver(nvp, solver_parameters=sp_krylov)

u, psi = z.subfunctions
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
    psi_old.assign(psi)

    if nrm < 1e-3:
        break
    if float(alpha) < 30:
        alpha.assign(2*alpha)

out.write(u)

print(f"\nPG Steps: {i}, Newton iterations: {history["newton_its"]}, Avg KSP its: {history["ksp_its"]/history["newton_its"]}, Max KSP its: {history["max_ksp_its"]}")