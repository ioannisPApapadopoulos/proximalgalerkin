from firedrake import *

"""
This script implements a block preconditioner for the Newton linear systems of proximal Galerkin applied 
to a strain-constrained elasticity problem.

Let Ω be the the rectangle (0, 1)x(0, 0.1). The strain-constrained elasticity problem implemented here seeks to minimize
   ∫_Ω μ |symgrad u|^2 + λ |div u|^2/2 - f u dx
subject to |symgrad u|(x) <= 0.4 almost everywhere. The right-most edge is compressed 0.2 to the left and the
left-most edge is fixed in place.

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
n=10
base_mesh = RectangleMesh(10*n, n, 1, 0.1, distribution_parameters=distribution_parameters)
mh = MeshHierarchy(base_mesh, 1)
mesh = mh[-1]


V = VectorFunctionSpace(mesh, "CG", 1)
W = TensorFunctionSpace(mesh, "DG", 0, symmetry=True)
Z = V*W


z = Function(Z)
u, psi = split(z)
psi_old = Function(W)
u_old = Function(V)
z_trial = TrialFunction(Z)
v, q = split(TestFunction(Z))


Id = Identity(2)

mu = Constant(70)
lmbda = Constant(1e2)

def symgrad(u):
    return sym(grad(u))
def sigma(u):
    return lmbda*div(u)*Id + 2*mu*symgrad(u)
def R(phi,psi):
    return phi*psi/sqrt(Constant(1)+inner(psi,psi))

alpha = Constant(1e1)

f = Constant((0,-1e1))
phi = Constant(0.4)
Id = Identity(2)


gamma = Constant(1e2)

F = inner(alpha*sigma(u), symgrad(v))*dx
F -= inner(alpha*f, v)*dx
F += inner(psi-psi_old, symgrad(v))*dx
F += gamma*inner(symgrad(u)- R(phi,psi), q)*dx(degree=2)

bcs = [DirichletBC(Z.sub(0), 0, [1]), DirichletBC(Z.sub(0), Constant((-0.2,0)), [2])]


def inverse_dR(psi, phi, eps, X, Y):
    s = sqrt(1.0 + inner(psi,psi))
    return s/(eps*s+phi)*(inner(X,Y) + phi/(eps*s**3+phi) * inner(psi, Y) * inner(psi, X))

z_trial = TrialFunction(Z)
eps = Constant(1e-3)
u_trial, psi_trial =  split(z_trial)
J = derivative(F, z, z_trial)
Jp = J + inverse_dR(psi,phi,eps,symgrad(u_trial),symgrad(v))*dx(degree=2) - inner(gamma*eps*psi_trial, q)*dx


sp_mg = {
    "mat_type": "nest",
    "snes_monitor": None,
    "snes_converged_reason": None,
    "snes_stol": 0,
    "snes_atol": 1e-5,
    "ksp_type": "fgmres",
    "ksp_converged_reason": None,
    "ksp_monitor_true_residual": None,
    "ksp_max_it": 200,
    "ksp_atol": 1e-6,
    "ksp_rtol": 1e-6,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_factorization_type": "full",
    "pc_fieldsplit_0_fields": "1",
    "pc_fieldsplit_1_fields": "0",
    "fieldsplit_ksp_type": "preonly",
    "fieldsplit_0": {
        "ksp_converged_reason": None,
        "ksp_type": "cg",
        "pc_use_amat": False,
        "pc_type": "jacobi"
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

nvp = NonlinearVariationalProblem(F, z, J=J, Jp=Jp, bcs=bcs)
nvs = NonlinearVariationalSolver(nvp, solver_parameters=sp_mg)

out = VTKFile("out/Strain.pvd")
u, psi = z.subfunctions
u.rename("Displacement")

Q = TensorFunctionSpace(mesh, "DG", 0)
strain = Function(Q)
strain.rename("Strain")


u.assign(0)
u_old.assign(0)
psi.assign(0)
psi_old.assign(0)
alpha.assign(1e-2)

newton_its = 0
ksp_its = 0

for i in range(40):
    nvs.solve()
    nrm = norm(u-u_old, "H1")

    newton_its  += nvs.snes.its
    ksp_its += nvs.snes.getLinearSolveIterations()

    u_old.assign(u)
    psi_old.assign(psi)
    
    print(f"PG iteration {i+1}, alpha = {float(alpha):.2e}, Cauchy Error = {nrm:.2e}")

    if nrm < 1e-3:
        break
    if float(alpha) < 10:
        alpha.assign(sqrt(2)*alpha)

print(f"\nProximal Steps: {i+1}, Total Newton iterations: {newton_its}, Avg FGMRES its: {ksp_its/newton_its}")
strain.project(symgrad(u))
out.write(u,strain)