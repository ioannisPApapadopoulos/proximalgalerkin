from firedrake import *

"""
This script implements a block preconditioner for the Newton linear systems of proximal Galerkin applied to a Signorini problem.

Let Ω be the unit square. The Signorini problem implemented here seeks to minimize
   ∫_Ω μ |symgrad u|^2 + λ |div u|^2/2  dx
subject to u ⋅ (0, -1) >= 0 almost everywhere at y = 0 where the top edge of the domain is compressed 0.1 vertically downwards.

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
base_mesh = UnitSquareMesh(32,32)
mh = MeshHierarchy(base_mesh, 1)
mh_contact = SubmeshHierarchy(mh, subdomain_id="on_boundary")
mesh = mh[-1]
contact_boundary = mh_contact[-1]

V = VectorFunctionSpace(mesh, "CG", 1)
W = FunctionSpace(contact_boundary, "CG", 1)
Z = V*W


z = Function(Z)
u, psi = split(z)
psi_old = Function(W)
u_old = Function(V)
z_trial = TrialFunction(Z)
v, q = split(TestFunction(Z))


# Create measures
dx_1 = Measure("dx", mesh, intersect_measures=(Measure("dx", mesh),Measure("ds", contact_boundary)),metadata={'max_quadrature_degree': 4})
ds_2 = Measure("dx", contact_boundary, intersect_measures=(Measure("ds", mesh),),metadata={'max_quadrature_degree': 4})


n = FacetNormal(mesh)
n_def = -Constant((0,-1))*sign(n[1])

x, y = SpatialCoordinate(mesh)

Id = Identity(2)

mu = Constant(1)
lmbda = Constant(1)

def epsilon(u):
    return sym(grad(u))
def sigma(u):
    return lmbda*div(u)*Id + 2*mu*epsilon(u)

alpha = Constant(1e1)

F = inner(alpha*sigma(u), epsilon(v))*dx_11
F += inner(psi-psi_old, dot(v, n_def))*ds_2
F += inner(dot(u, n_def) + exp(-psi), q)*ds_2


z_trial = TrialFunction(Z)
eps = Constant(1e-5)
u_trial, psi_trial =  split(z_trial)
J = derivative(F, z, z_trial)
Jp = J + inner(1.0/(exp(-psi)+eps)*dot(u_trial,n_def),dot(v,n_def))*ds_2 - inner(eps*psi_trial, q)*ds_2


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

bcs = DirichletBC(Z.sub(0), Constant((0,-0.1)), 4)
nvp = NonlinearVariationalProblem(F, z, J=J, Jp=Jp, bcs=bcs)
nvs = NonlinearVariationalSolver(nvp, solver_parameters=sp_mg)

out = VTKFile("out/Signorini.pvd")
u, psi = z.subfunctions
u.rename("Displacement")

Q = TensorFunctionSpace(mesh, "DG", 0)
stress = Function(Q)
stress.rename("Stress")

E = 10
nu = 0.4
u.assign(0)
u_old.assign(0)
psi.assign(0)
psi_old.assign(0)
alpha.assign(1e0)

lmbda.assign(E*nu / (1+nu) / (1-2*nu))
mu.assign(E/(2*(1+nu)))
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

    if nrm < 1e-4:
        break
    if float(alpha) < 10:
        alpha.assign(sqrt(2)*alpha)

print(f"\nProximal Steps: {i+1}, Total Newton iterations: {newton_its}, Avg FGMRES its: {ksp_its/newton_its}")
stress.project(sigma(u))
out.write(u,stress)