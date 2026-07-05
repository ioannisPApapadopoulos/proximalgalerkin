from firedrake import *
from collections import defaultdict


degree = 2
n = 10
base_mesh = RectangleMesh(10*n, n, 1, 0.1)
nref = 1
mh = MeshHierarchy(base_mesh, nref)
mesh = mh[-1]

V = VectorFunctionSpace(mesh, "CG", degree)
W = TensorFunctionSpace(mesh, "DG", degree-1)
# W = TensorFunctionSpace(mesh, "DG", degree-1, symmetry=True) # does not work with MG!
Z = V*W

z = Function(Z)
u, psi = split(z)
psi_old = Function(W)
u_old = Function(V)
z_trial = TrialFunction(Z)
v, q = split(TestFunction(Z))


Id = Identity(2)

mu = Constant(1)
lmbda = Constant(1)

def epsilon(u):
    return sym(grad(u))
def sigma(u):
    return lmbda*div(u)*Id + 2*mu*epsilon(u)
def R(psi):
    return psi/sqrt(Constant(1)+inner(psi,psi))

def inverse_per_dR(psi, phi, eps, X, Y):
    denom = sqrt(1.0 + inner(psi,psi))
    scale = eps + phi / denom
    return inner(X,Y)/scale + phi * inner(psi, Y) * inner(psi, X) / (scale * (scale * denom**3 - phi * inner(psi,psi)))

def inverse_per_dR2(psi, phi, eps, X, Y):
    s = sqrt(1.0 + inner(psi,psi))
    return s/(eps*s+phi)*(inner(X,Y) + phi/(eps*s**3+phi) * inner(psi, Y) * inner(psi, X))


def dR2(psi):
    denom = sqrt(1.0 + inner(psi,psi))
    return Id/denom - outer(psi,psi)/denom**3


alpha = Constant(1e1)
f = Constant((0,-2e-1))
phi = Constant(0.01)

F = inner(alpha*sigma(u), epsilon(v))*dx(degree=10*degree)
F -= inner(alpha*f, v)*ds(2)
F += inner(psi-psi_old, epsilon(v))*dx(degree=10*degree)
F += inner(epsilon(u) - phi*R(psi), q)*dx(degree=10*degree)


z_trial = TrialFunction(Z)
eps = Constant(1e-4)
eps_J = Constant(1e-4)
u_trial, psi_trial =  split(z_trial)
J = derivative(F, z, z_trial)  - inner(eps_J*psi_trial, q)*dx
Jp = (derivative(F, z, z_trial) 
      + inverse_per_dR2(psi,phi,eps,epsilon(u_trial),epsilon(v))*dx(degree=10*degree)
      - inner(eps*psi_trial, q)*dx
)



sp = {"snes_type": "newtonls",
      "snes_monitor": None,
      "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps",
      "snes_atol": 1e-6,
    #   "snes_linesearch_type": "l2"
}

sp_krylov = {
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
        "ksp_type": "preonly",
        "pc_use_amat": False,
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled" : {
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    },
    "fieldsplit_1": {
        "ksp_type": "preonly",
        "pc_use_amat": False,
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled" : {
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    },
}

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
        # "ksp_monitor_true_residual": None,
        "ksp_type": "preonly",
        "pc_use_amat": False,
        "pc_type": "bjacobi"
        # "pc_type": "python",
        # "pc_python_type": "firedrake.AssembledPC",
        # "pc_python_type": "firedrake.ASMStarPC",
        # "assembled":{                
            # "pc_type": "python",
            # "pc_python_type": "firedrake.ASMStarPC",}
        },
    "fieldsplit_1": {
        "ksp_type": "preonly",
        "pc_use_amat": False,
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled":{
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
                "ksp_type": "gmres",
                "pc_type": "python",
                "pc_python_type": "firedrake.ASMStarPC",
            },
        }
    }
}

bcs = DirichletBC(Z.sub(0), 0, [1])
nvp = NonlinearVariationalProblem(F, z, bcs=bcs, J=J, Jp=Jp)
nvs = NonlinearVariationalSolver(nvp, solver_parameters=sp_mg)

out = VTKFile("out/strain_limited_beam.pvd")
u, psi = z.subfunctions
u.rename("Displacement")
# out.write(u,strain)

Q = TensorFunctionSpace(mesh, "CG", degree)
strain = Function(Q)
strain.rename("strain")
strain_obs = Function(Q)
strain_obs.rename("observed strain")

E = 200

# for nu in [0.4,0.49,0.499,0.4999,0.49999,0.499999, 0.4999999]:
# for nu in [0.4]:

nu = 0.4
u.assign(0)
u_old.assign(0)
psi.assign(0)
psi_old.assign(0)
alpha.assign(1e-2)

lmbda.assign(E*nu / (1+nu) / (1-2*nu))
mu.assign(E/(2*(1+nu)))
history = defaultdict(list)
history["newton_its"] = 0
history["ksp_its"] = 0
history["max_ksp_its"] = 0


for i in range(40):

    print(f"alpha = {float(alpha)}")
    u_old.assign(u)
    nvs.solve()

    history["newton_its"]  += nvs.snes.its
    history["ksp_its"] += nvs.snes.getLinearSolveIterations()
    if nvs.snes.its > 0:
        history["max_ksp_its"] = max(history["max_ksp_its"], nvs.snes.getLinearSolveIterations()/nvs.snes.its)
    else:
        history["max_ksp_its"] = 0


    psi_old.assign(psi)
    
    nrm = norm(u-u_old, "H1")
    print(f"PG iteration {i+1}, alpha = {float(alpha):.2e}, Cauchy Error = {nrm:.2e}")

    if nrm < 1e-5:
        break
    if float(alpha) < 10:
        alpha.assign(sqrt(2)*alpha)

print(f"\nE={E}, nu={nu}, PG Steps: {i+1}, Newton iterations: {history["newton_its"]}, Avg KSP its: {history["ksp_its"]/history["newton_its"]}, Max KSP its: {history["max_ksp_its"]}")
strain.project(epsilon(u))
strain_obs.interpolate(phi*R(psi))
out.write(u,strain,strain_obs)