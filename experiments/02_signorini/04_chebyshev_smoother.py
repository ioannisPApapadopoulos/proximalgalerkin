from firedrake import *
from netgen.occ import *
from collections import defaultdict


degree = 2
maxh = 0.05

disk = WorkPlane(Axes((0,0,0), n=Z, h=X)).Circle(1).Face()
geo = OCCGeometry(disk, dim=2)
ngmesh = geo.GenerateMesh(maxh=maxh)

distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.NONE, 1),}

if degree > 1:
    base_mesh = Mesh(Mesh(ngmesh).curve_field(degree), distribution_parameters=distribution_parameters)
else:
    base_mesh = Mesh(ngmesh, distribution_parameters=distribution_parameters)

nref = 1
mh = MeshHierarchy(base_mesh, nref)
mh_contact = SubmeshHierarchy(mh, subdomain_id="on_boundary")
mesh = mh[-1]
contact_boundary = mh_contact[-1]

V = VectorFunctionSpace(mesh, "CG", degree)
W = FunctionSpace(contact_boundary, "CG", degree)
Z = V*W


z = Function(Z)
u, psi = split(z)
psi_old = Function(W)
u_old = Function(V)
z_trial = TrialFunction(Z)
v, q = split(TestFunction(Z))


# Create measures
dx_1 = Measure("dx", mesh, intersect_measures=(Measure("dx", mesh),Measure("ds", contact_boundary)),metadata={'quadrature_degree': 4*degree})
ds_2 = Measure("dx", contact_boundary, intersect_measures=(Measure("ds", mesh),),metadata={'quadrature_degree': 4*degree})


n = FacetNormal(mesh)
n_def = -Constant((0,-1))*sign(n[1])

x, y = SpatialCoordinate(mesh)

Id = Identity(2)

mu = Constant(1)
lmbda = Constant(1)

def epsilon(u):
    return sym(grad(u))
def sigma(u):
    return 2*mu*epsilon(u) + lmbda*div(u)*Id
def obstacle_v(x):
    circle = as_vector([0, -1+sqrt(1-x**2)])
    # circle = as_vector([0., 0])
    circle_translated = circle #+ Constant([0, 0.1])
    return circle_translated

alpha = Constant(1e1)

F = inner(alpha*sigma(u), epsilon(v))*dx_1
F -= inner(alpha*Constant((0,-1)), v)*dx_1
F += inner(psi-psi_old, dot(v, n_def))*ds_2
F += inner(dot(u, n_def) + exp(-psi), q)*ds_2
F -= inner(dot(obstacle_v(x), Constant((0,-1))), q)*ds_2


z_trial = TrialFunction(Z)
eps = Constant(1e-5)
u_trial, psi_trial =  split(z_trial)
J = derivative(F, z, z_trial)
Jp = J + inner(1.0/(exp(-psi)+eps)*dot(u_trial,n_def),dot(v,n_def))*ds_2 - inner(eps*psi_trial, q)*ds_2



sp = {"snes_type": "newtonls",
      "snes_monitor": None,
      "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps",
      "snes_atol": 1e-5,
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
    "snes_atol": 1e-7,
    "ksp_type": "fgmres",
    "ksp_converged_reason": None,
    "ksp_monitor_true_residual": None,
    "ksp_max_it": 200,
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_factorization_type": "full",
    "pc_fieldsplit_0_fields": "1",
    "pc_fieldsplit_1_fields": "0",
    "fieldsplit_ksp_type": "preonly",
    "fieldsplit_0": {
        "ksp_converged_reason": None,
        # "ksp_monitor_true_residual": None,
        "ksp_type": "cg",
        "pc_use_amat": False,
        "pc_type": "jacobi",
        "ksp_rtol": 1e-7,
        "ksp_atol": 1e-7,
        # "pc_type": "python",
        # "pc_python_type": "firedrake.AssembledPC",
        # # "pc_python_type": "firedrake.ASMStarPC",
        # "assembled":{                
        #     "pc_type": "python",
        #     "pc_python_type": "firedrake.ASMStarPC",}
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
                "ksp_max_it": 2,
                "ksp_type": "chebyshev",
                "pc_type": "jacobi"
            },
        }
    }
}


def free_body_motion(Z, mesh):
    # (Near) nullspaces due to the fact that
    # the disk has not been fixed anywhere
    x, y = SpatialCoordinate(mesh)

    tx = Function(Z)
    tx_u, _ = tx.subfunctions
    tx_u.interpolate(Constant((1.0, 0.0)))

    ty = Function(Z)
    ty_u, _ = ty.subfunctions
    ty_u.interpolate(Constant((0.0, 1.0)))

    rot = Function(Z)
    rot_u, _ = rot.subfunctions
    rot_u.interpolate(as_vector((-y, x)))

    horizontal = VectorSpaceBasis([tx_u])
    horizontal.orthonormalize()

    rigid_like = VectorSpaceBasis([tx_u, ty_u, rot_u])
    rigid_like.orthonormalize()

    return (
        MixedVectorSpaceBasis(Z, [horizontal, Z.sub(1)]),
        MixedVectorSpaceBasis(Z, [rigid_like, Z.sub(1)]),
    )

nvp = NonlinearVariationalProblem(F, z, J=J, Jp=Jp)
exact_nullspace, near_nullspace = free_body_motion(Z, mesh)
nvs = NonlinearVariationalSolver(nvp, 
                                    solver_parameters=sp_mg,
                                    nullspace=exact_nullspace,
                                    transpose_nullspace=exact_nullspace,
                                    near_nullspace=near_nullspace)

out = VTKFile("out/Incompressible_bouncy_ball.pvd")
u, psi = z.subfunctions
u.rename("Displacement")
# out.write(u,stress)

Q = TensorFunctionSpace(mesh, "DG", degree-1)
stress = Function(Q)
stress.rename("Stress")


E = 10

# for nu in [0.4,0.49,0.499,0.4999,0.49999,0.499999, 0.4999999]:
# for nu in [0.4]:

nu = 0.4
u.assign(0)
u_old.assign(0)
psi.assign(0)
psi_old.assign(0)
alpha.assign(1e0)

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
    history["max_ksp_its"] = max(history["max_ksp_its"], nvs.snes.getLinearSolveIterations()/nvs.snes.its)


    psi_old.assign(psi)
    
    nrm = norm(u-u_old, "H1")
    print(f"PG iteration {i+1}, alpha = {float(alpha):.2e}, Cauchy Error = {nrm:.2e}")

    if nrm < 1e-6:
        break
    if float(alpha) < 10:
        alpha.assign(sqrt(2)*alpha)

print(f"\nE={E}, nu={nu}, PG Steps: {i+1}, Newton iterations: {history["newton_its"]}, Avg KSP its: {history["ksp_its"]/history["newton_its"]}, Max KSP its: {history["max_ksp_its"]}")

stress.project(sigma(u))
out.write(u,stress)