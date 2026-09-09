BLOCK_VARIANTS = {
    "block_bjacobi_star": {"top_left": "bjacobi", "bottom": "star"},
    "block_jacobi_star": {"top_left": "jacobi", "bottom": "star"},
    "block_cg_bjacobi_star": {"top_left": "cg_bjacobi", "bottom": "star"},
    "block_cg_bjacobi_cg_star": {"top_left": "cg_bjacobi", "bottom": "cg_star"},
    "block_cg_jacobi_star": {"top_left": "cg_jacobi", "bottom": "star"},
    "block_star_star": {"top_left": "star", "bottom": "star"},
    "block_cg_bjacobi_chebyshev_jacobi": {"top_left": "cg_bjacobi", "bottom": "chebyshev_jacobi"},
    "block_cg_jacobi_chebyshev_jacobi": {"top_left": "cg_jacobi", "bottom": "chebyshev_jacobi"},
    "block_cg_bjacobi_chebyshev_bjacobi": {"top_left": "cg_bjacobi", "bottom": "chebyshev_bjacobi"},
    "block_cg_jacobi_chebyshev_bjacobi": {"top_left": "cg_jacobi", "bottom": "chebyshev_bjacobi"},
    "block_cg_bjacobi_gmres_bjacobi": {"top_left": "cg_bjacobi", "bottom": "gmres_bjacobi"},
    "block_cg_bjacobi_cg_bjacobi": {"top_left": "cg_bjacobi", "bottom": "cg_bjacobi"},
    "block_cg_bjacobi_lu": {"top_left": "cg_bjacobi", "bottom": "lu"},
}

MONOLITHIC_SOLVERS = {"monolithic_vanka"}

def lu_parameters(atol):
    sp_lu = {"snes_type": "newtonls",
        "snes_monitor": None,
        "ksp_type": "preonly",
        # "ksp_monitor": None,
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_atol": atol,
    }
    return sp_lu

def block_lu_parameters(atol):
    sp_krylov = {
        "mat_type": "nest",
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_stol": 0,
        "snes_atol": atol,
        "ksp_type": "fgmres",
        "ksp_converged_reason": None,
        "ksp_monitor_true_residual": None,
        "ksp_max_it": 200,
        "ksp_atol": atol,
        "ksp_rtol": atol,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_factorization_type": "full",
        "pc_fieldsplit_0_fields": "1",
        "pc_fieldsplit_1_fields": "0",
        # "fieldsplit_ksp_type": "preonly",
        "fieldsplit_0": {
            "ksp_type": "preonly",
            "ksp_monitor": None,
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            # "pc_type": "python",
            # "pc_python_type": "firedrake.AssembledPC",
            # "assembled" : {
            #     "pc_type": "lu",
            #     "pc_factor_mat_solver_type": "mumps",
            # }
        },
        "fieldsplit_1": {
            "ksp_type": "preonly",
            "pc_use_amat": False,
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            # "pc_type": "python",
            # "pc_python_type": "firedrake.AssembledPC",
            # "assembled" : {
            #     "pc_type": "lu",
            #     "pc_factor_mat_solver_type": "mumps",
            # }
        },
    }
    return sp_krylov

def top_left_parameters(kind, atol):
    options = {
        "jacobi": {
            "ksp_type": "preonly",
            "ksp_converged_reason": None,
            "pc_use_amat": False,
            "pc_type": "jacobi",
        },
        "bjacobi": {
            "ksp_type": "preonly",
            "pc_use_amat": False,
            "pc_type": "bjacobi",
        },
        "cg_bjacobi": {
            "ksp_type": "cg",
            "ksp_converged_reason": None,
            "pc_use_amat": False,
            "pc_type": "bjacobi",
        },
        "cg_jacobi": {
            "ksp_type": "cg",
            # "ksp_atol": atol/1e2,
            # "ksp_rtol": 0,
            "ksp_converged_reason": None,
            "pc_use_amat": False,
            "pc_type": "jacobi",
        },
        "star": {
            "ksp_type": "cg",
            "ksp_converged_reason": None,
            "pc_use_amat": False,
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled": {
                "pc_type": "python",
                "pc_python_type": "firedrake.ASMStarPC",
            },
        },
        "lu": {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled": {
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        },
    }
    return options[kind]


def bottom_mg_levels(kind, smoothing_its):
    options = {
        "lu": {
            "ksp_type": "preonly",
            "pc_use_amat": False,
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled": {
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        },
        "star": {
            "ksp_convergence_test": "skip",
            "ksp_max_it": smoothing_its,
            "ksp_type": "chebyshev",
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMStarPC",
            "pc_star_use_coloring": True,
        },
        "cg_star": {
            "ksp_convergence_test": "skip",
            "ksp_max_it": smoothing_its,
            "ksp_type": "cg",
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMStarPC",
            "pc_star_use_coloring": True,
        },
        "chebyshev_jacobi": {
            "ksp_convergence_test": "skip",
            "ksp_max_it": smoothing_its,
            "ksp_type": "chebyshev",
            "pc_type": "jacobi"
        },
        "chebyshev_bjacobi": {
            "ksp_convergence_test": "skip",
            "ksp_max_it": smoothing_its,
            "ksp_type": "chebyshev",
            "pc_type": "bjacobi"
        },
        "gmres_bjacobi": {
            "ksp_convergence_test": "skip",
            "ksp_max_it": smoothing_its,
            "ksp_type": "gmres",
            "pc_type": "bjacobi"
        },
        "cg_bjacobi": {
            "ksp_convergence_test": "skip",
            "ksp_max_it": smoothing_its,
            "ksp_type": "cg",
            "pc_type": "bjacobi"
        },
    }
    return options[kind]


def block_parameters(top_left, bottom, smoothing_its, atol):
    return {
        "snes_monitor": None,
        "mat_type": "nest",
        # "pmat_type": "aij",
        "snes_stol": 0,
        "snes_atol": atol,
        "ksp_type": "fgmres",
        "ksp_monitor": None,
        "ksp_converged_reason": None,
        "ksp_max_it": 200,
        "ksp_atol": atol/1e1,
        "ksp_rtol": atol/1e1,
        "pc_use_amat": False,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_factorization_type": "full",
        "pc_fieldsplit_0_fields": "1",
        "pc_fieldsplit_1_fields": "0",
        # "fieldsplit_ksp_type": "preonly",
        "fieldsplit_0": top_left_parameters(top_left, atol),
        "fieldsplit_1": {
            "ksp_type": "preonly",
            "pc_use_amat": False,
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled": {
                "pc_use_amat": False,
                "pc_type": "mg",
                "pc_mg_type": "full",
                "mg_coarse_mat_type": "aij",
                "mg_coarse_pc_type": "lu",
                "mg_coarse_pc_factor_mat_solver_type": "mumps",
                "mg_coarse_mat_mumps_icntl_14": 1000,
                "mg_levels": bottom_mg_levels(bottom, smoothing_its),
            },
        },
    }


def monolithic_vanka_parameters(smoothing_its,atol):
    return {
        "mat_type": "aij",
        "snes_monitor": None,
        "snes_stol": 0,
        "snes_atol": atol,
        "ksp_type": "fgmres",
        "ksp_monitor": None,
        "ksp_converged_reason": None,
        "ksp_max_it": 200,
        "ksp_atol": atol/10,
        "ksp_rtol": atol/10,
        "pc_use_amat": False,
        "pc_type": "mg",
        "mg_levels": {
            "ksp_type": "chebyshev",
            "ksp_max_it": smoothing_its,
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMVankaPC",
            "pc_vanka_construct_dim": 0,
            "pc_vanka_backend": "tinyasm",
        },
        "mg_coarse": {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    }
