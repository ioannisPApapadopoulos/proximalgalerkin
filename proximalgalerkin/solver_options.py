BLOCK_VARIANTS = {
    "block_lu": {"top_left": "lu", "bottom": "lu"},
    "block_bjacobi_star": {"top_left": "bjacobi", "bottom": "star"},
    "block_jacobi_star": {"top_left": "jacobi", "bottom": "star"},
    "block_cg_bjacobi_star": {"top_left": "cg_bjacobi", "bottom": "star"},
    "block_cg_jacobi_star": {"top_left": "cg_bjacobi", "bottom": "star"},
    "block_star_star": {"top_left": "star", "bottom": "star"},
    "block_cg_bjacobi_chebyshev": {"top_left": "cg_bjacobi", "bottom": "chebyshev"},
    "block_cg_jacobi_chebyshev": {"top_left": "cg_jacobi", "bottom": "chebyshev"},
}

MONOLITHIC_SOLVERS = {"monolithic_vanka"}

def lu_parameters():
    sp_lu = {"snes_type": "newtonls",
        "snes_monitor": None,
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_atol": 1e-5,
    }
    return sp_lu

def top_left_parameters(kind):
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
            "ksp_type": "gmres",
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMStarPC",
        },
        "chebyshev": {
            "ksp_convergence_test": "skip",
            "ksp_max_it": smoothing_its,
            "ksp_type": "chebyshev",
        },
    }
    return options[kind]


def block_parameters(top_left, bottom, smoothing_its):
    return {
        "snes_monitor": None,
        "mat_type": "nest",
        "snes_stol": 0,
        "snes_atol": 1e-6,
        "ksp_type": "fgmres",
        "ksp_monitor": None,
        "ksp_converged_reason": None,
        "ksp_max_it": 200,
        "ksp_atol": 1e-7,
        "ksp_rtol": 1e-7,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_factorization_type": "full",
        "pc_fieldsplit_0_fields": "1",
        "pc_fieldsplit_1_fields": "0",
        "fieldsplit_ksp_type": "preonly",
        "fieldsplit_0": top_left_parameters(top_left),
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


def monolithic_vanka_parameters(smoothing_its):
    return {
        "mat_type": "nest",
        "snes_stol": 0,
        "snes_atol": 1e-6,
        "ksp_type": "fgmres",
        "ksp_converged_reason": None,
        "ksp_max_it": 200,
        "ksp_atol": 1e-7,
        "ksp_rtol": 1e-7,
        "pc_use_amat": False,
        "pc_type": "mg",
        "mg_levels": {
            "ksp_type": "gmres",
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