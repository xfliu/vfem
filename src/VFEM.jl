module VFEM

# ============================================================================
# VFEM.jl — verified finite element library, Julia port of MATLAB VFEM_LIB.
#
# Two modes, distinguished only by the element type T of inputs:
#   * approximation: T = Float64        (fast, not rigorous)
#   * verified     : T = Interval{Float64}  (rigorous, via IntervalArithmetic)
#
# All numeric routines are written generically on T<:Real and let interval
# enclosures flow through. There is no MATLAB-style I_intval/I_inf/I_sup
# dispatch layer — Julia's type system covers it.
#
# Source layout:
#   src/core/         — general-purpose FEM kernel (mesh, bernstein,
#                       assembly, eigensolvers, potentials). Reusable
#                       across any 2D / 3D problem on triangles / tets.
#   src/applications/ — concrete problem drivers (Hydrogen, H₂⁺,
#                       constant_V, …). Each is a thin submodule that
#                       composes the kernel with problem-specific
#                       parameters, meshes, and reference values.
#
# Public API is built up phase-by-phase per plan.md. See Rule.md for the
# non-negotiable test contract: every routine has its own test file.
# ============================================================================

using LinearAlgebra
using SparseArrays
using IntervalArithmetic: Interval, interval, inf, sup, mid, hull, mag

# ---- Phase 1: triangle singular moments (port of matlab_lib/) --------------
include("core/singular_moments/poly_power_linear.jl")
include("core/singular_moments/polymul.jl")
include("core/singular_moments/tri_polar_sing_moment_le4_exact.jl")
include("core/singular_moments/tri_general_sing_moment_le4_exact.jl")
include("core/singular_moments/tri_polar_sing_moment_1_over_r2.jl")
include("core/singular_moments/tri_general_sing_moment_1_over_r2.jl")

# ---- Phase 2: Bernstein quadrature kernel (port of VFEM3D/lib/quadrature/) -
include("core/bernstein/simplex_dof.jl")
include("core/bernstein/ijkl_list.jl")
include("core/bernstein/ijkl_index.jl")
include("core/bernstein/bernstein_multinomial.jl")
include("core/bernstein/bernstein_eval.jl")
include("core/bernstein/bernstein_product.jl")
include("core/bernstein/inner_prod_matrix.jl")

# ---- Phase 2 (cont): vertex-singular Coulomb closed forms -----------------
include("core/quadrature_singular/tri3d_invR_face_moments.jl")
include("core/quadrature_singular/tet_vertex_sing_bernstein_moments.jl")
include("core/quadrature_singular/tet_vertex_sing_poly_integral.jl")
include("core/quadrature_singular/singular_potential_matrix_vertex_exact.jl")

# ---- Phase 3: tetrahedral mesh layer (port of VFEM3D/lib/mesh/) -----------
include("core/mesh/mesh_struct.jl")
include("core/mesh/get_facet_list.jl")
include("core/mesh/get_edge_list.jl")
include("core/mesh/facet_element_connectivity.jl")
include("core/mesh/mesh_load_from_folder.jl")
include("core/mesh/mesh_info.jl")
include("core/mesh/dof_on_facet.jl")
include("core/mesh/facet_id_at_element.jl")
include("core/mesh/find_mesh_hmax_3d.jl")

# ---- Phase 4a: 2D mesh layer + CR/ECR assembly (port of vfem2d/lib/) ------
include("core/mesh2d/mesh2d_struct.jl")
include("core/mesh2d/find_tri2edge.jl")
include("core/mesh2d/find_is_edge_bd.jl")
include("core/mesh2d/find_mesh_hmax.jl")
include("core/mesh2d/mesh2d_load.jl")
include("core/assembly2d/dunavant_rule_6.jl")
include("core/assembly2d/create_matrix_crouzeix_raviart.jl")
include("core/assembly2d/create_matrix_ecr.jl")

# ---- Phase 4b: CECR + Lagrange + enriched CR + element-V ------------------
include("core/assembly2d/build_ecr_dof_ordering.jl")
include("core/assembly2d/create_matrix_enriched_crouzeix_raviart.jl")
include("core/assembly2d/create_matrix_cecr.jl")
include("core/assembly2d/elem_V_bernstein.jl")
include("core/assembly2d/create_matrix_lagrange.jl")

# ---- Phase 4c: 2D Schrödinger eigensolvers --------------------------------
include("core/eigensolve2d/apply_dirichlet_bc.jl")
include("core/eigensolve2d/schrodinger_eig_cecr.jl")

# ---- Phase 4d: Lehmann–Goerisch RT auxiliary problem ----------------------
include("core/eigensolve2d/rt_hdiv_problem.jl")
include("core/eigensolve2d/laplace_eig_lagrange.jl")
include("core/eigensolve2d/lg_lower_eig_bound_laplace.jl")
include("core/eigensolve2d/verified_lg_lower_eig.jl")

# ---- Phase 5: 3D ECR/CECR assembly (port of VFEM3D/) ----------------------
include("core/assembly3d/create_matrix_ecr_3d.jl")
include("core/assembly3d/create_matrix_cecr_3d.jl")
include("core/eigensolve3d/schrodinger_eig_cecr_3d.jl")
include("core/eigensolve3d/compute_truncation_correction.jl")

# ---- Phase 5: Coulomb potential helpers -----------------------------------
include("core/potentials/elem_V_coulomb_average.jl")
include("core/potentials/elem_V_coulomb_bounds.jl")
include("core/potentials/elem_V_coulomb_Lp_integral_3d.jl")

export tri_polar_sing_moment_le4_exact,
       tri_general_sing_moment_le4_exact,
       tri_polar_sing_moment_1_over_r2,
       tri_general_sing_moment_1_over_r2,
       simplex_dof,
       ijkl_list,
       ijkl_index,
       ijkl_index_map,
       bernstein_multinomial_3d,
       bernstein_eval,
       bernstein3d_eval_bary,
       bernstein_product_exact,
       bernstein_product_monomial,
       inner_prod_matrix,
       inner_prod_matrix_reference,
       inner_prod_matrix_reference3d,
       tri3d_invR_face_moments,
       tet_vertex_sing_bernstein_moments,
       tet_vertex_sing_poly_integral,
       singular_potential_matrix_vertex_exact,
       Mesh3D,
       mesh_load_from_folder,
       mesh_info,
       get_facet_list,
       get_edge_list,
       facet_element_connectivity,
       facet_element_connectivity_with_sign,
       dof_on_facet,
       common_dof_on_facets,
       facet_id_at_element,
       Mesh2D,
       mesh2d_load,
       find_tri2edge,
       find_is_edge_bd,
       find_mesh_hmax,
       dunavant_rule_6,
       create_matrix_crouzeix_raviart,
       create_matrix_ecr,
       EcrDofOrdering,
       build_ecr_dof_ordering,
       create_matrix_enriched_crouzeix_raviart,
       create_matrix_cecr,
       bernstein4_multiindices_2d,
       elem_V_bernstein,
       create_matrix_lagrange,
       interior_ecr_dofs,
       restrict_to_interior,
       SchrodingerEig,
       schrodinger_eig_cecr,
       rt_hdiv_problem,
       verified_rt_hdiv_problem,
       LaplaceEigLagrange,
       laplace_eig_lagrange,
       lagrange_laplace_matrices,
       LGLaplaceLowerBound,
       lg_lower_eig_bound_laplace,
       verified_cr_liu_lower,
       verified_lg_transform,
       find_mesh_hmax_3d,
       EcrDof3D,
       create_matrix_ecr_3d,
       create_matrix_cecr_3d,
       SchrodingerEig3D,
       schrodinger_eig_cecr_3d,
       TruncationParams,
       compute_truncation_correction,
       CoulombInfo,
       elem_V_coulomb_average,
       elem_V_coulomb_bounds,
       elem_V_coulomb_Lp_integral_3d

# ---- Applications submodule namespace -------------------------------------
# Concrete problem drivers (Hydrogen, H₂⁺, constant_V, …) live as
# submodules under `VFEM.Applications`. Each submodule composes the
# kernel exports above with problem-specific parameters and meshes.
# Today it's a placeholder; populate as drivers are ported. See
# `src/applications/README.md` for the per-application convention.
include("applications/Applications.jl")

# ---- CECR certification pipeline (m4–m7) ----------------------------------
include("applications/cecr_pipeline/CecrPipeline.jl")

export mesh2d_load_ne,
       CaseConfig, MeshConstants, update_sigma,
       compute_mesh_constants,
       cecr_lower_bound,
       p1_upper_bound,
       eps_h_method_A_2d, eps_h_method_A_3d, compute_eps_h,
       corrected_lower_bound,
       ceps_diagnostic_2d, ceps_diagnostic_3d, ceps_diagnostic,
       coulomb_average_2d,
       run_cecr_pipeline

end # module
