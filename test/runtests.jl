# test/runtests.jl
#
# Top-level test entry. One file per routine, lumped via include.
# Rule.md mandate: every routine has its own test file with (a) typical,
# (b) corner-case taxonomy at the top, (c) mathematical contract,
# (d) at least one efficiency benchmark with regression threshold.

using Test
import VFEM

# Shared support modules — included once at the top so per-routine test
# files just `using` them instead of re-defining methods on each include.
include(joinpath(@__DIR__, "support", "refmoment.jl"))

@testset "VFEM.jl" begin
    # Phase 1: triangle singular moments
    include("test_poly_power_linear.jl")
    include("test_polymul.jl")
    include("test_tri_polar_sing_moment_le4_exact.jl")
    include("test_tri_polar_sing_moment_1_over_r2.jl")
    include("test_tri_general_sing_moment_le4_exact.jl")
    include("test_tri_general_sing_moment_1_over_r2.jl")

    # Phase 2: Bernstein quadrature kernel — foundation
    include("test_simplex_dof.jl")
    include("test_ijkl_list.jl")
    include("test_ijkl_index.jl")
    include("test_bernstein_multinomial.jl")
    include("test_bernstein_eval.jl")

    # Phase 2: Bernstein product + inner product matrices
    include("test_bernstein_product.jl")
    include("test_inner_prod_matrix.jl")

    # Phase 2: vertex-singular Coulomb closed forms
    include("test_tri3d_invR_face_moments.jl")
    include("test_tet_vertex_sing_bernstein_moments.jl")
    include("test_singular_potential_matrix_vertex_exact.jl")
    include("test_tet_vertex_sing_poly_integral.jl")

    # Phase 3: tetrahedral mesh layer
    include("test_mesh_load_cube_r1.jl")
    include("test_dof_on_facet.jl")
    include("test_facet_id_at_element.jl")

    # Phase 4a: 2D mesh + CR/ECR assembly
    include("test_mesh2d_unit_square.jl")
    include("test_create_matrix_crouzeix_raviart.jl")
    include("test_create_matrix_ecr.jl")

    # Phase 4b: CECR + Lagrange + enriched CR + element-V
    include("test_build_ecr_dof_ordering.jl")
    include("test_create_matrix_enriched_crouzeix_raviart.jl")
    include("test_create_matrix_cecr.jl")
    include("test_elem_V_bernstein.jl")
    include("test_create_matrix_lagrange.jl")

    # Phase 4c: 2D Schrödinger eigensolvers
    include("test_schrodinger_eig_cecr.jl")

    # Phase 4d: Lehmann–Goerisch RT auxiliary
    include("test_rt_hdiv_problem.jl")
    include("test_laplace_eig_lagrange.jl")
    include("test_lg_lower_eig_bound_laplace.jl")
    include("test_verified_lg_lower_eig.jl")

    # Phase 5: 3D ECR/CECR assembly
    include("test_create_matrix_ecr_3d.jl")
    include("test_create_matrix_cecr_3d.jl")
    include("test_schrodinger_eig_cecr_3d.jl")
    include("test_elem_V_coulomb_average.jl")
    include("test_elem_V_coulomb_bounds.jl")
    include("test_elem_V_coulomb_Lp_integral_3d.jl")
    include("test_hydrogen_e2e.jl")
    include("test_compute_truncation_correction.jl")
end
