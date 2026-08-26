# test/test_dg_space_3d.jl
#
# CORNER-CASE TAXONOMY:
#   1. Degree 0, 1, 2 dimensions match NumElt * binomial(p+3, 3).
#   2. Local-to-global DG blocks are element-owned and contiguous.
#   3. Mass matrix is symmetric positive definite on the midpoint matrix.
#   4. Constant vector integrates to the tetrahedron domain volume.
#   5. Interval mode encloses Float64 summaries.
#   6. Degenerate tetrahedron raises DomainError.
#   7. Debug helper returns consistent summary fields.
#
# MATHEMATICAL CONTRACT:
#   DG^p on a tetrahedral mesh is the broken scalar polynomial space with
#   local basis ordered by `ijkl_list(p)`. There are no boundary DOFs and
#   the mass matrix is block diagonal by element.

using Test
using LinearAlgebra: Symmetric, dot, eigvals
using IntervalArithmetic: Interval, inf, sup
using VFEM: Mesh3D, get_facet_list, get_edge_list, facet_element_connectivity,
            special_tetrahedron_mesh, simplex_dof, dg_l2g_3d,
            create_matrix_dg_3d, debug_dg_3d, ijkl_list,
            bernstein_multinomial_3d

function _dg_constant_coeffs(info)
    local_one = bernstein_multinomial_3d(info.degree, ijkl_list(info.degree))
    c = Vector{Float64}(undef, info.DimDG)
    for e in 1:(info.DimDG ÷ info.DegK)
        c[((e - 1) * info.DegK + 1):(e * info.DegK)] .= local_one
    end
    return c
end

@testset "dg_space_3d" begin
    m = special_tetrahedron_mesh()

    @testset "1. dimensions for p = 0, 1, 2" begin
        for p in 0:2
            _, info = dg_l2g_3d(m, p)
            @test info.DegK == simplex_dof(3, p)
            @test info.DimDG == m.NumElt * simplex_dof(3, p)
        end
    end

    @testset "2. local-to-global blocks" begin
        for p in 0:3
            L2G, info = dg_l2g_3d(m, p)
            for e in 1:m.NumElt
                @test L2G[e, :] == collect((e - 1) * info.DegK .+ (1:info.DegK))
            end
        end
    end

    @testset "3. mass symmetry and SPD" begin
        for p in 0:2
            M, _ = create_matrix_dg_3d(m, p)
            @test M ≈ M'
            @test minimum(eigvals(Symmetric(Matrix(M)))) > 0
        end
    end

    @testset "4. constant mass is domain volume" begin
        for p in 0:2
            M, info = create_matrix_dg_3d(m, p)
            c = _dg_constant_coeffs(info)
            @test dot(c, M * c) ≈ 1 / 12 atol = 1e-14
        end
    end

    @testset "5. interval mode encloses Float64 summaries" begin
        M_f, _ = create_matrix_dg_3d(m, 2)
        M_i, _ = create_matrix_dg_3d(m, 2; T = Interval{Float64})
        @test inf(sum(M_i)) ≤ sum(M_f) ≤ sup(sum(M_i))
        @test maximum(sup.(M_i) .- inf.(M_i)) < 1e-12
    end

    @testset "6. degenerate tetrahedron" begin
        nodes = [0.0 0.0 0.0;
                 1.0 0.0 0.0;
                 2.0 0.0 0.0;
                 3.0 0.0 0.0]
        elements = [1 2 3 4]
        facets = get_facet_list(elements)
        edges = get_edge_list(elements)
        f2e, e2f = facet_element_connectivity(elements, facets)
        md = Mesh3D(nodes, elements, facets, edges, f2e, e2f, 4, 1, 4, 6)
        @test_throws DomainError create_matrix_dg_3d(md, 1)
    end

    @testset "7. debug helper" begin
        s = debug_dg_3d(m, 2)
        @test s.DimDG == s.expected_DimDG
        @test s.symmetric
        @test s.constant_mass ≈ s.domain_volume atol = 1e-14
        @test s.min_eig_mid > 0
    end
end
