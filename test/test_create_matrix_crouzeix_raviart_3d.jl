# test/test_create_matrix_crouzeix_raviart_3d.jl
#
# CORNER-CASE TAXONOMY:
#   1. Fundamental tetrahedron centroid split: matrix sizes match the
#      facet count and DOF info separates 4 boundary / 6 interior facets.
#   2. Symmetry: mass and stiffness are symmetric.
#   3. Constant mode: the unrestricted stiffness annihilates ones.
#   4. Mass integral: ones' M ones equals |T_F| = 1/12.
#   5. Interval mode encloses the Float64 summaries with narrow widths.
#   6. Preconditions: a degenerate tetrahedron raises DomainError.
#   7. Performance: small exact-coordinate benchmark assembles in < 2 s.
#
# MATHEMATICAL CONTRACT:
#   The 3D CR basis is phi_i = 1 - 3 L_i on each tetrahedron, associated
#   with the face opposite vertex i. Local mass entries are 2|K|/5 on the
#   diagonal and -|K|/20 off-diagonal; local stiffness is
#   9 |K| grad(L_i) dot grad(L_j).

using Test
using LinearAlgebra: dot, norm
using IntervalArithmetic: Interval, inf, sup
using VFEM: Mesh3D, get_facet_list, get_edge_list, facet_element_connectivity,
            special_tetrahedron_mesh, create_matrix_crouzeix_raviart_3d,
            CrDof3D

@testset "create_matrix_crouzeix_raviart_3d" begin
    m = special_tetrahedron_mesh()

    @testset "1. size and DOF info" begin
        M, A, info = create_matrix_crouzeix_raviart_3d(m)
        @test info isa CrDof3D
        @test size(M) == (m.NumF, m.NumF)
        @test size(A) == (m.NumF, m.NumF)
        @test length(info.boundary_dofs) == 4
        @test length(info.interior_dofs) == 6
        @test isempty(intersect(info.boundary_dofs, info.interior_dofs))
    end

    @testset "2. symmetry" begin
        M, A, _ = create_matrix_crouzeix_raviart_3d(m)
        @test M ≈ M'
        @test A ≈ A'
    end

    @testset "3. stiffness annihilates constants" begin
        _, A, _ = create_matrix_crouzeix_raviart_3d(m)
        @test maximum(abs, A * ones(size(A, 1))) < 1e-12
    end

    @testset "4. mass integrates constants on T_F" begin
        M, _, _ = create_matrix_crouzeix_raviart_3d(m)
        c = ones(size(M, 1))
        @test dot(c, M * c) ≈ 1 / 12 atol = 1e-14
    end

    @testset "5. interval mode encloses Float64 summaries" begin
        M_f, A_f, _ = create_matrix_crouzeix_raviart_3d(m)
        M_i, A_i, _ = create_matrix_crouzeix_raviart_3d(m; T = Interval{Float64})
        @test inf(sum(M_i)) ≤ sum(M_f) ≤ sup(sum(M_i))
        @test inf(sum(A_i)) ≤ sum(A_f) ≤ sup(sum(A_i))
        @test maximum(sup.(M_i) .- inf.(M_i)) < 1e-12
        @test maximum(sup.(A_i) .- inf.(A_i)) < 1e-10
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
        @test_throws DomainError create_matrix_crouzeix_raviart_3d(md)
    end

    @testset "7. performance < 2 s" begin
        create_matrix_crouzeix_raviart_3d(m)
        t0 = time_ns()
        create_matrix_crouzeix_raviart_3d(m)
        @test (time_ns() - t0) / 1e9 < 2.0
    end
end
