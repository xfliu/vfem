# test/test_one_piece_bubble_laplace_3d.jl
#
# One-piece degree-p bubble polynomial Dirichlet solve on a single
# tetrahedron, with no mesh subdivision.

using Test
using VFEM: one_piece_bubble_laplace_3d, special_tetrahedron_vertices,
            simplex_dof

@testset "one_piece_bubble_laplace_3d" begin
    vertices = special_tetrahedron_vertices()

    @testset "degree 12 special tetrahedron" begin
        r = one_piece_bubble_laplace_3d(vertices, 12, 5)
        exact = (pi^2 / 4.0) .* [80.0, 140.0, 140.0, 160.0, 208.0]
        @test size(r.A) == (simplex_dof(3, 8), simplex_dof(3, 8))
        @test size(r.M) == size(r.A)
        @test r.eig_value[1] ≈ exact[1] rtol = 1e-7
        @test r.eig_value[2] ≈ exact[2] rtol = 1e-4
        @test r.eig_value[3] ≈ exact[3] rtol = 1e-4
        @test r.eig_value[4] ≈ exact[4] rtol = 1e-4
        @test r.eig_value[5] ≈ exact[5] rtol = 1e-3
    end

    @testset "preconditions" begin
        @test_throws DomainError one_piece_bubble_laplace_3d(vertices, 3, 1)
        @test_throws DomainError one_piece_bubble_laplace_3d(vertices, 12, 0)
        @test_throws DimensionMismatch one_piece_bubble_laplace_3d(vertices[1:3, :], 12, 1)
    end
end
