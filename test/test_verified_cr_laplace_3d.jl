# test/test_verified_cr_laplace_3d.jl
#
# CORNER-CASE TAXONOMY:
#   1. `special_tetrahedron_mesh` encodes the vertices from
#      `docs/notes/special_tetrahedron.md` and the exact centroid split.
#   2. Verified CR solve on T_F returns interval eigenvalues for the
#      reduced Dirichlet problem.
#   3. The verified intervals enclose the Float64 generalized eigenvalues
#      of the same assembled CR pencil.
#   4. Eigenvalues are positive and sorted by lower bound.
#   5. Error preconditions: neig = 0 and zero-dimensional Dirichlet CR
#      spaces are rejected.
#   6. Performance: the T_F interval solve finishes in < 30 s.
#
# MATHEMATICAL CONTRACT:
#   This verifies the discrete 3D CR Laplace eigenproblem
#   A u = lambda M u after removing boundary facet DOFs. Mesh points are
#   treated as exactly given; the T_F coordinates use binary-rational
#   coordinates from `docs/notes/special_tetrahedron.md`.

using Test
using LinearAlgebra: Symmetric, eigen
using IntervalArithmetic: Interval, inf, sup
using VFEM: Mesh3D, get_facet_list, get_edge_list, facet_element_connectivity,
            special_tetrahedron_mesh, create_matrix_crouzeix_raviart_3d,
            verified_cr_laplace_3d, CrLaplaceEig3D, red_refine_mesh_3d,
            special_tetrahedron_red_mesh, find_mesh_hmax_3d

@testset "verified_cr_laplace_3d" begin
    @testset "1. special_tetrahedron_mesh geometry" begin
        m = special_tetrahedron_mesh()
        @test m.NodeList[1:4, :] == [ 0.0  0.0  0.0;
                                      0.0  0.0  1.0;
                                      0.5  0.5  0.5;
                                     -0.5  0.5  0.5 ]
        @test m.NodeList[5, :] == [0.0, 0.25, 0.5]
        @test m.NumElt == 4
        @test m.NumF == 10
    end

    @testset "1b. red-refined special tetrahedron mesh" begin
        m0 = special_tetrahedron_red_mesh(0)
        m1 = special_tetrahedron_red_mesh(1)
        @test m0.NumNode == 4
        @test m0.NumElt == 1
        @test m1.NumNode == 10
        @test m1.NumElt == 8
        @test m1.NodeList[5:10, :] == [ 0.0   0.0   0.5;
                                         0.25  0.25  0.25;
                                        -0.25  0.25  0.25;
                                         0.25  0.25  0.75;
                                        -0.25  0.25  0.75;
                                         0.0   0.5   0.5 ]
        @test find_mesh_hmax_3d(m1) ≈ sqrt(0.5) * find_mesh_hmax_3d(m0)

        m_centroid = special_tetrahedron_mesh()
        @test red_refine_mesh_3d(m_centroid).NumElt == 8 * m_centroid.NumElt
    end

    @testset "2-4. verified T_F discrete eigenvalues" begin
        m = special_tetrahedron_mesh()
        r = verified_cr_laplace_3d(m, 4)
        @test r isa CrLaplaceEig3D
        @test r.eig_value isa Vector{Interval{Float64}}
        @test length(r.eig_value) ≥ 4

        M_f, A_f, info = create_matrix_crouzeix_raviart_3d(m)
        int_dofs = info.interior_dofs
        F = eigen(Symmetric(Matrix(A_f[int_dofs, int_dofs])),
                  Symmetric(Matrix(M_f[int_dofs, int_dofs])))
        λ_float = sort(F.values)

        for k in 1:4
            @test inf(r.eig_value[k]) > 0
            @test inf(r.eig_value[k]) ≤ λ_float[k] ≤ sup(r.eig_value[k])
            @test sup(r.eig_value[k]) - inf(r.eig_value[k]) < 1e-8
        end
        @test all(inf(r.eig_value[k]) ≤ inf(r.eig_value[k + 1])
                  for k in 1:3)
    end

    @testset "5. error preconditions" begin
        m = special_tetrahedron_mesh()
        @test_throws DomainError verified_cr_laplace_3d(m, 0)

        nodes = m.NodeList[1:4, :]
        elements = [1 2 3 4]
        facets = get_facet_list(elements)
        edges = get_edge_list(elements)
        f2e, e2f = facet_element_connectivity(elements, facets)
        one_tet = Mesh3D(nodes, elements, facets, edges, f2e, e2f,
                         4, 1, 4, 6)
        @test_throws ArgumentError verified_cr_laplace_3d(one_tet, 1)
    end

    @testset "6. performance < 30 s" begin
        m = special_tetrahedron_mesh()
        verified_cr_laplace_3d(m, 2)
        t0 = time_ns()
        verified_cr_laplace_3d(m, 2)
        @test (time_ns() - t0) / 1e9 < 30.0
    end
end
