# test/test_create_matrix_lagrange_3d.jl
#
# CORNER-CASE TAXONOMY:
#   1. p = 1, 2, 3 local/global dimensions are consistent.
#   2. Boundary DOFs are removed by facet ownership, with nonempty interior
#      space on the centroid-split special tetrahedron.
#   3. Stiffness and mass are symmetric.
#   4. Constant function has zero stiffness and mass equal to |T_F|.
#   5. Interval assembly encloses Float64 summaries.
#   6. Dirichlet eigenvalues on the special tetrahedron are positive and
#      above the closed-form first eigenvalue.
#   7. Debug helper and invalid input preconditions.
#
# MATHEMATICAL CONTRACT:
#   The conforming tetrahedral Lagrange basis is represented in monomial
#   barycentric coefficients ordered by `ijkl_list(p)`. Shared DOFs are
#   canonicalized on vertices, edges, and faces; cell DOFs are local.

using Test
using LinearAlgebra: dot
using IntervalArithmetic: Interval, inf, sup
using VFEM: Mesh3D, get_facet_list, get_edge_list, facet_element_connectivity,
            special_tetrahedron_mesh, simplex_dof, ijkl_list,
            bernstein_multinomial_3d, lagrange_l2g_3d,
            create_matrix_lagrange_3d, laplace_eig_lagrange_3d,
            debug_lagrange_3d

function _cg_constant_coeffs(info, L2G)
    local_one = Float64.(bernstein_multinomial_3d(info.degree, ijkl_list(info.degree)))
    c = zeros(Float64, info.DimCG)
    for e in axes(L2G, 1)
        c[L2G[e, :]] .= local_one
    end
    return c
end

@testset "create_matrix_lagrange_3d" begin
    m = special_tetrahedron_mesh()

    @testset "1. dimensions" begin
        for p in 1:3
            L2G, info = lagrange_l2g_3d(m, p)
            @test size(L2G) == (m.NumElt, simplex_dof(3, p))
            @test info.DegK == simplex_dof(3, p)
            @test info.DimCG ≥ m.NumNode
        end
    end

    @testset "2. boundary/interior DOFs" begin
        for p in 1:3
            _, info = lagrange_l2g_3d(m, p)
            @test !isempty(info.bd_dofs)
            @test !isempty(info.interior_dofs)
            @test isempty(intersect(info.bd_dofs, info.interior_dofs))
        end
    end

    @testset "3. symmetry" begin
        for p in 1:3
            A, M, _, _ = create_matrix_lagrange_3d(m, p)
            @test A ≈ A'
            @test M ≈ M'
        end
    end

    @testset "4. constant identities" begin
        for p in 1:3
            A, M, info, L2G = create_matrix_lagrange_3d(m, p)
            c = _cg_constant_coeffs(info, L2G)
            @test maximum(abs, A * c) < 1e-10
            @test dot(c, M * c) ≈ 1 / 12 atol = 1e-14
        end
    end

    @testset "5. interval mode encloses Float64 summaries" begin
        A_f, M_f, _, _ = create_matrix_lagrange_3d(m, 2)
        A_i, M_i, _, _ = create_matrix_lagrange_3d(m, 2; T = Interval{Float64})
        @test inf(sum(A_i)) ≤ sum(A_f) ≤ sup(sum(A_i))
        @test inf(sum(M_i)) ≤ sum(M_f) ≤ sup(sum(M_i))
    end

    @testset "6. eigenvalue upper-bound sanity" begin
        r = laplace_eig_lagrange_3d(m, 2, 1)
        λ_exact_1 = (π^2 / 4) * 80
        @test length(r.eig_value) == 1
        @test r.eig_value[1] > λ_exact_1
    end

    @testset "7. debug and preconditions" begin
        s = debug_lagrange_3d(m, 2)
        @test s.symmetric_A
        @test s.symmetric_M
        @test s.constant_stiffness_norm < 1e-10
        @test s.constant_mass ≈ 1 / 12 atol = 1e-14
        @test s.min_eig_M_int > 0
        @test_throws ArgumentError lagrange_l2g_3d(m, 0)

        nodes = [0.0 0.0 0.0;
                 1.0 0.0 0.0;
                 2.0 0.0 0.0;
                 3.0 0.0 0.0]
        elements = [1 2 3 4]
        facets = get_facet_list(elements)
        edges = get_edge_list(elements)
        f2e, e2f = facet_element_connectivity(elements, facets)
        md = Mesh3D(nodes, elements, facets, edges, f2e, e2f, 4, 1, 4, 6)
        @test_throws DomainError create_matrix_lagrange_3d(md, 1)
    end
end
