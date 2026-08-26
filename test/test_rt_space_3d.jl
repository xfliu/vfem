# test/test_rt_space_3d.jl
#
# CORNER-CASE TAXONOMY:
#   1. RT1 dimension contracts on the centroid-split special tetrahedron.
#   2. RT mass and DG mass matrix shapes and symmetry.
#   3. RT mass and DG mass midpoint matrices are positive definite.
#   4. Divergence coupling has DG rows and RT columns.
#   5. Debug helper returns the same dimensions.
#   6. RT2/RT3 arbitrary-degree transforms assemble without index errors.
#   7. RT0 is explicitly rejected by the current MATLAB-parity port.
#
# MATHEMATICAL CONTRACT:
#   This ports VFEM3D/build_scalar_rt_matrices.m for scalar RT_M mixed
#   matrices: A_rt = (sigma_i, sigma_j), B_rt = (div sigma_j, q_i), and
#   M_dg = (q_i, q_j), with facet normal DOFs globally shared.

using Test
using LinearAlgebra: Symmetric, eigvals
using SparseArrays: nnz
using VFEM: special_tetrahedron_mesh, simplex_dof, create_matrix_rt_3d,
            debug_rt_3d, RtData3D

@testset "rt_space_3d" begin
    m = special_tetrahedron_mesh()

    @testset "1. RT1 dimensions" begin
        rt = create_matrix_rt_3d(m, 1)
        @test rt isa RtData3D
        @test rt.DegK == simplex_dof(3, 1)
        @test rt.DegF == simplex_dof(2, 1)
        @test rt.DegRTInner == 3 * simplex_dof(3, 0)
        @test rt.DegRTElt == 4 * rt.DegF + rt.DegRTInner
        @test rt.DimDG == m.NumElt * rt.DegK
        @test rt.DimRT == m.NumF * rt.DegF + m.NumElt * rt.DegRTInner
    end

    @testset "2. shapes and symmetry" begin
        rt = create_matrix_rt_3d(m, 1)
        @test size(rt.A_rt) == (rt.DimRT, rt.DimRT)
        @test size(rt.B_rt) == (rt.DimDG, rt.DimRT)
        @test size(rt.M_dg) == (rt.DimDG, rt.DimDG)
        @test rt.A_rt ≈ rt.A_rt'
        @test rt.M_dg ≈ rt.M_dg'
    end

    @testset "3. positive definiteness" begin
        rt = create_matrix_rt_3d(m, 1)
        @test minimum(eigvals(Symmetric(Matrix(rt.A_rt)))) > 0
        @test minimum(eigvals(Symmetric(Matrix(rt.M_dg)))) > 0
    end

    @testset "4. divergence matrix nonzero" begin
        rt = create_matrix_rt_3d(m, 1)
        @test nnz(rt.B_rt) > 0
    end

    @testset "5. debug helper" begin
        s = debug_rt_3d(m, 1)
        @test s.DimDG == m.NumElt * simplex_dof(3, 1)
        @test s.symmetric_A
        @test s.symmetric_Mdg
        @test s.min_eig_A > 0
        @test s.min_eig_Mdg > 0
    end

    @testset "6. higher-degree dimensions" begin
        for p in 2:3
            rt = create_matrix_rt_3d(m, p)
            @test rt.DegK == simplex_dof(3, p)
            @test rt.DegF == simplex_dof(2, p)
            @test rt.DimDG == m.NumElt * rt.DegK
            @test rt.DimRT == m.NumF * rt.DegF + m.NumElt * 3 * simplex_dof(3, p - 1)
            @test size(rt.B_rt) == (rt.DimDG, rt.DimRT)
            @test rt.A_rt ≈ rt.A_rt'
            @test rt.M_dg ≈ rt.M_dg'
        end
    end

    @testset "7. RT0 currently rejected" begin
        @test_throws DomainError create_matrix_rt_3d(m, 0)
    end
end
