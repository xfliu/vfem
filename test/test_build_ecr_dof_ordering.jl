# test/test_build_ecr_dof_ordering.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. Single-element mesh: 3 edges + 1 cell → ndof = 4. Edge DOFs
#      get [1, 2, 3], cell DOF gets 4.
#   2. Bijection: every DOF in 1..ndof appears exactly once across
#      `edge` and `cell`.
#   3. local_dof[k, :] consistent: first 3 columns equal `edge_dof[tri2edge[k, :]]`,
#      column 4 equals `cell_dof[k]`.
#   4. old_to_new ∘ new_to_old == identity.
#   5. Performance: < 5 ms on UnitSquare8x8 fixture (208 + 128 = 336 DOFs).

using Test
using VFEM: build_ecr_dof_ordering, mesh2d_load

const _DOF_DIR = joinpath(@__DIR__, "fixtures", "unit_square_8x8")

@testset "build_ecr_dof_ordering" begin
    @testset "1. single triangle" begin
        tri2edge = reshape([1 2 3], 1, 3)
        d = build_ecr_dof_ordering(tri2edge, 3)
        @test d.edge == [1, 2, 3]
        @test d.cell == [4]
        @test d.local_dof == reshape([1 2 3 4], 1, 4)
        @test sort(d.old_to_new) == 1:4
    end

    @testset "Cube fixture: bijection + local_dof consistency" begin
        m = mesh2d_load(_DOF_DIR)
        d = build_ecr_dof_ordering(m.tri2edge, m.ne)
        ndof = m.ne + m.nt
        @test sort(vcat(d.edge, d.cell)) == 1:ndof
        for k in 1:m.nt
            @test d.local_dof[k, 1] == d.edge[m.tri2edge[k, 1]]
            @test d.local_dof[k, 2] == d.edge[m.tri2edge[k, 2]]
            @test d.local_dof[k, 3] == d.edge[m.tri2edge[k, 3]]
            @test d.local_dof[k, 4] == d.cell[k]
        end
        # 4. round-trip permutation
        for i in 1:ndof
            @test d.new_to_old[d.old_to_new[i]] == i
        end
    end

    @testset "5. performance < 5 ms on UnitSquare8x8" begin
        m = mesh2d_load(_DOF_DIR)
        build_ecr_dof_ordering(m.tri2edge, m.ne)
        T = 50
        t0 = time_ns()
        for _ in 1:T
            build_ecr_dof_ordering(m.tri2edge, m.ne)
        end
        tns = (time_ns() - t0) / T
        @test tns < 5_000_000
    end
end
