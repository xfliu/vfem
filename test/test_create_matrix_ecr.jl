# test/test_create_matrix_ecr.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. UnitSquare8x8 fixture: trace, sum, Frobenius, max abs entry, nnz
#      match MATLAB to ~1e-12.
#   2. Symmetry of A and M.
#   3. Mass matrix M ⪰ 0 (positive on diagonal; total mass = |Ω| = 1).
#   4. Stiffness A annihilates the constant (A·ones_dof = 0 on interior DOFs).
#      Note: ECR DOFs are edge averages + cell averages. The constant
#      function p ≡ 1 is represented by ones(ndof), since each edge
#      average and cell average of `1` equals 1.
#   5. Performance: < 500 ms on this fixture.
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   ECR space: P_1(K) ⊕ span{x²+y²}. Mass and stiffness assembled by
#   Dunavant degree-4-exact quadrature with the basis reconstructed
#   from DOF definitions (4×4 system per element). Reproduces MATLAB
#   matrix invariants on the canonical fixture.

using Test
using SparseArrays: nnz
using LinearAlgebra: tr, norm
using VFEM: mesh2d_load, create_matrix_ecr

const _ECR_DIR = joinpath(@__DIR__, "fixtures", "unit_square_8x8")

function _load_ecr_fixture()
    path = joinpath(_ECR_DIR, "cr_ecr_ref.txt")
    floats = Dict{String, Float64}()
    ints = Dict{String, Int}()
    for line in eachline(path)
        line = strip(line)
        isempty(line) && continue
        if (m = match(r"^(A_ecr_(?:trace|sum|frob|max)|M_ecr_(?:trace|sum|frob|max))=(.+)$", line)) !== nothing
            floats[m[1]] = parse(Float64, m[2])
        elseif (m = match(r"^(A_ecr_nnz|M_ecr_nnz)=(\d+)$", line)) !== nothing
            ints[m[1]] = parse(Int, m[2])
        end
    end
    return floats, ints
end

@testset "create_matrix_ecr" begin
    floats, ints = _load_ecr_fixture()
    m = mesh2d_load(_ECR_DIR)
    A, M = create_matrix_ecr(m)

    @testset "1. invariants match MATLAB" begin
        @test tr(A) ≈ floats["A_ecr_trace"] atol = 1e-10 rtol = 1e-10
        # A_ecr_sum is on the order of 1e-14, so tighten the absolute tol.
        @test abs(sum(A) - floats["A_ecr_sum"]) < 1e-9
        @test norm(A) ≈ floats["A_ecr_frob"] atol = 1e-10 rtol = 1e-10
        @test maximum(abs, A) ≈ floats["A_ecr_max"] atol = 1e-10 rtol = 1e-10
        @test nnz(A) == ints["A_ecr_nnz"]

        @test tr(M) ≈ floats["M_ecr_trace"] atol = 1e-12 rtol = 1e-12
        @test sum(M) ≈ floats["M_ecr_sum"] atol = 1e-12 rtol = 1e-12
        @test norm(M) ≈ floats["M_ecr_frob"] atol = 1e-12 rtol = 1e-12
        @test maximum(abs, M) ≈ floats["M_ecr_max"] atol = 1e-12 rtol = 1e-12
        @test nnz(M) == ints["M_ecr_nnz"]
    end

    @testset "2. symmetry" begin
        @test A ≈ A'
        @test M ≈ M'
    end

    @testset "3. mass total = domain area" begin
        @test sum(M) ≈ 1.0 atol = 1e-12
    end

    @testset "4. stiffness annihilates the constant" begin
        ndof = m.ne + m.nt
        @test maximum(abs, A * ones(ndof)) < 1e-9
    end

    @testset "5. performance < 500 ms on this fixture" begin
        create_matrix_ecr(m)
        T = 3
        t0 = time_ns()
        for _ in 1:T
            create_matrix_ecr(m)
        end
        tns = (time_ns() - t0) / T
        @test tns < 500_000_000
    end
end
