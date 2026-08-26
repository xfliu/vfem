# test/test_create_matrix_enriched_crouzeix_raviart.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. UnitSquare8x8 fixture: trace, sum, Frobenius norm, nnz match
#      MATLAB to ~1e-10.
#   2. Symmetry: A ≈ A', M ≈ M'.
#   3. Mass total = |Ω| = 1 (unit square).
#   4. Stiffness annihilates the constant: A · ones = 0 (interior dof).
#   5. Performance: < 1 s on this fixture.
#
# MATHEMATICAL CONTRACT (docs/testing-contract.md (c)):
#   ECR is exact via degree-2 Bernstein basis on each triangle. The
#   resulting Gram and stiffness matrices match the no-quadrature
#   MATLAB assembly element-wise.

using Test
using SparseArrays: nnz
using LinearAlgebra: tr, norm
using VFEM: mesh2d_load, create_matrix_enriched_crouzeix_raviart

const _ENR_DIR = joinpath(@__DIR__, "fixtures", "unit_square_8x8")

function _load_enr_fixture()
    path = joinpath(_ENR_DIR, "cecr_lag_ref.txt")
    f = Dict{String, Float64}(); n = Dict{String, Int}()
    for line in eachline(path)
        line = strip(line)
        isempty(line) && continue
        if (mm = match(r"^(A_ecr_exact_(?:trace|sum|frob)|M_ecr_exact_(?:trace|sum|frob))=(.+)$", line)) !== nothing
            f[mm[1]] = parse(Float64, mm[2])
        elseif (mm = match(r"^(A_ecr_exact_nnz|M_ecr_exact_nnz)=(\d+)$", line)) !== nothing
            n[mm[1]] = parse(Int, mm[2])
        end
    end
    return f, n
end

@testset "create_matrix_enriched_crouzeix_raviart" begin
    floats, ints = _load_enr_fixture()
    m = mesh2d_load(_ENR_DIR)
    A, M, _ = create_matrix_enriched_crouzeix_raviart(m)

    @testset "1. invariants match MATLAB" begin
        @test tr(A) ≈ floats["A_ecr_exact_trace"] atol = 1e-10 rtol = 1e-10
        @test abs(sum(A) - floats["A_ecr_exact_sum"]) < 1e-10
        @test norm(A) ≈ floats["A_ecr_exact_frob"] atol = 1e-10 rtol = 1e-10
        @test nnz(A) == ints["A_ecr_exact_nnz"]
        @test tr(M) ≈ floats["M_ecr_exact_trace"] atol = 1e-12 rtol = 1e-12
        @test sum(M) ≈ floats["M_ecr_exact_sum"] atol = 1e-12 rtol = 1e-12
        @test norm(M) ≈ floats["M_ecr_exact_frob"] atol = 1e-12 rtol = 1e-12
        @test nnz(M) == ints["M_ecr_exact_nnz"]
    end

    @testset "2. symmetry" begin
        @test A ≈ A'
        @test M ≈ M'
    end

    @testset "3. mass total = |Ω| = 1" begin
        @test sum(M) ≈ 1.0 atol = 1e-12
    end

    @testset "4. stiffness annihilates the constant" begin
        ndof = m.ne + m.nt
        @test maximum(abs, A * ones(ndof)) < 1e-9
    end

    @testset "5. performance < 1 s" begin
        create_matrix_enriched_crouzeix_raviart(m)
        T = 3
        t0 = time_ns()
        for _ in 1:T
            create_matrix_enriched_crouzeix_raviart(m)
        end
        tns = (time_ns() - t0) / T
        @test tns < 1_000_000_000
    end
end
