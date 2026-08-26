# test/test_create_matrix_lagrange.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. UnitSquare8x8 fixture, V = x²+y², degree=1 (P1) and degree=2 (P2):
#      trace, sum, Frobenius, nnz match MATLAB to ~1e-10.
#   2. Symmetry: A ≈ A', M ≈ M' for both P1 and P2.
#   3. Mass total = |Ω| = 1.
#   4. With V ≡ 0, the stiffness annihilates the constant function:
#      A · ones = 0 (interior dof; on a closed domain with no Dirichlet
#      removal — checking just the constant mode).
#   5. Wrong V_bern shape -> DimensionMismatch.
#   6. Bad degree -> ArgumentError.
#   7. Performance: < 1 s on this fixture.

using Test
using SparseArrays: nnz
using LinearAlgebra: tr, norm
using VFEM: mesh2d_load, create_matrix_lagrange, elem_V_bernstein

const _LAG_DIR = joinpath(@__DIR__, "fixtures", "unit_square_8x8")

function _load_lag_fixture()
    f = Dict{String, Float64}(); n = Dict{String, Int}()
    for line in eachline(joinpath(_LAG_DIR, "cecr_lag_ref.txt"))
        line = strip(line)
        if (mm = match(r"^([AM]_lag\d_(?:trace|sum|frob))=(.+)$", line)) !== nothing
            f[mm[1]] = parse(Float64, mm[2])
        elseif (mm = match(r"^([AM]_lag\d_nnz)=(\d+)$", line)) !== nothing
            n[mm[1]] = parse(Int, mm[2])
        end
    end
    return f, n
end

@testset "create_matrix_lagrange" begin
    floats, ints = _load_lag_fixture()
    m = mesh2d_load(_LAG_DIR)
    Vb = elem_V_bernstein(m, (x, y) -> x * x + y * y)

    @testset "1. P1 fixture matches MATLAB" begin
        A, M = create_matrix_lagrange(m, 1, Vb)
        @test tr(A) ≈ floats["A_lag1_trace"] atol = 1e-10 rtol = 1e-10
        @test sum(A) ≈ floats["A_lag1_sum"] atol = 1e-10 rtol = 1e-10
        @test norm(A) ≈ floats["A_lag1_frob"] atol = 1e-10 rtol = 1e-10
        @test nnz(A) == ints["A_lag1_nnz"]
        @test tr(M) ≈ floats["M_lag1_trace"] atol = 1e-12 rtol = 1e-12
        @test sum(M) ≈ floats["M_lag1_sum"] atol = 1e-12 rtol = 1e-12
        @test norm(M) ≈ floats["M_lag1_frob"] atol = 1e-12 rtol = 1e-12
        @test nnz(M) == ints["M_lag1_nnz"]
        @test A ≈ A'
        @test M ≈ M'
        @test sum(M) ≈ 1.0 atol = 1e-12
    end

    @testset "2. P2 fixture matches MATLAB" begin
        A, M = create_matrix_lagrange(m, 2, Vb)
        @test tr(A) ≈ floats["A_lag2_trace"] atol = 1e-9 rtol = 1e-9
        @test sum(A) ≈ floats["A_lag2_sum"] atol = 1e-10 rtol = 1e-10
        @test norm(A) ≈ floats["A_lag2_frob"] atol = 1e-10 rtol = 1e-10
        @test nnz(A) == ints["A_lag2_nnz"]
        @test tr(M) ≈ floats["M_lag2_trace"] atol = 1e-12 rtol = 1e-12
        @test sum(M) ≈ floats["M_lag2_sum"] atol = 1e-12 rtol = 1e-12
        @test norm(M) ≈ floats["M_lag2_frob"] atol = 1e-12 rtol = 1e-12
        @test nnz(M) == ints["M_lag2_nnz"]
        @test A ≈ A'
        @test M ≈ M'
        @test sum(M) ≈ 1.0 atol = 1e-12
    end

    @testset "4. V = 0 stiffness annihilates the constant" begin
        Vb0 = elem_V_bernstein(m, (x, y) -> 0.0)
        A1, _ = create_matrix_lagrange(m, 1, Vb0)
        A2, _ = create_matrix_lagrange(m, 2, Vb0)
        @test maximum(abs, A1 * ones(m.nv)) < 1e-10
        @test maximum(abs, A2 * ones(m.nv + m.ne)) < 1e-9
    end

    @testset "5–6. error preconditions" begin
        @test_throws DimensionMismatch create_matrix_lagrange(m, 1, zeros(m.nt, 14))
        @test_throws ArgumentError create_matrix_lagrange(m, 3, Vb)
    end

    @testset "7. performance < 1 s" begin
        create_matrix_lagrange(m, 1, Vb)
        T = 3
        t0 = time_ns()
        for _ in 1:T
            create_matrix_lagrange(m, 1, Vb)
        end
        tns = (time_ns() - t0) / T
        @test tns < 1_000_000_000
    end
end
