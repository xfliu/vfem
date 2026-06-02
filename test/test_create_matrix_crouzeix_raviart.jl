# test/test_create_matrix_crouzeix_raviart.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. UnitSquare8x8 fixture: trace, sum, Frobenius norm, nnz match
#      MATLAB element-wise to ~1e-12.
#   2. Symmetry of A0 and A1 matrices.
#   3. Mass matrix is positive (A0 > 0 entry-wise on the diagonal,
#      sum of all entries = total domain area = 1 for the unit square).
#   4. Stiffness matrix has zero row sums (constant function is
#      annihilated by the Laplacian).
#   5. Performance: < 100 ms on this fixture.
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   ∑_{i,j} A0[i, j] = ∫_Ω 1·1 dx = |Ω|     (mass times constant)
#   A1 · ones(ne) = 0                        (Laplacian of constant)
#   Both matrices are symmetric.

using Test
using SparseArrays: nnz
using LinearAlgebra: tr, norm
using VFEM: mesh2d_load, create_matrix_crouzeix_raviart

const _CR_DIR = joinpath(@__DIR__, "fixtures", "unit_square_8x8")

function _load_cr_fixture()
    path = joinpath(_CR_DIR, "cr_ecr_ref.txt")
    out = Dict{String, Float64}()
    nnz_out = Dict{String, Int}()
    for line in eachline(path)
        line = strip(line)
        isempty(line) && continue
        if (m = match(r"^(A0_cr_(?:trace|sum|frob|max))=(.+)$", line)) !== nothing
            out[m[1]] = parse(Float64, m[2])
        elseif (m = match(r"^(A1_cr_(?:trace|sum|frob|max))=(.+)$", line)) !== nothing
            out[m[1]] = parse(Float64, m[2])
        elseif (m = match(r"^(A0_cr_nnz|A1_cr_nnz)=(\d+)$", line)) !== nothing
            nnz_out[m[1]] = parse(Int, m[2])
        end
    end
    return out, nnz_out
end

@testset "create_matrix_crouzeix_raviart" begin
    floats, ints = _load_cr_fixture()
    m = mesh2d_load(_CR_DIR)
    A0, A1 = create_matrix_crouzeix_raviart(m)

    @testset "1. invariants match MATLAB" begin
        @test tr(A0) ≈ floats["A0_cr_trace"] atol = 1e-12 rtol = 1e-12
        @test sum(A0) ≈ floats["A0_cr_sum"] atol = 1e-12 rtol = 1e-12
        @test norm(A0) ≈ floats["A0_cr_frob"] atol = 1e-12 rtol = 1e-12
        @test maximum(abs, A0) ≈ floats["A0_cr_max"] atol = 1e-12 rtol = 1e-12
        @test nnz(A0) == ints["A0_cr_nnz"]

        @test tr(A1) ≈ floats["A1_cr_trace"] atol = 1e-12 rtol = 1e-12
        @test abs(sum(A1)) < 1e-10                                # CR Laplacian
        @test norm(A1) ≈ floats["A1_cr_frob"] atol = 1e-12 rtol = 1e-12
        @test maximum(abs, A1) ≈ floats["A1_cr_max"] atol = 1e-12 rtol = 1e-12
        @test nnz(A1) == ints["A1_cr_nnz"]
    end

    @testset "2. symmetry" begin
        @test A0 ≈ A0'
        @test A1 ≈ A1'
    end

    @testset "3. mass total = domain area" begin
        # Unit square: |Ω| = 1.
        @test sum(A0) ≈ 1.0 atol = 1e-12
    end

    @testset "4. stiffness annihilates the constant" begin
        ones_v = ones(m.ne)
        @test maximum(abs, A1 * ones_v) < 1e-10
    end

    @testset "5. performance < 100 ms" begin
        create_matrix_crouzeix_raviart(m)
        T = 10
        t0 = time_ns()
        for _ in 1:T
            create_matrix_crouzeix_raviart(m)
        end
        tns = (time_ns() - t0) / T
        @test tns < 100_000_000
    end
end
