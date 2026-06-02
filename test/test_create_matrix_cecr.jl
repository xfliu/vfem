# test/test_create_matrix_cecr.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. UnitSquare8x8 fixture, c = 0.5 constant: trace, sum, Frobenius
#      match MATLAB to ~1e-10.
#   2. c = 0 reduces to plain enriched CR exactly.
#   3. Reaction-only check: with c constant, A_cecr − A_ecr_exact has
#      `c · |Ω|` total mass on cell DOFs and zero elsewhere.
#   4. c_data forms: scalar, function (x, y) -> Real, vector all match.
#   5. Wrong-length vector c -> DimensionMismatch.
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   A_cecr = A_ecr_exact + diag(c_K · |K|) on cell DOFs.
#   M_cecr = M_ecr_exact (since b̂ uses only the first component).

using Test
using SparseArrays: nnz
using LinearAlgebra: tr, norm
using VFEM: mesh2d_load, create_matrix_cecr,
            create_matrix_enriched_crouzeix_raviart

const _CECR_DIR = joinpath(@__DIR__, "fixtures", "unit_square_8x8")

function _load_cecr_fixture()
    f = Dict{String, Float64}(); n = Dict{String, Int}()
    for line in eachline(joinpath(_CECR_DIR, "cecr_lag_ref.txt"))
        line = strip(line)
        if (mm = match(r"^(A_cecr_(?:trace|sum|frob)|M_cecr_(?:trace|sum|frob))=(.+)$", line)) !== nothing
            f[mm[1]] = parse(Float64, mm[2])
        elseif (mm = match(r"^(A_cecr_nnz|M_cecr_nnz)=(\d+)$", line)) !== nothing
            n[mm[1]] = parse(Int, mm[2])
        end
    end
    return f, n
end

@testset "create_matrix_cecr" begin
    floats, ints = _load_cecr_fixture()
    m = mesh2d_load(_CECR_DIR)

    @testset "1. invariants match MATLAB at c = 0.5" begin
        A, M, _ = create_matrix_cecr(m, 0.5)
        @test tr(A) ≈ floats["A_cecr_trace"] atol = 1e-10 rtol = 1e-10
        @test sum(A) ≈ floats["A_cecr_sum"] atol = 1e-10 rtol = 1e-10
        @test norm(A) ≈ floats["A_cecr_frob"] atol = 1e-10 rtol = 1e-10
        @test nnz(A) == ints["A_cecr_nnz"]
        @test tr(M) ≈ floats["M_cecr_trace"] atol = 1e-12 rtol = 1e-12
        @test sum(M) ≈ floats["M_cecr_sum"] atol = 1e-12 rtol = 1e-12
        @test norm(M) ≈ floats["M_cecr_frob"] atol = 1e-12 rtol = 1e-12
        @test nnz(M) == ints["M_cecr_nnz"]
    end

    @testset "2. c = 0 reduces to enriched CR exactly" begin
        A0, M0, _ = create_matrix_cecr(m, 0.0)
        A_ref, M_ref, _ = create_matrix_enriched_crouzeix_raviart(m)
        @test A0 ≈ A_ref atol = 1e-12
        @test M0 ≈ M_ref atol = 1e-12
    end

    @testset "3. reaction adds c·|Ω| to total sum on cell DOFs" begin
        A0, _, _ = create_matrix_cecr(m, 0.0)
        A1, _, _ = create_matrix_cecr(m, 1.0)
        # Sum of (A1 - A0) entries = sum of c·|K| over elements = |Ω|.
        @test sum(A1 - A0) ≈ 1.0 atol = 1e-12
    end

    @testset "4. c_data forms agree" begin
        Aa, _, _ = create_matrix_cecr(m, 0.7)
        Ab, _, _ = create_matrix_cecr(m, fill(0.7, m.nt))
        Ac, _, _ = create_matrix_cecr(m, (x, y) -> 0.7)
        @test Aa ≈ Ab atol = 1e-13
        @test Aa ≈ Ac atol = 1e-13
    end

    @testset "5. wrong-length vector c errors" begin
        @test_throws DimensionMismatch create_matrix_cecr(m, fill(0.7, m.nt + 1))
    end
end
