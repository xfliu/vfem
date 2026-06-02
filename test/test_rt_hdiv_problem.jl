# test/test_rt_hdiv_problem.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. UnitSquare8x8 fixture, RT_order = 2, single all-ones f column.
#      Result is a 1×1 matrix matching MATLAB to ≤ 1e-10.
#   2. Same fixture, two-column f (sin/cos along the global numbering).
#      The 2×2 result matches MATLAB entry-by-entry.
#   3. Symmetry of the 2×2 result.
#   4. RT_order < 0 -> DomainError.
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   For input f (Lagrange CG coefficients), `mat_b_w_w[i, j]` is the
#   Goerisch w-w bilinear form ⟨A_RT u_i, u_j⟩ where u_i, u_j are the
#   RT-mixed solutions driven by the i-th, j-th columns of f. Result
#   is symmetric. Cross-validated against MATLAB R2024a.

using Test
using LinearAlgebra: tr, norm
using VFEM: mesh2d_load, rt_hdiv_problem

const _RT_DIR = joinpath(@__DIR__, "fixtures", "unit_square_8x8")

function _load_rt_fixture()
    f = Dict{String, Float64}(); s = Dict{String, String}()
    for line in eachline(joinpath(_RT_DIR, "rt_hdiv_ref.txt"))
        line = strip(line)
        isempty(line) && continue
        if (mm = match(r"^(\w+)=(.+)$", line)) !== nothing
            try
                f[mm[1]] = parse(Float64, mm[2])
            catch
                s[mm[1]] = mm[2]
            end
        end
    end
    return f, s
end

@testset "rt_hdiv_problem" begin
    floats, strs = _load_rt_fixture()
    m = mesh2d_load(_RT_DIR)
    RT_order = 2
    nlag = m.nv + m.ne   # P2 Lagrange (no interior DOFs at order = 2)

    @testset "1. all-ones f, scalar result matches MATLAB" begin
        f = ones(nlag, 1)
        result = rt_hdiv_problem(m, RT_order, f)
        @test size(result) == (1, 1)
        @test result[1, 1] ≈ floats["value"] atol = 1e-10 rtol = 1e-10
    end

    @testset "2. sin/cos two-column f matches MATLAB entry-by-entry" begin
        nlag = m.nv + m.ne
        f = zeros(nlag, 2)
        f[:, 1] = sin.(range(0, π; length = nlag))
        f[:, 2] = cos.(range(0, π; length = nlag))
        result = rt_hdiv_problem(m, RT_order, f)
        @test size(result) == (2, 2)
        @test result[1, 1] ≈ floats["M2_11"] atol = 1e-10 rtol = 1e-10
        @test result[1, 2] ≈ floats["M2_12"] atol = 1e-10 rtol = 1e-10
        @test result[2, 2] ≈ floats["M2_22"] atol = 1e-10 rtol = 1e-10
        @test tr(result) ≈ floats["M2_trace"] atol = 1e-10 rtol = 1e-10
        @test norm(result) ≈ floats["M2_frob"] atol = 1e-10 rtol = 1e-10
    end

    @testset "3. symmetry of the result" begin
        nlag = m.nv + m.ne
        f = randn(nlag, 3)
        result = rt_hdiv_problem(m, RT_order, f)
        @test result ≈ result' atol = 1e-10
    end

    @testset "4. RT_order < 0 -> DomainError" begin
        f = ones(1, 1)
        @test_throws DomainError rt_hdiv_problem(m, -1, f)
    end
end
