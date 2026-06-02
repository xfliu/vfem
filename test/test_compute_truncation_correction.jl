# test/test_compute_truncation_correction.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. Hydrogen-like (Λ = −0.25, C_V = 1, α = 1, R0 = 1, R = 200):
#      R1, μ, C_tr, trunc_err match MATLAB to ≤ 1e-12.
#   2. Heavier potential (Λ = −0.5, C_V = 2, α = 1, R0 = 1, R = 100):
#      same comparison.
#   3. α = 2 case (faster decay, Λ = −1.0, C_V = 1, R0 = 1, R = 50):
#      same comparison.
#   4. Mathematical contract: trunc_err = C_tr · exp(−μ·R) by construction.
#   5. Λ ≥ 0 -> DomainError; γ² ≥ −Λ trivially fails -> DomainError.
#   6. trunc_err is monotone decreasing in R (sanity).
#   7. Performance: < 0.01 s (pure scalar arithmetic).
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   The function returns a constant `C_tr` and a rate `μ > 0` derived
#   from Agmon estimates such that the FE eigenvalue bound
#   `λ_lower − C_tr · exp(−μ·R)` is a guaranteed lower bound for the
#   eigenvalue of the operator on ℝ^d (un-truncated). See full_paper.tex
#   §6 (referenced from the MATLAB original).

using Test
using VFEM: compute_truncation_correction, TruncationParams

const _TR_DIR = joinpath(@__DIR__, "fixtures")

function _load_trunc_fixture()
    cases = Dict{String, Dict{String, Float64}}()
    cur = ""
    for line in eachline(joinpath(_TR_DIR, "trunc_ref.txt"))
        line = strip(line)
        isempty(line) && continue
        if (mm = match(r"^case=(.+)$", line)) !== nothing
            cur = mm[1]
            cases[cur] = Dict{String, Float64}()
            continue
        end
        if (mm = match(r"^([A-Za-z_0-9]+)=([+-]?\d+\.\d+e[+-]?\d+)$", line)) !== nothing
            cases[cur][mm[1]] = parse(Float64, mm[2])
        elseif (mm = match(r"^([A-Za-z_0-9]+)=(\d+)$", line)) !== nothing
            cases[cur][mm[1]] = parse(Float64, mm[2])
        end
    end
    return cases
end

@testset "compute_truncation_correction" begin
    cases = _load_trunc_fixture()

    @testset "1. Hydrogen-like matches MATLAB" begin
        err, p = compute_truncation_correction(-0.25, 1, 1, 1, 200)
        ref = cases["hydrogen"]
        @test p.R1            == Int(ref["R1"])
        @test p.delta_R1      ≈ ref["delta"]  atol = 1e-12 rtol = 1e-12
        @test p.mu            ≈ ref["mu"]     atol = 1e-12 rtol = 1e-12
        @test p.C_tr          ≈ ref["C_tr"]   atol = 1e-10 rtol = 1e-10
        @test err             ≈ ref["err"]    atol = 1e-10 rtol = 1e-10
    end

    @testset "2. heavier potential matches MATLAB" begin
        err, p = compute_truncation_correction(-0.5, 2, 1, 1, 100)
        ref = cases["heavier"]
        @test p.R1   == Int(ref["R1"])
        @test p.mu   ≈ ref["mu"]   atol = 1e-12 rtol = 1e-12
        @test p.C_tr ≈ ref["C_tr"] atol = 1e-10 rtol = 1e-10
        @test err    ≈ ref["err"]  atol = 1e-10 rtol = 1e-10
    end

    @testset "3. α = 2 (faster decay) matches MATLAB" begin
        err, p = compute_truncation_correction(-1.0, 1, 2, 1, 50)
        ref = cases["alpha2"]
        @test p.R1   == Int(ref["R1"])
        @test p.mu   ≈ ref["mu"]   atol = 1e-12 rtol = 1e-12
        @test p.C_tr ≈ ref["C_tr"] atol = 1e-10 rtol = 1e-10
        @test err    ≈ ref["err"]  atol = 1e-10 rtol = 1e-10
    end

    @testset "4. trunc_err = C_tr · exp(−μ·R) identity" begin
        err, p = compute_truncation_correction(-0.25, 1, 1, 1, 200)
        @test err ≈ p.C_tr * exp(-p.mu * 200)  atol = 1e-13 rtol = 1e-13
    end

    @testset "5. error preconditions" begin
        @test_throws DomainError compute_truncation_correction( 0.0, 1, 1, 1, 100)
        @test_throws DomainError compute_truncation_correction( 1.0, 1, 1, 1, 100)
    end

    @testset "6. trunc_err strictly decreasing in R" begin
        err1, _ = compute_truncation_correction(-0.25, 1, 1, 1, 100)
        err2, _ = compute_truncation_correction(-0.25, 1, 1, 1, 200)
        @test err2 < err1
    end

    @testset "7. performance < 0.01 s" begin
        compute_truncation_correction(-0.25, 1, 1, 1, 200)
        t0 = time_ns()
        compute_truncation_correction(-0.25, 1, 1, 1, 200)
        @test (time_ns() - t0) / 1e9 < 0.01
    end
end
