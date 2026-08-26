# test/test_poly_power_linear.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. p = 0       -> [1]   (the empty product)
#   1b. (c0+c1*t)^0 with c0,c1=0 -> [1]
#   2. p = 1       -> [c0, c1]
#   3. p = 2       -> [c0², 2 c0 c1, c1²]
#   4. p = 4       -> binomial expansion sanity check
#   5. p < 0       -> DomainError
#   6. c0 = 0      -> [0,…,0, c1^p]   (zero leading coefficients)
#   7. c1 = 0      -> [c0^p, 0,…,0]
#   8. Interval inputs: result encloses the Float64 result entry-wise.
#   9. Performance: p = 50 must run < 50 µs.
#
# MATHEMATICAL CONTRACT (docs/testing-contract.md (c)):
#   Σ_{k=0}^{p} coeff[k+1] · t^k  ≡  (c0 + c1·t)^p   for all t ∈ ℝ.
#   Verified by evaluating both sides at random t and checking equality.

using Test, Random
using IntervalArithmetic: Interval, interval, inf, sup
using VFEM: poly_power_linear

const _PPL = VFEM.poly_power_linear  # local alias

# Internal Horner evaluator; must NOT use the same machinery as the routine.
function _horner(coeff::AbstractVector, t)
    s = zero(promote_type(eltype(coeff), typeof(t)))
    for k in length(coeff):-1:1
        s = s * t + coeff[k]
    end
    return s
end

@testset "poly_power_linear" begin
    Random.seed!(20260501)

    @testset "1. p = 0" begin
        @test _PPL(2.0, 3.0, 0) == [1.0]
        @test _PPL(0.0, 0.0, 0) == [1.0]      # corner 1b
        @test _PPL(-7.5, 0.25, 0) == [1.0]
    end

    @testset "2. p = 1" begin
        @test _PPL(2.0, 3.0, 1) == [2.0, 3.0]
    end

    @testset "3. p = 2 (matches MATLAB docstring example)" begin
        @test _PPL(2.0, 3.0, 2) == [4.0, 12.0, 9.0]
    end

    @testset "4. p = 4 binomial expansion" begin
        c0, c1 = 1.5, -0.7
        coeff = _PPL(c0, c1, 4)
        @test length(coeff) == 5
        # Direct binomial check: coeff[k+1] = C(4,k) c0^(4-k) c1^k
        for k in 0:4
            @test coeff[k + 1] ≈ binomial(4, k) * c0^(4 - k) * c1^k
        end
    end

    @testset "5. negative p throws DomainError" begin
        @test_throws DomainError _PPL(1.0, 1.0, -1)
        @test_throws DomainError _PPL(1.0, 1.0, -3)
    end

    @testset "6. c0 = 0" begin
        @test _PPL(0.0, 5.0, 3) == [0.0, 0.0, 0.0, 125.0]
    end

    @testset "7. c1 = 0" begin
        @test _PPL(2.0, 0.0, 3) == [8.0, 0.0, 0.0, 0.0]
    end

    @testset "Mathematical contract: identity at random t" begin
        for _ in 1:50
            c0 = randn(); c1 = randn(); p = rand(0:6)
            t = randn()
            coeff = _PPL(c0, c1, p)
            @test _horner(coeff, t) ≈ (c0 + c1 * t)^p atol = 1e-10 rtol = 1e-10
        end
    end

    @testset "8. Interval inputs enclose Float64 result" begin
        c0f = 1.7; c1f = -0.4; p = 4
        coeff_f = _PPL(c0f, c1f, p)
        c0i = interval(c0f, c0f); c1i = interval(c1f, c1f)
        coeff_i = _PPL(c0i, c1i, p)
        @test length(coeff_i) == length(coeff_f)
        for k in eachindex(coeff_f)
            @test inf(coeff_i[k]) ≤ coeff_f[k] ≤ sup(coeff_i[k])
        end
    end

    @testset "9. performance budget p=50 < 50µs" begin
        c0, c1 = 1.1, 0.9
        # Warm-up & compile.
        _PPL(c0, c1, 50)
        N = 1000
        t0 = time_ns()
        @inbounds for _ in 1:N
            _PPL(c0, c1, 50)
        end
        tns = (time_ns() - t0) / N
        @test tns < 50_000     # 50 µs per call
    end
end
