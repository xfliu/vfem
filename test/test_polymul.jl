# test/test_polymul.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. (length 1) × (length 1) — scalar product
#   2. (length 1) × (length k) — pure scalar scaling
#   3. (length 2) × (length 2) — known product (1+t)(1+t) = 1+2t+t²
#   4. Asymmetric sizes (m=3, n=5)
#   5. Float64 inputs vs Integer inputs (promotion sanity)
#   6. Interval inputs enclose Float64 result entry-wise
#   7. Performance: 21×21 product < 25 µs.
#
# MATHEMATICAL CONTRACT (docs/testing-contract.md (c)):
#   For random polynomials p, q and random t,
#       _horner(polymul(p, q), t) ≈ _horner(p, t) * _horner(q, t).

using Test, Random
using IntervalArithmetic: Interval, interval, inf, sup
const _PMUL = VFEM.polymul

function _horner_pm(coeff::AbstractVector, t)
    s = zero(promote_type(eltype(coeff), typeof(t)))
    for k in length(coeff):-1:1
        s = s * t + coeff[k]
    end
    return s
end

@testset "polymul" begin
    Random.seed!(20260502)

    @testset "1. scalar × scalar" begin
        @test _PMUL([3.0], [4.0]) == [12.0]
    end

    @testset "2. scalar × vector" begin
        @test _PMUL([2.0], [1.0, 0.5, -0.25]) == [2.0, 1.0, -0.5]
    end

    @testset "3. (1+t)(1+t) = 1 + 2t + t²" begin
        @test _PMUL([1.0, 1.0], [1.0, 1.0]) == [1.0, 2.0, 1.0]
    end

    @testset "4. asymmetric sizes" begin
        p = [1.0, -1.0, 2.0]              # 1 - t + 2t²  (deg 2)
        q = [3.0, 0.0, -1.0, 0.5, 1.0]    # 3 - t² + 0.5t³ + t⁴  (deg 4)
        r = _PMUL(p, q)
        @test length(r) == 7              # deg 2+4 = 6  -> length 7
        # Verify via Horner identity at several points.
        for t in (-1.3, 0.0, 0.4, 1.7)
            @test _horner_pm(r, t) ≈ _horner_pm(p, t) * _horner_pm(q, t)
        end
    end

    @testset "5. integer + float promotion" begin
        r = _PMUL([1, 2, 3], [4.0, 5.0])
        @test eltype(r) === Float64
        @test r ≈ [4.0, 13.0, 22.0, 15.0]
    end

    @testset "Mathematical contract" begin
        for _ in 1:30
            p = randn(rand(1:6)); q = randn(rand(1:6)); t = randn()
            @test _horner_pm(_PMUL(p, q), t) ≈ _horner_pm(p, t) * _horner_pm(q, t) atol=1e-10 rtol=1e-10
        end
    end

    @testset "6. interval enclosure" begin
        p = [1.7, -0.4, 0.2]
        q = [-0.5, 1.1]
        r_f = _PMUL(p, q)
        p_i = interval.(p, p); q_i = interval.(q, q)
        r_i = _PMUL(p_i, q_i)
        for k in eachindex(r_f)
            @test inf(r_i[k]) ≤ r_f[k] ≤ sup(r_i[k])
        end
    end

    @testset "7. performance budget 21×21 < 25 µs" begin
        p = randn(21); q = randn(21)
        _PMUL(p, q)              # warm
        N = 200
        t0 = time_ns()
        @inbounds for _ in 1:N
            _PMUL(p, q)
        end
        tns = (time_ns() - t0) / N
        @test tns < 25_000
    end
end
