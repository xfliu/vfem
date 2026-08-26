# test/test_tri_polar_sing_moment_1_over_r2.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. Right triangle (0,0),(1,0),(0,1), 1 ≤ m+n ≤ 4
#   2. 60° corner triangle
#   3. Narrow triangle
#   4. Triangle far from origin
#   5. Tiny triangle near origin
#   6. m+n = 0 -> DomainError (divergent integral)
#   7. m+n > 4 -> DomainError
#   8. Negative m or n -> DomainError
#   9. r1 ≤ 0 / r2 ≤ 0 / t2 ≤ t1 / span ≥ π -> DomainError
#  10. Interval inputs enclose Float64 result
#  11. Performance: < 10 µs per call
#
# MATHEMATICAL CONTRACT (docs/testing-contract.md (c)):
#   For each (m, n) with 1 ≤ m+n ≤ 4 and the panel of triangles, the
#   closed-form result matches `ref_polar_inv_r2` (independent Gauss
#   reference) to relative tolerance 1e-8. Recurrence-based K_k vs
#   Gauss τ-integration ensures the two implementations are independent.

using Test, Random
using IntervalArithmetic: Interval, interval, inf, sup

const _F2 = VFEM.tri_polar_sing_moment_1_over_r2

const POLAR_TRIANGLES_INVR2 = [
    (1.0, 0.0,  1.0, π/2,  "right tri 1×1"),
    (1.0, 0.0,  1.0, π/3,  "60°"),
    (0.5, π/6,  0.3, π/6 + π/8, "narrow"),
    (2.0, π/4,  3.0, π/4 + π/4, "far"),
    (0.05, 0.1, 0.08, 0.1 + π/6, "small near origin"),
]

@testset "tri_polar_sing_moment_1_over_r2" begin
    Random.seed!(20260504)

    @testset "1–5. closed-form vs Gauss, 1 ≤ m+n ≤ 4" begin
        for (r1, t1, r2, t2, label) in POLAR_TRIANGLES_INVR2
            for m in 0:4, n in 0:(4 - m)
                m + n == 0 && continue
                got = _F2(r1, t1, r2, t2, m, n)
                ref = ref_polar_inv_r2(r1, t1, r2, t2, m, n; ngauss = 80)
                rel = abs(got - ref) / max(abs(ref), 1e-15)
                @test rel < 1e-8
            end
        end
    end

    @testset "DomainError preconditions" begin
        @test_throws DomainError _F2(1.0, 0.0, 1.0, π/3, 0, 0)        # m+n = 0
        @test_throws DomainError _F2(1.0, 0.0, 1.0, π/3, 5, 0)        # m+n > 4
        @test_throws DomainError _F2(1.0, 0.0, 1.0, π/3, -1, 1)
        @test_throws DomainError _F2(1.0, 0.0, 1.0, π/3, 1, -1)
        @test_throws DomainError _F2(0.0, 0.0, 1.0, π/3, 1, 0)
        @test_throws DomainError _F2(1.0, 0.0, 0.0, π/3, 1, 0)
        @test_throws DomainError _F2(1.0, 0.5, 1.0, 0.5, 1, 0)
        @test_throws DomainError _F2(1.0, 0.0, 1.0, π,   1, 0)
    end

    @testset "10. interval enclosure" begin
        for (r1, t1, r2, t2, _) in POLAR_TRIANGLES_INVR2
            for (m, n) in ((1, 0), (1, 1), (2, 1), (2, 2), (3, 1))
                got_f = _F2(r1, t1, r2, t2, m, n)
                got_i = _F2(interval(r1, r1), interval(t1, t1),
                            interval(r2, r2), interval(t2, t2), m, n)
                @test inf(got_i) ≤ got_f ≤ sup(got_i)
            end
        end
    end

    @testset "11. performance budget < 10 µs" begin
        _F2(1.0, 0.0, 1.0, π/3, 2, 2)
        N = 2000
        t0 = time_ns()
        @inbounds for _ in 1:N
            _F2(1.0, 0.0, 1.0, π/3, 2, 2)
        end
        tns = (time_ns() - t0) / N
        @test tns < 10_000
    end
end
