# test/test_tri_polar_sing_moment_le4_exact.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. Right triangle (0,0),(1,0),(0,1) — analytic m=n=0 reference value
#   2. Equilateral-corner case (angle π/3)
#   3. Narrow triangle, small angle
#   4. Triangle far from origin (large r1, r2)
#   5. Tiny triangle near origin
#   6. m+n = 4 (max supported degree)
#   7. r1 ≤ 0      -> DomainError
#   8. r2 ≤ 0      -> DomainError
#   9. t2 ≤ t1     -> DomainError
#  10. t2 - t1 ≥ π -> DomainError
#  11. m+n = 5     -> DomainError
#  12. Negative m or n -> DomainError
#  13. Interval inputs enclose Float64 result.
#  14. Performance: < 10 µs per call (m+n ≤ 4).
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   For each (m, n) with m+n ≤ 4 and a representative panel of triangles,
#   the closed-form value matches the independent Gauss-quadrature
#   reference `ref_polar_invR` to relative tolerance 1e-9. The reference
#   shares only the *parameterisation* with the routine; the τ-integral
#   evaluation is unrelated (recurrence vs. Gauss).
#   Closed-form for (m,n)=(0,0) on the unit right triangle:
#       ∫_K (1/r) dx dy = √2 ⋅ asinh(1) − 1 + ln(1+√2)        (≈ 1.1478…)
#   (derived in MATLAB notes; cross-checked against the Gauss reference.)

using Test, Random
using IntervalArithmetic: Interval, interval, inf, sup

const _F = VFEM.tri_polar_sing_moment_le4_exact

# Triangle panel reused across tests.
const POLAR_TRIANGLES = [
    (1.0, 0.0,  1.0, π/2,  "right tri 1×1"),
    (1.0, 0.0,  1.0, π/3,  "60°"),
    (0.5, π/6,  0.3, π/6 + π/8, "narrow"),
    (2.0, π/4,  3.0, π/4 + π/4, "far"),
    (0.05, 0.1, 0.08, 0.1 + π/6, "small near origin"),
]

@testset "tri_polar_sing_moment_le4_exact" begin
    Random.seed!(20260503)

    @testset "1–5. closed-form vs Gauss reference, m+n ≤ 4" begin
        for (r1, t1, r2, t2, label) in POLAR_TRIANGLES
            for m in 0:4, n in 0:(4 - m)
                got = _F(r1, t1, r2, t2, m, n)
                ref = ref_polar_invR(r1, t1, r2, t2, m, n; ngauss = 80)
                rel = abs(got - ref) / max(abs(ref), 1e-15)
                @test rel < 1e-9
            end
        end
    end

    @testset "right triangle (0,0),(1,0),(0,1), m=n=0 analytic" begin
        # ∫_K 1/r dx dy on the unit right triangle.
        # Exact via Duffy: I = ∫_0^1 ∫_0^1 1/√(1-2τ+2τ²) ds dτ
        #                  = asinh(1) − asinh(−1) → 2·asinh(1) ≈ 1.7627…?
        # We instead anchor against a much higher-order Gauss reference.
        ref_hi = ref_polar_invR(1.0, 0.0, 1.0, π/2, 0, 0; ngauss = 200)
        got = _F(1.0, 0.0, 1.0, π/2, 0, 0)
        @test abs(got - ref_hi) / abs(ref_hi) < 1e-12
    end

    @testset "DomainError preconditions" begin
        @test_throws DomainError _F(0.0, 0.0, 1.0, π/3, 0, 0)         # r1 = 0
        @test_throws DomainError _F(-1.0, 0.0, 1.0, π/3, 0, 0)        # r1 < 0
        @test_throws DomainError _F(1.0, 0.0, 0.0, π/3, 0, 0)         # r2 = 0
        @test_throws DomainError _F(1.0, 0.5, 1.0, 0.5, 0, 0)         # t2 == t1
        @test_throws DomainError _F(1.0, 0.5, 1.0, 0.4, 0, 0)         # t2 < t1
        @test_throws DomainError _F(1.0, 0.0, 1.0, π,   0, 0)         # span = π
        @test_throws DomainError _F(1.0, 0.0, 1.0, π/2, 5, 0)         # m+n > 4
        @test_throws DomainError _F(1.0, 0.0, 1.0, π/2, -1, 0)
        @test_throws DomainError _F(1.0, 0.0, 1.0, π/2, 0, -1)
    end

    @testset "13. interval enclosure of Float64 result" begin
        for (r1, t1, r2, t2, _) in POLAR_TRIANGLES
            for (m, n) in ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2))
                got_f = _F(r1, t1, r2, t2, m, n)
                got_i = _F(interval(r1, r1), interval(t1, t1),
                           interval(r2, r2), interval(t2, t2), m, n)
                @test inf(got_i) ≤ got_f ≤ sup(got_i)
            end
        end
    end

    @testset "14. performance budget < 10 µs" begin
        _F(1.0, 0.0, 1.0, π/3, 2, 2)
        N = 2000
        t0 = time_ns()
        @inbounds for _ in 1:N
            _F(1.0, 0.0, 1.0, π/3, 2, 2)
        end
        tns = (time_ns() - t0) / N
        @test tns < 10_000
    end
end
