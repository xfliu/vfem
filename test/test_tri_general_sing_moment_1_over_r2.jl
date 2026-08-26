# test/test_tri_general_sing_moment_1_over_r2.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. Triangle with vertex at origin: dispatch must match polar closed form.
#   2. General triangle, origin outside K (not strictly inside) — Gauss check.
#   3. Triangle with reversed orientation gives the same value.
#   4. m + n ∈ {1, 2, 3, 4} sweep across the panel.
#   5. m + n = 0    -> DomainError.
#   6. m + n > 4    -> DomainError.
#   7. Negative m or n -> DomainError.
#   8. Interval inputs enclose Float64 result.
#   9. Performance: < 30 µs per call.
#
# MATHEMATICAL CONTRACT (docs/testing-contract.md (c)):
#   For triangles whose origin is *outside* K, agree with high-order
#   Gauss quadrature on the physical triangle to relative tol 1e-7.
#   For triangles with a vertex at the origin, match the polar
#   closed-form 1/r² routine to within 1e-12 (same path, just dispatched).

using Test, Random
using IntervalArithmetic: Interval, interval, inf, sup

const _G2 = VFEM.tri_general_sing_moment_1_over_r2
const _P2 = VFEM.tri_polar_sing_moment_1_over_r2

function _tri_gauss_invr2(verts::NTuple{3, NTuple{2, Float64}}, m::Integer, n::Integer;
                          ngauss::Integer = 60)
    v1, v2, v3 = verts
    Jdet = (v2[1] - v1[1]) * (v3[2] - v1[2]) -
           (v3[1] - v1[1]) * (v2[2] - v1[2])
    area2 = abs(Jdet)
    ξ, wξ = gauss_legendre_01(ngauss)
    η, wη = gauss_legendre_01(ngauss)
    s = 0.0
    @inbounds for i in eachindex(ξ), j in eachindex(η)
        λ1 = 1 - ξ[i]
        λ2 = ξ[i] * (1 - η[j])
        λ3 = ξ[i] * η[j]
        x = λ1 * v1[1] + λ2 * v2[1] + λ3 * v3[1]
        y = λ1 * v1[2] + λ2 * v2[2] + λ3 * v3[2]
        r2 = x * x + y * y
        s += wξ[i] * wη[j] * ξ[i] * x^m * y^n / r2
    end
    return s * area2
end

@testset "tri_general_sing_moment_1_over_r2" begin
    Random.seed!(20260506)

    @testset "1. vertex-at-origin matches polar closed form" begin
        r1, t1, r2, t2 = 1.0, 0.0, 1.0, π/3
        x1 = r1 * cos(t1); y1 = r1 * sin(t1)
        x2 = r2 * cos(t2); y2 = r2 * sin(t2)
        for m in 0:4, n in 0:(4 - m)
            m + n == 0 && continue
            ref = _P2(r1, t1, r2, t2, m, n)
            got = _G2(0.0, 0.0, x1, y1, x2, y2, m, n)
            @test abs(got - ref) / max(abs(ref), 1e-15) < 1e-12
        end
    end

    @testset "2. origin outside K, Gauss reference" begin
        verts = (
            ((2.0, 1.0), (3.0, 1.5), (2.5, 2.0)),
            ((5.0, 3.0), (6.0, 3.0), (5.5, 4.0)),
        )
        for V in verts, m in 0:4, n in 0:(4 - m)
            m + n == 0 && continue
            got = _G2(V[1]..., V[2]..., V[3]..., m, n)
            ref = _tri_gauss_invr2(V, m, n; ngauss = 60)
            rel = abs(got - ref) / max(abs(ref), 1e-15)
            @test rel < 1e-7
        end
    end

    @testset "3. orientation flip negates result (faithful to MATLAB)" begin
        # Unlike the 1/r general routine, the 1/r² general routine uses an
        # interior-angle decomposition with explicit `sgn = sign(det12)`.
        # Reversing input vertex order flips det12 for each consecutive
        # edge, so each wedge's contribution flips sign and the total
        # negates. This matches matlab_lib/tri_general_sing_moment_1_over_r2.m
        # exactly — callers are expected to pass CCW vertices to get the
        # signed integral with the conventional positive sign.
        V1 = (2.0, 1.0); V2 = (3.0, 1.5); V3 = (2.5, 2.0)
        for (m, n) in ((1, 1), (2, 1), (3, 0), (1, 3))
            a = _G2(V1..., V2..., V3..., m, n)
            b = _G2(V1..., V3..., V2..., m, n)
            @test a ≈ -b atol = 1e-13 rtol = 1e-12
        end
    end

    @testset "5–7. DomainError preconditions" begin
        @test_throws DomainError _G2(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0, 0)
        @test_throws DomainError _G2(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5, 0)
        @test_throws DomainError _G2(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1, 1)
        @test_throws DomainError _G2(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1, -1)
    end

    @testset "8. interval enclosure" begin
        V1 = (2.0, 1.0); V2 = (3.0, 1.5); V3 = (2.5, 2.0)
        for (m, n) in ((1, 0), (1, 1), (2, 2))
            got_f = _G2(V1..., V2..., V3..., m, n)
            got_i = _G2(interval(V1[1], V1[1]), interval(V1[2], V1[2]),
                        interval(V2[1], V2[1]), interval(V2[2], V2[2]),
                        interval(V3[1], V3[1]), interval(V3[2], V3[2]),
                        m, n)
            @test inf(got_i) ≤ got_f ≤ sup(got_i)
        end
    end

    @testset "9. performance budget < 30 µs" begin
        _G2(2.0, 1.0, 3.0, 1.5, 2.5, 2.0, 2, 2)
        N = 1000
        t0 = time_ns()
        @inbounds for _ in 1:N
            _G2(2.0, 1.0, 3.0, 1.5, 2.5, 2.0, 2, 2)
        end
        tns = (time_ns() - t0) / N
        @test tns < 30_000
    end
end
