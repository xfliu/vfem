# test/test_tri_general_sing_moment_le4_exact.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. Triangle with one vertex at the origin — must equal the polar
#      closed-form on the same triangle.
#   2. General triangle, origin outside (panel of cases).
#   3. Triangle with vertex order reversed (CW vs CCW) — same value
#      (signed decomposition handles orientation).
#   4. m + n in {0, 1, 2, 3, 4}.
#   5. m + n > 4 -> DomainError.
#   6. Negative m or n -> DomainError.
#   7. Interval inputs enclose Float64 result.
#   8. Performance: < 30 µs per call (3× the polar routine).
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   Decomposition matches a high-order Dunavant-type Gauss quadrature
#   on the *physical* triangle for the non-singular case (origin
#   strictly outside K). For the origin-vertex case, decomposition
#   matches the polar closed-form exactly (same path, just dispatched).

using Test, Random
using IntervalArithmetic: Interval, interval, inf, sup

const _G = VFEM.tri_general_sing_moment_le4_exact
const _P = VFEM.tri_polar_sing_moment_le4_exact

# 2D tensor Gauss–Legendre over a triangle via the conical map
# (λ1, λ2, λ3) = (1−ξ, ξ(1−η), ξη), Jacobian ξ on [0,1]².
function _tri_gauss_invR(verts::NTuple{3, NTuple{2, Float64}}, m::Integer, n::Integer;
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
        r = hypot(x, y)
        s += wξ[i] * wη[j] * ξ[i] * x^m * y^n / r
    end
    return s * area2
end

@testset "tri_general_sing_moment_le4_exact" begin
    Random.seed!(20260505)

    @testset "1. vertex-at-origin matches polar closed form" begin
        # General-triangle dispatch with one vertex at origin must equal
        # the polar closed form on the same triangle.
        r1, t1, r2, t2 = 1.0, 0.0, 1.0, π/3
        x1 = r1 * cos(t1); y1 = r1 * sin(t1)
        x2 = r2 * cos(t2); y2 = r2 * sin(t2)
        for m in 0:4, n in 0:(4 - m)
            ref = _P(r1, t1, r2, t2, m, n)
            got = _G(0.0, 0.0, x1, y1, x2, y2, m, n)
            @test abs(got - ref) / max(abs(ref), 1e-15) < 1e-12
        end
    end

    @testset "2. origin outside K vs Gauss reference" begin
        verts = (
            ((2.0, 1.0), (3.0, 1.5), (2.5, 2.0)),
            ((5.0, 3.0), (6.0, 3.0), (5.5, 4.0)),
            ((1.0, 1.0), (2.0, 0.7), (1.5, 2.5)),
        )
        for V in verts, m in 0:4, n in 0:(4 - m)
            got = _G(V[1]..., V[2]..., V[3]..., m, n)
            ref = _tri_gauss_invR(V, m, n; ngauss = 60)
            rel = abs(got - ref) / max(abs(ref), 1e-15)
            @test rel < 1e-7
        end
    end

    @testset "3. orientation flip negates result" begin
        # With the fixed shoelace decomposition (using the *original*
        # det12 sign per edge), reversing the input vertex order flips
        # the sign of every edge's orig_det12 and hence the sign of the
        # total. Callers should pass CCW-ordered vertices to get the
        # conventional positive integral. Origin-inside-K is out of scope
        # — the polar wedges would wrap past π and trip the precondition.
        for (V1, V2, V3) in (
                ((2.0, 1.0), (3.0, 1.5), (2.5, 2.0)),
            )
            for (m, n) in ((0, 0), (2, 1), (1, 3), (4, 0))
                a = _G(V1..., V2..., V3..., m, n)
                b = _G(V1..., V3..., V2..., m, n)   # swap orientation
                @test a ≈ -b atol = 1e-13 rtol = 1e-12
            end
        end
    end

    @testset "5–6. DomainError preconditions" begin
        @test_throws DomainError _G(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5, 0)
        @test_throws DomainError _G(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1, 0)
        @test_throws DomainError _G(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0, -1)
    end

    @testset "7. interval enclosure" begin
        V1 = (2.0, 1.0); V2 = (3.0, 1.5); V3 = (2.5, 2.0)
        for (m, n) in ((0, 0), (1, 1), (2, 2))
            got_f = _G(V1..., V2..., V3..., m, n)
            got_i = _G(interval(V1[1], V1[1]), interval(V1[2], V1[2]),
                       interval(V2[1], V2[1]), interval(V2[2], V2[2]),
                       interval(V3[1], V3[1]), interval(V3[2], V3[2]),
                       m, n)
            @test inf(got_i) ≤ got_f ≤ sup(got_i)
        end
    end

    @testset "8. performance budget < 30 µs" begin
        _G(2.0, 1.0, 3.0, 1.5, 2.5, 2.0, 2, 2)
        N = 1000
        t0 = time_ns()
        @inbounds for _ in 1:N
            _G(2.0, 1.0, 3.0, 1.5, 2.5, 2.0, 2, 2)
        end
        tns = (time_ns() - t0) / N
        @test tns < 30_000
    end
end
