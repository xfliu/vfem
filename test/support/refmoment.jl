# test/support/refmoment.jl
#
# Independent reference values for triangle singular moments, computed
# from a high-order Gauss–Legendre quadrature on the parameter
# (s, τ) ∈ [0,1]² using the same Duffy-type parameterisation that the
# closed forms are derived from. The two implementations differ in
# everything except the parameterisation: closed-form J_k recurrence
# and atan-based K_0 in the routine, vs. tensor Gauss quadrature here.
#
# This is an independent oracle for tests, not part of the public API.

using LinearAlgebra: SymTridiagonal, eigen

"""
    gauss_legendre_01(n) -> (x, w)

Gauss–Legendre nodes and weights on `[0, 1]`.
"""
function gauss_legendre_01(n::Integer)
    β = [k / sqrt(4k * k - 1) for k in 1:(n - 1)]
    T = SymTridiagonal(zeros(n), β)
    F = eigen(T)
    x0 = F.values
    w0 = 2 .* (F.vectors[1, :]) .^ 2
    return 0.5 .* (x0 .+ 1), 0.5 .* w0
end

"""
    ref_polar_invR(r1, t1, r2, t2, m, n; ngauss=80) -> Float64

Reference value of `∫_K (1/r) x^m y^n dxdy` over the polar triangle,
via the s-then-τ split of the Duffy parameterisation. The s-integral
of `s^(m+n)` is exact (= 1/(m+n+1)); the τ-integral of
`wx^m wy^n / sqrt(q(τ))` is computed by an `ngauss`-point Gauss rule.
"""
function ref_polar_invR(r1, t1, r2, t2, m::Integer, n::Integer; ngauss::Integer = 80)
    x1 = r1 * cos(t1); y1 = r1 * sin(t1)
    x2 = r2 * cos(t2); y2 = r2 * sin(t2)
    dx = x2 - x1; dy = y2 - y1
    det12 = x1 * y2 - y1 * x2
    ξ, w = gauss_legendre_01(ngauss)
    s = 0.0
    @inbounds for k in eachindex(ξ)
        τ = ξ[k]
        wx = x1 + dx * τ
        wy = y1 + dy * τ
        q = wx * wx + wy * wy
        s += w[k] * wx^m * wy^n / sqrt(q)
    end
    return det12 / (m + n + 1) * s
end

"""
    ref_polar_inv_r2(r1, t1, r2, t2, m, n; ngauss=80) -> Float64

Reference value of `∫_K (1/r²) x^m y^n dxdy` over the polar triangle,
via the same Duffy split (s-integral = 1/(m+n) for `m+n ≥ 1`).
"""
function ref_polar_inv_r2(r1, t1, r2, t2, m::Integer, n::Integer; ngauss::Integer = 80)
    @assert m + n ≥ 1
    x1 = r1 * cos(t1); y1 = r1 * sin(t1)
    x2 = r2 * cos(t2); y2 = r2 * sin(t2)
    dx = x2 - x1; dy = y2 - y1
    det12 = x1 * y2 - y1 * x2
    ξ, w = gauss_legendre_01(ngauss)
    s = 0.0
    @inbounds for k in eachindex(ξ)
        τ = ξ[k]
        wx = x1 + dx * τ
        wy = y1 + dy * τ
        q = wx * wx + wy * wy
        s += w[k] * wx^m * wy^n / q
    end
    return det12 / (m + n) * s
end
