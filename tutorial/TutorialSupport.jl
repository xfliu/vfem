# tutorial/TutorialSupport.jl
#
# ============================================================================
#  TutorialSupport — the tutorial's own scaffolding around VFEM.jl
# ============================================================================
#
#  VFEM.jl is an *eigenvalue* library. It gives you assembled stiffness and
#  mass matrices and verified eigenvalue bounds, and it deliberately stops
#  there: there is no load vector, no Dirichlet lifting, no error norm, and no
#  plotting. Those pieces belong to the *application*, not the library.
#
#  This module supplies exactly those pieces, so the tutorial can pose and
#  solve honest boundary-value problems
#
#       -Delta u = f   in Omega,        u = g   on dOmega
#
#  on top of the library's own assembly. Nothing here reimplements anything
#  VFEM.jl already does: the stiffness and mass matrices always come from
#  `VFEM.lagrange_laplace_matrices`, and the triangle quadrature defaults to
#  the library's exported `VFEM.dunavant_rule_6`.
#
#  Dependencies: VFEM, LinearAlgebra, SparseArrays, DelimitedFiles, Printf.
#  Nothing else — in particular NO plotting package. Figures are hand-written
#  SVG (project decision D-10).
#
# ----------------------------------------------------------------------------
#  THE ONE THING YOU MUST UNDERSTAND BEFORE READING FURTHER
# ----------------------------------------------------------------------------
#
#  `lagrange_laplace_matrices(m, p)` does NOT use the nodal Lagrange basis.
#  It uses the *monomial* barycentric basis
#
#       phi_alpha = L1^i * L2^j * L3^k ,      alpha = (i,j,k),  i+j+k = p
#
#  (see the header comment of `src/core/eigensolve2d/laplace_eig_lagrange.jl`:
#  this is the basis that `rt_hdiv_problem` consumes, so the Lehmann-Goerisch
#  pipeline stays self-consistent). VFEM.jl also ships a *nodal* Lagrange
#  assembly, `create_matrix_lagrange`, used by the CECR pipeline — the two are
#  different matrices for the same space. Eigenvalues agree; coefficient
#  vectors do not.
#
#  Consequence for us: a coefficient vector `uh` returned by `solve_poisson`
#  is NOT a list of nodal values. For p = 2 the local basis is
#
#       (L1^2, L2^2, L3^2, L2*L3, L1*L3, L1*L2)
#
#  and e.g. the value of u_h at vertex 1 is just the L1^2 coefficient (the
#  other five basis functions vanish there), while the value at the midpoint
#  of edge (v1,v2) is (c_{L1^2} + c_{L2^2})/4 + c_{L1L2}/4. Every routine in
#  this module that has to *evaluate* a finite element function goes through
#  `_local_basis_values`, and every routine that has to *build* a coefficient
#  vector from point values goes through `interpolate_nodal`. Do not hand-roll
#  either one.
#
#  The exact DOF numbering, transcribed from `_cg_lagrange_local_to_global`
#  in `src/core/eigensolve2d/rt_hdiv_problem.jl`, is:
#
#     p = 1 : dof(v)    = v                     for v in 1:nv
#     p = 2 : dof(v)    = v                     for v in 1:nv
#             dof(edge) = nv + edge_id          for edge_id in 1:ne
#
#  and, per element k, local index -> global DOF is
#
#     p = 1 : (v1, v2, v3)
#     p = 2 : (v1, v2, v3,
#              nv + tri2edge[k,1],       # local basis 4 = L2*L3, edge (v2,v3)
#              nv + tri2edge[k,2],       # local basis 5 = L1*L3, edge (v1,v3)
#              nv + tri2edge[k,3])       # local basis 6 = L1*L2, edge (v1,v2)
#
#  This pairing is not taken on faith. `verify_dof_ordering` below reassembles
#  the stiffness and mass matrices from scratch using the map above and
#  compares them against `lagrange_laplace_matrices`; the smoke test calls it
#  and it agrees to ~1e-15 relative. If you ever change p, run it again.
#
# ----------------------------------------------------------------------------
#  QUADRATURE, AND A CORRECTION TO A NATURAL MISREADING
# ----------------------------------------------------------------------------
#
#  `VFEM.dunavant_rule_6()` returns `(lambda, w)` with `lambda[q, :]` the
#  barycentric coordinates of point q and `w[q]` its weight, the weights
#  summing to 1/2 (the area of the reference triangle). So
#
#       integral over K of h  =  2*|K| * sum_q w[q] * h(x_q)
#
#  The "6" is the POINT COUNT, not the degree: the rule is exact for total
#  degree <= 4. That is ample for a load vector (P2 needs only degree 2 for
#  optimal order) but it is *not* generous for measuring an L2 error, whose
#  integrand for P2 is degree 4 in u_h alone before you even account for the
#  smooth exact solution. Error norms therefore default to `tri_quad_rule(8)`,
#  a 64-point tensor-Gauss/Duffy rule exact to total degree 14, generated here
#  with the same `(lambda, w)` convention so the two are drop-in compatible.
#
# ============================================================================

module TutorialSupport

using VFEM: Mesh2D, dunavant_rule_6, lagrange_laplace_matrices, find_mesh_hmax
using LinearAlgebra: norm, dot
using SparseArrays: SparseMatrixCSC, spzeros, sparse
using DelimitedFiles: writedlm
using Printf: @sprintf, @printf

export
    # quadrature
    tri_quad_rule,
    # DOF bookkeeping
    lagrange_ndof, lagrange_monomial_exponents, lagrange_local_dofs,
    verify_dof_ordering,
    # assembly / solve
    assemble_load_vector, interpolate_nodal, PoissonSolution, solve_poisson,
    # post-processing
    l2_error, h1_seminorm_error, observed_order,
    fe_vertex_values, mesh_hmax,
    # meshes (fallback; the Meshes track owns the canonical generator)
    mesh2d_from_nodes_elements, tutorial_square_mesh,
    # figures
    svg_mesh, svg_solution, svg_loglog

# ============================================================================
#  1. Quadrature
# ============================================================================

"""
    _gauss_legendre(n) -> (x, w)

`n`-point Gauss-Legendre nodes and weights on `[0, 1]`, weights summing to 1.
Computed by Newton's method on the Legendre polynomial `P_n` using the
three-term recurrence, so there is no table to mistranscribe and no
dependency to add.
"""
function _gauss_legendre(n::Integer)
    n >= 1 || throw(ArgumentError("need n >= 1 Gauss points (got $n)"))
    x = zeros(Float64, n)
    w = zeros(Float64, n)
    for i in 1:n
        # Classical asymptotic starting guess for the i-th root of P_n.
        z = cos(pi * (i - 0.25) / (n + 0.5))
        p1 = 0.0
        dp = 1.0
        for _ in 1:100
            p0 = 1.0          # P_0
            p1 = z            # P_1
            for k in 2:n
                p2 = ((2k - 1) * z * p1 - (k - 1) * p0) / k
                p0 = p1
                p1 = p2
            end
            # p1 = P_n(z), p0 = P_{n-1}(z); derivative from the standard identity.
            dp = n * (z * p1 - p0) / (z * z - 1.0)
            dz = -p1 / dp
            z += dz
            abs(dz) < 1e-15 && break
        end
        # Nodes/weights on [-1,1], then affinely mapped to [0,1].
        x[i] = 0.5 * (z + 1.0)
        w[i] = 0.5 * (2.0 / ((1.0 - z * z) * dp * dp))
    end
    return x, w
end

"""
    tri_quad_rule(n::Integer) -> (lambda::Matrix{Float64}, w::Vector{Float64})

Symmetric-in-convention triangle rule with `n^2` points, exact for total
degree `<= 2n - 2`. Built by the Duffy (collapsed-square) map

    L2 = u,   L3 = v*(1 - u),   L1 = (1 - u)*(1 - v),   jacobian = (1 - u)

applied to a tensor Gauss-Legendre grid on `[0,1]^2`.

The return format matches `VFEM.dunavant_rule_6()` exactly: `lambda[q, :]` are
barycentric coordinates and `sum(w) == 1/2`, so

    integral over K of h  =  2*|K| * sum_q w[q] * h(x_q)

`n = 8` (64 points, degree 14) is the default for the error norms. Use
`dunavant_rule_6()` where degree 4 is enough and 6 points is cheaper.
"""
function tri_quad_rule(n::Integer)
    gx, gw = _gauss_legendre(n)
    nq = n * n
    lambda = Matrix{Float64}(undef, nq, 3)
    w = Vector{Float64}(undef, nq)
    q = 1
    for a in 1:n, b in 1:n
        u = gx[a]
        v = gx[b]
        lambda[q, 1] = (1.0 - u) * (1.0 - v)   # L1
        lambda[q, 2] = u                       # L2
        lambda[q, 3] = v * (1.0 - u)           # L3
        w[q] = gw[a] * gw[b] * (1.0 - u)
        q += 1
    end
    return lambda, w
end

# ============================================================================
#  2. DOF bookkeeping for the monomial Lagrange basis
# ============================================================================

"""
    lagrange_ndof(m::Mesh2D, p::Integer) -> Int

Global dimension of the conforming Lagrange space of order `p` in the
numbering used by `VFEM.lagrange_laplace_matrices`:
`nv + (p-1)*ne + max(0, (p-1)(p-2)/2)*nt`. For `p = 1` this is `m.nv`; for
`p = 2` it is `m.nv + m.ne`.
"""
function lagrange_ndof(m::Mesh2D, p::Integer)
    p >= 1 || throw(ArgumentError("Lagrange order must be >= 1 (got $p)"))
    return m.nv + (p - 1) * m.ne + max(0, (p - 1) * (p - 2) ÷ 2) * m.nt
end

"""
    lagrange_monomial_exponents(p::Integer) -> Matrix{Int}

`nbasis x 3` table of barycentric exponents `(i, j, k)` of the local basis
functions `phi = L1^i L2^j L3^k`, in the order VFEM.jl's `_Lagrange_basis`
produces them. `p = 1` gives `(L1, L2, L3)`; `p = 2` gives
`(L1^2, L2^2, L3^2, L2*L3, L1*L3, L1*L2)`.

Only `p = 1, 2` are supported here — that is what `solve_poisson` supports,
and it is where the local-to-global map below is verified.
"""
function lagrange_monomial_exponents(p::Integer)
    if p == 1
        return [1 0 0;
                0 1 0;
                0 0 1]
    elseif p == 2
        return [2 0 0;      # 1  L1^2      <-> vertex 1
                0 2 0;      # 2  L2^2      <-> vertex 2
                0 0 2;      # 3  L3^2      <-> vertex 3
                0 1 1;      # 4  L2*L3     <-> edge opposite vertex 1
                1 0 1;      # 5  L1*L3     <-> edge opposite vertex 2
                1 1 0]      # 6  L1*L2     <-> edge opposite vertex 3
    else
        throw(ArgumentError("TutorialSupport supports p = 1 or 2 (got $p)"))
    end
end

"""
    lagrange_local_dofs(m::Mesh2D, k::Integer, p::Integer) -> NTuple{N,Int}

Local-basis-index -> global-DOF map for element `k`, transcribed from
`_cg_lagrange_local_to_global` in `src/core/eigensolve2d/rt_hdiv_problem.jl`.

    p = 1 : (v1, v2, v3)
    p = 2 : (v1, v2, v3, nv + tri2edge[k,1], nv + tri2edge[k,2], nv + tri2edge[k,3])

Recall `tri2edge[k, j]` is the edge *opposite* local vertex `j`, which is why
local basis 4 (`L2*L3`, the function supported on edge (v2,v3)) pairs with
`tri2edge[k, 1]`. For `p = 2` there is one DOF per edge, so the edge
orientation branch in the library's map collapses and no sign or reversal
bookkeeping is needed; for `p >= 3` it would be.
"""
function lagrange_local_dofs(m::Mesh2D, k::Integer, p::Integer)
    v1 = m.elements[k, 1]
    v2 = m.elements[k, 2]
    v3 = m.elements[k, 3]
    if p == 1
        return (v1, v2, v3)
    elseif p == 2
        nv = m.nv
        return (v1, v2, v3,
                nv + m.tri2edge[k, 1],
                nv + m.tri2edge[k, 2],
                nv + m.tri2edge[k, 3])
    else
        throw(ArgumentError("TutorialSupport supports p = 1 or 2 (got $p)"))
    end
end

# ---- element geometry -------------------------------------------------------

# Vertex coordinates, signed 2*area (= det of the affine Jacobian), and the
# three constant barycentric gradients of element k.
#
#   B = [x2-x1  x3-x1; y2-y1  y3-y1],  L2 = xi, L3 = eta,  L1 = 1-xi-eta
#   grad L2 = row 1 of B^-1,  grad L3 = row 2 of B^-1,  grad L1 = -(gL2 + gL3)
#
# `detB` is signed (negative for clockwise triangles); quadrature uses
# `abs(detB)` while the gradients need the signed value.
@inline function _element_geometry(m::Mesh2D, k::Integer)
    v1 = m.elements[k, 1]; v2 = m.elements[k, 2]; v3 = m.elements[k, 3]
    x1 = m.nodes[v1, 1]; y1 = m.nodes[v1, 2]
    x2 = m.nodes[v2, 1]; y2 = m.nodes[v2, 2]
    x3 = m.nodes[v3, 1]; y3 = m.nodes[v3, 2]
    detB = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    gL2 = ((y3 - y1) / detB, (x1 - x3) / detB)
    gL3 = ((y1 - y2) / detB, (x2 - x1) / detB)
    gL1 = (-(gL2[1] + gL3[1]), -(gL2[2] + gL3[2]))
    return (x1, y1, x2, y2, x3, y3), detB, (gL1, gL2, gL3)
end

# Values of all local basis functions at one barycentric point.
@inline function _local_basis_values!(vals::Vector{Float64},
                                      expo::Matrix{Int},
                                      L1::Float64, L2::Float64, L3::Float64)
    @inbounds for i in 1:size(expo, 1)
        vals[i] = L1^expo[i, 1] * L2^expo[i, 2] * L3^expo[i, 3]
    end
    return vals
end

# Physical gradients of all local basis functions at one barycentric point.
# d(phi)/dx = sum_d (d phi / d L_d) * (grad L_d)_x, with
# d(L1^i L2^j L3^k)/dL1 = i * L1^(i-1) L2^j L3^k   (zero when i == 0).
@inline function _local_basis_grads!(gx::Vector{Float64}, gy::Vector{Float64},
                                     expo::Matrix{Int},
                                     gL::Tuple{Tuple{Float64,Float64},
                                               Tuple{Float64,Float64},
                                               Tuple{Float64,Float64}},
                                     L1::Float64, L2::Float64, L3::Float64)
    Lv = (L1, L2, L3)
    @inbounds for i in 1:size(expo, 1)
        sx = 0.0
        sy = 0.0
        for d in 1:3
            e = expo[i, d]
            e == 0 && continue
            # partial derivative w.r.t. L_d
            pd = Float64(e) * Lv[d]^(e - 1)
            for d2 in 1:3
                d2 == d && continue
                pd *= Lv[d2]^expo[i, d2]
            end
            sx += pd * gL[d][1]
            sy += pd * gL[d][2]
        end
        gx[i] = sx
        gy[i] = sy
    end
    return gx, gy
end

"""
    verify_dof_ordering(m::Mesh2D, p::Integer) -> NamedTuple

Independently reassemble the stiffness `A = (grad phi, grad psi)` and mass
`M = (phi, psi)` matrices by straight quadrature over the monomial basis,
using `lagrange_local_dofs` for the scatter, and compare against
`VFEM.lagrange_laplace_matrices(m, p)`.

Returns `(; ndof, relA, relM, ok)` where `relA`, `relM` are relative Frobenius
differences and `ok` is `true` when both are below `1e-11`.

This is the *empirical* proof that our DOF map and basis convention agree with
the library's. It is cheap, it is the single highest-value check in this
module, and the smoke test runs it before it trusts any solve. If it ever
fails, nothing downstream means anything.
"""
function verify_dof_ordering(m::Mesh2D, p::Integer)
    A_lib, M_lib, _ = lagrange_laplace_matrices(m, p)
    ndof = lagrange_ndof(m, p)
    expo = lagrange_monomial_exponents(p)
    nb = size(expo, 1)
    lam, wq = tri_quad_rule(6)          # degree 10: exact for p <= 2 here
    nq = length(wq)

    A = spzeros(Float64, ndof, ndof)
    M = spzeros(Float64, ndof, ndof)
    vals = zeros(Float64, nb); gx = zeros(Float64, nb); gy = zeros(Float64, nb)
    Aloc = zeros(Float64, nb, nb); Mloc = zeros(Float64, nb, nb)

    for k in 1:m.nt
        _, detB, gL = _element_geometry(m, k)
        w_geo = abs(detB)               # = 2*|K|
        fill!(Aloc, 0.0); fill!(Mloc, 0.0)
        for q in 1:nq
            L1 = lam[q, 1]; L2 = lam[q, 2]; L3 = lam[q, 3]
            _local_basis_values!(vals, expo, L1, L2, L3)
            _local_basis_grads!(gx, gy, expo, gL, L1, L2, L3)
            cw = w_geo * wq[q]
            for i in 1:nb, j in 1:nb
                Aloc[i, j] += cw * (gx[i] * gx[j] + gy[i] * gy[j])
                Mloc[i, j] += cw * vals[i] * vals[j]
            end
        end
        g = lagrange_local_dofs(m, k, p)
        for i in 1:nb, j in 1:nb
            A[g[i], g[j]] += Aloc[i, j]
            M[g[i], g[j]] += Mloc[i, j]
        end
    end

    relA = norm(Matrix(A - A_lib)) / max(norm(Matrix(A_lib)), eps())
    relM = norm(Matrix(M - M_lib)) / max(norm(Matrix(M_lib)), eps())
    return (; ndof, relA, relM, ok = (relA < 1e-11 && relM < 1e-11))
end

# ============================================================================
#  3. Load vector
# ============================================================================

"""
    assemble_load_vector(m::Mesh2D, f::Function, p::Integer;
                         rule = dunavant_rule_6()) -> Vector{Float64}

The `ndof`-vector `b[i] = integral over Omega of f * phi_i`, i.e. the right
hand side of `-Delta u = f` discretised in the conforming Lagrange space of
order `p`. `f` is called as `f(x, y)` and returns a scalar.

DOF numbering matches `VFEM.lagrange_laplace_matrices(m, p)` exactly (verified
by `verify_dof_ordering`), so `A` from the library and `b` from here may be
used in the same linear system.

`rule` is a `(lambda, w)` pair in the library's convention: barycentric rows
and weights summing to `1/2`, applied as `2*|K| * sum_q w_q h(x_q)`. The
default `dunavant_rule_6()` is 6 points, degree-4 exact — enough for optimal
order at `p <= 2` with a smooth `f`. Pass `rule = tri_quad_rule(8)` when `f`
oscillates on the scale of an element and you want the load vector itself to
stop being the error-dominating term.
"""
function assemble_load_vector(m::Mesh2D, f::Function, p::Integer;
                              rule = dunavant_rule_6())
    lam, wq = rule
    size(lam, 2) == 3 || throw(DimensionMismatch("rule lambda must have 3 columns"))
    length(wq) == size(lam, 1) ||
        throw(DimensionMismatch("rule weights and points disagree in length"))

    ndof = lagrange_ndof(m, p)
    expo = lagrange_monomial_exponents(p)
    nb = size(expo, 1)
    b = zeros(Float64, ndof)
    vals = zeros(Float64, nb)
    nq = length(wq)

    for k in 1:m.nt
        (x1, y1, x2, y2, x3, y3), detB, _ = _element_geometry(m, k)
        w_geo = abs(detB)                       # = 2*|K|
        g = lagrange_local_dofs(m, k, p)
        for q in 1:nq
            L1 = lam[q, 1]; L2 = lam[q, 2]; L3 = lam[q, 3]
            # Barycentric -> physical: x = L1*x1 + L2*x2 + L3*x3.
            xq = L1 * x1 + L2 * x2 + L3 * x3
            yq = L1 * y1 + L2 * y2 + L3 * y3
            fq = Float64(f(xq, yq))
            _local_basis_values!(vals, expo, L1, L2, L3)
            cw = w_geo * wq[q] * fq
            @inbounds for i in 1:nb
                b[g[i]] += cw * vals[i]
            end
        end
    end
    return b
end

# ============================================================================
#  4. Nodal interpolation into the monomial basis
# ============================================================================

"""
    interpolate_nodal(m::Mesh2D, g::Function, p::Integer) -> Vector{Float64}

Coefficient vector of the order-`p` Lagrange interpolant of `g(x, y)`, in the
*monomial* basis used by `lagrange_laplace_matrices`.

This needs care and is the second place (after the DOF map) where a plausible
guess is wrong. The monomial basis is not nodal, so you cannot just write
point values into the vector. Instead we solve the tiny nodal-to-monomial
change of basis once and reuse it:

* `p = 1`: `L1, L2, L3` *are* the nodal hat functions, so the coefficient at
  vertex `v` is simply `g` at that vertex. The change of basis is the identity.
* `p = 2`: the nodal basis is
  `psi_v = L_v(2 L_v - 1)` at vertices and `psi_e = 4 L_a L_b` at midpoints.
  The vertex functions are not quadratic monomials, so use `sum L = 1` to lift
  the linear term: `L1 = L1*(L1+L2+L3) = L1^2 + L1*L2 + L1*L3`, hence

      psi_1 = 2*L1^2 - L1 = L1^2 - L1*L2 - L1*L3.

  Collecting terms for the interpolant with vertex values `u1,u2,u3` and
  midpoint values `u4 = u(mid of edge (v2,v3))`, `u5 = u(mid (v1,v3))`,
  `u6 = u(mid (v1,v2))` gives, in the basis
  `(L1^2, L2^2, L3^2, L2L3, L1L3, L1L2)`:

      c_{L1^2} = u1,  c_{L2^2} = u2,  c_{L3^2} = u3,
      c_{L2L3} = 4*u4 - u2 - u3,
      c_{L1L3} = 4*u5 - u1 - u3,
      c_{L1L2} = 4*u6 - u1 - u2.

  (Note the coefficient is `-1`, not `-2`: each vertex value contributes
  `-L_a*L_b` from exactly one of the two `psi_v` expansions. Check it at the
  midpoint of edge (v1,v2), where `L1 = L2 = 1/2, L3 = 0`:
  `u1/4 + u2/4 + (4*u6 - u1 - u2)/4 = u6`, as required. With `-2` you would
  get `u6 - u1/4 - u2/4`, which is wrong by an amount that vanishes as the
  mesh refines — so it degrades the observed order instead of failing loudly.
  That is exactly the kind of error the smoke test exists to catch.)

  Because the interpolant is continuous and single-valued at every vertex and
  every edge midpoint, the per-element formulas above agree wherever two
  elements meet, so we can simply write global DOFs directly: vertex DOF `v`
  gets `g` at that vertex, and edge DOF `nv + e` gets
  `4*g(midpoint of e) - g(endpoint a) - g(endpoint b)`.

Used by `solve_poisson` for Dirichlet lifting, and useful on its own to
compute the interpolation error for comparison against the FE error.
"""
function interpolate_nodal(m::Mesh2D, g::Function, p::Integer)
    ndof = lagrange_ndof(m, p)
    c = zeros(Float64, ndof)
    @inbounds for v in 1:m.nv
        c[v] = Float64(g(m.nodes[v, 1], m.nodes[v, 2]))
    end
    if p == 1
        return c
    elseif p == 2
        @inbounds for e in 1:m.ne
            a = m.edges[e, 1]; bb = m.edges[e, 2]
            xa = m.nodes[a, 1]; ya = m.nodes[a, 2]
            xb = m.nodes[bb, 1]; yb = m.nodes[bb, 2]
            gm = Float64(g(0.5 * (xa + xb), 0.5 * (ya + yb)))
            c[m.nv + e] = 4.0 * gm - c[a] - c[bb]
        end
        return c
    else
        throw(ArgumentError("TutorialSupport supports p = 1 or 2 (got $p)"))
    end
end

"""
    fe_vertex_values(m::Mesh2D, uh::AbstractVector, p::Integer) -> Vector{Float64}

Vertex values of the finite element function with coefficient vector `uh`.
For both `p = 1` and `p = 2` the value at vertex `v` is `uh[v]`, because every
basis function other than `L_v^p` vanishes at vertex `v`. Provided as a named
routine so plotting code does not have to re-derive that fact.
"""
function fe_vertex_values(m::Mesh2D, uh::AbstractVector, p::Integer)
    length(uh) == lagrange_ndof(m, p) ||
        throw(DimensionMismatch("uh has length $(length(uh)), expected $(lagrange_ndof(m, p))"))
    return Float64[uh[v] for v in 1:m.nv]
end

"""
    mesh_hmax(m::Mesh2D) -> Float64

Maximum edge length, via the library's `VFEM.find_mesh_hmax(m.nodes, m.edges)`.
Thin wrapper so tutorial scripts do not have to remember that `find_mesh_hmax`
takes the two arrays rather than the mesh.
"""
mesh_hmax(m::Mesh2D) = find_mesh_hmax(m.nodes, m.edges)

# ============================================================================
#  5. Poisson solve with Dirichlet conditions
# ============================================================================

"""
    PoissonSolution

Everything a caller might want back from `solve_poisson`:

* `uh`      :: `Vector{Float64}` — full-length (`ndof`) coefficient vector,
               boundary DOFs carrying the Dirichlet data.
* `p`       :: `Int`             — Lagrange order.
* `A`, `M`  :: `SparseMatrixCSC{Float64,Int}` — full stiffness and mass from
               `lagrange_laplace_matrices` (no BC applied), so the caller can
               form energy norms or reuse them.
* `b`       :: `Vector{Float64}` — full load vector before BC elimination.
* `int_dofs`, `bd_dofs` :: `Vector{Int}` — interior / Dirichlet DOF indices.
* `ug`      :: `Vector{Float64}` — the Dirichlet lift (all zeros in the
               homogeneous case), so `uh = ug + correction`.
"""
struct PoissonSolution
    uh::Vector{Float64}
    p::Int
    A::SparseMatrixCSC{Float64, Int}
    M::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    int_dofs::Vector{Int}
    bd_dofs::Vector{Int}
    ug::Vector{Float64}
end

Base.show(io::IO, s::PoissonSolution) =
    print(io, "PoissonSolution(p=", s.p, ", ndof=", length(s.uh),
              ", n_interior=", length(s.int_dofs), ")")

"""
    solve_poisson(m::Mesh2D, f::Function, p::Integer;
                  g = nothing, rule = dunavant_rule_6()) -> PoissonSolution

Solve `-Delta u = f` on the mesh `m` with conforming Lagrange elements of
order `p`, subject to Dirichlet data `u = g` on the boundary (`g = nothing`
means homogeneous, `u = 0`).

Method:

1. `A, M, bd_dofs = lagrange_laplace_matrices(m, p)` — the library assembles;
   we never touch the bilinear form.
2. `b = assemble_load_vector(m, f, p; rule)`.
3. Non-homogeneous data by **Dirichlet lifting**: build `ug =
   interpolate_nodal(m, g, p)` but keep only its boundary entries (interior
   entries zeroed, so the lift is supported near the boundary), then solve for
   the correction `w` in the interior from

       A[I,I] * w = b[I] - (A * ug)[I]

   and set `uh = ug + w` (with `w = 0` on the boundary). For `g = nothing`,
   `ug = 0` and this reduces to the homogeneous restriction
   `A[I,I] uh[I] = b[I]`.
4. Sparse Cholesky/LU via `\\` on `A[I,I]`, which is symmetric positive
   definite once the Dirichlet rows and columns are removed.

Returns a `PoissonSolution` (see its docstring). The returned `uh` is
full-length with the boundary values in place, so it can be handed straight to
`l2_error`, `h1_seminorm_error`, or `svg_solution`.
"""
function solve_poisson(m::Mesh2D, f::Function, p::Integer;
                       g = nothing, rule = dunavant_rule_6())
    A, M, bd_dofs = lagrange_laplace_matrices(m, p)
    ndof = size(A, 1)
    ndof == lagrange_ndof(m, p) ||
        error("library ndof $(ndof) disagrees with lagrange_ndof $(lagrange_ndof(m, p))")

    b = assemble_load_vector(m, f, p; rule = rule)

    is_bd = falses(ndof)
    @inbounds for d in bd_dofs
        is_bd[d] = true
    end
    int_dofs = findall(!, is_bd)
    bd_sorted = findall(is_bd)

    ug = zeros(Float64, ndof)
    if g !== nothing
        gc = interpolate_nodal(m, g, p)
        # Keep ONLY the boundary coefficients: the lift must not prejudge the
        # interior solution, it only has to reproduce the boundary trace.
        @inbounds for d in bd_sorted
            ug[d] = gc[d]
        end
    end

    rhs = b[int_dofs]
    if g !== nothing
        rhs .-= (A * ug)[int_dofs]
    end

    A_ii = A[int_dofs, int_dofs]
    w_int = A_ii \ rhs

    uh = copy(ug)
    @inbounds for (r, d) in enumerate(int_dofs)
        uh[d] += w_int[r]
    end

    return PoissonSolution(uh, Int(p), A, M, b, int_dofs, bd_sorted, ug)
end

# ============================================================================
#  6. Error norms and observed convergence order
# ============================================================================

"""
    l2_error(m::Mesh2D, uh::AbstractVector, u_exact::Function, p::Integer;
             rule = tri_quad_rule(8)) -> Float64

`sqrt(integral over Omega of (u_h - u)^2)`, evaluated element by element with
`rule`. `u_exact` is called as `u_exact(x, y)`.

The default rule is 64-point, degree-14 exact — deliberately much stronger
than the solution space, so that the reported number is the finite element
error and not a quadrature artefact. Using `dunavant_rule_6()` (degree 4) here
would under-integrate the P2 error integrand and can bend the observed order
on fine meshes.
"""
function l2_error(m::Mesh2D, uh::AbstractVector, u_exact::Function, p::Integer;
                  rule = tri_quad_rule(8))
    length(uh) == lagrange_ndof(m, p) ||
        throw(DimensionMismatch("uh has length $(length(uh)), expected $(lagrange_ndof(m, p))"))
    lam, wq = rule
    expo = lagrange_monomial_exponents(p)
    nb = size(expo, 1)
    vals = zeros(Float64, nb)
    acc = 0.0
    nq = length(wq)
    for k in 1:m.nt
        (x1, y1, x2, y2, x3, y3), detB, _ = _element_geometry(m, k)
        w_geo = abs(detB)
        gdofs = lagrange_local_dofs(m, k, p)
        for q in 1:nq
            L1 = lam[q, 1]; L2 = lam[q, 2]; L3 = lam[q, 3]
            xq = L1 * x1 + L2 * x2 + L3 * x3
            yq = L1 * y1 + L2 * y2 + L3 * y3
            _local_basis_values!(vals, expo, L1, L2, L3)
            uhq = 0.0
            @inbounds for i in 1:nb
                uhq += uh[gdofs[i]] * vals[i]
            end
            d = uhq - Float64(u_exact(xq, yq))
            acc += w_geo * wq[q] * d * d
        end
    end
    return sqrt(acc)
end

"""
    h1_seminorm_error(m::Mesh2D, uh::AbstractVector, grad_u_exact::Function,
                      p::Integer; rule = tri_quad_rule(8)) -> Float64

`sqrt(integral over Omega of |grad u_h - grad u|^2)`. `grad_u_exact(x, y)`
must return a 2-tuple or 2-vector `(du/dx, du/dy)`.

Same quadrature remark as `l2_error`: the default rule is degree-14 exact.
"""
function h1_seminorm_error(m::Mesh2D, uh::AbstractVector,
                           grad_u_exact::Function, p::Integer;
                           rule = tri_quad_rule(8))
    length(uh) == lagrange_ndof(m, p) ||
        throw(DimensionMismatch("uh has length $(length(uh)), expected $(lagrange_ndof(m, p))"))
    lam, wq = rule
    expo = lagrange_monomial_exponents(p)
    nb = size(expo, 1)
    gx = zeros(Float64, nb); gy = zeros(Float64, nb)
    acc = 0.0
    nq = length(wq)
    for k in 1:m.nt
        (x1, y1, x2, y2, x3, y3), detB, gL = _element_geometry(m, k)
        w_geo = abs(detB)
        gdofs = lagrange_local_dofs(m, k, p)
        for q in 1:nq
            L1 = lam[q, 1]; L2 = lam[q, 2]; L3 = lam[q, 3]
            xq = L1 * x1 + L2 * x2 + L3 * x3
            yq = L1 * y1 + L2 * y2 + L3 * y3
            _local_basis_grads!(gx, gy, expo, gL, L1, L2, L3)
            dhx = 0.0; dhy = 0.0
            @inbounds for i in 1:nb
                c = uh[gdofs[i]]
                dhx += c * gx[i]
                dhy += c * gy[i]
            end
            ge = grad_u_exact(xq, yq)
            ex = Float64(ge[1]); ey = Float64(ge[2])
            acc += w_geo * wq[q] * ((dhx - ex)^2 + (dhy - ey)^2)
        end
    end
    return sqrt(acc)
end

"""
    observed_order(errs::AbstractVector, hs::AbstractVector) -> Vector{Float64}

Fitted convergence rates between consecutive refinement levels,

    order[i] = log(errs[i] / errs[i+1]) / log(hs[i] / hs[i+1]),

returned with `length(errs) - 1` entries. `NaN` where the ratio is not
well defined (zero or non-finite error, equal mesh sizes) rather than throwing,
so a table can still be printed when one level happens to be exact.
"""
function observed_order(errs::AbstractVector, hs::AbstractVector)
    length(errs) == length(hs) ||
        throw(DimensionMismatch("errs and hs must have equal length"))
    n = length(errs)
    out = fill(NaN, max(n - 1, 0))
    for i in 1:(n - 1)
        e1 = Float64(errs[i]); e2 = Float64(errs[i + 1])
        h1 = Float64(hs[i]);   h2 = Float64(hs[i + 1])
        if e1 > 0 && e2 > 0 && isfinite(e1) && isfinite(e2) && h1 > 0 && h2 > 0 && h1 != h2
            out[i] = log(e1 / e2) / log(h1 / h2)
        end
    end
    return out
end

# ============================================================================
#  7. A mesh of last resort
# ============================================================================
#
#  The canonical tutorial mesh generator lives in the Meshes track and writes
#  vert.dat / tri.dat / edge.dat / bd.dat folders for `VFEM.mesh2d_load`. The
#  two routines below exist only so a script is never *blocked* on that: they
#  build a structured criss-cross square in memory. Prefer the generated
#  meshes when they are on disk.

"""
    mesh2d_from_nodes_elements(nodes::AbstractMatrix, elements::AbstractMatrix) -> Mesh2D

Build a complete `Mesh2D` (edges, boundary edges, `tri2edge`) from just the
vertex coordinates and triangle list. Same algorithm as the library's
`mesh2d_load_ne` — edges from the triangle loop with local order
`((v2,v3), (v1,v3), (v1,v2))` so `tri2edge[k,j]` is the edge opposite local
vertex `j`, and boundary edges are those belonging to exactly one triangle —
but taking arrays rather than a folder, so a script can generate a mesh
in memory with no file I/O.
"""
function mesh2d_from_nodes_elements(nodes::AbstractMatrix, elements::AbstractMatrix)
    size(nodes, 2) == 2 || throw(ArgumentError("nodes must have 2 columns"))
    size(elements, 2) == 3 || throw(ArgumentError("elements must have 3 columns"))
    nodes_f = Matrix{Float64}(nodes)
    elems = Matrix{Int}(elements)
    nv = size(nodes_f, 1)
    nt = size(elems, 1)

    edge_map = Dict{NTuple{2, Int}, Int}()
    edge_cnt = Dict{NTuple{2, Int}, Int}()
    edge_list = NTuple{2, Int}[]
    for k in 1:nt
        v1 = elems[k, 1]; v2 = elems[k, 2]; v3 = elems[k, 3]
        for (a, b) in ((v2, v3), (v1, v3), (v1, v2))
            key = a < b ? (a, b) : (b, a)
            if haskey(edge_map, key)
                edge_cnt[key] += 1
            else
                push!(edge_list, key)
                edge_map[key] = length(edge_list)
                edge_cnt[key] = 1
            end
        end
    end
    ne = length(edge_list)
    edges = Matrix{Int}(undef, ne, 2)
    for (i, (a, b)) in enumerate(edge_list)
        edges[i, 1] = a; edges[i, 2] = b
    end
    bd_ids = findall(i -> edge_cnt[edge_list[i]] == 1, 1:ne)
    nb = length(bd_ids)
    bd_edges = Matrix{Int}(undef, nb, 2)
    for (i, e) in enumerate(bd_ids)
        bd_edges[i, 1] = edges[e, 1]; bd_edges[i, 2] = edges[e, 2]
    end
    tri2edge = Matrix{Int}(undef, nt, 3)
    for k in 1:nt
        v1 = elems[k, 1]; v2 = elems[k, 2]; v3 = elems[k, 3]
        for (j, (a, b)) in enumerate(((v2, v3), (v1, v3), (v1, v2)))
            key = a < b ? (a, b) : (b, a)
            tri2edge[k, j] = edge_map[key]
        end
    end
    return Mesh2D(nodes_f, elems, edges, bd_edges, bd_ids, tri2edge, nv, nt, ne, nb)
end

"""
    tutorial_square_mesh(n::Integer) -> Mesh2D

Structured triangulation of the unit square `[0,1]^2` with `n x n` squares,
each split along the diagonal from `(i,j)` to `(i+1,j+1)`, giving
`nv = (n+1)^2`, `nt = 2n^2`, and `hmax = sqrt(2)/n`. Fallback mesh so the
smoke test can run before the Meshes track has written its folders.
"""
function tutorial_square_mesh(n::Integer)
    n >= 1 || throw(ArgumentError("n must be >= 1 (got $n)"))
    h = 1.0 / n
    nv = (n + 1)^2
    nodes = Matrix{Float64}(undef, nv, 2)
    idx(i, j) = (j - 1) * (n + 1) + i          # i, j in 1:n+1, column-major-ish
    for j in 1:(n + 1), i in 1:(n + 1)
        nodes[idx(i, j), 1] = (i - 1) * h
        nodes[idx(i, j), 2] = (j - 1) * h
    end
    elements = Matrix{Int}(undef, 2 * n * n, 3)
    r = 1
    for j in 1:n, i in 1:n
        a = idx(i, j); b = idx(i + 1, j); c = idx(i + 1, j + 1); d = idx(i, j + 1)
        # Counter-clockwise so detB > 0 (not required, but tidier to look at).
        elements[r, 1] = a; elements[r, 2] = b; elements[r, 3] = c; r += 1
        elements[r, 1] = a; elements[r, 2] = c; elements[r, 3] = d; r += 1
    end
    return mesh2d_from_nodes_elements(nodes, elements)
end

# ============================================================================
#  8. SVG figures — no plotting dependency (decision D-10)
# ============================================================================
#
#  Everything below writes SVG text directly. The pattern in each routine is:
#  compute a data -> pixel affine map, emit a <g> of primitives, emit axes and
#  labels last so they sit on top. Colours are from a colourblind-safe set
#  (Okabe-Ito), and font sizes are absolute pixels chosen to stay legible at
#  the ~700 px width the tutorial page renders at.

# Okabe-Ito qualitative palette: safe for deuteranopia/protanopia.
const PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7",
                 "#E69F00", "#56B4E9", "#F0E442", "#000000")

const _AXIS_COLOR = "#333333"
const _GRID_COLOR = "#dddddd"

_fmt(v::Real) = @sprintf("%.3f", v)

# XML-escape text destined for an SVG text node.
function _xesc(s::AbstractString)
    out = replace(String(s), "&" => "&amp;")
    out = replace(out, "<" => "&lt;")
    out = replace(out, ">" => "&gt;")
    return out
end

# Format a power-of-ten tick label as 10^k using an SVG tspan superscript.
function _pow10_label(k::Int)
    return "10<tspan baseline-shift=\"super\" font-size=\"9\">$(k)</tspan>"
end

"""
    svg_mesh(path, m::Mesh2D; width = 520, height = 520, margin = 26,
             show_boundary = true, title = "") -> String

Write the triangulation of `m` to `path` as SVG and return `path`.

Interior edges are thin grey, boundary edges (`m.bd_edges`) are drawn thicker
in the palette's orange so the Dirichlet set is visually obvious. The aspect
ratio of the domain is preserved. `title` is optional; when non-empty the
drawing area shrinks to make room.
"""
function svg_mesh(path::AbstractString, m::Mesh2D;
                  width::Integer = 520, height::Integer = 520,
                  margin::Integer = 26, show_boundary::Bool = true,
                  title::AbstractString = "")
    xs = @view m.nodes[:, 1]
    ys = @view m.nodes[:, 2]
    xmin, xmax = minimum(xs), maximum(xs)
    ymin, ymax = minimum(ys), maximum(ys)
    top = margin + (isempty(title) ? 0 : 22)
    # Preserve aspect ratio: one scale for both axes.
    sx = (width - 2 * margin) / max(xmax - xmin, eps())
    sy = (height - margin - top) / max(ymax - ymin, eps())
    s = min(sx, sy)
    # Centre the drawing in the available box.
    ox = margin + ((width - 2 * margin) - s * (xmax - xmin)) / 2
    oy = top + ((height - margin - top) - s * (ymax - ymin)) / 2
    px(x) = ox + s * (x - xmin)
    py(y) = oy + s * (ymax - y)                 # flip y: SVG grows downward

    io = IOBuffer()
    println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$width\" height=\"$height\" viewBox=\"0 0 $width $height\">")
    println(io, "<rect width=\"$width\" height=\"$height\" fill=\"white\"/>")
    if !isempty(title)
        println(io, "<text x=\"$(width / 2)\" y=\"20\" text-anchor=\"middle\" ",
                    "font-family=\"sans-serif\" font-size=\"14\" fill=\"$_AXIS_COLOR\">",
                    _xesc(title), "</text>")
    end
    # Interior edges.
    println(io, "<g stroke=\"#8c8c8c\" stroke-width=\"0.7\" fill=\"none\">")
    for e in 1:m.ne
        a = m.edges[e, 1]; b = m.edges[e, 2]
        print(io, "<line x1=\"", _fmt(px(m.nodes[a, 1])), "\" y1=\"", _fmt(py(m.nodes[a, 2])),
                  "\" x2=\"", _fmt(px(m.nodes[b, 1])), "\" y2=\"", _fmt(py(m.nodes[b, 2])), "\"/>")
    end
    println(io, "\n</g>")
    if show_boundary && m.nb > 0
        println(io, "<g stroke=\"$(PALETTE[2])\" stroke-width=\"2.0\" fill=\"none\">")
        for r in 1:m.nb
            a = m.bd_edges[r, 1]; b = m.bd_edges[r, 2]
            print(io, "<line x1=\"", _fmt(px(m.nodes[a, 1])), "\" y1=\"", _fmt(py(m.nodes[a, 2])),
                      "\" x2=\"", _fmt(px(m.nodes[b, 1])), "\" y2=\"", _fmt(py(m.nodes[b, 2])), "\"/>")
        end
        println(io, "\n</g>")
    end
    println(io, "<text x=\"$margin\" y=\"$(height - 6)\" font-family=\"sans-serif\" ",
                "font-size=\"11\" fill=\"#555555\">nv=$(m.nv), nt=$(m.nt), ne=$(m.ne), nb=$(m.nb)</text>")
    println(io, "</svg>")
    write(path, String(take!(io)))
    return String(path)
end

# Diverging-through-light sequential map from the Okabe-Ito blue to orange,
# with a light middle so that both extremes read as "far from zero".
function _colormap(t::Float64)
    t = clamp(t, 0.0, 1.0)
    # Anchor colours: deep blue -> light grey -> deep orange.
    stops = ((0.0,  (5,  48,  97)),
             (0.25, (67, 147, 195)),
             (0.5,  (240, 240, 240)),
             (0.75, (214, 96,  77)),
             (1.0,  (103, 0,   31)))
    for i in 1:(length(stops) - 1)
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1 || i == length(stops) - 1
            f = (t - t0) / (t1 - t0)
            f = clamp(f, 0.0, 1.0)
            r = round(Int, c0[1] + f * (c1[1] - c0[1]))
            g = round(Int, c0[2] + f * (c1[2] - c0[2]))
            b = round(Int, c0[3] + f * (c1[3] - c0[3]))
            return @sprintf("#%02x%02x%02x", r, g, b)
        end
    end
    return "#000000"
end

"""
    svg_solution(path, m::Mesh2D, uh::AbstractVector; p = 1, width = 600,
                 height = 520, margin = 26, title = "", draw_mesh = false,
                 colorbar = true) -> String

Write a per-triangle colour map of the finite element function `uh` to `path`.

Each triangle is filled with a flat colour taken from the mean of its three
vertex values (`fe_vertex_values`), which is robust, needs no contouring, and
reads well at tutorial figure sizes. A vertical colourbar with min/mid/max
labels is drawn on the right when `colorbar = true`; `draw_mesh = true`
overlays thin element outlines.
"""
function svg_solution(path::AbstractString, m::Mesh2D, uh::AbstractVector;
                      p::Integer = 1, width::Integer = 600,
                      height::Integer = 520, margin::Integer = 26,
                      title::AbstractString = "", draw_mesh::Bool = false,
                      colorbar::Bool = true)
    vv = fe_vertex_values(m, uh, p)
    vmin, vmax = minimum(vv), maximum(vv)
    span = vmax - vmin
    span <= 0 && (span = 1.0)

    cb_w = colorbar ? 64 : 0
    xs = @view m.nodes[:, 1]
    ys = @view m.nodes[:, 2]
    xmin, xmax = minimum(xs), maximum(xs)
    ymin, ymax = minimum(ys), maximum(ys)
    top = margin + (isempty(title) ? 0 : 22)
    plot_w = width - 2 * margin - cb_w
    plot_h = height - margin - top
    s = min(plot_w / max(xmax - xmin, eps()), plot_h / max(ymax - ymin, eps()))
    ox = margin + (plot_w - s * (xmax - xmin)) / 2
    oy = top + (plot_h - s * (ymax - ymin)) / 2
    px(x) = ox + s * (x - xmin)
    py(y) = oy + s * (ymax - y)

    io = IOBuffer()
    println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$width\" height=\"$height\" viewBox=\"0 0 $width $height\">")
    println(io, "<rect width=\"$width\" height=\"$height\" fill=\"white\"/>")
    if !isempty(title)
        println(io, "<text x=\"$((width - cb_w) / 2)\" y=\"20\" text-anchor=\"middle\" ",
                    "font-family=\"sans-serif\" font-size=\"14\" fill=\"$_AXIS_COLOR\">",
                    _xesc(title), "</text>")
    end
    println(io, "<g shape-rendering=\"crispEdges\">")
    for k in 1:m.nt
        v1 = m.elements[k, 1]; v2 = m.elements[k, 2]; v3 = m.elements[k, 3]
        val = (vv[v1] + vv[v2] + vv[v3]) / 3
        col = _colormap((val - vmin) / span)
        print(io, "<polygon points=\"",
              _fmt(px(m.nodes[v1, 1])), ",", _fmt(py(m.nodes[v1, 2])), " ",
              _fmt(px(m.nodes[v2, 1])), ",", _fmt(py(m.nodes[v2, 2])), " ",
              _fmt(px(m.nodes[v3, 1])), ",", _fmt(py(m.nodes[v3, 2])),
              "\" fill=\"", col, "\" stroke=\"", col, "\" stroke-width=\"0.4\"/>")
    end
    println(io, "\n</g>")
    if draw_mesh
        println(io, "<g stroke=\"#444444\" stroke-width=\"0.35\" fill=\"none\" opacity=\"0.6\">")
        for e in 1:m.ne
            a = m.edges[e, 1]; b = m.edges[e, 2]
            print(io, "<line x1=\"", _fmt(px(m.nodes[a, 1])), "\" y1=\"", _fmt(py(m.nodes[a, 2])),
                      "\" x2=\"", _fmt(px(m.nodes[b, 1])), "\" y2=\"", _fmt(py(m.nodes[b, 2])), "\"/>")
        end
        println(io, "\n</g>")
    end
    if colorbar
        bx = width - cb_w + 8
        by = top + 6
        bh = height - margin - by - 6
        bw = 14
        nseg = 64
        for i in 1:nseg
            t = 1.0 - (i - 1) / nseg          # top of bar = vmax
            yy = by + (i - 1) * bh / nseg
            print(io, "<rect x=\"$bx\" y=\"", _fmt(yy), "\" width=\"$bw\" height=\"",
                      _fmt(bh / nseg + 0.6), "\" fill=\"", _colormap(t), "\"/>")
        end
        println(io)
        println(io, "<rect x=\"$bx\" y=\"", _fmt(by), "\" width=\"$bw\" height=\"", _fmt(bh),
                    "\" fill=\"none\" stroke=\"$_AXIS_COLOR\" stroke-width=\"0.8\"/>")
        for (frac, val) in ((0.0, vmax), (0.5, (vmin + vmax) / 2), (1.0, vmin))
            yy = by + frac * bh
            println(io, "<text x=\"$(bx + bw + 4)\" y=\"", _fmt(yy + 3.5),
                        "\" font-family=\"sans-serif\" font-size=\"10\" fill=\"$_AXIS_COLOR\">",
                        @sprintf("%.3g", val), "</text>")
        end
    end
    println(io, "</svg>")
    write(path, String(take!(io)))
    return String(path)
end

"""
    svg_loglog(path, series; width = 640, height = 500, margin = 70,
               xlabel = "h", ylabel = "error", title = "",
               slopes = Float64[], legend = true) -> String

Log-log convergence plot. `series` is a vector of `NamedTuple`s or `Dict`s,
each with

* `x`     — abscissae (e.g. `hmax` per level), all strictly positive,
* `y`     — ordinates (errors), all strictly positive,
* `label` — legend text,

optionally `slope` — a reference slope to draw as a triangle beside that
series (e.g. `2.0` for the expected P1 L2 rate). `slopes` is an alternative
way to give one reference slope per series positionally.

Draws decade gridlines, `10^k` tick labels on both axes, axis titles, one
coloured polyline with circular markers per series, and the requested slope
triangles annotated with their slope value.
"""
function svg_loglog(path::AbstractString, series;
                    width::Integer = 640, height::Integer = 500,
                    margin::Integer = 70,
                    xlabel::AbstractString = "h",
                    ylabel::AbstractString = "error",
                    title::AbstractString = "",
                    slopes = Float64[], legend::Bool = true)
    isempty(series) && throw(ArgumentError("series must be non-empty"))
    getf(s, k, dflt) = s isa Dict ? get(s, k, dflt) :
                       (hasproperty(s, k) ? getproperty(s, k) : dflt)

    LX = Vector{Vector{Float64}}()
    LY = Vector{Vector{Float64}}()
    labels = String[]
    ref_slopes = Union{Nothing, Float64}[]
    for (i, s) in enumerate(series)
        x = Float64.(collect(getf(s, :x, Float64[])))
        y = Float64.(collect(getf(s, :y, Float64[])))
        length(x) == length(y) || throw(DimensionMismatch("series $i: x and y differ in length"))
        keep = [j for j in eachindex(x) if x[j] > 0 && y[j] > 0 && isfinite(x[j]) && isfinite(y[j])]
        push!(LX, log10.(x[keep]))
        push!(LY, log10.(y[keep]))
        push!(labels, String(getf(s, :label, "series $i")))
        sl = getf(s, :slope, nothing)
        if sl === nothing && i <= length(slopes)
            sl = slopes[i]
        end
        push!(ref_slopes, sl === nothing ? nothing : Float64(sl))
    end
    all_x = vcat(LX...); all_y = vcat(LY...)
    isempty(all_x) && throw(ArgumentError("no positive finite data to plot"))
    x0, x1 = minimum(all_x), maximum(all_x)
    y0, y1 = minimum(all_y), maximum(all_y)
    # Pad by 8% of range (or half a decade if a range is degenerate).
    padx = max((x1 - x0) * 0.10, 0.25); pady = max((y1 - y0) * 0.12, 0.25)
    x0 -= padx; x1 += padx; y0 -= pady; y1 += pady

    top = margin - 28 + (isempty(title) ? 8 : 30)
    right = legend ? 24 : margin
    pl = margin; pr = width - right; pt = top; pb = height - margin + 10
    sx(lx) = pl + (lx - x0) / (x1 - x0) * (pr - pl)
    sy(ly) = pb - (ly - y0) / (y1 - y0) * (pb - pt)

    io = IOBuffer()
    println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$width\" height=\"$height\" viewBox=\"0 0 $width $height\">")
    println(io, "<rect width=\"$width\" height=\"$height\" fill=\"white\"/>")
    if !isempty(title)
        println(io, "<text x=\"$((pl + pr) / 2)\" y=\"22\" text-anchor=\"middle\" ",
                    "font-family=\"sans-serif\" font-size=\"15\" fill=\"$_AXIS_COLOR\">",
                    _xesc(title), "</text>")
    end
    # Decade gridlines + ticks.
    println(io, "<g stroke=\"$_GRID_COLOR\" stroke-width=\"0.8\">")
    for k in ceil(Int, x0):floor(Int, x1)
        print(io, "<line x1=\"", _fmt(sx(k)), "\" y1=\"", _fmt(pt), "\" x2=\"",
                  _fmt(sx(k)), "\" y2=\"", _fmt(pb), "\"/>")
    end
    for k in ceil(Int, y0):floor(Int, y1)
        print(io, "<line x1=\"", _fmt(pl), "\" y1=\"", _fmt(sy(k)), "\" x2=\"",
                  _fmt(pr), "\" y2=\"", _fmt(sy(k)), "\"/>")
    end
    println(io, "\n</g>")
    # Axis box.
    println(io, "<rect x=\"", _fmt(pl), "\" y=\"", _fmt(pt), "\" width=\"", _fmt(pr - pl),
                "\" height=\"", _fmt(pb - pt), "\" fill=\"none\" stroke=\"$_AXIS_COLOR\" stroke-width=\"1.1\"/>")
    # Tick marks and labels.
    println(io, "<g font-family=\"sans-serif\" font-size=\"11\" fill=\"$_AXIS_COLOR\">")
    for k in ceil(Int, x0):floor(Int, x1)
        xx = sx(k)
        println(io, "<line x1=\"", _fmt(xx), "\" y1=\"", _fmt(pb), "\" x2=\"", _fmt(xx),
                    "\" y2=\"", _fmt(pb + 5), "\" stroke=\"$_AXIS_COLOR\" stroke-width=\"1\"/>")
        println(io, "<text x=\"", _fmt(xx), "\" y=\"", _fmt(pb + 19),
                    "\" text-anchor=\"middle\">", _pow10_label(k), "</text>")
    end
    for k in ceil(Int, y0):floor(Int, y1)
        yy = sy(k)
        println(io, "<line x1=\"", _fmt(pl - 5), "\" y1=\"", _fmt(yy), "\" x2=\"", _fmt(pl),
                    "\" y2=\"", _fmt(yy), "\" stroke=\"$_AXIS_COLOR\" stroke-width=\"1\"/>")
        println(io, "<text x=\"", _fmt(pl - 9), "\" y=\"", _fmt(yy + 4),
                    "\" text-anchor=\"end\">", _pow10_label(k), "</text>")
    end
    println(io, "</g>")
    # Axis titles.
    println(io, "<text x=\"$((pl + pr) / 2)\" y=\"$(height - 12)\" text-anchor=\"middle\" ",
                "font-family=\"sans-serif\" font-size=\"13\" fill=\"$_AXIS_COLOR\">", _xesc(xlabel), "</text>")
    println(io, "<text x=\"16\" y=\"$((pt + pb) / 2)\" text-anchor=\"middle\" ",
                "font-family=\"sans-serif\" font-size=\"13\" fill=\"$_AXIS_COLOR\" ",
                "transform=\"rotate(-90 16 $((pt + pb) / 2))\">", _xesc(ylabel), "</text>")
    # Data.
    for (i, (lx, ly)) in enumerate(zip(LX, LY))
        col = PALETTE[mod1(i, length(PALETTE))]
        isempty(lx) && continue
        pts = join([string(_fmt(sx(lx[j])), ",", _fmt(sy(ly[j]))) for j in eachindex(lx)], " ")
        println(io, "<polyline points=\"$pts\" fill=\"none\" stroke=\"$col\" stroke-width=\"2.0\"/>")
        for j in eachindex(lx)
            println(io, "<circle cx=\"", _fmt(sx(lx[j])), "\" cy=\"", _fmt(sy(ly[j])),
                        "\" r=\"3.4\" fill=\"$col\" stroke=\"white\" stroke-width=\"0.8\"/>")
        end
    end
    # Reference slope triangles, one per series that asked for one.
    #
    # Placement matters more than it sounds: several series can lie almost on
    # top of each other (P1-L2 and P2-H1 both decay like h^2 and are often
    # within a factor of two), so a triangle placed by global extent collides
    # with its neighbour and the reader cannot tell which curve it annotates.
    # Instead each triangle is anchored under *its own* curve — interpolate
    # that series in log-log at the triangle's left edge, then drop by a fixed
    # number of decades — with the drop and the horizontal window staggered by
    # series index so two near-coincident curves get visibly separated marks.
    for (i, sl) in enumerate(ref_slopes)
        sl === nothing && continue
        lx = LX[i]; ly = LY[i]
        length(lx) >= 2 || continue
        col = PALETTE[mod1(i, length(PALETTE))]
        xlo, xhi = minimum(lx), maximum(lx)
        span = xhi - xlo
        # Stagger the horizontal window slightly per series.
        shift = 0.06 * span * ((i - 1) % 2)
        xa = xlo + (0.24 + shift) * span
        xb = xa + 0.34 * span
        # Interpolate this series' curve at xa (data is sorted by x either way).
        ord = sortperm(lx)
        lxs = lx[ord]; lys = ly[ord]
        ycurve = lys[1]
        for j in 1:(length(lxs) - 1)
            if xa >= lxs[j] && xa <= lxs[j + 1]
                t = (xa - lxs[j]) / max(lxs[j + 1] - lxs[j], eps())
                ycurve = lys[j] * (1 - t) + lys[j + 1] * t
                break
            end
        end
        # Drop below the curve; alternate the drop so stacked curves separate.
        drop = 0.42 + 0.30 * ((i - 1) % 2)
        ya = ycurve - drop
        yb = ya + sl * (xb - xa)
        X1 = sx(xa); Y1 = sy(ya); X2 = sx(xb); Y2 = sy(yb)
        println(io, "<polyline points=\"", _fmt(X1), ",", _fmt(Y1), " ", _fmt(X2), ",", _fmt(Y1),
                    " ", _fmt(X2), ",", _fmt(Y2), " ", _fmt(X1), ",", _fmt(Y1),
                    "\" fill=\"none\" stroke=\"$col\" stroke-width=\"1.2\" stroke-dasharray=\"4,3\"/>")
        println(io, "<text x=\"", _fmt(X2 + 6), "\" y=\"", _fmt((Y1 + Y2) / 2 + 4),
                    "\" font-family=\"sans-serif\" font-size=\"11\" fill=\"$col\">",
                    @sprintf("%.3g", sl), "</text>")
    end
    # Legend, top-left inside the axes box.
    if legend
        lx0 = pl + 12
        ly0 = pt + 16
        boxh = 16 * length(labels) + 8
        boxw = 20 + 7 * maximum(length.(labels)) + 16
        println(io, "<rect x=\"", _fmt(lx0 - 6), "\" y=\"", _fmt(ly0 - 13), "\" width=\"", _fmt(boxw),
                    "\" height=\"", _fmt(boxh), "\" fill=\"white\" fill-opacity=\"0.86\" ",
                    "stroke=\"#bbbbbb\" stroke-width=\"0.7\"/>")
        for (i, lab) in enumerate(labels)
            col = PALETTE[mod1(i, length(PALETTE))]
            yy = ly0 + 16 * (i - 1)
            println(io, "<line x1=\"", _fmt(lx0), "\" y1=\"", _fmt(yy - 4), "\" x2=\"", _fmt(lx0 + 18),
                        "\" y2=\"", _fmt(yy - 4), "\" stroke=\"$col\" stroke-width=\"2.0\"/>")
            println(io, "<circle cx=\"", _fmt(lx0 + 9), "\" cy=\"", _fmt(yy - 4),
                        "\" r=\"3.2\" fill=\"$col\"/>")
            println(io, "<text x=\"", _fmt(lx0 + 24), "\" y=\"", _fmt(yy),
                        "\" font-family=\"sans-serif\" font-size=\"11.5\" fill=\"$_AXIS_COLOR\">",
                        _xesc(lab), "</text>")
        end
    end
    println(io, "</svg>")
    write(path, String(take!(io)))
    return String(path)
end

end # module TutorialSupport
