# tutorial/examples/12_schrodinger.jl
#
# ============================================================================
#  Chapter 12 — Schrodinger eigenproblems: how a potential enters a FEM matrix
# ============================================================================
#
#  We solve the 2D Dirichlet eigenproblem for the Schrodinger operator
#
#       H u = (-Delta + V) u = lambda u   in Omega,   u = 0 on dOmega.
#
#  Compared with Chapter 11 (pure Laplace) exactly one thing is new: the
#  bilinear form gains the *reaction* term
#
#       a(u,v) = (grad u, grad v) + (V u, v).
#
#  Everything hard about a Schrodinger discretisation lives in that second
#  term. The stiffness and mass matrices are integrals of polynomials and
#  VFEM.jl evaluates them in closed form; (V phi_i, phi_j) is an integral of
#  V times a polynomial, and V is whatever physics hands you. VFEM.jl offers
#  two different answers, and this chapter runs both:
#
#   (1) CECR / `schrodinger_eig_cecr` — V is replaced by its value at the
#       element centroid, i.e. by a piecewise CONSTANT. That is not a
#       convenience: the CECR space carries a second, piecewise-constant
#       component and Liu's theory needs the reaction coefficient to act on
#       exactly that component. The payoff is a *guaranteed lower* bound
#       (`eig_lower`) with no eigenvector information required.
#
#   (2) Conforming Lagrange / `create_matrix_lagrange(m, p, V_bern)` — V is
#       represented as a degree-4 BERNSTEIN polynomial per element and
#       (V phi_i, phi_j) is then integrated in closed form (the weight tensor
#       `_precompute_W_pot` inside the library). The payoff is Rayleigh-Ritz:
#       every computed eigenvalue is an *upper* bound for the corresponding
#       exact one.
#
#  Together they bracket the truth. This chapter also finds, and fixes, a
#  subtle order-limiting trap in route (2) — see PART 2.
#
#  UNITS AND CONVENTIONS (stated once; every number below obeys them)
#  -----------------------------------------------------------------
#  H = -Delta + V. No hbar, no mass, no factor 1/2. Readers used to the
#  physicists' H_phys = -(1/2) Laplacian + V_phys should note
#
#       -Delta + V = 2 * ( -(1/2) Delta + V/2 )   =>   lambda = 2 * E_phys.
#
#  * Empty box: on the square (-L/2, L/2)^2, lambda_{m,n} = pi^2 (m^2+n^2)/L^2.
#  * Harmonic well V = c*(x^2+y^2): separating into two 1D oscillators
#    -u'' + c x^2 u gives 2*sqrt(c)*(n + 1/2), so on the WHOLE PLANE
#
#       lambda_{nx,ny} = 2*sqrt(c) * (nx + ny + 1),   nx, ny = 0, 1, 2, ...
#
#    i.e. with omega := 2*sqrt(c), lambda = omega*(N+1) with level N = nx+ny
#    carrying multiplicity N+1: omega, 2*omega (x2), 3*omega (x3), ...
#    A Dirichlet box truncates the whole-plane problem, so the computed
#    eigenvalues are ABOVE these numbers by a box-truncation error that we
#    measure (PART 3a) rather than assume away.
#
#  Run:
#    julia --project=. tutorial/12_schrodinger.jl
#  Expects `TutorialSupport.jl` in the working directory and the tutorial
#  meshes under $VFEM_TUTORIAL_MESHES (default below).
# ============================================================================

using VFEM: Mesh2D, mesh2d_load, find_mesh_hmax,
            elem_V_bernstein, bernstein4_multiindices_2d,
            create_matrix_lagrange, lagrange_laplace_matrices,
            schrodinger_eig_cecr, dunavant_rule_6
using LinearAlgebra: Symmetric, Diagonal, cholesky, eigen, norm, factorize
using SparseArrays: SparseMatrixCSC, sparse, spzeros
using Random: MersenneTwister
using Printf: @printf, @sprintf

include("TutorialSupport.jl")
using .TutorialSupport: mesh2d_from_nodes_elements, tri_quad_rule,
                        fe_vertex_values, svg_solution, svg_loglog, svg_mesh

const MESHROOT = get(ENV, "VFEM_TUTORIAL_MESHES",
                     joinpath(@__DIR__, "meshdata"))
const OUT = get(ENV, "VFEM_TUTORIAL_OUT", ".")

# ============================================================================
#  0. Scaffolding: boundary DOFs, mesh scaling, and a robust eigensolver
# ============================================================================

"""
    lagrange_bd_dofs(m, p) -> Vector{Int}

Dirichlet DOFs in the numbering used by `create_matrix_lagrange`: vertex DOFs
`1..nv`, then edge DOFs `nv + edge_id`. A DOF is Dirichlet iff its vertex or
edge lies on the boundary. `create_matrix_lagrange` returns only `(A, M)` and
applies no boundary conditions, so the caller must build this set; we
cross-check it against the `bd_dofs` that `lagrange_laplace_matrices` returns
for the same mesh and order (the two assemblies use different bases but the
SAME global numbering).
"""
function lagrange_bd_dofs(m::Mesh2D, p::Integer)
    (p == 1 || p == 2) || throw(ArgumentError("p must be 1 or 2"))
    onbd = falses(m.nv)
    for r in 1:m.nb
        onbd[m.bd_edges[r, 1]] = true
        onbd[m.bd_edges[r, 2]] = true
    end
    dofs = findall(onbd)
    if p == 2
        append!(dofs, [m.nv + e for e in m.bd_edge_ids])
    end
    return sort!(dofs)
end

lagrange_ndof_nodal(m::Mesh2D, p::Integer) = p == 1 ? m.nv : m.nv + m.ne

"""
    affine_mesh(m, scale, shift) -> Mesh2D

Rescale/translate a mesh: `x -> scale*x + shift`. Connectivity is unchanged,
but edges, boundary edges and `tri2edge` are rebuilt by
`mesh2d_from_nodes_elements` so the result is a fully consistent `Mesh2D`.
Used to turn the unit-square meshes into the centred box (-L/2, L/2)^2 that a
centred potential wants.
"""
function affine_mesh(m::Mesh2D, scale::Real, shift::Tuple{<:Real,<:Real})
    nodes = similar(m.nodes)
    @inbounds for v in 1:m.nv
        nodes[v, 1] = scale * m.nodes[v, 1] + shift[1]
        nodes[v, 2] = scale * m.nodes[v, 2] + shift[2]
    end
    return mesh2d_from_nodes_elements(nodes, m.elements)
end

"""
    eigs_smallest(A, M, k; shift = 0.0, block = k+8, tol = 1e-11, maxit = 400)
        -> (lambda, X, resid, iters)

The `k` algebraically SMALLEST eigenvalues of the symmetric pencil
`A x = lambda M x`, by shift-invert block inverse iteration with
Rayleigh-Ritz:

    B = A + shift*M   (must be positive definite: shift > -lambda_1)
    repeat:  Y <- B^{-1} (M X);  M-orthonormalise Y;  Rayleigh-Ritz on
             (Y'A Y, Y'M Y);  X <- Y * (Ritz vectors)

Why not Arpack? Two reasons this chapter runs into for real:

* `eigs(A, M; which = :SM)` — which is what `schrodinger_eig_cecr` uses —
  selects smallest **magnitude**. For a well deep enough to produce NEGATIVE
  eigenvalues (Chapter 13) that is the wrong set: it returns eigenvalues near
  zero and silently skips the bound states.
* `eigs(...; sigma = 0.0, which = :LM)` failed to converge on a P2 square
  problem of this size during Chapter 11 (`XYAUPD_Exception: Maximum number
  of iterations taken`).

25 lines of block iteration on top of a sparse Cholesky is more robust here,
and it returns eigenVECTORS in the same call, which we need for the figures.
`resid` is the largest `||A x - lambda M x|| / max(|lambda|,1)` over the
returned pairs — always report it; a converged eigenvalue with a residual of
1e-3 is not an eigenvalue.
"""
function eigs_smallest(A::SparseMatrixCSC{Float64,Int}, M::SparseMatrixCSC{Float64,Int},
                       k::Integer; shift::Real = 0.0, block::Integer = 0,
                       tol::Real = 1e-11, maxit::Integer = 400)
    n = size(A, 1)
    k = min(k, n - 1)
    p = min(block > 0 ? block : k + 8, n)
    B = A + Float64(shift) * M
    F = try
        cholesky(Symmetric(B))
    catch err
        error("eigs_smallest: A + $(shift)*M is not positive definite " *
              "($(typeof(err))); pass a larger `shift`")
    end
    rng = MersenneTwister(20240517)
    X = randn(rng, n, p)
    lam = fill(NaN, p)
    resmax = Inf
    iters = maxit
    for it in 1:maxit
        Y = F \ (M * X)
        G = Symmetric(Matrix(Y' * (M * Y)))
        U = cholesky(G).U
        Y = Y / U                       # now Y' M Y = I
        Ar = Symmetric(Matrix(Y' * (A * Y)))
        E = eigen(Ar)
        ord = sortperm(E.values)
        lam = E.values[ord]
        X = Y * E.vectors[:, ord]
        Xk = @view X[:, 1:k]
        R = A * Xk - (M * Xk) * Diagonal(lam[1:k])
        resmax = 0.0
        for j in 1:k
            resmax = max(resmax, norm(@view R[:, j]) / max(abs(lam[j]), 1.0))
        end
        if resmax < tol
            iters = it
            break
        end
    end
    return lam[1:k], X[:, 1:k], resmax, iters
end

# ============================================================================
#  1. The potential's Bernstein representation — and how to make it exact
# ============================================================================
#
#  `create_matrix_lagrange(m, p, V_bern)` wants `V_bern` to be `nt x 15`
#  degree-4 BERNSTEIN COEFFICIENTS: internally it integrates
#
#      sum_over_15 V_bern[k, mu] * B^4_mu(L1,L2,L3)   times   phi_i phi_j
#
#  where B^4_mu = (4!/(a!b!c!)) L1^a L2^b L3^c are the Bernstein basis
#  polynomials indexed by `bernstein4_multiindices_2d()`.
#
#  `elem_V_bernstein(m, V)` fills that argument with the VALUES of V at the
#  15 control points (a*P1+b*P2+c*P3)/4. Coefficients and values are not the
#  same thing. The Bernstein operator reproduces only AFFINE functions:
#  feeding it control values of V = x^2 represents, instead of V, the
#  polynomial B_4[V] = V + O(h^2) (in 1D, B_n[x^2] = x^2 + x(1-x)/n exactly).
#  The error is second order in the element size, which is invisible next to
#  a P1 eigenvalue error but DOMINATES the P2 one. PART 2 measures this.
#
#  The fix costs one 15x15 solve, done once for the whole mesh: the 15
#  control points of the principal lattice of order 4 are unisolvent for
#  total degree 4, so we can *interpolate* — find the coefficients `c` with
#
#      sum_mu c_mu B^4_mu(control point nu) = V(control point nu),  nu = 1..15
#
#  The resulting degree-4 polynomial equals V exactly whenever deg V <= 4, and
#  for smooth V it is a degree-4 interpolant (error O(h^5)) instead of a
#  second-order approximation. The control-point barycentrics are the same on
#  every element, so the 15x15 matrix is assembled and factorised once.

const _BERN4_MI = bernstein4_multiindices_2d()          # 15 x 3, canonical order

_fact(n::Int) = n <= 1 ? 1 : prod(2:n)

"""
    bernstein4_basis(lambda) -> Vector{Float64}

The 15 degree-4 Bernstein basis polynomials `B^4_mu = (4!/(a!b!c!)) L^mu`
evaluated at barycentric `lambda`, in `bernstein4_multiindices_2d()` order.
"""
function bernstein4_basis(lam::AbstractVector{<:Real})
    out = Vector{Float64}(undef, 15)
    @inbounds for r in 1:15
        a = _BERN4_MI[r, 1]; b = _BERN4_MI[r, 2]; c = _BERN4_MI[r, 3]
        out[r] = (24 / (_fact(a) * _fact(b) * _fact(c))) *
                 lam[1]^a * lam[2]^b * lam[3]^c
    end
    return out
end

# The 15x15 "control value -> Bernstein coefficient" matrix, built once.
const _BERN4_VANDERMONDE = begin
    Bm = Matrix{Float64}(undef, 15, 15)
    for nu in 1:15
        lam = (_BERN4_MI[nu, 1] / 4, _BERN4_MI[nu, 2] / 4, _BERN4_MI[nu, 3] / 4)
        Bm[nu, :] = bernstein4_basis(collect(lam))
    end
    Bm
end
const _BERN4_FACT = factorize(_BERN4_VANDERMONDE)

"""
    elem_V_bernstein_exact(m, V_func) -> Matrix{Float64}   (nt x 15)

Drop-in replacement for `elem_V_bernstein` that returns true degree-4
Bernstein COEFFICIENTS rather than control values, by interpolating V at the
15 principal-lattice points of each element. Exact for any V of total degree
<= 4 (so exact for the harmonic well used below); a degree-4 interpolant
otherwise.
"""
function elem_V_bernstein_exact(m::Mesh2D, V_func)
    vals = elem_V_bernstein(m, V_func)             # nt x 15 control VALUES
    coef = Matrix{Float64}(undef, size(vals))
    @inbounds for k in 1:size(vals, 1)
        coef[k, :] = _BERN4_FACT \ vals[k, :]
    end
    return coef
end

# ---- nodal Lagrange basis, for independent quadrature of (V phi_i, phi_j) --
#
# Local basis order used by `create_matrix_lagrange` (read off
# `_lagrange_phi_monomials`):
#   p = 1:  (L1, L2, L3)
#   p = 2:  (2L1^2-L1, 2L2^2-L2, 2L3^2-L3, 4L2L3, 4L1L3, 4L1L2)
# local -> global: (v1,v2,v3) and, for p = 2, (nv+tri2edge[k,1..3]).
function nodal_basis_values(p::Integer, L1::Float64, L2::Float64, L3::Float64)
    if p == 1
        return (L1, L2, L3)
    else
        return (2L1^2 - L1, 2L2^2 - L2, 2L3^2 - L3, 4L2 * L3, 4L1 * L3, 4L1 * L2)
    end
end

function nodal_local_dofs(m::Mesh2D, k::Integer, p::Integer)
    v1 = m.elements[k, 1]; v2 = m.elements[k, 2]; v3 = m.elements[k, 3]
    p == 1 && return (v1, v2, v3)
    return (v1, v2, v3, m.nv + m.tri2edge[k, 1],
            m.nv + m.tri2edge[k, 2], m.nv + m.tri2edge[k, 3])
end

"""
    potential_matrix_quadrature(m, p, V_func; rule = tri_quad_rule(10)) -> SparseMatrixCSC

`(V phi_i, phi_j)` assembled by brute-force high-order quadrature in the
nodal Lagrange basis. Not how you would do production work — it is the
independent yardstick against which PART 2 measures the Bernstein route.
"""
function potential_matrix_quadrature(m::Mesh2D, p::Integer, V_func;
                                     rule = tri_quad_rule(10))
    lam, wq = rule
    ndof = lagrange_ndof_nodal(m, p)
    nb = p == 1 ? 3 : 6
    # Triplet (COO) accumulation, then one `sparse` call. Scalar `A[i,j] += v`
    # into a CSC matrix is O(nnz) per insertion, which is minutes at ndof ~ 1e4.
    Ii = Int[]; Jj = Int[]; Vv = Float64[]
    sizehint!(Ii, m.nt * nb * nb); sizehint!(Jj, m.nt * nb * nb)
    sizehint!(Vv, m.nt * nb * nb)
    for k in 1:m.nt
        v1 = m.elements[k, 1]; v2 = m.elements[k, 2]; v3 = m.elements[k, 3]
        x1, y1 = m.nodes[v1, 1], m.nodes[v1, 2]
        x2, y2 = m.nodes[v2, 1], m.nodes[v2, 2]
        x3, y3 = m.nodes[v3, 1], m.nodes[v3, 2]
        w_geo = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))   # = 2|K|
        g = nodal_local_dofs(m, k, p)
        Aloc = zeros(nb, nb)
        for q in eachindex(wq)
            L1 = lam[q, 1]; L2 = lam[q, 2]; L3 = lam[q, 3]
            xq = L1 * x1 + L2 * x2 + L3 * x3
            yq = L1 * y1 + L2 * y2 + L3 * y3
            ph = nodal_basis_values(p, L1, L2, L3)
            cw = w_geo * wq[q] * Float64(V_func(xq, yq))
            for i in 1:nb, j in 1:nb
                Aloc[i, j] += cw * ph[i] * ph[j]
            end
        end
        for i in 1:nb, j in 1:nb
            push!(Ii, g[i]); push!(Jj, g[j]); push!(Vv, Aloc[i, j])
        end
    end
    return sparse(Ii, Jj, Vv, ndof, ndof)
end

# ============================================================================
#  2. The conforming Schrodinger solve
# ============================================================================

"""
    schrodinger_conforming(m, p, V_func, neig; exact_bernstein = true, shift = 0.0)
        -> (lambda, U, ndof, resid, iters)

Rayleigh-Ritz for `-Delta + V` in the conforming Lagrange space of order `p`:

1. `A, M = create_matrix_lagrange(m, p, V_bern)` — the library assembles both
   the Laplace part and the potential part, given `V_bern`.
2. Remove Dirichlet DOFs (`lagrange_bd_dofs`).
3. `eigs_smallest` on the interior pencil.
4. Scatter eigenvectors back to full length so they can be plotted.

`exact_bernstein = true` uses `elem_V_bernstein_exact` (interpolation
coefficients); `false` reproduces the library's `elem_V_bernstein` control
values. Every eigenvalue returned is an upper bound for the corresponding
eigenvalue of the discretised operator -Delta + V_h, where V_h is whichever
degree-4 polynomial the `V_bern` argument encodes.
"""
function schrodinger_conforming(m::Mesh2D, p::Integer, V_func, neig::Integer;
                                exact_bernstein::Bool = true, shift::Real = 0.0)
    V_bern = exact_bernstein ? elem_V_bernstein_exact(m, V_func) :
                               elem_V_bernstein(m, V_func)
    A, M = create_matrix_lagrange(m, p, V_bern)
    bd = lagrange_bd_dofs(m, p)
    ndof = size(A, 1)
    isbd = falses(ndof); isbd[bd] .= true
    int = findall(!, isbd)
    A0 = A[int, int]; M0 = M[int, int]
    lam, X, res, iters = eigs_smallest(A0, M0, neig; shift = shift)
    U = zeros(ndof, size(X, 2))
    U[int, :] = X
    return lam, U, ndof, res, iters
end

# ============================================================================
#  PART 1 — V = 0 as the gate
# ============================================================================
#
#  Nothing below is trustworthy unless the potential plumbing reproduces the
#  pure Laplacian when the potential is switched off. `zeros(nt, 15)` is the
#  documented pure-Laplace `V_bern`; `(x,y) -> 0.0` is its function form and
#  must give a bit-comparable matrix.

println("="^76)
println("PART 1 — V = 0 control on the unit square (the gate)")
println("="^76)

square_exact(m::Int, n::Int, L::Float64) = pi^2 * (m^2 + n^2) / L^2
# First six Dirichlet Laplace eigenvalues of a square of side L, ascending.
function square_exact_list(L::Float64, k::Int)
    vals = Float64[]
    for a in 1:6, b in 1:6
        push!(vals, square_exact(a, b, L))
    end
    return sort(vals)[1:k]
end

const NEIG = 6
rows_ctl = Vector{NamedTuple}()

for n in (4, 8, 16, 32)
    m = mesh2d_load(joinpath(MESHROOT, "unit_square_$(n)"))
    h = find_mesh_hmax(m.nodes, m.edges)
    ref = square_exact_list(1.0, NEIG)

    # (a) two spellings of "no potential" must agree exactly.
    A_fun, M_fun = create_matrix_lagrange(m, 1, elem_V_bernstein(m, (x, y) -> 0.0))
    A_zer, M_zer = create_matrix_lagrange(m, 1, zeros(m.nt, 15))
    dz = maximum(abs.(A_fun - A_zer))

    # (b) our Dirichlet DOF set must match the library's.
    _, _, bd_lib = lagrange_laplace_matrices(m, 1)
    bd_ok = sort(bd_lib) == lagrange_bd_dofs(m, 1)

    for p in (1, 2)
        lam, _, ndof, res, its = schrodinger_conforming(m, p, (x, y) -> 0.0, NEIG)
        err = abs.(lam .- ref) ./ ref
        @printf("n=%2d p=%d ndof=%5d hmax=%.4f  lam1=%.10f (exact %.10f)  max rel err=%.3e  resid=%.1e its=%d\n",
                n, p, ndof, h, lam[1], ref[1], maximum(err), res, its)
        for k in 1:NEIG
            push!(rows_ctl, (case = "control_V0", solver = "conforming_P$(p)",
                             c = 0.0, L = 1.0, n = n, p = p, ndof = ndof,
                             hmax = h, level = k, lambda = lam[k],
                             reference = ref[k], resid = res))
        end
        # Rayleigh-Ritz: conforming eigenvalues are upper bounds.
        @assert all(lam .>= ref .* (1 - 1e-9)) "conforming eigenvalue below exact!"
    end
    @printf("        V_bern function-vs-zeros max|dA| = %.2e ; bd_dofs match library: %s\n",
            dz, bd_ok)
    @assert dz == 0.0
    @assert bd_ok

    # (c) the CECR route, for comparison. NOTE the sign of the discrepancy.
    #
    # Wrapped: `schrodinger_eig_cecr` calls Arpack `eigs(...; which = :SM)`,
    # whose convergence at n_int of a few thousand is not guaranteed (Chapter 11
    # hit `XYAUPD_Exception` on a comparable pencil). A non-convergence here is a
    # library limitation worth REPORTING, not a reason to lose the whole run.
    r = try
        schrodinger_eig_cecr(m, (x, y) -> 0.0, NEIG)
    catch err
        @printf("        CECR FAILED at n=%d (n_int=%d): %s\n",
                n, m.ne + m.nt - m.nb, sprint(showerror, err))
        nothing
    end
    if r === nothing
        continue
    end
    @printf("        CECR: Ch=%.6f gamma_h=%.1f  eig_h[1]=%.10f  eig_lower[1]=%.10f\n",
            r.Ch, r.gamma_h, r.eig_h[1], r.eig_lower[1])
    @printf("        CECR eig_h - exact (all 6): %s\n",
            join([@sprintf("%+.4f", r.eig_h[k] - ref[k]) for k in 1:length(r.eig_h)], " "))
    for k in 1:length(r.eig_h)
        push!(rows_ctl, (case = "control_V0", solver = "CECR_eig_h",
                         c = 0.0, L = 1.0, n = n, p = 0, ndof = m.ne + m.nt,
                         hmax = h, level = k, lambda = r.eig_h[k],
                         reference = ref[k], resid = NaN))
        push!(rows_ctl, (case = "control_V0", solver = "CECR_eig_lower",
                         c = 0.0, L = 1.0, n = n, p = 0, ndof = m.ne + m.nt,
                         hmax = h, level = k, lambda = r.eig_lower[k],
                         reference = ref[k], resid = NaN))
    end
    if any(r.eig_h .> square_exact_list(1.0, length(r.eig_h)))
        println("        !! some CECR eig_h EXCEED the exact value")
    else
        println("        note: every CECR eig_h is BELOW the exact eigenvalue " *
                "(so `eig_upper` is not an upper bound)")
    end
end

# ============================================================================
#  PART 2 — how accurately is (V phi, psi) integrated?
# ============================================================================

println()
println("="^76)
println("PART 2 — the potential term: control values vs Bernstein coefficients")
println("="^76)

const C_HARM = 100.0                      # V = C_HARM * r^2  =>  omega = 2*sqrt(c) = 20
V_harm(x, y) = C_HARM * (x * x + y * y)
V_gauss(x, y) = 40.0 * exp(-8.0 * (x * x + y * y))   # smooth but not polynomial

rows_pot = Vector{NamedTuple}()
for n in (4, 8, 16, 32)
    m0 = mesh2d_load(joinpath(MESHROOT, "unit_square_$(n)"))
    m = affine_mesh(m0, 2.0, (-1.0, -1.0))            # box (-1,1)^2
    h = find_mesh_hmax(m.nodes, m.edges)
    for (vname, Vf) in (("quadratic", V_harm), ("gaussian", V_gauss))
        Aq = potential_matrix_quadrature(m, 2, Vf)     # yardstick, degree-18 rule
        nrm = norm(Aq)                                 # Frobenius, sparse-aware
        Az, _ = create_matrix_lagrange(m, 2, zeros(m.nt, 15))
        for (mode, Vb) in (("control_values", elem_V_bernstein(m, Vf)),
                           ("exact_coeffs", elem_V_bernstein_exact(m, Vf)))
            Av, _ = create_matrix_lagrange(m, 2, Vb)
            rel = norm(Av - Az - Aq) / nrm
            @printf("n=%2d hmax=%.4f  V=%-9s  %-14s  rel |A_pot - quad| = %.3e\n",
                    n, h, vname, mode, rel)
            push!(rows_pot, (n = n, hmax = h, potential = vname, mode = mode,
                             rel_diff = rel))
        end
    end
end
println()
println("Reading of the table above: with control values the potential term is")
println("wrong by O(h^2) even for a QUADRATIC V (the Bernstein operator only")
println("reproduces affine functions); with interpolation coefficients it is")
println("exact to round-off for deg V <= 4, and O(h^5) for the Gaussian.")

# ============================================================================
#  PART 3 — the harmonic well
# ============================================================================

println()
println("="^76)
println("PART 3a — box-truncation error at fixed c = $(C_HARM), P2, n = 32")
println("="^76)
println("whole-plane reference: lambda_N = 2*sqrt(c)*(N+1) = " *
        "$(join([@sprintf("%.4f", 2*sqrt(C_HARM)*(N+1)) for N in 0:2], ", ")) ...")

osc_ref(c::Float64, k::Int) = begin
    vals = Float64[]
    for nx in 0:8, ny in 0:8
        push!(vals, 2 * sqrt(c) * (nx + ny + 1))
    end
    sort(vals)[1:k]
end

rows_box = Vector{NamedTuple}()
m0_32 = mesh2d_load(joinpath(MESHROOT, "unit_square_32"))
for L in (1.0, 1.5, 2.0, 3.0, 4.0)
    m = affine_mesh(m0_32, L, (-L / 2, -L / 2))
    h = find_mesh_hmax(m.nodes, m.edges)
    lam, _, ndof, res, its = schrodinger_conforming(m, 2, V_harm, NEIG)
    ref = osc_ref(C_HARM, NEIG)
    @printf("L=%4.1f hmax=%.4f ndof=%5d  lam = %s   (resid %.1e)\n",
            L, h, ndof, join([@sprintf("%9.5f", v) for v in lam], " "), res)
    @printf("            lam - plane ref = %s\n",
            join([@sprintf("%+9.2e", lam[k] - ref[k]) for k in 1:NEIG], " "))
    for k in 1:NEIG
        push!(rows_box, (case = "harmonic_box", solver = "conforming_P2",
                         c = C_HARM, L = L, n = 32, p = 2, ndof = ndof, hmax = h,
                         level = k, lambda = lam[k], reference = ref[k], resid = res))
    end
end
println()
println("The L = 1 row is dominated by box truncation (the well's ground state")
println("has width c^(-1/4) = $(round(C_HARM^(-0.25), digits = 4)) and does not fit).")
println("As L grows the truncation error collapses and the residual gap is the")
println("discretisation error, which GROWS with L at fixed n because h = L*sqrt(2)/n.")

println()
println("="^76)
println("PART 3b — mesh convergence at c = $(C_HARM) on the box L = 2")
println("="^76)

conv = Dict{String, Vector{Tuple{Float64, Float64}}}()   # label -> (h, err)
rows_conv = Vector{NamedTuple}()
ref_osc = osc_ref(C_HARM, NEIG)
lam_ref_fine = NaN
for p in (1, 2), mode in (true, false)
    mode == false && p == 1 && continue          # only P2 is order-limited
    label = "P$(p)" * (mode ? " exact coeffs" : " control values")
    pts = Tuple{Float64, Float64}[]
    lams = Float64[]; hs = Float64[]
    for n in (4, 8, 16, 32, 64)
        m = affine_mesh(mesh2d_load(joinpath(MESHROOT, "unit_square_$(n)")),
                        2.0, (-1.0, -1.0))
        h = find_mesh_hmax(m.nodes, m.edges)
        t = @elapsed begin
            lam, _, ndof, res, its = schrodinger_conforming(m, p, V_harm, NEIG;
                                                            exact_bernstein = mode)
        end
        push!(lams, lam[1]); push!(hs, h)
        push!(rows_conv, (case = "harmonic_conv", solver = "conforming_P$(p)_" *
                          (mode ? "exact" : "control"), c = C_HARM, L = 2.0,
                          n = n, p = p, ndof = ndof, hmax = h, level = 1,
                          lambda = lam[1], reference = ref_osc[1], resid = res))
        @printf("%-22s n=%2d ndof=%6d hmax=%.5f  lam1=%.12f  err=%.3e  %5.1fs\n",
                label, n, ndof, h, lam[1], abs(lam[1] - ref_osc[1]), t)
    end
    # Observed order against the whole-plane reference (box error is negligible
    # at L = 2, c = 100 -- PART 3a quantified it).
    errs = abs.(lams .- ref_osc[1])
    for i in 1:(length(errs) - 1)
        @printf("%-22s   observed order (levels %d->%d): %.3f\n",
                label, i, i + 1, log(errs[i] / errs[i+1]) / log(hs[i] / hs[i+1]))
    end
    conv[label] = collect(zip(hs, errs))
end

println()
println("="^76)
println("PART 3c — spectrum vs potential strength c (box L = 2, P2, n = 32)")
println("="^76)

rows_c = Vector{NamedTuple}()
m_box32 = affine_mesh(m0_32, 2.0, (-1.0, -1.0))
h_box32 = find_mesh_hmax(m_box32.nodes, m_box32.edges)
c_list = (0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0)
lam_by_c = Dict{Float64, Vector{Float64}}()
vec_by_c = Dict{Float64, Matrix{Float64}}()
box_ref = square_exact_list(2.0, NEIG)
@printf("%9s %10s %10s | %s\n", "c", "omega", "box ref", "lambda_1..6")
for c in c_list
    Vc = (x, y) -> c * (x * x + y * y)
    lam, U, ndof, res, its = schrodinger_conforming(m_box32, 2, Vc, NEIG)
    lam_by_c[c] = lam
    vec_by_c[c] = U
    oref = c > 0 ? osc_ref(c, NEIG) : fill(NaN, NEIG)
    @printf("%9.1f %10.4f %10.4f | %s  (resid %.1e)\n",
            c, 2 * sqrt(c), box_ref[1],
            join([@sprintf("%11.5f", v) for v in lam], " "), res)
    if c > 0
        @printf("%9s %10s %10s | %s\n", "", "", "osc ref",
                join([@sprintf("%11.5f", v) for v in oref], " "))
    end
    for k in 1:NEIG
        push!(rows_c, (case = "harmonic_strength", solver = "conforming_P2",
                       c = c, L = 2.0, n = 32, p = 2, ndof = ndof, hmax = h_box32,
                       level = k, lambda = lam[k],
                       reference = c > 0 ? oref[k] : box_ref[k], resid = res))
    end
    # CECR lower bounds for the same problem (V >= 0, so no Liu shift needed).
    rc = try
        schrodinger_eig_cecr(m_box32, Vc, NEIG)
    catch err
        @printf("%9s CECR FAILED: %s\n", "", sprint(showerror, err))
        nothing
    end
    rc === nothing && continue
    @printf("%9s %10s %10s | %s   [CECR lower, Ch=%.4f gamma=%.1f]\n", "", "", "",
            join([@sprintf("%11.5f", v) for v in rc.eig_lower], " "), rc.Ch, rc.gamma_h)
    for k in 1:length(rc.eig_lower)
        push!(rows_c, (case = "harmonic_strength", solver = "CECR_eig_lower",
                       c = c, L = 2.0, n = 32, p = 0,
                       ndof = m_box32.ne + m_box32.nt, hmax = h_box32,
                       level = k, lambda = rc.eig_lower[k],
                       reference = c > 0 ? oref[k] : box_ref[k], resid = NaN))
    end
end

# ============================================================================
#  PART 4 — figures and CSV
# ============================================================================

println()
println("="^76)
println("PART 4 — artifacts")
println("="^76)

# Eigenfunction figures: normalise sign and scale so the colourbar is readable.
function normalise_mode!(u::Vector{Float64})
    i = argmax(abs.(u))
    u ./= u[i]
    return u
end

for (c, tag) in ((0.0, "c0"), (100.0, "c100"), (10000.0, "c10000"))
    U = vec_by_c[c]
    u1 = normalise_mode!(copy(U[:, 1]))
    p = svg_solution(joinpath(OUT, "schrodinger_ground_$(tag).svg"), m_box32, u1;
                     p = 2, title = "ground state, V = $(Int(c))*r^2, " *
                     "lambda_1 = $(round(lam_by_c[c][1], digits = 4))")
    println("wrote ", p)
end
u3 = normalise_mode!(copy(vec_by_c[100.0][:, 3]))
println("wrote ", svg_solution(joinpath(OUT, "schrodinger_mode3_c100.svg"),
                               m_box32, u3; p = 2,
                               title = "third mode, V = 100*r^2, lambda_3 = " *
                               "$(round(lam_by_c[100.0][3], digits = 4))"))

# Spectrum vs strength, log-log: lambda_k ~ 2*sqrt(c)*(N+1) has slope 1/2.
series_c = [(x = [c for c in c_list if c > 0],
             y = [lam_by_c[c][k] for c in c_list if c > 0],
             label = "lambda_$(k)", slope = k == 1 ? 0.5 : nothing)
            for k in 1:NEIG]
println("wrote ", svg_loglog(joinpath(OUT, "schrodinger_spectrum_vs_c.svg"), series_c;
                             xlabel = "potential strength c", ylabel = "lambda",
                             title = "harmonic well on (-1,1)^2, P2 n=32: " *
                             "spectrum vs strength"))

# Convergence figure.
series_conv = [(x = [t[1] for t in v], y = [t[2] for t in v], label = kk,
                slope = startswith(kk, "P1") ? 2.0 :
                        (occursin("exact", kk) ? 4.0 : 2.0))
               for (kk, v) in sort(collect(conv), by = first)]
println("wrote ", svg_loglog(joinpath(OUT, "schrodinger_convergence.svg"), series_conv;
                             xlabel = "hmax", ylabel = "|lambda_1 - 2*sqrt(c)|",
                             title = "harmonic well, c=$(Int(C_HARM)), L=2: " *
                             "eigenvalue convergence"))

# ---- CSVs -------------------------------------------------------------------
function write_spectrum_csv(path, rowsets)
    open(path, "w") do io
        println(io, "case,solver,c,box_L,mesh_n,p,ndof,hmax,level,lambda," *
                    "reference,abs_error,rel_error,resid")
        for rows in rowsets, r in rows
            ae = isnan(r.reference) ? NaN : r.lambda - r.reference
            re = isnan(r.reference) || r.reference == 0 ? NaN : ae / r.reference
            @printf(io, "%s,%s,%.6g,%.6g,%d,%d,%d,%.10e,%d,%.16e,%.16e,%.6e,%.6e,%.3e\n",
                    r.case, r.solver, r.c, r.L, r.n, r.p, r.ndof, r.hmax,
                    r.level, r.lambda, r.reference, ae, re, r.resid)
        end
    end
    println("wrote ", path)
end
write_spectrum_csv(joinpath(OUT, "schrodinger_spectrum.csv"),
                   (rows_ctl, rows_box, rows_conv, rows_c))

open(joinpath(OUT, "schrodinger_potential_term.csv"), "w") do io
    println(io, "mesh_n,hmax,potential,V_bern_mode,rel_diff_vs_quadrature")
    for r in rows_pot
        @printf(io, "%d,%.10e,%s,%s,%.10e\n", r.n, r.hmax, r.potential, r.mode,
                r.rel_diff)
    end
end
println("wrote ", joinpath(OUT, "schrodinger_potential_term.csv"))

println()
println("12_schrodinger.jl DONE")
