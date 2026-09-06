# =============================================================================
#  03_eig_square.jl — Dirichlet Laplace eigenvalues on domains with a
#                     known closed-form spectrum
#
#  Problem:   -Delta u = lambda u  in Omega,   u = 0 on dOmega
#
#  Two domains whose spectrum is known exactly, so every digit we print can be
#  checked against analysis:
#
#    (a) the unit square (0,1)^2
#            lambda_{m,n} = pi^2 (m^2 + n^2),        m, n >= 1
#        First six:  2pi^2, 5pi^2 (double), 8pi^2, 10pi^2 (double)
#
#    (b) the equilateral triangle of side a  (vertices (0,0), (a,0),
#        (a/2, a*sqrt(3)/2))
#            lambda_{m,n} = (16 pi^2 / (9 a^2)) (m^2 + m n + n^2),  m >= n >= 1
#        multiplicity 1 when m == n, multiplicity 2 otherwise.
#
#  Two facts we want to *observe*, not merely assert:
#
#    1. UPPER BOUNDS.  Conforming Lagrange elements give V_h subset of
#       H^1_0(Omega).  The k-th eigenvalue is the min-max of the Rayleigh
#       quotient over k-dimensional subspaces; minimising over a *smaller*
#       family of subspaces can only raise the minimum.  Hence
#       lambda_k <= lambda_k^h for every k and every mesh: the computed values
#       must all sit ABOVE the exact ones.  We check this explicitly.
#
#    2. DOUBLE RATE.  If the eigenfunction is approximated to O(h^p) in the
#       energy seminorm, the eigenvalue error is O(h^{2p}), because the
#       Rayleigh quotient is stationary at an eigenfunction so the first-order
#       term cancels.  So P1 -> O(h^2) and P2 -> O(h^4).  We measure both.
#
#  Run with:
#     export PATH=$HOME/.juliaup/bin:$PATH
#     julia --project=. tutorial/03_eig_square.jl
# =============================================================================

using VFEM: Mesh2D, mesh2d_load, laplace_eig_lagrange, LaplaceEigLagrange,
            lagrange_laplace_matrices
using LinearAlgebra: Symmetric, cholesky, eigen
using SparseArrays: SparseMatrixCSC
using Printf: @printf, @sprintf

# TutorialSupport.jl is a plain module file (not a package): include it, then
# bring its exports into scope with the leading dot.
const HELPERS = get(ENV, "VFEM_TUTORIAL_HELPERS",
                    joinpath(@__DIR__, "TutorialSupport.jl"))
include(HELPERS)
using .TutorialSupport

const MESHROOT = get(ENV, "VFEM_TUTORIAL_MESHES",
                     joinpath(@__DIR__, "meshdata"))
const OUT = get(ENV, "VFEM_TUTORIAL_OUT", ".")

const NEIG = 6

# -----------------------------------------------------------------------------
#  Exact spectra
# -----------------------------------------------------------------------------

"""
    square_exact(neig) -> Vector{Float64}

First `neig` Dirichlet eigenvalues of -Delta on (0,1)^2, ascending, WITH
multiplicity: enumerate pi^2 (m^2 + n^2) over m, n >= 1 and sort.
"""
function square_exact(neig::Integer)
    vals = Float64[]
    K = 12                       # generous: (12,12) is far past the 6th value
    for m in 1:K, n in 1:K
        push!(vals, pi^2 * (m^2 + n^2))
    end
    sort!(vals)
    return vals[1:neig]
end

"""
    equilateral_exact(neig; a = 1.0) -> Vector{Float64}

First `neig` Dirichlet eigenvalues of -Delta on the equilateral triangle of
side `a`, ascending, with multiplicity.  Closed form (Lame):

    lambda_{m,n} = (16 pi^2 / (9 a^2)) (m^2 + m n + n^2),   m >= n >= 1

with multiplicity 1 if m == n and 2 otherwise -- so a pair m > n contributes
the SAME number twice to the sorted list.
"""
function equilateral_exact(neig::Integer; a::Real = 1.0)
    c = 16 * pi^2 / (9 * a^2)
    vals = Float64[]
    K = 12
    for m in 1:K, n in 1:m
        lam = c * (m^2 + m * n + n^2)
        push!(vals, lam)
        m == n || push!(vals, lam)          # multiplicity 2 when m != n
    end
    sort!(vals)
    return vals[1:neig]
end

# -----------------------------------------------------------------------------
#  Getting the eigenpairs: the library routine, plus a fallback
# -----------------------------------------------------------------------------
#
#  `laplace_eig_lagrange(m, p, neig)` returns a `LaplaceEigLagrange` with fields
#  `eig_value`, `eig_func`, `A`, `M`, `bd_dofs`.  `A` and `M` are the FULL
#  ndof x ndof matrices -- no boundary condition applied -- and `bd_dofs` lists
#  the Dirichlet DOFs removed before the eigensolve.  `eig_func` is padded back
#  to ndof rows with zeros on those DOFs.
#
#  Internally the routine branches on the interior DOF count: at n_int <= 6000
#  it densifies and calls `eigen`, and above that it calls Arpack's shift-invert
#  `eigs(A0, M0; sigma = 0)`.  On this tutorial's finest cases that Arpack path
#  does NOT converge -- P2 on unit_square_64 (n_int = 16129) fails with
#  "XYAUPD_Exception: Maximum number of iterations taken".  So for the large
#  cases we run our own shift-invert block iteration below.  It needs nothing
#  beyond LinearAlgebra and SparseArrays and is short enough to read.

"""
    smallest_eigs_shift_invert(A0, M0, k; block, maxit, tol) -> (lam, V, iters)

First `k` eigenvalues of the SPD pencil `A0 x = lambda M0 x`, smallest first.

This is shift-invert block (simultaneous inverse) iteration with Rayleigh-Ritz:

  1. Factor `A0` once with a sparse Cholesky.  `A0` is SPD precisely because the
     Dirichlet rows/columns have been removed, so this always succeeds.
  2. Repeat: `Y <- A0^{-1} M0 X`.  Because `A0^{-1} M0` has eigenvalues
     `1/lambda_i`, this amplifies the SMALLEST eigenvalues of the pencil -- the
     ones we want -- most strongly.
  3. `M0`-orthonormalise `Y` (`Y' M0 Y = I`) and solve the small `b x b`
     projected symmetric problem `Y' A0 Y`.  Its eigenvalues are the Ritz
     values; its eigenvectors rotate `X` into the next iterate.

  Convergence of eigenvalue `k` is linear with factor `(lambda_k/lambda_{b+1})`,
  so a block `b` comfortably larger than `k` converges fast.  The starting block
  is a fixed deterministic formula rather than a random matrix, so reruns are
  bit-identical.
"""
function smallest_eigs_shift_invert(A0::SparseMatrixCSC, M0::SparseMatrixCSC,
                                    k::Integer; block::Integer = k + 10,
                                    maxit::Integer = 400, tol::Real = 1e-13)
    n = size(A0, 1)
    b = min(max(block, k + 2), n)
    F = cholesky(Symmetric(A0))
    X = Float64[sin(0.7 * i * j + 0.3 * j) for i in 1:n, j in 1:b]
    lam_old = fill(Inf, k)
    lam = Float64[]; V = zeros(n, b)
    for it in 1:maxit
        Y = F \ (M0 * X)                       # inverse iteration
        R = cholesky(Symmetric(Y' * (M0 * Y))).U
        Y = Y / R                              # now Y' M0 Y = I
        E = eigen(Symmetric(Y' * (A0 * Y)))    # small projected problem
        lam = E.values
        X = Y * E.vectors
        V = X
        if maximum(abs.(lam[1:k] .- lam_old) ./ abs.(lam[1:k])) < tol
            return lam[1:k], V[:, 1:k], it
        end
        lam_old = lam[1:k]
    end
    return lam[1:k], V[:, 1:k], maxit
end

"""
    interior_count(m, p) -> (n_int, ndof)

Interior and total Lagrange DOF counts, computed from mesh counts alone so we
can pick a solver path without paying for an assembly first.
"""
function interior_count(m::Mesh2D, p::Integer)
    nbdv = length(unique(vec(m.bd_edges)))
    ndof = p == 1 ? m.nv : m.nv + m.ne
    nbd  = p == 1 ? nbdv : nbdv + m.nb
    return ndof - nbd, ndof
end

"""
    eig_pairs(m, p, neig) -> NamedTuple(lam, U, A, M, bd, ndof, nint, path)

`laplace_eig_lagrange` when its dense path applies (n_int <= 6000), our
shift-invert iteration otherwise.  `U` is always ndof x neig with zeros on the
Dirichlet DOFs, so downstream plotting code does not care which path ran.
"""
function eig_pairs(m::Mesh2D, p::Integer, neig::Integer)
    nint, _ = interior_count(m, p)
    if nint <= 6000
        r = laplace_eig_lagrange(m, p, neig)
        return (; lam = collect(r.eig_value), U = r.eig_func, A = r.A, M = r.M,
                  bd = r.bd_dofs, ndof = size(r.A, 1),
                  nint = size(r.A, 1) - length(r.bd_dofs), path = "library-dense")
    end
    A, M, bd = lagrange_laplace_matrices(m, p)
    ndof = size(A, 1)
    isbd = falses(ndof); for d in bd; isbd[d] = true; end
    int = findall(!, isbd)
    lam, V, its = smallest_eigs_shift_invert(A[int, int], M[int, int], neig)
    U = zeros(Float64, ndof, neig); U[int, :] .= V
    return (; lam, U, A, M, bd, ndof, nint = length(int),
              path = "shift-invert(iters=$its)")
end

# -----------------------------------------------------------------------------
#  One family sweep
# -----------------------------------------------------------------------------

function sweep(folders, levels, exact; ps = (1, 2))
    rows = NamedTuple[]
    for p in ps, (folder, lev) in zip(folders, levels)
        m = mesh2d_load(joinpath(MESHROOT, folder))
        # A mesh with fewer than NEIG+1 interior DOFs cannot supply NEIG
        # eigenvalues at all: `laplace_eig_lagrange` clamps to
        # k_eff = min(neig, n_int - 1) and returns a shorter vector.  The
        # coarsest equilateral mesh at P1 has n_int = 3, so skip such levels
        # rather than tabulate a ragged row.
        nint_pre, _ = interior_count(m, p)
        if nint_pre < NEIG + 1
            @printf("  %-22s p=%d  SKIPPED: only %d interior DOFs (< NEIG+1 = %d)\n",
                    folder, p, nint_pre, NEIG + 1)
            continue
        end
        # NOTE: find_mesh_hmax has NO Mesh2D overload -- it takes the two
        # arrays.  mesh_hmax(m) from TutorialSupport is the thin wrapper.
        h = mesh_hmax(m)
        t0 = time_ns()
        r = eig_pairs(m, p, NEIG)
        el = (time_ns() - t0) / 1e9
        ndof = r.ndof
        nint = r.nint
        lam = r.lam
        @printf("  %-22s p=%d  ndof=%6d  n_int=%6d  hmax=%.6e  %6.2f s  [%s]\n",
                folder, p, ndof, nint, h, el, r.path)
        for k in 1:min(NEIG, length(lam))
            @printf("      lambda_%d = %.10f   exact = %.10f   abs = %.3e   rel = %.3e   above = %s\n",
                    k, lam[k], exact[k], lam[k] - exact[k],
                    (lam[k] - exact[k]) / exact[k],
                    lam[k] >= exact[k] ? "yes" : "NO")
        end
        push!(rows, (; folder, level = lev, p, ndof, nint, hmax = h,
                       lam = collect(lam), secs = el, mesh = m,
                       U = r.U, path = r.path))
    end
    return rows
end

# -----------------------------------------------------------------------------
#  Wide CSV: one row per (family, p, level)
# -----------------------------------------------------------------------------

function write_csv(path, family, rows, exact; xkey = :hmax, xname = "hmax")
    open(path, "w") do io
        hdr = String["family", "p", "level", "ndof", "n_int", xname]
        for k in 1:NEIG; push!(hdr, "lambda_$k");     end
        for k in 1:NEIG; push!(hdr, "exact_$k");      end
        for k in 1:NEIG; push!(hdr, "abs_err_$k");    end
        for k in 1:NEIG; push!(hdr, "rel_err_$k");    end
        for k in 1:NEIG; push!(hdr, "order_$k");      end
        push!(hdr, "all_above_exact", "secs")
        println(io, join(hdr, ","))

        for p in sort(unique(getfield.(rows, :p)))
            sub = [r for r in rows if r.p == p]
            xs  = [getfield(r, xkey) for r in sub]
            # observed order per eigenvalue index, from consecutive levels
            ords = [observed_order([r.lam[k] - exact[k] for r in sub], xs)
                    for k in 1:NEIG]
            for (i, r) in enumerate(sub)
                f = String[family, string(p), string(r.level),
                           string(r.ndof), string(r.nint),
                           @sprintf("%.10e", getfield(r, xkey))]
                for k in 1:NEIG; push!(f, @sprintf("%.12f", r.lam[k])); end
                for k in 1:NEIG; push!(f, @sprintf("%.12f", exact[k])); end
                for k in 1:NEIG; push!(f, @sprintf("%.6e", r.lam[k] - exact[k])); end
                for k in 1:NEIG; push!(f, @sprintf("%.6e", (r.lam[k]-exact[k])/exact[k])); end
                for k in 1:NEIG
                    push!(f, i == 1 ? "" : @sprintf("%.4f", ords[k][i-1]))
                end
                push!(f, all(r.lam[k] >= exact[k] for k in 1:NEIG) ? "yes" : "NO")
                push!(f, @sprintf("%.3f", r.secs))
                println(io, join(f, ","))
            end
        end
    end
    println("wrote $path")
end

# -----------------------------------------------------------------------------
#  Eigenfunction figures.  eig_func columns are MONOMIAL-basis Lagrange
#  coefficients; fe_vertex_values extracts vertex values correctly for p = 1
#  and p = 2.  Eigenfunction sign and scale are arbitrary, so fix them:
#  scale so that max |u| over vertices is +1.
# -----------------------------------------------------------------------------

function normalized_mode(m::Mesh2D, U::AbstractMatrix, k::Integer, p::Integer)
    u  = collect(U[:, k])
    vv = fe_vertex_values(m, u, p)
    i  = argmax(abs.(vv))
    s  = vv[i] < 0 ? -1.0 : 1.0
    mx = maximum(abs.(vv)); mx == 0 && (mx = 1.0)
    return (s / mx) .* u
end

# =============================================================================
#  (a) UNIT SQUARE
# =============================================================================

println("=" ^ 78)
println("UNIT SQUARE (0,1)^2 -- Dirichlet Laplace eigenvalues")
println("  exact lambda_{m,n} = pi^2 (m^2 + n^2)")
sq_exact = square_exact(NEIG)
for k in 1:NEIG
    @printf("  exact lambda_%d = %.10f\n", k, sq_exact[k])
end
println("=" ^ 78)

sq_levels  = [4, 8, 16, 32, 64]
sq_folders = ["unit_square_$n" for n in sq_levels]
sq_rows    = sweep(sq_folders, sq_levels, sq_exact)
write_csv(joinpath(OUT, "eig_square.csv"), "unit_square", sq_rows, sq_exact)

# ---- teaching point 1: every computed value is an UPPER bound ---------------
println("\n--- Galerkin upper-bound check (all lambda_k^h >= lambda_k) ---")
viol = 0
for r in sq_rows, k in 1:NEIG
    if r.lam[k] < sq_exact[k]
        global viol += 1
        @printf("  VIOLATION %s p=%d k=%d: %.12f < %.12f\n",
                r.folder, r.p, k, r.lam[k], sq_exact[k])
    end
end
@printf("  %d violations out of %d (family, p, k) combinations\n",
        viol, length(sq_rows) * NEIG)

# ---- teaching point 2: order 2 for P1, order 4 for P2 ----------------------
println("\n--- observed order of lambda_1 error vs hmax ---")
for p in (1, 2)
    sub = [r for r in sq_rows if r.p == p]
    er  = [r.lam[1] - sq_exact[1] for r in sub]
    hs  = [r.hmax for r in sub]
    o   = observed_order(er, hs)
    @printf("  P%d  errors: %s\n", p, join((@sprintf("%.4e", e) for e in er), "  "))
    @printf("  P%d  orders: %s   (expected %d)\n", p,
            join((@sprintf("%.4f", x) for x in o), "  "), 2 * p)
end

# ---- figures ---------------------------------------------------------------
# Use the level-32 P1 result: fine enough for smooth contours, coarse enough
# that the SVG stays small.
let r = first(q for q in sq_rows if q.p == 1 && q.level == 32)
    m = r.mesh
    for k in 1:4
        u = normalized_mode(m, r.U, k, 1)
        svg_solution(joinpath(OUT, "eig_square_mode$(k).svg"), m, u; p = 1,
                     title = @sprintf("Unit square, mode %d: lambda_%d^h = %.6f (exact %.6f)",
                                      k, k, r.lam[k], sq_exact[k]))
    end
    println("wrote eig_square_mode1..4.svg")
end
let r = first(q for q in sq_rows if q.p == 1 && q.level == 16)
    svg_mesh(joinpath(OUT, "eig_square_mesh16.svg"), r.mesh;
             title = "unit_square_16 (nv=$(r.mesh.nv), nt=$(r.mesh.nt))")
    println("wrote eig_square_mesh16.svg")
end

# log-log convergence, lambda_1 and lambda_4, P1 and P2
let series = Any[]
    for p in (1, 2)
        sub = [r for r in sq_rows if r.p == p]
        push!(series, (x = [r.hmax for r in sub],
                       y = [r.lam[1] - sq_exact[1] for r in sub],
                       label = "P$p  lambda_1", slope = 2.0 * p))
        push!(series, (x = [r.hmax for r in sub],
                       y = [r.lam[4] - sq_exact[4] for r in sub],
                       label = "P$p  lambda_4", slope = nothing))
    end
    svg_loglog(joinpath(OUT, "eig_square_conv.svg"), series;
               xlabel = "hmax", ylabel = "lambda_k^h - lambda_k",
               title = "Unit square: eigenvalue error, O(h^2) for P1 and O(h^4) for P2")
    println("wrote eig_square_conv.svg")
end

# =============================================================================
#  (b) EQUILATERAL TRIANGLE (side 1)
#      A convex domain with a *genuinely degenerate* spectrum: lambda_2 and
#      lambda_3 coincide exactly (the (2,1) mode and its mirror image).  A good
#      test that the eigensolver resolves a multiplicity-2 eigenvalue rather
#      than splitting it artificially.  Note carefully what accuracy to expect:
#      the two computed values agree to ~1e-13 RELATIVE, i.e. to machine
#      precision -- far better than the ~1e-2 discretisation error in the
#      eigenvalue itself.  The reason is that the structured equilateral mesh is
#      exactly invariant under the reflection that maps one member of the pair
#      to the other, so the DISCRETE eigenvalue is exactly double too.  The
#      degeneracy is inherited from the mesh symmetry, not merely approximated.
#      On an unstructured mesh the same pair would split at the discretisation
#      error level instead.
# =============================================================================

println("\n" * "=" ^ 78)
println("EQUILATERAL TRIANGLE, side a = 1  (vertices (0,0), (1,0), (0.5, sqrt(3)/2))")
println("  exact lambda_{m,n} = (16 pi^2 / (9 a^2)) (m^2 + m n + n^2), m >= n >= 1")
eq_exact = equilateral_exact(NEIG; a = 1.0)
for k in 1:NEIG
    @printf("  exact lambda_%d = %.10f\n", k, eq_exact[k])
end
println("=" ^ 78)

eq_levels  = [4, 8, 16, 32]
eq_folders = ["equilateral_$n" for n in eq_levels]
eq_rows    = sweep(eq_folders, eq_levels, eq_exact)
write_csv(joinpath(OUT, "eig_equilateral.csv"), "equilateral", eq_rows, eq_exact)

println("\n--- multiplicity-2 check: lambda_2 == lambda_3 (exact value $(round(eq_exact[2], digits=6))) ---")
println("    the structured mesh is exactly reflection-symmetric, so the discrete")
println("    pair is degenerate to machine precision, not merely to O(h^2p)")
for r in eq_rows
    @printf("  %-18s p=%d  lambda_2 = %.10f  lambda_3 = %.10f  |gap| = %.3e  gap/lambda = %.3e\n",
            r.folder, r.p, r.lam[2], r.lam[3],
            abs(r.lam[3] - r.lam[2]), abs(r.lam[3] - r.lam[2]) / eq_exact[2])
end
println("  (also lambda_5 == lambda_6, the (3,1) pair)")
for r in eq_rows
    @printf("  %-18s p=%d  lambda_5 = %.10f  lambda_6 = %.10f  |gap| = %.3e\n",
            r.folder, r.p, r.lam[5], r.lam[6], abs(r.lam[6] - r.lam[5]))
end

println("\n--- observed order of lambda_1 error vs hmax (equilateral) ---")
for p in (1, 2)
    sub = [r for r in eq_rows if r.p == p]
    er  = [r.lam[1] - eq_exact[1] for r in sub]
    o   = observed_order(er, [r.hmax for r in sub])
    @printf("  P%d  errors: %s\n", p, join((@sprintf("%.4e", e) for e in er), "  "))
    @printf("  P%d  orders: %s   (expected %d)\n", p,
            join((@sprintf("%.4f", x) for x in o), "  "), 2 * p)
end

let r = first(q for q in eq_rows if q.p == 1 && q.level == 32), series = Any[]
    for k in 1:3
        u = normalized_mode(r.mesh, r.U, k, 1)
        svg_solution(joinpath(OUT, "eig_equilateral_mode$(k).svg"), r.mesh, u; p = 1,
                     title = @sprintf("Equilateral, mode %d: lambda_%d^h = %.5f (exact %.5f)",
                                      k, k, r.lam[k], eq_exact[k]))
    end
    for p in (1, 2)
        sub = [q for q in eq_rows if q.p == p]
        push!(series, (x = [q.hmax for q in sub],
                       y = [q.lam[1] - eq_exact[1] for q in sub],
                       label = "P$p  lambda_1", slope = 2.0 * p))
    end
    svg_loglog(joinpath(OUT, "eig_equilateral_conv.svg"), series;
               xlabel = "hmax", ylabel = "lambda_1^h - lambda_1",
               title = "Equilateral triangle: eigenvalue error")
    println("wrote eig_equilateral_mode1..3.svg, eig_equilateral_conv.svg")
end

println("\n03_eig_square.jl DONE")
