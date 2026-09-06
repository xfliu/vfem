# tutorial/examples/10_element_spaces.jl
# ============================================================================
#  Every 2D element family in VFEM.jl on ONE eigenproblem
# ============================================================================
#
#  Problem.  Dirichlet Laplace eigenproblem on the unit square
#
#       -Delta u = lambda u  in  Omega = (0,1)^2,      u = 0 on dOmega,
#
#  whose exact spectrum is  lambda_{m,n} = pi^2 (m^2 + n^2), so
#
#       lambda_1 = 2 pi^2 = 19.739208802178716 .
#
#  We assemble the SAME problem with every family the library ships and ask
#  three questions:
#
#    (1) how many unknowns does the family cost, and where do they live?
#    (2) what is lambda_1(h), and is it ABOVE or BELOW the exact value?
#    (3) at equal unknowns, which family is most accurate?
#
#  Question (2) is the one that matters for verified computing. A *conforming*
#  space V_h is a subspace of H^1_0(Omega), so the Rayleigh quotient is
#  minimised over a smaller set than the true one and the Galerkin eigenvalue
#  can only come out too LARGE: lambda_1^h >= lambda_1, an upper bound, always.
#  A *non-conforming* space (Crouzeix-Raviart) is NOT a subspace of H^1_0 --
#  its functions jump across edges -- so that argument is void and lambda_1^h
#  may fall BELOW lambda_1. That is not a defect: it is the raw material for
#  Liu's rigorous lower bound
#
#       lambda_1 >= lambda_1^{CR} / (1 + C_h^2 lambda_1^{CR}),
#
#  which needs a computed quantity that is already on the correct side.
#
#  RUN:
#     export PATH=$HOME/.juliaup/bin:$PATH
#     julia --project=. tutorial/10_element_spaces.jl
#
#  `--project` is mandatory (VFEM and Veigs are unregistered path packages).
# ============================================================================

using VFEM
using LinearAlgebra: Symmetric, eigen, cholesky, qr, isdiag, norm, I
using SparseArrays: SparseMatrixCSC, nnz
using Printf
using Random: seed!

# ---------------------------------------------------------------------------
# Where things live. Both paths are read-only inputs.
# ---------------------------------------------------------------------------
const MESHROOT = joinpath(@__DIR__, "meshdata")
const SUPPORT  = let cand = ["TutorialSupport.jl",
                             joinpath(@__DIR__, "TutorialSupport.jl")]
    i = findfirst(isfile, cand)
    i === nothing && error("TutorialSupport.jl not found in $cand")
    cand[i]
end
include(SUPPORT)
using .TutorialSupport: svg_loglog, PALETTE

const LAMBDA1_EXACT = 2 * pi^2           # = 19.739208802178716

# ===========================================================================
# 0.  A small generalised eigensolver we can trust at every size
# ===========================================================================
#
#  VFEM's own `laplace_eig_lagrange` switches to Arpack above n_int = 6000 and
#  that path has been observed to hit `Maximum number of iterations taken` on
#  this very problem. Since we compare EIGHT families across FOUR mesh levels
#  we want one solver whose behaviour does not change under us. Below
#  `dense_cutoff` we use LAPACK's dense generalised solver directly; above it
#  we run shift-invert block inverse iteration with Rayleigh-Ritz projection,
#  which needs only a sparse Cholesky of A0 (SPD once Dirichlet DOFs are gone).
#
#  Both branches solve the same pencil A0 x = lambda M0 x with A0, M0
#  symmetric positive definite; §1 below verifies the two agree.

"""
    smallest_eigs(A0, M0, k; ...) -> (lambda::Vector, X::Matrix, info)

`k` smallest eigenvalues of the SPD pencil `A0 x = lambda M0 x`, ascending.
`info` carries `(; method, iters, resid)` where `resid` is the largest
relative residual `||A0 x - lambda M0 x|| / ||A0 x||` over the returned pairs.
"""
function smallest_eigs(A0::AbstractMatrix, M0::AbstractMatrix, k::Integer;
                       block::Integer = k + 6, maxit::Integer = 400,
                       tol::Float64 = 1e-12, dense_cutoff::Integer = 1200,
                       seed::Integer = 20240)
    n = size(A0, 1)
    1 <= k <= n || throw(ArgumentError("need 1 <= k <= n (k=$k, n=$n)"))

    resid(lam, X) = maximum(1:size(X, 2)) do j
        x = @view X[:, j]
        norm(A0 * x - lam[j] * (M0 * x)) / max(norm(A0 * x), eps())
    end

    if n <= dense_cutoff
        F = eigen(Symmetric(Matrix(A0)), Symmetric(Matrix(M0)))
        lam = F.values[1:k]; X = F.vectors[:, 1:k]
        return lam, X, (; method = :dense, iters = 0, resid = resid(lam, X))
    end

    b = min(Int(block), n)
    C = cholesky(Symmetric(A0))              # sparse SPD factorisation, reused
    seed!(seed)
    X = randn(n, b)
    lam_old = fill(Inf, k)
    lam = similar(lam_old)
    it_used = maxit
    for it in 1:maxit
        Y  = C \ (M0 * X)                    # Y ~ A0^{-1} M0 X  (inverse iter.)
        # Thin Q of the n x b block. `Matrix(qr(Y).Q)` is version-dependent in
        # whether it returns the thin or the square factor; multiplying the
        # implicit Q by a thin identity always gives the n x b factor.
        Q  = qr(Y).Q * Matrix{Float64}(I, size(Y, 1), size(Y, 2))
        Ak = Symmetric(Q' * (A0 * Q))        # Rayleigh-Ritz on that subspace
        Mk = Symmetric(Q' * (M0 * Q))
        E  = eigen(Ak, Mk)
        lam .= E.values[1:k]
        X = Q * E.vectors                    # all b Ritz vectors -> next iterate
        if maximum(abs.(lam .- lam_old) ./ abs.(lam)) < tol
            it_used = it; break
        end
        lam_old .= lam
    end
    Xk = X[:, 1:k]
    return lam, Xk, (; method = :shift_invert, iters = it_used, resid = resid(lam, Xk))
end

side_of(lam) = lam > LAMBDA1_EXACT ? "above" : (lam < LAMBDA1_EXACT ? "below" : "exact")
relerr(lam)  = abs(lam - LAMBDA1_EXACT) / LAMBDA1_EXACT

hr(t) = println("\n" * "="^76 * "\n  " * t * "\n" * "="^76)

# ===========================================================================
# 1.  The return-order trap: which matrix is the mass matrix?
# ===========================================================================
#
#  This is the single easiest way to get a silently wrong answer out of
#  VFEM.jl, so we check it at runtime instead of trusting our memory:
#
#     create_matrix_crouzeix_raviart(m)  ->  (A0, A1) = (MASS, STIFFNESS)
#     create_matrix_ecr(m)               ->  (A,  M ) = (STIFFNESS, MASS)
#
#  Two intrinsic fingerprints tell them apart, no reference values needed:
#    * the CR mass matrix is DIAGONAL (|K|/3 per edge DOF, no edge couples to
#      another inside an element) and its total sum is |Omega|;
#    * any stiffness matrix ANNIHILATES CONSTANTS, so its row sums vanish --
#      the constant function is in the kernel of the Laplacian.

hr("1. Return order is not consistent -- verify it at runtime")

m8 = mesh2d_load(joinpath(MESHROOT, "unit_square_8"))
println("mesh: ", m8, "   h_max = ", find_mesh_hmax(m8.nodes, m8.edges))

X1, X2 = create_matrix_crouzeix_raviart(m8)
@printf("\ncreate_matrix_crouzeix_raviart -> (X1, X2)\n")
@printf("  X1: isdiag = %-5s  sum = %+.15e   max|rowsum| = %.3e\n",
        isdiag(X1), sum(X1), maximum(abs, sum(X1, dims = 2)))
@printf("  X2: isdiag = %-5s  sum = %+.15e   max|rowsum| = %.3e\n",
        isdiag(X2), sum(X2), maximum(abs, sum(X2, dims = 2)))
cr_mass_is_first = isdiag(X1) && maximum(abs, sum(X2, dims = 2)) < 1e-10
@printf("  => X1 is the %s, X2 is the %s   (mass first: %s)\n",
        cr_mass_is_first ? "MASS" : "STIFFNESS",
        cr_mass_is_first ? "STIFFNESS" : "MASS", cr_mass_is_first)
M_cr, A_cr = X1, X2                        # name them only after the check

Y1, Y2 = create_matrix_ecr(m8)
@printf("\ncreate_matrix_ecr -> (Y1, Y2)\n")
@printf("  Y1: isdiag = %-5s  sum = %+.15e   max|rowsum| = %.3e\n",
        isdiag(Y1), sum(Y1), maximum(abs, sum(Y1, dims = 2)))
@printf("  Y2: isdiag = %-5s  sum = %+.15e   max|rowsum| = %.3e\n",
        isdiag(Y2), sum(Y2), maximum(abs, sum(Y2, dims = 2)))
ecr_stiff_is_first = maximum(abs, sum(Y1, dims = 2)) < 1e-10
@printf("  => Y1 is the %s, Y2 is the %s   (stiffness first: %s)\n",
        ecr_stiff_is_first ? "STIFFNESS" : "MASS",
        ecr_stiff_is_first ? "MASS" : "STIFFNESS", ecr_stiff_is_first)

println("""
  The ECR mass matrix is NOT diagonal (the cell-average DOF couples to the
  three edge DOFs of its element), so `isdiag` alone cannot classify it --
  the vanishing-row-sum test is the reliable one and works for every family.
  Swapping A and M turns lambda into 1/lambda; on this mesh that would report
  lambda_1 ~ 0.05 instead of ~ 19.7, which is obvious. On a Schrodinger
  problem with a potential it is NOT obvious. Always check.""")

# Cross-check the two eigensolver branches on one medium pencil, so that every
# number further down rests on a verified-consistent solver.
hr("1b. Dense vs shift-invert branch agree on the same pencil")
let m = mesh2d_load(joinpath(MESHROOT, "unit_square_16"))
    Mc, Ac = create_matrix_crouzeix_raviart(m)
    int = setdiff(1:m.ne, m.bd_edge_ids)
    A0, M0 = Ac[int, int], Mc[int, int]
    ld, _, id = smallest_eigs(A0, M0, 6; dense_cutoff = typemax(Int))
    ls, _, is = smallest_eigs(A0, M0, 6; dense_cutoff = 0)
    @printf("  n_int = %d\n", length(int))
    @printf("  dense        : lambda_1 = %.12f  resid = %.2e\n", ld[1], id.resid)
    @printf("  shift-invert : lambda_1 = %.12f  resid = %.2e  (%d iters)\n",
            ls[1], is.resid, is.iters)
    @printf("  max rel. difference over 6 eigenvalues: %.3e\n",
            maximum(abs.(ls .- ld) ./ ld))
end

# ===========================================================================
# 2.  One assembler per family, behind a uniform interface
# ===========================================================================
#
#  Each entry returns the interior-restricted pencil (A0, M0) plus bookkeeping.
#  The DOF restriction differs per family and that is the point:
#
#   CR                     ndof = ne         one DOF per EDGE (edge average).
#                                            Boundary DOFs = m.bd_edge_ids.
#   ECR (quadrature)       ndof = ne + nt    edges 1:ne then cells ne+1:ne+nt,
#                                            the LEGACY ordering -> restrict by
#                                            hand with setdiff.
#   ECR (exact Bernstein)  ndof = ne + nt    PERMUTED ordering from
#   CECR                                     `build_ecr_dof_ordering` -> you
#                                            MUST use `interior_ecr_dofs`.
#   Lagrange P1            ndof = nv         VERTICES.
#   Lagrange P2            ndof = nv + ne    vertices + one per edge.
#
#  Both Lagrange assemblies appear: `lagrange_laplace_matrices` (monomial
#  basis L1^i L2^j L3^k, the basis the Lehmann-Goerisch pipeline consumes) and
#  `create_matrix_lagrange` (nodal basis, the one the CECR pipeline uses).
#  They span the SAME space, so their eigenvalues must agree to round-off
#  while their matrix entries do not -- a good invariant to display.

"Boundary/interior DOF split for the nodal Lagrange assembler."
function lagrange_nodal_dofs(m::Mesh2D, p::Integer)
    ndof = p == 1 ? m.nv : m.nv + m.ne
    isbd = falses(ndof)
    for r in 1:m.nb
        isbd[m.bd_edges[r, 1]] = true
        isbd[m.bd_edges[r, 2]] = true
    end
    if p == 2
        for eid in m.bd_edge_ids
            isbd[m.nv + eid] = true
        end
    end
    return ndof, findall(isbd), findall(!, isbd)
end

# name => (conforming?, dof_location, assemble(m) -> (A0, M0, ndof, n_bd))
const FAMILIES = [
 ("CR  (create_matrix_crouzeix_raviart)", false, "edge midpoints (1/edge)",
  function (m)
      Mx, Ax = create_matrix_crouzeix_raviart(m)          # MASS FIRST
      int = setdiff(1:m.ne, m.bd_edge_ids)
      return Ax[int, int], Mx[int, int], m.ne, m.ne - length(int)
  end),

 ("ECR (create_matrix_ecr)", false, "edges + cells (enriched)",
  function (m)
      Ax, Mx = create_matrix_ecr(m)                        # STIFFNESS FIRST
      int = setdiff(1:(m.ne + m.nt), m.bd_edge_ids)        # legacy ordering
      return Ax[int, int], Mx[int, int], m.ne + m.nt, m.ne + m.nt - length(int)
  end),

 ("ECR exact (create_matrix_enriched_crouzeix_raviart)", false,
  "edges + cells (enriched)",
  function (m)
      Ax, Mx, dof = create_matrix_enriched_crouzeix_raviart(m)
      int = interior_ecr_dofs(m, dof)                      # permuted ordering!
      return restrict_to_interior(Ax, int), restrict_to_interior(Mx, int),
             m.ne + m.nt, m.ne + m.nt - length(int)
  end),

 ("CECR c=0 (create_matrix_cecr)", false, "edges + cells (enriched)",
  function (m)
      Ax, Mx, dof = create_matrix_cecr(m, 0.0)
      int = interior_ecr_dofs(m, dof)
      return restrict_to_interior(Ax, int), restrict_to_interior(Mx, int),
             m.ne + m.nt, m.ne + m.nt - length(int)
  end),

 ("Lagrange P1 monomial (lagrange_laplace_matrices)", true, "vertices",
  function (m)
      Ax, Mx, bd = lagrange_laplace_matrices(m, 1)
      int = setdiff(1:size(Ax, 1), bd)
      return Ax[int, int], Mx[int, int], size(Ax, 1), length(bd)
  end),

 ("Lagrange P2 monomial (lagrange_laplace_matrices)", true, "vertices + edges",
  function (m)
      Ax, Mx, bd = lagrange_laplace_matrices(m, 2)
      int = setdiff(1:size(Ax, 1), bd)
      return Ax[int, int], Mx[int, int], size(Ax, 1), length(bd)
  end),

 ("Lagrange P1 nodal (create_matrix_lagrange)", true, "vertices",
  function (m)
      Ax, Mx = create_matrix_lagrange(m, 1, zeros(m.nt, 15))   # V_bern is nt x 15
      _, bd, int = lagrange_nodal_dofs(m, 1)
      return Ax[int, int], Mx[int, int], size(Ax, 1), length(bd)
  end),

 ("Lagrange P2 nodal (create_matrix_lagrange)", true, "vertices + edges",
  function (m)
      Ax, Mx = create_matrix_lagrange(m, 2, zeros(m.nt, 15))
      _, bd, int = lagrange_nodal_dofs(m, 2)
      return Ax[int, int], Mx[int, int], size(Ax, 1), length(bd)
  end),
]

"Assemble + solve one family on one mesh. Returns a NamedTuple row."
function run_family(name, conforming, where_dofs, assemble, m; neig = 3)
    t_asm = @elapsed ((A0, M0, ndof, n_bd) = assemble(m))
    n_int = size(A0, 1)
    k = min(neig, n_int)
    t_eig = @elapsed ((lam, _, info) = smallest_eigs(A0, M0, k))
    return (; name, conforming, where_dofs, ndof, n_bd, n_int,
              nnz_A = nnz(A0), lambda1 = lam[1],
              lambda = lam, err = relerr(lam[1]), side = side_of(lam[1]),
              t_asm, t_eig, method = String(info.method), resid = info.resid)
end

# ===========================================================================
# 3.  The comparison table, on the 8x8 mesh
# ===========================================================================

hr("2. All families on unit_square_8  (equivalent to the shipped 8x8 fixture)")

rows8 = [run_family(f..., m8) for f in FAMILIES]

@printf("\n%-52s %6s %5s %6s %-22s %18s %11s %7s %8s\n",
        "family", "ndof", "bd", "n_int", "DOFs live on",
        "lambda_1", "rel.err", "side", "eig [s]")
println("-"^148)
for r in rows8
    @printf("%-52s %6d %5d %6d %-22s %18.12f %11.3e %7s %8.3f\n",
            r.name, r.ndof, r.n_bd, r.n_int, r.where_dofs,
            r.lambda1, r.err, r.side, r.t_eig)
end
println("-"^148)
@printf("exact %-46s %6s %5s %6s %-22s %18.12f\n",
        "lambda_1 = 2 pi^2", "", "", "", "", LAMBDA1_EXACT)

# The two Lagrange assemblies must agree on eigenvalues and differ on entries.
let mo = rows8[5], no = rows8[7], mo2 = rows8[6], no2 = rows8[8]
    hr("2b. Monomial vs nodal Lagrange: same space, different matrices")
    @printf("  P1: |lambda_1(monomial) - lambda_1(nodal)| = %.3e\n",
            abs(mo.lambda1 - no.lambda1))
    @printf("  P2: |lambda_1(monomial) - lambda_1(nodal)| = %.3e\n",
            abs(mo2.lambda1 - no2.lambda1))
    Am, _, _ = lagrange_laplace_matrices(m8, 2)
    An, _    = create_matrix_lagrange(m8, 2, zeros(m8.nt, 15))
    @printf("  P2 stiffness: ||A_monomial - A_nodal||_F / ||A_nodal||_F = %.3e\n",
            norm(Am - An) / norm(An))
    println("""
  Eigenvalues agree to round-off (they are a property of the SPACE); the
  matrices do not (they are a property of the BASIS). If you need coefficient
  vectors -- to plot, or to feed `rt_hdiv_problem` -- you must know which
  assembler produced them.""")
end

# The CECR reaction term, for context: c > 0 raises every eigenvalue.
hr("2c. What the `c` in CECR does")
for c in (0.0, 1.0, 10.0)
    Ax, Mx, dof = create_matrix_cecr(m8, c)
    int = interior_ecr_dofs(m8, dof)
    lam, _, _ = smallest_eigs(restrict_to_interior(Ax, int),
                              restrict_to_interior(Mx, int), 1)
    @printf("  c = %5.1f  ->  lambda_1 = %.12f   (shift vs c=0 tracks (c u, u))\n",
            c, lam[1])
end
println("""
  CECR = ECR stiffness + (c*Pi_0 u, Pi_0 v) with Pi_0 the elementwise L2
  projection onto constants. Because the cell-average DOF *is* that
  projection, the extra term is a pure diagonal on cell DOFs, c_K*|K|. This is
  how the CECR pipeline carries a Schrodinger potential; with c = 0 it is
  exactly the ECR Laplacian, which is why rows 3 and 4 of the table match.""")

# ===========================================================================
# 4.  Convergence, and accuracy per unknown
# ===========================================================================

hr("3. Convergence over four mesh levels")

const LEVELS = [4, 8, 16, 32]
const CONV_FAMILIES = [FAMILIES[1], FAMILIES[3], FAMILIES[5], FAMILIES[6]]
const SHORT = Dict(FAMILIES[1][1] => "CR",
                   FAMILIES[3][1] => "ECR",
                   FAMILIES[5][1] => "Lagrange P1",
                   FAMILIES[6][1] => "Lagrange P2")

conv = Dict{String, Vector{NamedTuple}}()
for f in CONV_FAMILIES
    key = SHORT[f[1]]
    conv[key] = NamedTuple[]
    for n in LEVELS
        m = mesh2d_load(joinpath(MESHROOT, "unit_square_$(n)"))
        r = run_family(f..., m; neig = 1)
        push!(conv[key], (; n, hmax = find_mesh_hmax(m.nodes, m.edges), r...))
        @printf("  %-12s n=%-3d n_int=%-6d lambda_1 = %18.12f  err = %.4e  %-5s  %-13s %6.2f s\n",
                key, n, r.n_int, r.lambda1, r.err, r.side, r.method, r.t_asm + r.t_eig)
    end
end

hr("3b. Observed order in h, and accuracy at equal unknowns")
for key in ("CR", "ECR", "Lagrange P1", "Lagrange P2")
    rs = conv[key]
    ords = [log(rs[i].err / rs[i+1].err) / log(rs[i].hmax / rs[i+1].hmax)
            for i in 1:length(rs)-1]
    @printf("  %-12s order(h) = %s   (all %s)\n", key,
            join((@sprintf("%.3f", o) for o in ords), ", "),
            join(unique(r.side for r in rs), "/"))
end
println("""
  Every family shows order 2 in h for the eigenvalue: lambda_1 - lambda_1^h
  ~ h^2 for the P1-type spaces (CR, ECR, Lagrange P1) and h^4 for P2. The
  eigenvalue error is the SQUARE of the eigenfunction energy error, which is
  why a P1 space with O(h) energy error still gives O(h^2) eigenvalues.""")

@printf("\n  %-12s %8s %20s %11s %7s %14s\n",
        "family", "n_int", "lambda_1", "rel.err", "side", "err x n_int")
println("  " * "-"^76)
for key in ("CR", "ECR", "Lagrange P1", "Lagrange P2"), r in conv[key]
    @printf("  %-12s %8d %20.12f %11.3e %7s %14.4e\n",
            key, r.n_int, r.lambda1, r.err, r.side, r.err * r.n_int)
end

# Best accuracy at comparable unknown count.
hr("3c. Which family wins at equal unknowns?")
let allr = [(SHORT[f[1]], r) for f in CONV_FAMILIES for r in conv[SHORT[f[1]]]]
    for (lo, hi) in ((150, 450), (700, 1300), (2500, 5500))
        cand = [(k, r) for (k, r) in allr if lo <= r.n_int <= hi]
        isempty(cand) && continue
        sort!(cand, by = kr -> kr[2].err)
        @printf("  n_int in [%d, %d]:\n", lo, hi)
        for (k, r) in cand
            @printf("      %-12s n_int=%-6d err = %.4e  %s\n", k, r.n_int, r.err, r.side)
        end
    end
end

# ===========================================================================
# 5.  CSV + figures
# ===========================================================================

hr("4. Writing element_spaces.csv, element_spaces_convergence.csv, figures")

open("element_spaces.csv", "w") do io
    println(io, "family,conforming,dof_location,ndof,n_boundary_dof,n_interior_dof,",
                "nnz_A_interior,lambda_1,lambda_1_exact,abs_error,rel_error,side,",
                "assemble_seconds,eigensolve_seconds,eig_method,residual")
    for r in rows8
        @printf(io, "%s,%s,%s,%d,%d,%d,%d,%.15e,%.15e,%.6e,%.6e,%s,%.4f,%.4f,%s,%.3e\n",
                r.name, r.conforming ? "yes" : "no", r.where_dofs, r.ndof, r.n_bd,
                r.n_int, r.nnz_A, r.lambda1, LAMBDA1_EXACT,
                r.lambda1 - LAMBDA1_EXACT, r.err, r.side,
                r.t_asm, r.t_eig, r.method, r.resid)
    end
end

open("element_spaces_convergence.csv", "w") do io
    println(io, "family,n,hmax,ndof,n_interior_dof,lambda_1,rel_error,side,",
                "observed_order_h,total_seconds,eig_method")
    for key in ("CR", "ECR", "Lagrange P1", "Lagrange P2")
        rs = conv[key]
        for (i, r) in enumerate(rs)
            ord = i == 1 ? NaN :
                  log(rs[i-1].err / r.err) / log(rs[i-1].hmax / r.hmax)
            @printf(io, "%s,%d,%.15e,%d,%d,%.15e,%.6e,%s,%s,%.3f,%s\n",
                    key, r.n, r.hmax, r.ndof, r.n_int, r.lambda1, r.err, r.side,
                    isnan(ord) ? "" : @sprintf("%.4f", ord),
                    r.t_asm + r.t_eig, r.method)
        end
    end
end

# --- accuracy-per-DOF, log-log ---------------------------------------------
svg_loglog("element_spaces_accuracy_per_dof.svg",
           [(x = [r.n_int for r in conv[k]], y = [r.err for r in conv[k]],
             label = k) for k in ("CR", "ECR", "Lagrange P1", "Lagrange P2")];
           xlabel = "interior DOFs", ylabel = "relative error in lambda_1",
           title = "Accuracy per unknown, Dirichlet Laplace on (0,1)^2",
           width = 660, height = 500)

# --- signed error bar chart, exact value as the zero line ------------------
#
# Hand-written SVG (project decision: no plotting dependency anywhere). The
# chart's whole job is to make the SIDE visible, so the exact eigenvalue is
# the horizontal axis and bars grow up (above) or down (below) from it, on a
# symmetric log scale because the errors span four decades.
function svg_signed_error(path, rows; width = 900, height = 470,
                          decades = 5, floor_exp = -7)
    n = length(rows)
    ml, mr, mt, mb = 74, 22, 46, 150
    pw = width - ml - mr; ph = height - mt - mb
    zero_y = mt + ph / 2
    half = ph / 2
    # signed symlog: map |rel err| in [10^floor_exp, 10^(floor_exp+decades)]
    ypos(e) = begin
        a = clamp(log10(max(abs(e), 10.0^floor_exp)), floor_exp, floor_exp + decades)
        frac = (a - floor_exp) / decades
        sign(e) < 0 ? zero_y + frac * half : zero_y - frac * half
    end
    bw = pw / n * 0.62
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
    println(io, """<rect width="$width" height="$height" fill="white"/>""")
    println(io, """<text x="$(ml + pw/2)" y="24" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#222">Signed error in lambda_1 on unit_square_8; zero line = exact 2 pi^2</text>""")
    # gridlines at each decade, both sides
    println(io, """<g stroke="#e2e2e2" stroke-width="0.9" font-family="sans-serif" font-size="10" fill="#666">""")
    for d in 0:decades
        f = d / decades
        for s in (-1, 1)
            y = zero_y - s * f * half
            println(io, """<line x1="$ml" y1="$y" x2="$(ml+pw)" y2="$y"/>""")
            d == 0 && continue
            println(io, """<text x="$(ml-7)" y="$(y+3.5)" text-anchor="end" stroke="none">$(s>0 ? "+" : "-")1e$(floor_exp+d)</text>""")
        end
    end
    println(io, "</g>")
    println(io, """<line x1="$ml" y1="$zero_y" x2="$(ml+pw)" y2="$zero_y" stroke="#222" stroke-width="1.8"/>""")
    println(io, """<text x="$(ml+pw-4)" y="$(zero_y-7)" text-anchor="end" font-family="sans-serif" font-size="11" fill="#222">exact: lambda_1 = 19.7392088022  (ABOVE = upper bound)</text>""")
    for (i, r) in enumerate(rows)
        xc = ml + pw * (i - 0.5) / n
        e = (r.lambda1 - LAMBDA1_EXACT) / LAMBDA1_EXACT
        y = ypos(e)
        col = r.conforming ? PALETTE[1] : PALETTE[2]
        y0, y1 = min(y, zero_y), max(y, zero_y)
        println(io, """<rect x="$(xc-bw/2)" y="$y0" width="$bw" height="$(max(y1-y0,1.0))" fill="$col" fill-opacity="0.82" stroke="$col" stroke-width="0.8"/>""")
        lab = @sprintf("%+.1e", e)
        ty = e < 0 ? y1 + 13 : y0 - 5
        println(io, """<text x="$xc" y="$ty" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#333">$lab</text>""")
        short = replace(r.name, r" \(.*\)$" => "")
        println(io, """<text x="$xc" y="$(mt+ph+14)" text-anchor="end" font-family="sans-serif" font-size="11" fill="#222" transform="rotate(-38 $xc $(mt+ph+14))">$short</text>""")
        println(io, """<text x="$xc" y="$(mt+ph+130)" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#666">n_int=$(r.n_int)</text>""")
    end
    lx, ly = ml + 12, mt + 14
    for (j, (c, t)) in enumerate(((PALETTE[1], "conforming (always ABOVE)"),
                                  (PALETTE[2], "non-conforming (may be BELOW)")))
        yy = ly + (j - 1) * 17
        println(io, """<rect x="$lx" y="$(yy-9)" width="13" height="11" fill="$c" fill-opacity="0.82"/>""")
        println(io, """<text x="$(lx+19)" y="$yy" font-family="sans-serif" font-size="11" fill="#222">$t</text>""")
    end
    println(io, "</svg>")
    write(path, String(take!(io)))
    return path
end

svg_signed_error("element_spaces_signed_error.svg", rows8)
for f in ("element_spaces.csv", "element_spaces_convergence.csv",
          "element_spaces_accuracy_per_dof.svg", "element_spaces_signed_error.svg")
    @printf("  wrote %-42s %8d bytes\n", f, filesize(f))
end

hr("5. Summary")
println("""
  * Conforming families (Lagrange P1/P2, either basis) landed ABOVE 2 pi^2 at
    every level -- that is the Galerkin upper-bound property and it is a
    theorem, not luck.
  * The non-conforming families are the interesting ones; see the `side`
    column of element_spaces.csv for what actually happened on this mesh, and
    note that no theorem forces them either way.
  * The verified LOWER bound pipeline (`verified_cr_liu_lower`,
    `cr_liu_lower_bounds_3d`) is built on CR precisely because CR's
    non-conformity is what lets a computed number sit on the lower side after
    the h-dependent correction lambda^CR/(1 + C_h^2 lambda^CR) is applied.
  * Per unknown, higher order wins as soon as the solution is smooth: compare
    Lagrange P2 against everything else at comparable n_int in
    element_spaces_convergence.csv.
""")
println("ELEMENT_SPACES_DONE")
