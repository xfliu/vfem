# =============================================================================
#  04_eig_lshape.jl — what a reentrant corner does to the eigenvalue rate,
#                     and how mesh grading repairs it
#
#  Domain (cite this orientation with any reference value!):
#
#      Omega = (-1,1)^2  \  [0,1] x [-1,0]        <-- BOTTOM-RIGHT quadrant cut
#      area 3, one reentrant corner AT THE ORIGIN, interior angle omega = 3pi/2
#      alpha = pi / omega = 2/3
#
#  Reference first eigenvalue for exactly this domain:
#      lambda_1 = 9.6397238440219
#
#  Near the corner the first eigenfunction behaves like r^alpha = r^(2/3)
#  times a smooth angular factor.  That function is in H^1 but NOT in H^2:
#  its second derivatives blow up like r^(alpha-2).  Consequences:
#
#    * On a UNIFORM mesh the energy-norm interpolation error is O(h^alpha),
#      not O(h^p), so by the "double rate" rule the eigenvalue error is
#      O(h^{2 alpha}) = O(h^{4/3}) -- for BOTH P1 and P2.  Raising the
#      polynomial degree buys almost nothing, because the bottleneck is the
#      regularity of the solution, not the approximation power of the space.
#
#    * On a mesh GRADED toward the corner (here beta = 1.5, radial map
#      p -> p * r^(beta-1) with r = max(|x|,|y|)) the local mesh size shrinks
#      fast enough near the origin to equidistribute the error, and the rate is
#      restored.  How much is restored depends on beta: the requirement is
#      beta >= p/alpha, so
#          P1 needs beta >= 1/alpha = 1.5   -- exactly what we ship, full
#                                              O(ndof^-1) = O(h^2) recovered;
#          P2 needs beta >= 2/alpha = 3.0   -- MORE than we ship, so on this
#                                              family P2 lands near order 2 in
#                                              ndof^(-1/2), not 4.
#      That is not a defect in the run, it is the grading exponent showing up
#      in the answer, and it is worth stating plainly: grading is tuned to a
#      (alpha, p) pair, and beta = 1/alpha is the P1 tuning.  Even so P2 on the
#      graded family is ~9x more accurate than P2 on the uniform family at
#      identical ndof, because it has traded a rate of 4/3 for a rate of 2.
#
#      Because the graded meshes are stretched in the far field, hmax is LARGER
#      than uniform at the same level and is NOT a meaningful abscissa across
#      families -- so all cross-family convergence plots here use ndof^(-1/2),
#      the mean mesh size.
#
#    * The shipped uniform and graded L-shape families are DOF-MATCHED:
#      lshape_<n> and lshape_graded_<n> have identical nv/nt/ne/nb and differ
#      only in vertex coordinates.  The comparison is therefore exactly at
#      equal cost.
#
#  We also run the SLIT domain (alpha = 1/2, the hardest planar corner) as a
#  second data point on how the rate degrades with the reentrant angle.  The
#  shipped slit family is UNGRADED, so it is presented uniform-only.
#
#  Run with:
#     export PATH=$HOME/.juliaup/bin:$PATH
#     julia --project=. tutorial/04_eig_lshape.jl
# =============================================================================

using VFEM: Mesh2D, mesh2d_load, laplace_eig_lagrange, lagrange_laplace_matrices
using LinearAlgebra: Symmetric, cholesky, eigen
using SparseArrays: SparseMatrixCSC
using Printf: @printf, @sprintf

const HELPERS = get(ENV, "VFEM_TUTORIAL_HELPERS",
                    joinpath(@__DIR__, "TutorialSupport.jl"))
include(HELPERS)
using .TutorialSupport

const MESHROOT = get(ENV, "VFEM_TUTORIAL_MESHES",
                     joinpath(@__DIR__, "meshdata"))
const OUT = get(ENV, "VFEM_TUTORIAL_OUT", ".")

const NEIG      = 6
const LAM1_REF  = 9.6397238440219        # L-shape, bottom-right quadrant removed
const ALPHA_L   = 2 / 3
const ALPHA_SLIT = 1 / 2

# -----------------------------------------------------------------------------
#  Eigensolver, with the same fallback used in 03_eig_square.jl
# -----------------------------------------------------------------------------
#  `laplace_eig_lagrange` densifies and calls `eigen` when the interior DOF
#  count is <= 6000, and calls Arpack shift-invert above that.  On the finest
#  P2 L-shape meshes the Arpack path does not converge ("XYAUPD_Exception:
#  Maximum number of iterations taken"), so we supply our own shift-invert
#  block iteration.  See 03_eig_square.jl for the full commentary.

function smallest_eigs_shift_invert(A0::SparseMatrixCSC, M0::SparseMatrixCSC,
                                   k::Integer; block::Integer = k + 10,
                                   maxit::Integer = 400, tol::Real = 1e-13)
    n = size(A0, 1)
    b = min(max(block, k + 2), n)
    F = cholesky(Symmetric(A0))            # SPD once Dirichlet DOFs are removed
    X = Float64[sin(0.7 * i * j + 0.3 * j) for i in 1:n, j in 1:b]
    lam_old = fill(Inf, k); lam = Float64[]; V = zeros(n, b)
    for it in 1:maxit
        Y = F \ (M0 * X)                                  # inverse iteration
        Y = Y / cholesky(Symmetric(Y' * (M0 * Y))).U      # M0-orthonormalise
        E = eigen(Symmetric(Y' * (A0 * Y)))               # Rayleigh-Ritz
        lam = E.values; X = Y * E.vectors; V = X
        if maximum(abs.(lam[1:k] .- lam_old) ./ abs.(lam[1:k])) < tol
            return lam[1:k], V[:, 1:k], it
        end
        lam_old = lam[1:k]
    end
    return lam[1:k], V[:, 1:k], maxit
end

function interior_count(m::Mesh2D, p::Integer)
    nbdv = length(unique(vec(m.bd_edges)))
    ndof = p == 1 ? m.nv : m.nv + m.ne
    nbd  = p == 1 ? nbdv : nbdv + m.nb
    return ndof - nbd, ndof
end

function eig_pairs(m::Mesh2D, p::Integer, neig::Integer)
    nint, _ = interior_count(m, p)
    if nint <= 6000
        r = laplace_eig_lagrange(m, p, neig)
        return (; lam = collect(r.eig_value), U = r.eig_func,
                  ndof = size(r.A, 1),
                  nint = size(r.A, 1) - length(r.bd_dofs), path = "library-dense")
    end
    A, M, bd = lagrange_laplace_matrices(m, p)
    ndof = size(A, 1)
    isbd = falses(ndof); for d in bd; isbd[d] = true; end
    int = findall(!, isbd)
    lam, V, its = smallest_eigs_shift_invert(A[int, int], M[int, int], neig)
    U = zeros(Float64, ndof, neig); U[int, :] .= V
    return (; lam, U, ndof, nint = length(int), path = "shift-invert(iters=$its)")
end

# -----------------------------------------------------------------------------
#  Sweep one mesh family
# -----------------------------------------------------------------------------
function sweep(prefix, levels; ps = (1, 2), neig = NEIG)
    rows = NamedTuple[]
    for p in ps, n in levels
        folder = "$(prefix)_$(n)"
        m = mesh2d_load(joinpath(MESHROOT, folder))
        # Too few interior DOFs to supply `neig` eigenvalues: the library
        # clamps to k_eff = min(neig, n_int - 1).  lshape_2 at P1 has only 5
        # interior DOFs, so skip rather than tabulate a ragged row.
        nint_pre, _ = interior_count(m, p)
        if nint_pre < neig + 1
            @printf("  %-22s p=%d SKIPPED: only %d interior DOFs (< neig+1 = %d)\n",
                    folder, p, nint_pre, neig + 1)
            continue
        end
        h = mesh_hmax(m)                      # = find_mesh_hmax(m.nodes, m.edges)
        t0 = time_ns()
        r = eig_pairs(m, p, neig)
        el = (time_ns() - t0) / 1e9
        ndof = r.ndof; nint = r.nint
        @printf("  %-22s p=%d ndof=%6d n_int=%6d hmax=%.6e ndof^-1/2=%.6e %7.2f s [%s]\n",
                folder, p, ndof, nint, h, ndof^(-0.5), el, r.path)
        @printf("      lambda_1 = %.12f\n", r.lam[1])
        @printf("      lambda_2..%d = %s\n", neig,
                join((@sprintf("%.8f", r.lam[k]) for k in 2:neig), "  "))
        push!(rows, (; folder, level = n, p, ndof, nint, hmax = h,
                       xdof = ndof^(-0.5), lam = collect(r.lam),
                       secs = el, mesh = m, U = r.U, path = r.path))
    end
    return rows
end

# -----------------------------------------------------------------------------
#  Aitken delta-squared extrapolation.  For eigenvalues 2..6 of the L-shape and
#  for the slit we have no closed form, so we build a *numerical* reference from
#  our own three finest values.  For a sequence with error ~ C q^i this returns
#  the limit exactly; here it only estimates it, so we print it as an estimate
#  and never treat it as exact.
# -----------------------------------------------------------------------------
function aitken(v::AbstractVector)
    length(v) >= 3 || return NaN
    a, b, c = v[end-2], v[end-1], v[end]
    den = (c - b) - (b - a)
    return abs(den) < 1e-300 ? NaN : c - (c - b)^2 / den
end

# -----------------------------------------------------------------------------
#  CSV writer.  `xkey` picks the abscissa used for the observed order:
#  :hmax for the uniform families, :xdof (= ndof^(-1/2)) for graded / for any
#  cross-family comparison.
# -----------------------------------------------------------------------------
function write_csv(path, family, rows, refs; xkey = :hmax)
    xname = xkey === :hmax ? "hmax" : "ndof_pow_minus_half"
    open(path, "w") do io
        hdr = String["family", "p", "level", "ndof", "n_int", "hmax",
                     "ndof_pow_minus_half"]
        for k in 1:NEIG; push!(hdr, "lambda_$k"); end
        push!(hdr, "lambda_1_ref", "abs_err_1", "rel_err_1", "order_1_vs_$xname")
        for k in 2:NEIG; push!(hdr, "ref_est_$k", "abs_err_$k", "order_$k"); end
        push!(hdr, "secs")
        println(io, join(hdr, ","))
        for p in sort(unique(getfield.(rows, :p)))
            sub = [r for r in rows if r.p == p]
            xs  = [getfield(r, xkey) for r in sub]
            ords = [observed_order([abs(r.lam[k] - refs[k]) for r in sub], xs)
                    for k in 1:NEIG]
            for (i, r) in enumerate(sub)
                f = String[family, string(p), string(r.level), string(r.ndof),
                           string(r.nint), @sprintf("%.10e", r.hmax),
                           @sprintf("%.10e", r.xdof)]
                for k in 1:NEIG; push!(f, @sprintf("%.12f", r.lam[k])); end
                push!(f, @sprintf("%.13f", refs[1]),
                         @sprintf("%.6e", r.lam[1] - refs[1]),
                         @sprintf("%.6e", (r.lam[1] - refs[1]) / refs[1]),
                         i == 1 ? "" : @sprintf("%.4f", ords[1][i-1]))
                for k in 2:NEIG
                    push!(f, @sprintf("%.10f", refs[k]),
                             @sprintf("%.6e", r.lam[k] - refs[k]),
                             i == 1 ? "" : @sprintf("%.4f", ords[k][i-1]))
                end
                push!(f, @sprintf("%.3f", r.secs))
                println(io, join(f, ","))
            end
        end
    end
    println("wrote $path")
end

function report_order(tag, rows, ref, xkey, expect)
    println("\n--- $tag: lambda_1 error and observed order vs $(xkey) ---")
    for p in sort(unique(getfield.(rows, :p)))
        sub = [r for r in rows if r.p == p]
        er  = [r.lam[1] - ref for r in sub]
        # abs() so an extrapolated reference that slightly overshoots the
        # finest value does not turn the last ratio into a NaN
        o   = observed_order(abs.(er), [getfield(r, xkey) for r in sub])
        @printf("  P%d errors: %s\n", p, join((@sprintf("%.5e", e) for e in er), "  "))
        @printf("  P%d orders: %s   (expected ~%s)\n", p,
                join((@sprintf("%.4f", x) for x in o), "  "), expect)
    end
end

function normalized_mode(m, U, k, p)
    u  = collect(U[:, k]); vv = fe_vertex_values(m, u, p)
    i  = argmax(abs.(vv)); s = vv[i] < 0 ? -1.0 : 1.0
    mx = maximum(abs.(vv)); mx == 0 && (mx = 1.0)
    return (s / mx) .* u
end

# =============================================================================
#  1. Uniform L-shape
# =============================================================================
println("=" ^ 78)
println("L-SHAPE, UNIFORM MESHES.  Omega = (-1,1)^2 minus [0,1]x[-1,0]")
@printf("  reference lambda_1 = %.13f    alpha = pi/omega = 2/3\n", LAM1_REF)
@printf("  expected uniform rate for lambda_1: h^(2 alpha) = h^%.4f\n", 2 * ALPHA_L)
println("=" ^ 78)
lu_rows = sweep("lshape", [2, 4, 8, 16, 32])

# =============================================================================
#  2. Graded L-shape (DOF-matched to the uniform family, beta = 1.5)
# =============================================================================
println("\n" * "=" ^ 78)
println("L-SHAPE, GRADED MESHES (beta = 1.5, max-norm radial map)")
println("  DOF-matched to the uniform family; hmax is LARGER at equal level,")
println("  so the abscissa for convergence is ndof^(-1/2), not hmax.")
println("=" ^ 78)
lg_rows = sweep("lshape_graded", [2, 4, 8, 16, 32])

# ---- numerical references for lambda_2..6, from the graded P2 sequence -------
lg_p2 = [r for r in lg_rows if r.p == 2]
refs_L = Float64[LAM1_REF]
for k in 2:NEIG
    push!(refs_L, aitken([r.lam[k] for r in lg_p2]))
end
println("\n--- lambda_2..6 reference ESTIMATES (Aitken on the graded-P2 sequence) ---")
for k in 2:NEIG
    @printf("  lambda_%d ~ %.9f   (finest graded P2 value %.9f)\n",
            k, refs_L[k], lg_p2[end].lam[k])
end
@printf("  for comparison 2*pi^2 = %.10f, 5*pi^2 = %.10f\n", 2pi^2, 5pi^2)

write_csv(joinpath(OUT, "eig_lshape_uniform.csv"), "lshape_uniform",
          lu_rows, refs_L; xkey = :hmax)
write_csv(joinpath(OUT, "eig_lshape_graded.csv"), "lshape_graded",
          lg_rows, refs_L; xkey = :xdof)

report_order("L-shape UNIFORM", lu_rows, LAM1_REF, :hmax,
             "1.333 = 2*alpha for both P1 and P2 (P1 approaches it from above)")
report_order("L-shape UNIFORM (vs ndof^-1/2)", lu_rows, LAM1_REF, :xdof, "1.333")
report_order("L-shape GRADED", lg_rows, LAM1_REF, :xdof,
             "2 for P1 (beta = 1/alpha is the P1 tuning); ~2 for P2 as well, " *
             "since full P2 recovery would need beta >= 2/alpha = 3")

println("\n--- upper-bound check on the L-shape (lambda_1^h >= 9.6397238440219) ---")
for r in vcat(lu_rows, lg_rows)
    @printf("  %-22s p=%d  lambda_1 = %.12f  above ref: %s\n",
            r.folder, r.p, r.lam[1], r.lam[1] >= LAM1_REF ? "yes" : "NO")
end

# ---- DOF-matched head-to-head at each level --------------------------------
println("\n--- DOF-matched uniform vs graded, error in lambda_1 ---")
println("  level   ndof     uniform P1      graded P1     ratio |   uniform P2      graded P2     ratio")
for n in [2, 4, 8, 16, 32]
    pick(rows, p) = (i = findfirst(r -> r.p == p && r.level == n, rows);
                     i === nothing ? nothing : rows[i])
    u1 = pick(lu_rows, 1); g1 = pick(lg_rows, 1)
    u2 = pick(lu_rows, 2); g2 = pick(lg_rows, 2)
    any(x -> x === nothing, (u1, g1, u2, g2)) && continue   # level was skipped
    e(r) = r.lam[1] - LAM1_REF
    @printf("  %5d %6d   %.4e   %.4e   %6.1fx |  %6d  %.4e   %.4e   %6.1fx\n",
            n, u1.ndof, e(u1), e(g1), e(u1)/e(g1),
            u2.ndof, e(u2), e(g2), e(u2)/e(g2))
end

# =============================================================================
#  3. Slit domain (alpha = 1/2), uniform only
# =============================================================================
println("\n" * "=" ^ 78)
println("SLIT DOMAIN (-1,1)^2 minus [0,1]x{0}, crack tip at origin, omega = 2pi")
@printf("  alpha = 1/2, expected uniform rate for lambda_1: h^%.2f\n", 2 * ALPHA_SLIT)
println("  the shipped slit family is UNGRADED -- uniform-only results")
println("=" ^ 78)
sl_rows = sweep("slit", [2, 4, 8, 16])
sl_p2 = [r for r in sl_rows if r.p == 2]
refs_S = Float64[aitken([r.lam[k] for r in sl_p2]) for k in 1:NEIG]
println("\n--- slit reference ESTIMATES (Aitken on the P2 sequence) ---")
for k in 1:NEIG
    @printf("  lambda_%d ~ %.9f   (finest P2 value %.9f)\n",
            k, refs_S[k], sl_p2[end].lam[k])
end
write_csv(joinpath(OUT, "eig_slit_uniform.csv"), "slit_uniform",
          sl_rows, refs_S; xkey = :hmax)
report_order("SLIT", sl_rows, refs_S[1], :hmax, "1.0 for both P1 and P2")

# =============================================================================
#  4. Figures
# =============================================================================
# mode 1 and 2 on the finest uniform mesh: the corner singularity is the steep
# gradient of mode 1 right at the origin.
let r = first(z for z in lu_rows if z.p == 1 && z.level == 32)
    for k in 1:2
        u = normalized_mode(r.mesh, r.U, k, 1)
        svg_solution(joinpath(OUT, "eig_lshape_mode$(k).svg"), r.mesh, u; p = 1,
                     title = @sprintf("L-shape mode %d: lambda_%d^h = %.8f", k, k, r.lam[k]))
    end
    println("\nwrote eig_lshape_mode1..2.svg")
end
# the graded mesh itself, so a reader can see the refinement toward the origin
let r = first(q for q in lg_rows if q.p == 1 && q.level == 16)
    svg_mesh(joinpath(OUT, "eig_lshape_graded_mesh16.svg"), r.mesh;
             title = "lshape_graded_16 (nv=$(r.mesh.nv), nt=$(r.mesh.nt), beta=1.5)")
    u = normalized_mode(r.mesh, r.U, 1, 1)
    svg_solution(joinpath(OUT, "eig_lshape_graded_mode1.svg"), r.mesh, u; p = 1,
                 draw_mesh = true,
                 title = @sprintf("L-shape graded, mode 1: lambda_1^h = %.10f", r.lam[1]))
end
let r = first(q for q in lu_rows if q.p == 1 && q.level == 16)
    svg_mesh(joinpath(OUT, "eig_lshape_mesh16.svg"), r.mesh;
             title = "lshape_16 uniform (nv=$(r.mesh.nv), nt=$(r.mesh.nt))")
end
let r = first(q for q in sl_rows if q.p == 1 && q.level == 16)
    u = normalized_mode(r.mesh, r.U, 1, 1)
    svg_solution(joinpath(OUT, "eig_slit_mode1.svg"), r.mesh, u; p = 1,
                 title = @sprintf("Slit domain, mode 1: lambda_1^h = %.8f", r.lam[1]))
end
println("wrote eig_lshape_graded_mesh16.svg, eig_lshape_graded_mode1.svg, " *
        "eig_lshape_mesh16.svg, eig_slit_mode1.svg")

# convergence plot: all four L-shape curves against ndof^(-1/2)
let series = Any[]
    for (rows, tag) in ((lu_rows, "uniform"), (lg_rows, "graded"))
        for p in (1, 2)
            sub = [r for r in rows if r.p == p]
            # reference slope: 2*alpha = 4/3 on the uniform family (the corner
            # sets the rate); 2 on the graded family (beta = 1.5 = 1/alpha is
            # the P1 tuning, and P2 attains the same 2 for lack of beta = 3)
            push!(series, (x = [r.xdof for r in sub],
                           y = [r.lam[1] - LAM1_REF for r in sub],
                           label = "P$p $tag",
                           slope = tag == "uniform" ? 2 * ALPHA_L : 2.0))
        end
    end
    svg_loglog(joinpath(OUT, "eig_lshape_conv.svg"), series;
               xlabel = "ndof^(-1/2)", ylabel = "lambda_1^h - lambda_1",
               title = "L-shape: uniform is stuck at 4/3, beta=1.5 grading restores 2")
    println("wrote eig_lshape_conv.svg")
end

# how the rate degrades with the reentrant angle: square (alpha=1), L (2/3),
# slit (1/2), all uniform, all P1, against ndof^(-1/2)
let series = Any[]
    for (rows, ref, tag, sl) in ((lu_rows, LAM1_REF, "L-shape  alpha=2/3", 2*ALPHA_L),
                                 (sl_rows, refs_S[1], "slit  alpha=1/2", 2*ALPHA_SLIT))
        for p in (1, 2)
            sub = [r for r in rows if r.p == p]
            push!(series, (x = [r.xdof for r in sub],
                           y = [abs(r.lam[1] - ref) for r in sub],
                           label = "P$p $tag", slope = p == 1 ? sl : nothing))
        end
    end
    svg_loglog(joinpath(OUT, "eig_corner_rates.svg"), series;
               xlabel = "ndof^(-1/2)", ylabel = "lambda_1^h - lambda_1",
               title = "Uniform meshes: the rate is set by alpha = pi/omega, not by p")
    println("wrote eig_corner_rates.svg")
end

println("\n04_eig_lshape.jl DONE")
