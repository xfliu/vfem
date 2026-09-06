# ============================================================================
# 09_verified_bounds_lshape.jl
#
# The same verified pipeline as chapter 08, now on the L-shaped domain, where
# there is NO closed-form eigenvalue. This is exactly the situation in which a
# verified bound stops being an academic exercise: with no exact answer to
# compare against, "the numbers stopped changing in the 6th digit" is not
# evidence of anything, whereas a rigorous enclosure is a proof.
#
# DOMAIN (cite this orientation with any reference value -- the L-shape
# eigenvalue literature uses several, and they are not interchangeable):
#     Omega = (-1,1)^2  \  [0,1] x [-1,0]      (BOTTOM-RIGHT quadrant removed)
#     area 3; one reentrant corner AT THE ORIGIN with interior angle 3*pi/2
#     alpha = pi/omega = 2/3
#     REFERENCE  lambda_1 = 9.6397238440219
#
# Why the L-shape is hard, and why grading is the fix:
# the reentrant corner makes the first eigenfunction behave like r^alpha =
# r^(2/3) near the origin. Its second derivatives are unbounded there, so
# u_1 is NOT in H^2 and the standard interpolation estimate that gives P1 its
# O(h^2) eigenvalue convergence fails. On a UNIFORM mesh the eigenvalue error
# is limited to O(h^(2*alpha)) = O(h^(4/3)) -- and the resulting loss is
# visible in the enclosure width below. A mesh GRADED toward the corner
# restores the full order: the achievable order is
#     H1 = min(p, alpha*beta),   L2 = min(p+1, 2*alpha*beta)
# so with alpha = 2/3 the shipped grading exponent beta = 1.5 gives
# alpha*beta = 1, exactly enough for P1 and no more. Full P2 recovery would
# need beta >= p/alpha = 3. The graded-P2 rows below are therefore a
# GRADING-PARAMETER CEILING, not a defect of the method -- do not read them as
# a shortfall.
#
# `lshape_<n>` and `lshape_graded_<n>` have IDENTICAL nv/nt/ne/nb; only the
# coordinates differ. The uniform-vs-graded comparison is therefore exactly
# DOF-matched, which is the only fair way to ask "does grading buy anything".
#
# Because the graded meshes stretch the far field, their h_max is LARGER than
# the uniform h_max at the same level. Never compare the two families against
# h; plot against ndof^(-1/2).
#
# Run:
#   julia --project=. tutorial/\
#         09_verified_bounds_lshape.jl
# ============================================================================

using VFEM
using Veigs
using IntervalArithmetic: Interval, interval, inf, sup, mid, diam
using LinearAlgebra
using SparseArrays
using Printf

const MESHROOT = joinpath(@__DIR__, "meshdata")

# Reference value for THIS orientation (bottom-right quadrant removed).
# It is a high-accuracy literature/extrapolated value, NOT an exact closed
# form -- so it is what we test our enclosure AGAINST, never what we derive
# the enclosure FROM.
const REF_1 = 9.6397238440219

const NEIG = 4

# ---------------------------------------------------------------------------
# The verified pipeline, repeated verbatim from chapter 08 so this script runs
# standalone. Comments there explain each step; the short version:
#   verified_lg_lower  -- rigorous CR/Liu shift rho, interval Lagrange
#                         eigenpairs, verified RT H(div) auxiliary solve,
#                         verified LG transform.
#   lg_is_valid        -- the mu < 1 guard. The LG transform
#                         lambda_low = rho - rho/(1-mu) is a lower bound ONLY
#                         while mu < 1, equivalently low < rho. When it fails
#                         the routine returns a large POSITIVE number that
#                         looks like a superb lower bound and is actually
#                         ABOVE the true eigenvalue. On this domain that is
#                         not hypothetical: at P1 n=4 the float driver reports
#                         73.34 as a "lower bound" for lambda_1 = 9.6397.
#                         Always check it; fall back to CR/Liu, which is
#                         unconditionally valid.
#   verified_upper     -- rigorous Galerkin upper bound via Veigs.veigs on the
#                         interval-assembled interior pencil.
#
# OBSERVED ON THIS HOST -- the float LG solve is the fragile link
# ---------------------------------------------------------------------------
# The mu < 1 guard fires far more often in FLOAT mode than in interval mode.
# On the L-shape it rejected the float LG number in 5 of 10 cases while the
# interval pipeline passed all 10 -- including cases where the interval bound
# was the tightest in the whole table. The reason is the pencil (AL, BL): BL is
# indefinite, and `LinearAlgebra.eigvals(AL_sym, BL_sym)` on an indefinite
# pencil can return a spurious mu above 1, which the float driver then feeds
# straight into the transform. `verified_lg_transform` instead symmetrises via
# `Veigs.sym_hull` and calls `Veigs.veig`, whose inertia-counting search
# locates the correct mu. So interval arithmetic here is not merely more
# careful bookkeeping around the same answer -- on this problem it is also the
# more RELIABLE route to the answer.
# ---------------------------------------------------------------------------

"""
    verified_lg_lower(m, p, neig; RT_order = p)

Fully rigorous Lehmann-Goerisch lower bounds. Every arithmetic step runs in
`Interval{Float64}`, so `inf` of each returned interval is a certified lower
bound -- subject to the validity condition `sup(low[k]) < rho`, which the
caller must check (see `lg_is_valid`).
"""
function verified_lg_lower(m::Mesh2D, p::Integer, neig::Integer;
                           RT_order::Integer = p)
    cr_int, Ch = verified_cr_liu_lower(m, neig)
    _check_cr_length(cr_int, neig)
    rho   = inf(cr_int[neig + 1])          # rigorous a-priori shift
    rho_i = interval(rho)

    r_cg = laplace_eig_lagrange(m, p, neig)          # exactly 3 positional args
    size(r_cg.eig_func, 2) == neig ||
        error("laplace_eig_lagrange clamped neig to $(size(r_cg.eig_func, 2)) " *
              "(it silently uses k_eff = min(neig, n_int-1)); use a finer mesh.")

    A_i, M_i, _ = lagrange_laplace_matrices(m, p; T = Interval{Float64})
    X = interval.(r_cg.eig_func)

    A_proj = transpose(X) * A_i * X
    M_proj = transpose(X) * M_i * X
    A_lg   = verified_rt_hdiv_problem(m, RT_order, X)   # Goerisch b(w,w)

    AL = A_proj .- rho_i .* M_proj
    BL = A_proj .- (interval(2.0) * rho_i) .* M_proj .+ (rho_i * rho_i) .* A_lg

    low = verified_lg_transform(AL, BL, rho_i)
    return (low = low, rho = rho, cr = cr_int, Ch = Ch)
end

# `verified_cr_liu_lower` may return MORE than neig+1 entries when Veigs
# widens outward to cover a cluster; fewer means the mesh is too coarse.
function _check_cr_length(cr, neig)
    length(cr) >= neig + 1 ||
        error("CR/Liu returned only $(length(cr)) bounds; need $(neig + 1) " *
              "for the LG shift. Refine the mesh.")
end

"LG validity test: the transform is a lower bound only while mu < 1, i.e. low < rho."
lg_is_valid(low_k, rho) = sup(low_k) < rho

# ===========================================================================
# Step 3 (rigorous).  The upper side of the enclosure.
#
# The conforming Galerkin eigenvalue bounds the exact eigenvalue from above in
# EXACT arithmetic -- but the float we print does not, because it carries
# assembly and eigensolver round-off. To close a rigorous enclosure we need a
# certified upper bound for the DISCRETE eigenvalue as well. `Veigs.veigs` on
# the interval-assembled interior pencil supplies it: sup(lambda[1]) lies
# above the discrete eigenvalue of every pair in the hull, hence above the
# exact eigenvalue by min-max.
#
# NOTE on `ind_range`: veigs may widen outward to cover a cluster, so index
# off the returned range and never assume lambda[1] is eigenvalue number 1.
# ===========================================================================

"Rigorous Galerkin upper bounds via Veigs on the interval interior pencil."
function verified_upper(m::Mesh2D, p::Integer, neig::Integer)
    A, M, bd = lagrange_laplace_matrices(m, p; T = Interval{Float64})
    isb = falses(size(A, 1)); isb[bd] .= true
    idx = findall(!, isb)
    lam, ir = Veigs.veigs(A[idx, idx], M[idx, idx], neig, :sm)
    return lam, ir, length(idx)
end

# ===========================================================================
# Hand-written SVG: the enclosure band.  No plotting package -- the tutorial
# ships zero plotting dependencies, so figures are emitted as literal SVG.
# y axis is linear in the eigenvalue, x axis is the refinement level. The band
# between lower and upper collapses onto the exact line as the mesh refines,
# which is the whole point of the chapter in one picture.
# ===========================================================================
function svg_enclosure_band(path, levels, lower, upper, exact;
                            width = 660, height = 460, title = "",
                            ylabel = "lambda_1", refname = "exact")
    n = length(levels)
    n == length(lower) == length(upper) || error("length mismatch")
    pl, pr, pt, pb = 92, width - 26, 46, height - 62
    ymin = min(minimum(lower), exact); ymax = max(maximum(upper), exact)
    pad  = max((ymax - ymin) * 0.12, abs(exact) * 1e-3)
    ymin -= pad; ymax += pad
    sx(i) = n == 1 ? (pl + pr) / 2 : pl + (i - 1) / (n - 1) * (pr - pl)
    sy(v) = pb - (v - ymin) / (ymax - ymin) * (pb - pt)
    f(v)  = string(round(v; digits = 2))
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
    println(io, """<rect width="$width" height="$height" fill="white"/>""")
    isempty(title) || println(io, """<text x="$((pl+pr)/2)" y="24" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#222">$title</text>""")
    # horizontal gridlines with value labels
    for t in 0:4
        v = ymin + t * (ymax - ymin) / 4
        println(io, """<line x1="$pl" y1="$(sy(v))" x2="$pr" y2="$(sy(v))" stroke="#e4e4e4" stroke-width="0.8"/>""")
        println(io, """<text x="$(pl-8)" y="$(sy(v)+4)" text-anchor="end" font-family="sans-serif" font-size="11" fill="#555">$(f(v))</text>""")
    end
    # the enclosure band: upper path forward, lower path back
    pts = join([" $(sx(i)),$(sy(upper[i]))" for i in 1:n]) *
          join([" $(sx(i)),$(sy(lower[i]))" for i in n:-1:1])
    println(io, """<polygon points="$pts" fill="#0072B2" fill-opacity="0.22" stroke="none"/>""")
    for (vals, col) in ((upper, "#0072B2"), (lower, "#D55E00"))
        println(io, """<polyline points="$(join([" $(sx(i)),$(sy(vals[i]))" for i in 1:n]))" fill="none" stroke="$col" stroke-width="2"/>""")
        for i in 1:n
            println(io, """<circle cx="$(sx(i))" cy="$(sy(vals[i]))" r="3.2" fill="$col"/>""")
        end
    end
    println(io, """<line x1="$pl" y1="$(sy(exact))" x2="$pr" y2="$(sy(exact))" stroke="#009E73" stroke-width="1.6" stroke-dasharray="6,4"/>""")
    println(io, """<text x="$(pr-4)" y="$(sy(exact)-6)" text-anchor="end" font-family="sans-serif" font-size="11" fill="#009E73">$refname = $(round(exact; digits=7))</text>""")
    for i in 1:n
        # First/last labels are anchored inward so long level tags are not
        # clipped by the plot frame.
        anch = i == 1 ? "start" : (i == n ? "end" : "middle")
        println(io, """<text x="$(sx(i))" y="$(pb+18)" text-anchor="$anch" font-family="sans-serif" font-size="11" fill="#333">$(levels[i])</text>""")
    end
    println(io, """<line x1="$pl" y1="$pb" x2="$pr" y2="$pb" stroke="#333" stroke-width="1"/>""")
    println(io, """<line x1="$pl" y1="$pt" x2="$pl" y2="$pb" stroke="#333" stroke-width="1"/>""")
    println(io, """<text x="$((pl+pr)/2)" y="$(height-16)" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#333">refinement level</text>""")
    println(io, """<text x="18" y="$((pt+pb)/2)" transform="rotate(-90 18 $((pt+pb)/2))" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#333">$ylabel</text>""")
    println(io, """<text x="$(pl+10)" y="$(pt+14)" font-family="sans-serif" font-size="11" fill="#0072B2">verified upper</text>""")
    println(io, """<text x="$(pl+10)" y="$(pt+28)" font-family="sans-serif" font-size="11" fill="#D55E00">verified lower</text>""")
    println(io, "</svg>")
    write(path, String(take!(io)))
    return path
end

# Minimal hand-written log-log plot (same reason: no plotting dependency).
function svg_loglog_plain(path, series; width = 660, height = 470,
                          xlabel = "ndof", ylabel = "value", title = "")
    PAL = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00")
    pl, pr, pt, pb = 78, width - 190, 46, height - 62
    LX = [log10.(Float64.(s.x)) for s in series]
    LY = [log10.(Float64.(s.y)) for s in series]
    x0, x1 = minimum(minimum.(LX)), maximum(maximum.(LX))
    y0, y1 = minimum(minimum.(LY)), maximum(maximum.(LY))
    px = max((x1 - x0) * 0.10, 0.12); py = max((y1 - y0) * 0.10, 0.12)
    x0 -= px; x1 += px; y0 -= py; y1 += py
    sx(v) = pl + (v - x0) / (x1 - x0) * (pr - pl)
    sy(v) = pb - (v - y0) / (y1 - y0) * (pb - pt)
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
    println(io, """<rect width="$width" height="$height" fill="white"/>""")
    isempty(title) || println(io, """<text x="$((pl+pr)/2)" y="24" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#222">$title</text>""")
    for k in ceil(Int, x0):floor(Int, x1)
        println(io, """<line x1="$(sx(k))" y1="$pt" x2="$(sx(k))" y2="$pb" stroke="#e4e4e4" stroke-width="0.8"/>""")
        println(io, """<text x="$(sx(k))" y="$(pb+18)" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#555">1e$k</text>""")
    end
    for k in ceil(Int, y0):floor(Int, y1)
        println(io, """<line x1="$pl" y1="$(sy(k))" x2="$pr" y2="$(sy(k))" stroke="#e4e4e4" stroke-width="0.8"/>""")
        println(io, """<text x="$(pl-8)" y="$(sy(k)+4)" text-anchor="end" font-family="sans-serif" font-size="11" fill="#555">1e$k</text>""")
    end
    for (i, s) in enumerate(series)
        col = PAL[mod1(i, length(PAL))]
        println(io, """<polyline points="$(join([" $(sx(LX[i][j])),$(sy(LY[i][j]))" for j in eachindex(LX[i])]))" fill="none" stroke="$col" stroke-width="2"/>""")
        for j in eachindex(LX[i])
            println(io, """<circle cx="$(sx(LX[i][j]))" cy="$(sy(LY[i][j]))" r="3.2" fill="$col"/>""")
        end
        println(io, """<line x1="$(pr+14)" y1="$(pt+10+16*(i-1))" x2="$(pr+34)" y2="$(pt+10+16*(i-1))" stroke="$col" stroke-width="2"/>""")
        println(io, """<text x="$(pr+40)" y="$(pt+14+16*(i-1))" font-family="sans-serif" font-size="11" fill="#333">$(s.label)</text>""")
    end
    println(io, """<line x1="$pl" y1="$pb" x2="$pr" y2="$pb" stroke="#333" stroke-width="1"/>""")
    println(io, """<line x1="$pl" y1="$pt" x2="$pl" y2="$pb" stroke="#333" stroke-width="1"/>""")
    println(io, """<text x="$((pl+pr)/2)" y="$(height-16)" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#333">$xlabel</text>""")
    println(io, """<text x="18" y="$((pt+pb)/2)" transform="rotate(-90 18 $((pt+pb)/2))" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#333">$ylabel</text>""")
    println(io, "</svg>")
    write(path, String(take!(io)))
    return path
end

# ===========================================================================
# Driver
# ===========================================================================
#
# Sizes: the interval pipeline's binding cost is the dense saddle-matrix
# inverse inside `verified_rt_hdiv_problem` (dimension N below, O(N^3) time
# and 8*N^2 bytes). The L-shape has 6*n^2 triangles and 9*n^2+4*n edges, so N
# grows fast; the run stops at the finest level that completes in minutes and
# the table reports where that is.

saddle_dim(m, q) = (q + 1) * m.ne + q * (q + 1) * m.nt + m.nt * ((q + 1) * (q + 2) ÷ 2)

const CASES = [("n=4  P1 uniform", "lshape_4",         1, "uniform"),
               ("n=8  P1 uniform", "lshape_8",         1, "uniform"),
               ("n=16 P1 uniform", "lshape_16",        1, "uniform"),
               ("n=4  P1 graded",  "lshape_graded_4",  1, "graded"),
               ("n=8  P1 graded",  "lshape_graded_8",  1, "graded"),
               ("n=16 P1 graded",  "lshape_graded_16", 1, "graded"),
               ("n=4  P2 uniform", "lshape_4",         2, "uniform"),
               ("n=8  P2 uniform", "lshape_8",         2, "uniform"),
               ("n=4  P2 graded",  "lshape_graded_4",  2, "graded"),
               ("n=8  P2 graded",  "lshape_graded_8",  2, "graded")]

let mw = mesh2d_load(joinpath(MESHROOT, "lshape_4"))
    t = @elapsed begin
        lg_lower_eig_bound_laplace(mw, 1, 2; RT_order = 1)
        verified_lg_lower(mw, 1, 2; RT_order = 1)
        verified_upper(mw, 1, 2)
    end
    @printf("[warm-up / JIT: %.1f s -- excluded from the table]\n", t)
end

rows = NamedTuple[]

for (tag, folder, p, fam) in CASES
    m    = mesh2d_load(joinpath(MESHROOT, folder))
    hmax = find_mesh_hmax(m.nodes, m.edges)
    ndof = p == 1 ? m.nv : m.nv + m.ne
    N    = saddle_dim(m, p)

    @printf("\n=== %s : %s  nv=%d nt=%d ne=%d  ndof=%d  hmax=%.6e  saddle N=%d ===\n",
            tag, folder, m.nv, m.nt, m.ne, ndof, hmax, N)

    tf = @elapsed rf = lg_lower_eig_bound_laplace(m, p, NEIG; RT_order = p)
    fl_ok = rf.eig_lower[1] < rf.rho
    @printf("float    rho = %.15e   CR/Liu lower[1] = %.15e\n", rf.rho, rf.cr_eig_lower[1])
    @printf("float    LG lower[1] = %.15e   %s\n", rf.eig_lower[1],
            fl_ok ? "(passes mu<1)" : "*** FAILS mu<1 : NOT A BOUND ***")
    @printf("float    CG upper[1] = %.15e   (%.2f s)\n", rf.eig_upper[1], tf)
    lo_f = fl_ok ? max(rf.cr_eig_lower[1], rf.eig_lower[1]) : rf.cr_eig_lower[1]

    tv = @elapsed vr = verified_lg_lower(m, p, NEIG; RT_order = p)
    tu = @elapsed (lamU, irU, nint) = verified_upper(m, p, NEIG)
    lg_ok = lg_is_valid(vr.low[1], vr.rho)
    lo_lg = inf(vr.low[1]); lo_cr = inf(vr.cr[1])
    lo    = lg_ok ? max(lo_cr, lo_lg) : lo_cr
    up    = sup(lamU[1])
    @printf("verified CR/Liu lower[1] >= %.15e   Ch_cr in [%.12e, %.12e]\n",
            lo_cr, inf(vr.Ch), sup(vr.Ch))
    @printf("verified LG lower[1]     >= %.15e   %s   (%.2f s)\n", lo_lg,
            lg_ok ? "(passes mu<1)" : "*** FAILS mu<1 : DISCARDED ***", tv)
    @printf("verified CG upper[1]     <= %.15e   (%.2f s, n_int=%d, ind_range=%s)\n",
            up, tu, nint, string(irU))
    @printf("ENCLOSURE  lambda_1 in [%.15e, %.15e]   width = %.6e\n", lo, up, up - lo)
    @printf("           reference  = %.15e   strictly inside? %s\n", REF_1,
            (lo < REF_1 < up) ? "YES" : "NO  <-- CONTRACT VIOLATION")

    push!(rows, (tag = tag, folder = folder, family = fam, p = p, ndof = ndof,
                 hmax = hmax, saddle = N, nint = nint,
                 lo_f = lo_f, up_f = rf.eig_upper[1],
                 cr_f = rf.cr_eig_lower[1], lg_f = rf.eig_lower[1],
                 w_f = rf.eig_upper[1] - lo_f, t_f = tf, lg_ok_f = fl_ok,
                 lo_v = lo, up_v = up, w_v = up - lo, t_v = tv + tu,
                 lg_ok_v = lg_ok, lo_cr = lo_cr, lo_lg = lo_lg,
                 rho = vr.rho, inside = (lo < REF_1 < up)))
end

# ---------------------------------------------------------------------------
# Uniform vs graded at matched ndof -- the question grading is supposed to
# answer. Because the two families share nv/nt/ne exactly, "at matched ndof"
# means "at the same level n", so this is a direct pairing with no
# interpolation.
# ---------------------------------------------------------------------------
println("\n--- uniform vs graded, DOF-matched (enclosure width) ---")
@printf("%-8s %-6s %8s %14s %14s %10s\n", "level", "order", "ndof",
        "width uniform", "width graded", "ratio u/g")
for p in (1, 2), lev in ("n=4", "n=8", "n=16")
    u = findfirst(r -> r.p == p && r.family == "uniform" && startswith(r.tag, lev), rows)
    g = findfirst(r -> r.p == p && r.family == "graded"  && startswith(r.tag, lev), rows)
    (u === nothing || g === nothing) && continue
    @printf("%-8s P%-5d %8d %14.6e %14.6e %10.3f\n", lev, p, rows[u].ndof,
            rows[u].w_v, rows[g].w_v, rows[u].w_v / rows[g].w_v)
end

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
open(joinpath(@__DIR__, "verified_lshape.csv"), "w") do io
    println(io, "level,mesh,family,order,ndof,hmax,saddle_dim,n_int,mode,lower,upper,width,",
                "reference,inside,wall_time_s,cr_liu_lower,lg_lower,lg_valid,rho,rigorous")
    for r in rows
        @printf(io, "%s,%s,%s,%d,%d,%.16e,%d,%d,float,%.16e,%.16e,%.16e,%.16e,%s,%.3f,%.16e,%.16e,%s,%.16e,no\n",
                r.tag, r.folder, r.family, r.p, r.ndof, r.hmax, r.saddle, r.nint,
                r.lo_f, r.up_f, r.w_f, REF_1, (r.lo_f < REF_1 < r.up_f),
                r.t_f, r.cr_f, r.lg_f, r.lg_ok_f, r.rho)
        @printf(io, "%s,%s,%s,%d,%d,%.16e,%d,%d,interval,%.16e,%.16e,%.16e,%.16e,%s,%.3f,%.16e,%.16e,%s,%.16e,yes\n",
                r.tag, r.folder, r.family, r.p, r.ndof, r.hmax, r.saddle, r.nint,
                r.lo_v, r.up_v, r.w_v, REF_1, r.inside,
                r.t_v, r.lo_cr, r.lo_lg, r.lg_ok_v, r.rho)
    end
end

# ---------------------------------------------------------------------------
# Figures. Graded meshes are plotted against ndof (equivalently ndof^(-1/2)
# as the length scale) and NEVER against hmax -- graded hmax is larger than
# uniform at the same level, so an h abscissa would compare unlike things.
# ---------------------------------------------------------------------------
sel(p, fam) = [r for r in rows if r.p == p && r.family == fam]

for (p, fam) in ((1, "uniform"), (1, "graded"), (2, "uniform"), (2, "graded"))
    rs = sel(p, fam); isempty(rs) && continue
    svg_enclosure_band(joinpath(@__DIR__, "lshape_band_p$(p)_$(fam).svg"),
        [r.tag for r in rs], [r.lo_v for r in rs], [r.up_v for r in rs], REF_1;
        title = "L-shape P$p $fam: verified enclosure of lambda_1",
        ylabel = "lambda_1", refname = "reference")
end

svg_loglog_plain(joinpath(@__DIR__, "lshape_width.svg"),
    [(x = [r.ndof for r in sel(1, "uniform")], y = [r.w_v for r in sel(1, "uniform")], label = "P1 uniform"),
     (x = [r.ndof for r in sel(1, "graded")],  y = [r.w_v for r in sel(1, "graded")],  label = "P1 graded"),
     (x = [r.ndof for r in sel(2, "uniform")], y = [r.w_v for r in sel(2, "uniform")], label = "P2 uniform"),
     (x = [r.ndof for r in sel(2, "graded")],  y = [r.w_v for r in sel(2, "graded")],  label = "P2 graded")];
    xlabel = "ndof", ylabel = "verified enclosure width",
    title = "L-shape: enclosure width vs ndof (DOF-matched uniform vs graded)")

# Wall time, split by order and family (never joined across them -- P1 and P2
# at the same ndof are different discretisations).
svg_loglog_plain(joinpath(@__DIR__, "lshape_time.svg"),
    [(x = [r.ndof for r in sel(1, "uniform")], y = [r.t_v for r in sel(1, "uniform")], label = "P1 uniform, interval"),
     (x = [r.ndof for r in sel(1, "uniform")], y = [r.t_f for r in sel(1, "uniform")], label = "P1 uniform, float"),
     (x = [r.ndof for r in sel(2, "uniform")], y = [r.t_v for r in sel(2, "uniform")], label = "P2 uniform, interval"),
     (x = [r.ndof for r in sel(2, "uniform")], y = [r.t_f for r in sel(2, "uniform")], label = "P2 uniform, float")];
    xlabel = "ndof", ylabel = "wall time (s)",
    title = "L-shape: wall time, float vs interval")

println("\nwrote verified_lshape.csv, lshape_band_*.svg, lshape_width.svg, lshape_time.svg")
println("LSHAPE_DONE_MARKER")
