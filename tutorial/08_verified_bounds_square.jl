# ============================================================================
# 08_verified_bounds_square.jl
#
# Rigorous TWO-SIDED enclosure of the first Dirichlet Laplace eigenvalue on
# the unit square (0,1)^2, where the exact value 2*pi^2 = 19.7392088021787...
# is known and can be checked to lie strictly inside the computed enclosure.
#
# ---------------------------------------------------------------------------
# WHY THIS CHAPTER EXISTS
# ---------------------------------------------------------------------------
# Chapter 07 produced *upper* bounds. That is what a conforming Galerkin
# method gives you for free: the FEM space V_h is a subspace of H^1_0, so the
# Rayleigh-Ritz min-max over V_h ranges over strictly fewer functions than the
# min-max over H^1_0, and every discrete eigenvalue therefore sits ABOVE the
# true one. Refining the mesh drives that upper bound down toward lambda_k,
# but no amount of refinement tells you how far you still are: a decreasing
# sequence with an unknown limit is not a bound from below.
#
# A lower bound needs a genuinely different argument. Notice that "use a finer
# mesh" is not one of them -- every conforming space, however fine, is still a
# subspace, so it still yields an upper bound. You must leave the conforming
# setting, or add an a-posteriori argument. VFEM implements both:
#
#   (a) Crouzeix-Raviart + Liu's constant.  The CR space is NOT a subspace of
#       H^1_0 -- CR functions are continuous only at edge midpoints and jump
#       across element edges. The min-max argument therefore no longer forces
#       an upper bound, and the CR eigenvalue in fact falls below the true one
#       up to a computable interpolation defect. Liu (2015) turns that into an
#       explicit, fully computable inequality
#
#             lambda_k  >=  mu_k / (1 + mu_k * Ch^2),      Ch = 0.1893 * h_max
#
#       with mu_k the k-th CR discrete eigenvalue. What makes this usable is
#       that 0.1893 is an explicit mesh-independent constant and h_max is
#       measured from the mesh: nothing is estimated or fitted.
#
#   (b) Lehmann-Goerisch (LG) sharpening.  Liu's bound is rigorous but loose
#       -- its error is O(h^2) with a large prefactor. LG takes an a-priori
#       lower bound rho (which is what (a) supplies) plus the solution of an
#       auxiliary H(div) Raviart-Thomas problem, and SHARPENS the bound to
#       roughly the accuracy of the upper bound. LG cannot bootstrap itself:
#       with no valid starting rho there is nothing to sharpen, and an invalid
#       rho silently destroys the guarantee. That dependency -- CR/Liu first,
#       LG second -- is the architecture of this chapter.
#
# ---------------------------------------------------------------------------
# THE NAME-VERSUS-RIGOUR TRAP -- read before trusting any number
# ---------------------------------------------------------------------------
# `lg_lower_eig_bound_laplace` computes a lower bound but is FLOAT-ONLY: it
# runs the LG algebra in Float64, so round-off is unaccounted for and the
# output is a *floating-point lower bound*, not a certificate. The rigorous
# counterparts are `verified_cr_liu_lower` (CR/Liu) and
# `verified_rt_hdiv_problem` + `verified_lg_transform` (LG), all returning
# `Interval{Float64}`. This script runs BOTH and labels every number.
# Conflating them would be the worst error a verified-computing tutorial
# could make.
#
# ---------------------------------------------------------------------------
# WHAT "RIGOROUS" MEANS HERE
# ---------------------------------------------------------------------------
# In interval mode every matrix entry is an interval [a,b] that provably
# contains the exact bilinear-form value; the interval matrix is a *set* of
# matrices (its hull). `Veigs.veigs` / `Veigs.veig` return an interval that
# provably contains the corresponding eigenvalue of EVERY concrete pair in
# that hull -- including the exactly-rounded one. So `inf(lower)` lies below
# the true lambda_1 no matter how the rounding fell. That is a theorem about
# the printed digits, not a convergence observation.
#
# Run:
#   julia --project=. tutorial/\
#         08_verified_bounds_square.jl
# ============================================================================

using VFEM
using Veigs
using IntervalArithmetic: Interval, interval, inf, sup, mid, diam
using LinearAlgebra
using SparseArrays
using Printf

const MESHROOT = joinpath(@__DIR__, "meshdata")
const EXACT_1  = 2 * pi^2      # exact lambda_1 on (0,1)^2 = 19.739208802178716
const NEIG     = 4

# ===========================================================================
# Step 1 (rigorous).  Crouzeix-Raviart + Liu lower bound in interval mode.
#
# `verified_cr_liu_lower(m, neig)` assembles the CR mass and stiffness with
# T = Interval{Float64}, restricts to interior edge DOFs, calls
# `Veigs.veigs(A1, A0, neig+1, :sm)` for verified eigenvalue enclosures, and
# applies the Liu shift lambda/(1 + lambda*Ch^2) in interval arithmetic. It
# returns neig+1 ascending bounds plus Ch_cr as an interval -- so you can see
# that h_max itself was enclosed rather than silently rounded. Entry neig+1
# is precisely what LG needs as its shift parameter rho.
#
# CAUTION: `verified_cr_liu_lower` densifies the interior CR pencil before
# calling Veigs (Julia 1.12 dropped the eigvals(Symmetric(sparse)) overload
# that veigs relied on). Cost is therefore O(n_int^3) in the interior EDGE
# count, and `verified_ldl` densifies again inside -- roughly 2*8*n^2 bytes.
# That, not the 2x from interval storage, is the memory cliff.
# ===========================================================================

# ===========================================================================
# Step 2 (rigorous).  Lehmann-Goerisch sharpening in interval mode.
#
# Mechanism, in execution order:
#
#  1. rho <- inf(CR/Liu bound number neig+1).  Taking `inf` is the rigorous
#     choice: LG needs rho to be a TRUE lower bound for lambda_{neig+1}, and
#     the infimum of a verified enclosure certainly is one.
#
#  2. Conforming Lagrange eigenpairs; X = the eigenfunction coefficient
#     stack. The basis matters: `laplace_eig_lagrange` /
#     `lagrange_laplace_matrices` use the MONOMIAL Lagrange basis
#     phi_alpha = L1^i L2^j L3^k, which is exactly what `rt_hdiv_problem`
#     consumes. `create_matrix_lagrange` is the NODAL basis and is a
#     different object; mixing the two produces silent garbage.
#
#  3. Project the interval stiffness and mass onto those eigenfunctions,
#        A_proj = X' A X,   M_proj = X' M X            (neig x neig, tiny).
#     LG needs the operator only on the span of the computed eigenfunctions,
#     which is why the final eigenproblem stays neig x neig however fine the
#     mesh gets.
#
#  4. THE AUXILIARY H(div) PROBLEM -- Goerisch's ingredient. Goerisch's
#     refinement of Lehmann's method requires a third bilinear form b(w,w)
#     where w is a vector field with div w = -u_h. Discretising that in
#     Raviart-Thomas elements against a DG-Lagrange multiplier gives the
#     saddle-point system
#
#            [ A   B ] [ w ]   [  0 ]
#            [ B'  0 ] [ q ] = [ -F ]
#
#     with A the RT mass matrix, B the RT/DG coupling and F the load built
#     from the eigenfunctions. `verified_rt_hdiv_problem` solves it with a
#     verified residual correction: LU-factor the MIDPOINT matrix in Float64,
#     form the interval residual r = rhs_int - K_int * x_approx, apply the
#     midpoint inverse to r, and add that correction. The returned w'Aw is an
#     interval enclosing the true b(w,w).
#
#     This step dominates the cost, structurally rather than wastefully: it
#     forms an explicit DENSE inverse of the (ndof_RT + n_dg) saddle matrix,
#     because interval triangular back-substitution suffers dependency
#     blow-up (each solve stage re-uses intervals already widened by the
#     previous one, so widths grow geometrically with depth). A dense inverse
#     makes each output entry one sum of Float x Interval products, so widths
#     grow only linearly. The price is O(N^3) time and 8*N^2 bytes in the
#     saddle dimension N. Every timing below is reported so the reader sees
#     where that becomes the binding constraint.
#
#  5. Close the LG eigenproblem and transform:
#        AL = A_proj - rho*M_proj
#        BL = A_proj - 2*rho*M_proj + rho^2 * A_lg
#        mu = verified eigenvalues of (AL, BL)          [Veigs.veig]
#        lambda_low = rho - rho/(1 - mu)                [interval arithmetic]
#     `verified_lg_transform` performs step 5: it symmetrises via
#     `Veigs.sym_hull` (round-off can break the symmetry of X'AX in the last
#     bit), calls `Veigs.veig`, applies the transform to the descending-
#     ordered mu, and re-sorts ascending.
#
# ---------------------------------------------------------------------------
# THE LG VALIDITY CONDITION -- a trap this script actively guards against
# ---------------------------------------------------------------------------
# The transform lambda_low = rho - rho/(1 - mu) is a lower bound only while
# mu < 1. Reading off the three regimes:
#     mu < 0        ->  0 < lambda_low < rho     sharp and useful
#     0 < mu < 1    ->  lambda_low < 0           valid but worthless
#     mu > 1        ->  lambda_low > rho         NOT A BOUND AT ALL
# The last case is not hypothetical. On a coarse mesh with a low-order RT
# space, the auxiliary problem is too inaccurate for the LG machinery and mu
# crosses 1; the routine then returns a large positive number that looks like
# an excellent lower bound and is in fact above the true eigenvalue. We
# observed exactly this on the L-shape at P1, level n=4: the float driver
# reported 73.34 as a "lower bound" for lambda_1 = 9.6397.
#
# So: ALWAYS check lambda_low < rho, and never report an LG number that fails
# it. The CR/Liu bound is unconditionally valid and is the correct fallback.
# `lower_best` below is max(CR/Liu, valid LG), which is itself a valid lower
# bound because the maximum of two lower bounds is a lower bound.
# ===========================================================================

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
# Sizes are chosen so the interval pipeline finishes in minutes. The binding
# cost is the dense saddle inverse inside `verified_rt_hdiv_problem`, whose
# dimension is
#      N = (RT_order+1)*ne + RT_order*(RT_order+1)*nt   +   nt*nbasis_lag
# and which needs 8*N^2 bytes and O(N^3) time. The table prints N per case.

const CASES = [("n=4  P1", "unit_square_4",  1),
               ("n=8  P1", "unit_square_8",  1),
               ("n=16 P1", "unit_square_16", 1),
               ("n=32 P1", "unit_square_32", 1),
               ("n=4  P2", "unit_square_4",  2),
               ("n=8  P2", "unit_square_8",  2),
               ("n=16 P2", "unit_square_16", 2)]

saddle_dim(m, q) = (q + 1) * m.ne + q * (q + 1) * m.nt + m.nt * ((q + 1) * (q + 2) ÷ 2)

# Warm-up so the reported timings are compile-free.
let mw = mesh2d_load(joinpath(MESHROOT, "unit_square_4"))
    t = @elapsed begin
        lg_lower_eig_bound_laplace(mw, 1, 2; RT_order = 1)
        verified_lg_lower(mw, 1, 2; RT_order = 1)
        verified_upper(mw, 1, 2)
    end
    @printf("[warm-up / JIT: %.1f s -- excluded from the table]\n", t)
end

rows = NamedTuple[]

for (tag, folder, p) in CASES
    m    = mesh2d_load(joinpath(MESHROOT, folder))
    hmax = find_mesh_hmax(m.nodes, m.edges)     # NOT find_mesh_hmax(m)
    ndof = p == 1 ? m.nv : m.nv + m.ne
    N    = saddle_dim(m, p)

    @printf("\n=== %s : %s  nv=%d nt=%d ne=%d  ndof=%d  hmax=%.6e  saddle N=%d ===\n",
            tag, folder, m.nv, m.nt, m.ne, ndof, hmax, N)

    # ---- float mode (NOT a certificate) --------------------------------
    tf = @elapsed rf = lg_lower_eig_bound_laplace(m, p, NEIG; RT_order = p)
    fl_ok = rf.eig_lower[1] < rf.rho
    @printf("float    rho = %.15e   Ch_cr = %.15e\n", rf.rho, rf.Ch_cr)
    @printf("float    CR/Liu lower[1] = %.15e\n", rf.cr_eig_lower[1])
    @printf("float    LG lower[1]     = %.15e   %s\n", rf.eig_lower[1],
            fl_ok ? "(passes mu<1)" : "*** FAILS mu<1 : NOT A BOUND ***")
    @printf("float    CG upper[1]     = %.15e   (%.2f s)\n", rf.eig_upper[1], tf)
    lo_f = fl_ok ? max(rf.cr_eig_lower[1], rf.eig_lower[1]) : rf.cr_eig_lower[1]

    # ---- interval mode (rigorous) ---------------------------------------
    tv = @elapsed vr = verified_lg_lower(m, p, NEIG; RT_order = p)
    tu = @elapsed (lamU, irU, nint) = verified_upper(m, p, NEIG)
    lg_ok  = lg_is_valid(vr.low[1], vr.rho)
    lo_lg  = inf(vr.low[1])
    lo_cr  = inf(vr.cr[1])
    lo     = lg_ok ? max(lo_cr, lo_lg) : lo_cr        # max of valid lower bounds
    up     = sup(lamU[1])
    @printf("verified Ch_cr in [%.15e, %.15e]\n", inf(vr.Ch), sup(vr.Ch))
    @printf("verified CR/Liu lower[1] >= %.15e\n", lo_cr)
    @printf("verified LG lower[1]     >= %.15e   %s   (%.2f s)\n", lo_lg,
            lg_ok ? "(passes mu<1)" : "*** FAILS mu<1 : DISCARDED ***", tv)
    @printf("verified CG upper[1]     <= %.15e   (%.2f s, n_int=%d, ind_range=%s)\n",
            up, tu, nint, string(irU))
    @printf("ENCLOSURE  lambda_1 in [%.15e, %.15e]   width = %.6e\n", lo, up, up - lo)
    @printf("           2*pi^2   =    %.15e   inside? %s\n", EXACT_1,
            (lo <= EXACT_1 <= up) ? "YES" : "NO  <-- CONTRACT VIOLATION")

    push!(rows, (tag = tag, folder = folder, p = p, ndof = ndof, hmax = hmax,
                 saddle = N, nint = nint,
                 lo_f = lo_f, up_f = rf.eig_upper[1],
                 cr_f = rf.cr_eig_lower[1], lg_f = rf.eig_lower[1],
                 w_f = rf.eig_upper[1] - lo_f, t_f = tf, lg_ok_f = fl_ok,
                 lo_v = lo, up_v = up, w_v = up - lo, t_v = tv + tu,
                 lg_ok_v = lg_ok, lo_cr = lo_cr, lo_lg = lo_lg,
                 rho = vr.rho, inside = (lo <= EXACT_1 <= up)))
end

# ---------------------------------------------------------------------------
# CSV -- two rows per case (float, interval).
# ---------------------------------------------------------------------------
open(joinpath(@__DIR__, "verified_square.csv"), "w") do io
    println(io, "level,mesh,order,ndof,hmax,saddle_dim,n_int,mode,lower,upper,width,",
                "reference,inside,wall_time_s,cr_liu_lower,lg_lower,lg_valid,rho,rigorous")
    for r in rows
        @printf(io, "%s,%s,%d,%d,%.16e,%d,%d,float,%.16e,%.16e,%.16e,%.16e,%s,%.3f,%.16e,%.16e,%s,%.16e,no\n",
                r.tag, r.folder, r.p, r.ndof, r.hmax, r.saddle, r.nint,
                r.lo_f, r.up_f, r.w_f, EXACT_1, (r.lo_f <= EXACT_1 <= r.up_f),
                r.t_f, r.cr_f, r.lg_f, r.lg_ok_f, r.rho)
        @printf(io, "%s,%s,%d,%d,%.16e,%d,%d,interval,%.16e,%.16e,%.16e,%.16e,%s,%.3f,%.16e,%.16e,%s,%.16e,yes\n",
                r.tag, r.folder, r.p, r.ndof, r.hmax, r.saddle, r.nint,
                r.lo_v, r.up_v, r.w_v, EXACT_1, r.inside,
                r.t_v, r.lo_cr, r.lo_lg, r.lg_ok_v, r.rho)
    end
end

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
p1 = [r for r in rows if r.p == 1]
p2 = [r for r in rows if r.p == 2]

svg_enclosure_band(joinpath(@__DIR__, "square_band_p2.svg"),
    [r.tag for r in p2], [r.lo_v for r in p2], [r.up_v for r in p2], EXACT_1;
    title = "Unit square P2: verified enclosure of lambda_1 (band collapses onto 2*pi^2)",
    ylabel = "lambda_1", refname = "2*pi^2")

svg_enclosure_band(joinpath(@__DIR__, "square_band_p1.svg"),
    [r.tag for r in p1], [r.lo_v for r in p1], [r.up_v for r in p1], EXACT_1;
    title = "Unit square P1: verified enclosure of lambda_1",
    ylabel = "lambda_1", refname = "2*pi^2")

svg_loglog_plain(joinpath(@__DIR__, "square_enclosure.svg"),
    [(x = [r.ndof for r in p1], y = [EXACT_1 - r.lo_v for r in p1], label = "P1 exact-lower"),
     (x = [r.ndof for r in p1], y = [r.up_v - EXACT_1 for r in p1], label = "P1 upper-exact"),
     (x = [r.ndof for r in p2], y = [EXACT_1 - r.lo_v for r in p2], label = "P2 exact-lower"),
     (x = [r.ndof for r in p2], y = [r.up_v - EXACT_1 for r in p2], label = "P2 upper-exact")];
    xlabel = "ndof", ylabel = "distance from 2*pi^2",
    title = "Unit square: how far each verified bound sits from the exact value")

# Two separate cost figures. Series are split by polynomial order: P1 and P2
# at the same ndof are different discretisations, so joining them with one
# polyline would draw a zig-zag that means nothing.
svg_loglog_plain(joinpath(@__DIR__, "square_time.svg"),
    [(x = [r.ndof for r in p1], y = [r.t_f for r in p1], label = "P1 float"),
     (x = [r.ndof for r in p1], y = [r.t_v for r in p1], label = "P1 interval"),
     (x = [r.ndof for r in p2], y = [r.t_f for r in p2], label = "P2 float"),
     (x = [r.ndof for r in p2], y = [r.t_v for r in p2], label = "P2 interval")];
    xlabel = "ndof", ylabel = "wall time (s)",
    title = "Unit square: wall time, float vs interval")

svg_loglog_plain(joinpath(@__DIR__, "square_width.svg"),
    [(x = [r.ndof for r in p1], y = [r.w_f for r in p1], label = "P1 float"),
     (x = [r.ndof for r in p1], y = [r.w_v for r in p1], label = "P1 interval"),
     (x = [r.ndof for r in p2], y = [r.w_f for r in p2], label = "P2 float"),
     (x = [r.ndof for r in p2], y = [r.w_v for r in p2], label = "P2 interval")];
    xlabel = "ndof", ylabel = "enclosure width",
    title = "Unit square: enclosure width, float vs interval")

@printf("\nwrote verified_square.csv, square_band_p1.svg, square_band_p2.svg, square_enclosure.svg, square_time.svg, square_width.svg\n")
println("SQUARE_DONE_MARKER")
