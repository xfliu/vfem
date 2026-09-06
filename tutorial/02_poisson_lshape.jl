#!/usr/bin/env julia
# =============================================================================
#  tutorial/examples/02_poisson_lshape.jl
#
#  VFEM.jl tutorial, Chapter 2 — the same operator on a NON-CONVEX domain.
#
#      -Delta u = f   in  Omega_L,        u = g  on  dOmega_L
#
#  This chapter exists to separate two things that are easy to confuse:
#
#      *  the domain being non-convex, and
#      *  the solution being non-smooth.
#
#  They are not the same, and only the second one costs you convergence order.
#  Three experiments, in order:
#
#    (a) SMOOTH solution on the L.  Full order p+1 / p is recovered even though
#        the domain has a reentrant corner.  Non-convexity by itself is not the
#        problem.
#    (b) The CORNER-SINGULAR solution u = r^(2/3) sin(2 theta / 3).  On uniform
#        meshes the rate collapses to h^(4/3) in L2 and h^(2/3) in H1 — and P2
#        does *not* help.  The bottleneck is the regularity of u, not the
#        polynomial degree.  This is the most important lesson in the chapter.
#    (c) The REMEDY: the same singular problem on a corner-graded mesh family
#        that is DOF-matched to the uniform one.  Full order comes back.
#
# -----------------------------------------------------------------------------
#  GEOMETRY — quote this orientation with every number below
# -----------------------------------------------------------------------------
#      Omega_L = (-1, 1)^2  \  ( [0,1] x [-1,0] )
#
#  i.e. the square of side 2 centred at the origin with the BOTTOM-RIGHT
#  quadrant removed.  Area = 3.  There is exactly one reentrant corner, AT THE
#  ORIGIN, with interior angle omega = 3 pi / 2.
#
#         y
#      1  +---------+---------+
#         |         |         |
#         |    UL   |    UR   |
#      0  +---------o---------+   <- the two edges meeting at the reentrant
#         |         |             corner o = (0,0) are  {y = 0, 0 <= x <= 1}
#         |    LL   |  (removed)  and  {x = 0, -1 <= y <= 0}
#     -1  +---------+
#        -1         0         1   x
#
#  POLAR CONVENTION (state it, because the branch cut matters):
#      r = sqrt(x^2 + y^2),   theta = mod(atan(y, x), 2 pi)  in  [0, 3 pi / 2].
#  Measured counter-clockwise from the positive x-axis, the domain sweeps
#  theta = 0 (the horizontal edge into the notch) through pi/2 and pi up to
#  theta = 3 pi / 2 (the vertical edge into the notch).  The branch cut of
#  `mod(atan(y,x), 2pi)` sits on the positive x-axis at theta = 0, which is
#  *on the boundary* of Omega_L, so no interior point is affected by it.  Note
#  `mod` also maps atan(-0.0, x<0) = -pi to +pi, so the negative x-axis (which
#  is interior) is single-valued.  Getting this wrong is the classic way to
#  produce a "singular solution" that is actually discontinuous inside the
#  domain and converges at no rate at all.
#
#  Run with:
#      julia --project=/path/to/VFEM.jl  02_poisson_lshape.jl
# =============================================================================

using VFEM
using Printf

include(get(ENV, "VFEM_TUTORIAL_SUPPORT",
            joinpath(@__DIR__, "TutorialSupport.jl")))
using .TutorialSupport

const MESHROOT = get(ENV, "VFEM_TUTORIAL_MESHDATA",
                     joinpath(@__DIR__, "meshdata"))
const OUTDIR = get(ENV, "VFEM_TUTORIAL_OUT", pwd())

# lshape_n : nv = (2n+1)^2 - n^2, nt = 6n^2, ne = 9n^2 + 4n, nb = 8n,
#            hmax = sqrt(2)/n, min angle 45 deg.
# lshape_graded_n : IDENTICAL nv/nt/ne/nb, only the coordinates differ.  The
#            grading is the max-norm map  p -> p * r^(beta-1),  r = max(|x|,|y|),
#            beta = 1.5 = 1/alpha.  Square level sets mean the polygon is still
#            discretised exactly, and the min angle is 24.956 deg at EVERY
#            level (no slivers appear as you refine).
const LEVELS = [2, 4, 8, 16, 32]

# =============================================================================
#  Exact solutions
# =============================================================================

# --- (a1) smooth, and it happens to have zero boundary data ------------------
# u = sin(pi x) sin(pi y).  Every straight piece of dOmega_L lies on one of
# x = -1, x = 0, x = 1, y = -1, y = 0, y = 1, and sin(pi *) vanishes at every
# integer, so this u is zero on the WHOLE L-shaped boundary — a homogeneous
# Dirichlet problem on a non-convex domain.  u is entire, so the only thing
# that could spoil the order is the domain.
uS(x, y)     = sin(pi * x) * sin(pi * y)
fS(x, y)     = 2 * pi^2 * sin(pi * x) * sin(pi * y)
graduS(x, y) = (pi * cos(pi * x) * sin(pi * y),
                pi * sin(pi * x) * cos(pi * y))

# --- (a2) smooth with genuinely non-zero boundary data -----------------------
# u = exp(x) cos(y) is harmonic (f = 0) and analytic on the closed L, and its
# trace is non-zero on every edge.  This exercises the Dirichlet lift on the
# non-convex domain.
uH(x, y)     = exp(x) * cos(y)
fH(x, y)     = 0.0
graduH(x, y) = (exp(x) * cos(y), -exp(x) * sin(y))

# --- (b) the corner-singular solution ----------------------------------------
# alpha = pi / omega = pi / (3 pi / 2) = 2/3.
#
#     u = r^alpha sin(alpha theta)
#
# In polar coordinates Delta = d_rr + (1/r) d_r + (1/r^2) d_thetatheta, and
# r^alpha sin(alpha theta) is annihilated by it for ANY alpha, so f = 0: this
# is a harmonic function.  It vanishes at theta = 0 (sin 0 = 0) and at
# theta = 3 pi / 2 (sin(alpha * 3pi/2) = sin(pi) = 0), i.e. on exactly the two
# edges that meet at the reentrant corner, and it is non-zero on the rest of
# the boundary.  Its gradient behaves like r^(alpha - 1) = r^(-1/3), which is
# unbounded at the origin; u itself is in H^(1+alpha-eps) = H^(5/3 - eps) but
# NOT in H^2.  That missing regularity is the whole story of part (b).
const ALPHA = 2 / 3

# Polar coordinates in the convention documented in the header.
@inline function _polar(x::Float64, y::Float64)
    r = sqrt(x * x + y * y)
    th = mod(atan(y, x), 2 * pi)
    return r, th
end

function uSing(x, y)
    r, th = _polar(Float64(x), Float64(y))
    r == 0.0 && return 0.0
    return r^ALPHA * sin(ALPHA * th)
end

fSing(x, y) = 0.0          # harmonic

# grad(r^a sin(a t)) in Cartesian coordinates.  Using
#   u_r = a r^(a-1) sin(a t),  (1/r) u_t = a r^(a-1) cos(a t)
# and the rotation to (x,y):
#   u_x = cos t * u_r - sin t * (1/r) u_t = a r^(a-1) sin((a-1) t)
#   u_y = sin t * u_r + cos t * (1/r) u_t = a r^(a-1) cos((a-1) t)
# With a - 1 = -1/3 the prefactor r^(-1/3) blows up at the corner.
function graduSing(x, y)
    r, th = _polar(Float64(x), Float64(y))
    r == 0.0 && return (0.0, 0.0)     # measure-zero point; never a quad point
    c = ALPHA * r^(ALPHA - 1)
    return (c * sin((ALPHA - 1) * th), c * cos((ALPHA - 1) * th))
end

# =============================================================================
#  Convergence driver (same shape as chapter 1, plus ndof-based orders)
# =============================================================================

struct Study
    case::String
    family::String
    p::Int
    ns::Vector{Int}
    ndofs::Vector{Int}
    hmaxs::Vector{Float64}
    xs::Vector{Float64}          # ndof^(-1/2), the fair abscissa across families
    l2::Vector{Float64}
    h1::Vector{Float64}
    l2ord_h::Vector{Float64}
    h1ord_h::Vector{Float64}
    l2ord_n::Vector{Float64}
    h1ord_n::Vector{Float64}
    sols::Vector{Vector{Float64}}
    meshes::Vector{Mesh2D}
end

function run_study(case::String, family::String, ns::Vector{Int},
                   f::Function, u::Function, gradu::Function, p::Integer;
                   g = nothing, err_rule = tri_quad_rule(8))
    ndofs = Int[]; hmaxs = Float64[]; l2 = Float64[]; h1 = Float64[]
    sols = Vector{Vector{Float64}}(); meshes = Mesh2D[]

    @printf("\n  %s  on %s_*   (Lagrange P%d)\n", case, family, p)
    @printf("  %4s %7s %10s %11s   %12s %6s   %12s %6s  %6s\n",
            "n", "ndof", "hmax", "ndof^-1/2", "L2 error", "ord_n",
            "H1 error", "ord_n", "sec")
    println("  " * "-"^88)

    for n in ns
        m = mesh2d_load(joinpath(MESHROOT, "$(family)_$(n)"))
        h = mesh_hmax(m)
        t0 = time()
        sol = solve_poisson(m, f, p; g = g)
        el2 = l2_error(m, sol.uh, u, p; rule = err_rule)
        eh1 = h1_seminorm_error(m, sol.uh, gradu, p; rule = err_rule)
        dt = time() - t0

        push!(ndofs, length(sol.uh)); push!(hmaxs, h)
        push!(l2, el2); push!(h1, eh1)
        push!(sols, sol.uh); push!(meshes, m)

        xs = [nd^(-0.5) for nd in ndofs]
        o2 = length(l2) > 1 ? observed_order(l2[end-1:end], xs[end-1:end])[1] : NaN
        o1 = length(h1) > 1 ? observed_order(h1[end-1:end], xs[end-1:end])[1] : NaN
        @printf("  %4d %7d %10.3e %11.3e   %12.5e %6s   %12.5e %6s  %6.2f\n",
                n, ndofs[end], h, xs[end], el2,
                isnan(o2) ? "  -  " : @sprintf("%5.3f", o2), eh1,
                isnan(o1) ? "  -  " : @sprintf("%5.3f", o1), dt)
        flush(stdout)
    end

    xs = [nd^(-0.5) for nd in ndofs]
    return Study(case, family, Int(p), ns, ndofs, hmaxs, xs, l2, h1,
                 observed_order(l2, hmaxs), observed_order(h1, hmaxs),
                 observed_order(l2, xs),    observed_order(h1, xs),
                 sols, meshes)
end

# A zoom of the mesh near the reentrant corner: keep the triangles whose
# centroid lies in the max-norm box of radius `rad` around the origin and
# rebuild a standalone Mesh2D from them.  This is the honest way to show what
# grading actually does, because at n = 16 the whole-domain picture is a grey
# smudge near the corner in both families.
function corner_zoom(m::Mesh2D, rad::Float64)
    keep = Int[]
    for k in 1:m.nt
        cx = (m.nodes[m.elements[k, 1], 1] + m.nodes[m.elements[k, 2], 1] +
              m.nodes[m.elements[k, 3], 1]) / 3
        cy = (m.nodes[m.elements[k, 1], 2] + m.nodes[m.elements[k, 2], 2] +
              m.nodes[m.elements[k, 3], 2]) / 3
        max(abs(cx), abs(cy)) <= rad && push!(keep, k)
    end
    old2new = zeros(Int, m.nv)
    coords = Vector{NTuple{2, Float64}}()
    for k in keep, j in 1:3
        v = m.elements[k, j]
        if old2new[v] == 0
            push!(coords, (m.nodes[v, 1], m.nodes[v, 2]))
            old2new[v] = length(coords)
        end
    end
    nodes = Matrix{Float64}(undef, length(coords), 2)
    for (i, (x, y)) in enumerate(coords)
        nodes[i, 1] = x; nodes[i, 2] = y
    end
    elems = Matrix{Int}(undef, length(keep), 3)
    for (i, k) in enumerate(keep), j in 1:3
        elems[i, j] = old2new[m.elements[k, j]]
    end
    return mesh2d_from_nodes_elements(nodes, elems)
end

# =============================================================================
#  Main
# =============================================================================

println("="^88)
println(" VFEM.jl tutorial 02 : Poisson on the L-shaped domain")
println("="^88)
println(" Omega_L = (-1,1)^2 \\ ([0,1] x [-1,0])   -- BOTTOM-RIGHT quadrant removed")
println(" area 3,  one reentrant corner AT THE ORIGIN,  interior angle omega = 3pi/2")
println(" alpha = pi/omega = 2/3   -> singular exponent of the corner")
println(" theta = mod(atan(y,x), 2pi) in [0, 3pi/2], measured CCW from +x axis")
println(" mesh root : ", MESHROOT)
println(" levels    : lshape_", join(LEVELS, ", lshape_"),
        "   and the DOF-matched lshape_graded_* family")
println(" Julia     : ", VERSION)

# --- mesh bookkeeping, and the DOF-match claim, checked not assumed ----------
println("\n--- mesh families: uniform vs graded (DOF-matched?) ---")
@printf("  %4s | %6s %6s %6s %5s | %12s %12s | %s\n",
        "n", "nv", "nt", "ne", "nb", "hmax unif", "hmax grad", "counts equal")
for n in LEVELS
    mu = mesh2d_load(joinpath(MESHROOT, "lshape_$(n)"))
    mg = mesh2d_load(joinpath(MESHROOT, "lshape_graded_$(n)"))
    same = (mu.nv, mu.nt, mu.ne, mu.nb) == (mg.nv, mg.nt, mg.ne, mg.nb)
    @printf("  %4d | %6d %6d %6d %5d | %12.6e %12.6e | %s\n",
            n, mu.nv, mu.nt, mu.ne, mu.nb, mesh_hmax(mu), mesh_hmax(mg), same)
end
println("  NOTE the graded hmax is LARGER at equal n (the far field is stretched")
println("  to pay for the refinement at the corner).  That is precisely why the")
println("  graded-vs-uniform comparison below is plotted against ndof^(-1/2).")

# --- sanity gate -------------------------------------------------------------
println("\n--- DOF-ordering check on lshape_8 and lshape_graded_8 ---")
for fam in ("lshape", "lshape_graded")
    m = mesh2d_load(joinpath(MESHROOT, "$(fam)_8"))
    for p in (1, 2)
        v = verify_dof_ordering(m, p)
        @printf("  %-14s p=%d  ndof=%5d  relA=%.2e  relM=%.2e  ok=%s\n",
                fam, p, v.ndof, v.relA, v.relM, v.ok)
        v.ok || error("DOF ordering mismatch on $fam at p=$p")
    end
end

# =============================================================================
#  (a) SMOOTH solutions on the non-convex domain
# =============================================================================
println("\n" * "="^88)
println(" PART (a)  SMOOTH solution on the L-shape")
println("="^88)
println(" (a1) u = sin(pi x) sin(pi y),  f = 2 pi^2 u.   This u vanishes on the")
println("      ENTIRE L-shaped boundary (every edge lies on an integer line),")
println("      so (a1) is a homogeneous Dirichlet problem.")
println(" (a2) u = exp(x) cos(y),  f = 0,  non-zero trace on every edge.")
println(" Both are analytic on the closed domain -> expect the full p+1 / p.")

S1 = run_study("smooth_sin_sin",  "lshape", LEVELS, fS, uS, graduS, 1)
S2 = run_study("smooth_sin_sin",  "lshape", LEVELS, fS, uS, graduS, 2)
H1s = run_study("smooth_harmonic", "lshape", LEVELS, fH, uH, graduH, 1; g = uH)
H2s = run_study("smooth_harmonic", "lshape", LEVELS, fH, uH, graduH, 2; g = uH)

# =============================================================================
#  (b) the corner-singular solution on UNIFORM meshes
# =============================================================================
println("\n" * "="^88)
println(" PART (b)  CORNER-SINGULAR solution on UNIFORM meshes")
println("="^88)
println(" u = r^(2/3) sin(2 theta / 3),  f = 0,  u = 0 on the two notch edges")
println(" and u = trace elsewhere.  u is in H^(5/3 - eps) but NOT in H^2.")
println(" Theory: |u - u_h|_1 = O(h^alpha) = O(h^(2/3)) and")
println("         ||u - u_h||_0 = O(h^(2 alpha)) = O(h^(4/3)), FOR EVERY p >= 1.")
println(" A stronger error quadrature (tri_quad_rule(12), 144 pts, degree 22) is")
println(" used here because the H1 integrand has an integrable r^(-2/3)")
println(" singularity on the three elements touching the corner.  The rule is")
println(" the same at every level and the corner elements are self-similar, so")
println(" the residual quadrature error is level-independent and does not bend")
println(" the observed order.")

const SING_RULE = tri_quad_rule(12)

U1 = run_study("singular_r23", "lshape", LEVELS, fSing, uSing, graduSing, 1;
               g = uSing, err_rule = SING_RULE)
U2 = run_study("singular_r23", "lshape", LEVELS, fSing, uSing, graduSing, 2;
               g = uSing, err_rule = SING_RULE)

# =============================================================================
#  (c) the corner-singular solution on the GRADED family
# =============================================================================
println("\n" * "="^88)
println(" PART (c)  the SAME singular problem on the corner-GRADED family")
println("="^88)
println(" lshape_graded_n has exactly the same nv/nt/ne/nb as lshape_n, so this")
println(" is a DOF-for-DOF fair comparison.  Grading exponent beta = 1/alpha = 1.5.")
println(" beta = 1/alpha is the P1-tuned choice: alpha*beta = 1, so it restores")
println(" L2 order 2 and H1 order 1 in ndof^(-1/2) -- optimal for P1.")
println(" For P2 it should NOT reach 3/2: the ceiling min(p, alpha*beta) is still")
println(" 1.  Part (d) tests that by raising beta to 3.")

G1 = run_study("singular_r23", "lshape_graded", LEVELS, fSing, uSing, graduSing, 1;
               g = uSing, err_rule = SING_RULE)
G2 = run_study("singular_r23", "lshape_graded", LEVELS, fSing, uSing, graduSing, 2;
               g = uSing, err_rule = SING_RULE)

# =============================================================================
#  (d) HOW MUCH grading?  beta must match the polynomial degree too
# =============================================================================
#
#  Part (c) restores the full rate for P1 but visibly does NOT for P2.  That is
#  not a bug: beta = 1/alpha is the P1-tuned grading.  The mechanism is that
#  grading buys you resolution near the corner at a rate controlled by beta,
#  and the amount of resolution you need there grows with p, because the
#  interpolation error of a degree-p element involves the (p+1)-st derivative
#  of u ~ r^alpha, i.e. r^(alpha - p - 1).  The rate you can achieve is
#  therefore capped:
#
#      H1 order in ndof^(-1/2)  =  min(p,     alpha * beta)
#      L2 order in ndof^(-1/2)  =  min(p + 1, 2 * alpha * beta)
#
#  The factor 2 in the L2 line is the usual duality (Aubin-Nitsche) gain: the
#  L2 rate is twice the energy rate until it saturates at the interpolation
#  ceiling p+1.  Check the two ends against the runs above:
#    beta = 1 (uniform), any p : H1 min(p, 2/3) = 2/3, L2 min(p+1, 4/3) = 4/3
#                               -- part (b) measured 0.674/1.338 and 0.677/1.384
#    beta = 1.5, alpha*beta = 1: H1 min(p, 1),   L2 min(p+1, 2)
#                               -- P1-optimal; for P2 this CAPS the rate at 1/2
#
#  So beta = 1/alpha is the P1-tuned grading, and part (c)'s P2 rows are not a
#  bug: they are the ceiling.  To get the full P2 rate you need
#  alpha * beta >= p, i.e.
#
#      beta >= p / alpha = 2 / (2/3) = 3.
#
#  The shipped mesh family only has beta = 1.5, so we build the beta = 3 family
#  in memory by applying the SAME documented grading map to the uniform nodes:
#
#      p -> p * r^(beta - 1),      r = max(|x|, |y|)      (max-norm!)
#
#  The max norm is what keeps the square level sets, so the boundary of the
#  polygon is preserved pointwise and the graded mesh discretises exactly the
#  same L-shape.  We first apply the map with beta = 1.5 and check that it
#  reproduces the shipped lshape_graded_n coordinates to round-off; only then
#  do we trust it at beta = 3.

"""
    grade_mesh(m, beta) -> Mesh2D

Apply the tutorial's corner grading `p -> p * max(|x|,|y|)^(beta-1)` to every
vertex of `m`, keeping the connectivity.  `beta = 1` is the identity.
"""
function grade_mesh(m::Mesh2D, beta::Float64)
    nodes = copy(m.nodes)
    for v in 1:m.nv
        x = nodes[v, 1]; y = nodes[v, 2]
        r = max(abs(x), abs(y))
        if r > 0
            s = r^(beta - 1)
            nodes[v, 1] = x * s
            nodes[v, 2] = y * s
        end
    end
    return Mesh2D(nodes, m.elements, m.edges, m.bd_edges, m.bd_edge_ids,
                  m.tri2edge, m.nv, m.nt, m.ne, m.nb)
end

println("\n" * "="^88)
println(" PART (d)  choosing beta: the grading exponent must match p as well as alpha")
println("="^88)
println(" claim: order ceiling is  H1 min(p, alpha*beta),  L2 min(p+1, alpha*beta+1)")
println(" alpha = 2/3, so beta = 1.5 -> ceiling 1 (P1-optimal, P2 capped)")
println("          and beta = 3.0 -> ceiling 2 (P2-optimal)")

# --- first, validate the grading map against the shipped family --------------
println("\n --- grade_mesh(uniform, 1.5) vs the shipped lshape_graded_* ---")
for n in LEVELS
    mu = mesh2d_load(joinpath(MESHROOT, "lshape_$(n)"))
    mg = mesh2d_load(joinpath(MESHROOT, "lshape_graded_$(n)"))
    mm = grade_mesh(mu, 1.5)
    d = maximum(abs.(mm.nodes .- mg.nodes))
    @printf("   n=%2d   max|nodes_reconstructed - nodes_shipped| = %.3e\n", n, d)
    d < 1e-14 || error("grading map does not reproduce the shipped graded mesh")
end
println("   => the map above IS the shipped grading; beta = 3 is the same map,")
println("      stronger.")

# --- now the sweep -----------------------------------------------------------
struct Sweep
    beta::Float64
    p::Int
    ndofs::Vector{Int}
    xs::Vector{Float64}
    l2::Vector{Float64}
    h1::Vector{Float64}
    l2ord::Vector{Float64}
    h1ord::Vector{Float64}
    hmaxs::Vector{Float64}
end

function run_sweep(beta::Float64, p::Integer)
    ndofs = Int[]; l2 = Float64[]; h1 = Float64[]; hmaxs = Float64[]
    @printf("\n  beta = %.2f,  P%d   (in-memory graded mesh, alpha*beta = %.3f)\n",
            beta, p, ALPHA * beta)
    @printf("  %4s %7s %11s %11s   %12s %6s   %12s %6s\n",
            "n", "ndof", "hmax", "ndof^-1/2", "L2 error", "ord_n", "H1 error", "ord_n")
    println("  " * "-"^82)
    for n in LEVELS
        m = grade_mesh(mesh2d_load(joinpath(MESHROOT, "lshape_$(n)")), beta)
        sol = solve_poisson(m, fSing, p; g = uSing)
        el2 = l2_error(m, sol.uh, uSing, p; rule = SING_RULE)
        eh1 = h1_seminorm_error(m, sol.uh, graduSing, p; rule = SING_RULE)
        push!(ndofs, length(sol.uh)); push!(l2, el2); push!(h1, eh1)
        push!(hmaxs, mesh_hmax(m))
        xs = [nd^(-0.5) for nd in ndofs]
        o2 = length(l2) > 1 ? observed_order(l2[end-1:end], xs[end-1:end])[1] : NaN
        o1 = length(h1) > 1 ? observed_order(h1[end-1:end], xs[end-1:end])[1] : NaN
        @printf("  %4d %7d %11.3e %11.3e   %12.5e %6s   %12.5e %6s\n",
                n, ndofs[end], hmaxs[end], xs[end], el2,
                isnan(o2) ? "  -  " : @sprintf("%5.3f", o2), eh1,
                isnan(o1) ? "  -  " : @sprintf("%5.3f", o1))
        flush(stdout)
    end
    xs = [nd^(-0.5) for nd in ndofs]
    return Sweep(beta, Int(p), ndofs, xs, l2, h1,
                 observed_order(l2, xs), observed_order(h1, xs), hmaxs)
end

D_b15_p2 = run_sweep(1.5, 2)      # must match G2 -- consistency check
D_b30_p2 = run_sweep(3.0, 2)      # the prediction: L2 -> 3, H1 -> 2
D_b30_p1 = run_sweep(3.0, 1)      # over-graded for P1: still optimal, mildly worse constant

@printf("\n  consistency: beta=1.5 P2 in-memory vs shipped lshape_graded_32:\n")
@printf("    L2 %.6e vs %.6e   (rel %.2e)\n", D_b15_p2.l2[end], G2.l2[end],
        abs(D_b15_p2.l2[end] - G2.l2[end]) / G2.l2[end])

println("\n  measured order ceilings (finest pair, ndof^(-1/2)):")
println("  prediction: H1 = min(p, alpha*beta),  L2 = min(p+1, 2*alpha*beta)")
@printf("  %-26s %8s %8s   %s\n", "family", "L2", "H1", "predicted L2 / H1")
predL2(p, beta) = min(p + 1.0, 2 * ALPHA * beta)
predH1(p, beta) = min(Float64(p), ALPHA * beta)
for (nm, l2o, h1o, p, beta) in
        (("uniform (beta = 1), P1", U1.l2ord_n[end], U1.h1ord_n[end], 1, 1.0),
         ("uniform (beta = 1), P2", U2.l2ord_n[end], U2.h1ord_n[end], 2, 1.0),
         ("beta = 1.5, P1",         G1.l2ord_n[end], G1.h1ord_n[end], 1, 1.5),
         ("beta = 1.5, P2",         G2.l2ord_n[end], G2.h1ord_n[end], 2, 1.5),
         ("beta = 3.0, P2",   D_b30_p2.l2ord[end], D_b30_p2.h1ord[end], 2, 3.0),
         ("beta = 3.0, P1",   D_b30_p1.l2ord[end], D_b30_p1.h1ord[end], 1, 3.0))
    @printf("  %-26s %8.3f %8.3f   %.2f / %.2f\n",
            nm, l2o, h1o, predL2(p, beta), predH1(p, beta))
end

# =============================================================================
#  CSV export
# =============================================================================
function write_csv(path::AbstractString, studies::Vector{Study})
    open(path, "w") do io
        println(io, "case,family,degree,level,n,ndof,hmax,ndof_scale,",
                    "l2_err,l2_order,l2_order_ndof,h1_err,h1_order,h1_order_ndof")
        for s in studies
            for i in eachindex(s.ns)
                fmt(v) = isnan(v) ? "" : @sprintf("%.6f", v)
                o2h = i > 1 ? s.l2ord_h[i-1] : NaN
                o1h = i > 1 ? s.h1ord_h[i-1] : NaN
                o2n = i > 1 ? s.l2ord_n[i-1] : NaN
                o1n = i > 1 ? s.h1ord_n[i-1] : NaN
                @printf(io, "%s,%s,%d,%d,%d,%d,%.10e,%.10e,%.10e,%s,%s,%.10e,%s,%s\n",
                        s.case, s.family, s.p, i, s.ns[i], s.ndofs[i],
                        s.hmaxs[i], s.xs[i],
                        s.l2[i], fmt(o2h), fmt(o2n),
                        s.h1[i], fmt(o1h), fmt(o1n))
            end
        end
    end
    println("  wrote ", path)
end

println("\n--- exports ---")
write_csv(joinpath(OUTDIR, "bvp_lshape_uniform.csv"),
          Study[S1, S2, H1s, H2s, U1, U2])
write_csv(joinpath(OUTDIR, "bvp_lshape_graded.csv"), Study[G1, G2])

# The beta sweep gets its own file: it is a different independent variable.
open(joinpath(OUTDIR, "bvp_lshape_grading_sweep.csv"), "w") do io
    println(io, "beta,degree,level,n,ndof,hmax,ndof_scale,",
                "l2_err,l2_order_ndof,h1_err,h1_order_ndof,alpha_beta")
    for s in (D_b15_p2, D_b30_p2, D_b30_p1)
        for i in eachindex(s.ndofs)
            fmt(v) = isnan(v) ? "" : @sprintf("%.6f", v)
            o2 = i > 1 ? s.l2ord[i-1] : NaN
            o1 = i > 1 ? s.h1ord[i-1] : NaN
            @printf(io, "%.2f,%d,%d,%d,%d,%.10e,%.10e,%.10e,%s,%.10e,%s,%.6f\n",
                    s.beta, s.p, i, LEVELS[i], s.ndofs[i], s.hmaxs[i], s.xs[i],
                    s.l2[i], fmt(o2), s.h1[i], fmt(o1), ALPHA * s.beta)
        end
    end
end
println("  wrote ", joinpath(OUTDIR, "bvp_lshape_grading_sweep.csv"))

# =============================================================================
#  Figures
# =============================================================================
# Solutions.  The singular one at n = 32 shows the gradient blow-up: the colour
# bands crowd together as they approach the origin from inside the domain.
let i = findfirst(==(32), LEVELS)
    svg_solution(joinpath(OUTDIR, "fig_lshape_solution_singular_p1.svg"),
                 U1.meshes[i], U1.sols[i]; p = 1, draw_mesh = false,
                 title = "u_h,  P1 on lshape_32,  u = r^(2/3) sin(2theta/3)")
    svg_solution(joinpath(OUTDIR, "fig_lshape_solution_singular_graded_p1.svg"),
                 G1.meshes[i], G1.sols[i]; p = 1, draw_mesh = false,
                 title = "u_h,  P1 on lshape_graded_32,  same singular solution")
    svg_solution(joinpath(OUTDIR, "fig_lshape_solution_smooth_p2.svg"),
                 S2.meshes[i], S2.sols[i]; p = 2,
                 title = "u_h,  P2 on lshape_32,  smooth u = sin(pi x) sin(pi y)")
end

# Meshes: whole domain at n = 8 ...
let i = findfirst(==(8), LEVELS)
    svg_mesh(joinpath(OUTDIR, "fig_lshape_mesh_uniform.svg"), U1.meshes[i];
             title = "lshape_8  (uniform)")
    svg_mesh(joinpath(OUTDIR, "fig_lshape_mesh_graded.svg"), G1.meshes[i];
             title = "lshape_graded_8  (beta = 1.5, max-norm grading)")
end
# ... and zoomed into the reentrant corner at n = 16, which is where the two
# families visibly part company.
# `show_boundary = false`: in a zoom the "boundary" of the extracted patch is
# mostly the artificial cut, not dOmega, so highlighting it would mislead.
let i = findfirst(==(16), LEVELS)
    svg_mesh(joinpath(OUTDIR, "fig_lshape_corner_uniform.svg"),
             corner_zoom(U1.meshes[i], 0.25); show_boundary = false,
             title = "lshape_16 near the corner (|x|,|y| <= 0.25)")
    svg_mesh(joinpath(OUTDIR, "fig_lshape_corner_graded.svg"),
             corner_zoom(G1.meshes[i], 0.25); show_boundary = false,
             title = "lshape_graded_16 near the corner (same window)")
end

# Convergence, uniform family: smooth (full rate) against singular (reduced).
svg_loglog(joinpath(OUTDIR, "fig_conv_lshape_uniform.svg"),
           [(x = S1.hmaxs, y = S1.l2, label = "P1 L2  smooth",   slope = 2.0),
            (x = S2.hmaxs, y = S2.l2, label = "P2 L2  smooth",   slope = 3.0),
            (x = U1.hmaxs, y = U1.l2, label = "P1 L2  singular", slope = 4 / 3),
            (x = U2.hmaxs, y = U2.l2, label = "P2 L2  singular", slope = 4 / 3)];
           xlabel = "h_max", ylabel = "L2 error",
           title = "L-shape, uniform meshes: smooth vs r^(2/3) (P2 does not help)")

svg_loglog(joinpath(OUTDIR, "fig_conv_lshape_uniform_h1.svg"),
           [(x = S1.hmaxs, y = S1.h1, label = "P1 H1  smooth",   slope = 1.0),
            (x = S2.hmaxs, y = S2.h1, label = "P2 H1  smooth",   slope = 2.0),
            (x = U1.hmaxs, y = U1.h1, label = "P1 H1  singular", slope = 2 / 3),
            (x = U2.hmaxs, y = U2.h1, label = "P2 H1  singular", slope = 2 / 3)];
           xlabel = "h_max", ylabel = "H1 seminorm error",
           title = "L-shape, uniform meshes: H1 seminorm, smooth vs singular")

# The payoff figure: uniform vs graded, plotted against ndof^(-1/2) so that
# points at the same abscissa cost the same number of unknowns.
svg_loglog(joinpath(OUTDIR, "fig_conv_lshape_graded.svg"),
           [(x = U1.xs, y = U1.l2, label = "P1 L2  uniform (b=1)",   slope = 4 / 3),
            (x = G1.xs, y = G1.l2, label = "P1 L2  graded b=1.5",    slope = 2.0),
            (x = U2.xs, y = U2.l2, label = "P2 L2  uniform (b=1)",   slope = 4 / 3),
            (x = G2.xs, y = G2.l2, label = "P2 L2  graded b=1.5",    slope = 2.0)];
           xlabel = "ndof^(-1/2)   (DOF-matched abscissa)", ylabel = "L2 error",
           title = "L-shape, u = r^(2/3): beta = 1.5 grading restores order 2")

svg_loglog(joinpath(OUTDIR, "fig_conv_lshape_graded_h1.svg"),
           [(x = U1.xs, y = U1.h1, label = "P1 H1  uniform (b=1)",   slope = 2 / 3),
            (x = G1.xs, y = G1.h1, label = "P1 H1  graded b=1.5",    slope = 1.0),
            (x = U2.xs, y = U2.h1, label = "P2 H1  uniform (b=1)",   slope = 2 / 3),
            (x = G2.xs, y = G2.h1, label = "P2 H1  graded b=1.5",    slope = 1.0)];
           xlabel = "ndof^(-1/2)   (DOF-matched abscissa)",
           ylabel = "H1 seminorm error",
           title = "L-shape, u = r^(2/3): H1 seminorm, uniform vs graded (beta = 1.5)")

# Part (d): the beta sweep at P2, showing the min(p, alpha*beta) ceiling move
# from 1 (beta = 1.5) to 2 (beta = 3).
svg_loglog(joinpath(OUTDIR, "fig_conv_lshape_beta_sweep.svg"),
           [(x = U2.xs,        y = U2.h1,        label = "P2 H1  beta=1 (uniform)", slope = 2 / 3),
            (x = D_b15_p2.xs,  y = D_b15_p2.h1,  label = "P2 H1  beta=1.5",         slope = 1.0),
            (x = D_b30_p2.xs,  y = D_b30_p2.h1,  label = "P2 H1  beta=3",           slope = 2.0)];
           xlabel = "ndof^(-1/2)", ylabel = "H1 seminorm error",
           title = "P2 on the L-shape: H1 order = min(p, alpha*beta), alpha = 2/3")

svg_loglog(joinpath(OUTDIR, "fig_conv_lshape_beta_sweep_l2.svg"),
           [(x = U2.xs,        y = U2.l2,        label = "P2 L2  beta=1 (uniform)", slope = 4 / 3),
            (x = D_b15_p2.xs,  y = D_b15_p2.l2,  label = "P2 L2  beta=1.5",         slope = 2.0),
            (x = D_b30_p2.xs,  y = D_b30_p2.l2,  label = "P2 L2  beta=3",           slope = 3.0)];
           xlabel = "ndof^(-1/2)", ylabel = "L2 error",
           title = "P2 on the L-shape: L2 order = min(p+1, 2*alpha*beta)")

# What beta = 3 does to the mesh, for the reader to compare against beta = 1.5.
let i = findfirst(==(16), LEVELS)
    mu = mesh2d_load(joinpath(MESHROOT, "lshape_$(LEVELS[i])"))
    svg_mesh(joinpath(OUTDIR, "fig_lshape_corner_beta3.svg"),
             corner_zoom(grade_mesh(mu, 3.0), 0.25); show_boundary = false,
             title = "beta = 3 grading near the corner (same window)")
end

for fn in ("fig_lshape_solution_singular_p1.svg",
           "fig_lshape_solution_singular_graded_p1.svg",
           "fig_lshape_solution_smooth_p2.svg",
           "fig_lshape_mesh_uniform.svg", "fig_lshape_mesh_graded.svg",
           "fig_lshape_corner_uniform.svg", "fig_lshape_corner_graded.svg",
           "fig_conv_lshape_uniform.svg", "fig_conv_lshape_uniform_h1.svg",
           "fig_conv_lshape_graded.svg", "fig_conv_lshape_graded_h1.svg",
           "fig_conv_lshape_beta_sweep.svg", "fig_conv_lshape_beta_sweep_l2.svg",
           "fig_lshape_corner_beta3.svg")
    println("  wrote ", joinpath(OUTDIR, fn))
end

# =============================================================================
#  Summary
# =============================================================================
println("\n" * "="^88)
println(" SUMMARY  (orders on the finest pair of levels, in ndof^(-1/2))")
println("="^88)
@printf("  %-34s %-6s %8s %8s   %s\n", "case", "degree", "L2", "H1", "expected")
for (s, exp2, exp1) in ((S1, "2", "1"), (S2, "3", "2"),
                        (H1s, "2", "1"), (H2s, "3", "2"),
                        (U1, "4/3", "2/3"), (U2, "4/3", "2/3"),
                        (G1, "2", "1"), (G2, "2 (capped)", "1 (capped)"))
    @printf("  %-34s P%-5d %8.3f %8.3f   L2 %s / H1 %s\n",
            s.case * " [" * s.family * "]", s.p,
            s.l2ord_n[end], s.h1ord_n[end], exp2, exp1)
end
println("="^88)
println(" errors at the finest DOF-matched level (n = 32, singular solution):")
@printf("   P1 uniform L2 %.6e  H1 %.6e\n", U1.l2[end], U1.h1[end])
@printf("   P1 graded  L2 %.6e  H1 %.6e   (gain x%.1f / x%.1f)\n",
        G1.l2[end], G1.h1[end], U1.l2[end] / G1.l2[end], U1.h1[end] / G1.h1[end])
@printf("   P2 uniform L2 %.6e  H1 %.6e\n", U2.l2[end], U2.h1[end])
@printf("   P2 graded  L2 %.6e  H1 %.6e   (gain x%.1f / x%.1f)\n",
        G2.l2[end], G2.h1[end], U2.l2[end] / G2.l2[end], U2.h1[end] / G2.h1[end])
println("="^88)
println("02_poisson_lshape.jl DONE")
