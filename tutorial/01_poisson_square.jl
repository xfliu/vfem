#!/usr/bin/env julia
# =============================================================================
#  tutorial/examples/01_poisson_square.jl
#
#  VFEM.jl tutorial, Chapter 1 — the Poisson problem on a convex domain.
#
#      -Delta u = f   in  Omega = (0,1)^2,        u = g  on  dOmega
#
#  This is the reader's first contact with the library, so the script is
#  written to be read top-to-bottom.  It does three things:
#
#    Part 0  spells out the whole finite element pipeline by hand — assemble,
#            restrict to interior DOFs, solve — and then checks that the
#            hand-written version agrees to round-off with the helper
#            `TutorialSupport.solve_poisson` used afterwards.  If you only
#            read one section, read this one.
#    Part A  homogeneous Dirichlet data, manufactured solution
#                u = sin(pi x) sin(pi y),   f = 2 pi^2 sin(pi x) sin(pi y)
#            refined over unit_square_{4,8,16,32,64} at P1 and P2.
#    Part B  NON-homogeneous Dirichlet data with the harmonic solution
#                u = exp(x) cos(y),   f = 0,   g = u|_dOmega
#            which exercises the Dirichlet-lifting path (a lift that is
#            *not* identically zero) on the same mesh family.
#
#  Run it with (the --project flag is mandatory — VFEM.jl is unregistered and
#  its dependencies live in that project's manifest):
#
#      julia --project=. tutorial/01_poisson_square.jl
#
# -----------------------------------------------------------------------------
#  THE WEAK FORM  (the one piece of mathematics you need before the code)
# -----------------------------------------------------------------------------
#  Multiply -Delta u = f by a test function v that vanishes on the boundary,
#  integrate over Omega, and integrate by parts.  The boundary term drops
#  because v = 0 there, leaving: find u in H^1(Omega) with u = g on dOmega and
#
#      a(u, v) := integral_Omega  grad u . grad v  dx
#               = integral_Omega  f v  dx  =: (f, v)      for all v in H^1_0.
#
#  Discretely, with {phi_1, ..., phi_ndof} a basis of the order-p conforming
#  Lagrange space V_h on the triangulation:
#
#      A[i,j] = a(phi_j, phi_i)      <- VFEM.lagrange_laplace_matrices
#      b[i]   = (f, phi_i)           <- TutorialSupport.assemble_load_vector
#
#  and the Dirichlet condition is imposed by *elimination*: split the DOF
#  index set into interior I and boundary B, write u_h = u_g + w with u_g
#  carrying the boundary data and w vanishing on B, and solve
#
#      A[I,I] w[I] = b[I] - (A * u_g)[I].
#
#  Homogeneous data is the special case u_g = 0, i.e. A[I,I] u[I] = b[I].
# =============================================================================

using VFEM                 # Mesh2D, mesh2d_load, lagrange_laplace_matrices, ...
using Printf
using LinearAlgebra: norm

# TutorialSupport.jl is a plain module file (not a package): include it, then
# bring its exports into scope with the leading dot.
include(get(ENV, "VFEM_TUTORIAL_SUPPORT",
            joinpath(@__DIR__, "TutorialSupport.jl")))
using .TutorialSupport

# Where the pre-generated mesh folders live.  Each folder holds vert.dat,
# tri.dat, edge.dat, bd.dat and is read by `VFEM.mesh2d_load`.
const MESHROOT = get(ENV, "VFEM_TUTORIAL_MESHDATA",
                     joinpath(@__DIR__, "meshdata"))

const OUTDIR = get(ENV, "VFEM_TUTORIAL_OUT", pwd())

# The refinement ladder.  unit_square_n has nv=(n+1)^2, nt=2n^2, ne=3n^2+2n,
# nb=4n and hmax = sqrt(2)/n (right-triangle mesh, min angle 45 degrees).
const LEVELS = [4, 8, 16, 32, 64]

# =============================================================================
#  Part 0 — the pipeline, written out once, by hand
# =============================================================================
#
#  Everything `solve_poisson` does, in a dozen lines, so that the helper is
#  never a black box.  We run it on one mesh at P1 and P2 and compare.

"""
    poisson_by_hand(m, f, p) -> Vector{Float64}

Solve -Delta u = f, u = 0 on dOmega, spelling out every step.  Returned vector
is the full-length coefficient vector in the *monomial* barycentric basis that
`lagrange_laplace_matrices` uses (NOT nodal values — see the note below).
"""
function poisson_by_hand(m::Mesh2D, f::Function, p::Integer)
    # 1. The library assembles.  Note the return triple: stiffness FIRST, then
    #    mass, then the list of Dirichlet (boundary) DOF indices.  No boundary
    #    condition has been applied to A or M — that is the caller's job, and
    #    that is deliberate: the same A and M feed the verified eigenvalue
    #    pipeline, which needs them untouched.
    A, M, bd_dofs = lagrange_laplace_matrices(m, p)

    # 2. The load vector b[i] = integral f * phi_i.  VFEM.jl has no load
    #    vector (it is an eigenvalue library), so this comes from the tutorial
    #    helper.  It uses the library's own 6-point degree-4 triangle rule.
    b = assemble_load_vector(m, f, p)

    # 3. Split the DOFs.  `bd_dofs` are the constrained ones; everything else
    #    is a genuine unknown.
    ndof = size(A, 1)
    is_bd = falses(ndof)
    is_bd[bd_dofs] .= true
    int_dofs = findall(!, is_bd)

    # 4. Restrict.  A[I,I] is symmetric positive definite once the Dirichlet
    #    rows and columns are gone (before that, A is only semi-definite: the
    #    constant vector is in its kernel).
    A_ii = A[int_dofs, int_dofs]
    b_i  = b[int_dofs]

    # 5. Solve.  `\` on a sparse SPD matrix goes through CHOLMOD.
    w = A_ii \ b_i

    # 6. Scatter back into a full-length vector, zeros on the boundary.
    uh = zeros(Float64, ndof)
    uh[int_dofs] .= w
    return uh
end

# NOTE, and it is the single most important gotcha in the whole tutorial:
# `lagrange_laplace_matrices` uses the MONOMIAL barycentric basis
#     phi_(i,j,k) = L1^i L2^j L3^k,      i + j + k = p
# not the nodal Lagrange basis (VFEM.jl's *other* Lagrange assembler,
# `create_matrix_lagrange`, is the nodal one; the two span the same space and
# give the same eigenvalues but different coefficient vectors).  So for p = 2,
# `uh[dof]` is the value of u_h at a vertex only because L1^2 = 1 there, and it
# is NOT the value at an edge midpoint.  Use `fe_vertex_values(m, uh, p)` to
# plot and `interpolate_nodal(m, g, p)` to turn point values into coefficients.
# Never index `uh` yourself.

# =============================================================================
#  The two manufactured solutions
# =============================================================================

# --- Part A: homogeneous Dirichlet -------------------------------------------
# u = sin(pi x) sin(pi y) vanishes on all four sides of the unit square, and
#     -Delta u = 2 pi^2 sin(pi x) sin(pi y).
uA(x, y)      = sin(pi * x) * sin(pi * y)
fA(x, y)      = 2 * pi^2 * sin(pi * x) * sin(pi * y)
graduA(x, y)  = (pi * cos(pi * x) * sin(pi * y),
                 pi * sin(pi * x) * cos(pi * y))

# --- Part B: non-homogeneous Dirichlet ---------------------------------------
# u = exp(x) cos(y) is harmonic:  u_xx = exp(x) cos(y), u_yy = -exp(x) cos(y),
# so Delta u = 0 and f = 0 exactly.  All the information now enters through the
# boundary data g = u|_dOmega, which is non-zero on every side.  This is the
# case that would still converge at the right rate with a broken load vector
# and would fail immediately with a broken Dirichlet lift, so it is exactly
# complementary to Part A.
uB(x, y)      = exp(x) * cos(y)
fB(x, y)      = 0.0
graduB(x, y)  = (exp(x) * cos(y), -exp(x) * sin(y))

# =============================================================================
#  Convergence driver
# =============================================================================
#
#  One mesh family, one exact solution, one polynomial degree -> one table.
#  `g = nothing` means homogeneous Dirichlet data; passing a function switches
#  `solve_poisson` onto the lifting path.

struct Study
    label::String
    p::Int
    ns::Vector{Int}
    ndofs::Vector{Int}
    hmaxs::Vector{Float64}
    l2::Vector{Float64}
    h1::Vector{Float64}
    l2ord::Vector{Float64}
    h1ord::Vector{Float64}
    sols::Vector{Vector{Float64}}   # kept so we can draw a picture afterwards
    meshes::Vector{Mesh2D}
end

function run_study(label::String, family::String, ns::Vector{Int},
                   f::Function, u::Function, gradu::Function, p::Integer;
                   g = nothing, err_rule = tri_quad_rule(8))
    ndofs = Int[]; hmaxs = Float64[]; l2 = Float64[]; h1 = Float64[]
    sols = Vector{Vector{Float64}}(); meshes = Mesh2D[]

    @printf("\n  %s   (Lagrange P%d)\n", label, p)
    @printf("  %5s %7s %9s   %12s %6s   %12s %6s   %7s\n",
            "n", "ndof", "hmax", "L2 error", "order", "H1 error", "order", "sec")
    println("  " * "-"^75)

    for n in ns
        folder = joinpath(MESHROOT, "$(family)_$(n)")
        m = mesh2d_load(folder)

        # hmax: note that `find_mesh_hmax` has NO Mesh2D overload -- it takes
        # the two arrays.  `mesh_hmax` is the tutorial's one-line wrapper.
        h = mesh_hmax(m)

        t0 = time()
        sol = solve_poisson(m, f, p; g = g)
        el2 = l2_error(m, sol.uh, u, p; rule = err_rule)
        eh1 = h1_seminorm_error(m, sol.uh, gradu, p; rule = err_rule)
        dt = time() - t0

        push!(ndofs, length(sol.uh)); push!(hmaxs, h)
        push!(l2, el2); push!(h1, eh1)
        push!(sols, sol.uh); push!(meshes, m)

        o2 = length(l2) > 1 ? observed_order(l2[end-1:end], hmaxs[end-1:end])[1] : NaN
        o1 = length(h1) > 1 ? observed_order(h1[end-1:end], hmaxs[end-1:end])[1] : NaN
        @printf("  %5d %7d %9.3e   %12.5e %6s   %12.5e %6s   %7.2f\n",
                n, ndofs[end], h, el2,
                isnan(o2) ? "  -  " : @sprintf("%5.3f", o2), eh1,
                isnan(o1) ? "  -  " : @sprintf("%5.3f", o1), dt)
        flush(stdout)
    end

    return Study(label, Int(p), ns, ndofs, hmaxs, l2, h1,
                 observed_order(l2, hmaxs), observed_order(h1, hmaxs),
                 sols, meshes)
end

# =============================================================================
#  Main
# =============================================================================

println("="^79)
println(" VFEM.jl tutorial 01 : Poisson on the unit square (0,1)^2")
println("="^79)
println(" mesh root : ", MESHROOT)
println(" levels    : unit_square_", join(LEVELS, ", unit_square_"))
println(" Julia     : ", VERSION)

# ---------------------------------------------------------------------------
#  Sanity gate.  `verify_dof_ordering` reassembles A and M from scratch using
#  the tutorial's own local-to-global map and compares with the library's.
#  If this does not agree to ~1e-15, nothing below means anything, so it runs
#  first and it runs every time.
# ---------------------------------------------------------------------------
println("\n--- DOF-ordering check against VFEM.lagrange_laplace_matrices ---")
let m = mesh2d_load(joinpath(MESHROOT, "unit_square_8"))
    println("  mesh: ", m, "   hmax = ", mesh_hmax(m))
    for p in (1, 2)
        v = verify_dof_ordering(m, p)
        @printf("  p=%d  ndof=%5d   rel|A-A_lib| = %.2e   rel|M-M_lib| = %.2e   ok=%s\n",
                p, v.ndof, v.relA, v.relM, v.ok)
        v.ok || error("DOF ordering mismatch at p=$p -- refusing to continue")
    end
end

# ---------------------------------------------------------------------------
#  Part 0 — hand-written pipeline vs. the helper.
# ---------------------------------------------------------------------------
println("\n--- Part 0: hand-written pipeline == solve_poisson ---")
let m = mesh2d_load(joinpath(MESHROOT, "unit_square_16"))
    for p in (1, 2)
        u_hand = poisson_by_hand(m, fA, p)
        u_help = solve_poisson(m, fA, p).uh
        rel = norm(u_hand - u_help) / norm(u_help)
        @printf("  p=%d  ndof=%5d   ||u_hand - u_helper|| / ||u_helper|| = %.3e\n",
                p, length(u_hand), rel)
        rel < 1e-12 || error("hand-written pipeline disagrees with the helper")
    end
    println("  => solve_poisson is exactly the six steps above, nothing more.")
end

# ---------------------------------------------------------------------------
#  Part A — homogeneous Dirichlet, manufactured u = sin(pi x) sin(pi y).
# ---------------------------------------------------------------------------
println("\n" * "="^79)
println(" PART A  homogeneous Dirichlet:  u = sin(pi x) sin(pi y),  f = 2 pi^2 u")
println("="^79)
println(" expected: L2 order p+1, H1 seminorm order p  (domain convex, u analytic)")

A1 = run_study("A: u = sin(pi x) sin(pi y), g = 0", "unit_square", LEVELS,
               fA, uA, graduA, 1)
A2 = run_study("A: u = sin(pi x) sin(pi y), g = 0", "unit_square", LEVELS,
               fA, uA, graduA, 2)

# ---------------------------------------------------------------------------
#  Part B — non-homogeneous Dirichlet, harmonic u = exp(x) cos(y), f = 0.
# ---------------------------------------------------------------------------
println("\n" * "="^79)
println(" PART B  non-homogeneous Dirichlet:  u = exp(x) cos(y),  f = 0,  g = u")
println("="^79)
println(" the entire solution is driven by the boundary lift; f is identically 0")

B1 = run_study("B: u = exp(x) cos(y), g = u|bd", "unit_square", LEVELS,
               fB, uB, graduB, 1; g = uB)
B2 = run_study("B: u = exp(x) cos(y), g = u|bd", "unit_square", LEVELS,
               fB, uB, graduB, 2; g = uB)

# ---------------------------------------------------------------------------
#  CSV export
# ---------------------------------------------------------------------------
# One row per (case, degree, level).  `ndof_scale = ndof^(-1/2)` is the
# dimensionless mesh parameter that lets you compare mesh families with
# different hmax at equal cost; on a uniform family it is proportional to hmax,
# so the two order columns agree here.  On a graded family (chapter 2) they do
# not, and ndof_scale is the honest one.
function write_csv(path::AbstractString, rows::Vector)
    open(path, "w") do io
        println(io, "case,degree,level,n,ndof,hmax,ndof_scale,",
                    "l2_err,l2_order,l2_order_ndof,h1_err,h1_order,h1_order_ndof")
        for (case, s) in rows
            xs = [nd^(-0.5) for nd in s.ndofs]
            l2n = observed_order(s.l2, xs)
            h1n = observed_order(s.h1, xs)
            for i in eachindex(s.ns)
                o2  = i > 1 ? s.l2ord[i-1] : NaN
                o1  = i > 1 ? s.h1ord[i-1] : NaN
                o2n = i > 1 ? l2n[i-1] : NaN
                o1n = i > 1 ? h1n[i-1] : NaN
                fmt(v) = isnan(v) ? "" : @sprintf("%.6f", v)
                @printf(io, "%s,%d,%d,%d,%d,%.10e,%.10e,%.10e,%s,%s,%.10e,%s,%s\n",
                        case, s.p, i, s.ns[i], s.ndofs[i], s.hmaxs[i], xs[i],
                        s.l2[i], fmt(o2), fmt(o2n),
                        s.h1[i], fmt(o1), fmt(o1n))
            end
        end
    end
    println("  wrote ", path)
end

println("\n--- exports ---")
write_csv(joinpath(OUTDIR, "bvp_square.csv"),
          [("homogeneous_sin_sin", A1), ("homogeneous_sin_sin", A2),
           ("nonhomogeneous_expcos", B1), ("nonhomogeneous_expcos", B2)])

# ---------------------------------------------------------------------------
#  Figures — hand-written SVG, no plotting package anywhere in the stack.
# ---------------------------------------------------------------------------
# Solution colour maps on the finest-but-still-legible mesh (n = 32).
let i = findfirst(==(32), LEVELS)
    svg_solution(joinpath(OUTDIR, "fig_square_solution_p2.svg"),
                 A2.meshes[i], A2.sols[i]; p = 2,
                 title = "u_h,  P2,  -Delta u = 2pi^2 sin(pi x) sin(pi y),  u|bd = 0")
    svg_solution(joinpath(OUTDIR, "fig_square_solution_nonhom_p2.svg"),
                 B2.meshes[i], B2.sols[i]; p = 2,
                 title = "u_h,  P2,  harmonic u = exp(x) cos(y),  non-zero u|bd")
end
# The mesh itself, coarse enough to see the individual triangles.
let i = findfirst(==(8), LEVELS)
    svg_mesh(joinpath(OUTDIR, "fig_square_mesh.svg"), A1.meshes[i];
             title = "unit_square_8  (boundary edges in orange)")
end

# Convergence: four curves, with the theoretical slopes drawn as triangles.
svg_loglog(joinpath(OUTDIR, "fig_conv_square.svg"),
           [(x = A1.hmaxs, y = A1.l2, label = "P1  L2",         slope = 2.0),
            (x = A1.hmaxs, y = A1.h1, label = "P1  H1 seminorm", slope = 1.0),
            (x = A2.hmaxs, y = A2.l2, label = "P2  L2",         slope = 3.0),
            (x = A2.hmaxs, y = A2.h1, label = "P2  H1 seminorm", slope = 2.0)];
           xlabel = "h_max", ylabel = "error",
           title = "Unit square, homogeneous Dirichlet: optimal orders")

svg_loglog(joinpath(OUTDIR, "fig_conv_square_nonhom.svg"),
           [(x = B1.hmaxs, y = B1.l2, label = "P1  L2",         slope = 2.0),
            (x = B1.hmaxs, y = B1.h1, label = "P1  H1 seminorm", slope = 1.0),
            (x = B2.hmaxs, y = B2.l2, label = "P2  L2",         slope = 3.0),
            (x = B2.hmaxs, y = B2.h1, label = "P2  H1 seminorm", slope = 2.0)];
           xlabel = "h_max", ylabel = "error",
           title = "Unit square, non-homogeneous Dirichlet (harmonic u)")

for fn in ("fig_square_solution_p2.svg", "fig_square_solution_nonhom_p2.svg",
           "fig_square_mesh.svg", "fig_conv_square.svg",
           "fig_conv_square_nonhom.svg")
    println("  wrote ", joinpath(OUTDIR, fn))
end

println("\n" * "="^79)
@printf(" SUMMARY  final observed orders (finest pair of levels)\n")
for (nm, s, want2, want1) in (("A P1", A1, 2, 1), ("A P2", A2, 3, 2),
                              ("B P1", B1, 2, 1), ("B P2", B2, 3, 2))
    @printf("   %-6s  L2 %.3f (expect %d)   H1 %.3f (expect %d)\n",
            nm, s.l2ord[end], want2, s.h1ord[end], want1)
end
println("="^79)
println("01_poisson_square.jl DONE")
