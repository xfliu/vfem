# examples/laplace_equilateral_p5_lg_verified.jl
#
# Fully verified two-sided bound on the first Dirichlet Laplace
# eigenvalue λ_1 of the equilateral triangle of side 1, using:
#
#   * Liu CR lower bound for λ_2 → LG shift ρ      (verified_cr_liu_lower)
#   * P_p Lagrange stiffness / mass in intervals   (lagrange_laplace_matrices)
#   * Verified RT saddle solve                     (verified_rt_hdiv_problem)
#   * Verified LG eigensolve + transform           (verified_lg_transform)
#
# The Float64 P_p Galerkin computes the upper bound λ_h and the
# approximate eigenvectors that we project onto.  The matrices AL, BL
# fed into the LG step are built rigorously from interval P_p mass /
# stiffness and a verified Krawczyk-style RT saddle solve, so the
# resulting LG enclosure is a certified lower bound on λ_1.

using Printf
using IntervalArithmetic: Interval, interval, inf, sup, mid

const _PKG_ROOT = abspath(joinpath(@__DIR__, ".."))
import Pkg
Pkg.activate(_PKG_ROOT; io = devnull)

using VFEM
import Veigs
include(joinpath(@__DIR__, "laplace_equilateral_triangle.jl"))

const REFINEMENT_LEVEL = 3
const LAGRANGE_ORDER   = 5
const RT_ORDER         = 5
const NEIG             = 1

m, _ = build_equilateral_mesh(REFINEMENT_LEVEL)
h    = find_mesh_hmax(m.nodes, m.edges)
λex  = exact_equilateral_dirichlet_eigenvalues(NEIG + 1)

println("Verified two-sided bound on λ_1 — equilateral triangle, side 1")
println(repeat("=", 78))
@printf("Mesh: L%d   h_max = %.6f   nv = %d   nt = %d   ne = %d\n",
        REFINEMENT_LEVEL, h, m.nv, m.nt, m.ne)
@printf("Exact:  λ_1 = %.10f   λ_2 = %.10f\n", λex[1], λex[2])
println()

# --- Step 1: rigorous ρ from verified Liu CR. --------------------------------
println("[1] Rigorous Liu CR lower bounds (Veigs + interval CR matrices)...")
@time cr_low_int, Ch_cr_int = verified_cr_liu_lower(m, NEIG)

ρ_int = cr_low_int[NEIG + 1]
@printf("    Ch_cr  = [%.10e, %.10e]\n", inf(Ch_cr_int), sup(Ch_cr_int))
for k in 1:length(cr_low_int)
    @printf("    Liu λ_%d enclosure = [%.10f, %.10f]   inf-err = %+.3e\n",
            k, inf(cr_low_int[k]), sup(cr_low_int[k]),
            inf(cr_low_int[k]) - λex[min(k, length(λex))])
end
@printf("    → ρ = inf(Liu λ_2) = %.10f\n", inf(ρ_int))
@printf("    Check sup(ρ) < λ_2 = %.4f  →  %s\n",
        λex[2], sup(ρ_int) < λex[2] ? "YES" : "NO")
@printf("    Check inf(ρ) > λ_1 = %.4f  →  %s\n",
        λex[1], inf(ρ_int) > λex[1] ? "YES" : "NO")
println()

# --- Step 2: P_p Galerkin upper bound + approximate eigvecs (Float64). -------
println("[2] Float64 P$LAGRANGE_ORDER Galerkin (upper bound + eigenvector projector)...")
@time r_cg = laplace_eig_lagrange(m, LAGRANGE_ORDER, NEIG)
λ_h     = r_cg.eig_value[1]
LA_eigf = r_cg.eig_func
println()

# --- Step 3: interval P_p stiffness / mass (verified assembly). --------------
println("[3] Interval-mode P$LAGRANGE_ORDER stiffness / mass assembly...")
@time A_int, M_int, bd_dofs = lagrange_laplace_matrices(m, LAGRANGE_ORDER;
                                                          T = Interval{Float64})
A_proj_int = LA_eigf' * A_int * LA_eigf
M_proj_int = LA_eigf' * M_int * LA_eigf
println()

# --- Step 3b: verified Galerkin upper bound via Veigs on (A_int, M_int). ----
println("[3b] Verified P$LAGRANGE_ORDER Galerkin (Veigs.veigs on interval matrices)...")
ndof = size(A_int, 1)
is_bd = falses(ndof); is_bd[bd_dofs] .= true
int_dof = findall(!, is_bd)
A_red_int = Matrix(A_int[int_dof, int_dof])
M_red_int = Matrix(M_int[int_dof, int_dof])
@time disc_eig_int, _ = Veigs.veigs(A_red_int, M_red_int, NEIG, :sm)
λ_h_int = disc_eig_int[1]
@printf("    Verified discrete λ_h ∈ [%.13f, %.13f]   width = %.3e\n",
        inf(λ_h_int), sup(λ_h_int), sup(λ_h_int) - inf(λ_h_int))
println()

# --- Step 4: verified RT saddle solve (Krawczyk-residual + dense R). ---------
println("[4] Verified RT_$RT_ORDER saddle (interval assembly + verified solve)...")
LA_eigf_int = interval.(LA_eigf)
@time A_lg_int = verified_rt_hdiv_problem(m, RT_ORDER, LA_eigf_int)
println()

# --- Step 5: assemble AL, BL in intervals and run verified LG transform. ----
println("[5] Verified LG eigensolve (Veigs.veig + interval LG transform)...")
two_int = interval(2.0)
AL_int = A_proj_int .- ρ_int    .* M_proj_int
BL_int = A_proj_int .- two_int * ρ_int .* M_proj_int .+ ρ_int^2 .* A_lg_int
@time eig_lo_int = verified_lg_transform(AL_int, BL_int, ρ_int)
λ_lo_int = eig_lo_int[1]

println()
println("Verified bound on λ_1:")
println(repeat("-", 78))
@printf("  Verified Galerkin upper      λ_h  ∈ [%.13f, %.13f]\n",
        inf(λ_h_int), sup(λ_h_int))
@printf("                                      sup-err = %+.3e   width = %.3e\n",
        sup(λ_h_int) - λex[1], sup(λ_h_int) - inf(λ_h_int))
@printf("  Verified LG lower            λ_lo ∈ [%.13f, %.13f]\n",
        inf(λ_lo_int), sup(λ_lo_int))
@printf("                                      inf-err = %+.3e   width = %.3e\n",
        inf(λ_lo_int) - λex[1], sup(λ_lo_int) - inf(λ_lo_int))
@printf("  Verified Liu CR (reference)        ∈ [%.13f, %.13f]\n",
        inf(cr_low_int[1]), sup(cr_low_int[1]))
println()
@printf("  Certified bracket  [inf(λ_lo), sup(λ_h)] = [%.13f, %.13f]\n",
        inf(λ_lo_int), sup(λ_h_int))
@printf("    width = %.3e   (relative %.3e)\n",
        sup(λ_h_int) - inf(λ_lo_int),
        (sup(λ_h_int) - inf(λ_lo_int)) / λex[1])
println()
println("Every step is interval-rigorous: ρ from verified Liu CR, P5 stiffness/mass")
println("from interval assembly, λ_h from Veigs on the interval Galerkin pair, RT")
println("saddle solve via Krawczyk-residual + dense R, LG eigensolve via Veigs.veig.")
println("The bracket above is a certified two-sided enclosure of λ_1.")
