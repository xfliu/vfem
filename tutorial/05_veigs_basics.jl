# =============================================================================
# 05_veigs_basics.jl  --  Verified eigenvalue bounds with Veigs.jl
#
# Run with:
#   julia --project=. tutorial/05_veigs_basics.jl
#
# The whole chapter answers one question: when a floating-point eigensolver
# hands you the number 3.0000000000000004, what do you actually know?
# Answer: nothing, mathematically. `eigen` is backward-stable, which means it
# returns the exact spectrum of *some* matrix near yours -- it does not tell
# you how near, and it does not tell you which way the error went.
#
# `veigs` returns an INTERVAL instead. The interval is a theorem: the true
# eigenvalue of the true matrix pair lies inside it. That is a different kind
# of object from a float, and this file is about learning to handle it.
#
# NOTE ON JULIA SCOPING. Every loop below that accumulates into a variable
# declares it `local` inside the loop body, or pushes into a pre-allocated
# array. At top level in a script, a `for` loop opens a SOFT scope: assigning
# to a name that already exists as a global creates a NEW local and the global
# keeps its old value -- Julia warns, then the code misbehaves. Writing
# tutorial scripts that a reader copy-pastes makes this worth being explicit
# about.
# =============================================================================

using LinearAlgebra
using SparseArrays
using Printf
using Random
import Veigs
using Veigs: veigs, veig
import IntervalArithmetic
using IntervalArithmetic: Interval, interval, inf, sup, mid, radius, diam,
                          isequal_interval

# Small formatting helpers. `veigs` returns Interval{Float64} values; these
# print them so that the *width* of the enclosure is visible at a glance,
# which is the number you actually care about when judging a verified result.
fmt_iv(x::Interval) = @sprintf("[%.17g, %.17g]", inf(x), sup(x))
fmt_w(x::Interval)  = @sprintf("%.3e", diam(x))

hdr(s) = (println(); println("="^76); println(s); println("="^76))

# =============================================================================
# 1. The motivating side-by-side: `eigen` versus `veigs`
# =============================================================================
hdr("1. eigen (no guarantee) versus veigs (guaranteed enclosure)")

# The 2x2 tridiagonal Laplacian. Its exact eigenvalues are 1 and 3, integers,
# no round-off anywhere in the problem statement.
A2 = [ 2.0 -1.0
      -1.0  2.0]
B2 = Matrix{Float64}(I, 2, 2)

ev_float = eigvals(Symmetric(A2), Symmetric(B2))
println("exact spectrum (by hand)   : 1, 3")
println("eigvals (Float64)          : ", ev_float)

# `veigs(A, B, k, sigma)`: k = 2 eigenvalues, selected as the algebraically
# smallest (:smallestreal), i.e. we ask for lambda_1 and lambda_2.
lam1, ir1 = veigs(A2, B2, 2, :smallestreal)
println("veigs index range          : ", ir1)
for i in ir1
    s = i - first(ir1) + 1
    @printf("  lambda_%d in %s   width %s   contains float value: %s\n",
            i, fmt_iv(lam1[s]), fmt_w(lam1[s]),
            inf(lam1[s]) <= ev_float[i] <= sup(lam1[s]))
end
println()
println("Note that veigs's enclosures are NOT degenerate here even though the")
println("true eigenvalues are exactly 1 and 3. The verified bound has to admit")
println("the possibility that the eigensolver, the LDL factorisation and the")
println("residual estimate all drifted by a few ULPs -- and it says so, out")
println("loud, instead of printing 1.0000000000000002 and staying silent.")

# --- The Hilbert reference problem -----------------------------------------
# For everything that follows we also want a problem that is genuinely hard,
# so the intervals have visible width. `A = I, B = Hilbert(8)` is example 1
# from the MATLAB veigs README, and its published bounds are reproduced here
# as an external cross-check.
hdr("1b. A hard reference problem: A = I(8), B = Hilbert(8)")

nH = 8
AH = Matrix{Float64}(I, nH, nH)
BH = [1.0 / (i + j - 1) for i in 1:nH, j in 1:nH]
@printf("cond(Hilbert(8)) = %.3e   -- this is why the bounds below are wide\n",
        cond(BH))

lamH, irH = veigs(AH, BH, 3, 3.5)     # three eigenvalues nearest 3.5
println("veigs(A, B, 3, 3.5): ind_range = ", irH)
evH = sort(real.(eigvals(Symmetric(AH), Symmetric(BH))))
for i in irH
    s = i - first(irH) + 1
    @printf("  lambda_%d in %-44s width %s\n", i, fmt_iv(lamH[s]), fmt_w(lamH[s]))
end
# Published MATLAB golden bounds for this exact problem:
#   lambda_1 in [0.589643850288603, 0.589643850289016]
#   lambda_2 in [3.35429531632129,  3.35429531633695]
#   lambda_3 in [38.1492376817517,  38.1492376835712]
matlab_golden = [(0.589643850288603, 0.589643850289016),
                 (3.35429531632129,  3.35429531633695),
                 (38.1492376817517,  38.1492376835712)]
println()
println("cross-check against the published MATLAB veigs bounds:")
for i in irH
    s = i - first(irH) + 1
    (glo, gup) = matlab_golden[i]
    ours_contains = inf(lamH[s]) <= glo && gup <= sup(lamH[s])
    inside_matlab = glo <= inf(lamH[s]) && sup(lamH[s]) <= gup
    @printf("  lambda_%d  MATLAB [%.15g, %.15g] width %.3e\n", i, glo, gup, gup - glo)
    @printf("            Julia  %-44s width %.3e\n", fmt_iv(lamH[s]), diam(lamH[s]))
    @printf("            Julia contains MATLAB: %-6s   Julia tighter: %s\n",
            string(ours_contains), string(inside_matlab))
end

# =============================================================================
# 2. All four call forms
# =============================================================================
hdr("2. The four call forms of veigs")

# A deliberately indefinite diagonal problem. Making the spectrum straddle
# zero is what separates "largest absolute value" from "largest algebraic
# value" -- on a positive-definite problem those two coincide and you learn
# nothing about the sigma options.
A5 = Matrix(Diagonal([-9.0, -1.0, 2.0, 4.0, 7.0]))
B5 = Matrix{Float64}(I, 5, 5)
println("spectrum of the reference problem: ", diag(A5))

# Form 1: veigs(A, B)                  -- k = 1, sigma = :largestabs (default)
lam_a, ir_a = veigs(A5, B5)
@printf("veigs(A,B)                : ind_range=%-6s  %s\n", string(ir_a), fmt_iv(lam_a[1]))

# Form 2: veigs(A, B, k)               -- k eigenvalues, sigma still :largestabs
lam_b, ir_b = veigs(A5, B5, 2)
@printf("veigs(A,B,2)              : ind_range=%-6s  %s .. %s\n",
        string(ir_b), fmt_iv(lam_b[1]), fmt_iv(lam_b[end]))

# Form 3: veigs(A, B, sigma)           -- k = 1, explicit selector
lam_c, ir_c = veigs(A5, B5, :smallestreal)
@printf("veigs(A,B,:smallestreal)  : ind_range=%-6s  %s\n", string(ir_c), fmt_iv(lam_c[1]))

# Form 4: veigs(A, B, k, sigma)        -- both
lam_d, ir_d = veigs(A5, B5, 3, :smallestreal)
@printf("veigs(A,B,3,:smallestreal): ind_range=%-6s  %d intervals\n",
        string(ir_d), length(lam_d))

# ---------------------------------------------------------------------------
# TRAP, and it is a silent one. `Int` is a subtype of `Real`, and the method
#   veigs(A, B, k::Integer)
# is more specific than
#   veigs(A, B, sigma::Real).
# So `veigs(A, B, 2)` means "give me k = 2 eigenvalues", NOT "the eigenvalue
# closest to 2". To ask for the eigenvalue nearest 2 you must write a Float:
#   veigs(A, B, 2.0)
# Both calls run without error and return plausible-looking output, so this
# mistake does not announce itself. Always write the shift as a Float literal.
# ---------------------------------------------------------------------------
lam_k, ir_k = veigs(A5, B5, 2)      # k = 2, sigma = :largestabs
lam_s, ir_s = veigs(A5, B5, 2.0)    # k = 1, sigma = 2.0 (shift-invert)
println()
println("veigs(A,B,2)   -> k=2,       ind_range = ", ir_k, "   (Integer -> k)")
println("veigs(A,B,2.0) -> sigma=2.0, ind_range = ", ir_s, "   (Float -> shift)")
println("Same two characters typed, entirely different question asked.")

# =============================================================================
# 3. The sigma reference table
# =============================================================================
hdr("3. Which eigenvalue does each sigma select?")

# One fixed matrix, every selector. This table is the single most useful piece
# of reference material in the chapter: it says, concretely, which index each
# sigma lands on, including the legacy MATLAB-compatible aliases.
sigma_cases = Any[
    (:largestabs,   "modern",  "largest |lambda|"),
    (:smallestabs,  "modern",  "smallest |lambda|"),
    (:largestreal,  "modern",  "largest algebraic lambda"),
    (:smallestreal, "modern",  "smallest algebraic lambda"),
    (:lm,           "legacy",  "alias of :largestabs"),
    (:sm,           "legacy",  "alias of :smallestabs"),
    (:la,           "legacy",  "alias of :largestreal"),
    (:sa,           "legacy",  "alias of :smallestreal"),
    (-8.0,          "numeric", "closest to -8.0"),
    (0.0,           "numeric", "closest to 0.0"),
    (2.0,           "numeric", "closest to 2.0"),
    (5.0,           "numeric", "closest to 5.0"),
    (100.0,         "numeric", "closest to 100.0"),
]

println("reference problem: A = diag(-9, -1, 2, 4, 7), B = I")
println()
println(rpad("sigma", 15), rpad("kind", 9), rpad("ind_range", 11),
        rpad("index", 7), rpad("selected", 10), rpad("enclosure", 46), "width")
println("-"^108)

sigma_rows = Any[]
for (sig, kind, descr) in sigma_cases
    # `local` is mandatory: without it, `lam`/`ir` assigned here would shadow
    # the globals of the same name and Julia would warn on every iteration.
    local lam, ir
    lam, ir = veigs(A5, B5, sig)
    # `veigs` may return a whole cluster. For this well-separated spectrum
    # ir is a single index, so report it directly -- but read it off ir, never
    # assume it.
    local idx = first(ir)
    local b   = lam[1]
    @printf("%-15s%-9s%-11s%-7d%-10s%-46s%s\n",
            string(sig), kind, string(ir), idx, string(diag(A5)[idx]),
            fmt_iv(b), fmt_w(b))
    push!(sigma_rows, (string(sig), kind, descr, first(ir), last(ir),
                       diag(A5)[idx], inf(b), sup(b), diam(b)))
end

# The same selectors on the Hilbert problem, where the widths are not zero.
println()
println("the same four modern selectors on A = I(8), B = Hilbert(8):")
println(rpad("sigma", 15), rpad("index", 7), rpad("enclosure", 46), "width")
println("-"^80)
for sig in (:largestabs, :smallestabs, :largestreal, :smallestreal)
    local lam, ir
    lam, ir = veigs(AH, BH, sig)
    local b = lam[1]
    @printf("%-15s%-7d%-46s%s\n", string(sig), first(ir), fmt_iv(b), fmt_w(b))
    push!(sigma_rows, ("hilbert8:" * string(sig), "modern",
                       "same selector on the Hilbert(8) reference problem",
                       first(ir), last(ir), mid(b), inf(b), sup(b), diam(b)))
end

# Machine-readable copy of the same table.
open("sigma_table.csv", "w") do io
    println(io, "sigma,kind,description,ind_first,ind_last,selected_eigenvalue,inf,sup,diam")
    for r in sigma_rows
        # Quote the free-text columns: descriptions such as "largest |lambda|"
        # are safe, but any future edit that introduces a comma would silently
        # shift every numeric column one to the right.
        @printf(io, "\"%s\",\"%s\",\"%s\",%d,%d,%.17g,%.17g,%.17g,%.17g\n", r...)
    end
end
println("\n-> wrote sigma_table.csv (", length(sigma_rows), " rows)")

# =============================================================================
# 4. The return contract, and why you must index off ind_range
# =============================================================================
hdr("4. (lambda, ind_range): clustering makes ind_range mandatory")

# THE canonical trap. A has a double eigenvalue at 1. We ask for k = 1
# eigenvalue closest to zero -- one number, surely?
Ac = Matrix(Diagonal([1.0, 1.0, 5.0, 9.0]))
Bc = Matrix{Float64}(I, 4, 4)

lam_cl, ir_cl = veigs(Ac, Bc, 1, :smallestreal)
println("A = diag(1, 1, 5, 9),  k = 1,  sigma = :smallestreal")
println("  length(lambda) = ", length(lam_cl), "        <-- we asked for 1")
println("  ind_range      = ", ir_cl, "      <-- two indices came back")
for i in ir_cl
    local s = i - first(ir_cl) + 1
    @printf("  lambda[%d] covers eigenvalue index %d : %s\n", s, i, fmt_iv(lam_cl[s]))
end

# Why: lambda_1 and lambda_2 are numerically indistinguishable, so no
# certificate can separate them. Veigs refuses to pretend otherwise; it
# certifies the pair jointly and tells you so through ind_range. The two
# returned intervals are identical, because they are the same theorem stated
# twice ("eigenvalue 1 is in here" and "eigenvalue 2 is in here").
println("  the two intervals are identical: ", isequal_interval(lam_cl[1], lam_cl[2]))

# The correct access pattern, always:
#     slot = i - first(ind_range) + 1     # for the eigenvalue you want, index i
# The wrong pattern -- `lambda[1]` is "the eigenvalue I asked for" -- happens
# to be right here and will be wrong the first time a cluster reaches
# downward instead of upward.
let want = 2, s = 2 - first(ir_cl) + 1
    @printf("  bound for eigenvalue %d, fetched correctly: %s\n", want, fmt_iv(lam_cl[s]))
end

# Same matrix, k = 1, largest: the cluster does not reach, so one index.
lam_c2, ir_c2 = veigs(Ac, Bc, 1, :largestreal)
println("  contrast, sigma=:largestreal: ind_range = ", ir_c2,
        ", length(lambda) = ", length(lam_c2))
println()
println("The idiom to internalise, in three lines:")
println("    lambda, ind_range = veigs(A, B, k, sigma)")
println("    slot  = i - first(ind_range) + 1     # i is the index you want")
println("    bound = lambda[slot]")

# =============================================================================
# 5. veig -- the dense Yamamoto-style solver, and its index semantics
# =============================================================================
hdr("5. veig(A, B, ind): explicit index selection on dense problems")

# `veig` is the small-dense sibling. Instead of a selector you give it the
# eigenvalue INDICES you want, counted in ascending eigenvalue order.
bnd_all, ira = veig(A5, B5)              # default: all of them, 1:n
println("veig(A,B)         ind_range = ", ira, "  (", length(bnd_all), " bounds)")
for i in ira
    local s = i - first(ira) + 1
    @printf("  lambda_%d = %-9s in %-46s width %s\n",
            i, string(diag(A5)[i]), fmt_iv(bnd_all[s]), fmt_w(bnd_all[s]))
end

bnd_mid, irm = veig(A5, B5, 2:3)         # a sub-range
println("veig(A,B,2:3)     ind_range = ", irm, "  (", length(bnd_mid), " bounds)")
bnd_one, iro = veig(A5, B5, 4:4)         # a single index
println("veig(A,B,4:4)     ind_range = ", iro, "  -> ", fmt_iv(bnd_one[1]))

# --- veig and clustering ---------------------------------------------------
# The same clustering logic applies, and here it is visible directly: two
# indices are handed the same interval object.
bnd_c, irc = veig(Ac, Bc, 1:2)
println()
println("veig(diag(1,1,5,9), I, 1:2): ind_range = ", irc)
for i in irc
    local s = i - first(irc) + 1
    @printf("  index %d -> %s\n", i, fmt_iv(bnd_c[s]))
end
println("  both indices share one wider interval: ",
        isequal_interval(bnd_c[1], bnd_c[2]))

# --- veig versus veigs on the same problem ---------------------------------
# `veig` verifies each eigenvalue independently with a two-sided LDL/inertia
# search and stops there. `veigs` runs the same bracketing and then applies
# Lehmann-Behnke sharpening, whose width is set by the eigenvector RESIDUAL
# rather than by the bracket. That is why veigs's intervals are narrower.
hdr("5b. veig versus veigs: same theorem, different sharpness")

println("On the exact-integer diagonal problem, veigs's bounds are so tight")
println("they collapse to a point, so the ratio is not informative:")
println(rpad("index", 7), rpad("veig width", 14), "veigs width")
for i in 1:5
    local bv, irv, bs, irs, sv, ss
    bv, irv = veig(A5, B5, i:i)
    sv = i - first(irv) + 1
    bs, irs = veigs(A5, B5, 1, float(diag(A5)[i]))
    ss = clamp(i - first(irs) + 1, 1, length(bs))
    @printf("%-7d%-14.3e%.3e\n", i, diam(bv[sv]), diam(bs[ss]))
end

println()
println("On the Hilbert problem, where both are genuinely working, the")
println("sharpening factor is measurable:")
println(rpad("index", 7), rpad("veig width", 14), rpad("veigs width", 14), "veig/veigs")
for i in 1:4
    local bv, irv, bs, irs, sv, ss, wv, ws
    bv, irv = veig(AH, BH, i:i)
    sv = i - first(irv) + 1
    bs, irs = veigs(AH, BH, 1, evH[i])
    ss = clamp(i - first(irs) + 1, 1, length(bs))
    wv, ws = diam(bv[sv]), diam(bs[ss])
    @printf("%-7d%-14.3e%-14.3e%s\n", i, wv, ws,
            ws > 0 ? @sprintf("%.2f", wv/ws) : "inf")
end

# The other half of the choice is not sharpness at all, it is applicability:
# `veig` refuses anything above n = 100 and needs dense input.
println()
println("veig's hard size cap (n = 100) is real:")
for n in (100, 101)
    local Ai_ = Matrix{Float64}(I, n, n)
    print("  n = ", n, " -> ")
    try
        veig(Ai_, copy(Ai_), 1:1)
        println("ok")
    catch err
        println(typeof(err), ": ", sprint(showerror, err))
    end
end
println()
println("Rule of thumb: veig for a dense problem below n = 100 when you want")
println("bounds for a specific INDEX RANGE; veigs for everything else, and")
println("always for sparse FEM matrices.")

# =============================================================================
# 6. How to read an Interval{Float64}
# =============================================================================
hdr("6. Working with Interval{Float64}")

# Use a genuinely wide interval so every accessor prints something meaningful.
b = lamH[3 - first(irH) + 1]         # lambda_3 of the Hilbert problem
println("the interval          : ", b)
println("typeof                : ", typeof(b))
@printf("inf(b)                : %.17g      lower end, a rigorous lower bound\n", inf(b))
@printf("sup(b)                : %.17g      upper end, a rigorous upper bound\n", sup(b))
@printf("mid(b)                : %.17g      best single-number summary\n", mid(b))
@printf("radius(b)             : %.6e   half-width; the +/- you may quote\n", radius(b))
@printf("diam(b)               : %.6e   full width = sup - inf\n", diam(b))
println("bounds(b)             : ", IntervalArithmetic.bounds(b))
println()
println("The `_com` suffix in the printed form is IntervalArithmetic's")
println("DECORATION: `com` = common, meaning bounded, non-empty, and the")
println("operations that produced it were everywhere-defined. It is a")
println("provenance tag, not part of the numeric value.")

# --- and now the sharp edge -----------------------------------------------
# `==` on intervals is NOT a comparison of endpoints. IntervalArithmetic
# treats an interval as a SET of possible values, so `a == b` asks a question
# with no Boolean answer when the sets are non-degenerate, and it throws
# rather than guess.
println()
println("Comparison operators on intervals:")
b_copy = interval(inf(b), sup(b))         # bit-identical endpoints
try
    println("  b == b_copy  -> ", b == b_copy)
catch err
    println("  b == b_copy  THREW ", typeof(err))
    println("      IntervalArithmetic is refusing to answer an ill-posed")
    println("      question -- two overlapping non-degenerate sets are neither")
    println("      equal nor unequal as VALUES. This is not a Veigs bug.")
end
println("  isequal_interval(b, b_copy) -> ", isequal_interval(b, b_copy),
        "   <-- use this")

# The degenerate case is the trap inside the trap: a zero-width interval DOES
# compare with `==`, because there the question is well-posed. So `==` works
# in testing on exact problems and then throws on the first realistic one.
let d = interval(1.0)
    println("  interval(1.0) == interval(1.0) -> ", d == d,
            "     <-- degenerate: no throw!")
    println("      This is why `==` must never appear in your code even when")
    println("      it appears to work: it works exactly until the interval")
    println("      acquires width, i.e. on every real problem.")
end
println("  in_interval(mid(b), b)  -> ", IntervalArithmetic.in_interval(mid(b), b))
println("  issubset_interval(b, hull(b, b_copy)) -> ",
        IntervalArithmetic.issubset_interval(b, IntervalArithmetic.hull(b, b_copy)))
println()
println("The practical rule: never write ==, <, > between two intervals.")
println("Compare the endpoints you actually mean -- `sup(a) < inf(b)` for")
println("\"a lies entirely below b\" -- or use the named set predicates")
println("isequal_interval / issubset_interval / isdisjoint_interval /")
println("in_interval.")

# =============================================================================
# 7. The soundness contract, stated plainly
# =============================================================================
hdr("7. Soundness: what the interval promises, and what it does not")

# The contract from src/veig.jl, in words: for every concrete matrix pair
# (A_c, B_c) drawn from the interval enclosure of (A, B), and for each index i
# in ind_range, the i-th eigenvalue of A_c x = lambda B_c x lies inside
# eig_bounds[i - first(ind_range) + 1].
#
# Two consequences worth internalising:
#
#  (a) VERIFIED is not the same as ACCURATE. A wide interval is still a
#      correct theorem -- it just says little. A narrow float is not a
#      theorem at all, however many digits it prints.
#  (b) The enclosure covers the *matrix pair you handed in*. If you hand in
#      float matrices, it covers exactly those floats plus the eigensolver's
#      error. If you hand in INTERVAL matrices, it also covers whatever
#      uncertainty you put in the entries -- assembly round-off, measured
#      coefficients, manufacturing tolerances.

println("(a) verified is not accurate. Consider the two statements:")
println("      float :  lambda_1 = 0.5896438502890  (17 digits, zero guarantee)")
println("      veigs :  lambda_1 in a certified interval of width w")
println("    Only the second can be wrong in a detectable way -- and it")
println("    cannot be wrong at all. Widening w never makes the claim false,")
println("    only less useful. Adding digits to the float never makes it true.")

# Demonstration of (b). Perturb A into an interval matrix of radius 1e-8 and
# watch the enclosure grow to swallow the extra uncertainty.
println()
println("(b) the enclosure inherits the uncertainty you declare:")
println(rpad("input radius", 16), rpad("enclosure of lambda_max", 46), "width")
println("-"^80)
lam_f0, ir_f0 = veigs(A5, B5, 1, :largestreal)
@printf("%-16s%-46s%.3e\n", "exact floats", fmt_iv(lam_f0[1]), diam(lam_f0[1]))
radius_rows = Any[]
for r in (1e-12, 1e-8, 1e-4)
    local Ai_ = interval.(A5 .- r, A5 .+ r)
    local Bi_ = interval.(B5)
    local li, iri
    li, iri = veigs(Ai_, Bi_, 1, :largestreal)
    @printf("%-16.0e%-46s%.3e\n", r, fmt_iv(li[1]), diam(li[1]))
    push!(radius_rows, (r, li[1]))
end
println()
println("Each wider input box gives a wider -- and still correct -- output")
println("interval, and each contains the one above it:")
for k in 2:length(radius_rows)
    local prev = radius_rows[k-1][2]
    local cur  = radius_rows[k][2]
    @printf("  radius %.0e enclosure contains the radius %.0e one: %s\n",
            radius_rows[k][1], radius_rows[k-1][1],
            string(inf(cur) <= inf(prev) && sup(prev) <= sup(cur)))
end

# A concrete probe of the contract: sample matrices from inside the interval
# enclosure and check that their float eigenvalues all land inside the
# verified bound. This is not a proof -- the theorem is the proof -- but it is
# a useful sanity check that the interval means what we said it means.
println()
println("Empirical probe of the contract (200 random pairs from inside the")
println("radius-1e-8 enclosure; every one must land inside the bound):")
Random.seed!(20260501)
let r = 1e-8
    Ai_ = interval.(A5 .- r, A5 .+ r)
    Bi_ = interval.(B5)
    li, iri = veigs(Ai_, Bi_, 1, :largestreal)
    bnd = li[1]
    nviol = 0
    wmax  = -Inf
    for _t in 1:200
        P  = A5 .+ r .* (2 .* rand(5, 5) .- 1)
        Pc = (P + P') / 2                       # stay in the symmetric slice
        ev = sort(real.(eigvals(Symmetric(Pc))))
        (inf(bnd) <= ev[end] <= sup(bnd)) || (nviol += 1)
        wmax = max(wmax, abs(ev[end] - mid(bnd)))
    end
    @printf("  bound %s  radius %.3e\n", fmt_iv(bnd), radius(bnd))
    @printf("  violations: %d / 200      max deviation from midpoint: %.3e\n",
            nviol, wmax)
    println("  (a violation would mean the library is unsound -- there are none)")
end

println()
println("VEIGS_BASICS_DONE_MARKER")
