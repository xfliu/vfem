# =============================================================================
# 06_veigs_internals.jl  --  opening the box: what veigs assembles internally
#
# Run with:
#   julia --project=. tutorial/06_veigs_internals.jl
#
# `veigs` is not a monolith. It is five verified primitives stacked in a fixed
# order, and every one of them is exported and callable on its own. Reading
# this file you should come away able to answer: where does the certificate
# actually come from?
#
# The stack, bottom to top:
#
#   inertia          count how many eigenvalues sit below a test point
#   verified_ldl     factor mid(M) and bound the residual M - PLDL'P' rigorously
#   verified_isspd   decide "is every matrix in this enclosure positive definite?"
#   rough_lower      bracket the whole spectrum from below
#   rough_upper      bracket the whole spectrum from above
#   lehmann_behnke   sharpen a bracketed cluster into a tight two-sided bound
#
# Everything rigorous in Veigs traces back to one idea: you cannot compute an
# eigenvalue exactly, but you CAN certify the sign of a determinant, and signs
# are enough to count.
# =============================================================================

using LinearAlgebra
using SparseArrays
using Printf
import Veigs
using Veigs: veig, veigs, verified_ldl, verified_isspd, rough_lower,
             rough_upper, lehmann_behnke, sym_hull,
             VeigsError, VeigsSizeError, VeigsLDLFailureError,
             VeigsClusterTooLargeError, VeigsLehmannBehnkeError,
             VeigsRoughBoundError
import IntervalArithmetic
using IntervalArithmetic: Interval, interval, inf, sup, mid, radius, diam,
                          mag, hull, isequal_interval

fmt_iv(x::Interval) = @sprintf("[%.17g, %.17g]", inf(x), sup(x))
hdr(s) = (println(); println("="^76); println(s); println("="^76))

# =============================================================================
# 1. inertia -- the engine underneath everything
# =============================================================================
hdr("1. Veigs.inertia: counting eigenvalues by counting signs")

# ---------------------------------------------------------------------------
# GOTCHA, and you will hit it. On Julia 1.12 `LinearAlgebra` exports its own
# `inertia`. Under `using LinearAlgebra, Veigs` the bare name `inertia` is
# ambiguous and resolves to nothing at all -- you get an UndefVarError that
# looks like Veigs failed to load. Qualify it: `Veigs.inertia(D)`.
# ---------------------------------------------------------------------------
println("LinearAlgebra also exports `inertia`? ", isdefined(LinearAlgebra, :inertia))

# Reproduce the collision in a throwaway module that does the two BARE
# `using` statements a reader would naturally write.
module _AmbiguityDemo
    using LinearAlgebra
    using Veigs
    probe() = inertia(Diagonal([1.0, -1.0]))
end
try
    println("bare `inertia(...)` under `using LinearAlgebra, Veigs` -> ",
            _AmbiguityDemo.probe())
catch err
    println("bare `inertia(...)` under `using LinearAlgebra, Veigs` -> ",
            typeof(err), ": ", first(sprint(showerror, err), 160))
end
println("Veigs.inertia(Diagonal([1.0,-1.0])) = ", Veigs.inertia(Diagonal([1.0, -1.0])),
        "     <-- qualify, always")

# `inertia(D)` returns (neg, pos, zero, F). `D` is the block-diagonal factor
# of an LDL' factorisation: 1x1 and 2x2 blocks only. The counts are certified
# WHEN F == false. F == true means at least one block's sign could not be
# decided in interval arithmetic; the counts are then partial and worthless.
println()
println("returns (neg, pos, zero, F);  F == true means 'could not certify'")
for (label, D) in (
        ("Diagonal([1,-2,0])",     Diagonal([1.0, -2.0, 0.0])),
        ("[0 1; 1 0]  (2x2 blk)",  [0.0 1.0; 1.0 0.0]),
        ("[-2 1; 1 -2]",           [-2.0 1.0; 1.0 -2.0]),
        ("Diagonal([1e-300])",     Diagonal([1e-300])),
    )
    @printf("  %-24s -> %s\n", label, Veigs.inertia(D))
end

# --- Sylvester's law of inertia, made concrete -----------------------------
# The theorem: congruence preserves signs. If M = A - lambda*B and we factor
# M = P L D L' P', then D has exactly as many negative entries as M has
# negative eigenvalues. And M has k negative eigenvalues precisely when k
# eigenvalues of the pair (A, B) lie strictly below lambda.
#
# So: pick lambda, factor, count the negatives -> you have COUNTED how many
# eigenvalues are below lambda, without computing a single one of them.
hdr("1b. Sylvester's law: sweep lambda and watch the count step")

A = Matrix(Diagonal([1.0, 2.0, 4.0, 8.0, 16.0]))
B = Matrix{Float64}(I, 5, 5)
println("A = diag(1, 2, 4, 8, 16), B = I  -- spectrum known exactly")
println()
println(rpad("lambda", 10), rpad("neg", 6), rpad("pos", 6), rpad("zero", 6),
        rpad("F", 7), "interpretation")
println("-"^66)
for lam in (0.5, 1.5, 3.0, 6.0, 12.0, 20.0)
    local Msweep = A .- lam .* B
    local Lx, Dx, px, dMx, okx
    Lx, Dx, px, dMx, okx = verified_ldl(Msweep)
    local neg, pos, zer, Fflag
    neg, pos, zer, Fflag = Veigs.inertia(Dx)
    @printf("%-10.1f%-6d%-6d%-6d%-7s%d eigenvalue(s) lie below %.1f\n",
            lam, neg, pos, zer, string(Fflag), neg, lam)
end
println()
println("That column of counts IS the localisation. A bisection on lambda")
println("driven by this count brackets any eigenvalue you like to arbitrary")
println("precision, and every step is a certified sign, not an estimate.")

# =============================================================================
# 2. verified_ldl -- the factorisation plus a rigorous residual
# =============================================================================
hdr("2. verified_ldl: float factors + interval residual")

# `verified_ldl(M) -> (L, D, p, dM, ok)`.
#   L, D, p : the FLOAT Bunch-Kaufman (or sparse no-pivot LDL) factors of
#             mid(M), satisfying mid(M)[p,p] ~ L*D*L' up to round-off.
#   dM      : an INTERVAL matrix that rigorously encloses the residual
#             M - P*L*D*L'*P'. This is where the rigour enters: the factors
#             are ordinary floats, and dM accounts for every bit they got
#             wrong.
#   ok      : false only if LAPACK rejected the input outright.
M = [4.0 2.0 1.0
     2.0 5.0 3.0
     1.0 3.0 6.0]
L, D, p, dM, ok = verified_ldl(M)
println("ok            = ", ok)
println("permutation p = ", p)
println("float check  M[p,p] ~ L*D*L' : ", M[p, p] ≈ L * D * L')
@printf("max |mid(dM)| = %.3e   (float reconstruction error, midpoint)\n",
        maximum(abs, mid.(dM)))
@printf("max radius(dM)= %.3e   (the certified slack)\n", maximum(radius.(dM)))
println("inertia of D  = ", Veigs.inertia(D), "   (M is SPD -> (0,3,0,false))")

# The soundness statement: M is contained in P*L*D*L'*P' + dM. Verified.
println()
println("Sparse input keeps a sparse path (LDLFactorizations + AMD ordering):")
Ms = spdiagm(-1 => fill(-1.0, 4), 0 => fill(2.0, 5), 1 => fill(-1.0, 4))
Ls, Ds, ps, dMs, oks = verified_ldl(Ms)
println("  typeof(L) = ", typeof(Ls))
println("  typeof(D) = ", typeof(Ds))
println("  ok        = ", oks, "   inertia = ", Veigs.inertia(Ds))

# ---------------------------------------------------------------------------
# THE MEMORY CLIFF. When the sparse no-pivot LDL hits a zero pivot,
# verified_ldl falls back to dense Bunch-Kaufman -- and Julia's
# `bunchkaufman` is dense-only. That means an n x n Float64 densification
# plus its interval residual, roughly 2 * 8 * n^2 bytes inside the
# factorisation. At n = 5000 that is about 400 MB, and it is this, not the
# factor-of-2 from interval arithmetic, that sets the practical ceiling.
# ---------------------------------------------------------------------------
for n in (1000, 2000, 5000, 10000)
    @printf("  dense fallback footprint at n=%-6d ~ %7.1f MB\n", n, 2*8*n^2/2^20)
end

# =============================================================================
# 3. verified_isspd -- the positive-definiteness decision
# =============================================================================
hdr("3. verified_isspd and DEFAULT_ISSPD_METHOD")

# The contract is one-sided, and the asymmetry is the whole design:
#   returns true  => EVERY matrix in the interval enclosure is SPD. A theorem.
#   returns false => could not certify. Might genuinely be indefinite, might
#                    be PD but too close to the boundary for the residual
#                    bound to clear.
# A verified routine must fail closed. Never read `false` as "not PD".

# `DEFAULT_ISSPD_METHOD` is a Base.RefValue{Symbol}, not a Symbol -- read it
# with the empty-index syntax. It exists so every internal caller
# (rough_bounds, lehmann_behnke, veig, veigs) switches algorithm in lockstep.
println("typeof(DEFAULT_ISSPD_METHOD)  = ", typeof(Veigs.DEFAULT_ISSPD_METHOD))
println("DEFAULT_ISSPD_METHOD[]        = ", Veigs.DEFAULT_ISSPD_METHOD[])
println()
println("  :rump2006  -- Rump 2006 (BIT 46:433-452). One float Cholesky of")
println("                (mid(M) - shift*I) with a closed-form shift. Sparse-")
println("                friendly (SuiteSparse), and what INTLAB's isspd does.")
println("  :ldl_shift -- Bunch-Kaufman LDL of (M - rho*I), inertia of D, then")
println("                Weyl's inequality against the residual norm; doubles")
println("                rho and retries. Densifies sparse input.")

println()
println(rpad("matrix", 34), rpad(":rump2006", 12), rpad(":ldl_shift", 12), "isposdef")
println("-"^70)
cases = Any[
    ("I(5)",                        Matrix{Float64}(I, 5, 5)),
    ("diag(1e-12, 2e-12, 3e-12)",   Matrix(Diagonal([1e-12, 2e-12, 3e-12]))),
    ("diag(-1, 1, 2)",              Matrix(Diagonal([-1.0, 1.0, 2.0]))),
    ("diag(-1e-12, 1, 2)",          Matrix(Diagonal([-1e-12, 1.0, 2.0]))),
    ("Hilbert(8)",                  [1.0/(i+j-1) for i in 1:8, j in 1:8]),
    ("zeros(3,3)",                  zeros(3, 3)),
]
for (label, Mx) in cases
    local r1 = verified_isspd(Mx)                       # default = :rump2006
    local r2 = verified_isspd(Mx; method = :ldl_shift)  # per-call override
    local ip = try isposdef(Symmetric(Mx)) catch; "n/a" end
    @printf("%-34s%-12s%-12s%s\n", label, string(r1), string(r2), string(ip))
end

# The soundness battery, in miniature: a matrix whose smallest eigenvalue is
# exactly -eps must return false for every eps. If it ever returned true the
# entire library would be worthless.
println()
println("SOUNDNESS: lambda_min = -eps must never certify as PD")
using Random
Random.seed!(20260501)
Q, _ = qr(randn(10, 10)); Q = Matrix(Q)
for eps_ in (1e-2, 1e-8, 1e-14)
    local evn = collect(range(1.0, 10.0; length = 10)); evn[1] = -eps_
    local Mn = Symmetric(Q * Diagonal(evn) * Q')
    @printf("  lambda_min = %-9.0e -> :rump2006 %s, :ldl_shift %s\n", -eps_,
            string(verified_isspd(Matrix(Mn))),
            string(verified_isspd(Matrix(Mn); method = :ldl_shift)))
end

# Interval input widens the question from "is this matrix PD" to "is every
# matrix in this box PD", and at some radius the answer flips to "cannot say".
println()
println("Interval input: widening the box until certification fails")
Mb = Matrix(Diagonal([1.0, 2.0, 3.0]))
for r in (1e-12, 1e-3, 0.5, 0.99, 1.5)
    local Mi = interval.(Mb .- r, Mb .+ r)
    @printf("  radius %-8.0e -> verified_isspd = %s\n", r, string(verified_isspd(Mi)))
end

# =============================================================================
# 4. rough_lower / rough_upper -- bracketing the spectrum
# =============================================================================
hdr("4. rough_lower / rough_upper: the initial bracket")

# rough_lower(A, B, lambda_seed, lambda_floor) walks DOWN from the seed on a
# doubling schedule -- delta = max(eps, |seed - floor|/2^32), then
# lambda_k = seed - delta*2^(k-1) -- and returns the first lambda for which
# verified_isspd(A - lambda*B) succeeds. Because A - lambda*B is then
# certified PD, EVERY eigenvalue of the pair is provably above lambda.
#
# rough_upper mirrors it: it certifies lambda*B - A as PD, so every
# eigenvalue is provably below lambda.
ev_true = sort(eigvals(Symmetric(A), Symmetric(B)))
lo = rough_lower(A, B, ev_true[1]   - 0.01, ev_true[1]   - 20.0)
up = rough_upper(A, B, ev_true[end] + 0.01, ev_true[end] + 20.0)
@printf("spectrum          [%.6f, %.6f]\n", ev_true[1], ev_true[end])
@printf("rough_lower       %.17g   (verified: all eigenvalues > this)\n", lo)
@printf("rough_upper       %.17g   (verified: all eigenvalues < this)\n", up)
println("bracket is valid: ", lo < ev_true[1] && ev_true[end] < up)
@printf("slack below %.3e, slack above %.3e\n", ev_true[1] - lo, up - ev_true[end])

# The seed does not have to be valid. Start it INSIDE the spectrum and the
# doubling schedule walks out until the PD test clears.
lo2 = rough_lower(A, B, 3.0, -50.0)      # 3.0 sits between lambda_2 and lambda_3
@printf("seed inside the spectrum (3.0) still lands at %.6f\n", lo2)

# This is also how veigs bounds lambda_min(B): call rough_lower(B, I, ...).
# `veigs` needs that number as the energy denominator in Lehmann-Behnke.
n5 = 5
I5 = Matrix{Float64}(I, n5, n5)
lamB = rough_lower(B, I5, 0.9, 0.0)
@printf("rough_lower(B, I, 0.9, 0.0) = %.17g   (lower bound on lambda_min(B))\n", lamB)

# =============================================================================
# 5. lehmann_behnke -- the sharpening step
# =============================================================================
hdr("5. lehmann_behnke: turning a bracket into a tight two-sided bound")

# The bracket from rough_* is crude. Lehmann-Behnke converts approximate
# eigenvectors plus two SEPARATORS into a genuinely tight enclosure, using
# complementary variational principles:
#
#   rho   -- a verified UPPER bound on lambda_{r-1} (the left separator)
#   sigma -- a verified LOWER bound on lambda_{s+1} (the right separator)
#
# Given those two walls, the cluster lambda_r .. lambda_s is boxed in, and the
# method returns bounds whose width is driven by the RESIDUAL of the
# approximate eigenvectors rather than by the width of the bracket. That is
# why veigs's intervals are so much narrower than rough_lower/rough_upper.
#
# Signature (order matters, and it is long):
#   lehmann_behnke(A, B, eig_list, V, rho, sigma, lambda_B_min, r, s;
#                  do_shift = true)
F = eigen(Symmetric(A), Symmetric(B))
perm = sortperm(real.(F.values))
eig_list = real.(F.values[perm])
V = real.(F.vectors[:, perm])

println("Single eigenvalue (r = s), sweeping the whole spectrum:")
for r in 1:5
    local rho   = r > 1 ? (eig_list[r-1] + eig_list[r]) / 2 : eig_list[1] - 1.0
    local sigma = r < 5 ? (eig_list[r] + eig_list[r+1]) / 2 : eig_list[5] + 1.0
    local bd = lehmann_behnke(A, B, eig_list, V[:, r:r], rho, sigma, 1.0, r, r)
    @printf("  lambda_%d: %-46s width %.3e   encloses %.1f: %s\n",
            r, fmt_iv(bd[1]), diam(bd[1]), eig_list[r],
            string(inf(bd[1]) <= eig_list[r] <= sup(bd[1])))
end

# A genuine cluster: two eigenvalues 1e-3 apart, certified jointly.
println()
println("Cluster of two (lambda_1, lambda_2 separated by 1e-3):")
A2c = Matrix(Diagonal([3.0, 3.0 + 1e-3, 6.0, 8.0, 16.0]))
B2c = Matrix{Float64}(I, 5, 5)
F2 = eigen(Symmetric(A2c), Symmetric(B2c))
p2 = sortperm(real.(F2.values))
ev2, V2 = real.(F2.values[p2]), real.(F2.vectors[:, p2])
bd2 = lehmann_behnke(A2c, B2c, ev2, V2[:, 1:2],
                     ev2[1] - 1.0, (ev2[2] + ev2[3]) / 2, 1.0, 1, 2)
for i in 1:2
    @printf("  index %d: %-46s encloses %.6f: %s\n", i, fmt_iv(bd2[i]),
            ev2[i], string(inf(bd2[i]) <= ev2[i] <= sup(bd2[i])))
end

# do_shift: the conditioning shift A -> A - eig_list[r]*B moves the cluster
# near zero before the small Lehmann subproblem is formed. It is on by default
# because it materially improves the conditioning of that subproblem.
println()
println("do_shift on versus off (r = s = 3):")
rho3, sigma3 = (eig_list[2] + eig_list[3]) / 2, (eig_list[3] + eig_list[4]) / 2
for sh in (true, false)
    local bd = lehmann_behnke(A, B, eig_list, V[:, 3:3], rho3, sigma3, 1.0, 3, 3;
                              do_shift = sh)
    @printf("  do_shift = %-6s -> %-46s width %.3e\n",
            string(sh), fmt_iv(bd[1]), diam(bd[1]))
end

# =============================================================================
# 6. sym_hull and assume_symmetric
# =============================================================================
hdr("6. sym_hull: repairing symmetry that interval round-off broke")

# In exact arithmetic V'*A*V is symmetric. In INTERVAL arithmetic it need not
# be: the entry (i,j) and the entry (j,i) are computed by different summation
# orders, so their enclosures differ slightly even though the true values are
# equal. `verified_isspd` needs a symmetric input, so Veigs repairs the matrix
# first with sym_hull: entry (i,j) becomes hull(M[i,j], M[j,i]) -- the
# smallest interval containing both, which therefore contains the true value.
Ma = [interval(1.0, 1.0)      interval(2.0, 2.5)
      interval(2.1, 2.4)      interval(3.0, 3.0)]
H = sym_hull(Ma)
println("input  M[1,2] = ", fmt_iv(Ma[1,2]), "   M[2,1] = ", fmt_iv(Ma[2,1]))
println("hulled H[1,2] = ", fmt_iv(H[1,2]),  "   H[2,1] = ", fmt_iv(H[2,1]))
println("H is symmetric entrywise: ", isequal_interval(H[1,2], H[2,1]))
println("(note the comparison: isequal_interval, never ==)")

# `assume_symmetric = true` short-circuits the whole routine and returns M
# untouched. This is a performance lever, not a cosmetic flag -- the
# docstring records the broadcast form as roughly 21% of total veigs wall
# time at n = 5000. Pass it only when the matrix is genuinely symmetric in
# interval arithmetic, which for `A - lambda*B` built from two symmetric
# matrices it is, entry by entry.
Hs = sym_hull(Ma; assume_symmetric = true)
println("assume_symmetric=true returns the input unchanged: ", Hs === Ma)
println()
println("Safe to pass:      M = A .- interval(lam) .* B, A and B symmetric")
println("NOT safe to pass:  M = V'*A*nV - nV'*(B*nV - A*V) + Err")
println("                   (mixed products; round-off breaks symmetry)")

# Timing the lever on a size where it matters.
nb = 400
Ab = Matrix(Diagonal(collect(1.0:nb)))
Ab[1, 2] = Ab[2, 1] = 0.5
Abi = interval.(Ab)
sym_hull(Abi); sym_hull(Abi; assume_symmetric = true)   # warm up
t_full = @elapsed sym_hull(Abi)
t_skip = @elapsed sym_hull(Abi; assume_symmetric = true)
@printf("n=%d dense interval: sym_hull %.4f s, assume_symmetric %.6f s (%.0fx)\n",
        nb, t_full, t_skip, t_full / max(t_skip, eps()))

# =============================================================================
# 7. Every error type, with a snippet that really fires it
# =============================================================================
hdr("7. The Veigs error hierarchy")

println("abstract type VeigsError <: Exception, with five concrete subtypes.")
println("Below, each is triggered for real -- or documented as unprovoked.")
println()

results = Any[]
# NOTE the argument order: Julia's `f(args...) do ... end` syntax passes the
# anonymous function as the FIRST positional argument, so the callable has to
# come first in the signature even though it reads last at the call site.
function try_case(f::Function, name::String, expect::String)
    print(rpad(name, 46))
    got = try
        f()
        "NO ERROR (returned normally)"
    catch err
        string(typeof(err))
    end
    fired = occursin(expect, got)
    println(rpad(got, 34), fired ? "MATCH" : "MISMATCH")
    push!(results, (name, expect, got, fired))
    return fired
end

# --- VeigsSizeError: three distinct triggers, all confirmed ---------------
try_case("veig on n=101 (hard cap is 100)", "VeigsSizeError") do
    n = 101
    veig(Matrix{Float64}(I, n, n), Matrix{Float64}(I, n, n))
end
try_case("veigs with non-square A", "VeigsSizeError") do
    veigs(zeros(3, 4), zeros(3, 4))
end
try_case("veig with asymmetric A", "VeigsSizeError") do
    veig([1.0 0.5; 0.6 2.0], Matrix{Float64}(I, 2, 2))
end

# --- VeigsLDLFailureError -------------------------------------------------
# Trigger A: `inertia(..., abort_on_err=true)` on a 2x2 block whose
# determinant interval straddles zero. det = 1*1 - [0.99,1.01]^2 spans zero,
# so the sign cannot be certified and the routine refuses to guess.
try_case("inertia(undecidable 2x2; abort_on_err)", "VeigsLDLFailureError") do
    Dbad = [interval(1.0, 1.0)     interval(0.99, 1.01)
            interval(0.99, 1.01)   interval(1.0, 1.0)]
    Veigs.inertia(Dbad; abort_on_err = true)
end
# Trigger B: force the sparse LDL backend on a matrix with a zero pivot. With
# ldl_backend = :auto this would silently fall back to dense Bunch-Kaufman;
# :ldlfac forbids the fallback and raises instead.
try_case("verified_ldl(zero pivot, backend=:ldlfac)", "VeigsLDLFailureError") do
    Z = sparse([1, 1, 2, 3, 3, 4], [1, 3, 2, 1, 3, 4],
               [0.0, 1.0, 1.0, 1.0, 1.0, 1.0], 4, 4)
    verified_ldl(Z; ldl_backend = :ldlfac)
end
# Trigger C: the :mumps backend is an exposed seam, not an implementation.
try_case("verified_ldl(backend=:mumps) -- seam", "VeigsLDLFailureError") do
    verified_ldl([2.0 1.0; 1.0 2.0]; ldl_backend = :mumps)
end

# --- VeigsLehmannBehnkeError ----------------------------------------------
# Feed an INVALID right separator. sigma must be a lower bound on
# lambda_{s+1}; here s = 2 and lambda_2 = 2, so sigma = 1.5 lies below the
# cluster it is supposed to wall off. The lower-side Lehmann matrix SB is
# then indefinite, verified_isspd fails, and the routine throws rather than
# return a bound it cannot justify.
try_case("lehmann_behnke with sigma below the cluster", "VeigsLehmannBehnkeError") do
    lehmann_behnke(A, B, eig_list, V[:, 1:2], 0.0, 1.5, 1.0, 1, 2;
                   do_shift = false)
end

# --- VeigsRoughBoundError -------------------------------------------------
# Put BOTH the seed and the floor inside the spectrum. Every lambda the
# doubling schedule visits leaves at least one negative eigenvalue in
# A - lambda*B, so no candidate ever certifies PD and the search exhausts.
try_case("rough_lower with seed AND floor inside spectrum", "VeigsRoughBoundError") do
    rough_lower(A, B, 3.0, 1.5)      # lambda_1 = 1, lambda_2 = 2: both inside
end

# --- VeigsClusterTooLargeError -------------------------------------------
# This one fires deep inside veigs's cluster-expansion loop, when the inertia
# counts disagree with the float eigenvalue ordering by more than the loop can
# reconcile -- i.e. when the approximate eigenvalues fed to it are so badly
# conditioned that the certified count and the float count part company.
# Several adversarial candidates are attempted below.
println()
println("Attempts to provoke VeigsClusterTooLargeError from veigs:")
cluster_fired = false
cluster_attempts = Any[]
for (label, Ax, Bx, k, sg) in Any[
        ("exactly-degenerate diag(1,1,1,1,9), k=4", Matrix(Diagonal([1.0,1.0,1.0,1.0,9.0])),
         Matrix{Float64}(I,5,5), 4, :smallestreal),
        ("all-identical diag(2,2,2,2,2), k=5", Matrix(Diagonal(fill(2.0,5))),
         Matrix{Float64}(I,5,5), 5, :smallestreal),
        ("Hilbert(12) vs I, k=6 (cond ~1e16)", [1.0/(i+j-1) for i in 1:12, j in 1:12],
         Matrix{Float64}(I,12,12), 6, :smallestabs),
        ("near-degenerate 1+1e-15 pair, k=2", Matrix(Diagonal([1.0,1.0+1e-15,5.0])),
         Matrix{Float64}(I,3,3), 2, :smallestreal),
        ("ill-scaled diag(1e-14,1,1e14), k=3", Matrix(Diagonal([1e-14,1.0,1e14])),
         Matrix{Float64}(I,3,3), 3, :smallestreal),
    ]
    local got = try
        local lam, ir
        lam, ir = veigs(Ax, Bx, k, sg)
        "ok (ind_range=$(ir), $(length(lam)) bounds)"
    catch err
        string(typeof(err)) * ": " * sprint(showerror, err)
    end
    occursin("ClusterTooLarge", got) && (global cluster_fired = true)
    println("  ", rpad(label, 42), first(got, 62))
    push!(cluster_attempts, (label, got))
end
println()
if cluster_fired
    println("VeigsClusterTooLargeError WAS provoked above.")
else
    println("VeigsClusterTooLargeError was NOT provoked by any of these five")
    println("adversarial inputs. Reporting that honestly rather than faking a")
    println("trigger. What it means: inside the cluster-expansion loop, veigs")
    println("compares the certified inertia span (global_s - global_r) against")
    println("the span of the float eigenvalue block it is working on (s - r).")
    println("If the certified span is WIDER, the cluster has swallowed more")
    println("eigenvalues than the approximate eigensolver supplied vectors for,")
    println("and veigs raises this error -- the fix on the caller's side is to")
    println("request a larger k so more approximate eigenpairs are available.")
    println("Reaching it needs the approximate eigensolver to under-resolve a")
    println("cluster that the certified count then finds; on well-conditioned")
    println("small problems the dense LAPACK path resolves everything, so the")
    println("branch stays unreached. Its sibling VeigsRoughBoundError guards")
    println("the opposite inequality (certified span NARROWER than the float")
    println("span) and is reachable via rough_lower, as shown above.")
end

# --- summary table --------------------------------------------------------
println()
println("Error-type coverage summary")
println(rpad("scenario", 46), rpad("expected", 26), "fired")
println("-"^84)
for (name, expect, got, fired) in results
    @printf("%-46s%-26s%s\n", name, expect, fired ? "yes" : "NO -- " * got)
end

open("veigs_errors.csv", "w") do io
    println(io, "scenario,expected_error,observed,fired")
    for (name, expect, got, fired) in results
        println(io, "\"$name\",$expect,\"$(replace(got, '"' => '\''))\",$fired")
    end
    for (label, got) in cluster_attempts
        println(io, "\"cluster attempt: $label\",VeigsClusterTooLargeError,\"$(replace(first(got,70), '"' => '\''))\",$(occursin("ClusterTooLarge", got))")
    end
end
println("\n-> wrote veigs_errors.csv")

println()
println("VEIGS_INTERNALS_DONE_MARKER")
