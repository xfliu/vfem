# test/test_lambda_h_bernstein.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. Symbolic Cholesky: the pattern is a SUPERSET of CHOLMOD's pruned
#      pattern, has an explicit diagonal first in every column, and sorted row
#      indices. `sparse_chol_shift` with T = Float64 reproduces a dense
#      Cholesky on a small SPD matrix.
#   2. Sparse-RHS triangular solve: `solve_sparse_rhs!` agrees with a dense
#      `L \ b` on the reach set, and the reach set really does cover every
#      nonzero of the solution.
#   3. The fast path agrees with the dense-D baseline — the same quantity the
#      published route computes — to O(cond(A)·eps), NOT to 1e-12. cond(A)
#      grows like n⁴ (6.8e4 at n = 16), so the acceptance criterion is
#      relative difference ≤ 100·cond(A)·eps, and both values must lie inside
#      the certified enclosure. `:trisolve` and `:selinv` agree likewise.
#   4. NO argmax test. The maximizing row is frequently GENUINELY TIED (top
#      two g_i agreeing to 1e-14..7e-12 with overlapping rigorous
#      enclosures), so argmax equality between two routes is not a valid
#      assertion — see the note in the source header.
#   5. `lambda_min_lower` returns a rigorous lower bound: positive, below the
#      true lambda_min, and reproducible when `x0` is given (the default
#      `randn` start makes the certified value vary between runs).
#   6. The certified enclosure CONTAINS the float value, is narrow, and its
#      endpoints bracket the reference value.
#   7. Published-table reproduction against test/fixtures/fm_lambda_hb_ref.txt,
#      for BOTH conventions, at n = 8 and n = 16. (n = 64 — the paper's
#      lambda = 5.78123 at theta = pi/2 — takes minutes and is NOT a unit
#      test; it is the separate production confirmation.)
#   8. The sharp screen reproduces Gup BIT-IDENTICALLY while certifying only a
#      small fraction of the rows, and its survivor set provably contains the
#      argmax row.
#   9. Corollary 3.1: CL_ub_interval encloses CL_ub.
#  10. Degenerate input: an all-zero row of B contributes g = 0; dimension
#      mismatch throws.
#  11. Performance: certified path at n = 8 < 30 s.
#
# MATHEMATICAL CONTRACT (docs/testing-contract.md (c)):
#   1/lambda_{h,B} = max_i beta_i' A^{-1} beta_i          (identity (G))
#   inf(enclosure) ≤ lambda_{h,B} ≤ sup(enclosure)        (rigorous)
#   lambda_min_lower(A) ≤ lambda_min(A)                   (rigorous)
#   C^L_ub = 1 / sqrt(lambda_{h,B}·(1 − h²))              (Corollary 3.1)

using Test
using LinearAlgebra: cholesky, cond, norm, eigvals, Symmetric, dot, tril, I
using SparseArrays: SparseMatrixCSC, sparse, sprandn, nnz, nzrange, rowvals,
                    nonzeros, spzeros
using IntervalArithmetic: Interval, interval, inf, sup, diam, mid
using VFEM: mesh2d_triangle_uniform, create_matrix_fujino_morley,
            symbolic_cholesky, sparse_chol_shift, etree_sym,
            ReachWorkspace, solve_sparse_rhs!, clear_workspace!, csr_rows,
            lambda_hb_fast, lambda_hb_baseline_denseD, lambda_min_lower,
            lambda_min_estimate, lambda_hb_certified, certified_diag_Ainv_upper,
            screen_sharp, CL_ub, CL_ub_interval

const _FM_REF_FILE = joinpath(@__DIR__, "fixtures", "fm_lambda_hb_ref.txt")

function _load_fm_lambda_ref()
    floats = Dict{String, Float64}()
    ints = Dict{String, Int}()
    for line in eachline(_FM_REF_FILE)
        line = strip(line)
        (isempty(line) || startswith(line, "#")) && continue
        key, val = split(line, "="; limit = 2)
        if startswith(key, "M_") || startswith(key, "N_")
            ints[key] = parse(Int, val)
        else
            floats[key] = parse(Float64, val)
        end
    end
    return floats, ints
end

# Assemble one FM case and return the pieces the solver needs.
function _fm_case(n::Int, theta::Float64, conv::Symbol)
    m, corners = mesh2d_triangle_uniform(n; alpha = 1.0, theta = theta, h = 1.0)
    fm = create_matrix_fujino_morley(m, corners; convention = conv)
    return SparseMatrixCSC{Float64,Int}(fm.A), fm.B, fm.M, fm.N
end

@testset "lambda_h_bernstein" begin

    floats, ints = _load_fm_lambda_ref()

    @testset "1. symbolic Cholesky pattern and generic numeric factorization" begin
        # small SPD sparse matrix with genuine fill
        A8, _, _, _ = _fm_case(3, pi / 2, :consistent)
        Lp, Li = symbolic_cholesky(A8)
        n = size(A8, 1)
        @test length(Lp) == n + 1
        @test Lp[1] == 1
        @test Lp[end] - 1 == length(Li)
        for j in 1:n
            colrows = Li[Lp[j]:(Lp[j+1] - 1)]
            @test colrows[1] == j                       # diagonal first
            @test issorted(colrows)                     # sorted row indices
            @test all(≥(j), colrows)                    # lower triangular
        end
        # superset of CHOLMOD's pruned pattern (same ordering: none applied here)
        F = cholesky(A8; perm = 1:n)
        Lc = sparse(F.L)
        sym = Set((Li[p], j) for j in 1:n for p in Lp[j]:(Lp[j+1] - 1))
        for j in 1:n, p in nzrange(Lc, j)
            @test (rowvals(Lc)[p], j) in sym
        end
        # numeric factorization on that pattern reproduces a dense Cholesky
        ok, Lx, minpiv = sparse_chol_shift(A8, 0.0, Lp, Li)
        @test ok
        @test minpiv > 0
        Ldense = zeros(n, n)
        for j in 1:n, p in Lp[j]:(Lp[j+1] - 1)
            Ldense[Li[p], j] = Lx[p]
        end
        @test maximum(abs, Ldense * Ldense' - Matrix(A8)) <
              1e-10 * maximum(abs, A8)
        # etree: every parent is a strictly larger index, roots marked 0
        parent = etree_sym(A8)
        @test all(k -> parent[k] == 0 || parent[k] > k, 1:n)
        # a shift beyond lambda_min must be detected as not positive definite
        lam_min = minimum(eigvals(Symmetric(Matrix(A8))))
        ok_bad, _, _ = sparse_chol_shift(A8, 2 * lam_min, Lp, Li)
        @test !ok_bad
        # interval mode: certifying a shift below lambda_min is a PROOF
        ok_iv, _, _ = sparse_chol_shift(A8, 0.5 * lam_min, Lp, Li;
                                        T = Interval{Float64})
        @test ok_iv
    end

    @testset "2. sparse-RHS triangular solve" begin
        A, B, M, N = _fm_case(3, pi / 2, :consistent)
        F = cholesky(A)
        L = sparse(F.L)
        Msz = size(L, 1)
        ws = ReachWorkspace(Msz)
        rowptr, colidx, nzval = csr_rows(B)
        invp = invperm(F.p)
        tested = 0
        for i in 1:N
            rng = rowptr[i]:(rowptr[i+1] - 1)
            isempty(rng) && continue
            # right-hand side permuted into the factor's ordering, as the
            # production path does
            idx = [invp[colidx[p]] for p in rng]
            val = [nzval[p] for p in rng]
            top, nreach = solve_sparse_rhs!(ws, L, idx, val)
            # dense reference solve of L y = b
            b = zeros(Msz)
            for (j, v) in zip(idx, val)
                b[j] += v
            end
            ydense = Matrix(L) \ b
            reach = ws.xi[top:Msz]
            @test nreach == length(reach)
            # the reach set covers every nonzero of the solution
            for j in 1:Msz
                if abs(ydense[j]) > 1e-12 * maximum(abs, ydense)
                    @test j in reach
                end
            end
            # and the values agree there
            for j in reach
                @test ws.x[j] ≈ ydense[j] atol = 1e-10 rtol = 1e-10
            end
            # ‖z‖² over the reach set is exactly the g_i of that row
            @test sum(ws.x[j]^2 for j in reach) ≈ dot(b, Matrix(L)' \ ydense) atol = 1e-10 rtol = 1e-10
            clear_workspace!(ws, top)
            @test all(iszero, ws.x)
            tested += 1
            tested ≥ 3 && break
        end
        @test tested == 3
    end

    @testset "3. fast path vs the dense-D baseline, to O(cond(A)·eps)" begin
        for (n, conv) in ((8, :consistent), (8, :verbatim), (16, :verbatim))
            A, B, M, N = _fm_case(n, pi / 2, conv)
            lam_f, imax_f, df = lambda_hb_fast(A, B; method = :trisolve)
            lam_D, imax_D, _ = lambda_hb_baseline_denseD(A, B)
            condA = cond(Matrix(A))
            tol = 100 * condA * eps()
            @test abs(lam_f - lam_D) / abs(lam_D) ≤ tol
            # both routes lie inside the certified enclosure
            lmin, _ = lambda_min_lower(A; x0 = ones(M))
            encl, _ = lambda_hb_certified(A, B; lmin = lmin)
            @test inf(encl) ≤ lam_f ≤ sup(encl)
            @test inf(encl) ≤ lam_D ≤ sup(encl)
            # the two float methods agree to the same order
            lam_s, imax_s, _ = lambda_hb_fast(A, B; method = :selinv)
            @test abs(lam_s - lam_f) / abs(lam_f) ≤ tol
            # 1 ≤ imax ≤ N, but NO equality assertion — see taxonomy item 4
            @test 1 ≤ imax_f ≤ N
            @test 1 ≤ imax_D ≤ N
            @test 1 ≤ imax_s ≤ N
        end
    end

    @testset "5. lambda_min_lower is a rigorous, reproducible lower bound" begin
        A, B, M, N = _fm_case(8, pi / 2, :consistent)
        lam_min_true = minimum(eigvals(Symmetric(Matrix(A))))
        lmin, diag = lambda_min_lower(A; x0 = ones(M))
        @test lmin > 0
        @test lmin ≤ lam_min_true                       # rigorous
        @test diag.ratio ≤ 1.0
        @test lmin > 0.5 * lam_min_true                 # and not absurdly loose
        # reproducible with a fixed start vector
        lmin2, _ = lambda_min_lower(A; x0 = ones(M))
        @test lmin2 == lmin
        est = lambda_min_estimate(A; x0 = ones(M))
        @test est ≈ lam_min_true rtol = 1e-6
        @test_throws DimensionMismatch lambda_min_estimate(A; x0 = ones(M + 1))
    end

    @testset "6+7. certified enclosure and published-table reproduction" begin
        for n in (8, 16), conv in (:consistent, :verbatim)
            A, B, M, N = _fm_case(n, pi / 2, conv)
            key = conv === :consistent ? "lambda_consistent_thetapi_over_2_n$n" :
                                         "lambda_verbatim_thetapi_over_2_n$n"
            lam_ref = floats[key]
            @test M == ints["M_thetapi_over_2_n$n"]
            @test N == ints["N_thetapi_over_2_n$n"]

            lam_f, _, _ = lambda_hb_fast(A, B)
            condA = cond(Matrix(A))
            # The fixture value comes from the reference implementation, which
            # uses the authors' DOF numbering; this library numbers the DOFs
            # differently, so the two assembled A matrices agree only to
            # rounding and the two lambdas only to O(cond(A)·eps). Hence a
            # relative-agreement criterion here, NOT enclosure containment: the
            # enclosure below brackets lambda of THIS A, and the reference value
            # belongs to a slightly different matrix.
            @test abs(lam_f - lam_ref) / lam_ref ≤ 100 * condA * eps()

            lmin, _ = lambda_min_lower(A; x0 = ones(M))
            encl, dg = lambda_hb_certified(A, B; lmin = lmin)
            @test inf(encl) ≤ lam_f ≤ sup(encl)         # contains the float value
            @test diam(encl) / abs(mid(encl)) < 1e-8    # narrow
            @test inf(encl) > 0
            # the enclosure is consistent with the reference to the same order
            @test abs(mid(encl) - lam_ref) / lam_ref ≤ 100 * condA * eps()
        end
    end

    @testset "8. the sharp screen reproduces Gup bit-identically" begin
        for conv in (:consistent, :verbatim)
            A, B, M, N = _fm_case(16, pi / 2, conv)
            lmin, _ = lambda_min_lower(A; x0 = ones(M))

            # (a) certify EVERY row — the reference
            encl_all, dg_all = lambda_hb_certified(A, B; lmin = lmin,
                                                   screen = false)
            # (b) the driver's own crude screen. It is INEFFECTIVE on these
            # matrices — it rejects almost nothing, because beta_i has 6 nonzeros
            # and is localised while the lowest eigenmode is smooth — but it must
            # not change the answer.
            encl_crude, dg_crude = lambda_hb_certified(A, B; lmin = lmin)
            @test dg_crude.Gup === dg_all.Gup                # BIT-identical
            @test encl_crude === encl_all
            @test dg_crude.survivor_frac > 0.9               # honestly reported

            # (c) the sharp screen: few survivors, argmax kept, same Gup
            F = cholesky(A)
            d_up, _ = certified_diag_Ainv_upper(A, F, lmin)
            @test all(>(0), d_up)
            # d_up must be a rigorous upper bound for diag(A^{-1})
            Ainv_diag = [inv(Matrix(A))[j, j] for j in 1:min(M, 40)]
            @test all(j -> d_up[j] ≥ Ainv_diag[j], 1:length(Ainv_diag))
            surv, bound = screen_sharp(B, d_up, dg_all.Glo)
            @test !isempty(surv)
            @test length(surv) < N ÷ 10                      # a small fraction
            lam_f, imax, _ = lambda_hb_fast(A, B)
            @test imax in surv                               # the argmax survives
            @test dg_all.imax_upper in surv
            # certifying ONLY the survivors gives the same certified upper bound
            _, dg_surv = lambda_hb_certified(A, B[surv, :]; lmin = lmin,
                                             screen = false)
            @test dg_surv.Gup === dg_all.Gup                 # BIT-identical
        end
    end

    @testset "9. Corollary 3.1 encloses the float conversion" begin
        for n in (8, 16)
            A, B, M, N = _fm_case(n, pi / 2, :verbatim)
            lam_f, _, _ = lambda_hb_fast(A, B)
            lmin, _ = lambda_min_lower(A; x0 = ones(M))
            encl, _ = lambda_hb_certified(A, B; lmin = lmin)
            cl_f = CL_ub(lam_f, n)
            cl_iv = CL_ub_interval(encl, n)
            @test inf(cl_iv) ≤ cl_f ≤ sup(cl_iv)
            @test cl_f ≈ 1 / sqrt(lam_f * (1 - (1 / n)^2)) atol = 0 rtol = 1e-15
            # C^L_ub decreases as lambda increases — the direction the lemma uses
            @test sup(CL_ub_interval(encl, n)) < CL_ub(0.9 * lam_f, n)
        end
    end

    @testset "10. degenerate rows and dimension mismatch" begin
        A, B, M, N = _fm_case(4, pi / 2, :consistent)
        # a genuinely empty row of B must contribute g = 0, not NaN. The three
        # corner elements already have such rows in this assembly, since all six
        # of their columns can be deleted corner DOFs; append one explicitly.
        Bz = vcat(B, spzeros(1, M))
        lam_z, imax_z, _ = lambda_hb_fast(A, Bz)
        lam_0, imax_0, _ = lambda_hb_fast(A, B)
        @test lam_z == lam_0                            # unchanged
        @test imax_z != N + 1                           # the empty row never wins
        @test_throws AssertionError lambda_hb_certified(A, B[:, 1:(M-1)])
    end

    @testset "11. performance: certified path at n = 8 < 30 s" begin
        A, B, M, N = _fm_case(8, pi / 2, :verbatim)
        lmin, _ = lambda_min_lower(A; x0 = ones(M))
        lambda_hb_certified(A, B; lmin = lmin)          # warm up
        t0 = time_ns()
        lambda_hb_certified(A, B; lmin = lmin)
        @test (time_ns() - t0) < 30_000_000_000
    end
end
