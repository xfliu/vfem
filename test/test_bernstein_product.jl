# test/test_bernstein_product.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. Constant × constant: B^0 · B^0 = B^0 with c_new = c1·c2.
#   2. Bernstein basis function × Bernstein basis function (single pair).
#   3. Polynomial × constant 1: result must equal the original polynomial.
#   4. Round-trip: evaluate p, q, p·q at random barycentric points and
#      verify (p·q)(L) ≈ p(L) · q(L).
#   5. Length mismatch -> DimensionMismatch.
#   6. Negative degrees -> DomainError.
#   7. Performance: N=2 × N=2 (DOF 10 × 10) < 100 µs.
#
# MATHEMATICAL CONTRACT (docs/testing-contract.md (c)):
#   For p = Σ_α c1_α B_α^N and q = Σ_β c2_β B_β^M, the returned d_γ
#   represents p·q in degree N+M. We verify by evaluating both forms
#   at random points L (4 barycentric coords summing to 1).
#
# ADVERSARIAL TEST — multinomial-factor bug (Validation/RESULTS.md):
#   Construct an adversarial coefficient combination that gives the
#   right answer ONLY if C(N;α)·C(M;β)/C(N+M;γ) is applied. The naive
#   "add multi-indices" implementation (bernstein_product_monomial)
#   gives a different answer; assert that our exact form differs from
#   the naive form on this input — exactly the bug class fixed in
#   Validation/RESULTS.md.

using Test, Random
using VFEM: bernstein_product_exact, bernstein_product_monomial,
            bernstein3d_eval_bary, ijkl_list, simplex_dof

# Build a Bernstein-coefficient vector from a multi-index δ_α: returns
# a vector of zeros except 1 at the index of α. Used for single-basis
# product checks.
function _bernstein_delta(N, α::NTuple{4, Int})
    L = ijkl_list(N)
    c = zeros(Float64, size(L, 1))
    for r in axes(L, 1)
        if (L[r, 1], L[r, 2], L[r, 3], L[r, 4]) == α
            c[r] = 1.0
            return c
        end
    end
    error("α=$α not in ijkl_list($N)")
end

@testset "bernstein_product_exact" begin
    Random.seed!(20260509)

    @testset "1. constant × constant" begin
        new_c, _ = bernstein_product_exact(0, 0, [3.0], [4.0])
        @test new_c == [12.0]
    end

    @testset "3. polynomial × 1 reproduces input" begin
        N = 3
        c = randn(simplex_dof(3, N))
        new_c, _ = bernstein_product_exact(N, 0, c, [1.0])
        @test new_c ≈ c
    end

    @testset "Mathematical contract: (p·q)(L) ≈ p(L) · q(L) at random L" begin
        for (N, M) in ((1, 1), (2, 1), (2, 2), (3, 2))
            c1 = randn(simplex_dof(3, N))
            c2 = randn(simplex_dof(3, M))
            d, _ = bernstein_product_exact(N, M, c1, c2)
            for _ in 1:10
                L = rand(4); L ./= sum(L)
                pL = bernstein3d_eval_bary(N, c1, L)
                qL = bernstein3d_eval_bary(M, c2, L)
                pqL = bernstein3d_eval_bary(N + M, d, L)
                @test pqL ≈ pL * qL atol = 1e-10 rtol = 1e-10
            end
        end
    end

    @testset "5–6. error preconditions" begin
        @test_throws DimensionMismatch bernstein_product_exact(1, 1, ones(3), ones(4))
        @test_throws DomainError bernstein_product_exact(-1, 1, [1.0], ones(4))
    end

    @testset "ADVERSARIAL: detect missing multinomial factor" begin
        # If a future implementation drops the C(N;α)·C(M;β)/C(N+M;γ)
        # factor and accidentally returns the monomial-form product,
        # the result is wrong by exactly this ratio. Pick basis pair
        # α = (1,1,0,0), β = (0,0,1,1) so γ = (1,1,1,1):
        #   C(2; α) = 2,  C(2; β) = 2,  C(4; γ) = 24
        #   ratio   = 4/24 = 1/6
        # Bernstein product gives d_γ = 1/6, monomial product gives 1.
        N, M = 2, 2
        c1 = _bernstein_delta(N, (1, 1, 0, 0))
        c2 = _bernstein_delta(M, (0, 0, 1, 1))
        d_exact, _   = bernstein_product_exact(N, M, c1, c2)
        d_monomial   = bernstein_product_monomial(N, M, c1, c2)

        # Locate γ = (1,1,1,1) in ijkl_list(N+M).
        L4 = ijkl_list(N + M)
        γ_idx = findfirst(r -> (L4[r, 1], L4[r, 2], L4[r, 3], L4[r, 4]) == (1, 1, 1, 1),
                          axes(L4, 1))
        @test d_exact[γ_idx]    ≈ 1 / 6 rtol = 1e-14
        @test d_monomial[γ_idx] == 1
        # Critical: the two forms MUST disagree on this entry — that's
        # the fingerprint of the multinomial bug.
        @test d_exact[γ_idx] != d_monomial[γ_idx]
    end

    @testset "performance N=2 × N=2 < 100 µs" begin
        c1 = randn(10); c2 = randn(10)
        bernstein_product_exact(2, 2, c1, c2)
        T = 200
        t0 = time_ns()
        @inbounds for _ in 1:T
            bernstein_product_exact(2, 2, c1, c2)
        end
        tns = (time_ns() - t0) / T
        @test tns < 100_000
    end
end

@testset "bernstein_product_monomial (no multinomial)" begin
    Random.seed!(20260510)

    @testset "round-trip identity in monomial coords" begin
        # If c1 and c2 are MONOMIAL coefficients of p, q (so p = Σ c1_α L^α),
        # then p·q has monomial coefficients (c1 ⊗ c2) summed by index.
        # Verify by evaluating L^α explicitly at random points.
        for (N, M) in ((1, 1), (2, 2))
            ijkl1 = ijkl_list(N); ijkl2 = ijkl_list(M)
            c1 = randn(size(ijkl1, 1)); c2 = randn(size(ijkl2, 1))
            new_c = bernstein_product_monomial(N, M, c1, c2)
            ijkl_new = ijkl_list(N + M)
            for _ in 1:10
                L = rand(4); L ./= sum(L)
                pL = sum(c1[i] * L[1]^ijkl1[i, 1] * L[2]^ijkl1[i, 2] *
                                 L[3]^ijkl1[i, 3] * L[4]^ijkl1[i, 4]
                         for i in axes(ijkl1, 1))
                qL = sum(c2[j] * L[1]^ijkl2[j, 1] * L[2]^ijkl2[j, 2] *
                                 L[3]^ijkl2[j, 3] * L[4]^ijkl2[j, 4]
                         for j in axes(ijkl2, 1))
                pqL = sum(new_c[k] * L[1]^ijkl_new[k, 1] * L[2]^ijkl_new[k, 2] *
                                     L[3]^ijkl_new[k, 3] * L[4]^ijkl_new[k, 4]
                          for k in axes(ijkl_new, 1))
                @test pqL ≈ pL * qL atol = 1e-10 rtol = 1e-10
            end
        end
    end
end
