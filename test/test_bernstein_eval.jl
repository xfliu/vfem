# test/test_bernstein_eval.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. 1D: B_0^0(x) = 1 (degree-0 constant)
#   2. 1D: B_i^n endpoint values — B_0^n(0) = 1, B_n^n(1) = 1, all others 0.
#   3. 1D: partition of unity Σ_i B_i^n(x) = 1 for all x.
#   4. 1D: array-input form returns array.
#   5. 1D: i out of range -> DomainError.
#   6. 3D: vertex value — B_α^N at the α-vertex L = e_k equals C(N;α)
#      iff α has all weight at vertex k, else 0.
#   7. 3D: partition of unity — ∑_α B_α^N(L) = 1 (since (L₁+L₂+L₃+L₄)^N = 1).
#   8. 3D: c length mismatch -> DimensionMismatch.
#   9. Performance: 3D evaluation at 100 points, M=4 < 1 ms.
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   B_i^n(x) = C(n,i) x^i (1-x)^{n-i}.
#   B_α^N(L) = C(N;α) L^α with α a tetrahedral multi-index.
#   Σ_α B_α^N(L) = (L₁+L₂+L₃+L₄)^N = 1 by the multinomial identity.

using Test, Random
using VFEM: bernstein_eval, bernstein3d_eval_bary, ijkl_list,
            bernstein_multinomial_3d

@testset "bernstein_eval (1D)" begin
    Random.seed!(20260507)

    @testset "B_0^0 = 1" begin
        @test bernstein_eval(0, 0, 0.3) == 1.0
        @test bernstein_eval(0, 0, 0.0) == 1.0
    end

    @testset "endpoint values" begin
        for n in 0:5, i in 0:n
            v0 = bernstein_eval(n, i, 0.0)
            v1 = bernstein_eval(n, i, 1.0)
            @test v0 == (i == 0 ? 1.0 : 0.0)
            @test v1 == (i == n ? 1.0 : 0.0)
        end
    end

    @testset "partition of unity" begin
        for x in (0.0, 0.25, 0.5, 0.75, 1.0, rand())
            for n in 0:6
                s = sum(bernstein_eval(n, i, x) for i in 0:n)
                @test s ≈ 1.0 atol = 1e-12
            end
        end
    end

    @testset "array input" begin
        x = [0.0, 0.25, 0.5, 0.75, 1.0]
        v = bernstein_eval(2, 1, x)
        @test length(v) == length(x)
        @test v ≈ [0.0, 2 * 0.25 * 0.75, 0.5, 2 * 0.75 * 0.25, 0.0]
    end

    @testset "DomainError preconditions" begin
        @test_throws DomainError bernstein_eval(3, -1, 0.5)
        @test_throws DomainError bernstein_eval(3, 4, 0.5)
        @test_throws DomainError bernstein_eval(-1, 0, 0.5)
    end
end

@testset "bernstein3d_eval_bary" begin
    Random.seed!(20260508)

    @testset "vertex evaluation" begin
        # At barycentric vertex L = e_k, B_α^N(L) is C(N;α) if α concentrates
        # at vertex k, else 0. With c = δ_β (Bernstein coeff vector that
        # picks out one basis function), bernstein3d_eval_bary at L = e_k
        # equals C(N;β) · 1[β concentrates at k].
        for N in (1, 2, 3, 4)
            L = ijkl_list(N)
            multi = bernstein_multinomial_3d(N, L)
            for vidx in 1:4
                ev = zeros(4); ev[vidx] = 1.0
                for k in axes(L, 1)
                    c = zeros(size(L, 1)); c[k] = 1.0
                    val = bernstein3d_eval_bary(N, c, ev)
                    if L[k, vidx] == N
                        @test val ≈ multi[k]
                    else
                        @test val ≈ 0.0
                    end
                end
            end
        end
    end

    @testset "partition of unity Σ_α B_α^N(L) = 1" begin
        # With c[k] = 1 for all k, the sum Σ_k 1·B_k^N(L) does NOT equal 1
        # — it equals (L₁+L₂+L₃+L₄)^N = 1 only when we include the
        # multinomial factor. Our `bernstein3d_eval_bary` ALREADY includes
        # the C(N;α) factor via `multi[k]`, so we should pass `c = ones`
        # divided by `multi` to get the partition-of-unity test... actually
        # easier: bernstein3d_eval_bary with the *coefficient* vector that
        # represents the constant function 1 is c[k] = 1 for ALL k (one
        # coeff per Bernstein basis function), and the sum then evaluates
        # Σ_α B_α^N(L) = 1 (since Σ_α C(N;α) L^α = (Σ L)^N = 1).
        for N in 0:5
            L_list = ijkl_list(N)
            c = ones(size(L_list, 1))
            for L in (
                    [1.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.4, 0.3, 0.2, 0.1],
                )
                @test bernstein3d_eval_bary(N, c, L) ≈ 1.0 atol = 1e-12
            end
        end
    end

    @testset "matrix-of-points input" begin
        N = 2
        c = ones(size(ijkl_list(N), 1))
        Lm = [1.0 0.0 0.0 0.0;
              0.25 0.25 0.25 0.25;
              0.4 0.3 0.2 0.1]
        v = bernstein3d_eval_bary(N, c, Lm)
        @test length(v) == 3
        @test all(abs.(v .- 1.0) .< 1e-12)
    end

    @testset "DimensionMismatch on bad c length" begin
        @test_throws DimensionMismatch bernstein3d_eval_bary(2, ones(3), [0.25, 0.25, 0.25, 0.25])
    end

    @testset "performance: 100 points, M=4 < 1 ms" begin
        N = 4
        c = randn(size(ijkl_list(N), 1))
        Lm = rand(100, 4); Lm ./= sum(Lm, dims=2)
        bernstein3d_eval_bary(N, c, Lm)
        T = 50
        t0 = time_ns()
        for _ in 1:T
            bernstein3d_eval_bary(N, c, Lm)
        end
        tns = (time_ns() - t0) / T
        @test tns < 1_000_000
    end
end
