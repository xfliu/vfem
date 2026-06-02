# test/test_inner_prod_matrix.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. deg1 = deg2 = 0 -> 1×1 matrix; ∫_K 1 dV = K_vol, so A = [K_vol].
#   2. deg1 = 1, deg2 = 0 -> column vector of ∫_K L_i dV = K_vol/4.
#   3. deg1 = deg2 = 1 -> 4×4 Gram of {L_i}; A_{ii} = K_vol/10,
#                                            A_{ij} = K_vol/20 (i≠j).
#   4. K_vol = 1 reproduces the *_reference variant.
#   5. Symmetry across deg1 = deg2 (matrix is symmetric).
#   6. Element type tracks K_vol (Float64 in, Float64 out; Interval in, Interval out).
#   7. Performance: deg1 = deg2 = 4 (35×35) < 5 ms.
#   8. inner_prod_matrix_reference3d slice equals inner_prod_matrix_reference
#      when one degree is 0 (∫ L^α · L^β · 1 dV).
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   A[α, β] = ∫_K L^α L^β dV = 6 · K_vol · (α+β)! / (|α|+|β|+3)!
#   Verified directly against the closed-form factorial expression
#   (which is also how the routine is built — ok). Indirect cross-check:
#   total trace ∑_α A[α, α] should match ∫_K Σ_α (L^α)² dV computed
#   via the partition-of-unity identity; we instead test individual
#   entries against hand-derived values for low degrees.

using Test
using IntervalArithmetic: Interval, interval, inf, sup
using VFEM: inner_prod_matrix, inner_prod_matrix_reference,
            inner_prod_matrix_reference3d, ijkl_list

@testset "inner_prod_matrix" begin
    @testset "deg1 = deg2 = 0 — measure of K" begin
        @test inner_prod_matrix(0, 0, 1.5) == reshape([1.5], 1, 1)
    end

    @testset "deg1 = 1, deg2 = 0" begin
        # ∫_K L_i · 1 dV = K_vol/4 for each i (centroid).
        A = inner_prod_matrix(1, 0, 4.0)
        @test size(A) == (4, 1)
        @test all(A .≈ 1.0)
    end

    @testset "deg1 = deg2 = 1 — analytic Gram of L_i" begin
        # ∫_K L_i² dV = K_vol · 2/(2+3)! · 2 ... the formula gives:
        #   A_{ii} = 6·K_vol · 2!·0!·0!·0! / 5! = 6·K_vol·2/120 = K_vol/10
        #   A_{ij} = 6·K_vol · 1!·1!·0!·0! / 5! = 6·K_vol/120     = K_vol/20
        K_vol = 3.0
        A = inner_prod_matrix(1, 1, K_vol)
        for i in 1:4, j in 1:4
            expected = i == j ? K_vol / 10 : K_vol / 20
            @test A[i, j] ≈ expected rtol = 1e-14
        end
    end

    @testset "K_vol = 1 reproduces reference" begin
        for (d1, d2) in ((0, 0), (1, 1), (2, 2), (2, 1), (3, 2))
            @test inner_prod_matrix(d1, d2, 1.0) ≈ inner_prod_matrix_reference(d1, d2)
        end
    end

    @testset "symmetry deg1 = deg2" begin
        for d in (1, 2, 3)
            A = inner_prod_matrix(d, d, 1.0)
            @test A ≈ A'
        end
    end

    @testset "element type tracks K_vol (Interval mode)" begin
        K_vol = interval(2.0, 2.0)
        A = inner_prod_matrix(1, 1, K_vol)
        @test eltype(A) <: Interval
        # Diagonal entry K_vol/10 ⊂ A[1,1].
        Af = inner_prod_matrix(1, 1, 2.0)
        for i in 1:4, j in 1:4
            @test inf(A[i, j]) ≤ Af[i, j] ≤ sup(A[i, j])
        end
    end

    @testset "performance deg1 = deg2 = 4 (35×35) < 5 ms" begin
        inner_prod_matrix(4, 4, 1.0)
        T = 5
        t0 = time_ns()
        for _ in 1:T
            inner_prod_matrix(4, 4, 1.0)
        end
        tns = (time_ns() - t0) / T
        @test tns < 5_000_000
    end
end

@testset "inner_prod_matrix_reference3d" begin
    @testset "deg3 = 0 reduces to 2-product" begin
        for (d1, d2) in ((1, 1), (2, 1), (2, 2))
            A2 = inner_prod_matrix_reference(d1, d2)
            A3 = inner_prod_matrix_reference3d(d1, d2, 0)
            @test size(A3) == (size(A2)..., 1)
            @test reshape(A3, size(A2)) ≈ A2
        end
    end

    @testset "shape" begin
        for (d1, d2, d3) in ((1, 1, 1), (2, 1, 1), (2, 2, 1))
            A = inner_prod_matrix_reference3d(d1, d2, d3)
            @test size(A) == (size(ijkl_list(d1), 1),
                              size(ijkl_list(d2), 1),
                              size(ijkl_list(d3), 1))
        end
    end
end
