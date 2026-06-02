# test/test_bernstein_multinomial.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. N = 0, α = (0,0,0,0) -> 1
#   2. N = 1: each unit vertex gives 1
#   3. N = 2: known values — (2,0,0,0)→1, (1,1,0,0)→2, (0,1,1,0)→2 etc.
#   4. N = 4 corners (4,0,0,0) etc. -> 1
#   5. N = 4 fully spread (1,1,1,1) -> 24
#   6. sum(α) ≠ N -> DomainError
#   7. α with negative entry -> DomainError
#   8. Vectorised matrix form preserves order
#   9. Performance: < 1 µs per scalar call.
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   For all α with |α| = N, C(N;α) = N! / (α₁! α₂! α₃! α₄!).
#   Sum over all α of C(N;α) equals 4^N (multinomial theorem).
#   Verified explicitly for N = 0..5.

using Test
using VFEM: bernstein_multinomial_3d, ijkl_list

@testset "bernstein_multinomial_3d" begin
    @testset "trivial cases" begin
        @test bernstein_multinomial_3d(0, (0, 0, 0, 0)) == 1
        @test bernstein_multinomial_3d(1, (1, 0, 0, 0)) == 1
        @test bernstein_multinomial_3d(1, (0, 1, 0, 0)) == 1
        @test bernstein_multinomial_3d(1, (0, 0, 1, 0)) == 1
        @test bernstein_multinomial_3d(1, (0, 0, 0, 1)) == 1
    end

    @testset "N = 2 known values" begin
        @test bernstein_multinomial_3d(2, (2, 0, 0, 0)) == 1
        @test bernstein_multinomial_3d(2, (1, 1, 0, 0)) == 2
        @test bernstein_multinomial_3d(2, (0, 1, 1, 0)) == 2
        @test bernstein_multinomial_3d(2, (1, 0, 1, 0)) == 2
    end

    @testset "N = 4 representative" begin
        @test bernstein_multinomial_3d(4, (4, 0, 0, 0)) == 1
        @test bernstein_multinomial_3d(4, (1, 1, 1, 1)) == 24       # 4!/1!1!1!1!
        @test bernstein_multinomial_3d(4, (2, 1, 1, 0)) == 12       # 24/(2)
        @test bernstein_multinomial_3d(4, (2, 2, 0, 0)) == 6        # 24/(2·2)
    end

    @testset "DomainError preconditions" begin
        @test_throws DomainError bernstein_multinomial_3d(2, (3, 0, 0, 0))     # |α| ≠ N
        @test_throws DomainError bernstein_multinomial_3d(3, (-1, 4, 0, 0))    # negative
        @test_throws DomainError bernstein_multinomial_3d(2, [1, 0, 0, 0])     # |α| ≠ N (Vec)
    end

    @testset "matrix form preserves row order" begin
        for N in 0:5
            L = ijkl_list(N)
            v = bernstein_multinomial_3d(N, L)
            @test length(v) == size(L, 1)
            for r in axes(L, 1)
                @test v[r] == bernstein_multinomial_3d(N,
                                                       (L[r, 1], L[r, 2], L[r, 3], L[r, 4]))
            end
        end
    end

    @testset "Mathematical contract: Σ_α C(N;α) = 4^N" begin
        # The multinomial theorem gives (1+1+1+1)^N = Σ_α C(N;α).
        for N in 0:5
            v = bernstein_multinomial_3d(N, ijkl_list(N))
            @test sum(v) == 4^N
        end
    end

    @testset "performance < 1 µs per scalar call" begin
        bernstein_multinomial_3d(4, (1, 1, 1, 1))
        N = 10_000
        t0 = time_ns()
        @inbounds for _ in 1:N
            bernstein_multinomial_3d(4, (1, 1, 1, 1))
        end
        tns = (time_ns() - t0) / N
        @test tns < 1_000
    end
end
