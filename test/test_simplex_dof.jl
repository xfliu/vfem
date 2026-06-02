# test/test_simplex_dof.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. n = 0 (point) -> always 1 (constant only)
#   2. M = 0 -> always 1 (constant)
#   3. n = 1 (interval), M up to 5: M+1
#   4. n = 2 (triangle), M up to 5: (M+1)(M+2)/2
#   5. n = 3 (tetrahedron), M up to 5: (M+1)(M+2)(M+3)/6 — primary use
#   6. negative n -> DomainError
#   7. negative M -> DomainError
#   8. Performance: < 1 µs per call.

using Test
using VFEM: simplex_dof

@testset "simplex_dof" begin
    @testset "n = 0 (point)" begin
        for M in 0:5
            @test simplex_dof(0, M) == 1
        end
    end
    @testset "M = 0" begin
        for n in 0:5
            @test simplex_dof(n, 0) == 1
        end
    end
    @testset "n = 1 (interval)" begin
        for M in 0:5
            @test simplex_dof(1, M) == M + 1
        end
    end
    @testset "n = 2 (triangle)" begin
        for M in 0:5
            @test simplex_dof(2, M) == ((M + 1) * (M + 2)) ÷ 2
        end
    end
    @testset "n = 3 (tetrahedron) — primary use" begin
        # Hard-coded reference values for sanity.
        @test simplex_dof(3, 0) == 1
        @test simplex_dof(3, 1) == 4
        @test simplex_dof(3, 2) == 10
        @test simplex_dof(3, 3) == 20
        @test simplex_dof(3, 4) == 35
        @test simplex_dof(3, 5) == 56
    end
    @testset "DomainError preconditions" begin
        @test_throws DomainError simplex_dof(-1, 2)
        @test_throws DomainError simplex_dof(3, -1)
    end
    @testset "performance < 1 µs" begin
        simplex_dof(3, 5)
        N = 10_000
        t0 = time_ns()
        @inbounds for _ in 1:N
            simplex_dof(3, 5)
        end
        tns = (time_ns() - t0) / N
        @test tns < 1_000
    end
end
