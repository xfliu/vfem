# test/test_ijkl_list.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. M = 0 -> single row (0,0,0,0)
#   2. M = 1 -> 4 rows, vertex DOFs in canonical order:
#                (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)
#   3. M = 2 -> 10 rows; corners first then edge-midpoint multi-indices
#   4. M = 5 -> 56 rows
#   5. row order matches MATLAB's lex-descending convention (load-bearing
#      contract — downstream caches depend on it)
#   6. each row sums to M
#   7. all rows are unique
#   8. negative M -> DomainError
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   Number of rows == simplex_dof(3, M); every (i,j,k,l) with
#   i+j+k+l = M and i,j,k,l ≥ 0 appears exactly once.

using Test
using VFEM: ijkl_list, simplex_dof

@testset "ijkl_list" begin
    @testset "M = 0" begin
        L = ijkl_list(0)
        @test size(L) == (1, 4)
        @test L[1, :] == [0, 0, 0, 0]
    end

    @testset "M = 1 — canonical vertex order" begin
        L = ijkl_list(1)
        @test size(L) == (4, 4)
        @test L[1, :] == [1, 0, 0, 0]
        @test L[2, :] == [0, 1, 0, 0]
        @test L[3, :] == [0, 0, 1, 0]
        @test L[4, :] == [0, 0, 0, 1]
    end

    @testset "M = 2 — first row is (2,0,0,0), last is (0,0,0,2)" begin
        L = ijkl_list(2)
        @test size(L) == (10, 4)
        @test L[1, :] == [2, 0, 0, 0]
        @test L[end, :] == [0, 0, 0, 2]
    end

    @testset "row count matches simplex_dof(3, M)" begin
        for M in 0:6
            @test size(ijkl_list(M), 1) == simplex_dof(3, M)
        end
    end

    @testset "every row sums to M and entries are nonneg" begin
        for M in 0:6
            L = ijkl_list(M)
            for r in axes(L, 1)
                @test sum(L[r, :]) == M
                @test all(L[r, :] .≥ 0)
            end
        end
    end

    @testset "all rows are unique" begin
        for M in 0:6
            L = ijkl_list(M)
            tuples = [tuple(L[r, :]...) for r in axes(L, 1)]
            @test length(Set(tuples)) == size(L, 1)
        end
    end

    @testset "lex-descending row order (MATLAB convention)" begin
        # MATLAB get_IJKL.m enumerates with `for i = M:-1:0; for j = M-i:-1:0; ...`.
        # Verify by reproducing the reference list with a hand-written loop
        # and comparing to ijkl_list output element by element.
        for M in 0:5
            ref = Vector{NTuple{4, Int}}()
            for i in M:-1:0, j in (M - i):-1:0, k in (M - i - j):-1:0
                push!(ref, (i, j, k, M - i - j - k))
            end
            L = ijkl_list(M)
            for r in eachindex(ref)
                @test (L[r, 1], L[r, 2], L[r, 3], L[r, 4]) == ref[r]
            end
        end
    end

    @testset "DomainError for negative M" begin
        @test_throws DomainError ijkl_list(-1)
    end
end
