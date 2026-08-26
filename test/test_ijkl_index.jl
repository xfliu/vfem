# test/test_ijkl_index.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. M = 0 -> only (0,0,0,0) at idx 1
#   2. Round-trip identity for every M in 0..5: list[k,:] -> idx -> k
#   3. ijkl_index_map equals ijkl_index for every entry
#   4. lookup of a multi-index NOT in the list -> KeyError
#   5. accept tuple input
#   6. accept Vector input
#   7. ijkl_index_map size and value count
#   8. Performance: hashmap lookup < 200 ns; linear scan < 1 µs at M=5

using Test
using VFEM: ijkl_list, ijkl_index, ijkl_index_map

@testset "ijkl_index / ijkl_index_map" begin
    @testset "M = 0 trivial case" begin
        L = ijkl_list(0)
        @test ijkl_index(L, (0, 0, 0, 0)) == 1
        m = ijkl_index_map(0)
        @test m[(0, 0, 0, 0)] == 1
        @test length(m) == 1
    end

    @testset "round-trip identity, every M in 0..5" begin
        for M in 0:5
            L = ijkl_list(M)
            for r in axes(L, 1)
                @test ijkl_index(L, (L[r, 1], L[r, 2], L[r, 3], L[r, 4])) == r
            end
        end
    end

    @testset "ijkl_index_map agrees with linear scan" begin
        for M in 0:5
            L = ijkl_list(M)
            m = ijkl_index_map(M)
            for r in axes(L, 1)
                key = (L[r, 1], L[r, 2], L[r, 3], L[r, 4])
                @test m[key] == ijkl_index(L, key)
            end
        end
    end

    @testset "tuple vs Vector inputs equivalent" begin
        L = ijkl_list(3)
        @test ijkl_index(L, (2, 1, 0, 0)) == ijkl_index(L, [2, 1, 0, 0])
    end

    @testset "missing multi-index throws KeyError" begin
        L = ijkl_list(2)
        @test_throws KeyError ijkl_index(L, (3, 0, 0, 0))   # |α| ≠ 2
        @test_throws KeyError ijkl_index(L, (1, 1, 0, 1))   # |α| = 3, not in M=2 list
    end

    @testset "ijkl_index_map size matches ijkl_list rows" begin
        for M in 0:6
            @test length(ijkl_index_map(M)) == size(ijkl_list(M), 1)
        end
    end

    @testset "performance: hashmap lookup < 200 ns" begin
        m = ijkl_index_map(5)
        keys_sample = [(2, 2, 1, 0), (5, 0, 0, 0), (0, 0, 0, 5), (1, 1, 1, 2)]
        for k in keys_sample
            @test haskey(m, k)
        end
        # Warm.
        for _ in 1:5
            for k in keys_sample
                m[k]
            end
        end
        N = 10_000
        t0 = time_ns()
        @inbounds for _ in 1:N
            for k in keys_sample
                m[k]
            end
        end
        tns = (time_ns() - t0) / (N * length(keys_sample))
        @test tns < 200
    end
end
