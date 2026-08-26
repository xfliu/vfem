# test/test_singular_potential_matrix_vertex_exact.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. N = 0 -> 1×1 matrix; equals the bare ∫_K 1/|x-P_s| dx.
#   2. MATLAB R2024a fixture cross-check at N = 2 (10×10 matrix, 55 entries).
#   3. Symmetry: T[i,j] == T[j,i].
#   4. Negative N -> DomainError.
#   5. Performance: N = 2 < 200 ms (the routine evaluates a degree-4
#      face-moment array internally).
#
# MATHEMATICAL CONTRACT (docs/testing-contract.md (c)):
#   T[i,j] = ∫_K B_i^N(x) · B_j^N(x) / |x - P_s| dx,
#   ordered by ijkl_list(N). Symmetric in (i, j).
#   For physical potential V(x) = -Z/|x-P_s|, the matrix is -Z·T.

using Test
using VFEM: singular_potential_matrix_vertex_exact, ijkl_list

const _SPM_LOCAL_NODES = [0.0 0.0 0.0;
                          1.0 0.0 0.0;
                          0.2 1.1 0.1;
                          0.1 0.3 0.9]

function _load_spm_fixture()
    path = joinpath(@__DIR__, "fixtures", "tet_vertex_singular_ref.txt")
    T_ref = Dict{NTuple{2, Int}, Float64}()
    for line in eachline(path)
        line = strip(line)
        (isempty(line) || startswith(line, "#")) && continue
        m = match(r"^T\[(\d+),(\d+)\]=(.+)$", line)
        m === nothing && continue
        i = parse(Int, m[1]); j = parse(Int, m[2])
        T_ref[(i, j)] = parse(Float64, m[3])
    end
    return T_ref
end

@testset "singular_potential_matrix_vertex_exact" begin
    @testset "MATLAB R2024a fixture cross-check (N = 2, 55 upper-tri entries)" begin
        T_ref = _load_spm_fixture()
        Tmat, info = singular_potential_matrix_vertex_exact(_SPM_LOCAL_NODES, 1, 2)
        @test size(Tmat) == (10, 10)
        @test info.degree == 2
        @test info.moment_degree == 4
        for ((i, j), expected) in T_ref
            @test Tmat[i, j] ≈ expected atol = 1e-12 rtol = 1e-12
        end
    end

    @testset "Symmetry T[i,j] == T[j,i]" begin
        Tmat, _ = singular_potential_matrix_vertex_exact(_SPM_LOCAL_NODES, 1, 2)
        @test Tmat ≈ Tmat'
    end

    @testset "1. N = 0 reduces to scalar bare integral" begin
        Tmat, info = singular_potential_matrix_vertex_exact(_SPM_LOCAL_NODES, 1, 0)
        @test size(Tmat) == (1, 1)
        @test info.degree == 0
        @test info.moment_degree == 0
    end

    @testset "Error preconditions" begin
        @test_throws DomainError singular_potential_matrix_vertex_exact(
            _SPM_LOCAL_NODES, 1, -1)
    end

    @testset "Performance N = 2 < 200 ms" begin
        singular_potential_matrix_vertex_exact(_SPM_LOCAL_NODES, 1, 2)
        T = 3
        t0 = time_ns()
        for _ in 1:T
            singular_potential_matrix_vertex_exact(_SPM_LOCAL_NODES, 1, 2)
        end
        tns = (time_ns() - t0) / T
        @test tns < 200_000_000
    end
end
