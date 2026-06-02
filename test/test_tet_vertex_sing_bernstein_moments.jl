# test/test_tet_vertex_sing_bernstein_moments.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. N = 0 returns a single moment ∫_K 1/|x-P_s| dx (= face_jac × beta).
#   2. MATLAB R2024a fixture cross-check at N = 2.
#   3. Sum invariant: Σ_k C(N;γ_k)·1 / multinomial · S[k] equals the
#      direct ∫_K 1/|x-P_s| dx result, since Σ_k B_γ^N(L) = 1.
#      Concretely: S_total = ∫_K 1/|x-P_s| dx is independent of N.
#   4. Singular vertex out of 1:4 -> DomainError.
#   5. LocalNodes wrong shape -> DimensionMismatch.
#   6. Negative N -> DomainError.
#   7. Performance: N = 4 < 100 ms.
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   S[k] = ∫_K B_γ_k^N(x) / |x - P_s| dx where γ_k = ijkl_list(N)[k, :]
#   and P_s = LocalNodes[singular_vertex, :].

using Test
using VFEM: tet_vertex_sing_bernstein_moments, ijkl_list

const _SV_LOCAL_NODES = [0.0 0.0 0.0;
                         1.0 0.0 0.0;
                         0.2 1.1 0.1;
                         0.1 0.3 0.9]

# Load only the moment+det6 sections from the fixture.
function _load_tvsm_fixture()
    path = joinpath(@__DIR__, "fixtures", "tet_vertex_singular_ref.txt")
    det6_ref = NaN
    vol_ref = NaN
    S_ref = Dict{NTuple{4, Int}, Float64}()
    for line in eachline(path)
        line = strip(line)
        (isempty(line) || startswith(line, "#")) && continue
        if startswith(line, "det6=")
            det6_ref = parse(Float64, line[6:end])
        elseif startswith(line, "volume=")
            vol_ref = parse(Float64, line[8:end])
        else
            m = match(r"^S2\[(\d+),(\d+),(\d+),(\d+)\]=(.+)$", line)
            m === nothing && continue
            key = (parse(Int, m[1]), parse(Int, m[2]),
                   parse(Int, m[3]), parse(Int, m[4]))
            S_ref[key] = parse(Float64, m[5])
        end
    end
    return det6_ref, vol_ref, S_ref
end

@testset "tet_vertex_sing_bernstein_moments" begin
    @testset "MATLAB R2024a fixture cross-check (N = 2)" begin
        det6_ref, vol_ref, S_ref = _load_tvsm_fixture()
        S, info = tet_vertex_sing_bernstein_moments(_SV_LOCAL_NODES, 1, 2)
        @test info.det6 ≈ det6_ref atol = 1e-13 rtol = 1e-13
        @test info.volume ≈ vol_ref atol = 1e-13 rtol = 1e-13
        list = info.ijkl
        for k in axes(list, 1)
            key = (list[k, 1], list[k, 2], list[k, 3], list[k, 4])
            @test S[k] ≈ S_ref[key] atol = 1e-12 rtol = 1e-12
        end
    end

    @testset "1. N = 0 returns scalar bare integral" begin
        S, info = tet_vertex_sing_bernstein_moments(_SV_LOCAL_NODES, 1, 0)
        @test length(S) == 1
        # B_γ=(0,0,0,0)^0 = 1, so S[1] = ∫_K 1/|x-P_s| dx.
        @test info.degree == 0
    end

    @testset "Mathematical contract: total invariant under N" begin
        # S_total(N) = Σ_k B_γ^N coefficient = 1 ⇒ Σ_k S[k] (where the
        # coefficient vector is c_k = 1/C(N;γ_k) so that Σ_k c_k B_γ^N = 1)
        # equals the bare integral. Easier check: compute the total
        # ∫_K 1/|x-P_s| dx via N = 0 and via N = 2 by reconstruction.
        S0, _ = tet_vertex_sing_bernstein_moments(_SV_LOCAL_NODES, 1, 0)
        # For N = 2, Σ_k (1/C(N;γ_k)) · S[k] · C(N;γ_k) = Σ_k S[k]·... actually
        # the partition-of-unity identity is Σ_k B_γ^N = 1, so writing
        # 1 = Σ_k 1·B_γ^N as a Bernstein expansion and integrating with the
        # 1/|x-P_s| weight, ∫ 1/|x-P_s| dx = Σ_k ∫ B_γ^N/|x-P_s| dx = Σ_k S[k]
        # — but only when the coefficient vector is c_k = 1 (representing
        # the constant 1 in Bernstein form, which DOES use c_k = 1 for all k).
        S2, _ = tet_vertex_sing_bernstein_moments(_SV_LOCAL_NODES, 1, 2)
        @test sum(S2) ≈ S0[1] atol = 1e-12 rtol = 1e-12
    end

    @testset "Error preconditions" begin
        @test_throws DomainError tet_vertex_sing_bernstein_moments(
            _SV_LOCAL_NODES, 0, 2)
        @test_throws DomainError tet_vertex_sing_bernstein_moments(
            _SV_LOCAL_NODES, 5, 2)
        @test_throws DimensionMismatch tet_vertex_sing_bernstein_moments(
            zeros(3, 3), 1, 2)
        @test_throws DomainError tet_vertex_sing_bernstein_moments(
            _SV_LOCAL_NODES, 1, -1)
    end

    @testset "Performance N = 4 < 100 ms" begin
        tet_vertex_sing_bernstein_moments(_SV_LOCAL_NODES, 1, 4)
        T = 5
        t0 = time_ns()
        for _ in 1:T
            tet_vertex_sing_bernstein_moments(_SV_LOCAL_NODES, 1, 4)
        end
        tns = (time_ns() - t0) / T
        @test tns < 100_000_000
    end
end
