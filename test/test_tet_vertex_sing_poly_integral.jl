# test/test_tet_vertex_sing_poly_integral.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. p = q = 1 (constant) reduces to bare ∫_K 1/|x-P_s| dx.
#   2. MATLAB fixture: cp = δ_1 + 0.5·δ_5,  cq = δ_3 - 0.25·δ_8 at N = 2.
#   3. Bilinearity: poly_integral(αp, q) == α·poly_integral(p, q).
#   4. Symmetry: poly_integral(p, q) == poly_integral(q, p).
#   5. Length mismatch -> DimensionMismatch.
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   value = ∫_K p(x) · q(x) / |x - P_s| dx
#   where p, q are tetrahedral Bernstein polynomials given by
#   coefficient vectors `coeff_p`, `coeff_q` ordered by ijkl_list.

using Test
using VFEM: tet_vertex_sing_poly_integral, ijkl_list

const _PI_LOCAL_NODES = [0.0 0.0 0.0;
                         1.0 0.0 0.0;
                         0.2 1.1 0.1;
                         0.1 0.3 0.9]

function _load_poly_int_fixture()
    path = joinpath(@__DIR__, "fixtures", "tet_vertex_singular_ref.txt")
    for line in eachline(path)
        line = strip(line)
        if startswith(line, "poly_int=")
            return parse(Float64, line[10:end])
        end
    end
    return NaN
end

@testset "tet_vertex_sing_poly_integral" begin
    @testset "MATLAB fixture: cp/cq at N = 2" begin
        ref = _load_poly_int_fixture()
        N = 2
        DegK = size(ijkl_list(N), 1)
        cp = zeros(DegK); cp[1] = 1.0; cp[5] = 0.5
        cq = zeros(DegK); cq[3] = 1.0; cq[8] = -0.25
        v = tet_vertex_sing_poly_integral(_PI_LOCAL_NODES, 1, N, cp, N, cq)
        @test v ≈ ref atol = 1e-12 rtol = 1e-12
    end

    @testset "1. constant p, q reduces to scalar bare integral" begin
        # In Bernstein basis, p ≡ 1 has coeff vector all-ones at each
        # degree. Then ∫_K 1·1 / |x-P_s| dx is the bare integral.
        N = 2
        DegK = size(ijkl_list(N), 1)
        cp = ones(DegK)
        cq = ones(DegK)
        v = tet_vertex_sing_poly_integral(_PI_LOCAL_NODES, 1, N, cp, N, cq)
        # Equivalent: same with N = 0.
        v0 = tet_vertex_sing_poly_integral(_PI_LOCAL_NODES, 1, 0, [1.0], 0, [1.0])
        @test v ≈ v0 atol = 1e-12 rtol = 1e-12
    end

    @testset "Bilinearity in p" begin
        N = 2
        DegK = size(ijkl_list(N), 1)
        cp = randn(DegK); cq = randn(DegK)
        v  = tet_vertex_sing_poly_integral(_PI_LOCAL_NODES, 1, N, cp, N, cq)
        v2 = tet_vertex_sing_poly_integral(_PI_LOCAL_NODES, 1, N, 2 * cp, N, cq)
        @test v2 ≈ 2 * v atol = 1e-12 rtol = 1e-12
    end

    @testset "Symmetry: p ↔ q" begin
        N = 2
        DegK = size(ijkl_list(N), 1)
        cp = randn(DegK); cq = randn(DegK)
        a = tet_vertex_sing_poly_integral(_PI_LOCAL_NODES, 1, N, cp, N, cq)
        b = tet_vertex_sing_poly_integral(_PI_LOCAL_NODES, 1, N, cq, N, cp)
        @test a ≈ b atol = 1e-12 rtol = 1e-12
    end

    @testset "Length mismatch propagates" begin
        @test_throws DimensionMismatch tet_vertex_sing_poly_integral(
            _PI_LOCAL_NODES, 1, 2, ones(3), 2, ones(10))
    end
end
