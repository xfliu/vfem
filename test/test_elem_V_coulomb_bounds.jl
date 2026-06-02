# test/test_elem_V_coulomb_bounds.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. cube_r1, single nucleus at the origin (Z = 1): origin lies on a
#      vertex shared by several elements; one element on this fixture
#      has the origin as a vertex (so d_min = 0 and V_bar = −∞ for
#      that element). Per-element V_bar (finite) and V_hat values
#      match MATLAB to ≤ 1e-10 for the first 6 elements.
#   2. cube_r1, single nucleus at (0.1, 0.1, 0.1): one interior
#      element contains this center (V_bar = −∞ there); rest match
#      MATLAB.
#   3. cube_r1, single nucleus at (10, 10, 10): far outside, no
#      singularities; min/max of V_bar/V_hat match MATLAB.
#   4. V_bar ≤ V_hat element-wise (including the −∞ entries).
#   5. Multi-center: two nuclei → V_bar/V_hat sum over centers; check
#      additivity by comparing single-center vs two-identical-centers
#      (latter doubles the bound).
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   For V(x) = − Σ_c Z_c / |x − c|, V is monotone in |x − c|. The
#   element max/min of V is therefore at points where |x − c| is
#   min/max over K. The max distance is always at a vertex (|·−c| is
#   convex; the convex hull of the vertices = K). The min distance is
#   the closest projection of c onto K (vertex / edge / face / 0 if
#   c ∈ K).

using Test
using VFEM: mesh_load_from_folder, elem_V_coulomb_bounds, CoulombInfo

const _CB_DIR = joinpath(@__DIR__, "fixtures", "cube_r1")

function _load_cb_fixture()
    cases = Dict{String, Dict{String, Any}}()
    cur = ""
    for line in eachline(joinpath(_CB_DIR, "coulomb_bounds_ref.txt"))
        line = strip(line)
        isempty(line) && continue
        if (mm = match(r"^case=(.+)$", line)) !== nothing
            cur = mm[1]
            cases[cur] = Dict{String, Any}()
            continue
        end
        if (mm = match(r"^([A-Za-z_0-9]+)=([+-]?\d+\.\d+e[+-]?\d+)$", line)) !== nothing
            target = isempty(cur) ? get!(cases, "_meta", Dict{String, Any}()) : cases[cur]
            target[mm[1]] = parse(Float64, mm[2])
        elseif (mm = match(r"^([A-Za-z_0-9]+)=(\d+)$", line)) !== nothing
            target = isempty(cur) ? get!(cases, "_meta", Dict{String, Any}()) : cases[cur]
            target[mm[1]] = parse(Int, mm[2])
        elseif (mm = match(r"^([A-Za-z_0-9]+)=-Inf$", line)) !== nothing
            cases[cur][mm[1]] = -Inf
        end
    end
    return cases
end

@testset "elem_V_coulomb_bounds" begin
    cases = _load_cb_fixture()
    m = mesh_load_from_folder(_CB_DIR)

    @testset "1. nucleus at origin (vertex of cube_r1) matches MATLAB" begin
        info = CoulombInfo([0.0 0.0 0.0], [1.0])
        V_bar, V_hat = elem_V_coulomb_bounds(m, info)
        ref = cases["H_origin"]
        @test sum(isinf, V_bar) == ref["n_singular_bar"]
        finite_bar = V_bar[isfinite.(V_bar)]
        @test minimum(finite_bar) ≈ ref["V_bar_min_finite"] atol = 1e-10 rtol = 1e-10
        @test maximum(finite_bar) ≈ ref["V_bar_max_finite"] atol = 1e-10 rtol = 1e-10
        @test minimum(V_hat) ≈ ref["V_hat_min"] atol = 1e-10 rtol = 1e-10
        @test maximum(V_hat) ≈ ref["V_hat_max"] atol = 1e-10 rtol = 1e-10
        for k in 1:6
            if haskey(ref, "V_bar_$(k)_inf")
                @test V_bar[k] == -Inf
            else
                @test V_bar[k] ≈ ref["V_bar_$k"] atol = 1e-10 rtol = 1e-10
            end
            @test V_hat[k] ≈ ref["V_hat_$k"] atol = 1e-10 rtol = 1e-10
        end
    end

    @testset "2. nucleus at (0.1,0.1,0.1) matches MATLAB" begin
        info = CoulombInfo([0.1 0.1 0.1], [1.0])
        V_bar, V_hat = elem_V_coulomb_bounds(m, info)
        ref = cases["offset"]
        # min(V_bar) is -Inf when a singularity lies inside an element.
        @test minimum(V_bar) == -Inf
        finite = V_bar[isfinite.(V_bar)]
        @test maximum(finite) ≈ ref["V_bar_max"] atol = 1e-10 rtol = 1e-10
        @test minimum(V_hat) ≈ ref["V_hat_min"] atol = 1e-10 rtol = 1e-10
        @test maximum(V_hat) ≈ ref["V_hat_max"] atol = 1e-10 rtol = 1e-10
        for k in 1:6
            if ref["V_bar_$k"] == -Inf
                @test V_bar[k] == -Inf
            else
                @test V_bar[k] ≈ ref["V_bar_$k"] atol = 1e-10 rtol = 1e-10
            end
            @test V_hat[k] ≈ ref["V_hat_$k"] atol = 1e-10 rtol = 1e-10
        end
    end

    @testset "3. nucleus far outside cube matches MATLAB" begin
        info = CoulombInfo([10.0 10.0 10.0], [1.0])
        V_bar, V_hat = elem_V_coulomb_bounds(m, info)
        ref = cases["far"]
        @test all(isfinite, V_bar)
        @test minimum(V_bar) ≈ ref["V_bar_min"] atol = 1e-10 rtol = 1e-10
        @test maximum(V_bar) ≈ ref["V_bar_max"] atol = 1e-10 rtol = 1e-10
        @test minimum(V_hat) ≈ ref["V_hat_min"] atol = 1e-10 rtol = 1e-10
        @test maximum(V_hat) ≈ ref["V_hat_max"] atol = 1e-10 rtol = 1e-10
    end

    @testset "4. V_bar ≤ V_hat element-wise" begin
        info = CoulombInfo([0.5 0.5 0.5], [1.0])
        V_bar, V_hat = elem_V_coulomb_bounds(m, info)
        for e in 1:m.NumElt
            @test V_bar[e] ≤ V_hat[e]
        end
    end

    @testset "5. multi-center is sum over centers" begin
        info1 = CoulombInfo([2.0 2.0 2.0], [1.0])
        info2 = CoulombInfo([2.0 2.0 2.0; 2.0 2.0 2.0], [1.0, 1.0])
        Vb1, Vh1 = elem_V_coulomb_bounds(m, info1)
        Vb2, Vh2 = elem_V_coulomb_bounds(m, info2)
        @test Vb2 ≈ 2 .* Vb1   atol = 1e-12
        @test Vh2 ≈ 2 .* Vh1   atol = 1e-12
    end
end
