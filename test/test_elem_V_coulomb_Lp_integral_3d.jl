# test/test_elem_V_coulomb_Lp_integral_3d.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. cube_r1, Coulomb at (0.1, 0.1, 0.1) inside an interior element,
#      cK = element-average V, p = 2: per-element values for the first
#      6 elements plus sum and sumsq match MATLAB to ≤ 1e-10.
#   2. Same fixture, p = 1: same comparison.
#   3. p = 2 and cK = V_avg: this is the L²-projection error for V on
#      the piecewise-constant approximation; check it's positive
#      element-wise and finite (Duffy handles the singularity).
#   4. cK length mismatch -> DimensionMismatch.
#   5. Performance: < 5 s on this fixture.
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   `I[K] = ∫_K |V(x) − cK[K]|^{p₀} dx` provides an upper bound on the
#   L^{p₀} approximation error when V is replaced by the per-element
#   constant `cK[K]`. The Duffy-transform `η₁²` Jacobian cancels the
#   1/r factor of V near a singularity vertex, so for any p₀ < 3 the
#   integral is finite and the quadrature converges.

using Test
using VFEM: mesh_load_from_folder, elem_V_coulomb_average, CoulombInfo,
            elem_V_coulomb_Lp_integral_3d

const _LP_DIR = joinpath(@__DIR__, "fixtures", "cube_r1")

function _load_lp_fixture()
    cases = Dict{String, Dict{String, Float64}}()
    cur = ""
    for line in eachline(joinpath(_LP_DIR, "lp_ref.txt"))
        line = strip(line)
        isempty(line) && continue
        if (mm = match(r"^case=(.+)$", line)) !== nothing
            cur = mm[1]
            cases[cur] = Dict{String, Float64}()
            continue
        end
        if (mm = match(r"^([A-Za-z_0-9]+)=([+-]?\d+\.\d+e[+-]?\d+)$", line)) !== nothing
            target = isempty(cur) ? get!(cases, "_meta", Dict{String, Float64}()) : cases[cur]
            target[mm[1]] = parse(Float64, mm[2])
        end
    end
    return cases
end

@testset "elem_V_coulomb_Lp_integral_3d" begin
    cases = _load_lp_fixture()
    m = mesh_load_from_folder(_LP_DIR)

    info = CoulombInfo([0.1 0.1 0.1], [1.0])
    cK = elem_V_coulomb_average(m, info)

    @testset "1. p = 2 matches MATLAB" begin
        I = elem_V_coulomb_Lp_integral_3d(m, info, cK, 2.0)
        ref = cases["p2"]
        @test sum(I)        ≈ ref["sum"]   atol = 1e-10 rtol = 1e-10
        @test sum(I .^ 2)   ≈ ref["sumsq"] atol = 1e-10 rtol = 1e-10
        for k in 1:6
            @test I[k] ≈ ref["val_$k"] atol = 1e-10 rtol = 1e-10
        end
    end

    @testset "2. p = 1 matches MATLAB" begin
        I = elem_V_coulomb_Lp_integral_3d(m, info, cK, 1.0)
        ref = cases["p1"]
        @test sum(I)        ≈ ref["sum"]   atol = 1e-10 rtol = 1e-10
        @test sum(I .^ 2)   ≈ ref["sumsq"] atol = 1e-10 rtol = 1e-10
        for k in 1:6
            @test I[k] ≈ ref["val_$k"] atol = 1e-10 rtol = 1e-10
        end
    end

    @testset "3. all I[K] ≥ 0 and finite" begin
        I = elem_V_coulomb_Lp_integral_3d(m, info, cK, 2.0)
        @test all(I .≥ 0)
        @test all(isfinite, I)
    end

    @testset "4. error preconditions" begin
        @test_throws DimensionMismatch elem_V_coulomb_Lp_integral_3d(m, info, ones(m.NumElt - 1), 2.0)
    end

    @testset "5. performance < 5 s" begin
        elem_V_coulomb_Lp_integral_3d(m, info, cK, 2.0)
        t0 = time_ns()
        elem_V_coulomb_Lp_integral_3d(m, info, cK, 2.0)
        @test (time_ns() - t0) / 1e9 < 5.0
    end
end
