# test/test_elem_V_coulomb_average.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. Hydrogen-like (single nucleus at origin, Z = 1): per-element
#      averages on cube_r1 — first 10 entries plus min/max/sum/sumsq —
#      match MATLAB to ≤ 1e-10. Default Duffy n=10, standard n=4.
#   2. H₂⁺-like (two nuclei at ±0.5 along x, both Z = 1): same.
#   3. The element containing the origin yields a finite (very negative)
#      value: 1/r is L¹-integrable so the average is finite even when
#      the singularity is interior.
#   4. Z = 0 gives V_avg ≡ 0.
#   5. CoulombInfo construction: shape mismatches -> DimensionMismatch.
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   For V(x) = − Σ_c Z_c / |x − c_c| and a tetrahedron K, the integral
#   `(1/|K|) ∫_K V dx` is finite (since 1/|x − c| ∈ L¹(ℝ³)). The Duffy
#   transform near a singularity vertex has Jacobian η₁²·η₂ which
#   exactly cancels the 1/r factor, leaving a smooth integrand that
#   Gauss-Legendre integrates to high accuracy.

using Test
using VFEM: mesh_load_from_folder, elem_V_coulomb_average, CoulombInfo

const _COUL_DIR = joinpath(@__DIR__, "fixtures", "cube_r1")

function _load_coulomb_fixture()
    cases = Dict{String, Dict{String, Float64}}()
    cur = ""
    for line in eachline(joinpath(_COUL_DIR, "coulomb_avg_ref.txt"))
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

@testset "elem_V_coulomb_average" begin
    cases = _load_coulomb_fixture()
    m = mesh_load_from_folder(_COUL_DIR)

    @testset "1. Hydrogen-like (Z=1 at origin) matches MATLAB" begin
        info = CoulombInfo([0.0 0.0 0.0], [1.0])
        V = elem_V_coulomb_average(m, info)
        ref = cases["H"]
        @test length(V) == m.NumElt
        @test minimum(V) ≈ ref["min"]   atol = 1e-10 rtol = 1e-10
        @test maximum(V) ≈ ref["max"]   atol = 1e-10 rtol = 1e-10
        @test sum(V)     ≈ ref["sum"]   atol = 1e-10 rtol = 1e-10
        @test sum(V .^ 2)≈ ref["sumsq"] atol = 1e-10 rtol = 1e-10
        for k in 1:10
            @test V[k] ≈ ref["val_$k"] atol = 1e-10 rtol = 1e-10
        end
    end

    @testset "2. H₂⁺-like matches MATLAB" begin
        info = CoulombInfo([-0.5 0.0 0.0; 0.5 0.0 0.0], [1.0, 1.0])
        V = elem_V_coulomb_average(m, info)
        ref = cases["H2"]
        @test minimum(V) ≈ ref["min"]   atol = 1e-10 rtol = 1e-10
        @test maximum(V) ≈ ref["max"]   atol = 1e-10 rtol = 1e-10
        @test sum(V)     ≈ ref["sum"]   atol = 1e-10 rtol = 1e-10
        @test sum(V .^ 2)≈ ref["sumsq"] atol = 1e-10 rtol = 1e-10
        for k in 1:10
            @test V[k] ≈ ref["val_$k"] atol = 1e-10 rtol = 1e-10
        end
    end

    @testset "3. element with interior singularity yields finite very-negative average" begin
        info = CoulombInfo([0.0 0.0 0.0], [1.0])
        V = elem_V_coulomb_average(m, info)
        @test all(isfinite, V)
        @test minimum(V) < -1.0      # at least one element is "deep"
        @test maximum(V) < 0.0       # all attractive
    end

    @testset "4. Z = 0 gives V_avg ≡ 0" begin
        info = CoulombInfo([0.0 0.0 0.0], [0.0])
        V = elem_V_coulomb_average(m, info)
        @test all(iszero, V)
    end

    @testset "5. CoulombInfo error preconditions" begin
        @test_throws DimensionMismatch CoulombInfo([0.0 0.0], [1.0])      # 1×2, not 1×3
        @test_throws DimensionMismatch CoulombInfo([0.0 0.0 0.0], [1.0, 2.0])
    end

    @testset "6. performance < 5 s on cube_r1" begin
        info = CoulombInfo([0.0 0.0 0.0], [1.0])
        elem_V_coulomb_average(m, info)
        t0 = time_ns()
        elem_V_coulomb_average(m, info)
        @test (time_ns() - t0) / 1e9 < 5.0
    end
end
