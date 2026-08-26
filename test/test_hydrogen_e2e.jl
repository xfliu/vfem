# test/test_hydrogen_e2e.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. End-to-end Hydrogen-like Liu lower bound on cube_r1: a single
#      Z=1 nucleus at (0.1, 0.1, 0.1), neig = 4, gamma_h_override = 0.3
#      (the paper's Hydrogen value). Compares h_max, C_h, raw eig_h,
#      and Liu eig_lower against MATLAB R2024a to ≤ 1e-9.
#   2. Sanity: eig_lower is *less* than eig_h (the Liu shift always
#      reduces, even with positive shift γ_h).
#   3. The Liu correction with shift formula:
#        ν = eig_h + γ_h
#        eig_lower = ν / (1 + Ch² · ν) − γ_h.
#      Verify this exact equality in Julia.
#
# This integration test exercises the whole Phase 5 chain:
#   mesh load → Coulomb-Duffy element-average → CECR matrices →
#   Dirichlet BC removal → eigensolve → Liu shift.

using Test
using VFEM: mesh_load_from_folder, elem_V_coulomb_average, CoulombInfo,
            schrodinger_eig_cecr_3d, find_mesh_hmax_3d

const _HYD_DIR = joinpath(@__DIR__, "fixtures", "cube_r1")

function _load_hyd_fixture()
    floats = Dict{String, Float64}()
    for line in eachline(joinpath(_HYD_DIR, "hydrogen_e2e_ref.txt"))
        line = strip(line)
        isempty(line) && continue
        if (mm = match(r"^([A-Za-z_0-9]+)=([+-]?\d+\.\d+e[+-]?\d+)$", line)) !== nothing
            floats[mm[1]] = parse(Float64, mm[2])
        elseif (mm = match(r"^([A-Za-z_0-9]+)=([+-]?\d+\.?\d*)$", line)) !== nothing
            floats[mm[1]] = parse(Float64, mm[2])
        end
    end
    return floats
end

@testset "hydrogen end-to-end (cube_r1)" begin
    f = _load_hyd_fixture()
    m = mesh_load_from_folder(_HYD_DIR)

    @testset "1. full pipeline matches MATLAB" begin
        info = CoulombInfo([0.1 0.1 0.1], [1.0])
        c_data = elem_V_coulomb_average(m, info)
        r = schrodinger_eig_cecr_3d(m, c_data, 4; gamma_h_override = 0.3)

        @test find_mesh_hmax_3d(m) ≈ f["h_max"] atol = 1e-12 rtol = 1e-12
        @test r.Ch ≈ f["C_h"]      atol = 1e-12 rtol = 1e-12
        @test r.gamma_h ≈ 0.3      atol = 1e-14
        for k in 1:4
            @test r.eig_h[k]     ≈ f["eig_h_$k"]     atol = 1e-9 rtol = 1e-10
            @test r.eig_lower[k] ≈ f["eig_lower_$k"] atol = 1e-9 rtol = 1e-10
        end
    end

    @testset "2. eig_lower < eig_h (Liu shift always tightens upward to lower bound)" begin
        info = CoulombInfo([0.1 0.1 0.1], [1.0])
        c_data = elem_V_coulomb_average(m, info)
        r = schrodinger_eig_cecr_3d(m, c_data, 4; gamma_h_override = 0.3)
        for k in 1:4
            @test r.eig_lower[k] < r.eig_h[k]
        end
    end

    @testset "3. Liu shift formula identity (γ_h = 0.3)" begin
        info = CoulombInfo([0.1 0.1 0.1], [1.0])
        c_data = elem_V_coulomb_average(m, info)
        r = schrodinger_eig_cecr_3d(m, c_data, 4; gamma_h_override = 0.3)
        for k in 1:4
            ν = r.eig_h[k] + 0.3
            expected = ν / (1 + ν * r.Ch^2) - 0.3
            @test r.eig_lower[k] ≈ expected atol = 1e-13 rtol = 1e-13
        end
    end
end
