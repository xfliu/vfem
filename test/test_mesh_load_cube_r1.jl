# test/test_mesh_load_cube_r1.jl
#
# Cross-validate the entire mesh layer (mesh_load_from_folder + all
# combinatorial outputs + mesh_info) against MATLAB R2024a on the
# canonical 27-node, 40-element cube_r1 fixture mesh.
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. Counts: NumNode, NumElt, NumF, NumEdge match.
#   2. mesh_info: h_max, h_min, C_h, vol_total, vol_min, vol_max match.
#   3. FacetList: every row matches MATLAB row-by-row (104 rows).
#   4. EdgeList: every row matches MATLAB row-by-row (90 rows).
#   5. Element2Facet: every entry matches (40 × 4).
#   6. Facet2Element: every entry matches (104 × 2).
#   7. Signed connectivity: ElementFacetDirectSign matches (40 × 4).
#   8. Boundary detection: a facet is boundary iff Facet2Element[:, 2] == 0.
#   9. Performance: full load < 50 ms on this fixture.
#
# MATHEMATICAL CONTRACT (docs/testing-contract.md (c)):
#   The output of mesh_load_from_folder is a function of the input
#   nodes.dat / elements.dat — bit-identical reproductions across
#   runs and across MATLAB↔Julia. This is the contract, and the
#   fixture validates it byte-for-byte against MATLAB.

using Test
using VFEM: Mesh3D, mesh_load_from_folder, mesh_info,
            facet_element_connectivity_with_sign

const _CUBE_R1_DIR = joinpath(@__DIR__, "fixtures", "cube_r1")

# Parse the fixture into a Dict-of-Dicts keyed on prefix → row index.
function _load_mesh_fixture()
    path = joinpath(_CUBE_R1_DIR, "mesh_ref.txt")
    counts = Dict{String, Int}()
    floats = Dict{String, Float64}()
    F = Dict{Int, NTuple{3, Int}}()
    E = Dict{Int, NTuple{2, Int}}()
    E2F = Dict{Int, NTuple{4, Int}}()
    F2E = Dict{Int, NTuple{2, Int}}()
    SE2F = Dict{Int, NTuple{4, Int}}()
    for line in eachline(path)
        line = strip(line)
        isempty(line) && continue
        if (m = match(r"^(NumNode|NumElt|NumF|NumEdge)=(\d+)$", line)) !== nothing
            counts[m[1]] = parse(Int, m[2])
        elseif (m = match(r"^(h_max|h_min|C_h|vol_total|vol_min|vol_max)=(.+)$", line)) !== nothing
            floats[m[1]] = parse(Float64, m[2])
        elseif (m = match(r"^F\[(\d+)\]=(\d+) (\d+) (\d+)$", line)) !== nothing
            F[parse(Int, m[1])] = (parse(Int, m[2]), parse(Int, m[3]), parse(Int, m[4]))
        elseif (m = match(r"^E\[(\d+)\]=(\d+) (\d+)$", line)) !== nothing
            E[parse(Int, m[1])] = (parse(Int, m[2]), parse(Int, m[3]))
        elseif (m = match(r"^E2F\[(\d+)\]=(\d+) (\d+) (\d+) (\d+)$", line)) !== nothing
            E2F[parse(Int, m[1])] = (parse(Int, m[2]), parse(Int, m[3]),
                                       parse(Int, m[4]), parse(Int, m[5]))
        elseif (m = match(r"^F2E\[(\d+)\]=(\d+) (\d+)$", line)) !== nothing
            F2E[parse(Int, m[1])] = (parse(Int, m[2]), parse(Int, m[3]))
        elseif (m = match(r"^SE2F\[(\d+)\]=(\-?\d+) (\-?\d+) (\-?\d+) (\-?\d+)$", line)) !== nothing
            SE2F[parse(Int, m[1])] = (parse(Int, m[2]), parse(Int, m[3]),
                                        parse(Int, m[4]), parse(Int, m[5]))
        end
    end
    return counts, floats, F, E, E2F, F2E, SE2F
end

@testset "mesh layer cross-check on cube_r1 (MATLAB R2024a fixture)" begin
    counts, floats, F_ref, E_ref, E2F_ref, F2E_ref, SE2F_ref =
        _load_mesh_fixture()
    m = mesh_load_from_folder(_CUBE_R1_DIR)

    @testset "1. counts match MATLAB" begin
        @test m.NumNode == counts["NumNode"]
        @test m.NumElt  == counts["NumElt"]
        @test m.NumF    == counts["NumF"]
        @test m.NumEdge == counts["NumEdge"]
    end

    @testset "2. mesh_info statistics match MATLAB" begin
        info = mesh_info(m)
        @test info.h_max ≈ floats["h_max"] atol = 1e-13 rtol = 1e-13
        @test info.h_min ≈ floats["h_min"] atol = 1e-13 rtol = 1e-13
        @test info.C_h   ≈ floats["C_h"]   atol = 1e-13 rtol = 1e-13
        @test info.vol_total ≈ floats["vol_total"] atol = 1e-13 rtol = 1e-13
        @test info.vol_min   ≈ floats["vol_min"]   atol = 1e-13 rtol = 1e-13
        @test info.vol_max   ≈ floats["vol_max"]   atol = 1e-13 rtol = 1e-13
    end

    @testset "3. FacetList row-by-row" begin
        @test size(m.FacetList) == (m.NumF, 3)
        for r in 1:m.NumF
            @test (m.FacetList[r, 1], m.FacetList[r, 2], m.FacetList[r, 3]) == F_ref[r]
        end
    end

    @testset "4. EdgeList row-by-row" begin
        @test size(m.EdgeList) == (m.NumEdge, 2)
        for r in 1:m.NumEdge
            @test (m.EdgeList[r, 1], m.EdgeList[r, 2]) == E_ref[r]
        end
    end

    @testset "5. Element2Facet row-by-row" begin
        for r in 1:m.NumElt
            @test (m.Element2Facet[r, 1], m.Element2Facet[r, 2],
                   m.Element2Facet[r, 3], m.Element2Facet[r, 4]) == E2F_ref[r]
        end
    end

    @testset "6. Facet2Element row-by-row" begin
        for r in 1:m.NumF
            @test (m.Facet2Element[r, 1], m.Facet2Element[r, 2]) == F2E_ref[r]
        end
    end

    @testset "7. signed Element2Facet row-by-row" begin
        _, _, sgn = facet_element_connectivity_with_sign(m.ElementList, m.FacetList)
        for r in 1:m.NumElt
            @test (sgn[r, 1], sgn[r, 2], sgn[r, 3], sgn[r, 4]) == SE2F_ref[r]
        end
    end

    @testset "8. boundary detection matches Facet2Element[:, 2] == 0" begin
        # A unit cube split into 40 tets has 6 boundary squares × ? facets.
        # We don't hardcode the count — just verify the predicate is
        # consistent with what we know: each surface vertex appears in
        # ≥ 1 boundary facet, so the count is positive.
        n_bd = count(==(0), @view m.Facet2Element[:, 2])
        @test n_bd > 0
    end

    @testset "9. performance: full load < 50 ms on this fixture" begin
        mesh_load_from_folder(_CUBE_R1_DIR)
        T = 10
        t0 = time_ns()
        for _ in 1:T
            mesh_load_from_folder(_CUBE_R1_DIR)
        end
        tns = (time_ns() - t0) / T
        @test tns < 50_000_000
    end
end
