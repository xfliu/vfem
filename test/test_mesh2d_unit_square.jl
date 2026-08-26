# test/test_mesh2d_unit_square.jl
#
# Cross-validate 2D mesh I/O + the small helpers (find_tri2edge,
# find_is_edge_bd, find_mesh_hmax) against MATLAB on UnitSquare8x8.
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. Counts: nv, nt, ne, nb match MATLAB.
#   2. hmax matches to ≤ 1e-13.
#   3. Boundary-edge count via bd_edge_ids matches nb.
#   4. tri2edge round-trip: for every (k, j), the j-th local edge of
#      element k connects the two vertices opposite vertex j.
#   5. find_is_edge_bd: every edge marked boundary appears in bd.dat
#      and vice versa.
#   6. Performance: full load + invariants < 50 ms.
#
# MATHEMATICAL CONTRACT (docs/testing-contract.md (c)):
#   Given the mesh fixture files, mesh2d_load is a deterministic
#   function — the output struct matches MATLAB byte-for-byte on
#   counts, hmax, tri2edge, and bd_edge_ids.

using Test
using VFEM: Mesh2D, mesh2d_load, find_mesh_hmax, find_tri2edge, find_is_edge_bd

const _UNIT_SQUARE_DIR = joinpath(@__DIR__, "fixtures", "unit_square_8x8")

function _load_2d_fixture()
    path = joinpath(_UNIT_SQUARE_DIR, "cr_ecr_ref.txt")
    counts = Dict{String, Int}()
    floats = Dict{String, Float64}()
    for line in eachline(path)
        line = strip(line)
        isempty(line) && continue
        if (m = match(r"^(\w+)=(.+)$", line)) !== nothing
            key, val = m[1], m[2]
            try
                counts[key] = parse(Int, val)
            catch
                floats[key] = parse(Float64, val)
            end
        end
    end
    return counts, floats
end

@testset "Mesh2D / UnitSquare8x8 cross-check" begin
    counts, floats = _load_2d_fixture()
    m = mesh2d_load(_UNIT_SQUARE_DIR)

    @testset "1. counts match MATLAB" begin
        @test m.nv == counts["nv"]
        @test m.nt == counts["nt"]
        @test m.ne == counts["ne"]
        @test m.nb == counts["nb"]
    end

    @testset "2. hmax matches MATLAB" begin
        h = find_mesh_hmax(m.nodes, m.edges)
        @test h ≈ floats["hmax"] atol = 1e-13 rtol = 1e-13
    end

    @testset "3. boundary edges count" begin
        @test length(m.bd_edge_ids) == counts["nb"]
    end

    @testset "4. tri2edge round-trip" begin
        # For triangle k, local edge j connects vertices opposite vertex j,
        # i.e. (v[j+1], v[j+2]) cyclically — same convention as MATLAB.
        for k in 1:m.nt
            v = (m.elements[k, 1], m.elements[k, 2], m.elements[k, 3])
            for j in 1:3
                eid = m.tri2edge[k, j]
                a, b = m.edges[eid, 1], m.edges[eid, 2]
                # Local edge j is opposite vertex j: (v[j%3+1], v[(j+1)%3+1]).
                v1 = v[mod(j, 3) + 1]
                v2 = v[mod(j + 1, 3) + 1]
                pair = a < b ? (a, b) : (b, a)
                expected = v1 < v2 ? (v1, v2) : (v2, v1)
                @test pair == expected
            end
        end
    end

    @testset "5. find_is_edge_bd matches bd_edge_ids" begin
        is_bd = find_is_edge_bd(m.edges, m.bd_edges)
        @test sum(is_bd) == m.nb
        @test sort(findall(==(1), is_bd)) == sort(m.bd_edge_ids)
    end

    @testset "6. performance: full load < 50 ms" begin
        mesh2d_load(_UNIT_SQUARE_DIR)
        T = 10
        t0 = time_ns()
        for _ in 1:T
            mesh2d_load(_UNIT_SQUARE_DIR)
        end
        tns = (time_ns() - t0) / T
        @test tns < 50_000_000
    end
end
