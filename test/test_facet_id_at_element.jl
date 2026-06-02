# test/test_facet_id_at_element.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. Each of the 4 local faces of a sample element returns its
#      correct local index (1, 2, 3, 4).
#   2. Round-trip property on a real mesh: for every (e, k),
#      facet_id_at_element(FacetList[Element2Facet[e, k], :],
#                          ElementList[e, :]) == k.
#   3. facet_nodes wrong length -> DimensionMismatch.
#   4. element_nodes wrong length -> DimensionMismatch.
#   5. Unrelated facet -> ArgumentError.

using Test
using VFEM: facet_id_at_element, mesh_load_from_folder

const _CUBE_DIR = joinpath(@__DIR__, "fixtures", "cube_r1")

@testset "facet_id_at_element" begin
    @testset "1. trivial 4-vertex element" begin
        e = [10, 20, 30, 40]
        @test facet_id_at_element([20, 30, 40], e) == 1
        @test facet_id_at_element([10, 30, 40], e) == 2
        @test facet_id_at_element([10, 20, 40], e) == 3
        @test facet_id_at_element([10, 20, 30], e) == 4
    end

    @testset "2. round-trip on cube_r1 mesh" begin
        m = mesh_load_from_folder(_CUBE_DIR)
        for e in 1:m.NumElt, k in 1:4
            f_idx = m.Element2Facet[e, k]
            @test facet_id_at_element(m.FacetList[f_idx, :],
                                      m.ElementList[e, :]) == k
        end
    end

    @testset "preconditions" begin
        @test_throws DimensionMismatch facet_id_at_element([1, 2], [1, 2, 3, 4])
        @test_throws DimensionMismatch facet_id_at_element([1, 2, 3], [1, 2, 3])
        # Facet not contained in element.
        @test_throws ArgumentError facet_id_at_element([100, 200, 300], [1, 2, 3, 4])
    end
end
