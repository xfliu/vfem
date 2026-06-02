# test/test_dof_on_facet.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. M = 0: facet-DOF count = 1, the single (0,0,0,0) DOF lies on every facet.
#   2. M = 1: 4 DOFs total; each facet (opposite vertex k) has 3 DOFs (γ[k] == 0).
#   3. M = 2: 10 DOFs total; each facet has simplex_dof(2, 2) = 6 DOFs.
#   4. dof_on_facet output length always == simplex_dof(2, M).
#   5. common_dof_on_facets returns vectors of length M+1 (DOFs on shared edge).
#   6. common_dof_on_facets indices are positions WITHIN each facet's DOF list,
#      not global DOFs.
#   7. FacetIdx out of range -> DomainError; equal facets -> ArgumentError.
#   8. M < 0 -> DomainError.
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   A degree-M Bernstein DOF γ ∈ ijkl_list(M) lies on the local face
#   opposite vertex k iff γ[k] == 0 (i.e. the face restricted to that
#   plane corresponds to multi-indices with no contribution from k).

using Test
using VFEM: dof_on_facet, common_dof_on_facets, ijkl_list, simplex_dof

@testset "dof_on_facet" begin
    @testset "1. M = 0" begin
        for f in 1:4
            @test dof_on_facet(0, f) == [1]
        end
    end

    @testset "2. M = 1: 3 DOFs per facet" begin
        # ijkl_list(1) = (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1).
        # Facet f opposite vertex f → DOFs whose γ[f] == 0 → all rows EXCEPT row f.
        @test dof_on_facet(1, 1) == [2, 3, 4]
        @test dof_on_facet(1, 2) == [1, 3, 4]
        @test dof_on_facet(1, 3) == [1, 2, 4]
        @test dof_on_facet(1, 4) == [1, 2, 3]
    end

    @testset "3. M = 2: 6 DOFs per facet" begin
        for f in 1:4
            v = dof_on_facet(2, f)
            @test length(v) == 6
            list = ijkl_list(2)
            @test all(list[v[i], f] == 0 for i in 1:6)
        end
    end

    @testset "4. length always = simplex_dof(2, M)" begin
        for M in 0:5, f in 1:4
            @test length(dof_on_facet(M, f)) == simplex_dof(2, M)
        end
    end

    @testset "preconditions" begin
        @test_throws DomainError dof_on_facet(-1, 1)
        @test_throws DomainError dof_on_facet(2, 0)
        @test_throws DomainError dof_on_facet(2, 5)
    end
end

@testset "common_dof_on_facets" begin
    @testset "5. lengths == M + 1" begin
        for M in 0:5, f1 in 1:3, f2 in (f1 + 1):4
            d1, d2 = common_dof_on_facets(M, f1, f2)
            @test length(d1) == M + 1
            @test length(d2) == M + 1
        end
    end

    @testset "6. indices are positions within each facet's DOF list" begin
        # M = 2, facets 1 (γ[1]=0) and 2 (γ[2]=0). Common: γ[1]=γ[2]=0.
        # The shared edge has degree 2 + 1 = 3 DOFs.
        d1, d2 = common_dof_on_facets(2, 1, 2)
        @test length(d1) == 3
        @test length(d2) == 3
        list = ijkl_list(2)
        f1 = dof_on_facet(2, 1)
        f2 = dof_on_facet(2, 2)
        for k in 1:3
            # The k-th common DOF, viewed in facet-1 numbering, is
            # f1[d1[k]]; viewed in facet-2 numbering, f2[d2[k]]. Both
            # must point at the same global DOF (a multi-index with
            # γ[1] == γ[2] == 0).
            @test f1[d1[k]] == f2[d2[k]]
            r = f1[d1[k]]
            @test list[r, 1] == 0
            @test list[r, 2] == 0
        end
    end

    @testset "preconditions" begin
        @test_throws ArgumentError common_dof_on_facets(2, 2, 2)
        @test_throws DomainError common_dof_on_facets(2, 0, 2)
        @test_throws DomainError common_dof_on_facets(2, 2, 5)
        @test_throws DomainError common_dof_on_facets(-1, 1, 2)
    end
end
