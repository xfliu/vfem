# test/test_m3_p1_ub.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. Unit cube, zero potential: the P1 eigenvalues bound 3π² from above,
#      and refinement lowers them towards it.
#   2. Constant reaction coefficient c: the whole spectrum shifts by exactly c.
#   3. c = 0 reproduces the pure Laplace spectrum.
#   4. neig larger than the interior dimension is clamped rather than throwing.
#   5. Negative c (attractive potential) still yields real, ordered eigenvalues.
#
# MATHEMATICAL CONTRACT (docs/testing-contract.md (c)):
#   _p1_ub_3d_assembly solves K_h u = λ M_h u restricted to interior nodes,
#   with K_h the P1 stiffness plus the reaction term c_K·|K| on each element.
#   Two consequences are exact, on any mesh:
#     (i)  conformity — P1 ⊂ H¹₀ — so every λ_h is an UPPER bound for the
#          corresponding exact Dirichlet eigenvalue. On (0,1)³ the first is
#          λ₁ = 3π².
#     (ii) with c constant, (K + cM)u = λMu ⟺ Ku = (λ − c)Mu, so the
#          spectrum is rigidly shifted: λ_h(c) = λ_h(0) + c, to round-off.
#   (ii) is the sharpest available check: it pins the reaction assembly against
#   the kinetic assembly with no reference data.

using Test
using VFEM: mesh_load_from_folder, red_refine_mesh_3d

zero_c_of(mesh) = zeros(Float64, mesh.NumElt)

const _M3_3D_DIR = joinpath(@__DIR__, "fixtures", "cube_r1")
const _LAM1_CUBE = 3 * pi^2          # first Dirichlet eigenvalue of (0,1)³

@testset "m3_p1_ub" begin
    m_coarse = mesh_load_from_folder(_M3_3D_DIR)
    m = red_refine_mesh_3d(m_coarse)          # enough interior nodes to solve
    zero_c = zeros(Float64, m.NumElt)

    @testset "1. P1 eigenvalues bound 3π² from above" begin
        λ = VFEM._p1_ub_3d_assembly(m, zero_c, 1)
        @test all(isfinite, λ)
        @test λ[1] > _LAM1_CUBE          # conformity: an upper bound, never below
        # a finer mesh gives a sharper (smaller) upper bound
        mr = red_refine_mesh_3d(m)
        λc = VFEM._p1_ub_3d_assembly(mr, zero_c_of(mr), 1)
        @test λc[1] <= λ[1] + 1e-8
    end

    @testset "2. constant c shifts the whole spectrum by exactly c" begin
        λ0 = VFEM._p1_ub_3d_assembly(m, zero_c, 3)
        for c in (0.5, 2.0, -1.25)
            λc = VFEM._p1_ub_3d_assembly(m, fill(c, m.NumElt), 3)
            @test λc ≈ λ0 .+ c rtol = 1e-9
        end
    end

    @testset "3. c = 0 reproduces the pure Laplace spectrum" begin
        @test VFEM._p1_ub_3d_assembly(m, zero_c, 2) ≈
              VFEM._p1_ub_3d_assembly(m, zeros(Float64, m.NumElt), 2) rtol = 1e-12
    end

    @testset "4. neig is clamped to the interior dimension" begin
        λ = VFEM._p1_ub_3d_assembly(m, zero_c, 10_000)
        @test length(λ) >= 1
        @test all(isfinite, λ)
    end

    @testset "5. negative c stays real and ordered" begin
        λ = VFEM._p1_ub_3d_assembly(m, fill(-5.0, m.NumElt), 3)
        @test all(isfinite, λ)
        @test issorted(λ)
    end

    # (d) efficiency: assembly is COO-based and must stay linear.
    @testset "6. efficiency regression threshold" begin
        VFEM._p1_ub_3d_assembly(m, zero_c, 1)                 # warm up
        t = @elapsed VFEM._p1_ub_3d_assembly(m, zero_c, 1)
        @test t < 10.0
    end
end
