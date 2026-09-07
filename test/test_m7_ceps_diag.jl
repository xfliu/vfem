# test/test_m7_ceps_diag.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. Unit square, one nucleus at the centre: total mass of D_h equals the
#      closed form 4·ln(1+√2), independently of the mesh.
#   2. Nucleus exactly on a mesh vertex — the singular branch (2D and 3D).
#   3. Zero charge gives the zero matrix; negative charge flips the sign.
#   4. Several nuclei: superposition against the one-nucleus results.
#   5. 3D far nucleus — the conical-product Gauss branch — and a nucleus at a
#      vertex, the exact-moment branch.
#   6. Rigid translation of mesh and nucleus together leaves D_h unchanged.
#
# MATHEMATICAL CONTRACT (docs/testing-contract.md (c)):
#   D_h[i,j] = Σ_c Z_c ∫_Ω ψ_i ψ_j / |x − a_c| dx  with P1 basis ψ.
#   Hence, exactly and on any mesh:
#     (i)   D_h is symmetric;
#     (ii)  Σ_ij D_h[i,j] = Σ_c Z_c ∫_Ω 1/|x−a_c| dx, because Σ_i ψ_i ≡ 1;
#     (iii) D_h is linear in the charges, so scaling and superposition hold
#           exactly;
#     (iv)  for Z > 0 the weight 1/r is positive, so D_h is PSD;
#     (v)   D_h is invariant under rigid translation of mesh and nuclei.
#   For the unit square with the nucleus at its centre, splitting into eight
#   congruent sectors and integrating in polar coordinates gives
#     ∫_Ω 1/r dx = 8 ∫_0^{π/4} (1/2)·sec θ dθ = 4·ln(1+√2),
#   which is the value (ii) must reproduce for any triangulation.

using Test
using LinearAlgebra: issymmetric, eigvals, Symmetric, norm
using VFEM: mesh2d_load, mesh_load_from_folder, red_refine_mesh_3d

const _M7_2D_DIR = joinpath(@__DIR__, "fixtures", "unit_square_8x8")
const _M7_3D_DIR = joinpath(@__DIR__, "fixtures", "cube_r1")

# ∫ over the unit square of 1/|x − centre|, in closed form.
const _SQUARE_INV_R = 4 * log(1 + sqrt(2))

@testset "m7_ceps_diag" begin
    m2 = mesh2d_load(_M7_2D_DIR)
    m3 = mesh_load_from_folder(_M7_3D_DIR)

    @testset "1. total mass matches the closed form 4·ln(1+√2)" begin
        D = VFEM._assemble_Dh_2d(m2, [0.5 0.5], [1.0])
        # Σ_i ψ_i ≡ 1, so the double sum collapses to ∫_Ω 1/r dx exactly.
        @test sum(D) ≈ _SQUARE_INV_R rtol = 1e-10
        @test issymmetric(Matrix(D))
        # 1/r > 0 ⟹ the Gram matrix is positive semi-definite.
        @test minimum(eigvals(Symmetric(Matrix(D)))) > -1e-10
    end

    @testset "2. nucleus sitting on a mesh vertex (singular branch)" begin
        D = VFEM._assemble_Dh_2d(m2, [0.0 0.0], [1.0])
        @test all(isfinite, D)
        @test issymmetric(Matrix(D))
        @test sum(D) > 0
    end

    @testset "3. charge scaling and sign" begin
        D1 = VFEM._assemble_Dh_2d(m2, [0.5 0.5], [1.0])
        @test VFEM._assemble_Dh_2d(m2, [0.5 0.5], [0.0]) ≈ 0 * D1 atol = 1e-14
        @test VFEM._assemble_Dh_2d(m2, [0.5 0.5], [2.0]) ≈ 2 * D1 rtol = 1e-12
        @test VFEM._assemble_Dh_2d(m2, [0.5 0.5], [-1.0]) ≈ -D1 rtol = 1e-12
    end

    @testset "4. superposition over nuclei (2D)" begin
        a, b = [0.25 0.25], [0.75 0.6]
        Da = VFEM._assemble_Dh_2d(m2, a, [1.0])
        Db = VFEM._assemble_Dh_2d(m2, b, [0.5])
        Dab = VFEM._assemble_Dh_2d(m2, [0.25 0.25; 0.75 0.6], [1.0, 0.5])
        @test Dab ≈ Da + Db rtol = 1e-12
    end

    @testset "5. 3D: both branches" begin
        # (a) far nucleus -> conical-product Gauss branch
        far = [0.3137 0.2718 0.4142]
        Df = VFEM._assemble_Dh_3d(m3, far, [1.0])
        @test issymmetric(Matrix(Df))
        @test minimum(eigvals(Symmetric(Matrix(Df)))) > -1e-10
        @test VFEM._assemble_Dh_3d(m3, far, [2.0]) ≈ 2 * Df rtol = 1e-12
        # (b) nucleus at a vertex -> exact singular-moment branch
        Dv = VFEM._assemble_Dh_3d(m3, m3.NodeList[1:1, :], [1.0])
        @test all(isfinite, Dv)
        @test issymmetric(Matrix(Dv))
        # (c) superposition across the two branches
        both = VFEM._assemble_Dh_3d(m3, vcat(far, m3.NodeList[1:1, :]), [1.0, 1.0])
        @test both ≈ Df + Dv rtol = 1e-12
    end

    @testset "5b. 3D Gauss branch: absolute magnitude vs. grid quadrature" begin
        # Relative identities (symmetry, Z-linearity, superposition) are all
        # invariant under a global rescaling of the Gauss branch, so they cannot
        # detect a wrong quadrature weight. This pins the magnitude against an
        # independent midpoint rule. Nucleus placed OUTSIDE the cube, so 1/r is
        # smooth on Ω and a plain grid converges fast.
        a = [1.7, 0.3, 0.45]
        m3r = red_refine_mesh_3d(m3)
        D = VFEM._assemble_Dh_3d(m3r, reshape(a, 1, 3), [1.0])
        # Σ_i ψ_i ≡ 1 ⟹ sum(D) = ∫_Ω 1/|x−a| dV over the unit cube.
        n = 120; h = 1 / n; ref = 0.0
        for i in 1:n, j in 1:n, k in 1:n
            dx = (i - 0.5) * h - a[1]
            dy = (j - 0.5) * h - a[2]
            dz = (k - 0.5) * h - a[3]
            ref += 1 / sqrt(dx * dx + dy * dy + dz * dz)
        end
        ref *= h^3
        @test sum(D) ≈ ref rtol = 2e-3
    end

    @testset "6. rigid translation invariance (3D)" begin
        shift = [0.37, -0.11, 0.52]
        m3s = deepcopy(m3)
        m3s.NodeList .= m3.NodeList .+ shift'
        a  = [0.3137 0.2718 0.4142]
        as = a .+ shift'
        @test VFEM._assemble_Dh_3d(m3s, as, [1.0]) ≈
              VFEM._assemble_Dh_3d(m3, a, [1.0]) rtol = 1e-10
    end

    @testset "7. ceps_diagnostic returns a finite certificate" begin
        c2 = VFEM.ceps_diagnostic_2d(m2, [0.5 0.5], [1.0], 0.5)
        @test isfinite(c2)
        m3r = red_refine_mesh_3d(m3)
        c3 = VFEM.ceps_diagnostic_3d(m3r, [0.3137 0.2718 0.4142], [1.0], 0.5)
        @test isfinite(c3)
    end

    # (d) efficiency: assembly must stay linear. Pre-2026-09 the scatter-add
    # build was O(nt^2); this ceiling is ~20x the measured 0.043 s at nt=8192.
    @testset "8. efficiency regression threshold" begin
        VFEM._assemble_Dh_2d(m2, [0.5 0.5], [1.0])          # warm up
        t = @elapsed VFEM._assemble_Dh_2d(m2, [0.5 0.5], [1.0])
        @test t < 1.0
    end
end
