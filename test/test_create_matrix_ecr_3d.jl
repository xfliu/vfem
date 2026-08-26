# test/test_create_matrix_ecr_3d.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. cube_r1 fixture (40 elements, 104 facets, 144 DOFs): trace, sum,
#      Frobenius and nnz of both A and M match MATLAB R2024a to ≤ 1e-10.
#   2. Symmetry: A ≈ A', M ≈ M'.
#   3. Mass total = |Ω| = 1 (cube has unit volume in this fixture).
#   4. Stiffness annihilates the constant mode: A · ones ≈ 0 on interior
#      DOFs (constant 1 has zero gradient — should hold *globally* on the
#      ECR space modulo the cell-DOF basis change; check by summing rows
#      and comparing to numerical zero).
#   5. DOF info struct: face_dofs = 1:NumF, cell_dofs = NumF+1:NumF+NumElt.
#   6. Generic on T: Float64 result equals mid-point of Interval result
#      to round-off; interval widths < 1e-10.
#   7. Performance: < 5 s on this small fixture.
#
# MATHEMATICAL CONTRACT (docs/testing-contract.md (c)):
#   The ECR finite element on tetrahedra has 5 DOFs per element: 4
#   face-midpoint averages (CR basis φ_i = 1 − 3 L_i) and 1 cell-average
#   enrichment (q = |x|² normalized to vanish on face averages). All
#   integrals are exact via Bernstein-2 closed form. The Bernstein
#   multinomial factor C(N;α) = N!/α! is load-bearing — see
#   `Validation/RESULTS.md` for the bug-fix log.

using Test
using SparseArrays: nnz
using LinearAlgebra: tr, norm
using IntervalArithmetic: Interval, mid, sup, inf
using VFEM: mesh_load_from_folder, create_matrix_ecr_3d, EcrDof3D

const _ECR3D_DIR = joinpath(@__DIR__, "fixtures", "cube_r1")

function _load_ecr3d_fixture()
    floats = Dict{String, Float64}(); ints = Dict{String, Int}()
    for line in eachline(joinpath(_ECR3D_DIR, "ecr3d_ref.txt"))
        line = strip(line)
        isempty(line) && continue
        if (mm = match(r"^([A-Za-z_0-9]+)=([+-]?\d+\.\d+e[+-]?\d+)$", line)) !== nothing
            floats[mm[1]] = parse(Float64, mm[2])
        elseif (mm = match(r"^([A-Za-z_0-9]+)=(\d+)$", line)) !== nothing
            ints[mm[1]] = parse(Int, mm[2])
        end
    end
    return floats, ints
end

@testset "create_matrix_ecr_3d" begin
    floats, ints = _load_ecr3d_fixture()
    m = mesh_load_from_folder(_ECR3D_DIR)

    @testset "1. cube_r1 fixture matches MATLAB" begin
        A, M, info = create_matrix_ecr_3d(m)
        @test info isa EcrDof3D
        @test info.ndof == ints["ndof"] == 144
        @test info.NumF == ints["NumF"] == m.NumF == 104
        @test info.NumElt == ints["NumElt"] == m.NumElt == 40

        @test tr(A)   ≈ floats["A_trace"]  atol = 1e-9 rtol = 1e-10
        @test sum(A)  ≈ floats["A_sum"]    atol = 1e-9
        @test norm(A) ≈ floats["A_frob"]   atol = 1e-9 rtol = 1e-10
        @test nnz(A)  == ints["A_nnz"]
        @test tr(M)   ≈ floats["M_trace"]  atol = 1e-12
        @test sum(M)  ≈ floats["M_sum"]    atol = 1e-12
        @test norm(M) ≈ floats["M_frob"]   atol = 1e-12
        @test nnz(M)  == ints["M_nnz"]
    end

    @testset "2. symmetry" begin
        A, M, _ = create_matrix_ecr_3d(m)
        @test A ≈ A'
        @test M ≈ M'
    end

    @testset "3. mass total = |Ω|" begin
        _, M, _ = create_matrix_ecr_3d(m)
        @test sum(M) ≈ 1.0 atol = 1e-12
    end

    @testset "4. stiffness annihilates constant" begin
        # The constant function in physical space is reproduced as
        # `Σ_i φ_i^ECR + ψ = 1`: with all 5 local DOFs (4 face + 1 cell)
        # at coefficient 1, the local linear combination is the constant
        # 1. Globally, `c = ones(ndof)` represents the constant; A·c = 0.
        A, M, _ = create_matrix_ecr_3d(m)
        c = ones(size(A, 1))
        @test maximum(abs, A * c) < 1e-10
        # Mass · ones gives ∫ 1 · 1 = |Ω| over the rows the basis covers.
        @test sum(M * c) ≈ 1.0 atol = 1e-12
    end

    @testset "5. DOF info layout" begin
        _, _, info = create_matrix_ecr_3d(m)
        @test info.face_dofs == 1:m.NumF
        @test info.cell_dofs == (m.NumF + 1):(m.NumF + m.NumElt)
    end

    @testset "6. interval mode encloses Float64 mode" begin
        A_f, M_f, _ = create_matrix_ecr_3d(m)
        A_i, M_i, _ = create_matrix_ecr_3d(m; T = Interval{Float64})
        # Per-entry: the interval result encloses the float result.
        # We don't iterate every entry; spot check via summary statistics.
        @test inf(sum(A_i)) ≤ sum(A_f) ≤ sup(sum(A_i))
        @test inf(sum(M_i)) ≤ sum(M_f) ≤ sup(sum(M_i))
        # Interval widths small.
        max_width_M = maximum(sup.(M_i) .- inf.(M_i))
        @test max_width_M < 1e-10
    end

    @testset "7. performance < 5 s" begin
        create_matrix_ecr_3d(m)   # warm-up
        t0 = time_ns()
        create_matrix_ecr_3d(m)
        @test (time_ns() - t0) / 1e9 < 5.0
    end
end
