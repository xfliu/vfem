# test/test_create_matrix_cecr_3d.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. cube_r1 fixture, c = ones(NumElt): trace, sum, Frobenius, nnz of A
#      and tr(M) match MATLAB to ≤ 1e-10. The reaction adds Σ_e |K_e| =
#      |Ω| = 1 to the cell-DOF diagonal entries → tr(A) increases by 1
#      vs the bare ECR.
#   2. cube_r1, c = (1:NumElt)·0.1 (non-constant): A trace, sum,
#      Frobenius match MATLAB.
#   3. M is unchanged from `create_matrix_ecr_3d` (the reaction touches
#      A only).
#   4. Symmetry: A ≈ A', M ≈ M'.
#   5. c_data length mismatch -> DimensionMismatch.
#   6. Performance: < 5 s on this fixture.
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   The CECR reaction term is `c_K · ∫_K (Π₀ u)(Π₀ v) dx = c_K · |K_e|
#   · u_cell · v_cell`. It contributes to A only on the diagonal entry
#   indexed by the cell-average DOF of element e.

using Test
using SparseArrays: nnz
using LinearAlgebra: tr, norm
using VFEM: mesh_load_from_folder, create_matrix_ecr_3d,
            create_matrix_cecr_3d

const _CECR3D_DIR = joinpath(@__DIR__, "fixtures", "cube_r1")

function _load_cecr3d_fixture()
    cases = Dict{String, Dict{String, Float64}}()
    nnz_cases = Dict{String, Dict{String, Int}}()
    cur = ""
    for line in eachline(joinpath(_CECR3D_DIR, "cecr3d_ref.txt"))
        line = strip(line)
        isempty(line) && continue
        if (mm = match(r"^case=(.+)$", line)) !== nothing
            cur = mm[1]
            cases[cur] = Dict{String, Float64}()
            nnz_cases[cur] = Dict{String, Int}()
            continue
        end
        if (mm = match(r"^([A-Za-z_0-9]+)=([+-]?\d+\.\d+e[+-]?\d+)$", line)) !== nothing
            cases[cur][mm[1]] = parse(Float64, mm[2])
        elseif (mm = match(r"^([A-Za-z_0-9]+)=(\d+)$", line)) !== nothing
            nnz_cases[cur][mm[1]] = parse(Int, mm[2])
        end
    end
    return cases, nnz_cases
end

@testset "create_matrix_cecr_3d" begin
    cases, ints = _load_cecr3d_fixture()
    m = mesh_load_from_folder(_CECR3D_DIR)

    @testset "1. c = ones (constant) matches MATLAB" begin
        c = ones(m.NumElt)
        A, M, info = create_matrix_cecr_3d(m, c)
        ref = cases["const_1"]
        @test tr(A)   ≈ ref["A_trace"]  atol = 1e-9 rtol = 1e-10
        @test sum(A)  ≈ ref["A_sum"]    atol = 1e-9
        @test norm(A) ≈ ref["A_frob"]   atol = 1e-9 rtol = 1e-10
        @test nnz(A)  == ints["const_1"]["A_nnz"]
        @test tr(M)   ≈ ref["M_trace"]  atol = 1e-12
    end

    @testset "2. c = (1:NumElt)·0.1 (non-constant) matches MATLAB" begin
        c = collect(1.0:m.NumElt) .* 0.1
        A, _, _ = create_matrix_cecr_3d(m, c)
        ref = cases["var"]
        @test tr(A)   ≈ ref["A_trace"]  atol = 1e-9 rtol = 1e-10
        @test sum(A)  ≈ ref["A_sum"]    atol = 1e-9 rtol = 1e-10
        @test norm(A) ≈ ref["A_frob"]   atol = 1e-9 rtol = 1e-10
    end

    @testset "3. M is unchanged from ECR" begin
        c = collect(1.0:m.NumElt) .* 0.5
        _, M_cecr, _ = create_matrix_cecr_3d(m, c)
        _, M_ecr,  _ = create_matrix_ecr_3d(m)
        @test M_cecr ≈ M_ecr
    end

    @testset "4. symmetry" begin
        A, M, _ = create_matrix_cecr_3d(m, ones(m.NumElt))
        @test A ≈ A'
        @test M ≈ M'
    end

    @testset "5. error preconditions" begin
        @test_throws DimensionMismatch create_matrix_cecr_3d(m, ones(m.NumElt - 1))
        @test_throws DimensionMismatch create_matrix_cecr_3d(m, ones(m.NumElt + 5))
    end

    @testset "6. performance < 5 s" begin
        c = ones(m.NumElt)
        create_matrix_cecr_3d(m, c)
        t0 = time_ns()
        create_matrix_cecr_3d(m, c)
        @test (time_ns() - t0) / 1e9 < 5.0
    end
end
