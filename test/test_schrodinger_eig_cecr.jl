# test/test_schrodinger_eig_cecr.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. Laplace (V = 0) on the unit square, Dirichlet BC, neig = 6:
#      eig_h matches MATLAB to ~1e-9; eig_lower matches MATLAB to ~1e-9;
#      eig_lower < eig_h component-wise.
#   2. V(x, y) = x² + y², same fixture.
#   3. eig_h[1] ≈ 19.5 (close to but below the analytic 2π² ≈ 19.74).
#   4. Liu's lower bound for V = 0 reduces to λ_h / (1 + λ_h · C_h²)
#      (no shift needed since γ_h = 0).
#   5. neig = 0 -> DomainError.
#   6. bc = :foo -> ArgumentError.
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   For the Dirichlet Laplacian on a domain Ω, every CECR FE eigenvalue
#   λ_h is also a numerically computed upper bound on the truncated
#   problem, and the Liu correction λ_h / (1 + λ_h · C_h²) is a
#   guaranteed lower bound. λ_lower ≤ λ_true ≤ λ_h with
#   C_h = 0.1490 · h_max.

using Test
using VFEM: mesh2d_load, schrodinger_eig_cecr

const _SCHR_DIR = joinpath(@__DIR__, "fixtures", "unit_square_8x8")

function _load_eig_fixture(filename)
    path = joinpath(_SCHR_DIR, filename)
    out = Dict{String, Float64}()
    n = 0
    for line in eachline(path)
        line = strip(line)
        isempty(line) && continue
        if (mm = match(r"^(eig_(?:h|lower|upper)_\d+)=(.+)$", line)) !== nothing
            out[mm[1]] = parse(Float64, mm[2])
        elseif startswith(line, "eig_h_count=")
            n = parse(Int, line[13:end])
        end
    end
    return out, n
end

@testset "schrodinger_eig_cecr" begin
    m = mesh2d_load(_SCHR_DIR)

    @testset "1. Laplace (V = 0), Dirichlet, neig = 6" begin
        ref, nref = _load_eig_fixture("laplace_dirichlet_unit_square_ref.txt")
        r = schrodinger_eig_cecr(m, (x, y) -> 0.0, 6)
        @test length(r.eig_h) == nref
        @test r.gamma_h ≈ 0.0 atol = 1e-14
        for k in 1:nref
            @test r.eig_h[k] ≈ ref["eig_h_$k"] atol = 1e-9 rtol = 1e-9
            @test r.eig_lower[k] ≈ ref["eig_lower_$k"] atol = 1e-9 rtol = 1e-9
            @test r.eig_upper[k] ≈ ref["eig_upper_$k"] atol = 1e-9 rtol = 1e-9
            @test r.eig_lower[k] < r.eig_h[k]
        end
    end

    @testset "2. V = x²+y², Dirichlet" begin
        ref, nref = _load_eig_fixture("quadV_dirichlet_unit_square_ref.txt")
        r = schrodinger_eig_cecr(m, (x, y) -> x * x + y * y, 6)
        for k in 1:nref
            @test r.eig_h[k] ≈ ref["eig_h_$k"] atol = 1e-9 rtol = 1e-9
            @test r.eig_lower[k] ≈ ref["eig_lower_$k"] atol = 1e-9 rtol = 1e-9
        end
    end

    @testset "3. λ_h[1] is below 2π² (analytic lowest Dirichlet Laplacian)" begin
        r = schrodinger_eig_cecr(m, (x, y) -> 0.0, 1)
        @test r.eig_h[1] < 2π^2          # numerical λ_h underestimates λ
        @test r.eig_lower[1] < r.eig_h[1]
    end

    @testset "4. Liu lower for V = 0 (no shift)" begin
        r = schrodinger_eig_cecr(m, (x, y) -> 0.0, 3)
        for k in 1:length(r.eig_h)
            expected = r.eig_h[k] / (1 + r.eig_h[k] * r.Ch^2)
            @test r.eig_lower[k] ≈ expected atol = 1e-13 rtol = 1e-13
        end
    end

    @testset "5–6. error preconditions" begin
        @test_throws DomainError schrodinger_eig_cecr(m, (x, y) -> 0.0, 0)
        @test_throws ArgumentError schrodinger_eig_cecr(m, (x, y) -> 0.0, 6; bc = :foo)
    end
end
