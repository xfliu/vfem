# test/test_lg_lower_eig_bound_laplace.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. UnitSquare8x8, lagrange_order = 2, neig = 4: ρ, Ch_cr, the
#      `neig + 1` Liu-CR lower bounds, the `neig` Lehmann–Goerisch
#      lower bounds, and the `neig` Lagrange CG upper bounds all match
#      MATLAB to ~1e-9. (`example_lower_eig_bound_laplace.m` reference.)
#   2. eig_lower < eig_upper component-wise (lower-bound contract).
#   3. eig_lower[1] < 2π² < eig_upper[1] (the analytic first eigenvalue
#      is bracketed).
#   4. The LG lower bound is sharper than the Liu-CR lower bound for
#      every k ≤ neig: cr_eig_lower[k] ≤ eig_lower[k].
#   5. lagrange_order = 0 -> ArgumentError; neig = 0 -> DomainError;
#      RT_order = -1 -> DomainError.
#   6. Performance: < 30 s on this fixture.
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   For the homogeneous Dirichlet Laplacian on a Lipschitz domain Ω
#   in 2D, the Lehmann–Goerisch sharpening combines a CR-based shift
#   parameter ρ with a Goerisch correction `A_lg` from the RT auxiliary
#   problem to produce a guaranteed lower bound that is ≥ Liu's CR
#   bound. Together with the conforming-Lagrange CG eigenvalues
#   (Galerkin upper bound), this yields a validated bracket
#   `eig_lower[k] ≤ λ_true[k] ≤ eig_upper[k]`.

using Test
using VFEM: mesh2d_load, lg_lower_eig_bound_laplace, LGLaplaceLowerBound

const _LGB_DIR = joinpath(@__DIR__, "fixtures", "unit_square_8x8")

function _load_lg_fixture()
    floats = Dict{String, Float64}()
    ints   = Dict{String, Int}()
    for line in eachline(joinpath(_LGB_DIR, "lg_lower_eig_ref.txt"))
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

@testset "lg_lower_eig_bound_laplace" begin
    f, i = _load_lg_fixture()
    m = mesh2d_load(_LGB_DIR)

    @testset "1. Full pipeline matches MATLAB" begin
        r = lg_lower_eig_bound_laplace(m, i["lagrange_order"], i["neig"];
                                        RT_order = i["RT_order"])
        @test r isa LGLaplaceLowerBound
        @test r.rho   ≈ f["rho"]   atol = 1e-9 rtol = 1e-9
        @test r.Ch_cr ≈ f["Ch_cr"] atol = 1e-12 rtol = 1e-12

        @test length(r.cr_eig_lower) == i["neig"] + 1
        for k in 1:(i["neig"] + 1)
            @test r.cr_eig_lower[k] ≈ f["cr_eig_lower_$k"] atol = 1e-9 rtol = 1e-9
        end

        @test length(r.eig_lower) == i["neig"]
        @test length(r.eig_upper) == i["neig"]
        for k in 1:i["neig"]
            @test r.eig_lower[k] ≈ f["lg_eig_lower_$k"] atol = 1e-9 rtol = 1e-9
            @test r.eig_upper[k] ≈ f["lg_eig_upper_$k"] atol = 1e-9 rtol = 1e-9
        end
    end

    @testset "2. eig_lower < eig_upper component-wise" begin
        r = lg_lower_eig_bound_laplace(m, 2, 4)
        for k in 1:4
            @test r.eig_lower[k] < r.eig_upper[k]
        end
    end

    @testset "3. λ_true[1] = 2π² is bracketed" begin
        r = lg_lower_eig_bound_laplace(m, 2, 1)
        @test r.eig_lower[1] < 2π^2 < r.eig_upper[1]
    end

    @testset "4. LG lower ≥ Liu-CR lower for every k" begin
        r = lg_lower_eig_bound_laplace(m, 2, 4)
        for k in 1:4
            @test r.cr_eig_lower[k] ≤ r.eig_lower[k]
        end
    end

    @testset "5. error preconditions" begin
        @test_throws ArgumentError lg_lower_eig_bound_laplace(m, 0, 4)
        @test_throws DomainError   lg_lower_eig_bound_laplace(m, 2, 0)
        @test_throws DomainError   lg_lower_eig_bound_laplace(m, 2, 4; RT_order = -1)
    end

    @testset "6. performance < 30 s" begin
        # warm-up
        lg_lower_eig_bound_laplace(m, 2, 4)
        t0 = time_ns()
        lg_lower_eig_bound_laplace(m, 2, 4)
        @test (time_ns() - t0) / 1e9 < 30.0
    end
end
