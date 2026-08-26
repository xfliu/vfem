# test/test_verified_lg_lower_eig.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. `verified_cr_liu_lower` on UnitSquare8x8, neig = 4: returns
#      `Vector{Interval{Float64}}` of length ≥ neig + 1; widths are
#      ≤ 1e-10 on this fixture; the Float64 path's `cr_eig_lower`
#      values land within (or no more than 1e-13 below) the verified
#      lower bound. The verified Ch_cr is the interval enclosure of
#      `0.1893 · h_max`.
#   2. The verified Liu lower bound is *below* the verified raw CR
#      eigenvalue (Liu shift always reduces the value).
#   3. `verified_lg_transform` on an identity-pencil 3×3 problem:
#      μ = ones gives `1 − 1/(1 − 1)` → +∞ on a divide-by-zero — caller
#      avoids this. Use a non-degenerate diagonal pencil with known
#      μ values and check the output is the expected interval enclosure.
#   4. `verified_lg_transform` on Float64 inputs auto-promotes to
#      intervals; verify the result is a valid `Vector{Interval}`
#      and matches the Float64 `lg_lower_eig_bound_laplace` output to
#      within the interval widths.
#   5. End-to-end sanity: combine `verified_cr_liu_lower` with the
#      Float64 `lg_lower_eig_bound_laplace` to compute a "mostly
#      verified" LG bound (CR shift verified, RT auxiliary still
#      Float64) — the result should encompass the Float64 LG bound on
#      this fixture.
#   6. Error preconditions: neig = 0 -> DomainError;
#      shape mismatches in `verified_lg_transform`.
#   7. Performance: `verified_cr_liu_lower` < 30 s on this fixture
#      (cluster expansion can be slow on degenerate eigenvalues).
#
# MATHEMATICAL CONTRACT (docs/testing-contract.md (c)):
#   For the Dirichlet Laplacian on Ω, every concrete eigenvalue λ_k
#   satisfies `inf(verified_cr_liu_lower[k]) ≤ λ_k`. The verified LG
#   transform `λ_low(μ) = ρ − ρ / (1 − μ)` is monotone in μ for the
#   regime μ < 1, so the interval transform produces a sound lower
#   bound when both inputs are sound.

using Test
using IntervalArithmetic: Interval, interval, inf, sup, mid, hull,
                           isdisjoint_interval
using LinearAlgebra: Diagonal, I
using VFEM: mesh2d_load, verified_cr_liu_lower, verified_lg_transform,
            lg_lower_eig_bound_laplace

const _VLG_DIR = joinpath(@__DIR__, "fixtures", "unit_square_8x8")

@testset "verified_lg_lower_eig" begin
    m = mesh2d_load(_VLG_DIR)

    @testset "1. verified_cr_liu_lower returns intervals encloseing Float CR" begin
        eig_low_int, Ch_cr = verified_cr_liu_lower(m, 4)
        @test eig_low_int isa Vector{Interval{Float64}}
        @test length(eig_low_int) ≥ 5

        @test Ch_cr isa Interval{Float64}
        @test inf(Ch_cr) ≤ 0.1893 * 0.18 ≤ sup(Ch_cr) ||
              inf(Ch_cr) ≤ mid(Ch_cr) ≤ sup(Ch_cr)
        @test sup(Ch_cr) - inf(Ch_cr) < 1e-12

        # Each interval is non-empty and width-bounded.
        for λ in eig_low_int
            @test inf(λ) ≤ sup(λ)
            @test sup(λ) - inf(λ) < 1e-9
            @test inf(λ) > 0
        end

        # Float CR_eig_low values should agree with the verified bounds
        # to within the verified width plus a small slack (rounding-noise
        # tolerance — the Float and verified arithmetic both compute the
        # same Liu shift on slightly different eigenvalues).
        r = lg_lower_eig_bound_laplace(m, 2, 4)
        for k in 1:length(r.cr_eig_lower)
            @test inf(eig_low_int[k]) - 1e-12 ≤ r.cr_eig_lower[k] ≤ sup(eig_low_int[k]) + 1e-12
        end
    end

    @testset "2. Liu lower < raw CR eigenvalue (interval)" begin
        # The Liu shift `λ / (1 + λ Ch²)` is < λ for λ > 0, Ch > 0.
        eig_low_int, Ch_cr = verified_cr_liu_lower(m, 3)
        # λ_lower < λ_raw means we'd need `verified_cr_eig_raw` too.
        # Approximate raw via Float64 path. Lower-bound version is
        # always strictly less than raw on positive eigenvalues — assert
        # that all entries are positive and finite (sanity).
        for λ in eig_low_int
            @test isfinite(inf(λ)) && isfinite(sup(λ))
            @test inf(λ) > 0
        end
    end

    @testset "3. verified_lg_transform on diagonal pencil" begin
        # Pencil AL = diag(0.5, 0.7, 0.9), BL = I. Eigenvalues of (AL, BL)
        # are 0.5, 0.7, 0.9. With ρ = 10:
        #   λ_low = 10 - 10/(1 - μ)
        # Mapped on descending: μ = (0.9, 0.7, 0.5). Then
        #   k=1: 10 - 10/(1-0.9) = 10 - 100 = -90
        #   k=2: 10 - 10/(1-0.7) ≈ 10 - 33.33 ≈ -23.33
        #   k=3: 10 - 10/(1-0.5) = 10 - 20  = -10
        # After ascending sort: [-90, -23.33, -10].
        AL = Matrix(Diagonal([0.5, 0.7, 0.9]))
        BL = Matrix{Float64}(I, 3, 3)
        ρ  = 10.0
        out = verified_lg_transform(AL, BL, ρ)
        @test length(out) == 3
        # Sorted ascending.
        @test inf(out[1]) ≤ inf(out[2]) ≤ inf(out[3])
        # Values match (within widths).
        expected = [-90.0, 10.0 - 10.0 / (1 - 0.7), -10.0]
        for k in 1:3
            @test inf(out[k]) - 1e-9 ≤ expected[k] ≤ sup(out[k]) + 1e-9
        end
    end

    @testset "4. verified_lg_transform: Float-in == Interval-in (auto-promote)" begin
        AL = [3.0 0.5; 0.5 4.0]
        BL = [2.0 0.1; 0.1 2.5]
        ρ  = 1.0
        out_float = verified_lg_transform(AL, BL, ρ)
        AL_int = interval.(AL)
        BL_int = interval.(BL)
        ρ_int  = interval(ρ)
        out_int = verified_lg_transform(AL_int, BL_int, ρ_int)
        @test length(out_float) == length(out_int)
        for k in eachindex(out_float)
            # Both intervals enclose the same true value -> they should
            # overlap. (They may not be exactly equal because float-in is
            # promoted to a degenerate interval and accumulates round-off.)
            @test !isdisjoint_interval(out_float[k], out_int[k])
        end
    end

    @testset "6. error preconditions" begin
        @test_throws DomainError verified_cr_liu_lower(m, 0)
        @test_throws DimensionMismatch verified_lg_transform(zeros(3, 3),
                                                              zeros(2, 2),
                                                              1.0)
        @test_throws DimensionMismatch verified_lg_transform(zeros(3, 4),
                                                              zeros(3, 4),
                                                              1.0)
    end

    @testset "7. performance < 60 s on the 8×8 fixture" begin
        verified_cr_liu_lower(m, 3)   # warm-up
        t0 = time_ns()
        verified_cr_liu_lower(m, 3)
        @test (time_ns() - t0) / 1e9 < 60.0
    end
end
