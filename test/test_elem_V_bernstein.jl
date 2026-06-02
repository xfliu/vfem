# test/test_elem_V_bernstein.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. Constant V ≡ c: every entry of the 15-column row equals c.
#   2. Linear V(x, y) = a·x + b·y + c0: Bernstein interpolant is
#      exact, so the row reproduces V at the 15 control points.
#   3. UnitSquare8x8 fixture: V = x² + y² → MATLAB sum and max match.
#   4. Multi-index ordering: index 1 ↔ (4,0,0), 11 ↔ (0,4,0), 15 ↔ (0,0,4).
#   5. Performance: < 50 ms on UnitSquare8x8.

using Test
using VFEM: mesh2d_load, elem_V_bernstein, bernstein4_multiindices_2d

const _EVB_DIR = joinpath(@__DIR__, "fixtures", "unit_square_8x8")

function _load_evb_fixture()
    f = Dict{String, Float64}(); s = Dict{String, String}()
    for line in eachline(joinpath(_EVB_DIR, "cecr_lag_ref.txt"))
        line = strip(line)
        if startswith(line, "V_bern_sum=")
            f["V_bern_sum"] = parse(Float64, line[12:end])
        elseif startswith(line, "V_bern_max=")
            f["V_bern_max"] = parse(Float64, line[12:end])
        elseif startswith(line, "V_bern_size=")
            s["V_bern_size"] = line[13:end]
        end
    end
    return f, s
end

@testset "elem_V_bernstein" begin
    @testset "1. constant V" begin
        m = mesh2d_load(_EVB_DIR)
        Vb = elem_V_bernstein(m, (x, y) -> 3.14)
        @test size(Vb) == (m.nt, 15)
        @test all(abs.(Vb .- 3.14) .< 1e-14)
    end

    @testset "2. linear V exact at control points" begin
        m = mesh2d_load(_EVB_DIR)
        a, b, c0 = 1.5, -0.7, 0.3
        Vb = elem_V_bernstein(m, (x, y) -> a * x + b * y + c0)
        bern4 = bernstein4_multiindices_2d()
        for k in 1:m.nt
            v1 = (m.nodes[m.elements[k, 1], 1], m.nodes[m.elements[k, 1], 2])
            v2 = (m.nodes[m.elements[k, 2], 1], m.nodes[m.elements[k, 2], 2])
            v3 = (m.nodes[m.elements[k, 3], 1], m.nodes[m.elements[k, 3], 2])
            for r in 1:15
                a_, b_, c_ = bern4[r, 1], bern4[r, 2], bern4[r, 3]
                xq = (a_ * v1[1] + b_ * v2[1] + c_ * v3[1]) / 4
                yq = (a_ * v1[2] + b_ * v2[2] + c_ * v3[2]) / 4
                @test Vb[k, r] ≈ a * xq + b * yq + c0 atol = 1e-13
            end
        end
    end

    @testset "3. fixture: V = x²+y² sum and max match MATLAB" begin
        floats, sizes = _load_evb_fixture()
        m = mesh2d_load(_EVB_DIR)
        Vb = elem_V_bernstein(m, (x, y) -> x * x + y * y)
        @test sum(Vb) ≈ floats["V_bern_sum"] atol = 1e-10 rtol = 1e-10
        @test maximum(Vb) ≈ floats["V_bern_max"] atol = 1e-12
        @test sizes["V_bern_size"] == "$(size(Vb, 1))x$(size(Vb, 2))"
    end

    @testset "4. multi-index ordering at indices 1, 11, 15" begin
        bern4 = bernstein4_multiindices_2d()
        @test (bern4[1, 1], bern4[1, 2], bern4[1, 3]) == (4, 0, 0)
        @test (bern4[11, 1], bern4[11, 2], bern4[11, 3]) == (0, 4, 0)
        @test (bern4[15, 1], bern4[15, 2], bern4[15, 3]) == (0, 0, 4)
    end

    @testset "5. performance < 50 ms" begin
        m = mesh2d_load(_EVB_DIR)
        elem_V_bernstein(m, (x, y) -> x * x + y * y)
        T = 5
        t0 = time_ns()
        for _ in 1:T
            elem_V_bernstein(m, (x, y) -> x * x + y * y)
        end
        tns = (time_ns() - t0) / T
        @test tns < 50_000_000
    end
end
