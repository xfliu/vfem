# test/test_schrodinger_eig_cecr_3d.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. cube_r1, V = 0, neig = 4: Liu's CR Liu lower bound `eig_lower` and
#      raw CECR `eig_h` match MATLAB to ≤ 1e-9. C_h matches to ≤ 1e-12.
#   2. cube_r1, V(x,y,z) = x²+y²+z², neig = 4: same comparison. The
#      reaction part is sampled at element centroids exactly as MATLAB
#      `c_data(e) = norm(centroid(e))^2`.
#   3. eig_lower < eig_h component-wise (Liu shift always reduces).
#   4. With γ_h = 0 and V ≥ 0, `eig_lower[k] = eig_h[k] / (1 + Ch²·eig_h[k])`.
#   5. Vector V_input (precomputed c_data) gives the same answer as
#      function-handle V_input that produces the same per-element values.
#   6. neig = 0 -> DomainError; bc = :foo -> ArgumentError; vector
#      V_input of wrong length -> DimensionMismatch.
#   7. Performance: < 5 s on this fixture.
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   For the Dirichlet truncated Schrödinger operator on a 3D domain Ω,
#   every CECR FE eigenvalue ν_h is a numerical upper bound on the
#   truncated problem and the Liu correction
#       λ_lower = ν / (1 + Ch² · ν)        (γ_h = 0)
#   gives a guaranteed lower bound, with `Ch = h_max / √40`.

using Test
using VFEM: mesh_load_from_folder, schrodinger_eig_cecr_3d,
            SchrodingerEig3D, find_mesh_hmax_3d

const _SCHR3D_DIR = joinpath(@__DIR__, "fixtures", "cube_r1")

function _load_schr3d_fixture()
    cases = Dict{String, Dict{String, Float64}}()
    cur = ""
    for line in eachline(joinpath(_SCHR3D_DIR, "schrodinger3d_ref.txt"))
        line = strip(line)
        isempty(line) && continue
        if (mm = match(r"^case=(.+)$", line)) !== nothing
            cur = mm[1]
            cases[cur] = Dict{String, Float64}()
            continue
        end
        if (mm = match(r"^([A-Za-z_0-9]+)=([+-]?\d+\.\d+e[+-]?\d+)$", line)) !== nothing
            cases[cur][mm[1]] = parse(Float64, mm[2])
        end
    end
    # h_max / C_h are stored under v_zero; lift to top-level.
    cases[""]   = Dict{String, Float64}()
    cases[""]["h_max"] = cases["v_zero"]["h_max"]
    cases[""]["C_h"]   = cases["v_zero"]["C_h"]
    return cases
end

@testset "schrodinger_eig_cecr_3d" begin
    cases = _load_schr3d_fixture()
    m = mesh_load_from_folder(_SCHR3D_DIR)

    @testset "1. V = 0, neig = 4 matches MATLAB" begin
        r = schrodinger_eig_cecr_3d(m, (x, y, z) -> 0.0, 4)
        @test r isa SchrodingerEig3D
        ref = cases["v_zero"]
        @test r.Ch ≈ cases[""]["C_h"] atol = 1e-12 rtol = 1e-12
        @test find_mesh_hmax_3d(m) ≈ cases[""]["h_max"] atol = 1e-12
        for k in 1:4
            @test r.eig_h[k]     ≈ ref["eig_h_$k"]     atol = 1e-9 rtol = 1e-10
            @test r.eig_lower[k] ≈ ref["eig_lower_$k"] atol = 1e-9 rtol = 1e-10
        end
        @test r.gamma_h ≈ 0.0 atol = 1e-14
    end

    @testset "2. V = x²+y²+z² matches MATLAB" begin
        r = schrodinger_eig_cecr_3d(m, (x, y, z) -> x^2 + y^2 + z^2, 4)
        ref = cases["v_sq"]
        for k in 1:4
            @test r.eig_h[k]     ≈ ref["eig_h_$k"]     atol = 1e-9 rtol = 1e-10
            @test r.eig_lower[k] ≈ ref["eig_lower_$k"] atol = 1e-9 rtol = 1e-10
        end
    end

    @testset "3. eig_lower < eig_h" begin
        r = schrodinger_eig_cecr_3d(m, (x, y, z) -> 0.0, 4)
        for k in 1:4
            @test r.eig_lower[k] < r.eig_h[k]
        end
    end

    @testset "4. Liu formula at γ_h = 0" begin
        r = schrodinger_eig_cecr_3d(m, (x, y, z) -> 0.0, 4)
        for k in 1:4
            expected = r.eig_h[k] / (1 + r.eig_h[k] * r.Ch^2)
            @test r.eig_lower[k] ≈ expected atol = 1e-13 rtol = 1e-13
        end
    end

    @testset "5. vector V_input vs function V_input" begin
        # Precompute c_data per centroid and compare.
        c = Vector{Float64}(undef, m.NumElt)
        for e in 1:m.NumElt
            xc = sum(m.NodeList[m.ElementList[e, k], 1] for k in 1:4) / 4
            yc = sum(m.NodeList[m.ElementList[e, k], 2] for k in 1:4) / 4
            zc = sum(m.NodeList[m.ElementList[e, k], 3] for k in 1:4) / 4
            c[e] = xc^2 + yc^2 + zc^2
        end
        r_vec  = schrodinger_eig_cecr_3d(m, c, 4)
        r_func = schrodinger_eig_cecr_3d(m, (x, y, z) -> x^2 + y^2 + z^2, 4)
        for k in 1:4
            @test r_vec.eig_h[k] ≈ r_func.eig_h[k] atol = 1e-12 rtol = 1e-13
        end
    end

    @testset "6. error preconditions" begin
        @test_throws DomainError schrodinger_eig_cecr_3d(m, (x, y, z) -> 0.0, 0)
        @test_throws ArgumentError schrodinger_eig_cecr_3d(m, (x, y, z) -> 0.0, 4; bc = :foo)
        @test_throws DimensionMismatch schrodinger_eig_cecr_3d(m, ones(m.NumElt + 1), 4)
    end

    @testset "7. performance < 5 s" begin
        schrodinger_eig_cecr_3d(m, (x, y, z) -> 0.0, 4)
        t0 = time_ns()
        schrodinger_eig_cecr_3d(m, (x, y, z) -> 0.0, 4)
        @test (time_ns() - t0) / 1e9 < 5.0
    end
end
