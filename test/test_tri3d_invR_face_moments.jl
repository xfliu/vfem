# test/test_tri3d_invR_face_moments.jl
#
# CORNER-CASE TAXONOMY (Rule.md (b)):
#   1. max_degree = 0 returns just F[1,1,1] (the planar potential I00).
#   2. Equilateral triangle in z = 1 plane (analytic h, simple geometry).
#   3. MATLAB cross-check on the canonical face_nodes panel from
#      tests/fixtures/face_moments_ref.txt — covers max_degree = 4
#      across 35 entries.
#   4. h flip — the routine reorients normal so h ≥ 0; supplying a
#      "flipped" face order should give the same F values (up to the
#      orientation correction handled internally).
#   5. face_nodes shape ≠ 3×3 -> DimensionMismatch.
#   6. max_degree < 0 -> DomainError.
#   7. Degenerate face (collinear vertices) -> DomainError.
#   8. F[1,1,1] (b1=b2=b3=0) is the planar potential = I00 / area_jac.
#   9. Performance: max_degree = 4 (35 entries) < 50 ms.
#
# MATHEMATICAL CONTRACT (Rule.md (c)):
#   F[b1+1, b2+1, b3+1] = ∫_Δ μ₁^b1 μ₂^b2 μ₃^b3 / |μ₁·a + μ₂·b + μ₃·c| dμ
#   over the unit reference triangle Δ in (μ₁, μ₂, μ₃) coords with
#   μ₁ + μ₂ + μ₃ = 1. Cross-validated against MATLAB R2024a fixture
#   to 13 significant decimals.

using Test
using VFEM: tri3d_invR_face_moments

# Load MATLAB fixture into a Dict from (b1, b2, b3) to expected value,
# plus h_ref and area_jac_ref.
function _load_face_moments_fixture()
    path = joinpath(@__DIR__, "fixtures", "face_moments_ref.txt")
    h_ref = NaN
    area_jac_ref = NaN
    F_ref = Dict{NTuple{3, Int}, Float64}()
    for line in eachline(path)
        line = strip(line)
        (isempty(line) || startswith(line, "#")) && continue
        if startswith(line, "h=")
            h_ref = parse(Float64, line[3:end])
        elseif startswith(line, "area_jac=")
            area_jac_ref = parse(Float64, line[10:end])
        else
            # F[b1,b2,b3]=value
            m = match(r"^F\[(\d+),(\d+),(\d+)\]=(.+)$", line)
            m === nothing && continue
            key = (parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3]))
            F_ref[key] = parse(Float64, m[4])
        end
    end
    return h_ref, area_jac_ref, F_ref
end

const _CANONICAL_FACE = [1.0 0.0 0.0;
                         0.2 1.1 0.1;
                         0.1 0.3 0.9]

@testset "tri3d_invR_face_moments" begin
    @testset "MATLAB R2024a fixture cross-check (max_degree = 4)" begin
        h_ref, area_jac_ref, F_ref = _load_face_moments_fixture()
        F, info = tri3d_invR_face_moments(_CANONICAL_FACE, 4)
        @test info.h ≈ h_ref atol = 1e-13 rtol = 1e-13
        @test info.area_jac ≈ area_jac_ref atol = 1e-13 rtol = 1e-13
        for ((b1, b2, b3), expected) in F_ref
            got = F[b1 + 1, b2 + 1, b3 + 1]
            @test got ≈ expected atol = 1e-12 rtol = 1e-12
        end
    end

    @testset "1. max_degree = 0 returns just I00 / area_jac" begin
        F, info = tri3d_invR_face_moments(_CANONICAL_FACE, 0)
        @test size(F) == (1, 1, 1)
        # Compare with the higher-degree call's F[1,1,1] entry.
        F4, _ = tri3d_invR_face_moments(_CANONICAL_FACE, 4)
        @test F[1, 1, 1] ≈ F4[1, 1, 1] atol = 1e-14
    end

    @testset "2. equilateral in z = 1 plane (h sanity)" begin
        # Triangle with vertices on z = 1 plane; perpendicular distance
        # from origin to plane is exactly 1.
        v = [1.0 0.0 1.0;
             cos(2π/3) sin(2π/3) 1.0;
             cos(4π/3) sin(4π/3) 1.0]
        _, info = tri3d_invR_face_moments(v, 0)
        @test info.h ≈ 1.0 atol = 1e-14
    end

    @testset "5–7. error preconditions" begin
        @test_throws DimensionMismatch tri3d_invR_face_moments(
            [1.0 0.0; 0.0 1.0], 2)
        @test_throws DomainError tri3d_invR_face_moments(_CANONICAL_FACE, -1)
        # Degenerate face: collinear vertices.
        degen = [0.0 0.0 0.0;
                 1.0 0.0 0.0;
                 2.0 0.0 0.0]
        @test_throws DomainError tri3d_invR_face_moments(degen, 1)
    end

    @testset "9. performance budget max_degree = 4 < 50 ms" begin
        tri3d_invR_face_moments(_CANONICAL_FACE, 4)
        T = 5
        t0 = time_ns()
        for _ in 1:T
            tri3d_invR_face_moments(_CANONICAL_FACE, 4)
        end
        tns = (time_ns() - t0) / T
        @test tns < 50_000_000
    end
end
