# test/test_laplace_eig_lagrange.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. P1 on UnitSquare8x8: matrix invariants (size, trace, sum, Frobenius
#      for both A and M), the first 6 Dirichlet Laplace eigenvalues, and
#      the per-column norms of `eig_func` match MATLAB to ~1e-10 / ~1e-9.
#   2. P2 on UnitSquare8x8: same comparison, with both Julia and MATLAB
#      using the monomial Lagrange basis φ_α = L^α (the basis consumed
#      by `rt_hdiv_problem`).
#   3. λ_h is above the analytic 2π² for the smallest eigenvalue (Galerkin
#      upper bound) and P2 is sharper than P1.
#   4. Boundary DOFs are correctly set: P1 removes only boundary-vertex
#      DOFs (32 on an 8×8 grid); P2 also removes boundary-edge midpoints.
#   5. Returned `A`, `M` are square ndof × ndof; `eig_func` is ndof × neig
#      with zeros on the rows of the Dirichlet DOFs.
#   6. order = 0 -> ArgumentError; neig = 0 -> DomainError.
#   7. Performance: < 5 s on this fixture for both P1 and P2.
#
# MATHEMATICAL CONTRACT (docs/testing-contract.md (c)):
#   For the homogeneous Dirichlet Laplacian on the unit square, the
#   first eigenvalue is 2π² ≈ 19.7392. The Lagrange CG approximation
#   converges from above: λ_h ≥ λ_true. The lowest CG eigenvalue
#   satisfies λ_h(P2) < λ_h(P1) (P2 is sharper).

using Test
using LinearAlgebra: norm, tr
using VFEM: mesh2d_load, laplace_eig_lagrange, LaplaceEigLagrange

const _LEL_DIR = joinpath(@__DIR__, "fixtures", "unit_square_8x8")

# Parse the per-order fixture: lines come in two blocks separated by
# `order=1` / `order=2` markers. Returns ((f1, i1), (f2, i2)).
function _load_lel_fixture()
    f1 = Dict{String, Float64}(); f2 = Dict{String, Float64}()
    i1 = Dict{String, Int}();     i2 = Dict{String, Int}()
    cur = 0
    for line in eachline(joinpath(_LEL_DIR, "laplace_eig_lagrange_ref.txt"))
        line = strip(line)
        isempty(line) && continue
        if startswith(line, "order=")
            cur = parse(Int, line[7:end]); continue
        end
        if (mm = match(r"^([A-Za-z_0-9]+)=([+-]?\d+\.\d+e[+-]?\d+)$", line)) !== nothing
            (cur == 1 ? f1 : f2)[mm[1]] = parse(Float64, mm[2])
        elseif (mm = match(r"^([A-Za-z_0-9]+)=(\d+)$", line)) !== nothing
            (cur == 1 ? i1 : i2)[mm[1]] = parse(Int, mm[2])
        end
    end
    return (f1, i1), (f2, i2)
end

@testset "laplace_eig_lagrange" begin
    (f1, i1), (f2, i2) = _load_lel_fixture()
    m = mesh2d_load(_LEL_DIR)

    @testset "1. P1: matrices + eigenvalues + eigfunc norms match MATLAB" begin
        r = laplace_eig_lagrange(m, 1, 6)
        @test r isa LaplaceEigLagrange
        @test size(r.A, 1) == i1["A_size"] == m.nv == 81
        @test size(r.A, 2) == size(r.A, 1)
        @test tr(r.A)   ≈ f1["A_trace"]  atol = 1e-9 rtol = 1e-10
        @test sum(r.A)  ≈ f1["A_sum"]    atol = 1e-9
        @test norm(r.A) ≈ f1["A_frob"]   atol = 1e-9 rtol = 1e-10
        @test tr(r.M)   ≈ f1["M_trace"]  atol = 1e-12
        @test sum(r.M)  ≈ f1["M_sum"]    atol = 1e-12
        @test norm(r.M) ≈ f1["M_frob"]   atol = 1e-12
        @test length(r.eig_value) == 6
        for k in 1:6
            @test r.eig_value[k] ≈ f1["lam_1_$k"] atol = 1e-9 rtol = 1e-9
        end
        @test norm(r.eig_func) ≈ f1["efunc_1_frob"] atol = 1e-8 rtol = 1e-9
        for k in 1:6
            @test norm(@view r.eig_func[:, k]) ≈ f1["efunc_1_colnorm_$k"] atol = 1e-8 rtol = 1e-9
        end
    end

    @testset "2. P2: matrices + eigenvalues + eigfunc norms match MATLAB" begin
        r = laplace_eig_lagrange(m, 2, 6)
        @test size(r.A, 1) == i2["A_size"] == m.nv + m.ne == 289
        @test tr(r.A)   ≈ f2["A_trace"]  atol = 1e-9 rtol = 1e-10
        @test sum(r.A)  ≈ f2["A_sum"]    atol = 1e-9 rtol = 1e-10
        @test norm(r.A) ≈ f2["A_frob"]   atol = 1e-9 rtol = 1e-10
        @test tr(r.M)   ≈ f2["M_trace"]  atol = 1e-12
        @test sum(r.M)  ≈ f2["M_sum"]    atol = 1e-12
        @test norm(r.M) ≈ f2["M_frob"]   atol = 1e-12
        for k in 1:6
            @test r.eig_value[k] ≈ f2["lam_2_$k"] atol = 1e-9 rtol = 1e-9
        end
        @test norm(r.eig_func) ≈ f2["efunc_2_frob"] atol = 1e-8 rtol = 1e-9
        for k in 1:6
            @test norm(@view r.eig_func[:, k]) ≈ f2["efunc_2_colnorm_$k"] atol = 1e-8 rtol = 1e-9
        end
    end

    @testset "3. λ_h(P2) < λ_h(P1) and both > 2π²" begin
        r1 = laplace_eig_lagrange(m, 1, 1)
        r2 = laplace_eig_lagrange(m, 2, 1)
        @test r1.eig_value[1] > 2π^2 - 1e-10
        @test r2.eig_value[1] > 2π^2 - 1e-10
        @test r2.eig_value[1] < r1.eig_value[1]
    end

    @testset "4. boundary DOF set has the right cardinality" begin
        r1 = laplace_eig_lagrange(m, 1, 1)
        @test length(r1.bd_dofs) == 32                  # 4·8 boundary vertices
        r2 = laplace_eig_lagrange(m, 2, 1)
        @test length(r2.bd_dofs) == 32 + m.nb           # + boundary-edge mids
    end

    @testset "5. shape + zero rows on Dirichlet" begin
        r = laplace_eig_lagrange(m, 2, 4)
        @test size(r.eig_func) == (m.nv + m.ne, 4)
        @test all(iszero, r.eig_func[r.bd_dofs, :])
    end

    @testset "6. error preconditions" begin
        @test_throws ArgumentError laplace_eig_lagrange(m, 0, 6)
        @test_throws DomainError   laplace_eig_lagrange(m, 1, 0)
    end

    @testset "7. performance < 5 s for P1 + P2" begin
        laplace_eig_lagrange(m, 1, 6)
        t0 = time_ns()
        laplace_eig_lagrange(m, 1, 6)
        @test (time_ns() - t0) / 1e9 < 5.0

        laplace_eig_lagrange(m, 2, 6)
        t0 = time_ns()
        laplace_eig_lagrange(m, 2, 6)
        @test (time_ns() - t0) / 1e9 < 5.0
    end
end
