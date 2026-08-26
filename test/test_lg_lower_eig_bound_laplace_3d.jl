# test/test_lg_lower_eig_bound_laplace_3d.jl
#
# CORNER-CASE TAXONOMY:
#   1. Float64 RT auxiliary solve produces a small divergence residual.
#   2. Interval RT auxiliary solve encloses the Float64 A2 value.
#   3. CG3+RT3 Lehmann-Goerisch driver brackets the closed-form first
#      eigenvalue on the special tetrahedron with an explicit separation rho.
#   4. Final interval LG driver gives an interval lower bound below the
#      closed-form value and keeps the conforming upper bound above it.

using Test
using IntervalArithmetic: inf, sup
using LinearAlgebra: norm
import VFEM
using VFEM: special_tetrahedron_mesh, create_matrix_lagrange_3d,
            laplace_eig_lagrange_3d, rt_hdiv_problem_3d,
            verified_rt_hdiv_problem_3d,
            lg_lower_eig_bound_laplace_3d,
            verified_lg_lower_eig_bound_laplace_3d

@testset "lg_lower_eig_bound_laplace_3d" begin
    m = special_tetrahedron_mesh()
    exact_lambda1 = (pi^2 / 4) * 80.0
    rho = 250.0

    A, M, info, L2G = create_matrix_lagrange_3d(m, 3)
    cg = laplace_eig_lagrange_3d(m, 3, 1)

    @testset "1. RT auxiliary residual" begin
        A2, rt, W = rt_hdiv_problem_3d(m, 3, cg.eig_func, L2G)
        rhs = Matrix(rt.M_dg) * VFEM._cg_to_dg_3d(cg.eig_func[:, 1], L2G, rt)
        res = norm(Matrix(rt.B_rt) * W[:, 1] + rhs) / norm(rhs)
        @test size(A2) == (1, 1)
        @test res < 1e-8
    end

    @testset "2. interval RT auxiliary encloses Float64 A2" begin
        A2, _, _ = rt_hdiv_problem_3d(m, 3, cg.eig_func, L2G)
        A2_int, _, _ = verified_rt_hdiv_problem_3d(m, 3, cg.eig_func, L2G)
        @test inf(A2_int[1, 1]) <= A2[1, 1] <= sup(A2_int[1, 1])
        @test sup(A2_int[1, 1]) - inf(A2_int[1, 1]) < 1e-10
    end

    @testset "3. Float64 LG special tetrahedron bracket" begin
        r = lg_lower_eig_bound_laplace_3d(m, 3, 1; RT_order = 3, rho = rho)
        @test r.eig_lower[1] < exact_lambda1 < r.eig_upper[1]
        @test r.rho == rho
    end

    @testset "4. interval LG special tetrahedron bracket" begin
        r = verified_lg_lower_eig_bound_laplace_3d(m, 3, 1; RT_order = 3, rho = rho)
        @test inf(r.eig_lower[1]) < exact_lambda1 < sup(r.eig_upper[1])
        @test sup(r.eig_lower[1]) < exact_lambda1
    end
end
