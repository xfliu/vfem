#!/usr/bin/env julia

using IntervalArithmetic: inf, sup
using Printf
using VFEM

const EXACT_LAMBDA1 = (pi^2 / 4.0) * 80.0

function run_case(; interval_mode::Bool = true)
    m = special_tetrahedron_red_mesh(0)
    println("Special tetrahedron one-piece CG6+RT6 Lehmann-Goerisch")
    @printf("exact lambda_1 = %.15f\n", EXACT_LAMBDA1)
    @printf("mesh: nodes=%d tets=%d facets=%d edges=%d\n",
            m.NumNode, m.NumElt, m.NumF, m.NumEdge)

    cg = laplace_eig_lagrange_3d(m, 6, 1)
    rt = create_matrix_rt_3d(m, 6)
    @printf("CG6 interior DOFs = %d, upper = %.15f\n",
            length(cg.dof.interior_dofs), cg.eig_value[1])
    @printf("RT6 DimRT = %d, DimDG = %d\n\n", rt.DimRT, rt.DimDG)

    for rho in (220.0, 250.0, 280.0, 300.0, 320.0, 330.0, 340.0)
        r = lg_lower_eig_bound_laplace_3d(m, 6, 1; RT_order = 6, rho = rho)
        @printf("rho=%6.1f  lower=%18.12f  upper=%18.12f  exact_inside=%s\n",
                rho, r.eig_lower[1], r.eig_upper[1],
                string(r.eig_lower[1] < EXACT_LAMBDA1 < r.eig_upper[1]))
    end

    if interval_mode
        println("\ninterval check at rho=340")
        r = verified_lg_lower_eig_bound_laplace_3d(m, 6, 1;
                                                   RT_order = 6, rho = 340.0)
        @printf("lower=[%.15f, %.15f]\n", inf(r.eig_lower[1]), sup(r.eig_lower[1]))
        @printf("upper=[%.15f, %.15f]\n", inf(r.eig_upper[1]), sup(r.eig_upper[1]))
        @printf("A2=[%.15f, %.15f]\n", inf(r.A2[1, 1]), sup(r.A2[1, 1]))
        @printf("A2 width=%.3e\n", sup(r.A2[1, 1]) - inf(r.A2[1, 1]))
    end
end

run_case(interval_mode = get(ENV, "VFEM_INTERVAL", "1") != "0")
