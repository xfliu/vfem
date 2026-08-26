#!/usr/bin/env julia

using Printf
using IntervalArithmetic: inf, sup
using VFEM

const EXACT_LAMBDA1 = (pi^2 / 4.0) * 80.0

function run_case(; interval_mode::Bool = true)
    m0 = special_tetrahedron_mesh()
    m1 = red_refine_mesh_3d(m0)

    println("Special tetrahedron CG3+RT3 Lehmann-Goerisch refinement scan")
    @printf("exact lambda_1 = %.15f\n", EXACT_LAMBDA1)
    @printf("coarse centroid mesh: nodes=%d tets=%d hmax=%.12f\n",
            m0.NumNode, m0.NumElt, find_mesh_hmax_3d(m0))
    @printf("one red refinement: nodes=%d tets=%d hmax=%.12f\n\n",
            m1.NumNode, m1.NumElt, find_mesh_hmax_3d(m1))

    for rho in (220.0, 250.0, 280.0, 320.0, 340.0)
        r = lg_lower_eig_bound_laplace_3d(m1, 3, 1; RT_order = 3, rho = rho)
        @printf("rho=%6.1f  lower=%18.12f  upper=%18.12f  exact_inside=%s\n",
                rho, r.eig_lower[1], r.eig_upper[1],
                string(r.eig_lower[1] < EXACT_LAMBDA1 < r.eig_upper[1]))
    end

    if interval_mode
        println("\ninterval check at rho=340")
        r = verified_lg_lower_eig_bound_laplace_3d(m1, 3, 1;
                                                   RT_order = 3, rho = 340.0)
        @printf("lower=[%.15f, %.15f]\n", inf(r.eig_lower[1]), sup(r.eig_lower[1]))
        @printf("upper=[%.15f, %.15f]\n", inf(r.eig_upper[1]), sup(r.eig_upper[1]))
        @printf("A2 width=%.3e\n", sup(r.A2[1, 1]) - inf(r.A2[1, 1]))
    end
end

run_case(interval_mode = get(ENV, "VFEM_INTERVAL", "1") != "0")
