#!/usr/bin/env julia

using LinearAlgebra: cond
using Printf
using VFEM

exact = (pi^2 / 4.0) .* [80.0, 140.0, 140.0, 160.0, 208.0]
result = one_piece_bubble_laplace_3d(special_tetrahedron_vertices(), 12, 5)

println("Special tetrahedron one-piece bubble polynomial solve")
println("space: lambda1*lambda2*lambda3*lambda4 * P_8, total degree 12")
@printf("DOFs: %d\n", size(result.A, 1))
@printf("cond(M): %.6e\n\n", cond(result.M))
println(" k        Ritz value             exact value          relative error")
for k in 1:5
    relerr = abs(result.eig_value[k] - exact[k]) / exact[k]
    @printf("%2d  %22.14f  %22.14f  %.6e\n",
            k, result.eig_value[k], exact[k], relerr)
end
