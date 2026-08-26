#!/usr/bin/env julia

using LinearAlgebra
using Printf

const ROOT = abspath(joinpath(@__DIR__, ".."))
import Pkg
Pkg.activate(ROOT; io = devnull)

using VFEM

const MATLAB_OUT = joinpath(@__DIR__, "reference", "vfem3d_matlab_debug.txt")

function read_kv(path)
    d = Dict{String, Float64}()
    for line in eachline(path)
        isempty(strip(line)) && continue
        k, v = split(strip(line), "="; limit = 2)
        d[k] = parse(Float64, v)
    end
    return d
end

function check_close(label, got, ref; atol = 1e-9, rtol = 1e-9)
    ok = isapprox(got, ref; atol = atol, rtol = rtol)
    @printf("%-18s Julia=% .16e  MATLAB=% .16e  diff=% .3e  %s\n",
            label, got, ref, got - ref, ok ? "OK" : "FAIL")
    return ok
end

function main()
    isfile(MATLAB_OUT) || error("Missing MATLAB export: $MATLAB_OUT")
    ref = read_kv(MATLAB_OUT)

    m = special_tetrahedron_mesh()
    A, M, info, _ = create_matrix_lagrange_3d(m, 2)
    int = info.interior_dofs
    λ = eigen(Symmetric(Matrix(A[int, int])), Symmetric(Matrix(M[int, int]))).values
    rt = create_matrix_rt_3d(m, 1)

    println("VFEM3D MATLAB-vs-Julia debug comparison")
    println(repeat("=", 78))
    ok = true
    ok &= check_close("NumNode", m.NumNode, ref["NumNode"]; atol = 0)
    ok &= check_close("NumElt", m.NumElt, ref["NumElt"]; atol = 0)
    ok &= check_close("NumF", m.NumF, ref["NumF"]; atol = 0)
    ok &= check_close("NumEdge", m.NumEdge, ref["NumEdge"]; atol = 0)
    ok &= check_close("CG2_DimCG", info.DimCG, ref["CG2_DimCG"]; atol = 0)
    ok &= check_close("CG2_NumBD", length(info.bd_dofs), ref["CG2_NumBD"]; atol = 0)
    ok &= check_close("CG2_NumInt", length(info.interior_dofs), ref["CG2_NumInt"]; atol = 0)
    ok &= check_close("CG2_A_trace", tr(A), ref["CG2_A_trace"])
    ok &= check_close("CG2_A_sum", sum(A), ref["CG2_A_sum"])
    ok &= check_close("CG2_A_frob", norm(A), ref["CG2_A_frob"])
    ok &= check_close("CG2_M_trace", tr(M), ref["CG2_M_trace"])
    ok &= check_close("CG2_M_sum", sum(M), ref["CG2_M_sum"])
    ok &= check_close("CG2_M_frob", norm(M), ref["CG2_M_frob"])
    ok &= check_close("CG2_lambda1", sort(λ)[1], ref["CG2_lambda1"])
    ok &= check_close("RT1_DimRT", rt.DimRT, ref["RT1_DimRT"]; atol = 0)
    ok &= check_close("RT1_DimDG", rt.DimDG, ref["RT1_DimDG"]; atol = 0)
    ok &= check_close("RT1_DegK", rt.DegK, ref["RT1_DegK"]; atol = 0)
    ok &= check_close("RT1_A_trace", tr(rt.A_rt), ref["RT1_A_trace"]; atol = 1e-8, rtol = 1e-8)
    ok &= check_close("RT1_A_sum", sum(rt.A_rt), ref["RT1_A_sum"]; atol = 1e-8, rtol = 1e-8)
    ok &= check_close("RT1_A_frob", norm(rt.A_rt), ref["RT1_A_frob"]; atol = 1e-8, rtol = 1e-8)
    ok &= check_close("RT1_B_sum", sum(rt.B_rt), ref["RT1_B_sum"]; atol = 1e-8, rtol = 1e-8)
    ok &= check_close("RT1_B_frob", norm(rt.B_rt), ref["RT1_B_frob"]; atol = 1e-8, rtol = 1e-8)
    ok &= check_close("RT1_Mdg_trace", tr(rt.M_dg), ref["RT1_Mdg_trace"])
    ok &= check_close("RT1_Mdg_sum", sum(rt.M_dg), ref["RT1_Mdg_sum"])
    ok &= check_close("RT1_Mdg_frob", norm(rt.M_dg), ref["RT1_Mdg_frob"])
    ok || error("MATLAB-vs-Julia comparison failed")
    println("All comparisons passed.")
end

main()
