# examples/laplace_equilateral_triangle.jl
#
# Worked example: Dirichlet Laplace eigenvalues and eigenfunctions on
# an equilateral triangle of side 1. Compares Julia FE results to the
# closed-form Lamé spectrum derived in
#
#   B. J. McCartin, *Eigenstructure of the equilateral triangle, part I:
#   the Dirichlet problem*, SIAM Review 45 (2003), pp. 267–287.
#
# For an equilateral triangle of side a, the Dirichlet Laplacian
# eigenvalues are
#
#     λ_{m, n} = (16 π² / (9 a²)) · (m² + m n + n²),
#
# enumerated over integer pairs (m, n) with m ≥ n ≥ 1.  Multiplicity
# is 1 when m = n (symmetric mode only) and 2 when m > n (symmetric
# + antisymmetric).  Smallest few (a = 1):
#
#     (m, n) = (1, 1)   ⇒   m² + mn + n² = 3   ⇒   λ ≈ 52.638  (mult 1)
#     (m, n) = (2, 1)   ⇒                = 7   ⇒   λ ≈ 122.822 (mult 2)
#     (m, n) = (2, 2)   ⇒                = 12  ⇒   λ ≈ 210.553 (mult 1)
#     (m, n) = (3, 1)   ⇒                = 13  ⇒   λ ≈ 228.099 (mult 2)
#
# Run with:
#     julia --project=. examples/laplace_equilateral_triangle.jl
#
# Output: one block of numbers per refinement level — λ_h(P1), λ_h(P2),
# the Liu CR lower bound, and the analytical reference.

using LinearAlgebra: norm
using Printf
using DelimitedFiles: writedlm
using IntervalArithmetic: inf

# Activate the parent VFEM.jl package so this script runs from anywhere.
const _PKG_ROOT = abspath(joinpath(@__DIR__, ".."))
import Pkg
Pkg.activate(_PKG_ROOT; io = devnull)

using VFEM

# ----------------------------------------------------------------------------
# Mesh construction: refined equilateral triangle.
# ----------------------------------------------------------------------------

# Subdivide each triangle (v1, v2, v3) into 4 sub-triangles by taking
# edge midpoints. Shared midpoints are deduped via a hash on sorted
# vertex-pair keys.
function _refine_4way(nodes::Matrix{Float64}, elements::Matrix{Int})
    nv0 = size(nodes, 1)
    nt0 = size(elements, 1)

    # Mutable list of node coordinates, seeded with the existing nodes.
    new_nodes = [nodes[i, j] for i in 1:nv0, j in 1:2]
    new_nodes = Matrix{Float64}(new_nodes)
    midpoint_cache = Dict{Tuple{Int, Int}, Int}()

    function midpoint_index!(a::Int, b::Int)
        key = a < b ? (a, b) : (b, a)
        haskey(midpoint_cache, key) && return midpoint_cache[key]
        mx = (new_nodes[key[1], 1] + new_nodes[key[2], 1]) / 2
        my = (new_nodes[key[1], 2] + new_nodes[key[2], 2]) / 2
        new_nodes = vcat(new_nodes, [mx my])
        idx = size(new_nodes, 1)
        midpoint_cache[key] = idx
        return idx
    end

    new_elements = Vector{NTuple{3, Int}}()
    sizehint!(new_elements, 4 * nt0)
    for k in 1:nt0
        v1 = elements[k, 1]; v2 = elements[k, 2]; v3 = elements[k, 3]
        m12 = midpoint_index!(v1, v2)
        m23 = midpoint_index!(v2, v3)
        m31 = midpoint_index!(v3, v1)
        push!(new_elements, (v1, m12, m31))
        push!(new_elements, (m12, v2, m23))
        push!(new_elements, (m31, m23, v3))
        push!(new_elements, (m12, m23, m31))
    end

    elements_out = Matrix{Int}(undef, length(new_elements), 3)
    for (k, t) in enumerate(new_elements)
        elements_out[k, 1] = t[1]
        elements_out[k, 2] = t[2]
        elements_out[k, 3] = t[3]
    end
    return new_nodes, elements_out
end

# Build the four mesh-folder files (vert / tri / edge / bd) so
# `mesh2d_load` can ingest. Edges and boundary edges are derived from
# the element-vertex topology (each edge appearing in exactly one
# triangle is on the boundary).
function _build_mesh_files(nodes::Matrix{Float64},
                            elements::Matrix{Int}, dir::AbstractString)
    nt = size(elements, 1)

    edge_count = Dict{Tuple{Int, Int}, Int}()
    for k in 1:nt
        v = (elements[k, 1], elements[k, 2], elements[k, 3])
        for (a, b) in ((v[1], v[2]), (v[2], v[3]), (v[3], v[1]))
            key = a < b ? (a, b) : (b, a)
            edge_count[key] = get(edge_count, key, 0) + 1
        end
    end
    edges_list = sort(collect(keys(edge_count)))
    edges = reduce(vcat, [reshape([e[1], e[2]], 1, 2) for e in edges_list])

    bd_list = [e for e in edges_list if edge_count[e] == 1]
    bd_edges = reduce(vcat, [reshape([e[1], e[2]], 1, 2) for e in bd_list])

    mkpath(dir)
    writedlm(joinpath(dir, "vert.dat"), nodes)
    writedlm(joinpath(dir, "tri.dat"),  elements)
    writedlm(joinpath(dir, "edge.dat"), edges)
    writedlm(joinpath(dir, "bd.dat"),   bd_edges)
end

"""
    build_equilateral_mesh(refinement_levels; side = 1.0, dir = mktempdir())
        -> (Mesh2D, dir::String)

Build a uniformly-refined equilateral triangle mesh. `refinement_levels`
applies 4-way midpoint refinement that many times: level 0 has 1
triangle / 3 vertices, level k has `4^k` triangles.
"""
function build_equilateral_mesh(refinement_levels::Integer;
                                  side::Float64 = 1.0,
                                  dir::AbstractString = mktempdir())
    h = side * sqrt(3) / 2
    nodes = Float64[0.0      0.0;
                    side     0.0;
                    side / 2 h]
    elements = Int[1 2 3]

    for _ in 1:refinement_levels
        nodes, elements = _refine_4way(nodes, elements)
    end

    _build_mesh_files(nodes, elements, dir)
    return mesh2d_load(dir), dir
end

# ----------------------------------------------------------------------------
# Closed-form reference (McCartin 2003).
# ----------------------------------------------------------------------------

"""
    exact_equilateral_dirichlet_eigenvalues(n_eigs; side = 1.0)
        -> Vector{Float64}

First `n_eigs` exact Dirichlet Laplace eigenvalues of an equilateral
triangle of `side`. Enumerate (m, n) with m ≥ n ≥ 1: multiplicity is
1 when m = n (symmetric mode only, McCartin §3) and 2 when m > n
(one symmetric + one antisymmetric mode at the same λ).
"""
function exact_equilateral_dirichlet_eigenvalues(n_eigs::Integer; side::Float64 = 1.0)
    coef = 16π^2 / (9 * side^2)
    out = Tuple{Float64, Int, Int, Int}[]   # (λ, m, n, multiplicity)
    for m in 1:30, n in 1:30
        m ≥ n || continue
        λ = coef * (m^2 + m * n + n^2)
        mult = m == n ? 1 : 2
        push!(out, (λ, m, n, mult))
    end
    sort!(out; by = first)

    expanded = Float64[]
    for (λ, m, n, mult) in out
        for _ in 1:mult
            push!(expanded, λ)
        end
        length(expanded) ≥ n_eigs && break
    end
    return expanded[1:min(n_eigs, length(expanded))]
end

# ----------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------

function run_case(; refinement_levels::Vector{Int} = [3, 4, 5], neig::Int = 4)
    λ_exact = exact_equilateral_dirichlet_eigenvalues(neig)
    println("Dirichlet Laplace eigenvalues on the unit-side equilateral triangle")
    println(repeat("=", 78))
    @printf("%-6s  %-6s  %-12s  %-12s  %-12s  %-12s\n",
            "level", "nv", "λ_h(P1)", "λ_h(P2)", "λ_low(CR)", "λ_exact")
    println(repeat("-", 78))

    finest_mesh = nothing
    finest_r2   = nothing

    for level in refinement_levels
        m, _ = build_equilateral_mesh(level)
        r1 = laplace_eig_lagrange(m, 1, neig)
        r2 = laplace_eig_lagrange(m, 2, neig)
        cr_low, _ = verified_cr_liu_lower(m, neig)

        for k in 1:neig
            label_lvl = k == 1 ? @sprintf("L%d", level) : ""
            label_nv  = k == 1 ? string(m.nv) : ""
            @printf("%-6s  %-6s  %-12.6f  %-12.6f  %-12.6f  %-12.6f\n",
                    label_lvl, label_nv,
                    r1.eig_value[k], r2.eig_value[k],
                    inf(cr_low[k]), λ_exact[k])
        end
        println(repeat("-", 78))

        finest_mesh = m
        finest_r2   = r2
    end

    println()
    println("Finest level — P2 eigenfunction coefficient column norms:")
    for k in 1:neig
        @printf("    ‖φ_%d‖_2 = %.6f\n", k, norm(@view finest_r2.eig_func[:, k]))
    end
    println()
    println("Eigenfunctions are returned in monomial-Lagrange-basis coefficients;")
    println("see `laplace_eig_lagrange` for the basis convention.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_case()
end
