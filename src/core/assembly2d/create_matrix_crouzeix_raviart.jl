# src/core/assembly2d/create_matrix_crouzeix_raviart.jl
#
# Port of vfem2d/lib/fem_assembly/create_matrix_crouzeix_raviart.m.
#
# Crouzeix–Raviart P1 mass and stiffness on a triangulation. CR
# DOFs live on edges; one DOF per edge ⇒ `ne × ne` global matrices.
#
# Local matrices on triangle K with edges e₁, e₂, e₃ (opposite
# vertices 1, 2, 3 respectively) are:
#   A0_loc = (|K| / 3) · I₃             (mass)
#   A1_loc = (e_i · e_j) / |K|          (stiffness; edges as 2D vectors)
# where edge i is the directed vector v_{i+2} − v_{i+1} (cyclic), i.e.
# the side opposite vertex i. The MATLAB code uses `node(t([3,1,2]),:) −
# node(t([2,3,1]),:)` which produces the same set of vector edges.

using SparseArrays: spzeros, sparse, SparseMatrixCSC

"""
    create_matrix_crouzeix_raviart(m::Mesh2D; T::Type = Float64)
        -> (A0::SparseMatrixCSC{T}, A1::SparseMatrixCSC{T})

Assemble the global CR mass and stiffness matrices for the 2D
mesh `m`. DOFs are edge-midpoint averages; ordering is the row order
of `m.edges`.

`T` controls the element type (`Float64` for approximation,
`Interval{Float64}` for verified — the routine is generic).
"""
function create_matrix_crouzeix_raviart(m::Mesh2D; T::Type = Float64)
    ne = m.ne
    A0 = spzeros(T, ne, ne)
    A1 = spzeros(T, ne, ne)

    @inbounds for k in 1:m.nt
        t = (m.elements[k, 1], m.elements[k, 2], m.elements[k, 3])
        # Three edge vectors: opposite vertices 1, 2, 3 — same as MATLAB
        # `node(t([3,1,2]),:) − node(t([2,3,1]),:)`.
        e = (
            (T(m.nodes[t[3], 1]) - T(m.nodes[t[2], 1]),
             T(m.nodes[t[3], 2]) - T(m.nodes[t[2], 2])),
            (T(m.nodes[t[1], 1]) - T(m.nodes[t[3], 1]),
             T(m.nodes[t[1], 2]) - T(m.nodes[t[3], 2])),
            (T(m.nodes[t[2], 1]) - T(m.nodes[t[1], 1]),
             T(m.nodes[t[2], 2]) - T(m.nodes[t[1], 2])),
        )
        # |K| = ½ |e₁ × e₂| (using e₁ rotated by 90° dot e₂).
        S = abs(T(0.5) * (e[1][1] * e[2][2] - e[1][2] * e[2][1]))

        eidx = (m.tri2edge[k, 1], m.tri2edge[k, 2], m.tri2edge[k, 3])
        # Mass: diagonal S/3 on local edge-DOFs.
        for i in 1:3
            A0[eidx[i], eidx[i]] += S / T(3)
        end
        # Stiffness: e_i · e_j / S.
        for i in 1:3, j in 1:3
            A1[eidx[i], eidx[j]] += (e[i][1] * e[j][1] + e[i][2] * e[j][2]) / S
        end
    end
    return A0, A1
end
