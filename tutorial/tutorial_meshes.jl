# tutorial/meshes/tutorial_meshes.jl
#
# Mesh generators for the VFEM.jl tutorial.
# =========================================
#
# VFEM.jl deliberately ships no mesh generator: `mesh2d_load` reads a
# *folder* of four whitespace-separated text files and that is the whole
# interface.  This file is the tutorial's own generator.  It has NO
# dependency on VFEM — it only writes the four files — so you can read it,
# run it, and diff its output without loading the library at all.
#
#   include("tutorial_meshes.jl")
#   generate_unit_square(8;  path = "meshdata/unit_square_8")
#   generate_lshape(4;       path = "meshdata/lshape_4")
#   generate_lshape(4;       path = "meshdata/lshape_graded_4", graded = true, beta = 1.5)
#   generate_slit(4;         path = "meshdata/slit_4")
#   generate_equilateral(8;  path = "meshdata/equilateral_8")
#
# ---------------------------------------------------------------------------
# THE FILE FORMAT `mesh2d_load` EXPECTS
# ---------------------------------------------------------------------------
# A mesh folder contains exactly four files, all whitespace-separated ASCII,
# all read with `DelimitedFiles.readdlm`:
#
#   vert.dat   nv x 2  Float64   vertex coordinates (x, y)
#   tri.dat    nt x 3  Int       triangle vertex indices, 1-BASED
#   edge.dat   ne x 2  Int       the edge table; EACH ROW SORTED ASCENDING
#   bd.dat     nb x 2  Int       boundary edges; a SUBSET of the edge.dat rows
#
# Two properties of the loader drive the design of `write_mesh` below:
#
#  1. The 2D loader does NOT recompute edges from triangles (unlike the 3D
#     layer, and unlike `mesh2d_load_ne`).  It trusts `edge.dat`.  So the
#     edge table we write *is* the global edge numbering that every later
#     assembly routine (Crouzeix-Raviart, ECR, CECR, P2 Lagrange) will use
#     for its edge DOFs.  Getting it wrong is silent, not loud.
#
#  2. `bd.dat` is matched against `edge.dat` by *sorted endpoint pair*
#     (`find_is_edge_bd`), so the row order of `bd.dat` is irrelevant, but
#     each of its rows must actually occur in `edge.dat`.
#
# `find_tri2edge` then builds `tri2edge` with the convention that is easy to
# get backwards, so it is worth stating twice:
#
#     tri2edge[k, j] is the global index of the edge OPPOSITE local vertex j
#     of triangle k.   Local edge order is (v2,v3), (v1,v3), (v1,v2).
#
# ---------------------------------------------------------------------------
# ORIENTATION
# ---------------------------------------------------------------------------
# Every triangle written here is counter-clockwise (positive signed area).
# `write_mesh` enforces this by swapping the last two vertices of any
# clockwise triangle.  This is not cosmetic: assembly routines that use the
# signed area to form gradients will silently produce sign-flipped local
# stiffness contributions on a negatively oriented element, and the global
# matrix is then wrong without ever throwing.
#
# ---------------------------------------------------------------------------

using Printf

# ===========================================================================
# Small geometric primitives
# ===========================================================================

"""
    signed_area2(verts, a, b, c) -> Float64

Twice the signed area of triangle `(a, b, c)`.  Positive <=> counter-clockwise.
"""
function signed_area2(verts::AbstractMatrix{Float64}, a::Int, b::Int, c::Int)
    x1, y1 = verts[a, 1], verts[a, 2]
    x2, y2 = verts[b, 1], verts[b, 2]
    x3, y3 = verts[c, 1], verts[c, 2]
    return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
end

"""
    tri_min_angle_deg(verts, a, b, c) -> Float64

Smallest interior angle of triangle `(a, b, c)`, in degrees, from the three
side lengths via the law of cosines.
"""
function tri_min_angle_deg(verts::AbstractMatrix{Float64}, a::Int, b::Int, c::Int)
    p = (verts[a, 1], verts[a, 2])
    q = (verts[b, 1], verts[b, 2])
    r = (verts[c, 1], verts[c, 2])
    # Side lengths: la is opposite vertex a, etc.
    la = hypot(q[1] - r[1], q[2] - r[2])
    lb = hypot(r[1] - p[1], r[2] - p[2])
    lc = hypot(p[1] - q[1], p[2] - q[2])
    ang(o, u, v) = acos(clamp((u * u + v * v - o * o) / (2 * u * v), -1.0, 1.0))
    return rad2deg(min(ang(la, lb, lc), ang(lb, lc, la), ang(lc, la, lb)))
end

"""
    mesh_hmax(verts, edges) -> Float64

Longest edge length.  Mirrors `VFEM.find_mesh_hmax` so the generator can
self-check without importing VFEM.
"""
function mesh_hmax(verts::AbstractMatrix{Float64}, edges::AbstractMatrix{Int})
    h2 = 0.0
    for r in 1:size(edges, 1)
        i, j = edges[r, 1], edges[r, 2]
        d2 = (verts[i, 1] - verts[j, 1])^2 + (verts[i, 2] - verts[j, 2])^2
        d2 > h2 && (h2 = d2)
    end
    return sqrt(h2)
end

# ===========================================================================
# Topology: edges and boundary from triangles
# ===========================================================================

"""
    derive_edges(tris) -> (edges, bd_edges, edge_index, counts)

Build the global edge table from the triangle list.  This mirrors the logic
in `VFEM/src/applications/cecr_pipeline/mesh2d_load_ne.jl`: hash every
triangle side to its ascending-sorted endpoint pair, count how many
triangles own it, and call an edge a *boundary* edge iff exactly one
triangle owns it.  That rule needs no geometry and no knowledge of the
domain, which is why it is the robust way to do this.

The returned `edges` matrix is sorted lexicographically, so the global edge
numbering is a deterministic function of the triangle list alone -- two runs
of the generator produce byte-identical `edge.dat`.

Throws if any side is shared by more than two triangles (a non-manifold
mesh), which is the failure mode you get from a bad vertex deduplication.
"""
function derive_edges(tris::AbstractMatrix{Int})
    nt = size(tris, 1)
    counts = Dict{NTuple{2, Int}, Int}()
    sizehint!(counts, 3 * nt)
    for k in 1:nt
        v1, v2, v3 = tris[k, 1], tris[k, 2], tris[k, 3]
        for (a, b) in ((v2, v3), (v1, v3), (v1, v2))
            key = a < b ? (a, b) : (b, a)
            key[1] == key[2] &&
                error("degenerate side (repeated vertex $(key[1])) in triangle $k")
            counts[key] = get(counts, key, 0) + 1
        end
    end
    for (key, c) in counts
        c <= 2 || error("edge $key is shared by $c triangles (non-manifold mesh)")
    end

    keys_sorted = sort!(collect(keys(counts)))
    ne = length(keys_sorted)
    edges = Matrix{Int}(undef, ne, 2)
    edge_index = Dict{NTuple{2, Int}, Int}()
    sizehint!(edge_index, ne)
    for (r, key) in enumerate(keys_sorted)
        edges[r, 1] = key[1]
        edges[r, 2] = key[2]
        edge_index[key] = r
    end

    bd_rows = [r for r in 1:ne if counts[(edges[r, 1], edges[r, 2])] == 1]
    bd_edges = Matrix{Int}(undef, length(bd_rows), 2)
    for (i, r) in enumerate(bd_rows)
        bd_edges[i, 1] = edges[r, 1]
        bd_edges[i, 2] = edges[r, 2]
    end
    return edges, bd_edges, edge_index, counts
end

# ===========================================================================
# write_mesh -- the one function every generator below funnels through
# ===========================================================================

"""
    write_mesh(path, verts, tris; fix_orientation = true, allow_dup_coords = false)
        -> NamedTuple

Write the four `.dat` files of a VFEM 2D mesh folder and return a summary.

* `verts` -- `nv x 2` `Float64`
* `tris`  -- `nt x 3` `Int`, 1-based

Steps, in order:

1. **Orientation.** Any triangle with negative signed area has its last two
   vertices swapped, so `tri.dat` is uniformly counter-clockwise.  A triangle
   with zero area is a hard error.
2. **Edges.** Derived from `tris` (see `derive_edges`), sorted ascending
   within a row and lexicographically between rows.
3. **Boundary.** Edges owned by exactly one triangle.
4. **Sanity.** Unreferenced vertices and coincident vertex coordinates are
   detected and reported.  Coincident coordinates are an *error* unless
   `allow_dup_coords = true` -- the slit domain needs them on purpose (the two
   faces of the crack carry geometrically identical, topologically distinct
   vertices), everywhere else they mean a deduplication bug.
5. **Write.** Coordinates as `%.16e` (17 significant digits, so every Float64
   round-trips exactly through the text file -- which matters because these
   coordinates feed verified interval computations later); indices as `%d`.

Returned fields: `nv, nt, ne, nb, hmax, min_angle_deg, area, euler_ok,
orientation_ok, n_flipped, n_unreferenced, n_dup_coords`, plus the arrays
`verts, tris, edges, bd_edges` for callers that want to draw the mesh.
"""
function write_mesh(path::AbstractString,
                    verts::AbstractMatrix{<:Real},
                    tris::AbstractMatrix{<:Integer};
                    fix_orientation::Bool = true,
                    allow_dup_coords::Bool = false)
    V = Matrix{Float64}(verts)
    T = Matrix{Int}(tris)
    size(V, 2) == 2 || error("verts must be nv x 2")
    size(T, 2) == 3 || error("tris must be nt x 3")
    nv, nt = size(V, 1), size(T, 1)

    # --- 1. orientation ----------------------------------------------------
    n_flipped = 0
    for k in 1:nt
        a2 = signed_area2(V, T[k, 1], T[k, 2], T[k, 3])
        if a2 == 0.0
            error("triangle $k has zero area (vertices $(T[k, :]))")
        elseif a2 < 0
            if fix_orientation
                T[k, 2], T[k, 3] = T[k, 3], T[k, 2]
                n_flipped += 1
            else
                error("triangle $k is clockwise and fix_orientation = false")
            end
        end
    end
    orientation_ok = all(signed_area2(V, T[k, 1], T[k, 2], T[k, 3]) > 0 for k in 1:nt)

    # --- 2/3. edges and boundary ------------------------------------------
    edges, bd_edges, _, _ = derive_edges(T)
    ne, nb = size(edges, 1), size(bd_edges, 1)

    # --- 4. sanity --------------------------------------------------------
    referenced = falses(nv)
    for k in 1:nt, j in 1:3
        1 <= T[k, j] <= nv || error("triangle $k references vertex $(T[k, j]) not in 1:$nv")
        referenced[T[k, j]] = true
    end
    n_unreferenced = count(!, referenced)
    n_unreferenced == 0 ||
        error("$n_unreferenced vertices are not referenced by any triangle")

    seen = Dict{NTuple{2, Float64}, Int}()
    n_dup_coords = 0
    for i in 1:nv
        key = (V[i, 1], V[i, 2])
        if haskey(seen, key)
            n_dup_coords += 1
        else
            seen[key] = i
        end
    end
    if n_dup_coords > 0 && !allow_dup_coords
        error("$n_dup_coords coincident vertex coordinates (pass " *
              "allow_dup_coords = true only for slit/crack domains)")
    end

    # Euler's formula for a triangulated simply-connected polygon:
    #   V - E + F = 1   with F counted as triangles (the unbounded face is
    #                   excluded, hence 1 rather than the usual 2).
    euler_ok = (nv - ne + nt == 1)

    area = sum(signed_area2(V, T[k, 1], T[k, 2], T[k, 3]) for k in 1:nt) / 2
    minang = minimum(tri_min_angle_deg(V, T[k, 1], T[k, 2], T[k, 3]) for k in 1:nt)

    # --- 5. write ---------------------------------------------------------
    mkpath(path)
    open(joinpath(path, "vert.dat"), "w") do io
        for i in 1:nv
            @printf(io, "%.16e %.16e\n", V[i, 1], V[i, 2])
        end
    end
    open(joinpath(path, "tri.dat"), "w") do io
        for k in 1:nt
            @printf(io, "%d %d %d\n", T[k, 1], T[k, 2], T[k, 3])
        end
    end
    open(joinpath(path, "edge.dat"), "w") do io
        for r in 1:ne
            @printf(io, "%d %d\n", edges[r, 1], edges[r, 2])
        end
    end
    open(joinpath(path, "bd.dat"), "w") do io
        for r in 1:nb
            @printf(io, "%d %d\n", bd_edges[r, 1], bd_edges[r, 2])
        end
    end

    return (nv = nv, nt = nt, ne = ne, nb = nb,
            hmax = mesh_hmax(V, edges),
            min_angle_deg = minang,
            area = area,
            euler_ok = euler_ok,
            orientation_ok = orientation_ok,
            n_flipped = n_flipped,
            n_unreferenced = n_unreferenced,
            n_dup_coords = n_dup_coords,
            verts = V, tris = T, edges = edges, bd_edges = bd_edges)
end

# ===========================================================================
# Family 1 -- unit square (0,1)^2
# ===========================================================================

"""
    generate_unit_square(n; path) -> NamedTuple

Right-triangle ("diagonal" / Friedrichs-Keller) triangulation of (0,1)^2 with
`n` subdivisions per side: an `n x n` grid of squares, each cut by its
lower-left -> upper-right diagonal into two triangles.

Counts, closed form:  `nv = (n+1)^2`, `nt = 2n^2`,
`ne = 3n^2 + 2n` (n(n+1) horizontal + n(n+1) vertical + n^2 diagonal),
`nb = 4n`, `hmax = sqrt(2)/n`, min angle 45 degrees.

This is the same triangulation as the library fixture
`VFEM.jl/test/fixtures/unit_square_8x8` (nv = 81, nt = 128, ne = 208,
nb = 32), so `generate_unit_square(8)` reproduces its counts exactly -- a
useful first check that the generator agrees with the library's own data.
"""
function generate_unit_square(n::Integer; path::AbstractString)
    n >= 1 || error("n must be >= 1")
    gid(i, j) = j * (n + 1) + i + 1          # grid node (i, j), i,j in 0:n

    nv = (n + 1)^2
    verts = Matrix{Float64}(undef, nv, 2)
    for j in 0:n, i in 0:n
        verts[gid(i, j), 1] = i / n
        verts[gid(i, j), 2] = j / n
    end

    tris = Matrix{Int}(undef, 2 * n^2, 3)
    k = 0
    for j in 0:(n - 1), i in 0:(n - 1)
        v00 = gid(i, j); v10 = gid(i + 1, j)
        v11 = gid(i + 1, j + 1); v01 = gid(i, j + 1)
        k += 1; tris[k, :] = [v00, v10, v11]     # lower-right half, CCW
        k += 1; tris[k, :] = [v00, v11, v01]     # upper-left  half, CCW
    end
    return write_mesh(path, verts, tris)
end

# ===========================================================================
# Family 2/3 -- L-shape, uniform and corner-graded
# ===========================================================================

"""
    generate_lshape(n; path, graded = false, beta = 1.0) -> NamedTuple

The classic L-shaped domain

    Omega = (-1, 1)^2  minus  ( [0, 1] x [-1, 0] )

i.e. the **bottom-right** quadrant is removed.  Omega is the union of the
top-left, top-right and bottom-left unit squares, area 3, with its single
reentrant corner **at the origin** and interior angle omega = 3*pi/2 there.
The boundary polygon, counter-clockwise, is

    (-1,-1) -> (0,-1) -> (0,0) -> (1,0) -> (1,1) -> (-1,1) -> (-1,-1)

State this orientation whenever you quote a reference eigenvalue: the first
Dirichlet Laplace eigenvalue of *this* domain is the standard L-shaped
membrane value lambda_1 = 9.6397238440219...

`n` is the number of cells per unit length, so the background grid on
(-1,1)^2 is 2n x 2n of size h = 1/n and the three kept unit squares carry
3n^2 cells = 6n^2 triangles.  Uniform counts: `nt = 6n^2`,
`nv = (2n+1)^2 - n^2`, `ne = 9n^2 + 4n`, `nb = 8n` (the L has perimeter 8),
`hmax = sqrt(2)/n`, min angle 45 degrees.

**Grading (`graded = true`).**  The corner singularity of the L-shape is
u ~ r^alpha with alpha = pi/omega = 2/3, so a uniform mesh loses the optimal
P1 rate.  The fix used here keeps the *topology* of the uniform mesh and
moves the vertices along rays from the reentrant corner:

    p  ->  p * r^(beta-1),        r = max(|x|, |y|)   (the max-norm radius)

With beta = 1 this is the identity.  With beta > 1 and r < 1 the factor
r^(beta-1) is < 1, so vertices are pulled toward the origin and the local
mesh size near the corner shrinks like h^beta.  The max-norm is the right
radius here because its level sets are squares: the map fixes
max(|x|,|y|) = 1 pointwise and maps every ray from the origin into itself,
so the two reentrant edges (origin -> (1,0) and origin -> (0,-1)) and the
whole outer square are preserved *exactly* -- the graded mesh discretises the
same polygon, not an approximation of it.

The standard choice is beta = 1/alpha, i.e. **beta = 1.5** for the L-shape
(beta = 2 for the r^(1/2) slit).  Because only coordinates change, the graded
family has *identical* nv/nt/ne/nb to the uniform family at the same `n`,
which makes the uniform-vs-graded convergence comparison exactly DOF-matched.
"""
function generate_lshape(n::Integer; path::AbstractString,
                         graded::Bool = false, beta::Real = 1.0)
    n >= 1 || error("n must be >= 1")
    N = 2n                                   # cells per side of (-1,1)^2
    h = 1 / n
    # Grid nodes (i, j), i,j in 0:N, coordinates (-1 + i*h, -1 + j*h).
    # Cell (i, j) spans x in [-1+i h, -1+(i+1) h], y in [-1+j h, -1+(j+1) h].
    # The removed quadrant is x >= 0 and y <= 0  <=>  i >= n and j <= n-1.
    keep(i, j) = !(i >= n && j <= n - 1)

    cells = [(i, j) for j in 0:(N - 1) for i in 0:(N - 1) if keep(i, j)]

    # Renumber only the grid nodes actually used, in row-major grid order so
    # the numbering is reproducible.
    used = Set{NTuple{2, Int}}()
    for (i, j) in cells
        push!(used, (i, j)); push!(used, (i + 1, j))
        push!(used, (i + 1, j + 1)); push!(used, (i, j + 1))
    end
    order = sort!(collect(used); by = p -> (p[2], p[1]))
    id = Dict(p => k for (k, p) in enumerate(order))

    nv = length(order)
    verts = Matrix{Float64}(undef, nv, 2)
    for (k, (i, j)) in enumerate(order)
        verts[k, 1] = -1 + i * h
        verts[k, 2] = -1 + j * h
    end

    if graded
        beta > 0 || error("beta must be > 0")
        for k in 1:nv
            x, y = verts[k, 1], verts[k, 2]
            r = max(abs(x), abs(y))
            if r > 0
                s = r^(beta - 1)
                verts[k, 1] = x * s
                verts[k, 2] = y * s
            end
        end
    end

    tris = Matrix{Int}(undef, 2 * length(cells), 3)
    k = 0
    for (i, j) in cells
        v00 = id[(i, j)]; v10 = id[(i + 1, j)]
        v11 = id[(i + 1, j + 1)]; v01 = id[(i, j + 1)]
        k += 1; tris[k, :] = [v00, v10, v11]
        k += 1; tris[k, :] = [v00, v11, v01]
    end
    return write_mesh(path, verts, tris)
end

# ===========================================================================
# Family 4 -- slit (cracked) domain
# ===========================================================================

"""
    generate_slit(n; path) -> NamedTuple

The slit (cracked) square

    Omega = (-1, 1)^2  minus  ( [0, 1] x {0} )

-- the full square with a cut along the positive x-axis from the crack tip at
the **origin** out to (1, 0).  The interior angle at the tip is 2*pi, the
largest possible, so the leading corner singularity is the hard one,
u ~ r^(1/2) (alpha = pi/omega = 1/2).

The crack is realised the only way it can be on a triangulation: the grid
nodes on the slit with x > 0 are **duplicated**, one copy owned by the
triangles above the cut and one by the triangles below, while the tip node at
the origin stays single.  Consequently `vert.dat` contains n pairs of
*coincident* coordinates.  That is intentional and is why `write_mesh` is
called here with `allow_dup_coords = true`.  Nothing is topologically
degenerate: the two copies are distinct vertices, the domain is still simply
connected (a slit disk), and Euler's V - E + F = 1 still holds.

`n` = cells per unit length, so 2n x 2n background cells, `nt = 8n^2`,
`nv = (2n+1)^2 + n`, `nb = 8n + 2n = 10n` (8n on the outer square plus n on
each of the two crack faces), `hmax = sqrt(2)/n`, min angle 45 degrees.
"""
function generate_slit(n::Integer; path::AbstractString)
    n >= 1 || error("n must be >= 1")
    N = 2n
    h = 1 / n
    base(i, j) = j * (N + 1) + i + 1          # base id of grid node (i, j)

    nv_base = (N + 1)^2
    # One extra vertex per slit node with x > 0 (i > n at j = n).  The base id
    # serves as the LOWER copy; the extra ids are the UPPER copies.
    upper = Dict{Int, Int}()
    nv = nv_base
    for i in (n + 1):N
        nv += 1
        upper[i] = nv
    end

    verts = Matrix{Float64}(undef, nv, 2)
    for j in 0:N, i in 0:N
        verts[base(i, j), 1] = -1 + i * h
        verts[base(i, j), 2] = -1 + j * h
    end
    for i in (n + 1):N
        verts[upper[i], 1] = -1 + i * h       # same coordinates as the
        verts[upper[i], 2] = 0.0              # lower copy -- on purpose
    end

    # Resolve a grid node for a cell that lies above (`above = true`) or below
    # the cut.  Only nodes strictly beyond the tip are split.
    function nid(i, j, above::Bool)
        if j == n && i > n
            return above ? upper[i] : base(i, j)
        end
        return base(i, j)
    end

    tris = Matrix{Int}(undef, 2 * N^2, 3)
    k = 0
    for j in 0:(N - 1), i in 0:(N - 1)
        above = (j >= n)                      # cell sits above the slit line
        v00 = nid(i, j, above);         v10 = nid(i + 1, j, above)
        v11 = nid(i + 1, j + 1, above); v01 = nid(i, j + 1, above)
        k += 1; tris[k, :] = [v00, v10, v11]
        k += 1; tris[k, :] = [v00, v11, v01]
    end
    return write_mesh(path, verts, tris; allow_dup_coords = true)
end

# ===========================================================================
# Family 5 -- equilateral triangle (closed-form spectrum)
# ===========================================================================

"""
    generate_equilateral(n; path, side = 1.0) -> NamedTuple

Uniform triangulation of the equilateral triangle with vertices
(0,0), (side,0), (side/2, side*sqrt(3)/2), obtained by cutting each side into
`n` equal parts: `n^2` congruent equilateral sub-triangles (n(n+1)/2 pointing
up, n(n-1)/2 pointing down), `nv = (n+1)(n+2)/2`, `ne = 3n(n+1)/2`,
`nb = 3n`, `hmax = side/n`, and **every** angle exactly 60 degrees.

This domain earns its place in the tutorial because its Dirichlet Laplace
spectrum is known in closed form (Lame; see McCartin, *SIAM Review* 45 (2003)
267-287): for side `a`,

    lambda_{m,n} = (16 pi^2 / (9 a^2)) * (m^2 + m n + n^2),   m >= n >= 1,

with multiplicity 1 when m = n and 2 when m > n.  For a = 1 the first few are
lambda ~ 52.638 (1,1), 122.822 (2,1, double), 210.553 (2,2), 228.099 (3,1,
double) -- exact targets against which a verified eigenvalue enclosure can be
checked rather than merely compared.
"""
function generate_equilateral(n::Integer; path::AbstractString, side::Real = 1.0)
    n >= 1 || error("n must be >= 1")
    A  = (0.0, 0.0)
    e1 = (side / n, 0.0)                                  # step toward (side,0)
    e2 = (side / (2n), side * sqrt(3) / (2n))             # step toward the apex

    # Lattice points (i, j) with i + j <= n, numbered row by row (j ascending).
    order = [(i, j) for j in 0:n for i in 0:(n - j)]
    id = Dict(p => k for (k, p) in enumerate(order))

    verts = Matrix{Float64}(undef, length(order), 2)
    for (k, (i, j)) in enumerate(order)
        verts[k, 1] = A[1] + i * e1[1] + j * e2[1]
        verts[k, 2] = A[2] + i * e1[2] + j * e2[2]
    end

    tris = Matrix{Int}(undef, n^2, 3)
    k = 0
    for j in 0:(n - 1), i in 0:(n - 1 - j)
        k += 1; tris[k, :] = [id[(i, j)], id[(i + 1, j)], id[(i, j + 1)]]       # up
        if i + j <= n - 2                                                    # down
            k += 1; tris[k, :] = [id[(i + 1, j)], id[(i + 1, j + 1)], id[(i, j + 1)]]
        end
    end
    return write_mesh(path, verts, tris)
end

# ===========================================================================
# The two-triangle mesh printed in full in the tutorial
# ===========================================================================

"""
    generate_example_2tri(path) -> NamedTuple

The smallest interesting mesh: the unit square as two triangles.  Small
enough that all four `.dat` files fit on a tutorial page, and it is exactly
`generate_unit_square(1)`.
"""
generate_example_2tri(path::AbstractString) = generate_unit_square(1; path = path)
