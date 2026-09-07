---
name: vfem
description: Operating VFEM.jl, a Julia library for verified (rigorous) finite element eigenvalue bounds on 2D triangular and 3D tetrahedral meshes. Covers environment setup for an unregistered package, the Float64/Interval dual mode, the two Lagrange bases, mesh formats, the assembly idiom the codebase requires, the testing contract, and the traps that cost real debugging time. Use whenever running, extending or reviewing VFEM.jl, or when a task needs guaranteed two-sided eigenvalue enclosures rather than approximations.
---

# VFEM.jl — operating guide

VFEM computes **enclosures**, not estimates. Every routine can return an interval
that provably contains the true value. That single fact drives most of the design
decisions below, and most of the ways an agent can silently get it wrong.

Repository: `github.com/xfliu/vfem` (MIT). Read `docs/decisions.md` for *why*,
`docs/testing-contract.md` before writing any test, `README.md` for the maths.

## 1. Environment — get this right first

`VFEM` is **unregistered**, and so is its dependency `Veigs`. `Manifest.toml` is
deliberately untracked because it pins an absolute local path.

```bash
julia --project=. -e 'using Pkg; Pkg.add(url="https://github.com/xfliu/veigs", subdir="VEIGS.jl"); Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'      # 7563 assertions, ~3 min
```

**`--project=.` is mandatory on every invocation.** Without it you get whatever
is in the default depot, which will not have `VFEM`.

Requires Julia 1.10+; developed on 1.12.6.

## 2. The dual mode is a type parameter, not a flag

```julia
A_f = create_matrix_crouzeix_raviart_3d(m)                        # Float64
A_i = create_matrix_crouzeix_raviart_3d(m; T = Interval{Float64}) # enclosure
```

Every numeric routine is generic on `T<:Real`. There is no separate verified API
and no global mode switch — the MATLAB ancestor needed a `mode_switch_interface/`
shim with a `global INTERVAL_MODE`; Julia does not.

When you add a routine, **keep it generic**. Hard-coding `Float64` anywhere in
the numeric path silently removes the ability to certify results downstream.

## 3. Traps that cost real time

**Two different Lagrange bases.** `lagrange_laplace_matrices` uses the *monomial
barycentric* basis `φ_α = L^α`; `create_matrix_lagrange` uses the *nodal* basis.
They span the same space, so **eigenvalues agree and coefficient vectors do
not**. `rt_hdiv_problem` consumes the monomial one — mixing them breaks the
Lehmann–Goerisch chain silently. Pick by which pipeline you are feeding.

**SLEPc is absent.** `firedrake.LinearEigensolver`-style calls will *import* but
cannot run: `slepc4py` is not installed. Extract the assembled PETSc/sparse
matrices and use `scipy.sparse.linalg.eigsh` / `Arpack` shift-invert instead.

**`C_h` in `schrodinger_eig_cecr_3d` is not yet rigorous.** It hard-codes
`C_h = 0.1581·h_max`, the truncation of `1/√40`, to match MATLAB fixtures
bit-for-bit. Because the bound decreases in `C_h`, rounding *down* overshoots by
~1e-3 — below discretisation error, but not a proof. Use `1/√40` rounded up, or
carry `C_h` as an interval, when the result must be a theorem.

**Interval width is not monotone in polynomial degree.** On the equilateral
triangle the enclosure is tightest at `p=5` (rel. width 1.0e-10) and gets
*wider* at `p=6` (3.7e-10) while costing twice the time. Approximation error
falls with `p`; accumulated interval width grows. Do not assume more degree is
more truth.

## 4. Assembly: the required idiom

**Never scatter-add into a `SparseMatrixCSC`.** `A[i,j] += v` inserts when the
entry is new — an O(nnz) memmove — making element-by-element assembly quadratic.
This was the shape of a real defect fixed on 2026-09-06 (up to 84× on the
affected routines).

```julia
Irow = Vector{Int}(undef, nent); Jcol = Vector{Int}(undef, nent)
Aval = Vector{T}(undef, nent);   pos = 0
# ... per element:
pos += 1; Irow[pos] = g[i]; Jcol[pos] = g[j]; Aval[pos] = A_local[i, j]
# ... after the loop:
A = dropzeros!(sparse(Irow, Jcol, Aval, ndof, ndof))
```

**`dropzeros!` is not optional.** `+= 0.0` on a non-stored entry is a no-op in
Julia, so the old code never stored exact zeros; every COO triplet does. Omit it
and Crouzeix–Raviart on right triangles gains 256 explicit zeros (`eᵢ·eⱼ`
vanishes for perpendicular edges) and the MATLAB `nnz` fixtures fail.

Use `push!` with `sizehint!` instead of a preallocated buffer only when the entry
count is not known ahead of the loop (boundary DOFs skipped, or branching per
element as in `_assemble_Dh_3d`).

*Caveat worth knowing:* not every scatter-add is quadratic. The cost driver is
**new entries per element**. P1 vertex matrices reuse their sparsity heavily, so
they were already linear; high-order Lagrange was not.

## 5. Meshes

| Loader | Reads |
|---|---|
| `mesh2d_load(dir)` | `vert.dat`, `tri.dat`, `edge.dat`, `bd.dat` |
| `mesh_load_from_folder(dir)` | `nodes.dat` (N×3), `elements.dat` (N×4, 1-based) |
| `mesh2d_triangle_uniform(n)` | returns `(Mesh2D, dims)` — a **tuple** |
| `red_refine_mesh_3d(m)` | uniform 3D refinement |

Fixtures: `test/fixtures/cube_r1` is the unit cube (λ₁ = 3π²);
`tutorial/meshdata/` holds 24 ready 2D meshes.

## 6. Function spaces

**2D:** Lagrange `P_k` (monomial and nodal), Crouzeix–Raviart, ECR, enriched CR,
CECR, Fujino–Morley, Raviart–Thomas `H(div)`.
**3D:** Lagrange `P_k`, CR, ECR, CECR, DG, Raviart–Thomas.
Plus closed-form vertex-singular Coulomb quadrature (`tri3d_invR_face_moments`,
`tet_vertex_sing_bernstein_moments`, `elem_V_coulomb_average`).

## 7. Tests: the contract is enforced

`docs/testing-contract.md` mandates four sections per test file, in order:
**(a)** a typical case checked against something *independent* — a closed form,
an analytic eigenvalue, or a MATLAB fixture; **(b)** a numbered corner-case
taxonomy in a header comment, one `@testset` each; **(c)** the mathematical
contract, stated then asserted; **(d)** a wall-clock ceiling, so an accidental
O(n²) rewrite fails rather than passing slowly.

**A high assertion count is not coverage.** Until 2026-09-07 the two CECR
pipeline routines had *no* tests; a change that made `_assemble_Dh_3d` throw on
every call still reported 7530/7530 passing. Before trusting a green suite for a
file you changed, check that something actually exercises it:

```bash
grep -rl '<routine_name>' test/
```

**Prefer identities that pin magnitude, not just shape.** Symmetry, linearity in
the charges and superposition are all invariant under a global rescaling — a
mutant that halved a quadrature weight passed every one of them. It was caught
only by comparing `sum(D)` against an independently computed integral. When
testing a quadrature, assert an absolute value from an independent source.

## 8. Verifying a refactor that should not change results

The suite alone is weak evidence for a file with thin coverage. Capture outputs
before and after in one session and diff them:

```julia
using Serialization
serialize(ENV["OUT"], Dict("case" => Matrix(f(args...))))
```

Run with the old file swapped in, then the new, then compare. Bit-identical is
achievable for pure reordering; expect ~1e-15 relative drift wherever a
reordered sum feeds a linear solve or eigensolve.
