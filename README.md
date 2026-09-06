# vfem

**Verified finite element method in 2D and 3D.** Guaranteed, computer-assisted
eigenvalue bounds for the Dirichlet Laplacian and for Schrödinger operators
with singular (Coulomb) potentials, on triangular and tetrahedral meshes.

Every bound this library produces is an *enclosure*, not an estimate: run the
same routine with `T = Interval{Float64}` and the result is a rigorous interval
that provably contains the true value, with all rounding accounted for.

The Julia module is named `VFEM`; this repository is `vfem`.

> **Status: development.** The kernel is complete in both dimensions and
> cross-validated against the MATLAB reference implementations (7530
> assertions, all passing). The 3D application drivers (Hydrogen, H₂⁺) are not
> yet ported — see [Roadmap](#roadmap).

> Formerly published as `vfem3d`. The old URL redirects here.

---

## Scope: both dimensions

The library covers **triangular meshes in 2D** and **tetrahedral meshes in
3D**, in parallel directory trees. The 2D half is a direct port of the MATLAB
`VFEM2D` library — 17 source files carry a `# Port of vfem2d/….m` header — and
is complete. The 3D half is where current development is concentrated.

### 2D — triangular meshes

| Layer | What it covers |
| --- | --- |
| `src/core/mesh2d/` | `Mesh2D`: triangles, edges, tri↔edge connectivity, uniform and graded meshes |
| `src/core/assembly2d/` | CR, ECR, enriched CR, CECR, Lagrange P_k, Fujino–Morley; Dunavant quadrature |
| `src/core/eigensolve2d/` | Laplace and Schrödinger eigensolvers, Lehmann–Goerisch RT auxiliary problem, verified LG lower bounds, certified `lambda_h_bernstein` |
| `src/applications/cecr_pipeline/` | the m2–m7 CECR certification pipeline |

### 3D — tetrahedral meshes

| Layer | What it covers |
| --- | --- |
| `src/core/mesh/` | `Mesh3D`: tetrahedra, facets, edges, facet↔element connectivity |
| `src/core/assembly3d/` | CR, ECR, CECR, Lagrange P_k, DG, Raviart–Thomas on tets |
| `src/core/eigensolve3d/` | Liu lower bounds, Lehmann–Goerisch sharpening, verified CR Laplace, truncation correction |
| `src/core/quadrature_singular/` | Closed-form vertex-singular integrals ∫_K 1/r, 1/r² over tets and their faces |
| `src/core/potentials/` | Multi-centre Coulomb element averages, bounds, L^p integrals |

### Shared

Two supporting layers are dimension-neutral and are required by the 3D code:

- `src/core/bernstein/` — Bernstein multi-index machinery and Gram matrices on
  the reference simplex, used by the 3D quadrature.
- `src/core/singular_moments/` — closed-form singular moments over a
  *triangle*. The 3D vertex-singular tetrahedron integrals reduce to these
  over the tetrahedron's faces (`tri3d_invR_face_moments`).

### How the two halves relate

- The 2D path is the **cross-validation baseline**: 2D and 3D share the Liu
  constant machinery and the Lehmann–Goerisch transform, and the 2D results
  are checked against closed forms on the equilateral triangle.
- The CECR m2–m7 certification pipeline is **2D today** and is the template
  for its 3D counterpart.

**The two halves are independent.** No file under `assembly3d/`,
`eigensolve3d/`, `potentials/`, or `quadrature_singular/` references any
2D-only symbol, so either half can be used — or extracted — on its own.

---

## Installation

`vfem` depends on [`Veigs`](https://github.com/xfliu/veigs) for verified
generalized eigenvalue enclosures. It is not in the Julia General registry, so
install it explicitly first:

```julia
using Pkg
Pkg.add(url = "https://github.com/xfliu/veigs", subdir = "VEIGS.jl")
Pkg.add(url = "https://github.com/xfliu/vfem")
```

To work on the library itself:

```bash
git clone https://github.com/xfliu/vfem.git
cd vfem
julia --project=. -e 'using Pkg; Pkg.add(url="https://github.com/xfliu/veigs", subdir="VEIGS.jl"); Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

Requires Julia 1.10 or newer. `Manifest.toml` is not tracked: it pins an
absolute local path for the unregistered `Veigs` dependency and is therefore
not portable.

---

## Quick start

### Guaranteed bounds for the Dirichlet Laplacian

The *fundamental tetrahedron* `T_F` = conv{(0,0,0), (0,0,1), (½,½,½),
(−½,½,½)} has closed-form eigenvalues λ = (π²/4)·|k|², which makes it the
benchmark of choice (see [`docs/notes/special_tetrahedron.md`](docs/notes/special_tetrahedron.md)).

```julia
using VFEM
using IntervalArithmetic: inf, sup

m = special_tetrahedron_mesh()          # T_F, split into 4 tetrahedra
exact = (pi^2 / 4) * 80                 # = 197.39208802178717

r = verified_lg_lower_eig_bound_laplace_3d(m, 3, 1; RT_order = 3, rho = 250.0)

inf(r.eig_lower[1])   # 75.01579090325714   <= lambda_1, rigorously
sup(r.eig_upper[1])   # 228.12588145708128  >= lambda_1, rigorously
```

Both ends are certified. The gap is wide because the mesh has four elements —
refine with `special_tetrahedron_red_mesh(level)` or `red_refine_mesh_3d(m)`
and raise the polynomial degree to tighten it.

### A Coulomb (Hydrogen-like) Schrödinger problem

```julia
using VFEM

m = mesh_load_from_folder("test/fixtures/cube_r1")

info   = CoulombInfo([0.1 0.1 0.1], [1.0])   # one Z=1 nucleus
c_data = elem_V_coulomb_average(m, info)     # exact element averages of -Z/|x-c|
s      = schrodinger_eig_cecr_3d(m, c_data, 4; gamma_h_override = 0.3)

s.eig_h      # [25.518052, 33.837496, 34.046367, 34.046370]  discrete
s.eig_lower  # [19.219663, 23.628535, 23.730971, 23.730973]  guaranteed lower
s.Ch         # 0.111794  — the Liu interpolation constant, 0.1581 * h_max
             #             (see the caveat under "Method" below)
```

The Coulomb element averages are computed in closed form via the Duffy-type
vertex-singular quadrature, not by numerical integration, so the 1/r
singularity is handled exactly.

---

## Two modes, one code path

Every numeric routine is generic on `T<:Real`. The element type selects the
mode; there is no separate verified API to learn.

```julia
A_f = create_matrix_crouzeix_raviart_3d(m)                        # Float64: fast
A_i = create_matrix_crouzeix_raviart_3d(m; T = Interval{Float64}) # rigorous enclosure
```

Interval enclosures flow through assembly, boundary-condition restriction, and
the eigensolve, so a bound computed in interval mode is valid as a proof.

---

## What is implemented

### 3D element spaces (`src/core/assembly3d/`)

| Space | Function | Used for |
| --- | --- | --- |
| Crouzeix–Raviart | `create_matrix_crouzeix_raviart_3d` | Liu lower bounds |
| ECR (enriched CR) | `create_matrix_ecr_3d` | sharper nonconforming bounds |
| CECR | `create_matrix_cecr_3d` | Schrödinger lower bounds |
| Lagrange P_k | `create_matrix_lagrange_3d` | conforming (Galerkin) upper bounds |
| Discontinuous Galerkin | `create_matrix_dg_3d` | Lehmann–Goerisch right-hand sides |
| Raviart–Thomas | `create_matrix_rt_3d` | H(div) auxiliary problem |

### 3D eigenvalue drivers (`src/core/eigensolve3d/`)

| Function | Result |
| --- | --- |
| `schrodinger_eig_cecr_3d` | Liu lower + Galerkin upper bounds for −Δ + V |
| `verified_cr_laplace_3d` | verified enclosures of the discrete CR pencil |
| `cr_liu_lower_bounds_3d` / `verified_cr_liu_lower_3d` | Liu lower bounds, plain and verified |
| `lg_lower_eig_bound_laplace_3d` / `verified_..._3d` | Lehmann–Goerisch sharpened lower bounds |
| `one_piece_bubble_laplace_3d` | single-element bubble bounds on one tetrahedron |
| `compute_truncation_correction` | correction for truncating an unbounded domain |

### Singular Coulomb quadrature

`tri3d_invR_face_moments`, `tet_vertex_sing_bernstein_moments`,
`tet_vertex_sing_poly_integral`, `singular_potential_matrix_vertex_exact`,
`elem_V_coulomb_average`, `elem_V_coulomb_bounds`,
`elem_V_coulomb_Lp_integral_3d` — all closed-form, all interval-capable.

### 2D element spaces and drivers (`src/core/assembly2d/`, `eigensolve2d/`)

| Space / driver | Function |
| --- | --- |
| Crouzeix–Raviart | `create_matrix_crouzeix_raviart` |
| ECR / enriched CR | `create_matrix_ecr`, `create_matrix_enriched_crouzeix_raviart` |
| CECR | `create_matrix_cecr` |
| Lagrange P_k | `create_matrix_lagrange` |
| Fujino–Morley | `create_matrix_fujino_morley` |
| Laplace eigenvalues | `laplace_eig_lagrange`, `lg_lower_eig_bound_laplace` |
| Schrödinger (CECR) | `schrodinger_eig_cecr` |
| Verified LG lower bounds | `verified_lg_lower_eig` |
| H(div) auxiliary problem | `rt_hdiv_problem` |
| Interpolation constant | `lambda_h_bernstein` |

plus the m2–m7 CECR certification pipeline in `src/applications/cecr_pipeline/`.

---

## Repository layout

```
vfem/
├── src/
│   ├── VFEM.jl              module entry point; include order and exports
│   ├── core/                problem-agnostic FEM kernel  (see src/core/README.md)
│   │   ├── mesh/            Mesh3D + connectivity
│   │   ├── mesh2d/          Mesh2D + connectivity
│   │   ├── bernstein/       Bernstein kernel on the reference simplex
│   │   ├── singular_moments/    closed-form triangle moments
│   │   ├── quadrature_singular/ vertex-singular tet/face integrals
│   │   ├── assembly2d/  assembly3d/
│   │   ├── eigensolve2d/  eigensolve3d/
│   │   └── potentials/      Coulomb helpers
│   └── applications/        problem drivers  (see src/applications/README.md)
│       └── cecr_pipeline/   CECR certification pipeline (m2–m7)
├── test/
│   ├── runtests.jl          one test file per routine
│   ├── fixtures/            MATLAB-generated reference values + meshes
│   └── support/
├── examples/                runnable case studies
│   └── reference/           MATLAB exporters for cross-checking
└── docs/
    ├── decisions.md         design decision log, with reasons
    ├── testing-contract.md  what every test file must contain
    ├── notes/               mathematical notes
    └── reports/             generated HTML result reports
```

## Mesh format

`mesh_load_from_folder(dir)` reads two whitespace-separated text files:

- `nodes.dat` — `NumNode × 3` vertex coordinates.
- `elements.dat` — `NumElt × 4` one-based vertex indices per tetrahedron.

Facets, edges, and connectivity tables are derived on load. Element rows are
sorted ascending, matching the MATLAB convention, so downstream routines can
rely on the ordering. `test/fixtures/cube_r1/` is a worked example.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Every routine has its own test file, and every test file states its
corner-case taxonomy and mathematical contract up front — see
[`docs/testing-contract.md`](docs/testing-contract.md). Numerical results are
pinned against MATLAB `VFEM_LIB` fixtures under `test/fixtures/`, and against
closed forms where they exist (equilateral triangle in 2D, fundamental
tetrahedron in 3D).

## Roadmap

Deferred deliberately; reasoning recorded in
[`docs/decisions.md`](docs/decisions.md).

- **Application drivers.** `Applications.Hydrogen`, `Applications.H2Plus`,
  `Applications.ConstantV` are specified in `src/applications/README.md` but
  not yet written. `test/test_hydrogen_e2e.jl` is the worked example in the
  meantime.
- **3D certification pipeline.** The m2–m7 pipeline exists for 2D only.
- **`schrodinger_eig_cr_3d` / `schrodinger_eig_cg_3d`.** Skipped: CECR
  generalizes CR, and the Galerkin upper bound is already covered by
  `laplace_eig_lagrange_3d`.
- **Verified linear solves.** `verified_lg_lower_eig` currently relies on
  `Veigs` for the eigenvalue step; a general verified linear solver would let
  more of the Lehmann–Goerisch chain run in interval arithmetic.

## Related repositories

| Repository | Relation |
| --- | --- |
| [`xfliu/veigs`](https://github.com/xfliu/veigs) | verified eigenvalue solver — a dependency of this package |
| [`xfliu/VFEM2D`](https://github.com/xfliu/VFEM2D) | the MATLAB 2D code base that `src/core/*2d/` is ported from |
| `xfliu/VFEM_LIB` | the MATLAB 2D/3D reference implementation (private) |

## Method

The lower bounds follow the Liu framework: a nonconforming (CR / ECR / CECR)
discretization gives eigenvalues λ_h that, corrected by the element
interpolation constant C_h, yield a guaranteed lower bound

    lambda >= nu / (1 + C_h^2 * nu),     nu = lambda_h + gamma_h

(then shifted back by `gamma_h`, a shift that keeps the pencil positive for
sign-changing potentials such as Coulomb). On tetrahedra the interpolation
constant is `C_h = h_K/sqrt(40)`. The Lehmann–Goerisch method then sharpens
that bound using a Raviart–Thomas H(div) auxiliary problem and a separation
parameter ρ. Both stages run in interval arithmetic, so the final number is a
proof rather than an approximation.

> **Caveat on `C_h` in `schrodinger_eig_cecr_3d`.** That driver hard-codes
> `C_h = 0.1581 * h_max`, the truncation of `1/sqrt(40) = 0.158113883…` used by
> the MATLAB reference, so the cross-validation fixtures match bit-for-bit.
> Because `nu/(1 + C_h^2 nu)` is *decreasing* in `C_h`, rounding `C_h` down
> raises the reported bound — it overshoots the certified value by ~1e-3
> absolute on the shipped fixtures instead of leaving margin below it. Below
> discretization error, but not rigorous. Use `1/sqrt(40)` rounded up, or carry
> `C_h` as an interval, when the result has to be a proof. See
> [`docs/decisions.md`](docs/decisions.md).

## Author and license

Xuefeng Liu. Released under the MIT License — see [LICENSE](LICENSE).
