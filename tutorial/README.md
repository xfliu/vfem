# VFEM.jl tutorial

Thirteen self-contained Julia scripts that build up from a first Poisson solve
to rigorous, computer-assisted eigenvalue enclosures for Schrödinger operators
with Coulomb potentials.

They are written to be **read top-to-bottom, then run** — each is a long
commented essay with executable code in it, not a bare demo. Chapter 1 spells
out the entire FEM pipeline by hand before ever calling a helper.

## Running

From the repository root:

```bash
julia --project=. tutorial/01_poisson_square.jl
```

The `--project` flag is required: `VFEM` is unregistered, so its dependencies
come from this repository's environment. Set up once with:

```bash
julia --project=. -e 'using Pkg; Pkg.add(url="https://github.com/xfliu/veigs", subdir="VEIGS.jl"); Pkg.instantiate()'
```

Figures and CSVs are written to the current directory by default. Override
with environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VFEM_TUTORIAL_OUT` | `pwd()` | where SVGs and CSVs are written |
| `VFEM_TUTORIAL_MESHDATA` | `tutorial/meshdata` | input mesh folders |
| `VFEM_TUTORIAL_SUPPORT` | `tutorial/TutorialSupport.jl` | the support module |

## Chapters

| # | File | What it covers |
| --- | --- | --- |
| 1 | `01_poisson_square.jl` | The Poisson problem on a convex domain. The whole pipeline by hand — assemble, restrict, solve — then homogeneous and non-homogeneous Dirichlet data, P1/P2, on a 5-level refinement ladder. |
| 2 | `02_poisson_lshape.jl` | The same operator on a **non-convex** domain. Separates the two things easy to confuse: approximation order and elliptic regularity. Uniform vs graded meshes. |
| 3 | `03_eig_square.jl` | Dirichlet Laplace eigenvalues on domains with known spectra, so every printed digit can be checked. |
| 4 | `04_eig_lshape.jl` | What a reentrant corner does to the eigenvalue convergence rate. |
| 5 | `05_veigs_basics.jl` | Verified eigenvalue bounds with `Veigs.jl` — the first rigorous numbers. |
| 6 | `06_veigs_internals.jl` | Opening the box: what `veigs` assembles internally. |
| 7 | `07_veigs_fem.jl` | `veigs` on a real FEM stiffness/mass pair. Reads the pre-exported pairs in `matrices/`. |
| 8 | `08_verified_bounds_square.jl` | Rigorous **two-sided** enclosure of λ₁ on the unit square, where 2π² = 19.7392088… is known exactly. |
| 9 | `09_verified_bounds_lshape.jl` | The same pipeline where there is **no closed form** — the situation a verified method actually exists for. |
| 10 | `10_element_spaces.jl` | Every 2D element family in VFEM.jl on one eigenproblem: CR, ECR, CECR, Lagrange, Fujino–Morley. |
| 12 | `12_schrodinger.jl` | Schrödinger eigenproblems: how a smooth potential enters a FEM matrix. |
| 13 | `13_coulomb.jl` | Coulomb potentials — why 1/r defeats quadrature, and what the closed-form singular moments do about it. |

> Chapters 0 and 11 do not exist. The numbering has always had these gaps.

## Supporting files

| Path | Role |
| --- | --- |
| `TutorialSupport.jl` | The tutorial's own scaffolding: load vectors, Dirichlet lifting, error norms, and hand-written SVG plotting. `VFEM` is an *eigenvalue* library and deliberately ships none of these — they belong to the application, not the library. |
| `tutorial_meshes.jl` | Mesh generation for the folders in `meshdata/`. |
| `meshdata/` | 24 pre-generated mesh folders (`vert.dat`, `tri.dat`, `edge.dat`, `bd.dat`), read by `VFEM.mesh2d_load`: unit squares, L-shapes uniform and graded, slits, equilateral triangles. |
| `matrices/` | Pre-exported stiffness/mass pairs in MatrixMarket format, consumed by Chapter 7. Chapter 7 detects their absence and skips that section. |

`TutorialSupport` has no plotting dependency — every figure is hand-written
SVG, by design.

## One thing worth knowing before Chapter 1

`lagrange_laplace_matrices(m, p)` does **not** use the nodal Lagrange basis. It
uses the monomial barycentric basis `φ_α = L₁ⁱ L₂ʲ L₃ᵏ`, which is what
`rt_hdiv_problem` consumes, so the Lehmann–Goerisch pipeline stays
self-consistent. VFEM.jl also ships a *nodal* Lagrange assembly,
`create_matrix_lagrange`, used by the CECR pipeline. The two are different
matrices for the same space: eigenvalues agree, coefficient vectors do not.
