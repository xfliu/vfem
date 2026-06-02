# Decision log — auto-loop continuation of VFEM.jl Phase 5+

## 2026-05-02 — Liu's 3D ECR constant

The 3D Liu constant has *two* values floating around the codebase:

* `VFEM3D/CLAUDE.md` (top-level) says `Cₕ = h_max / √10 ≈ 0.3162 · h_max`.
* `VFEM3D/lib/eigensolve/schrodinger_eig_cecr_3d.m` (the actual code)
  uses `Cₕ = 0.1581 · h_max = h_max / √40`, with a comment:
  "C_ECR(K) ≤ 0.1581·h_K for tetrahedra (= 1/sqrt(40), verified by
  exact monomial integration on reference tet)".

**Decision:** Use `1/√40` per the MATLAB code, not `1/√10` per CLAUDE.md.
Reason: the executable code is the authority; the doc is likely stale.
The 1/√40 number can be regenerated from the proof in
`VFEM3D/docs/notes/` if needed. Logged for traceability.

## 2026-05-02 — Schrödinger 3D driver: general V vs Coulomb-specific

MATLAB's `schrodinger_eig_cecr_3d.m` is hard-wired to call
`elem_V_coulomb_average(mesh, V_info)` for the per-element reaction
coefficient. That ties the driver to the (still-unported) Coulomb
helpers in `lib/eigensolve/potentials/`.

**Decision:** Port a *general* Julia driver
`schrodinger_eig_cecr_3d(m, V_func, neig)` that samples the
potential at element centroids — same shape as the 2D version
`schrodinger_eig_cecr.jl`. It works on any V (constant, polynomial,
function-of-position). For Coulomb specifically, callers will pass a
precomputed `c_data::Vector{Float64}` (override the centroid sampling)
once `elem_V_coulomb_average` lands. This decouples the driver from
the Coulomb-specific quadrature and gives us an end-to-end working 3D
pipeline immediately.

Centroid sampling on tetrahedra ≠ the exact element average for
non-affine V — but for smooth V it converges as h². For the Coulomb
singular case, the caller should bypass the centroid path entirely
and supply `c_data`.

## 2026-05-02 — Arpack shift-invert quirk in 3D

Same issue as the 2D path (documented in `schrodinger_eig_cecr.jl`):
`Arpack.eigs(A, B; sigma)` in Julia's binding does not back-transform
the shift-inverted eigenvalues. The 2D path works around this with
`which = :SM`. The MATLAB 3D code uses an explicit shift-invert via
LU. For Julia, `which = :SM` matches the 2D pattern and works for
problems where the smallest eigenvalues are positive.

**Decision:** Use `which = :SM` in the Julia 3D driver. For Coulomb
problems where some eigenvalues may be negative (Hydrogen has λ₁ ≈
−0.25), an explicit shift-invert path will need a separate code path;
deferred until Coulomb support is added.

## 2026-05-02 — Arpack vs dense for the 3D CECR eigensolve

The cube_r1 fixture has 4 lowest CECR eigenvalues `[26.57, 34.886, 34.886, 34.886]`
(a degenerate cluster of three). Arpack `which = :SM` returns only one
member of the cluster, then jumps to the next eigenvalue (43.88).

**Decision:** For 3D problems with `n_int ≤ 1000` use dense `eigen` —
robust on degenerate clusters, performance-fine at this scale. For
larger problems fall back to Arpack `which = :SM`. Threshold is a
conservative pick; can be tuned later.

## 2026-05-02 — Order of remaining Phase 5 work

After ECR / CECR / CECR-Schrödinger land, the plan lists:
* `schrodinger_eig_cr_3d` (CR variant — gives a coarser Liu lower bound)
* `schrodinger_eig_cg_3d` (CG / Lagrange — Galerkin upper bound)
* Coulomb helpers `elem_V_coulomb_*`
* Duffy-transform potential matrices `duffy_potential_matrix_*`

**Decision:** Skip the CR variant for now. Reasoning: CECR strictly
generalises CR (CR drops the cell-DOF enrichment), so anything CR
gives, CECR gives at least as well, and the test surface for the
shared Liu machinery is already validated on CECR. The CR-only
version is useful when the user wants a faster, looser bound — that's
a polish item, not a blocker.

**Decision:** Skip the CG variant for now. Reasoning: in the LG
pipeline (Phase 6), the upper-bound role is already played by the
Galerkin Lagrange eigensolver `laplace_eig_lagrange` (which we have in
2D and will need to port to 3D). The standalone `schrodinger_eig_cg`
is a thin wrapper that adds little — it's a Phase 6 dependency, not a
standalone goal.

**Decision:** Coulomb helpers next. They unblock the Hydrogen/H₂⁺
application, which is the main motivation for VFEM3D in the first
place. Within that group, prioritise `elem_V_coulomb_average` (the
input the CECR driver consumes), then `elem_V_coulomb_bernstein` if
needed for a finer reaction approximation.
