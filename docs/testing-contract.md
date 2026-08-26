# Test contract

Every routine in `src/` has its own test file in `test/`, named
`test_<routine>.jl` and included from `test/runtests.jl`. Each test file
must cover four things, in this order:

**(a) Typical case.** At least one call with ordinary, in-range inputs,
checked against a value that is known independently of this code —
a closed form, an analytic eigenvalue, or a MATLAB `VFEM_LIB` fixture
under `test/fixtures/`.

**(b) Corner-case taxonomy.** A comment block at the top of the file
enumerating the degenerate and boundary inputs the routine must
survive, numbered, with one `@testset` per entry. Typical entries:
empty input, single element, degenerate simplex, the singularity
sitting exactly on a vertex or face, `neig` exceeding the space
dimension, and — for the interval routines — the `Interval{Float64}`
path alongside the `Float64` one.

**(c) Mathematical contract.** A comment block stating what the routine
is *supposed* to compute, in mathematical terms, with the identities
the tests then assert: symmetry, positive semi-definiteness, exactness
degree of a quadrature rule, annihilation of constants by a stiffness
matrix, partition of unity, total mass equal to the domain measure.

**(d) An efficiency benchmark with a regression threshold.** At least
one `@test` with a wall-clock ceiling on the canonical fixture, so that
an accidental O(n²) rewrite fails the suite rather than passing slowly.

Two conventions that follow from (b):

- Routines that are generic on `T<:Real` are tested in both modes.
  `Float64` pins the value; `Interval{Float64}` pins that the enclosure
  actually contains it.
- MATLAB cross-checks compare invariants (trace, sum, Frobenius norm,
  max entry, nnz) rather than whole matrices. Where a comparison is
  sensitive to summation order — sparse `nnz` counts in particular —
  assert the order-independent quantity and say so at the assertion.

Older test headers cite this document as `Rule.md`, its name earlier in
the project's history.
