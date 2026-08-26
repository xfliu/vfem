Yes. The tetrahedron with good regularity in the paper is not the regular tetrahedron T_R, but the fundamental tetrahedron T_F.

The vertices are given in Section 5.2 as  ￼

P_0=(0,0,0),

P_1=(0,0,1),

P_2=\left(\frac12,\frac12,\frac12\right),

P_3=\left(-\frac12,\frac12,\frac12\right).

Thus

T_F=\operatorname{conv}
\left\{
(0,0,0),
(0,0,1),
\left(\frac12,\frac12,\frac12\right),
\left(-\frac12,\frac12,\frac12\right)
\right\}.

Geometrically:

* one edge is the vertical segment from (0,0,0) to (0,0,1);
* the opposite face lies in the plane y=\frac12;
* the tetrahedron is highly symmetric but not regular.

The reason this tetrahedron is special is not its shape regularity, but its connection with a lattice/Fourier structure.

In Appendix D the authors introduce homogeneous coordinates and construct explicit generalized sine functions

TS_{\mathbf k},

which satisfy

-\Delta TS_{\mathbf k}
=
\mu_{\mathbf k} TS_{\mathbf k},
\qquad
TS_{\mathbf k}|_{\partial T_F}=0,

with

\mu_{\mathbf k}
=
\frac{\pi^2}{4}
|\mathbf k|^2 .

Thus the eigenfunctions are known explicitly and are analytic.  ￼

This explains why they observe exponential convergence on T_F but only algebraic convergence on the regular tetrahedron T_R.  ￼

In fact, T_F is the fundamental domain of the tetrahedral reflection group (related to the face-centered cubic lattice). The exact eigenfunctions are obtained by antisymmetrizing Fourier modes, exactly as the equilateral triangle eigenfunctions are obtained from plane waves.

For your purposes, T_F is probably a much better benchmark than the regular tetrahedron if you want:

* exact eigenvalues,
* exact eigenfunctions,
* exponential convergence tests,
* verification of FEM eigenvalue bounds.

The regular tetrahedron is more suitable for testing robustness against singularities.  ￼

An interesting question is whether the regular tetrahedron actually has a singular exponent \lambda<\infty limiting the Sobolev regularity, or whether the observed algebraic convergence is partly due to the polynomial basis not respecting the tetrahedral symmetry. Dauge’s theory suggests the former, but it would be worthwhile to compute the first vertex singular exponent explicitly.
