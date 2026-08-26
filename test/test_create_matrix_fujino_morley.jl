# test/test_create_matrix_fujino_morley.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. DOF duality σ_a(φ_i) = δ_ai on one element, checked against
#      `fm_dof_values` — vertex values by direct evaluation, edge terms by
#      Gauss–Legendre quadrature, i.e. INDEPENDENTLY of `fm_sigma`.
#   2. Σ block structure: Σ[1:3, :] = [I 0] exactly, and B_loc = Σ⁻¹ to
#      round-off (the block inverse must agree with a dense inverse).
#   3. Local stiffness: symmetric, positive semidefinite, kernel dimension
#      EXACTLY 3 (= P1 on the element), and A_loc annihilates the FM DOF
#      vector of any P1 function.
#   4. P1 reproduction: the FM DOF vector of an affine function maps under
#      B_loc to that function's exact Bernstein coefficients.
#   5. Global counts M = (nv−3) + ne and N = 6·nt, including M = 8382 and
#      N = 24576 at n = 64 — the paper's production size.
#   6. Sparsity of B: at most 6 nonzeros per row (exactly 6 except in the
#      elements touching a deleted corner). This is the structural fact the
#      inverse-free solver relies on.
#   7. THE SIGN TEST. The edge-average of the jump of ∂v/∂n across interior
#      edges vanishes for :consistent, and equals exactly 2× the one-sided
#      average for :verbatim — the authors' convention applies tempT to the
#      local stiffness but not to the local FM→Bernstein block, so the two
#      elements sharing an edge disagree on the sign of the edge DOF.
#   8. Exactness on q = L₁·L₂, a global quadratic on K vanishing at all three
#      corners and hence a member of V^FM_h: reproduced to ~1e-15 by
#      :consistent, with O(1) relative error by :verbatim. Independent
#      confirmation of item 7 that does not reuse the jump machinery.
#   9. A is IDENTICAL between the two conventions (they differ only in B);
#      A is exactly symmetric with symmetrize=true (needed because CHOLMOD's
#      cholesky tests symmetry exactly) and merely symmetric to ~1e-16
#      without it; A is positive definite.
#  10. Interval mode: T = Interval{Float64} flows through and encloses the
#      Float64 result.
#  11. Invalid arguments throw (bad convention symbol, bad corners).
#  12. Performance: n = 16 assembly < 2 s.
#
# MATHEMATICAL CONTRACT (docs/testing-contract.md (c)):
#   σ_a(φ_i) = δ_ai                                   (Σ⁻¹ is the basis map)
#   ker(A_loc) = P1|_K,  dim = 3                      (H² seminorm)
#   M = (nv − 3) + ne,  N = 6·nt
#   nnz(row of B) ≤ 6
#   [∂v/∂n]_e = 0 on interior edges  ⟺  convention = :consistent

using Test
using LinearAlgebra: eigvals, Symmetric, norm, rank, opnorm, I
using SparseArrays: nnz, nzrange, SparseMatrixCSC
using IntervalArithmetic: Interval, interval, inf, sup
using VFEM: mesh2d_triangle_uniform, create_matrix_fujino_morley,
            fm_local_matrices, fm_sigma, fm_element_geometry, fm_inv3,
            fm_reference_elements, fm_edge_signs, fm_dof_values,
            fm_p2_eval, fm_p2_grad, fm_bary

# Consistent (reference-element) edge normals of element k, as used by A in
# both conventions and by B under :consistent.
function _fm_test_normals(m, ref, k)
    q = m.nodes[m.elements[k, :], :]
    sg = fm_edge_signs(m, ref, k)
    _, _, _, nout = fm_element_geometry(q)
    nrm = similar(nout)
    for j in 1:3
        nrm[j, 1] = sg[j] * nout[j, 1]
        nrm[j, 2] = sg[j] * nout[j, 2]
    end
    return q, nrm
end

# Exact degree-2 Bernstein coefficients of a function that is a polynomial of
# degree ≤ 2 on the element q: corners are vertex values, mixed coefficients
# come from inverting J_{3+m}(midpoint) = 1/2.
function _fm_bern_coeffs_exact(q, f)
    d = zeros(6)
    for a in 1:3
        d[a] = f(q[a, 1], q[a, 2])
    end
    for m in 1:3
        ia = mod(m, 3) + 1
        ib = mod(m + 1, 3) + 1
        mx = (q[ia, 1] + q[ib, 1]) / 2
        my = (q[ia, 2] + q[ib, 2]) / 2
        d[3+m] = 2 * f(mx, my) - (d[ia] + d[ib]) / 2
    end
    return d
end

@testset "create_matrix_fujino_morley" begin

    m4, corners4 = mesh2d_triangle_uniform(4; alpha = 1.0, theta = pi / 2)
    ref4 = fm_reference_elements(m4)

    @testset "1. DOF duality σ_a(φ_i) = δ_ai" begin
        for k in (1, 3, m4.nt)
            q, nrm = _fm_test_normals(m4, ref4, k)
            _, B_loc, _, _, _ = fm_local_matrices(q, nrm)
            for i in 1:6
                dv = fm_dof_values(q, nrm, B_loc[:, i])
                target = [j == i ? 1.0 : 0.0 for j in 1:6]
                @test maximum(abs, dv - target) < 1e-12
            end
        end
    end

    @testset "2. Σ block structure and Σ⁻¹" begin
        q, nrm = _fm_test_normals(m4, ref4, 2)
        A_loc, B_loc, Sig, A_bern, area = fm_local_matrices(q, nrm)
        @test Sig == fm_sigma(q, nrm)
        # first three rows are exactly [I 0] — the corner Bernstein coefficients
        # of a P2 function are its vertex values
        @test Sig[1:3, 1:3] == Matrix(1.0I, 3, 3)
        @test all(iszero, Sig[1:3, 4:6])
        @test maximum(abs, Sig * B_loc - Matrix(1.0I, 6, 6)) < 1e-12
        @test maximum(abs, B_loc - inv(Sig)) < 1e-12
        # fm_inv3 agrees with a dense inverse
        Db = Sig[4:6, 4:6]
        @test maximum(abs, fm_inv3(Db) - inv(Db)) < 1e-12
        @test area > 0
        @test maximum(abs, A_bern - A_bern') == 0.0     # exactly symmetric
    end

    @testset "3. local stiffness: PSD, kernel dimension exactly 3" begin
        for k in (1, 5, m4.nt)
            q, nrm = _fm_test_normals(m4, ref4, k)
            A_loc, _, _, _, _ = fm_local_matrices(q, nrm)
            # `fm_local_matrices` returns the RAW triple product
            # B_loc' A_bern B_loc, which is asymmetric at the rounding level even
            # though A_bern is exactly symmetric. The repair happens one level up,
            # in create_matrix_fujino_morley (see item 9).
            @test maximum(abs, A_loc - A_loc') < 1e-12 * maximum(abs, A_loc)
            ev = sort(eigvals(Symmetric(A_loc)))
            scale = maximum(abs, ev)
            @test all(>(-1e-12 * scale), ev)            # positive semidefinite
            @test count(<(1e-10 * scale), ev) == 3      # kernel dim exactly 3
            @test ev[4] > 1e-6 * scale                  # and no more than 3
        end
    end

    @testset "4. P1 reproduction" begin
        a, b, c = 0.3, -1.7, 0.55
        f(x, y) = a * x + b * y + c
        for k in (1, 4, m4.nt)
            q, nrm = _fm_test_normals(m4, ref4, k)
            A_loc, B_loc, _, _, _ = fm_local_matrices(q, nrm)
            # FM DOFs of f: vertex values, and grad f · n (constant along edges)
            x = zeros(6)
            for aa in 1:3
                x[aa] = f(q[aa, 1], q[aa, 2])
            end
            for j in 1:3
                x[3+j] = a * nrm[j, 1] + b * nrm[j, 2]
            end
            @test maximum(abs, B_loc * x - _fm_bern_coeffs_exact(q, f)) < 1e-12
            # P1 is in the kernel of the H² seminorm
            @test norm(A_loc * x) < 1e-10 * maximum(abs, A_loc)
        end
    end

    @testset "5. global counts M and N" begin
        for n in (2, 4, 8)
            m, corners = mesh2d_triangle_uniform(n)
            fm = create_matrix_fujino_morley(m, corners)
            @test fm.M == (m.nv - 3) + m.ne
            @test fm.N == 6 * m.nt
            @test size(fm.A) == (fm.M, fm.M)
            @test size(fm.B) == (fm.N, fm.M)
            @test count(==(0), fm.dofmap) == 3          # the three corners
        end
        # the paper's production size
        m64, corners64 = mesh2d_triangle_uniform(64)
        @test (m64.nv - 3) + m64.ne == 8382
        @test 6 * m64.nt == 24576
    end

    @testset "6. B has at most 6 nonzeros per row" begin
        m, corners = mesh2d_triangle_uniform(5)
        fm = create_matrix_fujino_morley(m, corners)
        Bt = SparseMatrixCSC(fm.B')                      # rows of B = cols of B'
        per_row = [length(nzrange(Bt, i)) for i in 1:fm.N]
        @test maximum(per_row) == 6
        @test all(≤(6), per_row)
        # only rows of elements touching a deleted corner can be shorter
        @test count(<(6), per_row) == 6 * 3              # three corner elements
    end

    @testset "7. interior-edge normal-derivative jump (THE SIGN TEST)" begin
        # An arbitrary nonzero FM coefficient vector; reconstruct per element and
        # compare the edge-average of ∂v/∂n from both sides, using the SAME
        # reference normal on both. For a function in V^FM_h this jump is zero.
        adj = [Tuple{Int,Int}[] for _ in 1:m4.ne]
        for k in 1:m4.nt, j in 1:3
            push!(adj[m4.tri2edge[k, j]], (k, j))
        end
        results = Dict{Symbol,Tuple{Float64,Float64,Float64}}()
        for conv in (:consistent, :verbatim)
            fm = create_matrix_fujino_morley(m4, corners4; convention = conv)
            x = collect(1.0:fm.M) ./ fm.M .+ 0.3
            dall = fm.B * x
            maxjump = 0.0
            maxone = 0.0
            ratios = Float64[]
            for e in 1:m4.ne
                length(adj[e]) == 2 || continue          # interior edges only
                kref = ref4[e]
                jref = findfirst(j -> m4.tri2edge[kref, j] == e, 1:3)
                qref = m4.nodes[m4.elements[kref, :], :]
                _, _, _, nout = fm_element_geometry(qref)
                nvec = nout[jref, :]
                vals = Float64[]
                for (k, j) in adj[e]
                    qk = m4.nodes[m4.elements[k, :], :]
                    nn = zeros(3, 2)
                    nn[j, :] = nvec                      # same normal both sides
                    push!(vals, fm_dof_values(qk, nn, dall[6*(k-1)+1 : 6*k])[3+j])
                end
                jump = abs(vals[1] - vals[2])
                one_sided = maximum(abs, vals)
                maxjump = max(maxjump, jump)
                maxone = max(maxone, one_sided)
                one_sided > 1e-10 && push!(ratios, jump / one_sided)
            end
            results[conv] = (maxjump, maxone, maximum(ratios))
        end
        # :consistent — the jump vanishes, so the reconstruction is in V^FM_h
        jc, onec, _ = results[:consistent]
        @test onec > 1e-3                                # the test is non-trivial
        @test jc < 1e-12 * onec
        # :verbatim — the two sides have opposite signs, so the jump is exactly
        # twice the one-sided value, on EVERY interior edge
        jv, onev, ratv = results[:verbatim]
        @test jv > onev                                  # not small
        @test ratv ≈ 2.0 atol = 1e-10
    end

    @testset "8. exactness on the global quadratic q = L₁·L₂" begin
        # q vanishes at all three corners of K and is a global quadratic, so it
        # lies in V^FM_h and must be reproduced EXACTLY by a correct assembly.
        for (n, al, th) in ((4, 1.0, pi / 2), (3, 1.0, 3pi / 4))
            m, corners = mesh2d_triangle_uniform(n; alpha = al, theta = th)
            ref = fm_reference_elements(m)
            P = [m.nodes[corners[i], :] for i in 1:3]
            Dt = (P[2][1]-P[1][1])*(P[3][2]-P[1][2]) -
                 (P[3][1]-P[1][1])*(P[2][2]-P[1][2])
            gL2 = [ (P[3][2]-P[1][2])/Dt, -(P[3][1]-P[1][1])/Dt ]
            gL3 = [-(P[2][2]-P[1][2])/Dt,  (P[2][1]-P[1][1])/Dt ]
            gL1 = -(gL2 + gL3)
            L2g(x, y) = ((x-P[1][1])*(P[3][2]-P[1][2]) -
                         (P[3][1]-P[1][1])*(y-P[1][2])) / Dt
            L3g(x, y) = ((P[2][1]-P[1][1])*(y-P[1][2]) -
                         (x-P[1][1])*(P[2][2]-P[1][2])) / Dt
            L1g(x, y) = 1 - L2g(x, y) - L3g(x, y)
            qfun(x, y) = L1g(x, y) * L2g(x, y)
            gradq(x, y) = L1g(x, y) .* gL2 .+ L2g(x, y) .* gL1

            errs = Dict{Symbol,Float64}()
            for conv in (:consistent, :verbatim)
                fm = create_matrix_fujino_morley(m, corners; convention = conv)
                x = zeros(fm.M)
                for v in 1:m.nv
                    dof = fm.dofmap[v]
                    dof == 0 && continue
                    x[dof] = qfun(m.nodes[v, 1], m.nodes[v, 2])
                end
                for e in 1:m.ne
                    dof = fm.dofmap[m.nv + e]
                    dof == 0 && continue
                    k = ref[e]
                    j = findfirst(jj -> m.tri2edge[k, jj] == e, 1:3)
                    qk = m.nodes[m.elements[k, :], :]
                    _, _, _, nout = fm_element_geometry(qk)
                    ia = mod(j, 3) + 1
                    ib = mod(j + 1, 3) + 1
                    mx = (qk[ia, 1] + qk[ib, 1]) / 2
                    my = (qk[ia, 2] + qk[ib, 2]) / 2
                    # grad q is affine, so the midpoint value IS the edge average
                    g = gradq(mx, my)
                    x[dof] = g[1] * nout[j, 1] + g[2] * nout[j, 2]
                end
                d = fm.B * x
                err = 0.0
                scale = 0.0
                for k in 1:m.nt
                    qk = m.nodes[m.elements[k, :], :]
                    dex = _fm_bern_coeffs_exact(qk, qfun)
                    err = max(err, maximum(abs, d[6*(k-1)+1 : 6*k] - dex))
                    scale = max(scale, maximum(abs, dex))
                end
                errs[conv] = err / scale
            end
            @test errs[:consistent] < 1e-13              # exact reproduction
            # Measured relative error for :verbatim on these two configurations:
            # 1.25 (n=4, theta=pi/2) and 2.05 (n=3, theta=3pi/4), i.e. O(1) —
            # the reconstruction is simply not the intended function. The
            # threshold is set well below those values, not at them.
            @test errs[:verbatim] > 0.2
        end
    end

    @testset "9. A is convention-independent, symmetric, positive definite" begin
        fc = create_matrix_fujino_morley(m4, corners4; convention = :consistent)
        fv = create_matrix_fujino_morley(m4, corners4; convention = :verbatim)
        @test fc.A == fv.A                               # bit-identical
        @test fc.B != fv.B
        @test fc.issym
        @test fc.asym == 0.0                             # EXACTLY symmetric
        ev = eigvals(Symmetric(Matrix(fc.A)))
        @test minimum(ev) > 0                            # positive definite
        # without the repair, A is asymmetric at the rounding level: harmless for
        # a dense LU, fatal for CHOLMOD's cholesky, which tests symmetry exactly
        fr = create_matrix_fujino_morley(m4, corners4; symmetrize = false)
        @test fr.asym > 0
        @test fr.asym < 1e-12 * maximum(abs, fr.A)
        @test !fr.issym
    end

    @testset "10. interval mode encloses the Float64 result" begin
        m2, corners2 = mesh2d_triangle_uniform(2)
        f64 = create_matrix_fujino_morley(m2, corners2)
        fiv = create_matrix_fujino_morley(m2, corners2; T = Interval{Float64})
        @test eltype(fiv.A) == Interval{Float64}
        @test (fiv.M, fiv.N) == (f64.M, f64.N)
        Ad = Matrix(f64.A); Ai = Matrix(fiv.A)
        @test all(inf.(Ai) .<= Ad .<= sup.(Ai))
        Bd = Matrix(f64.B); Bi = Matrix(fiv.B)
        @test all(inf.(Bi) .<= Bd .<= sup.(Bi))
    end

    @testset "11. invalid arguments throw" begin
        @test_throws ArgumentError create_matrix_fujino_morley(m4, corners4;
                                                              convention = :bogus)
        @test_throws ArgumentError create_matrix_fujino_morley(m4, (1, 1, 2))
        @test_throws ArgumentError create_matrix_fujino_morley(m4,
                                                              (1, 2, m4.nv + 1))
    end

    @testset "12. performance: n = 16 assembly < 2 s" begin
        m8, c8 = mesh2d_triangle_uniform(8)
        create_matrix_fujino_morley(m8, c8)              # warm up
        m16, c16 = mesh2d_triangle_uniform(16)
        t0 = time_ns()
        create_matrix_fujino_morley(m16, c16)
        @test (time_ns() - t0) < 2_000_000_000
    end
end
