# test/test_mesh2d_triangle_uniform.jl
#
# CORNER-CASE TAXONOMY (docs/testing-contract.md (b)):
#   1. Counts nv/nt/ne/nb against the closed forms, for n = 1 (single element,
#      every edge on the boundary), n = 2, 3, 8, and the paper's n = 64.
#   2. tri2edge convention: local edge j is opposite vertex j, i.e. connects
#      (v[j+1], v[j+2]) cyclically — the same assertion as
#      test_mesh2d_unit_square.jl item 4, since the generator must emit a
#      Mesh2D indistinguishable from a loaded one.
#   3. Corners: node indices (1, n+1, nv), coordinates exactly p1, p2, p3.
#   4. Geometry: all elements counter-clockwise; total area = |K|; every
#      sub-triangle has area |K|/n² (the subdivision is congruent in the
#      parameter triangle, and the push-forward is affine).
#   5. Degenerate-parameter guards: n = 0, alpha ≤ 0, h ≤ 0, theta outside
#      (0, π) all throw.
#   6. Boundary edges: exactly 3n, each with multiplicity 1; interior edges
#      have exactly two adjacent elements.
#   7. Reference-element tie-break: every edge's lowest-numbered adjacent
#      element is an UPWARD triangle (index ≤ n(n+1)/2). This is what makes
#      the :verbatim convention of create_matrix_fujino_morley coincide with
#      the authors' choice of reference element.
#   8. Performance: n = 64 generation < 2 s.
#
# MATHEMATICAL CONTRACT (docs/testing-contract.md (c)):
#   nv = (n+1)(n+2)/2,  nt = n²,  ne = 3n(n+1)/2,  nb = 3n
#   Euler: nv − ne + nt = 1 for a triangulated disc.
#   Σ_k |K_k| = |K| = ½·alpha·h²·sin(theta),  and each |K_k| = |K|/n².

using Test
using VFEM: Mesh2D, mesh2d_triangle_uniform, mesh2d_triangle_uniform_check,
            find_tri2edge

@testset "mesh2d_triangle_uniform" begin

    @testset "1. counts match the closed forms" begin
        for n in (1, 2, 3, 8, 64)
            m, corners = mesh2d_triangle_uniform(n)
            @test m.nv == div((n + 1) * (n + 2), 2)
            @test m.nt == n^2
            @test m.ne == div(3 * n * (n + 1), 2)
            @test m.nb == 3 * n
            # Euler characteristic of a triangulated disc
            @test m.nv - m.ne + m.nt == 1
            @test length(m.bd_edge_ids) == m.nb
            @test size(m.bd_edges) == (m.nb, 2)
            @test length(corners) == 3
        end
        # n = 1: the single element, every edge on the boundary
        m1, _ = mesh2d_triangle_uniform(1)
        @test (m1.nv, m1.nt, m1.ne, m1.nb) == (3, 1, 3, 3)
        # the paper's production size
        m64, _ = mesh2d_triangle_uniform(64)
        @test m64.nv == 2145
        @test m64.nt == 4096
        @test m64.ne == 6240
    end

    @testset "2. tri2edge convention (edge j opposite vertex j)" begin
        m, _ = mesh2d_triangle_uniform(6; alpha = 1.3, theta = 2pi / 5, h = 0.7)
        for k in 1:m.nt
            v = (m.elements[k, 1], m.elements[k, 2], m.elements[k, 3])
            for j in 1:3
                eid = m.tri2edge[k, j]
                a, b = m.edges[eid, 1], m.edges[eid, 2]
                v1 = v[mod(j, 3) + 1]
                v2 = v[mod(j + 1, 3) + 1]
                pair = a < b ? (a, b) : (b, a)
                expected = v1 < v2 ? (v1, v2) : (v2, v1)
                @test pair == expected
            end
        end
        # and it agrees with the library's own builder
        @test m.tri2edge == find_tri2edge(m.elements, m.edges)
        # edges rows ascending, so a single global orientation per edge
        @test all(m.edges[e, 1] < m.edges[e, 2] for e in 1:m.ne)
    end

    @testset "3. corners are p1, p2, p3" begin
        n, al, th, hh = 5, 1.4, pi / 3, 0.9
        m, corners = mesh2d_triangle_uniform(n; alpha = al, theta = th, h = hh)
        @test corners == (1, n + 1, m.nv)
        @test m.nodes[corners[1], :] == [0.0, 0.0]
        @test m.nodes[corners[2], 1] ≈ hh atol = 0 rtol = 1e-15
        @test m.nodes[corners[2], 2] == 0.0
        @test m.nodes[corners[3], 1] ≈ al * hh * cos(th) atol = 1e-15
        @test m.nodes[corners[3], 2] ≈ al * hh * sin(th) atol = 1e-15
    end

    @testset "4. orientation, total area, congruence" begin
        for (n, al, th, hh) in ((4, 1.0, pi / 2, 1.0),
                                (5, 1.0, pi / 6, 1.0),
                                (3, 1.7, 3pi / 4, 1.2))
            m, corners = mesh2d_triangle_uniform(n; alpha = al, theta = th, h = hh)
            chk = mesh2d_triangle_uniform_check(m, corners, n;
                                                alpha = al, theta = th, h = hh)
            area_K = 0.5 * al * hh^2 * sin(th)
            @test chk.area ≈ area_K rtol = 1e-13
            @test chk.min_signed_area > 0          # all counter-clockwise
            # every sub-triangle has the same area |K|/n²
            for k in 1:m.nt
                x1, y1 = m.nodes[m.elements[k, 1], 1], m.nodes[m.elements[k, 1], 2]
                x2, y2 = m.nodes[m.elements[k, 2], 1], m.nodes[m.elements[k, 2], 2]
                x3, y3 = m.nodes[m.elements[k, 3], 1], m.nodes[m.elements[k, 3], 2]
                s = ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2
                @test s ≈ area_K / n^2 rtol = 1e-12
            end
        end
    end

    @testset "5. invalid parameters throw" begin
        @test_throws DomainError mesh2d_triangle_uniform(0)
        @test_throws DomainError mesh2d_triangle_uniform(-1)
        @test_throws DomainError mesh2d_triangle_uniform(4; alpha = 0.0)
        @test_throws DomainError mesh2d_triangle_uniform(4; alpha = -1.0)
        @test_throws DomainError mesh2d_triangle_uniform(4; h = 0.0)
        @test_throws DomainError mesh2d_triangle_uniform(4; theta = 0.0)
        @test_throws DomainError mesh2d_triangle_uniform(4; theta = pi)
        @test_throws DomainError mesh2d_triangle_uniform(4; theta = -0.5)
    end

    @testset "6. edge multiplicities" begin
        n = 7
        m, _ = mesh2d_triangle_uniform(n)
        cnt = zeros(Int, m.ne)
        for k in 1:m.nt, j in 1:3
            cnt[m.tri2edge[k, j]] += 1
        end
        @test all(c -> c == 1 || c == 2, cnt)
        @test count(==(1), cnt) == 3 * n
        @test sort(findall(==(1), cnt)) == sort(m.bd_edge_ids)
        @test count(==(2), cnt) == m.ne - 3 * n
    end

    @testset "7. every edge's reference element is an upward triangle" begin
        n = 8
        m, _ = mesh2d_triangle_uniform(n)
        n_up = div(n * (n + 1), 2)
        first_adj = zeros(Int, m.ne)
        for k in 1:m.nt, j in 1:3
            e = m.tri2edge[k, j]
            first_adj[e] == 0 && (first_adj[e] = k)
        end
        @test all(k -> 1 ≤ k ≤ n_up, first_adj)
    end

    @testset "8. performance: n = 64 < 2 s" begin
        mesh2d_triangle_uniform(8)                     # warm up
        t0 = time_ns()
        mesh2d_triangle_uniform(64)
        @test (time_ns() - t0) < 2_000_000_000
    end
end
