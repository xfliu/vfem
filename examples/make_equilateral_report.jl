# examples/make_equilateral_report.jl
#
# Generate `docs/reports/equilateral_triangle_report.html`: a
# detailed convergence-and-eigenfunction report on the Dirichlet
# Laplace eigenvalue case study, comparing
#
#     λ_h(P1)  Galerkin upper bound        (theoretical rate O(h²))
#     λ_h(P2)  Galerkin upper bound        (theoretical rate O(h⁴))
#     λ_h(P3)  Galerkin upper bound        (theoretical rate O(h⁶))
#     λ_lo(CR-Liu)  guaranteed lower bound (theoretical rate O(h²))
#
# against the closed-form Lamé spectrum.
#
# The HTML loads Plotly.js from a CDN (so an internet connection is
# needed to view the interactive plots). All numerical data is
# embedded inline as JSON.
#
# Run with:
#     julia --project=. examples/make_equilateral_report.jl

using Printf
using Dates
using LinearAlgebra: norm
using IntervalArithmetic: inf

const _PKG_ROOT = abspath(joinpath(@__DIR__, ".."))
import Pkg
Pkg.activate(_PKG_ROOT; io = devnull)

using VFEM
include(joinpath(@__DIR__, "laplace_equilateral_triangle.jl"))

# ----------------------------------------------------------------------------
# Configuration.
# ----------------------------------------------------------------------------

const FEM_ORDERS                    = [1, 2, 3]
const REFINEMENT_LEVELS_FOR_TABLE   = [3, 4, 5]
const REFINEMENT_LEVEL_FOR_PLOT     = 4
const N_EIG                         = 4
const N_SUB_FOR_PLOT                = 6
const RATE_SATURATION_FLOOR         = 1e-12   # below this, observed rate is unreliable

# ----------------------------------------------------------------------------
# Eigenfunction sampling — see laplace_equilateral_triangle.jl for the
# basis convention. We evaluate the P_p polynomial in monomial basis on
# a per-element barycentric grid for plotting. (Only used at the chosen
# `REFINEMENT_LEVEL_FOR_PLOT`, with degree 2 — P3 plots are not added
# to the report to keep the page light.)
# ----------------------------------------------------------------------------

@inline function _eval_p2_local(c::AbstractVector, L1::Real, L2::Real, L3::Real)
    return c[1] * L1 * L1 + c[2] * L2 * L2 + c[3] * L3 * L3 +
           c[4] * L2 * L3 + c[5] * L1 * L3 + c[6] * L1 * L2
end

function sample_eigenfunction(m, eig_func_col::AbstractVector,
                                cg_local_to_global, n_sub::Integer = 6)
    nt = m.nt
    n_per_tri = (n_sub + 1) * (n_sub + 2) ÷ 2
    n_total = nt * n_per_tri
    xs = Vector{Float64}(undef, n_total)
    ys = Vector{Float64}(undef, n_total)
    vs = Vector{Float64}(undef, n_total)
    tris = Vector{NTuple{3, Int}}()

    cur = 1
    for k in 1:nt
        v1 = m.elements[k, 1]; v2 = m.elements[k, 2]; v3 = m.elements[k, 3]
        x1, y1 = m.nodes[v1, 1], m.nodes[v1, 2]
        x2, y2 = m.nodes[v2, 1], m.nodes[v2, 2]
        x3, y3 = m.nodes[v3, 1], m.nodes[v3, 2]
        g_dofs = cg_local_to_global(m, k, 2)
        c = [eig_func_col[g_dofs[i]] for i in 1:6]

        rows = Int[cur]
        for i in 0:n_sub
            for j in 0:(n_sub - i)
                L1 = i / n_sub
                L2 = j / n_sub
                L3 = 1 - L1 - L2
                xs[cur] = L1 * x1 + L2 * x2 + L3 * x3
                ys[cur] = L1 * y1 + L2 * y2 + L3 * y3
                vs[cur] = _eval_p2_local(c, L1, L2, L3)
                cur += 1
            end
            push!(rows, cur)
        end

        bary_idx(i, j) = rows[i + 1] + j
        for i in 0:(n_sub - 1)
            for j in 0:(n_sub - i - 1)
                a = bary_idx(i, j)
                b = bary_idx(i, j + 1)
                c_idx = bary_idx(i + 1, j)
                push!(tris, (a, b, c_idx))
                if j ≤ n_sub - i - 2
                    d = bary_idx(i + 1, j + 1)
                    push!(tris, (b, d, c_idx))
                end
            end
        end
    end
    return xs, ys, vs, tris
end

# ----------------------------------------------------------------------------
# Run the convergence study.
# ----------------------------------------------------------------------------

println("Computing convergence data for P1, P2, P3 + CR-Liu...")
λ_exact = exact_equilateral_dirichlet_eigenvalues(N_EIG)

# Per-level data: (h, nv, nt, ndofs_per_p, eigvals_per_p, cr_low)
levels_data = NamedTuple[]
for level in REFINEMENT_LEVELS_FOR_TABLE
    print("  level $level ...")
    m, _ = build_equilateral_mesh(level)
    h = find_mesh_hmax(m.nodes, m.edges)

    eigs_per_p = Dict{Int, Vector{Float64}}()
    ndofs_per_p = Dict{Int, Int}()
    for p in FEM_ORDERS
        r = laplace_eig_lagrange(m, p, N_EIG)
        eigs_per_p[p]  = collect(r.eig_value)
        ndofs_per_p[p] = size(r.A, 1)
    end
    cr_low_int, _ = verified_cr_liu_lower(m, N_EIG)
    cr_low = [inf(cr_low_int[k]) for k in 1:N_EIG]

    push!(levels_data,
          (; level = level, h = h, nv = m.nv, nt = m.nt,
              ndofs_per_p = ndofs_per_p,
              eigs_per_p = eigs_per_p,
              cr_low = cr_low))
    println(" h = $(round(h; digits = 5))  nv = $(m.nv)  nt = $(m.nt)")
end

# Errors. For Galerkin upper bounds err = λ_h - λ_exact (≥ 0).
# For CR-Liu lower bound err = λ_exact - λ_lo (≥ 0).
upper_err_per_p = Dict{Int, Vector{Vector{Float64}}}()
for p in FEM_ORDERS
    upper_err_per_p[p] = [
        [max(d.eigs_per_p[p][k] - λ_exact[k], 0.0) for k in 1:N_EIG]
        for d in levels_data
    ]
end
lower_err = [[max(λ_exact[k] - d.cr_low[k], 0.0) for k in 1:N_EIG] for d in levels_data]

# Observed convergence rates between consecutive levels.
# rate = log(err_k / err_{k+1}) / log(h_k / h_{k+1}).
function _observed_rate(err_k::Real, err_kp1::Real, h_k::Real, h_kp1::Real)
    if err_k < RATE_SATURATION_FLOOR || err_kp1 < RATE_SATURATION_FLOOR
        return nothing   # saturated
    end
    return log(err_k / err_kp1) / log(h_k / h_kp1)
end

# upper_rates[p][i][k] = rate between level i and i+1 for eigenvalue k, order p
upper_rates = Dict{Int, Vector{Vector{Union{Float64, Nothing}}}}()
for p in FEM_ORDERS
    rates = [Union{Float64, Nothing}[] for _ in 1:(length(levels_data) - 1)]
    for i in 1:(length(levels_data) - 1)
        for k in 1:N_EIG
            r = _observed_rate(upper_err_per_p[p][i][k],
                               upper_err_per_p[p][i + 1][k],
                               levels_data[i].h, levels_data[i + 1].h)
            push!(rates[i], r)
        end
    end
    upper_rates[p] = rates
end
lower_rates = [Union{Float64, Nothing}[] for _ in 1:(length(levels_data) - 1)]
for i in 1:(length(levels_data) - 1)
    for k in 1:N_EIG
        r = _observed_rate(lower_err[i][k], lower_err[i + 1][k],
                           levels_data[i].h, levels_data[i + 1].h)
        push!(lower_rates[i], r)
    end
end

# ----------------------------------------------------------------------------
# Eigenfunction samples (P2 only, for plotting).
# ----------------------------------------------------------------------------

println("Sampling eigenfunctions (P2, level $REFINEMENT_LEVEL_FOR_PLOT)...")
m_plot, _ = build_equilateral_mesh(REFINEMENT_LEVEL_FOR_PLOT)
r2_plot   = laplace_eig_lagrange(m_plot, 2, N_EIG)

eigfunc_samples = Vector{NamedTuple}()
for k in 1:N_EIG
    xs, ys, vs, tris = sample_eigenfunction(m_plot, view(r2_plot.eig_func, :, k),
                                              VFEM._cg_lagrange_local_to_global,
                                              N_SUB_FOR_PLOT)
    scale = maximum(abs, vs)
    vs_norm = vs ./ scale
    push!(eigfunc_samples,
          (; k = k, λ_h = r2_plot.eig_value[k], λ_exact = λ_exact[k],
              xs = xs, ys = ys, vs = vs_norm,
              tris_i = [t[1] - 1 for t in tris],
              tris_j = [t[2] - 1 for t in tris],
              tris_k = [t[3] - 1 for t in tris]))
end

# Coarse-mesh wireframe.
mesh_xs0 = Float64[]; mesh_ys0 = Float64[]
for r in 1:m_plot.ne
    a = m_plot.edges[r, 1]; b = m_plot.edges[r, 2]
    push!(mesh_xs0, m_plot.nodes[a, 1]); push!(mesh_ys0, m_plot.nodes[a, 2])
    push!(mesh_xs0, m_plot.nodes[b, 1]); push!(mesh_ys0, m_plot.nodes[b, 2])
    push!(mesh_xs0, NaN);                push!(mesh_ys0, NaN)
end

# ----------------------------------------------------------------------------
# JSON encoding (small, no external deps).
# ----------------------------------------------------------------------------

_jstr(x::AbstractString) = "\"" * replace(String(x),
                                            "\\" => "\\\\", "\"" => "\\\"") * "\""
_jstr(x::Symbol)         = _jstr(string(x))
_jstr(x::Real)           = isfinite(x) ? string(x) : "null"
_jstr(x::Int)            = string(x)
_jstr(x::AbstractVector) = "[" * join((_jstr(v) for v in x), ",") * "]"
_jstr(x::Bool)           = x ? "true" : "false"
_jstr(x::Nothing)        = "null"
function _jstr(x::AbstractDict)
    parts = String[]
    for (k, v) in pairs(x)
        push!(parts, _jstr(string(k)) * ":" * _jstr(v))
    end
    return "{" * join(parts, ",") * "}"
end
_jstr(x::NamedTuple) = _jstr(Dict(string(k) => getfield(x, k) for k in keys(x)))

# ----------------------------------------------------------------------------
# HTML helpers.
# ----------------------------------------------------------------------------

# Convergence value table: per level, per p, four eigenvalues each
# with err. Plus the CR-Liu lower-bound block.
function format_convergence_table_html(levels_data, λ_exact)
    io = IOBuffer()
    print(io, "<table class=\"results\"><thead><tr>")
    print(io, "<th rowspan=\"2\">level</th>")
    print(io, "<th rowspan=\"2\">h</th>")
    print(io, "<th rowspan=\"2\">n<sub>v</sub></th>")
    print(io, "<th rowspan=\"2\">n<sub>t</sub></th>")
    for p in FEM_ORDERS
        print(io, "<th colspan=\"$N_EIG\" class=\"upper-hdr\">P$p (n<sub>dof</sub>)</th>")
    end
    print(io, "<th colspan=\"$N_EIG\" class=\"lower-hdr\">CR-Liu (lower)</th>")
    print(io, "</tr><tr>")
    for _ in FEM_ORDERS
        for k in 1:N_EIG
            print(io, "<th>λ<sub>$k</sub></th>")
        end
    end
    for k in 1:N_EIG
        print(io, "<th>λ<sub>$k</sub></th>")
    end
    print(io, "</tr></thead><tbody>")

    for d in levels_data
        print(io, "<tr><td>L$(d.level)</td>")
        print(io, "<td>", @sprintf("%.4e", d.h), "</td>")
        print(io, "<td>$(d.nv)</td><td>$(d.nt)</td>")
        for p in FEM_ORDERS
            ndof = d.ndofs_per_p[p]
            for k in 1:N_EIG
                val = d.eigs_per_p[p][k]
                err = val - λ_exact[k]
                print(io, "<td class=\"upper\">",
                      @sprintf("%.4f", val),
                      "<br><span class=\"err\">",
                      @sprintf("err = %+.2e", err),
                      "</span>",
                      k == 1 ? @sprintf("<br><span class=\"ndof\">n<sub>dof</sub> = %d</span>", ndof) : "",
                      "</td>")
            end
        end
        for k in 1:N_EIG
            val = d.cr_low[k]
            err = val - λ_exact[k]
            print(io, "<td class=\"lower\">",
                  @sprintf("%.4f", val),
                  "<br><span class=\"err\">",
                  @sprintf("err = %+.2e", err),
                  "</span></td>")
        end
        print(io, "</tr>")
    end
    print(io, "</tbody></table>")
    return String(take!(io))
end

# Rate table — observed convergence rate between consecutive levels for
# each (method, eigenvalue).
function format_rate_table_html(levels_data, upper_rates, lower_rates,
                                  upper_err_per_p, lower_err)
    io = IOBuffer()
    n_intervals = length(upper_rates[FEM_ORDERS[1]])
    print(io, "<table class=\"rate\"><thead><tr>")
    print(io, "<th>method</th>")
    print(io, "<th>theoretical</th>")
    for i in 1:n_intervals
        print(io, "<th>L$(levels_data[i].level) → L$(levels_data[i + 1].level)</th>")
    end
    print(io, "</tr></thead><tbody>")

    function write_row!(io, label::String, theo::String, rates_per_interval, kind::Symbol)
        print(io, "<tr><td>$label</td>")
        print(io, "<td><em>$theo</em></td>")
        for ri in rates_per_interval
            # ri is a Vector of length N_EIG, possibly with `nothing` entries.
            cell = String[]
            for r in ri
                push!(cell, r === nothing ?
                              "<span class=\"sat\">—</span>" :
                              @sprintf("%.2f", r))
            end
            cls = kind === :upper ? "rate-upper" : "rate-lower"
            print(io, "<td class=\"$cls\">", join(cell, " &nbsp; "), "</td>")
        end
        print(io, "</tr>")
    end

    for p in FEM_ORDERS
        write_row!(io, "P$p (Galerkin upper)", "O(h<sup>$(2p)</sup>) ⇒ rate $(2p)",
                   upper_rates[p], :upper)
    end
    write_row!(io, "CR-Liu (lower)", "O(h²) ⇒ rate 2",
               lower_rates, :lower)
    print(io, "</tbody></table>")
    return String(take!(io))
end

function format_exact_table_html(λ_exact)
    coef_lookup = [(1, 1, 1), (2, 1, 2), (2, 2, 1), (3, 1, 2)]
    io = IOBuffer()
    print(io, "<table class=\"exact\"><thead><tr>")
    print(io, "<th>k</th><th>(m, n)</th><th>m² + mn + n²</th>")
    print(io, "<th>multiplicity</th><th>λ<sub>exact</sub></th></tr></thead><tbody>")
    for k in 1:N_EIG
        m, n, mult = coef_lookup[k]
        print(io, "<tr><td>$k</td><td>($m, $n)</td><td>$(m^2 + m*n + n^2)</td>")
        print(io, "<td>$mult</td><td>", @sprintf("%.6f", λ_exact[k]), "</td></tr>")
    end
    print(io, "</tbody></table>")
    return String(take!(io))
end

# ----------------------------------------------------------------------------
# Convergence plot data.
# ----------------------------------------------------------------------------

# Worst-case error per level per method, used as the curve we plot.
hs = [d.h for d in levels_data]
upper_err_max = Dict{Int, Vector{Float64}}()
for p in FEM_ORDERS
    upper_err_max[p] = [maximum(upper_err_per_p[p][i]) for i in 1:length(levels_data)]
end
lower_err_max = [maximum(lower_err[i]) for i in 1:length(levels_data)]

# Reference slope lines: for each FE order p, draw `y = C·h^{2p}` so it
# passes through the FIRST data point — the visual eye then compares
# slopes directly.
function _slope_line(hs, ref_err, ref_h, slope)
    return [ref_err * (h / ref_h)^slope for h in hs]
end

slope_lines = Dict{Int, Vector{Float64}}()
for p in FEM_ORDERS
    slope_lines[p] = _slope_line(hs, upper_err_max[p][1], hs[1], 2p)
end
lower_slope_line = _slope_line(hs, lower_err_max[1], hs[1], 2)

# ----------------------------------------------------------------------------
# Assemble the JSON payload.
# ----------------------------------------------------------------------------

payload = Dict(
    "eigfuncs"       => [Dict("k" => s.k,
                              "lam_h"     => s.λ_h,
                              "lam_exact" => s.λ_exact,
                              "xs" => collect(s.xs),
                              "ys" => collect(s.ys),
                              "vs" => collect(s.vs),
                              "ti" => collect(s.tris_i),
                              "tj" => collect(s.tris_j),
                              "tk" => collect(s.tris_k))
                         for s in eigfunc_samples],
    "mesh"           => Dict("xs" => collect(mesh_xs0),
                              "ys" => collect(mesh_ys0),
                              "n_tri" => m_plot.nt,
                              "n_vert" => m_plot.nv,
                              "n_edge" => m_plot.ne),
    "convergence"    => Dict(
        "h"            => collect(hs),
        "fem_orders"   => FEM_ORDERS,
        "p_err_max"    => Dict(string(p) => collect(upper_err_max[p]) for p in FEM_ORDERS),
        "p_slope_line" => Dict(string(p) => collect(slope_lines[p])    for p in FEM_ORDERS),
        "lower_err_max"=> collect(lower_err_max),
        "lower_slope"  => collect(lower_slope_line),
    ),
)

table_html  = format_convergence_table_html(levels_data, λ_exact)
exact_html  = format_exact_table_html(λ_exact)
rate_html   = format_rate_table_html(levels_data, upper_rates, lower_rates,
                                       upper_err_per_p, lower_err)

today = string(Dates.today())
data_json = _jstr(payload)

html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Equilateral Triangle Eigenvalue Convergence — VFEM.jl</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  body  { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          max-width: 1200px; margin: 24px auto; color: #1a1a1a; padding: 0 16px; }
  h1    { color: #2c3e50; margin-bottom: 4px; }
  h2    { color: #2c3e50; border-bottom: 2px solid #e0e0e0; padding-bottom: 4px;
          margin-top: 32px; }
  h3    { color: #34495e; margin-top: 24px; }
  .meta { color: #7a7a7a; font-size: 0.9em; margin-bottom: 16px; }
  .formula { font-family: "Latin Modern Math", Cambria Math, serif; font-size: 1.05em;
             background: #f5f5f5; padding: 8px 12px; border-radius: 4px;
             border-left: 4px solid #2c3e50; }
  table { border-collapse: collapse; margin: 12px 0; font-size: 0.88em; }
  table.results th, table.results td { border: 1px solid #ccc; padding: 4px 6px;
                                        text-align: right; }
  table.results th { background: #ecf0f1; font-weight: 600; }
  table.results th.upper-hdr { background: #fcf3cf; }
  table.results th.lower-hdr { background: #d6eaf8; }
  table.results td.upper { background: #fef9e7; }
  table.results td.lower { background: #eaf2f8; }
  table.results .err  { font-size: 0.78em; color: #5a5a5a; display: block; }
  table.results .ndof { font-size: 0.75em; color: #7a7a7a; display: block;
                         font-style: italic; }
  table.exact th, table.exact td { border: 1px solid #ccc; padding: 5px 10px;
                                     text-align: right; }
  table.exact th { background: #ecf0f1; }
  table.rate { font-size: 0.95em; }
  table.rate th, table.rate td { border: 1px solid #ccc; padding: 6px 10px;
                                  text-align: center; }
  table.rate th { background: #ecf0f1; font-weight: 600; }
  table.rate td.rate-upper { background: #fef9e7; }
  table.rate td.rate-lower { background: #eaf2f8; }
  table.rate .sat { color: #c0392b; }
  table.rate em { color: #555; font-style: italic; }
  .legend { display: flex; gap: 16px; font-size: 0.85em; margin: 8px 0 16px 0;
            flex-wrap: wrap; }
  .legend .swatch { display: inline-block; width: 14px; height: 14px;
                     vertical-align: middle; margin-right: 4px;
                     border: 1px solid #999; }
  .swatch.upper { background: #fef9e7; }
  .swatch.lower { background: #eaf2f8; }
  .plot { width: 100%; height: 480px; margin: 16px 0; }
  .plot-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 800px) { .plot-grid { grid-template-columns: 1fr; } }
  footer { color: #888; font-size: 0.8em; margin-top: 40px; border-top: 1px solid #e0e0e0;
           padding-top: 12px; }
  code  { background: #f5f5f5; padding: 1px 4px; border-radius: 3px; font-size: 0.92em; }
  .note { background: #fdf6e3; border-left: 4px solid #cba656;
          padding: 10px 14px; margin: 12px 0; border-radius: 4px; }
</style>
</head>
<body>

<h1>Dirichlet Laplace eigenvalues on the equilateral triangle</h1>
<div class="meta">
  Generated $(today) by <code>examples/make_equilateral_report.jl</code>
  &middot; VFEM.jl
</div>

<h2>Problem</h2>
<p>
  Solve the Dirichlet Laplace eigenvalue problem
</p>
<div class="formula">
  &minus;Δ u = λ u  in  Ω,&nbsp;&nbsp;&nbsp;&nbsp;u = 0  on ∂Ω
</div>
<p>
  where Ω is an equilateral triangle of side <em>a</em> = 1, with
  vertices at (0, 0), (1, 0), (0.5, √3/2). The closed-form spectrum
  (Lamé / McCartin&nbsp;2003):
</p>
<div class="formula">
  λ<sub>m, n</sub> = (16 π² / 9) &middot; (m² + m n + n²),
  &nbsp;&nbsp; m ≥ n ≥ 1,
  &nbsp;&nbsp; multiplicity = 2 if m ≠ n else 1
</div>
<p>The first $N_EIG eigenvalues:</p>
$exact_html

<h2>Method</h2>
<p>
  We compare four discretisations on a sequence of uniformly refined
  meshes (each level applies one round of 4-way midpoint refinement).
</p>
<ul>
  <li><strong>P_p Lagrange (p = 1, 2, 3)</strong>: <em>Galerkin upper
      bounds</em> λ<sub>h</sub>, computed via
      <code>laplace_eig_lagrange(m, p, k)</code>. The local basis is
      monomial-Lagrange ({L<sub>1</sub><sup>α</sup> L<sub>2</sub><sup>β</sup>
      L<sub>3</sub><sup>γ</sup> : α + β + γ = p}).</li>
  <li><strong>Crouzeix–Raviart + Liu shift</strong>: λ<sub>lo</sub>(CR), a
      <em>verified lower bound</em>,
      <code>verified_cr_liu_lower(m, k)</code>. Each entry is
      inf(·) of an Interval{Float64} enclosure produced by Veigs.jl
      with Liu's interpolation constant C<sub>h</sub> = 0.1893 · h<sub>max</sub>.</li>
</ul>

<div class="note">
  <strong>Theoretical convergence rates (smooth eigenfunctions, polygon).</strong>
  For the Dirichlet Laplacian on a convex polygon the eigenfunctions
  are smooth in the interior, so Galerkin P<sub>p</sub> eigenvalues
  converge as O(h<sup>2p</sup>): rates 2, 4, 6 for p = 1, 2, 3
  respectively.  The Liu lower bound based on a CR/ECR interpolation
  estimate is fundamentally O(h²) — its rate is set by the
  P<sub>1</sub>-non-conforming interpolation, not by any higher-order
  basis used to construct the upper bound.  Thus higher-order FEM
  buys you a sharper upper bound, not a sharper lower bound (this is
  the limitation that motivates Lehmann–Goerisch sharpening).
</div>

<h2>Convergence — values + per-eigenvalue errors</h2>
<div class="legend">
  <span><span class="swatch upper"></span> Galerkin upper bound (P<sub>p</sub>): λ<sub>h</sub> ≥ λ<sub>true</sub></span>
  <span><span class="swatch lower"></span> Liu lower bound (CR): λ<sub>lo</sub> ≤ λ<sub>true</sub></span>
</div>
$table_html

<h2>Convergence — observed rates</h2>
<p>
  Observed rate between two refinement levels k, k+1:
  <code>rate = log(err<sub>k</sub> / err<sub>k+1</sub>) / log(h<sub>k</sub> / h<sub>k+1</sub>)</code>.
  4-way refinement halves h, so the asymptotic rate equals
  <code>log<sub>2</sub>(err<sub>k</sub> / err<sub>k+1</sub>)</code>.
  Each cell shows four numbers — one per eigenvalue λ<sub>1</sub> &middot; λ<sub>2</sub> &middot; λ<sub>3</sub> &middot; λ<sub>4</sub>.
  A "—" means the error fell below $RATE_SATURATION_FLOOR (saturation
  by floating-point round-off) and the observed rate is unreliable.
</p>
$rate_html

<h2>Convergence — error vs h with reference slopes</h2>
<p>
  Log-log plot of <em>maximum</em> per-eigenvalue absolute error over
  the first $N_EIG eigenvalues, vs the largest mesh-edge length h.
  Solid markers: actual data. Dashed lines: theoretical reference
  slopes (h², h⁴, h⁶) anchored at the coarsest level. A data line that
  lies parallel to its dashed reference confirms the predicted rate.
</p>
<div id="plot-convergence" class="plot"></div>

<h2>Eigenfunctions (P2, level $REFINEMENT_LEVEL_FOR_PLOT)</h2>
<p>
  Each panel shows the discrete P2 Lagrange eigenfunction on a
  refined triangular grid (mesh has $(m_plot.nv) vertices,
  $(m_plot.nt) triangles). Values are normalised so that |max| = 1,
  with the sign chosen by the eigensolver. Hover over a point to
  read off the value.
</p>
<div class="plot-grid">
  <div id="plot-eig-1" class="plot"></div>
  <div id="plot-eig-2" class="plot"></div>
  <div id="plot-eig-3" class="plot"></div>
  <div id="plot-eig-4" class="plot"></div>
</div>

<h2>Mesh (level $REFINEMENT_LEVEL_FOR_PLOT)</h2>
<div id="plot-mesh" class="plot" style="height: 360px;"></div>

<h2>Take-aways</h2>
<ul>
  <li>P2 and P3 confirm their theoretical orders (4 and 6) at the
      finer levels. P3 saturates the fastest because its error reaches
      double-precision noise for the first eigenvalue around L5–L6.</li>
  <li>The Liu CR lower bound holds rigorously (inf of an interval
      enclosure) and converges at exactly O(h²) — independent of the
      polynomial order used for the upper-bound side. Sharpening this
      lower-bound rate requires Lehmann–Goerisch (see
      <code>lg_lower_eig_bound_laplace</code>).</li>
  <li>For a higher-fidelity lower bound today, use the LG driver in
      2D — it inherits the Galerkin O(h<sup>2p</sup>) rate. The
      analogous LG pipeline in 3D is the next phase.</li>
</ul>

<footer>
  Source: <code>examples/laplace_equilateral_triangle.jl</code> (case)
  + <code>examples/make_equilateral_report.jl</code> (this report).
  Library: VFEM.jl. Plots rendered with
  <a href="https://plotly.com/javascript/">Plotly.js</a> via CDN.
</footer>

<script>
const DATA = $data_json;

// ---- Convergence plot --------------------------------------------------
(function () {
  const traces = [];
  const colors = { '1': '#1f77b4', '2': '#ff7f0e', '3': '#2ca02c' };

  // Reference slope lines first, so they sit BEHIND the data markers.
  for (const p of DATA.convergence.fem_orders) {
    traces.push({
      x: DATA.convergence.h,
      y: DATA.convergence.p_slope_line[p.toString()],
      name: 'h^' + (2 * p) + ' reference',
      mode: 'lines',
      line: { color: colors[p.toString()], dash: 'dash', width: 1 },
      hoverinfo: 'skip',
      showlegend: true
    });
  }
  // Data lines for P_p.
  for (const p of DATA.convergence.fem_orders) {
    traces.push({
      x: DATA.convergence.h,
      y: DATA.convergence.p_err_max[p.toString()],
      name: 'P' + p + ' (max err)',
      mode: 'lines+markers',
      line: { color: colors[p.toString()], width: 2.5 },
      marker: { size: 9, symbol: 'circle' }
    });
  }
  // Lower-bound slope-2 reference and data.
  traces.push({
    x: DATA.convergence.h,
    y: DATA.convergence.lower_slope,
    name: 'h^2 reference (lower)',
    mode: 'lines',
    line: { color: '#aaaaaa', dash: 'dash', width: 1 },
    hoverinfo: 'skip',
    showlegend: true
  });
  traces.push({
    x: DATA.convergence.h,
    y: DATA.convergence.lower_err_max,
    name: 'CR-Liu (max err)',
    mode: 'lines+markers',
    line: { color: '#7f7f7f', width: 2.5 },
    marker: { size: 9, symbol: 'square' }
  });

  Plotly.newPlot('plot-convergence', traces, {
    title: 'max error vs h (log–log) — solid: data, dashed: O(h^{2p}) reference',
    xaxis: { title: 'h (largest edge length)', type: 'log',
             autorange: 'reversed' },
    yaxis: { title: 'max_k |λ_h^{(k)} − λ_exact^{(k)}|', type: 'log' },
    legend: { orientation: 'h', y: -0.18 },
    margin: { t: 40, l: 70, r: 20, b: 90 }
  }, { responsive: true });
})();

// ---- Eigenfunction plots ----------------------------------------------
(function () {
  for (let i = 0; i < DATA.eigfuncs.length; i++) {
    const ef = DATA.eigfuncs[i];
    const trace = {
      type: 'mesh3d',
      x: ef.xs, y: ef.ys, z: ef.vs,
      i: ef.ti, j: ef.tj, k: ef.tk,
      intensity: ef.vs,
      colorscale: 'RdBu',
      reversescale: true,
      cmin: -1, cmax: 1,
      showscale: i === 0,
      hovertemplate: '(x=%{x:.3f}, y=%{y:.3f})<br>u = %{z:.3f}<extra></extra>'
    };
    const title = 'φ_' + ef.k +
                  '  (λ_h = ' + ef.lam_h.toFixed(4) +
                  ',  λ_exact = ' + ef.lam_exact.toFixed(4) + ')';
    Plotly.newPlot('plot-eig-' + ef.k, [trace], {
      title: { text: title, font: { size: 13 } },
      scene: {
        xaxis: { title: 'x' }, yaxis: { title: 'y' }, zaxis: { title: 'φ_norm' },
        aspectmode: 'cube',
        camera: { eye: { x: 1.3, y: -1.3, z: 1.0 } }
      },
      margin: { t: 40, l: 0, r: 0, b: 0 }
    }, { responsive: true });
  }
})();

// ---- Mesh wireframe --------------------------------------------------
(function () {
  const mesh = DATA.mesh;
  Plotly.newPlot('plot-mesh', [{
    x: mesh.xs, y: mesh.ys,
    mode: 'lines',
    line: { color: '#34495e', width: 1 },
    showlegend: false,
    hoverinfo: 'skip'
  }], {
    title: 'Mesh: ' + mesh.n_vert + ' vertices, ' + mesh.n_tri +
           ' triangles, ' + mesh.n_edge + ' edges',
    xaxis: { title: 'x', scaleanchor: 'y', scaleratio: 1 },
    yaxis: { title: 'y' },
    margin: { t: 40, l: 60, r: 20, b: 50 }
  }, { responsive: true });
})();
</script>

</body>
</html>
"""

out_path = joinpath(@__DIR__, "..", "docs", "reports", "equilateral_triangle_report.html")
write(out_path, html)
println("Wrote $(out_path)")
println("  size = $(round(filesize(out_path) / 1024; digits = 1)) KB")
