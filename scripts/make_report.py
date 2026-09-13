"""Turn the results JSONs into the paper's tables and figures.

The paper never hard-codes a number: every table is generated from
``results/*.json`` by this script and pulled in with ``\\input``.  Re-running the
experiments and re-running this script is the only way a number in the PDF changes,
so the text cannot drift from the code.

Figure conventions follow the accessibility rules used throughout: a fixed
categorical hue order (never cycled), a distinct marker per series so identity
survives grayscale printing and colour-vision deficiency, a legend whenever more
than one series is drawn, and recessive grid lines.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "paper" / "tables"
FIGURES = ROOT / "paper" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Validated colourblind-safe categorical order; assigned by entity, never by rank.
PALETTE = {
    "equivariant": "#0072B2",
    "resnet": "#D55E00",
    "resnet-aug": "#009E73",
    "vcg-resnet": "#785EF0",
    "equivariant-relaxed": "#785EF0",
    "equivariant-pose-aware": "#009E73",
    "equivariant+pose-head": "#009E73",
}
MARKERS = {
    "equivariant": "o", "resnet": "s", "resnet-aug": "^", "vcg-resnet": "D",
    "equivariant-relaxed": "v", "equivariant-pose-aware": "P",
    "equivariant+pose-head": "P",
}
LABELS = {
    "equivariant": "Equivariant (ours)",
    "resnet": "ResNet-1D",
    "resnet-aug": "ResNet-1D + rot. aug.",
    "vcg-resnet": "VCG-ResNet-1D",
    "equivariant-relaxed": "Equivariant (relaxed)",
    "equivariant-pose-aware": "Equivariant + pose feats.",
    "equivariant+pose-head": "Equivariant + pose head",
}

plt.rcParams.update({
    "figure.dpi": 160, "savefig.bbox": "tight", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.markersize": 5.5,
})


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _mean_std(rows, key_fields, value):
    """Aggregate ``value`` over seeds, grouped by ``key_fields``."""
    acc = defaultdict(list)
    for r in rows:
        if value in r and r[value] is not None:
            acc[tuple(r[k] for k in key_fields)].append(r[value])
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in acc.items()}


def _tex(path: Path, body: str) -> None:
    path.write_text(body)
    print(f"[table] {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- theory
def theory_tables(d: dict) -> None:
    g = d["theorem1_geometry"]
    rows = "\n".join(
        f"  {name.capitalize()} & {v['singular_values_of_D'][0]:.3f} / "
        f"{v['singular_values_of_D'][1]:.3f} / {v['singular_values_of_D'][2]:.3f} & "
        f"{v['anisotropy_kappa_D']:.3f} & {v['metric_condition_number']:.3f} & "
        f"{v['max_euclidean_orthogonality_defect']:.3f} & "
        f"\\num{{{v['max_M_invariance_defect']:.1e}}} \\\\"
        for name, v in g.items()
    )
    _tex(TABLES / "geometry.tex", f"""\\begin{{tabular}}{{lccccc}}
\\toprule
Transform & $\\sigma(D)$ & $\\kappa(D)$ & $\\kappa(M)$ & Euclid.\\ defect & $M$ defect \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}""")

    rows = "\n".join(
        f"  {r['channels_in']} & {r['channels_out']} & {r['numerical_dimension']} & "
        f"{r['schur_prediction']} \\\\"
        for r in d["theorem1_intertwiner_dimensions"]
    )
    _tex(TABLES / "intertwiners.tex", f"""\\begin{{tabular}}{{cccc}}
\\toprule
$c_{{\\mathrm{{in}}}}$ & $c_{{\\mathrm{{out}}}}$ & numerical $\\dim$ & Schur prediction \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}""")

    rey = d["theorem2_reynolds"]
    _tex(TABLES / "reynolds.tex", f"""\\begin{{tabular}}{{lccc}}
\\toprule
Inner product & idempotency err.\\ & asymmetry $\\|S-S^{{\\!\\top}}\\|/\\|S\\|$ & $\\tr S$ \\\\
\\midrule
  Euclidean (standard assumption) & \\num{{{rey['euclidean']['relative_idempotency_error']:.1e}}} & {rey['euclidean']['relative_asymmetry']:.4f} & {rey['euclidean']['trace']:.1f} \\\\
  $M$ (Theorem~\\ref{{thm:gauge}}) & \\num{{{rey['M']['relative_idempotency_error']:.1e}}} & {rey['M']['relative_asymmetry']:.4f} & {rey['M']['trace']:.1f} \\\\
\\bottomrule
\\end{{tabular}}""")

    tbl = [r for r in d["theorem2_dimension_table"] if r["degree"] <= 3]
    rows = "\n".join(
        f"  {r['time_samples']} & {r['degree']} & {r['dim_total']} & {r['dim_invariant']} & "
        f"{100*r['fraction_removed']:.1f}\\% \\\\"
        for r in tbl
    )
    _tex(TABLES / "invariant_dims.tex", f"""\\begin{{tabular}}{{ccccc}}
\\toprule
$m$ samples & degree $k$ & $\\dim\\Sym^k V$ & $\\dim(\\Sym^k V)^G$ & removed \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}""")

    if "theorem2_norm_distortion" in d:
        rows = "\n".join(
            f"  {r['degree']} & {r['condition_number']:.3f} \\\\"
            for r in d["theorem2_norm_distortion"]
        )
        _tex(TABLES / "distortion.tex", """\\begin{tabular}{cc}
\\toprule
feature degree $k$ & congruence condition number $\\kappa(T)^k$ \\\\
\\midrule
""" + rows + """
\\bottomrule
\\end{tabular}""")

    rows = "\n".join(
        f"  {r['trajectory_rank']} & {r['noise_sigma']:.2f} & {r['pose_error_deg_mean']:.2f} & "
        f"{r['pose_error_deg_p95']:.2f} & {r['chirality_signal']:.4f} \\\\"
        for r in d["theorem3_identifiability"]
    )
    _tex(TABLES / "identifiability.tex", f"""\\begin{{tabular}}{{ccccc}}
\\toprule
$\\rank V$ & noise $\\sigma$ & pose err.\\ (deg) & p95 & chirality signal \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}""")


def equivariance_table(d: dict) -> None:
    rows = []
    for arch, regimes in d["architectures"].items():
        for regime, m in regimes.items():
            rows.append(
                f"  {LABELS.get(arch, arch)} & {regime.replace('_',' ')} & "
                f"\\num{{{m['relative_logit_error_max']:.2e}}} & "
                f"{m.get('pose_equivariance_error_deg_max', float('nan')):.2f} \\\\"
            )
    _tex(TABLES / "equivariance.tex", """\\begin{tabular}{llcc}
\\toprule
Architecture & Rotation regime & rel.\\ logit error & pose error (deg) \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}""")


# ------------------------------------------------------------------------------- H1
def h1_outputs(d: dict) -> None:
    rows = d["rows"]
    agg = _mean_std(rows, ("model", "n_train"), "test_macro_f1")
    sizes = sorted({r["n_train"] for r in rows})
    models = [m for m in ["equivariant", "resnet-aug", "vcg-resnet", "resnet"]
              if any(r["model"] == m for r in rows)]

    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    for m in models:
        xs = [n for n in sizes if (m, n) in agg]
        mu = np.array([agg[(m, n)][0] for n in xs])
        sd = np.array([agg[(m, n)][1] for n in xs])
        ax.plot(xs, mu, marker=MARKERS[m], color=PALETTE[m], label=LABELS[m])
        ax.fill_between(xs, mu - sd, mu + sd, color=PALETTE[m], alpha=0.13, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("labelled training records")
    ax.set_ylabel("test macro-F1")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.savefig(FIGURES / "h1_sample_efficiency.pdf")
    fig.savefig(FIGURES / "h1_sample_efficiency.png")
    plt.close(fig)
    print("[figure] paper/figures/h1_sample_efficiency.pdf")

    header = " & ".join(f"$n={n}$" for n in sizes)
    body = []
    for m in models:
        cells = []
        for n in sizes:
            if (m, n) in agg:
                mu, sd, _ = agg[(m, n)]
                cells.append(f"{mu:.3f}\\,\\tiny{{$\\pm${sd:.3f}}}")
            else:
                cells.append("--")
        body.append(f"  {LABELS[m]} & " + " & ".join(cells) + " \\\\")
    _tex(TABLES / "h1.tex", f"""\\begin{{tabular}}{{l{'c'*len(sizes)}}}
\\toprule
Model & {header} \\\\
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular}}""")

    # Data multiplier: labelled records each baseline needs to match the equivariant
    # model at each training-set size (by interpolation on its own curve).
    mult = []
    for n in sizes:
        if ("equivariant", n) not in agg:
            continue
        target = agg[("equivariant", n)][0]
        for m in models:
            if m == "equivariant":
                continue
            xs = [k for k in sizes if (m, k) in agg]
            ys = [agg[(m, k)][0] for k in xs]
            if not ys or target > max(ys):
                needed = float("inf")
            else:
                needed = float(np.interp(target, ys, np.log(xs)))
                needed = float(np.exp(needed))
            mult.append({"n_train": n, "baseline": m, "equivariant_f1": target,
                         "baseline_records_needed": needed,
                         "multiplier": needed / n if np.isfinite(needed) else float("inf")})
    (RESULTS / "h1_data_multiplier.json").write_text(json.dumps(mult, indent=2))
    print("[derived] results/h1_data_multiplier.json")


# ---------------------------------------------------------------------------- H2/H3
def h2_h3_outputs(d: dict) -> None:
    probes = d["probes"]
    agg = _mean_std(probes, ("model", "target"), "r2_mean")
    models = sorted({p["model"] for p in probes}, key=lambda m: list(LABELS).index(m)
                    if m in LABELS else 99)
    rows = []
    for m in models:
        cells = []
        for t in ("rotation_vector", "rotation_matrix"):
            if (m, t) in agg:
                mu, sd, _ = agg[(m, t)]
                cells.append(f"{mu:.3f}\\,\\tiny{{$\\pm${sd:.3f}}}")
            else:
                cells.append("--")
        rows.append(f"  {LABELS.get(m,m)} & " + " & ".join(cells) + " \\\\")
    _tex(TABLES / "h2_probe.tex", """\\begin{tabular}{lcc}
\\toprule
Encoder & probe $R^2$ (rotation vector) & probe $R^2$ (rotation matrix) \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}""")

    rob = d["robustness"]
    keys = [("none", 0), ("rotation", 60.0), ("rotation", 90.0), ("rotation", 180.0),
            ("lead_reversal", "LA-RA"), ("lead_reversal", "LA-LL"),
            ("displacement", 0.05), ("displacement", 0.1), ("displacement", 0.2)]
    agg = _mean_std(rob, ("model", "corruption", "strength"), "macro_f1")
    names = {("none", 0): "clean", ("rotation", 60.0): "rot.\\ $60^\\circ$",
             ("rotation", 90.0): "rot.\\ $90^\\circ$", ("rotation", 180.0): "rot.\\ Haar",
             ("lead_reversal", "LA-RA"): "LA--RA swap",
             ("lead_reversal", "LA-LL"): "LA--LL swap",
             ("displacement", 0.05): "electrode $0.05$",
             ("displacement", 0.1): "electrode $0.1$",
             ("displacement", 0.2): "electrode $0.2$"}
    models = [m for m in ["equivariant", "resnet-aug", "vcg-resnet", "resnet"]
              if any(r["model"] == m for r in rob)]
    body = []
    for m in models:
        cells = [f"{agg[(m,c,s)][0]:.3f}" if (m, c, s) in agg else "--" for c, s in keys]
        body.append(f"  {LABELS[m]} & " + " & ".join(cells) + " \\\\")
    _tex(TABLES / "h3_robustness.tex", f"""\\begin{{tabular}}{{l{'c'*len(keys)}}}
\\toprule
Model & {" & ".join(names[k] for k in keys)} \\\\
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular}}""")

    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    width = 0.8 / max(len(models), 1)
    xs = np.arange(len(keys))
    for i, m in enumerate(models):
        vals = [agg[(m, c, s)][0] if (m, c, s) in agg else np.nan for c, s in keys]
        ax.bar(xs + i * width - 0.4 + width / 2, vals, width * 0.88,
               color=PALETTE[m], label=LABELS[m], edgecolor="white", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([names[k].replace("\\\\", "").replace("$", "").replace("\\circ", "deg")
                        for k in keys], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("test macro-F1")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.savefig(FIGURES / "h3_robustness.pdf")
    fig.savefig(FIGURES / "h3_robustness.png")
    plt.close(fig)
    print("[figure] paper/figures/h3_robustness.pdf")


# ------------------------------------------------------------------------------- H4
def h4_outputs(d: dict) -> None:
    rows = d["nondipolar_sweep"]
    agg = _mean_std(rows, ("model", "nondipolar_fraction"), "macro_f1")
    fracs = sorted({r["nondipolar_fraction"] for r in rows})
    models = sorted({r["model"] for r in rows})

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for m in models:
        xs = [f for f in fracs if (m, f) in agg]
        mu = [agg[(m, f)][0] for f in xs]
        ax.plot(xs, mu, marker=MARKERS.get(m, "o"), color=PALETTE.get(m, "#555"),
                label=LABELS.get(m, m))
    ax.set_xlabel("non-dipolar energy fraction")
    ax.set_ylabel("test macro-F1")
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(FIGURES / "h4_nondipolar.pdf")
    fig.savefig(FIGURES / "h4_nondipolar.png")
    plt.close(fig)
    print("[figure] paper/figures/h4_nondipolar.pdf")

    body = []
    for m in models:
        cells = [f"{agg[(m,f)][0]:.3f}" if (m, f) in agg else "--" for f in fracs]
        body.append(f"  {LABELS.get(m,m)} & " + " & ".join(cells) + " \\\\")
    leak = _mean_std(rows, ("nondipolar_fraction",), "learned_leak_weight")
    leak_row = " & ".join(f"{leak[(f,)][0]:.3f}" if (f,) in leak else "--" for f in fracs)
    # Python 3.11 forbids backslashes inside f-string expressions, so build the
    # header separately rather than inline.
    eta_header = " & ".join("$\\eta=" + str(f) + "$" for f in fracs)
    _tex(TABLES / "h4_nondipolar.tex", f"""\\begin{{tabular}}{{l{'c'*len(fracs)}}}
\\toprule
Model & {eta_header} \\\\
\\midrule
{chr(10).join(body)}
\\midrule
  learned leak weight & {leak_row} \\\\
\\bottomrule
\\end{{tabular}}""")

    rows = d["pose_dependent_labels"]
    agg_f1 = _mean_std(rows, ("model",), "macro_f1")
    agg_ax = _mean_std(rows, ("model",), "recall_AXIS")
    agg_pi = _mean_std(rows, ("model",), "recall_pose_invariant_mean")
    body = []
    for m in agg_f1:
        body.append(
            f"  {LABELS.get(m[0], m[0])} & {agg_f1[m][0]:.3f} & "
            f"{agg_pi.get(m,(float('nan'),))[0]:.3f} & {agg_ax.get(m,(float('nan'),))[0]:.3f} \\\\"
        )
    _tex(TABLES / "h4_axis.tex", """\\begin{tabular}{lccc}
\\toprule
Model & macro-F1 (5-class) & recall, pose-invariant classes & recall, \\textsc{axis} \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}""")


def capacity_table(d: dict) -> None:
    agg = _mean_std(d["rows"], ("model", "width"), "macro_f1")
    par = _mean_std(d["rows"], ("model", "width"), "n_parameters")
    body = [
        f"  {LABELS.get(m,m)} & {w} & {int(par[(m,w)][0])} & {agg[(m,w)][0]:.3f} \\\\"
        for (m, w) in sorted(agg, key=lambda k: (k[0], k[1]))
    ]
    _tex(TABLES / "capacity.tex", """\\begin{tabular}{lccc}
\\toprule
Model & width & parameters & test macro-F1 \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}""")


def noise_frame_table(d: dict) -> None:
    agg = _mean_std(d["rows"], ("noise_frame", "model", "n_train"), "macro_f1")
    sizes = sorted({r["n_train"] for r in d["rows"]})
    body = []
    for frame in ("lead", "heart"):
        for m in ("equivariant", "resnet"):
            cells = [f"{agg[(frame,m,n)][0]:.3f}" if (frame, m, n) in agg else "--"
                     for n in sizes]
            body.append(f"  {frame} & {LABELS.get(m,m)} & " + " & ".join(cells) + " \\\\")
        gaps = []
        for n in sizes:
            a, b = (frame, "equivariant", n), (frame, "resnet", n)
            gaps.append(f"{agg[a][0]-agg[b][0]:+.3f}" if a in agg and b in agg else "--")
        body.append("  \\multicolumn{2}{l}{\\quad gap (equiv.\\ $-$ ResNet)} & "
                    + " & ".join(gaps) + " \\\\")
        body.append("  \\midrule" if frame == "lead" else "")
    header = " & ".join("$n=" + str(n) + "$" for n in sizes)
    _tex(TABLES / "noise_frame.tex", f"""\\begin{{tabular}}{{ll{'c'*len(sizes)}}}
\\toprule
noise frame & model & {header} \\\\
\\midrule
{chr(10).join(x for x in body if x)}
\\bottomrule
\\end{{tabular}}""")


def lr_fairness_table(d: dict) -> None:
    agg = _mean_std(d["rows"], ("model", "n_train", "lr"), "macro_f1")
    sizes = sorted({r["n_train"] for r in d["rows"]})
    lrs = d["learning_rates"]
    body = []
    for m in ("equivariant", "resnet"):
        for n in sizes:
            cells = [f"{agg[(m,n,lr)][0]:.3f}" if (m, n, lr) in agg else "--" for lr in lrs]
            best = max((agg[(m, n, lr)][0] for lr in lrs if (m, n, lr) in agg), default=0.0)
            body.append(f"  {LABELS.get(m,m)} & {n} & " + " & ".join(cells)
                        + f" & \\textbf{{{best:.3f}}} \\\\")
    header = " & ".join("$\\eta=" + f"{lr:.0e}" + "$" for lr in lrs)
    _tex(TABLES / "lr_fairness.tex", f"""\\begin{{tabular}}{{ll{'c'*len(lrs)}c}}
\\toprule
model & $n$ & {header} & best \\\\
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular}}""")


def main() -> None:
    handlers = [
        ("theory", theory_tables), ("equivariance", equivariance_table),
        ("h1_sample_efficiency", h1_outputs),
        ("h2_h3_probe_robustness", h2_h3_outputs),
        ("h4_dipole_breakdown", h4_outputs), ("capacity_ablation", capacity_table),
        ("noise_frame", noise_frame_table), ("lr_fairness", lr_fairness_table),
    ]
    for name, fn in handlers:
        d = load(name)
        if d is None:
            print(f"[skip] results/{name}.json not present yet")
            continue
        fn(d)


if __name__ == "__main__":
    main()
