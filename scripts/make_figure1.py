#!/usr/bin/env python3
"""Figure 1: the induced action, and the sense in which it is non-orthogonal.

Four panels, each carrying one claim of Section 3:

(a) the VCG loop is a 3-D trajectory that the nuisance rotates;
(b) the twelve observed leads change substantially under that rotation, so the
    nuisance is not a small perturbation to be ignored;
(c) the Euclidean norm of a lead vector is *not* preserved by the action, while the
    norm in the metric of Theorem 1 is preserved exactly -- this is the whole
    argument of the paper in one panel;
(d) the Gram matrix of the VCG trajectory, the complete invariant of Theorem 3, is
    unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eqecg.data.synthetic import DipoleECGSimulator, SimulatorConfig  # noqa: E402
from eqecg.group import random_small_rotation, rotation_matrix  # noqa: E402
from eqecg.leads import GEOMETRY as geo  # noqa: E402
from eqecg.leads import LEAD_NAMES  # noqa: E402

BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 170, "savefig.bbox": "tight", "font.size": 8.5,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
})


def main() -> None:
    rng = np.random.default_rng(4)
    sim = DipoleECGSimulator(
        SimulatorConfig(rotation="none", baseline_wander=0.0, muscle_noise=0.0,
                        powerline_noise=0.0, class_effect_scale=0.0)
    )
    d = sim.sample(1, rng)
    x = d["x"][0]
    v = geo.to_vcg(x)
    # A fixed 35-degree rotation about the sagittal axis: a realistic posture change,
    # and large enough that the panel shows what it does rather than hinting at it.
    R = rotation_matrix(np.array([0.0, 1.0, 0.0]), np.deg2rad(35.0))
    xr = geo.rho_ext(R) @ x
    vr = geo.to_vcg(xr)

    fig = plt.figure(figsize=(10.2, 2.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.25, 1.0, 1.05], wspace=0.42)

    # (a) the VCG loop, before and after the nuisance rotation
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    peak = int(np.argmax(np.linalg.norm(v, axis=0)))
    seg = slice(peak - 28, peak + 42)          # one P-QRS-T loop around the R peak
    ax.plot(*v[:, seg], color=BLUE, lw=1.5, label="VCG $v(t)$")
    ax.plot(*vr[:, seg], color=ORANGE, lw=1.5, label=r"$Rv(t)$")
    ax.set_xlabel("X", labelpad=-8); ax.set_ylabel("Y", labelpad=-8)
    ax.set_zlabel("Z", labelpad=-8)
    ax.tick_params(labelsize=5, pad=-2)
    ax.set_title("(a) heart vector", fontsize=9)
    ax.legend(fontsize=6, frameon=False, loc="upper left")

    # (b) the observed leads move a great deal
    ax = fig.add_subplot(gs[0, 1])
    win = slice(peak - 40, peak + 120)
    t = np.arange(win.stop - win.start) / 100.0
    step = 1.9 * max(np.abs(x[[0, 1, 6]][:, win]).max(), np.abs(xr[[0, 1, 6]][:, win]).max())
    for k, li in enumerate((0, 1, 6)):
        off = -k * step
        a, b = x[li, win] + off, xr[li, win] + off
        ax.fill_between(t, a, b, color=ORANGE, alpha=0.30, linewidth=0)
        ax.plot(t, a, color=BLUE, lw=1.3)
        ax.plot(t, b, color=ORANGE, lw=1.3, ls="--")
        ax.text(t[-1] * 1.01, off, LEAD_NAMES[li], fontsize=7.5, color="#444",
                va="center")
    ax.set_yticks([]); ax.set_xlabel("time (s)"); ax.set_xlim(0, t[-1] * 1.12)
    ax.set_title(r"(b) $x$ (solid) vs $\rho(R)x$ (dashed)", fontsize=9, pad=7)

    # (c) THE point: Euclidean norm is not preserved, the M norm is
    ax = fig.add_subplot(gs[0, 2])
    Rs = random_small_rotation(rng, 30.0, 400)
    cols = x[:, 60:260]
    eu, me = [], []
    base_e = np.linalg.norm(cols)
    base_m = np.sqrt(np.trace(cols.T @ geo.M_ext @ cols))
    for Ri in Rs:
        y = geo.rho_ext(Ri) @ cols
        eu.append(np.linalg.norm(y) / base_e)
        me.append(np.sqrt(np.trace(y.T @ geo.M_ext @ y)) / base_m)
    ax.hist(eu, bins=28, color=ORANGE, alpha=0.85, label="Euclidean")
    ax.axvline(1.0, color="#333", lw=1.0, ls=":")
    ax.plot([np.mean(me)], [0], marker="^", color=BLUE, ms=9, clip_on=False,
            label=r"$M$ (Thm. 1)")
    ax.set_xlabel(r"$\|\rho(R)x\| \,/\, \|x\|$")
    ax.set_ylabel("count")
    ax.set_title("(c) the action is not orthogonal", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)

    # (d) the complete invariant is untouched
    ax = fig.add_subplot(gs[0, 3])
    G = v[:, seg].T @ v[:, seg]
    Gr = vr[:, seg].T @ vr[:, seg]
    im = ax.imshow(G, cmap="viridis")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("(d) Gram matrix (invariant)", fontsize=9)
    ax.set_xlabel(rf"$\max|G(v)-G(Rv)| = ${np.abs(G - Gr).max():.1e}", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figure1_action.{ext}")
    plt.close(fig)
    print(f"[figure] {OUT/'figure1_action.pdf'}")
    print(f"  Euclidean norm ratio: {np.min(eu):.4f} .. {np.max(eu):.4f} "
          f"(spread {np.max(eu)-np.min(eu):.4f})")
    print(f"  M norm ratio        : {np.min(me):.6f} .. {np.max(me):.6f}")


if __name__ == "__main__":
    main()
