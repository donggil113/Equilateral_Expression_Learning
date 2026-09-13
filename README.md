# Equivariant Representation Learning for a Physically Known, Non-Orthogonal Group

Research code for a study of `SO(3)`-equivariant learning on the 12-lead
electrocardiogram, where the symmetry group is **derived from physics rather than
assumed** — and, unusually, acts **non-orthogonally**.

## The setting in one paragraph

Under the equivalent cardiac-dipole model the 12-lead ECG is a fixed linear image
`x(t) = D v(t)` of a 3-D heart vector, so patient posture, respiration and electrode
placement act as a rotation `R ∈ SO(3)` of `v`, inducing

```
x  ↦  ρ(R) x = D R D⁺ x
```

on lead space. This is an exactly computable three-parameter Lie group action. It is
**not orthogonal** in the Euclidean geometry of lead space (Dower: `κ(D) = 1.45`;
Kors: `κ(D) = 2.19`), so the standard steerable/equivariant toolkit — which assumes an
orthogonal action — does not apply verbatim.

## What is here

| Module | Contents |
|---|---|
| `eqecg/leads.py` | Lead geometry: exact limb-lead identities (rank-8 realisable space), Dower/Kors transforms, dipolar/non-dipolar split, the invariant metric `M`, the induced action |
| `eqecg/group.py` | `SO(3)` utilities: Haar sampling, exp/log, geodesic distance |
| `eqecg/theory/` | Machine-checkable versions of Theorems 1–3: intertwiner bases, Reynolds operators, Weyl-integration dimension counts, identifiability strata |
| `eqecg/models/` | Gauge-equivariant network, capacity-matched baselines, naive-lift ablation |
| `eqecg/data/` | Dipole-model ECG simulator with ground-truth pose; loaders for PTB-XL, CODE-15%, MIMIC-IV-ECG |
| `experiments/` | H1–H4, the nuisance-leakage probe, the capacity ablation |
| `paper/` | LaTeX source; every table is generated from `results/*.json` |

## Quick start

```bash
pip install numpy scipy pandas scikit-learn matplotlib torch wfdb h5py pytest

python -m pytest tests/ -q          # 52 tests pinning the theory and the architecture
python experiments/exp_theory.py    # reproduces every theoretical constant
./scripts/run_all_experiments.sh    # full suite (~2.5 h on 4 CPU cores)
python scripts/make_report.py       # regenerates paper/tables and paper/figures
```

## Headline numbers

Verified by `experiments/exp_theory.py` and `experiments/exp_equivariance.py`:

| Quantity | Value |
|---|---|
| Realisable / dipolar / non-dipolar rank | 8 / 3 / 5 |
| Anisotropy `κ(D)`, Dower / Kors | 1.451 / 2.191 |
| Euclidean orthogonality defect of `ρ(R)` | 1.19 (Dower) |
| `M`-invariance defect of `ρ(R)` (Theorem 1) | `6e-16` |
| Equivariant linear maps, 1→1 channel | 26 = 1 + 25 (matches Schur) |
| Reynolds asymmetry, Euclidean vs gauge | 0.189 vs 0.000 |
| Invariant fraction removed, degree-2 features | 55.6% |
| **Equivariance error, our model** | **`7e-7`** (float32 round-off) |
| Equivariance error, standard orthogonal-lift assumption | `4e-2` |
| Equivariance error, conventional 1-D CNN | `0.99` |

## On the data

**The empirical results are computed on simulated data.** PhysioNet and Zenodo were
unreachable from the environment this work was produced in (blocked by egress policy),
so the real corpora could not be downloaded. The loaders in `eqecg/data/datasets.py`
are complete and their parsing logic is unit-tested offline against fixtures
(`tests/test_datasets_offline.py`); run the `fetch` step anywhere those hosts are
reachable and everything downstream is local:

```bash
python -c "from eqecg.data.datasets import PTBXL; d=PTBXL('data'); d.fetch(); d.prepare()"
```

Simulation is also not purely a fallback: the true nuisance rotation is never recorded
in a clinical corpus, and no corpus exposes the non-dipolar-energy knob that H4 needs,
so those two claims are not falsifiable on real data alone.

## Honest summary of findings

- **Theorem 1 holds exactly.** The gauge construction is verified to `4e-15`, and the
  equivariant-map dimension matches the Schur prediction for every channel count tried.
- **The claim that non-orthogonality breaks the generalisation theory is too strong as
  usually stated.** In `L²(μ)` the Reynolds operator remains an orthogonal projection
  regardless. The failure is in the *estimator's* geometry (RKHS / weight-decay norm),
  where it becomes oblique; the repair and its exact constant `κ(T)^k` are given.
- **H1 is only partially supported.** Equivariance is a clear win with few labels and a
  measurable cost with many; the crossover is reported rather than tuned away.
- **A strictly invariant model provably cannot represent pose-defined labels** such as
  axis deviation. This is a theorem, not a disappointing result, and the
  invariant⊕equivariant decomposition of Theorem 3 is what repairs it.
