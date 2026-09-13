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

**Theory — holds exactly.**
- The gauge construction of Theorem 1 is verified to `4e-15`, and the dimension of the
  equivariant linear maps matches the Schur-lemma prediction for every channel count
  tried (26 = 1 + 25 for one channel).
- **The claim that non-orthogonality breaks the generalisation theory is too strong as
  usually stated.** In `L²(μ)` the Reynolds operator remains an *orthogonal* projection
  regardless of the action. The failure is in the *estimator's* geometry (RKHS /
  weight-decay norm), where it becomes oblique (asymmetry 0.189 vs 0.000 in the gauge).
  The repair and its exact constant `κ(T)^k` are given.
- Theorem 3's three identifiability strata are confirmed numerically, including the
  chirality ambiguity for planar VCG loops.

**Architecture — the headline empirical fact.** Our model is invariant to float32
round-off (`7e-7`); the standard orthogonal-lift assumption errs by `1.3e-1` and a
conventional CNN by `0.80`. Five orders of magnitude.

**H1 (sample efficiency) — NOT supported.** At matched parameters and a shared
optimiser configuration, the equivariant model is within ±0.025 macro-F1 of the best
baseline at every training-set size. There is no 10–100x data multiplier. A single
seed in our pilot showed +0.08 at n=125; it did not replicate (three-seed spread at
that size is ±0.026) and was noise.

But two controls show the large-sample *deficit* is not a property of the symmetry:
- **Capacity**: widening the equivariant model closes it entirely (0.779 → 0.798 at
  n=4000, versus 0.794–0.796 for baselines at two widths). Matched parameters is not
  matched capacity for these architectures.
- **Learning rate**: with per-architecture tuning the ordering reverses at n=4000
  (equivariant 0.799 at η=1e-2 vs baseline best 0.791). The usual shared-LR protocol
  was costing the constrained model most of its apparent deficit.

**H2 / Proposition 4 (nuisance leakage) — NOT supported for supervised encoders.** Every
probe R² is indistinguishable from zero. A model trained end-to-end on invariant labels
discards the pose, so the diagnostic is uninformative exactly where a label signal
exists. The hypothesis conflated "the input contains the nuisance" with "the
representation retains it". The self-supervised setting is where it has something to
say.

**H3 (robustness) — STRONGLY supported; the clearest empirical result.** In-group the
equivariant model is exactly flat (0.749 at 60°, 90° and Haar) while baselines fall to
0.47–0.52 under Haar. The advantage *persists out of group*, where equivariance offers
no guarantee: LA–RA swap 0.726 vs 0.518–0.673; LA–LL 0.610 vs 0.390–0.576; strongest
electrode displacement 0.728 vs 0.553–0.708. We did not predict this and do not claim
to explain it.

Notably the VCG-ResNet — same gauge, no constraint — is the *worst* model under every
corruption. The coordinate change alone is not merely insufficient, it is harmful
without the constraint that motivates it. That is the cleanest evidence that the
contribution is the equivariance and not the inverse-Dower transform existing
augmentation methods already use.

**H4 (dipole breakdown) — partially supported.** The effect exists but is mild, and it
hurts every model rather than selectively punishing the invariant one. The relaxed
variant is uniformly best. One sub-claim fails cleanly: the learned symmetry-breaking
weight does *not* track the non-dipolar fraction (0.208 → 0.178, if anything the wrong
way), so it is a useful architecture but not a useful measuring instrument.

**A strictly invariant model provably cannot represent pose-defined labels** such as
axis deviation (AXIS recall 0.298 vs 0.808 for a CNN). This is a theorem, not a
disappointing result, and the invariant ⊕ equivariant decomposition of Theorem 3 is
what repairs it.

**A fourth consequence of non-orthogonality, apparently new.** Isotropic noise in the
*electrode* frame becomes anisotropic in heart-vector coordinates, with condition
number exactly `κ(D)²` (2.11 Dower, 4.80 Kors), so the Bayes-optimal classifier is not
exactly invariant even when the label is. Making the noise rotation-covariant shrinks
the equivariant model's large-sample gap from −0.038 to −0.004 — about nine tenths of
it. This is a concrete cost of exact invariance with no analogue in the orthogonal
setting the literature assumes.

**One architectural detail dominates all of the above.** Without an equivariant
normalisation the gated nonlinearity starves the vector path (magnitudes decay ~80x
over four blocks), which reads as "equivariance doesn't scale". Fixing it moved
macro-F1 at n=4000 from 0.735 to 0.776. Any study reporting that an equivariant
architecture underperforms should check this first.
