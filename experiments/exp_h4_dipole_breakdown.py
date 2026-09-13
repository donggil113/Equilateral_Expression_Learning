"""H4: what happens when the dipole approximation -- and hence the symmetry -- fails.

H4 is the paper's designated falsification risk, so it is set up to be able to
fail.  Two distinct failure modes are separated, because they call for different
remedies.

**(a) Non-dipolar signal energy.**  The generator injects a controlled fraction of
signal energy into the five-dimensional residual subspace ``Vres``.  The group does
not move that component, so the *action* stays exactly correct; what degrades is the
premise that the diagnostic information is carried by the rotating part.  Sweeping
the fraction shows whether strict equivariance becomes a liability, and at what
point.

**(b) Pose-dependent labels.**  Some real ECG findings (axis deviation, rotation of
the heart) are *defined by* orientation.  A strictly invariant model provably cannot
represent them -- this is not an empirical question but a theorem, and the ``AXIS``
control class measures the resulting ceiling directly.  The remedy is the paper's
positive contribution rather than a retreat: because the architecture is
*equivariant* and not merely invariant, the pose head recovers exactly this
information, and the invariant/equivariant split of Theorem 3 lets one model serve
both label types.

The relaxed model (``equivariance_leak``) is the third option -- approximate
equivariance with a learned symmetry-breaking weight -- and the learned weight is
reported, since its magnitude is itself the measurement of how much symmetry the
data actually has.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from common import make_splits, save, set_threads
from eqecg.data.synthetic import CLASS_NAMES, POSE_INVARIANT_CLASSES
from eqecg.train import TrainConfig, evaluate, make_model, train_model

FRACTIONS = (0.0, 0.1, 0.25, 0.5)


def part_a(seeds: int, steps: int, n_train: int, fractions) -> list[dict]:
    """Sweep the non-dipolar energy fraction."""
    rows = []
    for seed in range(seeds):
        for frac in fractions:
            train, val, test, n_classes = make_splits(
                n_train, 400, 800, seed=seed, rotation="small",
                max_rotation_deg=30.0, nondipolar_fraction=frac,
            )
            for label in ("equivariant", "equivariant-relaxed", "resnet"):
                model = make_model(label, n_classes)
                train_model(model, train, val,
                            TrainConfig(steps=steps, seed=seed, patience=4), n_classes)
                m = evaluate(model, torch.as_tensor(test["x"]), test["y"], n_classes)
                row = {"model": label, "nondipolar_fraction": frac, "seed": seed, **m}
                if hasattr(model, "leak_weight"):
                    row["learned_leak_weight"] = float(model.leak_weight.detach().abs())
                rows.append(row)
                print(f"  [a] frac={frac:.2f} {label:20s} seed={seed} "
                      f"f1={m['macro_f1']:.4f}", flush=True)
    return rows


def part_b(seeds: int, steps: int, n_train: int) -> list[dict]:
    """The pose-defined ``AXIS`` class: the ceiling of strict invariance, and its fix."""
    rows = []
    for seed in range(seeds):
        train, val, test, n_classes = make_splits(
            n_train, 400, 800, seed=seed, classes=CLASS_NAMES,
            rotation="small", max_rotation_deg=30.0,
        )
        axis_idx = CLASS_NAMES.index("AXIS")
        for label in ("equivariant", "equivariant-relaxed", "resnet"):
            model = make_model(label, n_classes)
            train_model(model, train, val,
                        TrainConfig(steps=steps, seed=seed, patience=4), n_classes)
            from eqecg.train import predict
            logits = predict(model, torch.as_tensor(test["x"]))["logits"]
            pred, y = logits.argmax(1), np.asarray(test["y"])
            overall = evaluate(model, torch.as_tensor(test["x"]), y, n_classes)
            per_class = {
                CLASS_NAMES[c]: float((pred[y == c] == c).mean())
                for c in range(n_classes)
                if (y == c).any()
            }
            rows.append({
                "model": label, "seed": seed, "task": "5-class incl. AXIS",
                **overall,
                "recall_AXIS": per_class.get("AXIS", float("nan")),
                "recall_pose_invariant_mean": float(
                    np.mean([per_class[c] for c in POSE_INVARIANT_CLASSES if c in per_class])
                ),
            })
            print(f"  [b] {label:20s} seed={seed} f1={overall['macro_f1']:.4f} "
                  f"AXIS_recall={rows[-1]['recall_AXIS']:.3f}", flush=True)

        # The architecture that actually embodies the decomposition of Theorem 3:
        # the invariant features carry pathology, and the *equivariant* (time-pooled
        # vector) features are exposed to the classifier as well, so orientation-
        # defined labels become representable.  A pose head alone does not achieve
        # this -- its prediction never reaches the classifier, whose features remain
        # strictly invariant -- which is why both are measured.
        for label in ("equivariant-pose-aware", "equivariant+pose-head"):
            if label == "equivariant-pose-aware":
                model = make_model(label, n_classes)
                cfg = TrainConfig(steps=steps, seed=seed, patience=4)
            else:
                model = make_model("equivariant", n_classes, predict_pose=True)
                cfg = TrainConfig(steps=steps, seed=seed, patience=4, pose_weight=1.0)
            train_model(model, train, val, cfg, n_classes)
            from eqecg.train import predict

            out = predict(model, torch.as_tensor(test["x"]))
            pred, y = out["logits"].argmax(1), np.asarray(test["y"])
            overall = evaluate(model, torch.as_tensor(test["x"]), y, n_classes)
            rows.append({
                "model": label, "seed": seed, "task": "5-class incl. AXIS",
                **overall,
                "recall_AXIS": float((pred[y == axis_idx] == axis_idx).mean()),
                "recall_pose_invariant_mean": float(np.mean([
                    (pred[y == CLASS_NAMES.index(c)] == CLASS_NAMES.index(c)).mean()
                    for c in POSE_INVARIANT_CLASSES if (y == CLASS_NAMES.index(c)).any()
                ])),
            })
            print(f"  [b] {label:22s} seed={seed} f1={overall['macro_f1']:.4f} "
                  f"AXIS_recall={rows[-1]['recall_AXIS']:.3f}", flush=True)

    return rows


def main(seeds: int = 2, steps: int = 700, n_train: int = 1500) -> dict:
    set_threads()
    return {
        "nondipolar_sweep": part_a(seeds, steps, n_train, FRACTIONS),
        "pose_dependent_labels": part_b(seeds, steps, n_train),
        "fractions": list(FRACTIONS),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=700)
    a = ap.parse_args()
    save("h4_dipole_breakdown", main(a.seeds, a.steps))
