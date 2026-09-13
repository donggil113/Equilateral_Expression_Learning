"""Why exact invariance can be statistically suboptimal even for an invariant label.

A fourth consequence of non-orthogonality, and one that bears directly on the H1
crossover.  Measurement noise lives in the *electrode* frame: it is added after the
heart vector is rotated, and is roughly isotropic across leads.  But the gauge that
makes the action orthogonal is not itself orthogonal, so isotropic lead-space noise
becomes **anisotropic** in heart-vector coordinates, with covariance
$(D^{\\top}D)^{-1}$ and condition number exactly $\\kappa(D)^2$ ($2.11$ for Dower,
$4.80$ for Kors).

The consequence is that the pose carries information about how much noise corrupts
the discriminative directions, so the Bayes-optimal classifier is *not* exactly
invariant even when the label is.  An exactly invariant model is then slightly
misspecified -- by construction, not by bad architecture.

This experiment isolates the effect by re-running the comparison with noise injected
in the heart frame instead, where the whole observation is an exact group action on a
pose-independent random variable and exact invariance *is* optimal.  If the
equivariant model's large-sample deficit shrinks under heart-frame noise, this
mechanism is a genuine part of the explanation; if it does not, it is not.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from common import make_splits, save, set_threads
from eqecg.leads import geometry
from eqecg.train import TrainConfig, evaluate, make_model, train_model


def main(sizes=(250, 4000), seeds: int = 2, steps: int = 800) -> dict:
    set_threads()
    rows = []
    for seed in range(seeds):
        for frame in ("lead", "heart"):
            pool, val, test, n_classes = make_splits(
                max(sizes), 500, 1000, seed=seed, rotation="small",
                max_rotation_deg=30.0, noise_frame=frame,
            )
            for n in sizes:
                sub = {k: (v[:n] if isinstance(v, np.ndarray) else v)
                       for k, v in pool.items()}
                for name in ("equivariant", "resnet"):
                    model = make_model(name, n_classes)
                    train_model(model, sub, val,
                                TrainConfig(steps=steps, seed=seed, patience=4), n_classes)
                    m = evaluate(model, torch.as_tensor(test["x"]), test["y"], n_classes)
                    rows.append({"noise_frame": frame, "model": name, "n_train": n,
                                 "seed": seed, **m})
                    print(f"  [{frame:5s} {name:12s} n={n:5d} seed={seed}] "
                          f"f1={m['macro_f1']:.4f}", flush=True)
    return {
        "rows": rows,
        "noise_anisotropy": {
            name: float(geometry(name).anisotropy ** 2) for name in ("dower", "kors")
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=800)
    a = ap.parse_args()
    save("noise_frame", main(seeds=a.seeds, steps=a.steps))
