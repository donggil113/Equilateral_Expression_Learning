"""Is the sample-efficiency comparison confounded by a single shared learning rate?

The H1 curves use one optimiser configuration for every architecture, which is the
usual protocol but is not obviously fair: a constrained model can have a different
loss landscape and so a different optimal step size.  Reporting a deficit that is
really a tuning artefact would be a serious error, so this script sweeps the learning
rate for each architecture and reports the best each achieves.

If the ranking survives per-model tuning, the H1 conclusion stands; if it does not,
the H1 conclusion is the tuning and must be reported as such.
"""

from __future__ import annotations

import argparse

import torch

from common import make_splits, save, set_threads
from eqecg.train import TrainConfig, evaluate, make_model, train_model

LEARNING_RATES = (1e-3, 3e-3, 1e-2)


def main(sizes=(250, 4000), steps: int = 800, seeds: int = 1) -> dict:
    set_threads()
    rows = []
    for seed in range(seeds):
        pool, val, test, n_classes = make_splits(
            max(sizes), 500, 1000, seed=seed, rotation="small", max_rotation_deg=30.0
        )
        for n in sizes:
            import numpy as np

            sub = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in pool.items()}
            for name in ("equivariant", "resnet"):
                for lr in LEARNING_RATES:
                    model = make_model(name, n_classes)
                    train_model(model, sub, val,
                                TrainConfig(steps=steps, lr=lr, seed=seed, patience=4),
                                n_classes)
                    m = evaluate(model, torch.as_tensor(test["x"]), test["y"], n_classes)
                    rows.append({"model": name, "n_train": n, "lr": lr, "seed": seed, **m})
                    print(f"  [{name:12s} n={n:5d} lr={lr:.0e}] f1={m['macro_f1']:.4f}",
                          flush=True)
    return {"rows": rows, "learning_rates": list(LEARNING_RATES)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    a = ap.parse_args()
    save("lr_fairness", main(steps=a.steps))
