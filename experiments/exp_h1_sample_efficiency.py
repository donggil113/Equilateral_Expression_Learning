"""H1: sample efficiency of the equivariant model against matched baselines.

All four models have the same parameter budget (~137k) and the same optimisation
budget, and differ only in inductive bias:

``equivariant``   exact ``SO(3)``-equivariance in the gauge of Theorem 1.
``vcg-resnet``    same gauge, no symmetry constraint -- isolates the coordinate change.
``resnet-aug``    raw leads, trained with rotation augmentation -- the strongest
                  existing practice (this is what VCG-augmentation methods do).
``resnet``        raw leads, no augmentation.

The headline quantity is the *data multiplier*: how many labelled records each
baseline needs to match the equivariant model's score at a given training-set size.
"""

from __future__ import annotations

import argparse

import numpy as np

from common import make_splits, save, set_threads
from eqecg.train import TrainConfig, evaluate, make_model, train_model

SIZES = (60, 125, 250, 500, 1000, 2000, 4000)
MODELS = {
    "equivariant": ("equivariant", "none"),
    "vcg-resnet": ("vcg-resnet", "none"),
    "resnet-aug": ("resnet", "small"),
    "resnet": ("resnet", "none"),
}


def main(seeds: int = 3, steps: int = 800, sizes=SIZES) -> dict:
    set_threads()
    rows = []
    # One large pool per seed; the smaller training sets are nested prefixes of it so
    # the curves differ only in how much data is used, not in which data.
    for seed in range(seeds):
        pool, val, test, n_classes = make_splits(
            max(sizes), 600, 1200, seed=seed, rotation="small", max_rotation_deg=30.0
        )
        for label, (arch, augment) in MODELS.items():
            for n in sizes:
                sub = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in pool.items()}
                model = make_model(arch, n_classes)
                cfg = TrainConfig(steps=steps, augment=augment, seed=seed, eval_every=100,
                                  patience=4)
                info = train_model(model, sub, val, cfg, n_classes)
                metrics = evaluate(model, __import__("torch").as_tensor(test["x"]),
                                   test["y"], n_classes)
                rows.append(
                    {
                        "model": label, "n_train": n, "seed": seed,
                        "n_parameters": info["n_parameters"],
                        "train_seconds": info["train_seconds"],
                        **{f"test_{k}": v for k, v in metrics.items()},
                    }
                )
                print(f"  [{label:12s} n={n:5d} seed={seed}] "
                      f"test_f1={metrics['macro_f1']:.4f} ({info['train_seconds']:.0f}s)",
                      flush=True)
    return {"rows": rows, "sizes": list(sizes), "models": list(MODELS)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=800)
    a = ap.parse_args()
    save("h1_sample_efficiency", main(a.seeds, a.steps))
