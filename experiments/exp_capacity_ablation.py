"""Is the equivariant model's large-sample deficit a matter of bias or of capacity?

The sample-efficiency curves cross: the constrained model wins with little data and
loses with a lot.  There are two very different explanations, and the paper must not
guess between them.

*Capacity*  -- the equivariant layers, at a matched parameter count, simply have less
usable expressive power, in which case widening the model closes the gap.
*Bias*      -- the symmetry itself excludes functions the task needs, in which case
widening changes nothing and the constraint is the binding one.

This ablation trains the equivariant model at several widths on the largest training
set and reports whether the deficit closes.
"""

from __future__ import annotations

import argparse

import torch

from common import make_splits, save, set_threads
from eqecg.train import TrainConfig, evaluate, make_model, train_model

WIDTHS = ((48, 24), (72, 36), (96, 48))


def main(seeds: int = 2, steps: int = 800, n_train: int = 4000) -> dict:
    set_threads()
    rows = []
    for seed in range(seeds):
        train, val, test, n_classes = make_splits(
            n_train, 500, 1000, seed=seed, rotation="small", max_rotation_deg=30.0
        )
        for width, vwidth in WIDTHS:
            model = make_model("equivariant", n_classes, width=width, vector_width=vwidth)
            info = train_model(model, train, val,
                               TrainConfig(steps=steps, seed=seed, patience=5), n_classes)
            m = evaluate(model, torch.as_tensor(test["x"]), test["y"], n_classes)
            rows.append({"model": "equivariant", "width": width, "vector_width": vwidth,
                         "seed": seed, "n_parameters": info["n_parameters"], **m})
            print(f"  [width={width:3d}] seed={seed} params={info['n_parameters']:7d} "
                  f"f1={m['macro_f1']:.4f}", flush=True)
        for width in (53, 106):
            model = make_model("resnet", n_classes, width=width)
            info = train_model(model, train, val,
                               TrainConfig(steps=steps, seed=seed, patience=5), n_classes)
            m = evaluate(model, torch.as_tensor(test["x"]), test["y"], n_classes)
            rows.append({"model": "resnet", "width": width, "seed": seed,
                         "n_parameters": info["n_parameters"], **m})
            print(f"  [resnet w={width:3d}] seed={seed} params={info['n_parameters']:7d} "
                  f"f1={m['macro_f1']:.4f}", flush=True)
    return {"rows": rows, "n_train": n_train}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    save("capacity_ablation", main(a.seeds))
