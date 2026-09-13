"""Shared scaffolding for the experiment scripts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eqecg.data.synthetic import (  # noqa: E402
    CLASS_NAMES,
    POSE_INVARIANT_CLASSES,
    DipoleECGSimulator,
    SimulatorConfig,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def make_splits(
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 0,
    classes: tuple[str, ...] = POSE_INVARIANT_CLASSES,
    **sim_kwargs,
) -> tuple[dict, dict, dict, int]:
    """Generate train/val/test from the simulator with a chosen label subset.

    Restricting to :data:`POSE_INVARIANT_CLASSES` by default is a deliberate
    methodological choice: the main comparison must be on labels that an invariant
    model can in principle represent.  The pose-defined ``AXIS`` class is studied
    separately, where it belongs.
    """
    rng = np.random.default_rng(seed)
    sim = DipoleECGSimulator(SimulatorConfig(**sim_kwargs))
    label_ids = np.array([CLASS_NAMES.index(c) for c in classes])
    remap = {int(v): i for i, v in enumerate(label_ids)}

    def draw(n: int) -> dict:
        raw = rng.choice(label_ids, size=n)
        d = sim.sample(n, rng, labels=raw)
        d["y"] = np.array([remap[int(v)] for v in d["y"]], dtype=np.int64)
        return d

    return draw(n_train), draw(n_val), draw(n_test), len(classes)


def save(name: str, payload: dict) -> Path:
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_default))
    print(f"[saved] {path}")
    return path


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def set_threads(n: int | None = None) -> None:
    n = n or int(os.environ.get("EQECG_THREADS", "4"))
    torch.set_num_threads(n)
