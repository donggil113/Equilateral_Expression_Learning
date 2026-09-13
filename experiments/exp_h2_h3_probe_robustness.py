"""H2 / Proposition 4 (nuisance leakage) and H3 (robustness), from one set of models.

**H2 / Prop 4.**  If a representation encodes the nuisance pose, a *linear* probe can
read the rotation back out of it.  A high probe ``R^2`` is direct evidence that the
encoder spends capacity modelling nuisance orbits rather than pathology.  For the
equivariant model the features are invariant by construction, so its ``R^2`` is
zero up to round-off -- that is a sanity check on the probe, not a finding.  The
finding is how large the baselines' ``R^2`` is.

The same probe is the diagnostic the paper proposes for *any* pretrained ECG
encoder: :func:`probe_embeddings` takes an arbitrary embedding matrix, so it applies
unchanged to a foundation-model checkpoint once one is available.

**H3.**  Robustness is measured under three corruptions, only the first of which is
in the group:

``rotation``      larger nuisance rotation than seen in training (in-group).
``lead_reversal`` LA-RA / LA-LL electrode swap -- an exact linear map that is *not*
                  a group element (see ``results/equivariance.json``).
``displacement``  precordial electrode movement, modelled by perturbing the transform
                  matrix itself, which leaves the dipole subspace entirely.

Reporting the out-of-group corruptions is the honest part: equivariance gives no
a-priori guarantee there, so any gain has to be measured rather than assumed.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.linear_model import RidgeCV

from common import make_splits, save, set_threads
from eqecg.data.synthetic import (
    DipoleECGSimulator,
    SimulatorConfig,
    electrode_displacement_transform,
    lead_reversal_matrix,
)
from eqecg.group import random_small_rotation
from eqecg.leads import GEOMETRY as geo
from eqecg.train import TrainConfig, evaluate, make_model, predict, train_model

MODELS = {
    "equivariant": ("equivariant", "none"),
    "vcg-resnet": ("vcg-resnet", "none"),
    "resnet-aug": ("resnet", "small"),
    "resnet": ("resnet", "none"),
}


def probe_embeddings(features: np.ndarray, targets: np.ndarray, n_fit: int = None) -> dict:
    """Linear (ridge) probe from embeddings to nuisance parameters, scored out-of-sample.

    ``R^2`` is computed per target dimension on a held-out half and averaged; the
    ridge strength is selected by generalised cross-validation on the fit half, so a
    low score cannot be blamed on regularisation.
    """
    n = len(features)
    n_fit = n_fit or n // 2
    f = (features - features[:n_fit].mean(0)) / (features[:n_fit].std(0) + 1e-8)
    model = RidgeCV(alphas=np.logspace(-3, 4, 20)).fit(f[:n_fit], targets[:n_fit])
    pred = model.predict(f[n_fit:])
    true = targets[n_fit:]
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {
        "r2_per_dim": r2.tolist(),
        "r2_mean": float(np.mean(r2)),
        "n_fit": int(n_fit),
        "n_eval": int(n - n_fit),
    }


def corrupt(test: dict, kind: str, strength, rng) -> torch.Tensor:
    x = torch.as_tensor(test["x"])
    if kind == "rotation":
        R = random_small_rotation(rng, strength, len(x))
        rho = torch.tensor(geo.rho_ext(R), dtype=torch.float32)
        return torch.einsum("bij,bjt->bit", rho, x)
    if kind == "lead_reversal":
        T = torch.tensor(lead_reversal_matrix(strength), dtype=torch.float32)
        return torch.einsum("ij,bjt->bit", T, x)
    if kind == "displacement":
        # Re-project the *observed* signal through a perturbed transform matrix:
        # x -> D' D^+ x + P_res x.  Re-synthesising from the latent heart vector
        # instead would silently also strip the noise, which would flatter whichever
        # model is most noise-sensitive rather than measuring displacement robustness.
        Dp = torch.tensor(
            electrode_displacement_transform(rng, strength), dtype=torch.float32
        )
        Dpinv = torch.tensor(geo.D_pinv, dtype=torch.float32)
        Pres = torch.tensor(geo.P_res, dtype=torch.float32)
        P8 = torch.tensor(geo.P8, dtype=torch.float32)
        A = P8 @ (Dp @ Dpinv + Pres)
        return torch.einsum("ij,bjt->bit", A, x)
    raise ValueError(kind)


def main(seeds: int = 3, steps: int = 900, n_train: int = 2000) -> dict:
    set_threads()
    rows, probes = [], []
    for seed in range(seeds):
        train, val, test, n_classes = make_splits(
            n_train, 600, 1200, seed=seed, rotation="small", max_rotation_deg=30.0
        )
        rng = np.random.default_rng(seed + 99)
        for label, (arch, augment) in MODELS.items():
            model = make_model(arch, n_classes)
            train_model(model, train, val,
                        TrainConfig(steps=steps, augment=augment, seed=seed, patience=5),
                        n_classes)

            clean = evaluate(model, torch.as_tensor(test["x"]), test["y"], n_classes)
            rows.append({"model": label, "seed": seed, "corruption": "none",
                         "strength": 0, **clean})

            # --- H2 / Prop 4: can a linear probe recover the nuisance rotation? -----
            feats = predict(model, torch.as_tensor(test["x"]))["features"]
            probes.append({
                "model": label, "seed": seed, "target": "rotation_vector",
                **probe_embeddings(feats, np.asarray(test["rotvec"])),
            })
            probes.append({
                "model": label, "seed": seed, "target": "rotation_matrix",
                **probe_embeddings(feats, np.asarray(test["R"]).reshape(len(feats), 9)),
            })

            # --- H3: robustness -----------------------------------------------------
            for kind, strengths in (
                ("rotation", (60.0, 90.0, 180.0)),
                ("lead_reversal", ("LA-RA", "LA-LL")),
                ("displacement", (0.05, 0.1, 0.2)),
            ):
                for s in strengths:
                    xc = corrupt(test, kind, s, np.random.default_rng(seed + 7))
                    m = evaluate(model, xc, test["y"], n_classes)
                    rows.append({"model": label, "seed": seed, "corruption": kind,
                                 "strength": s, **m,
                                 "delta_macro_f1": m["macro_f1"] - clean["macro_f1"]})
            print(f"  [{label:12s} seed={seed}] clean_f1={clean['macro_f1']:.4f}", flush=True)
    return {"robustness": rows, "probes": probes, "n_train": n_train}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=900)
    a = ap.parse_args()
    save("h2_h3_probe_robustness", main(a.seeds, a.steps))
