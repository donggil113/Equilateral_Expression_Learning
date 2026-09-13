"""Proposition 4 in its intended setting: self-supervised encoders.

The nuisance-leakage diagnostic is aimed at *pretrained* ECG encoders, so testing it
only on supervised baselines would miss the point. Here we pretrain encoders with a
SimCLR-style contrastive objective on unlabelled records -- the recipe ECG foundation
models actually use -- and then ask two questions of the frozen representation:

1. *How much pose does it encode?*  A ridge probe predicts the applied rotation from
   the embedding. A high $R^2$ means capacity is being spent distinguishing points
   within a single group orbit.
2. *Does that cost anything downstream?*  A linear classifier on the frozen embedding
   is evaluated at several label budgets.

Three encoders are compared: a conventional CNN pretrained without rotation
augmentation (the typical foundation-model recipe); the same CNN pretrained *with*
rotation augmentation (what VCG-augmentation methods such as 3KG do, our strongest
existing-practice control); and our equivariant encoder, whose probe $R^2$ is zero by
construction -- that is the sanity check on the probe, not the finding.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from common import make_splits, save, set_threads
from eqecg.train import make_model, predict, rotate_batch
from exp_h2_h3_probe_robustness import probe_embeddings


def augment(x: torch.Tensor, rng: np.random.Generator, rotate: bool) -> torch.Tensor:
    """Standard ECG contrastive augmentations, optionally plus a nuisance rotation."""
    B, _, T = x.shape
    if rotate:
        x, _ = rotate_batch(x, rng, "small", 30.0)
    # Random temporal crop-and-pad (a time shift of up to 1 s).
    shift = int(rng.integers(-T // 10, T // 10))
    x = torch.roll(x, shifts=shift, dims=-1)
    # Per-record amplitude scaling and additive noise.
    x = x * torch.tensor(rng.uniform(0.8, 1.25, size=(B, 1, 1)), dtype=x.dtype)
    x = x + 0.05 * x.std() * torch.randn_like(x)
    # Baseline wander.
    t = torch.linspace(0, 1, T)[None, None, :]
    freq = torch.tensor(rng.uniform(0.2, 0.8, size=(B, 1, 1)), dtype=x.dtype)
    phase = torch.tensor(rng.uniform(0, 6.28, size=(B, 1, 1)), dtype=x.dtype)
    return x + 0.08 * x.std() * torch.sin(2 * np.pi * freq * t + phase)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = len(z1)
    sim = z @ z.T / temperature
    sim.fill_diagonal_(-1e9)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(sim, target)


def pretrain(model, x: torch.Tensor, rotate: bool, steps: int, seed: int,
             batch_size: int = 64, lr: float = 2e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 5)
    feat_dim = model(x[:2])["features"].shape[1]
    head = torch.nn.Sequential(torch.nn.Linear(feat_dim, 128), torch.nn.SiLU(),
                               torch.nn.Linear(128, 64))
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr)
    model.train()
    for step in range(steps):
        idx = rng.integers(0, len(x), size=batch_size)
        xb = x[idx]
        z1 = head(model(augment(xb, rng, rotate))["features"])
        z2 = head(model(augment(xb, rng, rotate))["features"])
        loss = nt_xent(z1, z2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (step + 1) % 200 == 0:
            print(f"    ssl step {step+1}/{steps} loss {loss.item():.4f}", flush=True)
    return model


def linear_eval(train_f, train_y, test_f, test_y, n_labels: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_f))[:n_labels]
    mu, sd = train_f[idx].mean(0), train_f[idx].std(0) + 1e-8
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit((train_f[idx] - mu) / sd, train_y[idx])
    from sklearn.metrics import f1_score

    pred = clf.predict((test_f - mu) / sd)
    return float(f1_score(test_y, pred, average="macro", zero_division=0))


def main(seeds: int = 2, ssl_steps: int = 800, n_unlabelled: int = 3000) -> dict:
    set_threads()
    probes, evals = [], []
    for seed in range(seeds):
        pool, _, test, n_classes = make_splits(
            n_unlabelled, 200, 1000, seed=seed, rotation="small", max_rotation_deg=30.0
        )
        x_unlab = torch.as_tensor(pool["x"])
        for label, (arch, rotate) in {
            "cnn-ssl": ("resnet", False),
            "cnn-ssl+rot-aug": ("resnet", True),
            "equivariant-ssl": ("equivariant", False),
        }.items():
            print(f"  pretraining {label} (seed {seed})", flush=True)
            model = make_model(arch, n_classes)
            pretrain(model, x_unlab, rotate, ssl_steps, seed)
            model.eval()

            feats = predict(model, torch.as_tensor(test["x"]))["features"]
            probes.append({"encoder": label, "seed": seed, "target": "rotation_vector",
                           **probe_embeddings(feats, np.asarray(test["rotvec"]))})

            train_f = predict(model, x_unlab)["features"]
            mine = []
            for n_labels in (100, 500, 2000):
                if n_labels > len(train_f):
                    continue
                mine.append({
                    "encoder": label, "seed": seed, "n_labels": n_labels,
                    "macro_f1": linear_eval(train_f, np.asarray(pool["y"]), feats,
                                            np.asarray(test["y"]), n_labels, seed),
                })
            evals.extend(mine)
            print(f"    probe R2={probes[-1]['r2_mean']:.3f} | "
                  f"linear-eval f1={[round(e['macro_f1'], 3) for e in mine]}",
                  flush=True)
    return {"probes": probes, "linear_eval": evals, "n_unlabelled": n_unlabelled}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--ssl-steps", type=int, default=800)
    a = ap.parse_args()
    save("prop4_ssl_probe", main(a.seeds, a.ssl_steps))
