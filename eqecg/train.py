"""Training and evaluation harness shared by every experiment.

Design choices that matter for the comparisons being made:

*Equal optimisation budget.*  Models are trained for a fixed number of gradient
steps rather than a fixed number of epochs, so the sample-efficiency curves of H1
compare inductive bias and not how many updates a given training-set size happens
to buy.

*Augmentation is a training-loop property.*  The rotation-augmented baseline uses
exactly the same architecture as the plain baseline, so any difference is
attributable to the augmentation and not to capacity.

*Corruption evaluation never touches training.*  The H3 corruptions (electrode
displacement, lead reversal, out-of-range rotation) are applied only at test time,
and the displacement corruption is deliberately *outside* the group so that
equivariance cannot trivially explain robustness to it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from eqecg.group import geodesic_distance, random_rotation, random_small_rotation
from eqecg.leads import GEOMETRY, LeadGeometry
from eqecg.models.baselines import ResNet1D, VCGResNet1D, count_parameters
from eqecg.models.equivariant import EquivariantECGNet

__all__ = ["TrainConfig", "make_model", "train_model", "evaluate", "rotate_batch", "MODEL_ZOO"]


MODEL_ZOO = {
    "equivariant": lambda n, **kw: EquivariantECGNet(n_classes=n, **kw),
    "equivariant-naive-lift": lambda n, **kw: EquivariantECGNet(n_classes=n, naive_lift=True, **kw),
    "equivariant-relaxed": lambda n, **kw: EquivariantECGNet(
        n_classes=n, equivariance_leak=kw.pop("leak", 0.1), **kw
    ),
    "equivariant-pose-aware": lambda n, **kw: EquivariantECGNet(
        n_classes=n, pose_in_head=True, **kw
    ),
    "resnet": lambda n, **kw: ResNet1D(n_classes=n, **kw),
    "vcg-resnet": lambda n, **kw: VCGResNet1D(n_classes=n, **kw),
}


@dataclass
class TrainConfig:
    steps: int = 1500
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 1e-4
    augment: str = "none"                 # "none" | "small" | "haar"
    augment_max_deg: float = 30.0
    eval_every: int = 100
    patience: int = 6                     # evaluations without improvement
    pose_weight: float = 0.0
    seed: int = 0
    threads: int = 4
    warmup: int = 50


def make_model(name: str, n_classes: int, **kwargs) -> torch.nn.Module:
    if name not in MODEL_ZOO:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODEL_ZOO)}")
    return MODEL_ZOO[name](n_classes, **kwargs)


def rotate_batch(
    x: torch.Tensor,
    rng: np.random.Generator,
    regime: str = "small",
    max_deg: float = 30.0,
    geo: LeadGeometry = GEOMETRY,
) -> tuple[torch.Tensor, np.ndarray]:
    """Apply an independent nuisance rotation to every record in the batch."""
    n = x.shape[0]
    R = (
        random_rotation(rng, n)
        if regime == "haar"
        else random_small_rotation(rng, max_deg, n)
    )
    rho = torch.tensor(geo.rho_ext(R), dtype=x.dtype)
    return torch.einsum("bij,bjt->bit", rho, x), R


def _metrics(logits: np.ndarray, y: np.ndarray, n_classes: int) -> dict[str, float]:
    pred = logits.argmax(1)
    out = {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    try:
        prob = torch.softmax(torch.tensor(logits), dim=1).numpy()
        present = np.unique(y)
        if len(present) == n_classes:
            out["macro_auroc"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    except ValueError:
        pass
    return out


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 128) -> dict[str, np.ndarray]:
    model.eval()
    logits, feats, rots = [], [], []
    for i in range(0, len(x), batch_size):
        out = model(x[i : i + batch_size])
        logits.append(out["logits"].numpy())
        feats.append(out["features"].numpy())
        if "rotation" in out:
            rots.append(out["rotation"].numpy())
    res = {"logits": np.concatenate(logits), "features": np.concatenate(feats)}
    if rots:
        res["rotation"] = np.concatenate(rots)
    return res


def evaluate(
    model: torch.nn.Module, x: torch.Tensor, y: np.ndarray, n_classes: int
) -> dict[str, float]:
    return _metrics(predict(model, x)["logits"], y, n_classes)


def train_model(
    model: torch.nn.Module,
    train: dict,
    val: dict,
    cfg: TrainConfig,
    n_classes: int,
    verbose: bool = False,
) -> dict:
    """Train with a fixed step budget, cosine schedule and early stopping on val macro-F1."""
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed + 1000)

    xtr = torch.as_tensor(train["x"])
    ytr = torch.as_tensor(train["y"])
    xva, yva = torch.as_tensor(val["x"]), np.asarray(val["y"])
    Rtr = train.get("R")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda step: min(1.0, (step + 1) / max(cfg.warmup, 1))
        * 0.5
        * (1 + np.cos(np.pi * min(step / cfg.steps, 1.0))),
    )

    best = {"macro_f1": -1.0}
    best_state, stale, history = None, 0, []
    t0 = time.time()

    for step in range(cfg.steps):
        model.train()
        idx = rng.integers(0, len(xtr), size=min(cfg.batch_size, len(xtr)))
        xb, yb = xtr[idx], ytr[idx]
        R_applied = None
        if cfg.augment != "none":
            xb, R_applied = rotate_batch(xb, rng, cfg.augment, cfg.augment_max_deg)

        out = model(xb)
        loss = F.cross_entropy(out["logits"], yb)
        if cfg.pose_weight > 0 and "rotation" in out and Rtr is not None:
            target = torch.as_tensor(np.asarray(Rtr)[idx])
            if R_applied is not None:
                target = torch.as_tensor(R_applied, dtype=target.dtype) @ target
            loss = loss + cfg.pose_weight * F.mse_loss(out["rotation"], target)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()

        if (step + 1) % cfg.eval_every == 0 or step == cfg.steps - 1:
            m = evaluate(model, xva, yva, n_classes)
            m.update(step=step + 1, loss=float(loss.item()))
            history.append(m)
            if verbose:
                print(f"  step {step+1:5d} loss {loss.item():.4f} val_f1 {m['macro_f1']:.4f}")
            if m["macro_f1"] > best["macro_f1"]:
                best = m
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= cfg.patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_val": best,
        "history": history,
        "train_seconds": time.time() - t0,
        "n_parameters": count_parameters(model),
    }
