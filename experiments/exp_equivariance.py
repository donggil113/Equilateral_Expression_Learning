"""Table 1: measured equivariance error of each architecture under the true action.

This is the experiment that converts Theorem 1 from a remark into a number.  Every
model is evaluated on realisable ECG, then on the same ECG acted on by ``rho_ext(R)``,
and the relative change of the logits is reported.  A model that is genuinely
equivariant changes by float32 round-off; one built on the orthogonality assumption
that the standard steerable literature makes does not.

The measurement is made with *untrained* weights on purpose: equivariance is a
property of the architecture, and training on augmented data can only ever make a
non-equivariant model approximately invariant on the data distribution it saw.
"""

from __future__ import annotations

import numpy as np
import torch

from common import RESULTS, make_splits, save, set_threads
from eqecg.group import geodesic_distance, random_rotation, random_small_rotation
from eqecg.leads import GEOMETRY as geo
from eqecg.train import make_model


def equivariance_error(model, x: torch.Tensor, rotations, pose: bool = False) -> dict:
    model.eval()
    with torch.no_grad():
        base = model(x)
        rel_logits, pose_err = [], []
        for R in rotations:
            rho = torch.tensor(geo.rho_ext(R), dtype=torch.float32)
            out = model(torch.einsum("ij,bjt->bit", rho, x))
            denom = base["logits"].abs().max().clamp_min(1e-12)
            rel_logits.append(float((out["logits"] - base["logits"]).abs().max() / denom))
            if pose and "rotation" in out:
                Rt = torch.tensor(R, dtype=torch.float32)
                pose_err.append(
                    float(
                        np.rad2deg(
                            geodesic_distance(
                                (Rt @ base["rotation"]).numpy(), out["rotation"].numpy()
                            )
                        ).max()
                    )
                )
    res = {
        "relative_logit_error_mean": float(np.mean(rel_logits)),
        "relative_logit_error_max": float(np.max(rel_logits)),
    }
    if pose_err:
        res["pose_equivariance_error_deg_max"] = float(np.max(pose_err))
    return res


def main() -> dict:
    set_threads()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    _, _, test, n_classes = make_splits(8, 8, 64, seed=0, rotation="none")
    x = torch.tensor(test["x"])

    regimes = {
        "small_rotation_30deg": random_small_rotation(rng, 30.0, 16),
        "haar_rotation": random_rotation(rng, 16),
    }

    out: dict = {"n_test": int(x.shape[0]), "architectures": {}}
    for name in ("equivariant", "equivariant-naive-lift", "resnet", "vcg-resnet"):
        model = make_model(name, n_classes, predict_pose=True)
        out["architectures"][name] = {
            regime: equivariance_error(model, x, Rs, pose=True)
            for regime, Rs in regimes.items()
        }

    # How far is a lead reversal from being an element of the group?  If it were, an
    # SO(3)-equivariant model would be invariant to it for free; the residual below
    # shows it is not, which is why H3 is a genuine out-of-group robustness test.
    from eqecg.data.synthetic import lead_reversal_matrix
    from eqecg.theory.identifiability import recover_rotation

    reversal = {}
    for kind in ("LA-RA", "LA-LL"):
        T = lead_reversal_matrix(kind)
        # Best approximation of T|_Vdip by a group element, in the gauge.
        A = geo.D_pinv @ T @ geo.D                       # 3x3 action on VCG coordinates
        R_best = recover_rotation(np.eye(3), A)
        reversal[kind] = {
            "residual_to_nearest_group_element": float(
                np.linalg.norm(A - R_best) / np.linalg.norm(A)
            ),
            "nearest_rotation_angle_deg": float(
                np.rad2deg(geodesic_distance(np.eye(3), R_best))
            ),
            "acts_on_residual_subspace": float(
                np.linalg.norm(geo.P_res @ T @ geo.P_res - geo.P_res)
                / np.linalg.norm(geo.P_res)
            ),
        }
    out["lead_reversal_vs_group"] = reversal
    return out


if __name__ == "__main__":
    save("equivariance", main())
