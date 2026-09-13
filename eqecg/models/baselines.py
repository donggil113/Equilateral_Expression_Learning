"""Non-equivariant reference architectures, matched in capacity.

The comparison the paper needs is "same budget, different inductive bias", so the
baseline is a conventional 1-D residual CNN over the twelve leads -- the standard
architecture in the ECG deep-learning literature -- with its width chosen to match
the equivariant model's parameter count.  Rotation *augmentation* is a property of
the training loop, not of the architecture, so the same class serves as both the
plain and the augmented baseline.

:class:`VCGResNet1D` is the important intermediate control: it consumes the
inverse-Dower VCG rather than the raw leads, which is exactly what existing
VCG-augmentation work does.  Including it prevents the paper from attributing to
equivariance a gain that merely comes from the change of coordinates.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from eqecg.leads import GEOMETRY, LeadGeometry

__all__ = ["ResNet1D", "VCGResNet1D", "count_parameters"]


class _ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int = 9, stride: int = 2):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, stride, pad)
        self.norm1 = nn.GroupNorm(1, c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, 1, pad)
        self.norm2 = nn.GroupNorm(1, c_out)
        self.act = nn.SiLU()
        self.skip = (
            nn.Identity() if (c_in == c_out and stride == 1)
            else nn.Conv1d(c_in, c_out, 1, stride)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class ResNet1D(nn.Module):
    """Standard 1-D residual CNN over the raw 12 leads."""

    def __init__(
        self,
        n_classes: int = 5,
        in_channels: int = 12,
        width: int = 46,
        depth: int = 4,
        kernel: int = 9,
        strides: tuple[int, ...] = (4, 2, 2, 2),
        predict_pose: bool = False,
    ):
        super().__init__()
        if len(strides) < depth:
            strides = tuple(strides) + (2,) * (depth - len(strides))
        blocks, c = [], in_channels
        for i in range(depth):
            c_out = width if i else width // 2
            blocks.append(_ResBlock(c, c_out, kernel, stride=strides[i]))
            c = c_out
        self.blocks = nn.Sequential(*blocks)
        self.feat_dim = 3 * c
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, 128), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(128, n_classes),
        )
        self.predict_pose = predict_pose
        if predict_pose:
            self.pose = nn.Linear(self.feat_dim, 6)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x)
        return torch.cat([h.mean(-1), h.amax(-1), h.std(-1)], dim=1)

    def invariant_features(self, x: torch.Tensor) -> torch.Tensor:
        """Named for interface parity with the equivariant model; these are *not*
        invariant, which is precisely what the H2 probe measures."""
        return self._features(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self._features(x)
        out = {"logits": self.head(feats), "features": feats}
        if self.predict_pose:
            from eqecg.models.equivariant import _gram_schmidt

            out["rotation"] = _gram_schmidt(self.pose(feats).reshape(-1, 2, 3))
        return out


class VCGResNet1D(ResNet1D):
    """Same CNN, but fed the 3-lead inverse-Dower VCG plus the 5 residual channels.

    This isolates the contribution of the *coordinate change* from that of
    *equivariance*: it sees exactly the same information as the equivariant model,
    in the same gauge, but with no symmetry constraint on its layers.
    """

    def __init__(self, n_classes: int = 5, **kwargs):
        kwargs.pop("in_channels", None)
        super().__init__(n_classes=n_classes, in_channels=8, **kwargs)
        geo: LeadGeometry = GEOMETRY
        import numpy as np

        lift = np.concatenate([geo.D_pinv, geo.residual_basis], axis=0)  # 8 x 12
        self.register_buffer("lift_matrix", torch.tensor(lift, dtype=torch.float32))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("ij,bjt->bit", self.lift_matrix, x)
        return super()._features(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
