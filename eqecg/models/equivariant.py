"""An ``SO(3)``-equivariant network for the 12-lead ECG, built in the gauge of Theorem 1.

The architecture is deliberately plain: every layer is one of five primitives that
provably commute with the induced action, so equivariance holds by construction and
is verified numerically to floating-point precision rather than encouraged by a
penalty.

Feature types
    ``s`` of shape ``(B, Cs, T)``     -- ``l = 0`` (invariant) channels.
    ``v`` of shape ``(B, Cv, 3, T)``  -- ``l = 1`` (vector) channels, each a heart
    vector that the group rotates.

The primitives
    *temporal convolution*   applied per channel and **shared across the three vector
      components**, with no bias on the vector path: a bias would add a fixed vector
      and destroy equivariance.  This is the one place where an off-the-shelf
      implementation silently breaks the symmetry.
    *channel mixing*         arbitrary linear maps on the multiplicity index -- by
      Theorem 1 these exhaust the equivariant linear maps between vector types.
    *invariant read-outs*    norms ``||v_c||`` and inner products ``<v_a, v_b>``,
      which by the first fundamental theorem generate all ``O(3)`` invariants.
    *cross products*         ``v_a x v_b``, the pseudo-vector that distinguishes
      ``SO(3)`` from ``O(3)`` and so lets the network see VCG-loop chirality.
    *gated nonlinearity*     vectors are rescaled by a scalar gate; applying a
      pointwise nonlinearity to vector components directly would not be equivariant.

The crucial difference from a standard steerable network is the entry point.
:class:`GaugeLift` maps the observed leads through ``D^+`` into the coordinates where
the action is an honest rotation.  Feeding the twelve leads to a conventional
``SO(3)``-equivariant block instead -- which implicitly assumes the action is
orthogonal -- is *not* equivariant, and :class:`NaiveOrthogonalLift` exists precisely
so that the paper can measure how large that error is.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eqecg.leads import GEOMETRY, LeadGeometry

__all__ = [
    "GaugeLift",
    "NaiveOrthogonalLift",
    "VectorNorm",
    "GVPBlock",
    "EquivariantECGNet",
    "gram_features",
]


class GaugeLift(nn.Module):
    """Observed leads -> (dipolar vector, non-dipolar scalars), in the Theorem-1 gauge.

    ``v = D^+ x`` transforms as ``v -> R v`` exactly, and ``s = C x`` is pointwise
    invariant, so the pair is a faithful, lossless re-coordinatisation of the
    realisable rank-8 lead space into ``l=1 (+) 5 x l=0``.
    """

    def __init__(self, geo: LeadGeometry = GEOMETRY):
        super().__init__()
        self.register_buffer("Dpinv", torch.tensor(geo.D_pinv, dtype=torch.float32))
        self.register_buffer("C", torch.tensor(geo.residual_basis, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = torch.einsum("ij,bjt->bit", self.Dpinv, x).unsqueeze(1)  # (B,1,3,T)
        s = torch.einsum("ij,bjt->bit", self.C, x)                   # (B,5,T)
        return v, s


class NaiveOrthogonalLift(nn.Module):
    """The lift a standard steerable network would implicitly use -- and why it fails.

    Existing equivariant architectures assume the group acts orthogonally, which for
    lead space amounts to reading the heart vector off an *orthonormalised* basis of
    the dipolar subspace (here the ``Q`` factor of ``D``) rather than through ``D^+``.
    The two agree only when ``D^T D`` is a multiple of the identity.  Because it is
    not, this lift is equivariant to the wrong group, and the resulting error is
    reported in the paper rather than assumed to be negligible.
    """

    def __init__(self, geo: LeadGeometry = GEOMETRY):
        super().__init__()
        Q = np.linalg.qr(geo.D)[0]                    # 12 x 3, orthonormal columns
        self.register_buffer("lift", torch.tensor(Q.T, dtype=torch.float32))
        self.register_buffer("C", torch.tensor(geo.residual_basis, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = torch.einsum("ij,bjt->bit", self.lift, x).unsqueeze(1)
        s = torch.einsum("ij,bjt->bit", self.C, x)
        return v, s


class VectorNorm(nn.Module):
    """Equivariant normalisation for vector channels -- the ``l=1`` analogue of LayerNorm.

    Without it the architecture has a real defect: the gated nonlinearity multiplies
    vectors by a sigmoid, which is always below one, so vector magnitudes decay
    geometrically with depth (measured: a factor of ~80 over four blocks) while the
    scalar stream stays healthy under its GroupNorm.  The vector path is then starved
    and the model degenerates towards its scalar half, which would look like evidence
    that equivariance is unhelpful when it is really a normalisation bug.

    Each channel is divided by the root-mean-square of its norm over time.  That
    divisor is an *invariant* scalar, so the operation commutes with the group action
    and equivariance is preserved exactly.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        scale = v.pow(2).sum(dim=2).mean(dim=-1, keepdim=True).sqrt()   # (B, C, 1)
        return v / (scale.unsqueeze(-1) + self.eps) * self.gain[None, :, None, None]


class GVPBlock(nn.Module):
    """One equivariant block: temporal convolution, tensor products, gated nonlinearity."""

    def __init__(
        self,
        cs_in: int,
        cv_in: int,
        cs_out: int,
        cv_out: int,
        kernel: int = 9,
        stride: int = 2,
        n_pairs: int = 8,
        use_cross: bool = True,
    ):
        super().__init__()
        pad = kernel // 2
        self.use_cross = use_cross
        self.n_pairs = n_pairs

        # Vector path: shared across the three components, and strictly bias-free.
        self.vector_conv = nn.Conv1d(cv_in, cv_out, kernel, stride, pad, bias=False)
        self.vector_norm = VectorNorm(cv_out)
        # Learned mixes whose inner/cross products supply invariants and pseudo-vectors.
        self.mix_a = nn.Conv1d(cv_out, n_pairs, 1, bias=False)
        self.mix_b = nn.Conv1d(cv_out, n_pairs, 1, bias=False)

        # Scalar path consumes the original scalars plus the invariants read off the
        # vectors: cv_out norms and n_pairs inner products.
        self.scalar_conv = nn.Conv1d(cs_in, cs_out, kernel, stride, pad)
        self.invariant_proj = nn.Conv1d(cv_out + n_pairs, cs_out, 1)
        self.scalar_norm = nn.GroupNorm(1, cs_out)
        self.gate = nn.Conv1d(cs_out, cv_out, 1)
        if use_cross:
            self.cross_proj = nn.Conv1d(n_pairs, cv_out, 1, bias=False)

    @staticmethod
    def _channel_conv(conv: nn.Conv1d, v: torch.Tensor) -> torch.Tensor:
        """Apply a 1-D convolution over time, shared across the vector components."""
        B, C, _, T = v.shape
        v = v.permute(0, 2, 1, 3).reshape(B * 3, C, T)
        v = conv(v)
        return v.reshape(B, 3, v.shape[1], v.shape[2]).permute(0, 2, 1, 3)

    def forward(self, v: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.vector_norm(self._channel_conv(self.vector_conv, v))  # (B, cv_out, 3, T')
        a = self._channel_conv(self.mix_a, v)
        b = self._channel_conv(self.mix_b, v)

        norms = torch.linalg.norm(v, dim=2)                  # (B, cv_out, T') invariant
        dots = (a * b).sum(dim=2)                            # (B, n_pairs, T') invariant

        s = self.scalar_conv(s) + self.invariant_proj(torch.cat([norms, dots], dim=1))
        s = F.silu(self.scalar_norm(s))

        if self.use_cross:
            # a x b is equivariant for SO(3) and sign-flips under reflection, giving
            # the network access to VCG-loop chirality (cf. Theorem 3).
            cross = torch.cross(a, b, dim=2)
            v = v + self._channel_conv(self.cross_proj, cross)

        v = v * torch.sigmoid(self.gate(s)).unsqueeze(2)     # gated, hence equivariant
        return v, s


def gram_features(v: torch.Tensor) -> torch.Tensor:
    """Invariant Gram descriptors of the vector channels.

    Theorem 3 identifies the Gram matrix *of the trajectory* -- the pairwise inner
    products of the vectors at every time point -- as a complete invariant.  Two
    contractions of it are used here:

    ``time-averaged Gram``   ``(1/T) sum_t <v_c(t), v_d(t)>``, which keeps the
        temporal covariance structure of the VCG loop, and
    ``Gram of the time means``  ``<mean_t v_c, mean_t v_d>``, which keeps the loop's
        net orientation-free magnitude.

    Taking only the second (the obvious implementation) averages the trajectory away
    before contracting it and discards nearly all morphology -- the distinction
    matters a great deal empirically, not just formally.
    """
    B, C, _, T = v.shape
    idx = torch.triu_indices(C, C, device=v.device)
    G_traj = torch.einsum("bcit,bdit->bcd", v, v) / T
    V = v.mean(dim=-1)
    G_mean = torch.einsum("bci,bdi->bcd", V, V)
    return torch.cat([G_traj[:, idx[0], idx[1]], G_mean[:, idx[0], idx[1]]], dim=1)


class EquivariantECGNet(nn.Module):
    """Gauge-equivariant encoder with an invariant classification head and a pose head.

    Parameters
    ----------
    n_classes:
        number of diagnostic outputs.
    predict_pose:
        also regress the nuisance rotation as a 3x3 matrix.  The pose head is
        *equivariant* rather than invariant, which is what lets the model separate
        posture from pathology instead of merely discarding one of them.
    equivariance_leak:
        if non-zero, a parallel non-equivariant path is mixed in with this initial
        weight and the weight is learned.  This is the approximate-equivariance
        fallback of H4: when the dipole model fails, exact invariance is too strong a
        constraint and the learned leak measures how much symmetry must be broken.
    naive_lift:
        use :class:`NaiveOrthogonalLift`, i.e. the incorrect orthogonality assumption
        made by off-the-shelf steerable architectures.  For ablation only.
    """

    def __init__(
        self,
        n_classes: int = 5,
        width: int = 48,
        vector_width: int = 24,
        depth: int = 4,
        kernel: int = 9,
        strides: tuple[int, ...] = (4, 2, 2, 2),
        predict_pose: bool = False,
        pose_in_head: bool = False,
        equivariance_leak: float = 0.0,
        naive_lift: bool = False,
        geo: LeadGeometry = GEOMETRY,
    ):
        super().__init__()
        self.lift = NaiveOrthogonalLift(geo) if naive_lift else GaugeLift(geo)
        self.predict_pose = predict_pose
        self.pose_in_head = pose_in_head
        self.equivariance_leak = equivariance_leak

        if len(strides) < depth:
            strides = tuple(strides) + (2,) * (depth - len(strides))
        blocks = []
        cs, cv = 5, 1
        for i in range(depth):
            cs_out = width if i else width // 2
            cv_out = vector_width if i else vector_width // 2
            blocks.append(
                GVPBlock(cs, cv, cs_out, cv_out, kernel=kernel, stride=strides[i])
            )
            cs, cv = cs_out, cv_out
        self.blocks = nn.ModuleList(blocks)
        self.cs, self.cv = cs, cv

        n_gram = cv * (cv + 1)  # two Gram contractions, upper triangles
        feat_dim = 3 * cs + cv + n_gram          # mean/max/std scalars, norms, Gram
        if pose_in_head:
            # Theorem 3 says the representation splits into an invariant part and an
            # equivariant part.  Labels that are *defined* by orientation (axis
            # deviation) live in the second, so exposing the time-pooled vector
            # features to the head is not a hack -- it is the other half of the
            # decomposition.  Invariance is then broken only through this named
            # channel, and only when the task requires it.
            feat_dim += 3 * cv
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.SiLU(), nn.Dropout(0.1), nn.Linear(128, n_classes)
        )
        if predict_pose:
            # Two equivariant vectors, orthonormalised into a rotation matrix.
            self.pose = nn.Conv1d(cv, 2, 1, bias=False)
        if equivariance_leak > 0:
            self.leak_path = nn.Sequential(
                nn.Conv1d(12, width, kernel, 4, kernel // 2), nn.SiLU(),
                nn.Conv1d(width, width, kernel, 4, kernel // 2), nn.SiLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width, n_classes),
            )
            self.leak_weight = nn.Parameter(torch.tensor(float(equivariance_leak)))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v, s = self.lift(x)
        for block in self.blocks:
            v, s = block(v, s)
        return v, s

    def invariant_features(self, x: torch.Tensor) -> torch.Tensor:
        v, s = self.encode(x)
        norms = torch.linalg.norm(v, dim=2)
        feats = [s.mean(-1), s.amax(-1), s.std(-1), norms.mean(-1), gram_features(v)]
        return torch.cat(feats, dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        v, s = self.encode(x)
        norms = torch.linalg.norm(v, dim=2)
        parts = [s.mean(-1), s.amax(-1), s.std(-1), norms.mean(-1), gram_features(v)]
        if self.pose_in_head:
            parts.append(v.mean(-1).flatten(1))
        feats = torch.cat(parts, dim=1)
        logits = self.head(feats)
        if self.equivariance_leak > 0:
            logits = logits + self.leak_weight * self.leak_path(x)
        out = {"logits": logits, "features": feats}
        if self.predict_pose:
            # (B, cv, 3): mix over the channel axis with the vector axis as 'length',
            # so the same weights act on all three components and the head stays equivariant.
            frame = self.pose(v.mean(-1))  # (B, 2, 3)
            out["rotation"] = _gram_schmidt(frame)
        return out


def _gram_schmidt(frame: torch.Tensor) -> torch.Tensor:
    """Two equivariant vectors -> a rotation matrix (the 6-D rotation parameterisation).

    Equivariance is preserved: rotating both inputs by ``R`` rotates the resulting
    frame by ``R``, so the head predicts pose in the correct covariant way.
    """
    a, b = frame[:, 0], frame[:, 1]
    e1 = F.normalize(a, dim=-1, eps=1e-8)
    b = b - (e1 * b).sum(-1, keepdim=True) * e1
    e2 = F.normalize(b, dim=-1, eps=1e-8)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)
