"""Lead geometry of the 12-lead ECG and the group action induced by cardiac-axis rotation.

The 12-lead ECG is *not* a generic 12-dimensional signal.  Two exact algebraic facts
constrain it, and this module makes both explicit and machine-checkable.

1.  **Exact linear redundancy (rank 8).**  By the definition of the augmented and
    bipolar limb leads,

        III  = II - I,        aVR = -(I + II)/2,
        aVL  = (I - III)/2,   aVF = (II + III)/2,

    so every physically realisable 12-lead vector lies in an 8-dimensional subspace
    ``V8 = range(B) <= R^12`` spanned by the independent leads ``(I, II, V1..V6)``.

2.  **The dipole (VCG) model, and the group it induces.**  Under the equivalent
    cardiac-dipole model the body-surface potentials are a fixed linear image of a
    3-dimensional heart vector ``v(t) in R^3``::

        x(t) = D v(t),        D in R^{12x3},  rank(D) = 3.

    Changes of patient posture, respiration, body habitus and electrode placement
    act, to first order, as a rotation ``R in SO(3)`` of the heart vector.  The
    action *induced on the observed lead space* is therefore

        x  |->  rho(R) x = D R D^+ x.

``rho`` is a genuine representation of ``SO(3)`` on the dipolar subspace
``Vdip = range(D)``, but -- and this is the point of the whole paper -- it is **not
orthogonal** with respect to the Euclidean inner product of lead space, because
``D^T D`` is not a multiple of the identity for any clinically used transform
matrix.  Off-the-shelf steerable/equivariant architectures assume an orthogonal
(unitary) action and therefore do not apply verbatim.

The remedy implemented here (Theorem 1 of the paper) is a fixed *gauge*: the
pullback metric ``M = D (D^T D)^{-2} D^T`` makes the action orthogonal, and the
whitening map ``Phi = D^+`` intertwines ``rho`` with the standard representation
of ``SO(3)`` on ``R^3``.

Physical 12-lead space then decomposes, as an ``SO(3)``-representation, into

        V8  =  Vdip  (+)  Vres         ~=   l=1   (+)   5 x (l=0),

with ``Vres`` the 5-dimensional *non-dipolar* residual that the rotation leaves
fixed.  The extended action ``rho_ext(R) = D R D^+ + Pi_res`` is the physically
correct one on real (rank-8, only approximately rank-3) recordings, and the gap
between ``rho`` and ``rho_ext`` is exactly what hypothesis H4 probes.

All transform matrices are treated as *parameters*, not constants: the theory is
stated for an arbitrary full-rank ``D``, so a different choice of transform
(Dower, Kors, a subject-specific regression matrix) changes the numbers but not a
single statement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "LEAD_NAMES",
    "INDEPENDENT_LEADS",
    "reconstruction_matrix",
    "lead_identity_matrix",
    "dower_forward",
    "kors_forward",
    "LeadGeometry",
    "GEOMETRY",
]

# Canonical clinical ordering used throughout the code base.
LEAD_NAMES: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)
INDEPENDENT_LEADS: tuple[str, ...] = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")

_LEAD_INDEX = {name: i for i, name in enumerate(LEAD_NAMES)}


def reconstruction_matrix() -> np.ndarray:
    """``B in R^{12x8}`` mapping the independent leads to the full 12-lead vector.

    Columns are ordered as :data:`INDEPENDENT_LEADS`.  ``B`` encodes exactly the
    four limb-lead identities, so ``range(B)`` is the space of physically
    realisable 12-lead vectors.
    """
    B = np.zeros((12, 8))
    col = {name: j for j, name in enumerate(INDEPENDENT_LEADS)}

    B[_LEAD_INDEX["I"], col["I"]] = 1.0
    B[_LEAD_INDEX["II"], col["II"]] = 1.0
    # III = II - I
    B[_LEAD_INDEX["III"], col["II"]] = 1.0
    B[_LEAD_INDEX["III"], col["I"]] = -1.0
    # aVR = -(I + II)/2
    B[_LEAD_INDEX["aVR"], col["I"]] = -0.5
    B[_LEAD_INDEX["aVR"], col["II"]] = -0.5
    # aVL = (I - III)/2 = I - II/2
    B[_LEAD_INDEX["aVL"], col["I"]] = 1.0
    B[_LEAD_INDEX["aVL"], col["II"]] = -0.5
    # aVF = (II + III)/2 = II - I/2
    B[_LEAD_INDEX["aVF"], col["II"]] = 1.0
    B[_LEAD_INDEX["aVF"], col["I"]] = -0.5

    for name in ("V1", "V2", "V3", "V4", "V5", "V6"):
        B[_LEAD_INDEX[name], col[name]] = 1.0
    return B


def lead_identity_matrix() -> np.ndarray:
    """``A in R^{4x12}`` with ``A x = 0`` for every realisable 12-lead vector ``x``."""
    A = np.zeros((4, 12))
    # III - II + I = 0
    A[0, _LEAD_INDEX["III"]] = 1.0
    A[0, _LEAD_INDEX["II"]] = -1.0
    A[0, _LEAD_INDEX["I"]] = 1.0
    # aVR + (I + II)/2 = 0
    A[1, _LEAD_INDEX["aVR"]] = 1.0
    A[1, _LEAD_INDEX["I"]] = 0.5
    A[1, _LEAD_INDEX["II"]] = 0.5
    # aVL - (I - III)/2 = 0
    A[2, _LEAD_INDEX["aVL"]] = 1.0
    A[2, _LEAD_INDEX["I"]] = -0.5
    A[2, _LEAD_INDEX["III"]] = 0.5
    # aVF - (II + III)/2 = 0
    A[3, _LEAD_INDEX["aVF"]] = 1.0
    A[3, _LEAD_INDEX["II"]] = -0.5
    A[3, _LEAD_INDEX["III"]] = -0.5
    return A


# --------------------------------------------------------------------------------------
# Transform matrices.  Rows follow INDEPENDENT_LEADS ordering; the full 12x3 matrix is
# obtained by propagating through B so that the limb-lead identities hold exactly.
# --------------------------------------------------------------------------------------

# Dower's forward transform (Dower, Machado & Osborne, 1980): XYZ -> 8 independent leads.
_DOWER_8x3 = np.array(
    [
        [0.632, -0.235, 0.059],   # I
        [0.235, 1.066, -0.132],   # II
        [-0.515, 0.157, -0.917],  # V1
        [0.044, 0.164, -1.387],   # V2
        [0.882, 0.098, -1.277],   # V3
        [1.213, 0.127, -0.601],   # V4
        [1.125, 0.127, -0.086],   # V5
        [0.831, 0.076, 0.230],    # V6
    ]
)

# Kors' regression transform (Kors et al., 1990) is published as an *inverse* map
# (8 leads -> XYZ); the forward map used here is its pseudo-inverse.
_KORS_INVERSE_3x8 = np.array(
    [
        [0.38, -0.07, -0.13, 0.05, -0.01, 0.14, 0.06, 0.54],
        [-0.07, 0.93, 0.06, -0.02, -0.05, 0.06, -0.17, 0.13],
        [0.11, -0.23, -0.43, -0.06, -0.14, -0.20, -0.11, 0.31],
    ]
)


def _lift_to_12(mat_8x3: np.ndarray) -> np.ndarray:
    """Lift an 8x3 transform on the independent leads to the full 12x3 transform."""
    return reconstruction_matrix() @ mat_8x3


def dower_forward() -> np.ndarray:
    """Dower forward transform ``D in R^{12x3}`` (VCG -> 12-lead)."""
    return _lift_to_12(_DOWER_8x3)


def kors_forward() -> np.ndarray:
    """Kors regression transform, expressed as a forward map ``D in R^{12x3}``."""
    return _lift_to_12(np.linalg.pinv(_KORS_INVERSE_3x8))


@dataclass(frozen=True)
class LeadGeometry:
    """All linear-algebraic objects attached to a choice of transform matrix ``D``.

    Attributes
    ----------
    D:
        ``12x3`` forward transform, VCG -> 12-lead.
    B:
        ``12x8`` reconstruction matrix; ``range(B)`` is the realisable lead space.
    D_pinv:
        ``3x12`` Moore-Penrose pseudo-inverse of ``D``; the *gauge map* that sends an
        observed lead vector to its VCG coordinates.
    P_dip, P_res, P8:
        Euclidean-orthogonal projectors onto the dipolar subspace, the non-dipolar
        residual subspace and the realisable subspace respectively.
    M:
        ``12x12`` PSD matrix, the pullback of the Euclidean VCG metric; makes ``rho``
        orthogonal on ``Vdip``.
    M_ext:
        ``M + P_res``; makes the physically extended action orthogonal on ``V8``.
    """

    name: str
    D: np.ndarray
    B: np.ndarray = field(default_factory=reconstruction_matrix)

    # ---- derived quantities -------------------------------------------------------
    @property
    def D_pinv(self) -> np.ndarray:
        return np.linalg.pinv(self.D)

    @property
    def P_dip(self) -> np.ndarray:
        return self.D @ self.D_pinv

    @property
    def P8(self) -> np.ndarray:
        return self.B @ np.linalg.pinv(self.B)

    @property
    def P_res(self) -> np.ndarray:
        return self.P8 - self.P_dip

    @property
    def M(self) -> np.ndarray:
        """Pullback metric ``M = D (D^T D)^{-2} D^T`` (PSD, rank 3)."""
        G = self.D.T @ self.D
        Ginv = np.linalg.inv(G)
        return self.D @ Ginv @ Ginv @ self.D.T

    @property
    def M_ext(self) -> np.ndarray:
        """Metric making the *extended* action orthogonal on the realisable subspace."""
        return self.M + self.P_res

    @property
    def residual_basis(self) -> np.ndarray:
        """``C in R^{5x12}``: an orthonormal basis of the non-dipolar residual space.

        Rows are Euclidean-orthonormal and span ``V8 (-) Vdip``.  ``C x`` are the five
        *invariant* (l=0) coordinates of a lead vector.
        """
        U, s, _ = np.linalg.svd(self.P_res)
        rank = int(np.sum(s > 1e-9))
        return U[:, :rank].T

    @property
    def dipole_basis(self) -> np.ndarray:
        """``3x12`` map to whitened VCG coordinates; identical to :attr:`D_pinv`."""
        return self.D_pinv

    # ---- group action -------------------------------------------------------------
    def rho(self, R: np.ndarray) -> np.ndarray:
        """Induced action on lead space, strict dipole model: ``D R D^+``.

        Supports a single ``3x3`` rotation or a batch ``(..., 3, 3)``.
        """
        R = np.asarray(R)
        return self.D @ R @ self.D_pinv if R.ndim == 2 else np.einsum(
            "ij,...jk,kl->...il", self.D, R, self.D_pinv
        )

    def rho_ext(self, R: np.ndarray) -> np.ndarray:
        """Physically extended action: rotate the dipolar part, fix the residual."""
        return self.rho(R) + self.P_res

    # ---- diagnostics --------------------------------------------------------------
    @property
    def singular_values(self) -> np.ndarray:
        return np.linalg.svd(self.D, compute_uv=False)

    @property
    def anisotropy(self) -> float:
        """``kappa(D) = sigma_max / sigma_min``.

        Equals 1 exactly when the action is already Euclidean-orthogonal (up to
        scale); the amount by which it exceeds 1 quantifies how badly the standard
        steerable assumption is violated, and is the constant that appears in the
        generalisation bound of Theorem 2.
        """
        s = self.singular_values
        return float(s[0] / s[-1])

    def orthogonality_defect(self, R: np.ndarray) -> float:
        """``||rho(R)^T rho(R) - P_dip||_F`` -- zero iff the action is orthogonal."""
        A = self.rho(R)
        return float(np.linalg.norm(A.T @ A - self.P_dip))

    def invariance_defect(self, R: np.ndarray) -> float:
        """``||rho(R)^T M rho(R) - M||_F`` -- zero by Theorem 1, up to round-off."""
        A = self.rho(R)
        return float(np.linalg.norm(A.T @ self.M @ A - self.M))

    def to_vcg(self, x: np.ndarray) -> np.ndarray:
        """Lead-space signal ``(..., 12, T)`` -> VCG ``(..., 3, T)``."""
        return np.einsum("ij,...jt->...it", self.D_pinv, x)

    def to_leads(self, v: np.ndarray) -> np.ndarray:
        """VCG ``(..., 3, T)`` -> lead-space signal ``(..., 12, T)``."""
        return np.einsum("ij,...jt->...it", self.D, v)

    def residual(self, x: np.ndarray) -> np.ndarray:
        """Non-dipolar residual coordinates ``(..., 5, T)`` of a lead signal."""
        return np.einsum("ij,...jt->...it", self.residual_basis, x)


def geometry(name: str = "dower") -> LeadGeometry:
    """Build a :class:`LeadGeometry` for a named transform (``dower`` or ``kors``)."""
    table = {"dower": dower_forward, "kors": kors_forward}
    if name not in table:
        raise KeyError(f"unknown transform {name!r}; choose from {sorted(table)}")
    return LeadGeometry(name=name, D=table[name]())


GEOMETRY = geometry("dower")
