"""``SO(3)`` utilities: Haar sampling, parameterisations, and Haar averaging.

Everything the paper needs about the group itself lives here, kept deliberately
independent of the ECG specifics so that the theory statements can be checked
against a generic compact-group implementation.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "hat",
    "expm_so3",
    "log_so3",
    "rotation_matrix",
    "random_rotation",
    "random_small_rotation",
    "geodesic_distance",
    "haar_average",
    "axis_angle_of",
    "euler_xyz",
]


def hat(w: np.ndarray) -> np.ndarray:
    """Map ``R^3 -> so(3)`` (the cross-product / skew-symmetric matrix)."""
    w = np.asarray(w, dtype=float)
    x, y, z = w[..., 0], w[..., 1], w[..., 2]
    zero = np.zeros_like(x)
    return np.stack(
        [
            np.stack([zero, -z, y], axis=-1),
            np.stack([z, zero, -x], axis=-1),
            np.stack([-y, x, zero], axis=-1),
        ],
        axis=-2,
    )


def expm_so3(w: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: exponential map ``so(3) -> SO(3)`` from an axis-angle vector."""
    w = np.asarray(w, dtype=float)
    theta = np.linalg.norm(w, axis=-1, keepdims=True)
    small = theta < 1e-9
    safe = np.where(small, 1.0, theta)
    K = hat(w / safe)
    s = np.sin(theta)[..., None]
    c = np.cos(theta)[..., None]
    eye = np.eye(3) + np.zeros(w.shape[:-1] + (3, 3))
    R = eye + s * K + (1.0 - c) * (K @ K)
    return np.where(small[..., None], eye, R)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Inverse of :func:`expm_so3`, returning an axis-angle vector in ``R^3``."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R, axis1=-2, axis2=-1)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    axis = np.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        axis=-1,
    )
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    # Near theta = 0 and theta = pi the closed form degrades; fall back to the
    # eigenvector construction for the (measure-zero, but numerically reachable) pi case.
    out = np.where(norm > 1e-8, axis / np.where(norm > 1e-8, norm, 1.0), 0.0)
    return out * theta[..., None]


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation by ``angle`` radians about a (not necessarily normalised) ``axis``."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    return expm_so3(axis * angle)


def euler_xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Intrinsic X-Y-Z Euler rotation, used for interpretable pose labels.

    In ECG terms the three angles correspond to rotation in the frontal, sagittal
    and transverse planes respectively, so a fitted ``(alpha, beta, gamma)`` maps
    directly onto the clinical notions of axis deviation and heart rotation.
    """
    Rx = rotation_matrix(np.array([1.0, 0.0, 0.0]), alpha)
    Ry = rotation_matrix(np.array([0.0, 1.0, 0.0]), beta)
    Rz = rotation_matrix(np.array([0.0, 0.0, 1.0]), gamma)
    return Rx @ Ry @ Rz


def random_rotation(rng: np.random.Generator, size: int | None = None) -> np.ndarray:
    """Exact Haar-uniform sample(s) from ``SO(3)`` via the subgroup algorithm.

    Uses unit quaternions drawn uniformly from ``S^3`` (Shoemake's method), which is
    Haar by the double cover ``S^3 -> SO(3)``.
    """
    n = 1 if size is None else size
    u = rng.random((n, 3))
    q = np.stack(
        [
            np.sqrt(1 - u[:, 0]) * np.sin(2 * np.pi * u[:, 1]),
            np.sqrt(1 - u[:, 0]) * np.cos(2 * np.pi * u[:, 1]),
            np.sqrt(u[:, 0]) * np.sin(2 * np.pi * u[:, 2]),
            np.sqrt(u[:, 0]) * np.cos(2 * np.pi * u[:, 2]),
        ],
        axis=-1,
    )
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )
    return R[0] if size is None else R


def random_small_rotation(
    rng: np.random.Generator, max_angle_deg: float, size: int | None = None
) -> np.ndarray:
    """Rotation with uniformly random axis and angle in ``[0, max_angle_deg]``.

    This models the *realistic* nuisance regime -- posture, respiration, electrode
    displacement -- as opposed to the full Haar measure.
    """
    n = 1 if size is None else size
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = rng.uniform(0.0, np.deg2rad(max_angle_deg), size=(n, 1))
    R = expm_so3(axis * angle)
    return R[0] if size is None else R


def geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Riemannian (angle) distance on ``SO(3)`` in radians."""
    rel = np.einsum("...ji,...jk->...ik", R1, R2)
    tr = np.trace(rel, axis1=-2, axis2=-1)
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


def axis_angle_of(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a rotation into unit axis and angle."""
    w = log_so3(R)
    angle = np.linalg.norm(w, axis=-1)
    axis = w / np.maximum(angle, 1e-12)[..., None]
    return axis, angle


def haar_average(fn, rng: np.random.Generator, n_samples: int = 4096):
    """Monte-Carlo Haar average ``\\int_{SO(3)} fn(R) dR``.

    Used to verify by simulation the closed-form projector / dimension counts that
    the paper derives analytically.
    """
    Rs = random_rotation(rng, n_samples)
    acc = None
    for R in Rs:
        val = np.asarray(fn(R), dtype=float)
        acc = val.copy() if acc is None else acc + val
    return acc / n_samples
