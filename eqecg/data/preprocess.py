"""Shared preprocessing so that every corpus lands in one canonical format.

Canonical record: ``float32`` array of shape ``(12, 1000)`` -- twelve leads in the
order of :data:`eqecg.leads.LEAD_NAMES`, ten seconds at 100 Hz, in millivolts.

Two steps deserve comment because they interact with the theory.

``reorder_leads``
    Corpora disagree on lead naming (``DI``/``I``, ``AVR``/``aVR``).  Getting this
    wrong would silently permute the lead axis and invalidate every statement about
    the transform matrix, so the mapping is explicit and validated rather than
    positional.

``project_to_realisable``
    Real recordings are quantised, filtered and occasionally re-derived, so the four
    limb-lead identities hold only approximately.  Projecting onto ``range(B)``
    enforces them exactly, which is what makes the rank-8 decomposition -- and hence
    the clean split into dipolar and non-dipolar coordinates -- well defined.  The
    size of the correction is returned so it can be reported rather than hidden.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from eqecg.leads import GEOMETRY, LEAD_NAMES, LeadGeometry

__all__ = [
    "TARGET_FS",
    "TARGET_LENGTH",
    "ALIASES",
    "reorder_leads",
    "resample_to",
    "bandpass",
    "project_to_realisable",
    "standardise_record",
]

TARGET_FS = 100
TARGET_LENGTH = 1000

#: Normalised spellings of lead names seen across corpora.
ALIASES: dict[str, str] = {
    "i": "I", "di": "I", "lead i": "I",
    "ii": "II", "dii": "II", "lead ii": "II",
    "iii": "III", "diii": "III", "lead iii": "III",
    "avr": "aVR", "avl": "aVL", "avf": "aVF",
    **{f"v{i}": f"V{i}" for i in range(1, 7)},
}


def canonical_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    if key in ALIASES:
        return ALIASES[key]
    raise KeyError(f"unrecognised lead name {name!r}")


def reorder_leads(x: np.ndarray, names: list[str]) -> np.ndarray:
    """Reorder ``(n_leads, T)`` (or ``(N, n_leads, T)``) into the canonical order."""
    canon = [canonical_name(n) for n in names]
    missing = set(LEAD_NAMES) - set(canon)
    if missing:
        raise ValueError(f"missing leads: {sorted(missing)}")
    order = [canon.index(n) for n in LEAD_NAMES]
    return x[..., order, :]


def resample_to(x: np.ndarray, fs_in: int, fs_out: int = TARGET_FS) -> np.ndarray:
    """Polyphase resampling along the last axis."""
    if fs_in == fs_out:
        return x
    from math import gcd

    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(x, int(fs_out // g), int(fs_in // g), axis=-1)


def bandpass(x: np.ndarray, fs: int = TARGET_FS, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """Zero-phase 0.5-40 Hz bandpass, the standard diagnostic ECG band.

    ``high`` is clipped below Nyquist so the same call is valid at any sample rate.
    """
    high = min(high, 0.45 * fs)
    sos = butter(3, [low / (fs / 2), high / (fs / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def project_to_realisable(
    x: np.ndarray, geo: LeadGeometry = GEOMETRY
) -> tuple[np.ndarray, float]:
    """Project onto the rank-8 realisable subspace; also return the relative correction."""
    proj = np.einsum("ij,...jt->...it", geo.P8, x)
    denom = np.linalg.norm(x) + 1e-12
    return proj.astype(np.float32), float(np.linalg.norm(proj - x) / denom)


def standardise_record(
    x: np.ndarray,
    fs: int,
    names: list[str] | None = None,
    do_bandpass: bool = True,
    geo: LeadGeometry = GEOMETRY,
) -> tuple[np.ndarray, float]:
    """Full pipeline: reorder -> resample -> bandpass -> crop/pad -> project.

    Returns the canonical ``(12, 1000)`` record and the relative size of the
    lead-identity correction.
    """
    if names is not None:
        x = reorder_leads(x, names)
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    x = resample_to(x, fs, TARGET_FS)
    if do_bandpass:
        x = bandpass(x, TARGET_FS)
    if x.shape[-1] >= TARGET_LENGTH:
        start = (x.shape[-1] - TARGET_LENGTH) // 2
        x = x[..., start : start + TARGET_LENGTH]
    else:
        pad = TARGET_LENGTH - x.shape[-1]
        x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
    return project_to_realisable(x, geo)
