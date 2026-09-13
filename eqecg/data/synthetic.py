"""A dipole-model ECG simulator with ground-truth pose, pathology and dipole-breakdown.

Why simulate at all when large public ECG corpora exist?  Because three of the
paper's claims are *unfalsifiable on real data alone*:

* Theorem 3 and hypothesis H2 need the true nuisance rotation ``R``, which is never
  recorded in a clinical corpus.
* H4 asks what happens as the dipole approximation degrades; that requires a knob
  controlling the non-dipolar energy fraction, which nature does not expose.
* Separating "the model is invariant" from "the label happened to be invariant"
  requires labels whose dependence on pose is known by construction.

The generator is the 3-D counterpart of the standard Gaussian-mixture ECG model
(McSharry et al.): each of the P, QRS and T complexes is a sum of Gaussian bumps in
time, each carrying a fixed direction in heart-vector space, so the heart vector is

    v(t) = sum_k a_k u_k exp( -(t - mu_k)^2 / (2 s_k^2) ),   u_k in S^2.

Passing ``v`` through the transform matrix yields a 12-lead signal that satisfies
the limb-lead identities exactly and has realistic morphology.  Departures from the
dipole model are injected *directly in the five-dimensional non-dipolar subspace*
``Vres``, which is the only physically meaningful place for them to live.

Label design deserves a note, because it is where a careless benchmark would beg
the question.  Four of the five classes are defined by genuinely rotation-invariant
quantities (durations, amplitude ratios, the QRS-T angle); the fifth, ``AXIS``, is
defined *by* the heart-vector orientation and is therefore provably unlearnable by a
strictly invariant model.  Keeping it in the benchmark is deliberate: it is the
control that shows invariance is a real constraint and not a free lunch, and it is
the concrete setting in which the approximate-equivariance fallback of H4 is
measured rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eqecg.group import expm_so3, random_rotation, random_small_rotation
from eqecg.leads import GEOMETRY, LeadGeometry

__all__ = [
    "CLASS_NAMES",
    "SimulatorConfig",
    "DipoleECGSimulator",
    "lead_reversal_matrix",
    "electrode_displacement_transform",
]

# Four pose-invariant classes plus one deliberately pose-defined control class.
CLASS_NAMES: tuple[str, ...] = ("NORM", "CD", "MI", "HYP", "AXIS")
POSE_INVARIANT_CLASSES: tuple[str, ...] = ("NORM", "CD", "MI", "HYP")


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@dataclass
class SimulatorConfig:
    """Knobs of the generator.  Defaults match the PTB-XL low-rate protocol."""

    fs: int = 100
    duration: float = 10.0
    heart_rate_mean: float = 70.0
    heart_rate_std: float = 12.0
    rr_jitter: float = 0.04
    #: standard deviation of the per-subject direction jitter, radians
    direction_jitter: float = 0.25
    #: log-normal spread of per-component amplitude and width across subjects
    amplitude_jitter: float = 0.30
    width_jitter: float = 0.20
    #: scales every class-conditional effect towards "no effect"; 1.0 reproduces the
    #: textbook-sized abnormalities, smaller values make classes overlap with normal
    #: anatomical variation, which is the regime where inductive bias actually matters
    class_effect_scale: float = 0.35
    #: energy fraction of the non-dipolar residual, relative to the dipolar signal
    nondipolar_fraction: float = 0.0
    #: nuisance rotation regime: "none", "small" (posture-like) or "haar"
    rotation: str = "small"
    max_rotation_deg: float = 30.0
    #: additive noise, as a fraction of the signal's RMS
    baseline_wander: float = 0.12
    muscle_noise: float = 0.10
    powerline_noise: float = 0.01
    powerline_hz: float = 50.0

    @property
    def n_samples(self) -> int:
        return int(round(self.fs * self.duration))


@dataclass
class _Bump:
    """One Gaussian component of the heart vector."""

    offset: float      # seconds relative to the R peak
    width: float       # seconds
    amplitude: float   # mV
    direction: np.ndarray  # unit vector in heart-vector space


# Base morphology, loosely following normal VCG loop orientations expressed in the
# Dower XYZ frame (x: left, y: inferior, z: posterior).
def _base_morphology(rng: np.random.Generator) -> list[_Bump]:
    return [
        # Offsets/widths give a ~100 ms QRS, ~180 ms PR and ~380 ms QT at rest,
        # i.e. intervals inside the normal adult range.
        _Bump(-0.180, 0.028, 0.13, _unit(np.array([0.55, 0.75, -0.35]))),  # P loop
        _Bump(-0.030, 0.010, 0.18, _unit(np.array([-0.65, 0.15, -0.74]))),  # septal q
        _Bump(0.000, 0.028, 1.35, _unit(np.array([0.72, 0.60, 0.35]))),    # R (main QRS)
        _Bump(0.038, 0.014, 0.28, _unit(np.array([-0.20, 0.35, 0.91]))),   # S terminal
        _Bump(0.240, 0.055, 0.33, _unit(np.array([0.62, 0.68, 0.10]))),    # T loop
    ]


class DipoleECGSimulator:
    """Generates 12-lead ECG together with every latent variable the paper needs."""

    def __init__(self, config: SimulatorConfig | None = None, geo: LeadGeometry = GEOMETRY):
        self.cfg = config or SimulatorConfig()
        self.geo = geo

    # -- class-conditional morphology ----------------------------------------------
    def _apply_class(self, bumps: list[_Bump], label: int, rng: np.random.Generator) -> list[_Bump]:
        """Apply the class-conditional morphology change.

        Every effect is shrunk towards "no effect" by ``class_effect_scale``, so the
        abnormal and normal populations overlap the way they do clinically.  Without
        this the benchmark saturates at a hundred training examples and can no longer
        distinguish inductive biases, which is the only thing it exists to measure.
        """
        name = CLASS_NAMES[label]
        k = self.cfg.class_effect_scale

        def mul(lo: float, hi: float) -> float:
            """Multiplicative effect, interpolated towards 1."""
            return 1.0 + (rng.uniform(lo, hi) - 1.0) * k

        out = [_Bump(b.offset, b.width, b.amplitude, b.direction.copy()) for b in bumps]
        if name == "CD":
            for b in out[1:4]:
                b.width *= mul(1.8, 2.6)
            out[3].amplitude *= mul(1.6, 2.4)
            out[3].offset += rng.uniform(0.012, 0.030) * k
        elif name == "MI":
            out[1].amplitude *= mul(2.5, 4.0)
            out[1].width *= mul(1.4, 2.0)
            out[2].amplitude *= mul(0.45, 0.7)
            out[4].amplitude *= 1.0 + (rng.uniform(-0.9, -0.4) - 1.0) * k
        elif name == "HYP":
            out[2].amplitude *= mul(1.5, 2.1)
            axis = _unit(rng.normal(size=3))
            angle = rng.uniform(np.deg2rad(80), np.deg2rad(140)) * k
            out[4].direction = expm_so3(axis * angle) @ out[2].direction
            out[4].amplitude *= mul(0.8, 1.3)
        elif name == "AXIS":
            axis = _unit(rng.normal(size=3))
            angle = rng.uniform(np.deg2rad(45), np.deg2rad(90))
            Rd = expm_so3(axis * angle)
            for b in out:
                b.direction = Rd @ b.direction
        return out

    # -- single record --------------------------------------------------------------
    def _heart_vector(self, bumps: list[_Bump], rng: np.random.Generator) -> np.ndarray:
        cfg = self.cfg
        T = cfg.n_samples
        t = np.arange(T) / cfg.fs
        hr = float(np.clip(rng.normal(cfg.heart_rate_mean, cfg.heart_rate_std), 40.0, 150.0))
        rr = 60.0 / hr

        peaks = []
        cur = rng.uniform(0.2, 0.2 + rr)
        while cur < cfg.duration + rr:
            peaks.append(cur)
            cur += rr * (1.0 + rng.normal(0.0, cfg.rr_jitter))

        v = np.zeros((3, T))
        for peak in peaks:
            # Beat-to-beat amplitude modulation (respiration, autonomic tone).
            gain = 1.0 + rng.normal(0.0, 0.04)
            for b in bumps:
                centre = peak + b.offset
                if centre < -0.5 or centre > cfg.duration + 0.5:
                    continue
                env = np.exp(-((t - centre) ** 2) / (2.0 * b.width**2))
                v += (gain * b.amplitude) * b.direction[:, None] * env[None, :]
        return v

    def _noise(self, rng: np.random.Generator, scale: float) -> np.ndarray:
        cfg = self.cfg
        T = cfg.n_samples
        t = np.arange(T) / cfg.fs
        out = np.zeros((12, T))
        if cfg.baseline_wander > 0:
            for _ in range(3):
                f = rng.uniform(0.05, 0.5)
                phase = rng.uniform(0, 2 * np.pi, size=(12, 1))
                amp = rng.uniform(0.3, 1.0, size=(12, 1))
                out += amp * np.sin(2 * np.pi * f * t[None, :] + phase)
            out *= cfg.baseline_wander * scale / max(np.std(out), 1e-9)
        if cfg.muscle_noise > 0:
            emg = rng.normal(size=(12, T))
            # Crude high-pass to put the energy in the EMG band.
            emg = np.diff(emg, axis=1, prepend=emg[:, :1])
            out += cfg.muscle_noise * scale * emg / max(np.std(emg), 1e-9)
        if cfg.powerline_noise > 0:
            phase = rng.uniform(0, 2 * np.pi, size=(12, 1))
            out += cfg.powerline_noise * scale * np.sin(
                2 * np.pi * cfg.powerline_hz * t[None, :] + phase
            )
        return out

    def sample(self, n: int, rng: np.random.Generator, labels: np.ndarray | None = None) -> dict:
        """Generate ``n`` records with all latent variables attached.

        Returns a dict with
        ``x`` ``(n, 12, T)`` observed leads, ``v_true`` ``(n, 3, T)`` the *un-rotated*
        heart vector, ``R`` ``(n, 3, 3)`` the applied nuisance rotation, ``y`` ``(n,)``
        the class index, ``axis_deg`` the frontal QRS axis, and ``rotvec`` the
        axis-angle encoding of ``R`` (the regression target of the H2 probe).
        """
        cfg = self.cfg
        geo = self.geo
        T = cfg.n_samples
        if labels is None:
            labels = rng.integers(0, len(CLASS_NAMES), size=n)
        labels = np.asarray(labels)

        if cfg.rotation == "none":
            R = np.tile(np.eye(3), (n, 1, 1))
        elif cfg.rotation == "haar":
            R = random_rotation(rng, n)
        elif cfg.rotation == "small":
            R = random_small_rotation(rng, cfg.max_rotation_deg, n)
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown rotation regime {cfg.rotation!r}")

        x = np.empty((n, 12, T))
        v_true = np.empty((n, 3, T))
        for i in range(n):
            bumps = _base_morphology(rng)
            # Per-subject anatomical variation, applied before the class effect.
            for b in bumps:
                b.direction = _unit(
                    b.direction + rng.normal(0.0, cfg.direction_jitter, size=3)
                )
                b.amplitude *= np.exp(rng.normal(0.0, cfg.amplitude_jitter))
                b.width *= np.exp(rng.normal(0.0, cfg.width_jitter))
            bumps = self._apply_class(bumps, int(labels[i]), rng)

            v = self._heart_vector(bumps, rng)
            v_true[i] = v
            signal = geo.to_leads(R[i] @ v)

            if cfg.nondipolar_fraction > 0:
                # Non-dipolar content lives in Vres and is *not* moved by the rotation,
                # which is exactly why it breaks strict equivariance (H4).
                C = geo.residual_basis                      # 5 x 12
                coeff = rng.normal(size=(5, T))
                # Smooth it so it occupies the ECG band rather than looking like noise.
                kernel = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2)
                kernel /= kernel.sum()
                coeff = np.apply_along_axis(
                    lambda r: np.convolve(r, kernel, mode="same"), 1, coeff
                )
                resid = C.T @ coeff
                resid *= (
                    cfg.nondipolar_fraction
                    * np.linalg.norm(signal)
                    / max(np.linalg.norm(resid), 1e-9)
                )
                signal = signal + resid

            signal = signal + self._noise(rng, float(np.std(signal)))
            # Re-impose the limb-lead identities, which noise would otherwise violate.
            x[i] = geo.P8 @ signal

        from eqecg.group import log_so3

        return {
            "x": x.astype(np.float32),
            "v_true": v_true.astype(np.float32),
            "R": R.astype(np.float32),
            "rotvec": log_so3(R).astype(np.float32),
            "y": labels.astype(np.int64),
            "axis_deg": _frontal_axis(v_true, R).astype(np.float32),
            "class_names": CLASS_NAMES,
        }


def _frontal_axis(v_true: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Frontal-plane QRS axis in degrees, computed from the rotated heart vector."""
    v = np.einsum("nij,njt->nit", R, v_true)
    peak = np.argmax(np.linalg.norm(v, axis=1), axis=1)
    main = v[np.arange(v.shape[0]), :, peak]
    return np.rad2deg(np.arctan2(main[:, 1], main[:, 0]))


# --------------------------------------------------------------------------------------
# Corruptions used by H3.  These are *not* elements of the group: that is the point --
# they test whether equivariance buys robustness beyond the symmetry it encodes.
# --------------------------------------------------------------------------------------

def lead_reversal_matrix(kind: str = "LA-RA") -> np.ndarray:
    """Exact 12x12 matrix of a limb-electrode reversal.

    Electrode swaps act as *exact* linear maps on the recorded leads, derived from
    the limb-lead definitions.  ``LA-RA`` (the commonest error) sends
    ``I -> -I``, swaps ``II <-> III`` and ``aVR <-> aVL``, and fixes ``aVF``.
    """
    from eqecg.leads import LEAD_NAMES

    idx = {n: i for i, n in enumerate(LEAD_NAMES)}
    Tm = np.eye(12)
    if kind == "LA-RA":
        Tm[idx["I"], idx["I"]] = -1.0
        for a, b in (("II", "III"), ("aVR", "aVL")):
            Tm[idx[a], idx[a]] = Tm[idx[b], idx[b]] = 0.0
            Tm[idx[a], idx[b]] = Tm[idx[b], idx[a]] = 1.0
    elif kind == "LA-LL":
        for a, b in (("I", "II"), ("aVL", "aVF")):
            Tm[idx[a], idx[a]] = Tm[idx[b], idx[b]] = 0.0
            Tm[idx[a], idx[b]] = Tm[idx[b], idx[a]] = 1.0
        Tm[idx["III"], idx["III"]] = -1.0
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown reversal {kind!r}")
    return Tm


def electrode_displacement_transform(
    rng: np.random.Generator, sigma: float = 0.1, geo: LeadGeometry = GEOMETRY
) -> np.ndarray:
    """A perturbed transform matrix modelling precordial electrode displacement.

    Moving a chest electrode changes the *row* of the transform matrix for that
    lead, which is a genuinely different map from a rotation of the heart vector --
    it leaves the dipole subspace.  This is the honest stress test for H3: a model
    equivariant to ``SO(3)`` has no a-priori guarantee here.
    """
    D = geo.D.copy()
    from eqecg.leads import LEAD_NAMES, reconstruction_matrix, INDEPENDENT_LEADS

    D8 = np.linalg.pinv(reconstruction_matrix()) @ D
    precordial = [i for i, n in enumerate(INDEPENDENT_LEADS) if n.startswith("V")]
    D8[precordial] += sigma * rng.normal(size=(len(precordial), 3)) * np.linalg.norm(
        D8[precordial], axis=1, keepdims=True
    )
    return reconstruction_matrix() @ D8
