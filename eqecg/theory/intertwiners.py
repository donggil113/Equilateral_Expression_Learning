"""Theorem 1: the space of ``G_D``-equivariant linear maps, and where the standard
steerable theory breaks for a non-orthogonal action.

Two distinct claims are made computable here.

**(a) Complete characterisation of equivariant linear layers.**  Because ``SO(3)``
is connected, ``L`` commutes with ``rho_ext(R)`` for every ``R`` iff it commutes
with the three Lie-algebra generators ``drho(e_i)``.  That turns an uncountable
family of constraints into a single finite linear system whose null space we solve
exactly.  The dimension is then compared against the analytic prediction obtained
from Schur's lemma applied to the decomposition ``V8 = l1 (+) 5 x l0``:

    dim Hom_G(V8^{c_in}, V8^{c_out}) = c_in c_out  +  25 c_in c_out.

The first term is the vector (``l=1``) multiplicity space -- one scalar per channel
pair, because the standard representation of ``SO(3)`` is absolutely irreducible
over ``R`` -- and the second is the unconstrained mixing of the five non-dipolar
scalar channels.

**(b) The failure point of the orthogonal theory.**  For a *linear* hypothesis
class the Reynolds (Haar-averaging) operator is still an orthogonal projection even
though ``rho`` is not orthogonal, so nothing breaks -- we say so explicitly rather
than overstating the gap.  The break appears at quadratic order and above: on the
class ``f_W(x) = x^T W x`` the Reynolds operator

    S(W) = \\int_{SO(3)} rho(R)^T W rho(R) dR

is idempotent but **not self-adjoint** in the Frobenius inner product, i.e. it is an
*oblique* projection.  Every generalisation argument that decomposes the parameter
norm as ``||W||^2 = ||S W||^2 + ||W - S W||^2`` therefore fails.  Re-deriving in the
metric of Theorem 1 restores self-adjointness exactly, which is what makes the
extension of Elesedy--Zaidi-type results possible at all.
"""

from __future__ import annotations

import numpy as np

from eqecg.group import hat
from eqecg.leads import GEOMETRY, LeadGeometry

__all__ = [
    "generators",
    "adapted_basis",
    "reynolds_quadratic_exact",
    "metric_condition_number",
    "polynomial_norm_distortion",
    "equivariant_linear_basis",
    "predicted_intertwiner_dimension",
    "reynolds_quadratic",
    "reynolds_selfadjointness_defect",
]


def generators(geo: LeadGeometry = GEOMETRY) -> np.ndarray:
    """Lie-algebra generators ``drho(e_i) in R^{12x12}`` of the induced action.

    ``rho_ext`` differs from ``rho`` by the constant projector ``P_res``, which the
    derivative annihilates, so both actions share these generators.
    """
    basis = np.eye(3)
    return np.stack([geo.D @ hat(basis[i]) @ geo.D_pinv for i in range(3)])


def equivariant_linear_basis(
    channels_in: int = 1,
    channels_out: int = 1,
    geo: LeadGeometry = GEOMETRY,
    tol: float = 1e-8,
) -> np.ndarray:
    """Orthonormal basis of the space of equivariant linear maps.

    The maps act on ``V8 (x) R^{channels}``, flattened as ``(channel, lead)``.  Two
    constraint families are imposed simultaneously:

    * commutation with each Lie-algebra generator (equivariance), and
    * support inside the realisable subspace, ``L = P8 L P8`` (the four limb-lead
      identities), which removes the physically meaningless directions.

    Returns
    -------
    ndarray of shape ``(dim, 12*channels_out, 12*channels_in)``
    """
    gens = generators(geo)
    P8 = geo.P8
    n_in, n_out = 12 * channels_in, 12 * channels_out
    eye_in = np.eye(channels_in)
    eye_out = np.eye(channels_out)

    rows = []
    for g in gens:
        # vec(L G_in - G_out L) = (G_in^T (x) I - I (x) G_out) vec(L) with row-major vec.
        G_in = np.kron(eye_in, g)
        G_out = np.kron(eye_out, g)
        rows.append(np.kron(np.eye(n_out), G_in.T) - np.kron(G_out, np.eye(n_in)))
    # Support constraint: L - P8_out L P8_in = 0.
    Pin = np.kron(eye_in, P8)
    Pout = np.kron(eye_out, P8)
    rows.append(np.eye(n_out * n_in) - np.kron(Pout, Pin.T))

    C = np.concatenate(rows, axis=0)
    _, s, Vt = np.linalg.svd(C)
    # Right-singular vectors whose singular value is numerically zero span the null
    # space; Vt can be taller than s, so pad before comparing.
    thresh = tol * max(C.shape) * (s[0] if s.size else 1.0)
    padded = np.concatenate([s, np.zeros(Vt.shape[0] - s.size)])
    return Vt[padded <= thresh].reshape(-1, n_out, n_in)


def predicted_intertwiner_dimension(channels_in: int = 1, channels_out: int = 1) -> int:
    """Analytic prediction from Schur's lemma for ``V8 = l1 (+) 5 x l0``."""
    vector_block = channels_in * channels_out          # End(l1)^G = R
    scalar_block = (5 * channels_in) * (5 * channels_out)
    return vector_block + scalar_block


def reynolds_quadratic(
    geo: LeadGeometry = GEOMETRY,
    n_samples: int = 20000,
    seed: int = 0,
    metric: str = "euclidean",
) -> np.ndarray:
    """Monte-Carlo Reynolds operator on quadratic forms, in the adapted basis.

    Provided as an independent check on :func:`reynolds_quadratic_exact`; both are
    expressed in the coordinates of :func:`adapted_basis` so they are directly
    comparable entry by entry.
    """
    from eqecg.group import random_rotation

    rng = np.random.default_rng(seed)
    Rs = random_rotation(rng, n_samples)
    E = adapted_basis(geo, metric)
    Epinv = np.linalg.pinv(E)
    A = np.einsum("ij,njk,kl->nil", Epinv, geo.rho_ext(Rs), E)
    return np.einsum("nki,nlj->ijkl", A, A).reshape(64, 64) / n_samples


def reynolds_selfadjointness_defect(S: np.ndarray) -> dict[str, float]:
    """Diagnostics for a Reynolds matrix: idempotency, symmetry, rank."""
    idem = float(np.linalg.norm(S @ S - S) / max(np.linalg.norm(S), 1e-30))
    sym = float(np.linalg.norm(S - S.T) / max(np.linalg.norm(S), 1e-30))
    return {
        "relative_idempotency_error": idem,
        "relative_asymmetry": sym,
        "trace": float(np.trace(S)),
    }


def adapted_basis(geo: LeadGeometry = GEOMETRY, metric: str = "euclidean") -> np.ndarray:
    """Basis ``E in R^{12x8}`` of the realisable subspace, dipolar coordinates first.

    With ``metric='M'`` the basis satisfies ``E^T M_ext E = I`` and the action becomes
    the block-diagonal orthogonal matrix ``diag(R', I_5)``; with ``metric='euclidean'``
    it is Euclidean-orthonormal and the action is *not* orthogonal.  The two differ
    by a fixed invertible change of coordinates, which is precisely the "gauge" of
    Theorem 1.
    """
    if metric == "M":
        # Dipolar block: columns of D scaled so that E^T M E = I on that block.
        Edip = geo.D
        res = geo.residual_basis.T                       # 12 x 5, Euclidean-orthonormal
        E = np.concatenate([Edip, res], axis=1)
        gram = E.T @ geo.M_ext @ E
        # gram is identity by construction up to round-off; symmetrise defensively.
        L = np.linalg.cholesky((gram + gram.T) / 2.0)
        return E @ np.linalg.inv(L).T
    if metric == "euclidean":
        Edip = np.linalg.qr(geo.D)[0]                    # 12 x 3, orthonormal
        res = geo.residual_basis.T
        return np.concatenate([Edip, res], axis=1)
    raise ValueError(f"unknown metric {metric!r}")


def reynolds_quadratic_exact(
    geo: LeadGeometry = GEOMETRY, metric: str = "euclidean"
) -> np.ndarray:
    """Closed-form Reynolds operator on quadratic forms -- no sampling error.

    In the ``M``-adapted basis the action is ``diag(R', I_5)`` and Haar integration is
    elementary, using ``E_R[R' A R'^T] = tr(A)/3 * I_3`` for the standard
    representation:

        S(W) = blockdiag( tr(W_vv)/3 * I_3 ,  W_ss ),

    an idempotent of rank ``1 + 25 = 26`` -- exactly the intertwiner dimension of
    Theorem 1, as it must be.  Conjugating by the fixed gauge change gives the
    operator in Euclidean coordinates, where it is no longer self-adjoint.
    """
    S = np.zeros((64, 64))
    # Build S in the M-adapted coordinates on vec(W) (row-major, 8x8 blocks).
    for a in range(8):
        for b in range(8):
            W = np.zeros((8, 8))
            W[a, b] = 1.0
            out = np.zeros((8, 8))
            out[:3, :3] = np.trace(W[:3, :3]) / 3.0 * np.eye(3)
            out[3:, 3:] = W[3:, 3:]
            S[:, a * 8 + b] = out.reshape(-1)
    if metric == "M":
        return S
    if metric == "euclidean":
        E_M = adapted_basis(geo, "M")
        E_E = adapted_basis(geo, "euclidean")
        T = np.linalg.pinv(E_M) @ E_E        # euclidean coords -> M coords
        Tinv = np.linalg.inv(T)
        # W_M = T^{-T} W_E T^{-1}; push S through the congruence.
        fwd = np.kron(Tinv.T, Tinv.T)
        bwd = np.kron(T.T, T.T)
        return bwd @ S @ fwd
    raise ValueError(f"unknown metric {metric!r}")


def metric_condition_number(geo: LeadGeometry = GEOMETRY) -> float:
    """``kappa(M_ext)``: ratio of extreme non-zero eigenvalues of the invariant metric.

    This is the constant by which the generalisation bound of Theorem 2 degrades
    relative to the orthogonal case; it equals 1 exactly when the induced action is
    already orthogonal.
    """
    w = np.linalg.eigvalsh(geo.M_ext)
    w = w[w > 1e-9]
    return float(w.max() / w.min())


def polynomial_norm_distortion(geo: LeadGeometry = GEOMETRY, max_degree: int = 3) -> list[dict]:
    """How much the gauge change distorts the RKHS norm of a degree-$k$ polynomial class.

    Theorem 2 is proved in the inner product of Theorem 1, where the Reynolds
    operator is self-adjoint and the classical argument applies verbatim.  Carrying
    the bound back to the Euclidean-parameterised class costs a multiplicative
    constant: the condition number of the congruence induced on degree-$k$ features.

    For the feature map ``x -> x^{\\otimes k}`` that congruence is ``T^{\\otimes k}``,
    whose singular values are the ``k``-fold products of those of ``T``; the
    condition number is therefore exactly ``kappa(T)^k``.  Computing it rather than
    bounding it keeps the constant honest -- it is modest at low degree and grows
    geometrically, which is itself worth stating.
    """
    E_M = adapted_basis(geo, "M")
    E_E = adapted_basis(geo, "euclidean")
    T = np.linalg.pinv(E_M) @ E_E
    s = np.linalg.svd(T, compute_uv=False)
    kappa = float(s[0] / s[-1])
    return [
        {
            "degree": k,
            "condition_number": kappa**k,
            "min_singular_value": float(s[-1] ** k),
            "max_singular_value": float(s[0] ** k),
        }
        for k in range(1, max_degree + 1)
    ]
