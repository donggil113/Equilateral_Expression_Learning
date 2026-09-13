"""Theorem 2: exactly how much of a hypothesis class the symmetry removes.

The sample-complexity benefit of an equivariant model is governed by the fraction
of the hypothesis class that survives Haar averaging.  For polynomial feature maps
that fraction is computable in closed form, so the paper can state a *number*
rather than an asymptotic.

Two independent routes are implemented and cross-checked against each other:

1.  **Weyl integration (exact).**  For a class function ``f`` on ``SO(3)``,

        \\int_{SO(3)} f(R) dR = (1/pi) \\int_0^pi f(theta) (1 - cos theta) d theta,

    and the characters of all symmetric powers are generated at once by

        sum_k chi_{Sym^k V}(R) t^k = 1 / det(I - t rho(R)).

    Since ``dim (Sym^k V)^G = \\int chi_{Sym^k V}``, one 1-D quadrature yields the
    invariant dimension for every degree simultaneously.

2.  **Monte-Carlo trace (independent check).**  The Reynolds operator is idempotent
    even when it is *oblique*, and the trace of an idempotent equals its rank
    regardless of self-adjointness.  Averaging ``tr`` of the induced action on
    ``Sym^k`` therefore estimates the same integer, using no representation theory
    at all.

The relevant representation for a window of ``m`` time samples of a 12-lead ECG is

    V = m x (l=1)  (+)  5m x (l=0),

the first summand being the dipolar (VCG) content that the rotation moves and the
second the non-dipolar residual it fixes.
"""

from __future__ import annotations

from math import comb

import numpy as np

__all__ = [
    "so3_quadrature",
    "representation_eigenvalues",
    "invariant_dimension_series",
    "total_dimension_series",
    "invariant_fraction_table",
    "monte_carlo_invariant_dimension",
]


def so3_quadrature(n: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for the ``SO(3)`` Weyl integration formula on ``[0, pi]``."""
    x, w = np.polynomial.legendre.leggauss(n)
    theta = 0.5 * np.pi * (x + 1.0)
    weight = 0.5 * np.pi * w * (1.0 - np.cos(theta)) / np.pi
    return theta, weight


def representation_eigenvalues(theta: np.ndarray, n_vector: int, n_scalar: int) -> np.ndarray:
    """Eigenvalues of ``rho(theta)`` for ``n_vector`` copies of ``l=1`` plus trivials.

    A rotation by ``theta`` acts on each ``l=1`` copy with eigenvalues
    ``{1, e^{i theta}, e^{-i theta}}`` and on each trivial copy with eigenvalue 1.
    """
    ones = np.ones((theta.size, n_vector + n_scalar), dtype=complex)
    plus = np.exp(1j * theta)[:, None] * np.ones((1, n_vector))
    minus = np.exp(-1j * theta)[:, None] * np.ones((1, n_vector))
    return np.concatenate([ones, plus, minus], axis=1)


def _symmetric_power_characters(eigs: np.ndarray, max_degree: int) -> np.ndarray:
    """Coefficients of ``prod_j 1/(1 - t e_j)`` up to ``t^max_degree``.

    The ``k``-th coefficient is ``chi_{Sym^k V}``, evaluated at each quadrature node.
    """
    coef = np.zeros((eigs.shape[0], max_degree + 1), dtype=complex)
    coef[:, 0] = 1.0
    for j in range(eigs.shape[1]):
        e = eigs[:, j]
        new = np.empty_like(coef)
        new[:, 0] = coef[:, 0]
        for k in range(1, max_degree + 1):
            new[:, k] = e * new[:, k - 1] + coef[:, k]
        coef = new
    return coef


def invariant_dimension_series(
    n_vector: int, n_scalar: int, max_degree: int, n_quad: int = 512
) -> np.ndarray:
    """``dim (Sym^k V)^{SO(3)}`` for ``k = 0 .. max_degree`` (exact up to quadrature)."""
    theta, weight = so3_quadrature(n_quad)
    chars = _symmetric_power_characters(
        representation_eigenvalues(theta, n_vector, n_scalar), max_degree
    )
    dims = (weight[None, :] @ chars.real).ravel()
    return dims


def total_dimension_series(n_vector: int, n_scalar: int, max_degree: int) -> np.ndarray:
    """``dim Sym^k V`` for ``k = 0 .. max_degree``."""
    d = 3 * n_vector + n_scalar
    return np.array([comb(k + d - 1, d - 1) for k in range(max_degree + 1)], dtype=float)


def invariant_fraction_table(
    n_vector_values: list[int], max_degree: int, n_scalar_per_vector: int = 5
) -> list[dict]:
    """Paper Table: dimension of the invariant sub-class versus the full class.

    ``n_scalar_per_vector`` is 5 for the 12-lead ECG, since each time sample
    contributes one dipolar vector and five non-dipolar scalars.
    """
    rows = []
    for m in n_vector_values:
        q = n_scalar_per_vector * m
        inv = invariant_dimension_series(m, q, max_degree)
        tot = total_dimension_series(m, q, max_degree)
        for k in range(1, max_degree + 1):
            rows.append(
                {
                    "time_samples": m,
                    "degree": k,
                    "dim_total": int(round(tot[k])),
                    "dim_invariant": int(round(inv[k])),
                    "fraction_removed": 1.0 - inv[k] / tot[k],
                }
            )
    return rows


def monte_carlo_invariant_dimension(
    n_vector: int, n_scalar: int, degree: int, n_samples: int = 20000, seed: int = 0
) -> float:
    """Independent estimate of ``dim (Sym^k V)^G`` as the trace of the Reynolds operator.

    Uses only Haar sampling and the character of the symmetric power computed from
    the eigenvalues of an explicit rotation matrix -- no Weyl formula.
    """
    from eqecg.group import random_rotation

    rng = np.random.default_rng(seed)
    Rs = random_rotation(rng, n_samples)
    acc = 0.0
    for R in Rs:
        eig = np.linalg.eigvals(R)
        full = np.concatenate(
            [np.tile(eig, n_vector), np.ones(n_scalar, dtype=complex)]
        )[None, :]
        acc += _symmetric_power_characters(full, degree)[0, degree].real
    return acc / n_samples
