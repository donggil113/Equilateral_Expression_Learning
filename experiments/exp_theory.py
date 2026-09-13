"""Numerical verification of Theorems 1-3, producing every constant the paper quotes.

Nothing here is a simulation of the claims: each item is the claim itself evaluated
in exact arithmetic up to floating-point round-off, so a regression in the theory
code shows up as a failing number rather than as prose that no longer matches.
"""

from __future__ import annotations

import numpy as np

from common import save
from eqecg.group import random_rotation
from eqecg.leads import geometry
from eqecg.theory.identifiability import identifiability_report
from eqecg.theory.intertwiners import (
    equivariant_linear_basis,
    metric_condition_number,
    polynomial_norm_distortion,
    predicted_intertwiner_dimension,
    reynolds_quadratic,
    reynolds_quadratic_exact,
    reynolds_selfadjointness_defect,
)
from eqecg.theory.invariants import (
    invariant_dimension_series,
    invariant_fraction_table,
    monte_carlo_invariant_dimension,
    total_dimension_series,
)


def main() -> dict:
    rng = np.random.default_rng(0)
    out: dict = {}

    # -- Theorem 1 -------------------------------------------------------------------
    per_transform = {}
    for name in ("dower", "kors"):
        geo = geometry(name)
        Rs = random_rotation(rng, 64)
        per_transform[name] = {
            "singular_values_of_D": geo.singular_values.tolist(),
            "anisotropy_kappa_D": geo.anisotropy,
            "metric_condition_number": metric_condition_number(geo),
            "max_euclidean_orthogonality_defect": max(
                geo.orthogonality_defect(R) for R in Rs
            ),
            "max_M_invariance_defect": max(geo.invariance_defect(R) for R in Rs),
            "homomorphism_error": float(
                max(
                    np.abs(geo.rho_ext(A) @ geo.rho_ext(B) - geo.rho_ext(A @ B)).max()
                    for A, B in zip(Rs[:16], Rs[16:32])
                )
            ),
            "rank_dipolar": int(np.linalg.matrix_rank(geo.P_dip)),
            "rank_residual": int(np.linalg.matrix_rank(geo.P_res)),
            "rank_realisable": int(np.linalg.matrix_rank(geo.P8)),
        }
    out["theorem1_geometry"] = per_transform

    dims = []
    for cin, cout in ((1, 1), (1, 2), (2, 2), (4, 3)):
        dims.append(
            {
                "channels_in": cin,
                "channels_out": cout,
                "numerical_dimension": int(equivariant_linear_basis(cin, cout).shape[0]),
                "schur_prediction": predicted_intertwiner_dimension(cin, cout),
            }
        )
    out["theorem1_intertwiner_dimensions"] = dims

    # -- Theorem 2 -------------------------------------------------------------------
    reynolds = {}
    for metric in ("euclidean", "M"):
        S = reynolds_quadratic_exact(metric=metric)
        d = reynolds_selfadjointness_defect(S)
        S_mc = reynolds_quadratic(n_samples=20000, seed=1, metric=metric)
        d["monte_carlo_max_deviation"] = float(np.abs(S - S_mc).max())
        reynolds[metric] = d
    out["theorem2_reynolds"] = reynolds

    table = invariant_fraction_table([1, 2, 4, 8], max_degree=4)
    out["theorem2_dimension_table"] = table
    # The constant by which the bound degrades when transported from the gauge back
    # to the Euclidean parameterisation.
    out["theorem2_norm_distortion"] = polynomial_norm_distortion(max_degree=4)
    out["theorem2_monte_carlo_check"] = [
        {
            "n_vector": m,
            "degree": k,
            "exact": int(round(invariant_dimension_series(m, 5 * m, k)[k])),
            "monte_carlo": monte_carlo_invariant_dimension(m, 5 * m, k, n_samples=20000),
            "total": int(total_dimension_series(m, 5 * m, k)[k]),
        }
        for m, k in ((1, 2), (1, 3), (2, 2), (2, 3))
    ]

    # -- Theorem 3 -------------------------------------------------------------------
    out["theorem3_identifiability"] = identifiability_report(n_trials=128, m=64)
    return out


if __name__ == "__main__":
    save("theory", main())
