"""Tests that pin the paper's mathematical claims to executable checks.

Each test corresponds to a statement in the paper.  If a refactor breaks one of
these, a theorem is wrong in the code, not merely a unit is broken.
"""

import numpy as np
import pytest

from eqecg.group import expm_so3, geodesic_distance, log_so3, random_rotation
from eqecg.leads import GEOMETRY, geometry, lead_identity_matrix, reconstruction_matrix
from eqecg.theory.identifiability import (
    gram_invariant,
    recover_rotation,
    triple_products,
)
from eqecg.theory.intertwiners import (
    adapted_basis,
    equivariant_linear_basis,
    metric_condition_number,
    predicted_intertwiner_dimension,
    reynolds_quadratic,
    reynolds_quadratic_exact,
    reynolds_selfadjointness_defect,
)
from eqecg.theory.invariants import (
    invariant_dimension_series,
    monte_carlo_invariant_dimension,
    total_dimension_series,
)

RNG = np.random.default_rng(0)
TRANSFORMS = ["dower", "kors"]


# --------------------------------------------------------------------------- group
def test_haar_sampler_produces_rotations():
    R = random_rotation(RNG, 256)
    assert np.allclose(np.einsum("nij,nkj->nik", R, R), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(R), 1.0)


def test_haar_sampler_is_uniform_in_angle():
    """The angle density of Haar measure on SO(3) is (1 - cos t)/pi on [0, pi]."""
    R = random_rotation(RNG, 40000)
    angles = np.linalg.norm(log_so3(R), axis=-1)
    edges = np.linspace(0, np.pi, 9)
    observed = np.histogram(angles, bins=edges)[0] / len(angles)
    expected = np.diff(edges - np.sin(edges)) / np.pi
    assert np.max(np.abs(observed - expected)) < 0.01


def test_log_exp_are_mutual_inverses():
    R = random_rotation(RNG, 64)
    assert np.allclose(expm_so3(log_so3(R)), R, atol=1e-10)


# ---------------------------------------------------------------------- lead algebra
def test_limb_lead_identities_span_the_null_space_of_realisable_signals():
    A, B = lead_identity_matrix(), reconstruction_matrix()
    assert np.allclose(A @ B, 0.0, atol=1e-12)
    assert np.linalg.matrix_rank(B) == 8
    assert np.linalg.matrix_rank(A) == 4


@pytest.mark.parametrize("name", TRANSFORMS)
def test_subspace_ranks_are_three_five_eight(name):
    geo = geometry(name)
    assert np.linalg.matrix_rank(geo.P_dip) == 3
    assert np.linalg.matrix_rank(geo.P_res) == 5
    assert np.linalg.matrix_rank(geo.P8) == 8
    # The dipolar subspace must sit inside the realisable subspace.
    assert np.allclose(geo.P8 @ geo.D, geo.D, atol=1e-12)


# ------------------------------------------------------------------------- Theorem 1
@pytest.mark.parametrize("name", TRANSFORMS)
def test_induced_action_is_a_representation(name):
    geo = geometry(name)
    A, B = random_rotation(RNG), random_rotation(RNG)
    assert np.allclose(geo.rho_ext(A) @ geo.rho_ext(B), geo.rho_ext(A @ B), atol=1e-12)
    assert np.allclose(geo.rho_ext(np.eye(3)), geo.P8, atol=1e-12)


@pytest.mark.parametrize("name", TRANSFORMS)
def test_action_is_not_orthogonal_but_is_M_orthogonal(name):
    """The paper's premise: the induced action needs a different metric."""
    geo = geometry(name)
    R = random_rotation(RNG)
    assert geo.anisotropy > 1.01, "transform would have to be non-degenerate to matter"
    assert geo.orthogonality_defect(R) > 1e-2, "action must NOT be Euclidean-orthogonal"
    assert geo.invariance_defect(R) < 1e-12, "action MUST be M-orthogonal (Theorem 1)"


@pytest.mark.parametrize("name", TRANSFORMS)
def test_gauge_turns_the_action_into_a_plain_rotation(name):
    """Constructive content of Theorem 1: block-diagonal diag(R', I_5) in the gauge."""
    geo = geometry(name)
    E = adapted_basis(geo, "M")
    assert np.allclose(E.T @ geo.M_ext @ E, np.eye(8), atol=1e-10)
    for _ in range(8):
        R = random_rotation(RNG)
        A = np.linalg.pinv(E) @ geo.rho_ext(R) @ E
        assert np.abs(A[:3, 3:]).max() < 1e-10 and np.abs(A[3:, :3]).max() < 1e-10
        assert np.allclose(A[:3, :3] @ A[:3, :3].T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(A[:3, :3]), 1.0, atol=1e-10)
        assert np.allclose(A[3:, 3:], np.eye(5), atol=1e-10)


@pytest.mark.parametrize("cin,cout", [(1, 1), (1, 2), (2, 2), (3, 2)])
def test_intertwiner_dimension_matches_schur(cin, cout):
    basis = equivariant_linear_basis(cin, cout)
    assert basis.shape[0] == predicted_intertwiner_dimension(cin, cout)


def test_intertwiner_basis_elements_actually_commute_with_the_action():
    basis = equivariant_linear_basis(1, 1)
    geo = GEOMETRY
    R = random_rotation(RNG)
    rho = geo.rho_ext(R)
    for L in basis[:8]:
        assert np.abs(L @ rho - rho @ L).max() < 1e-8


# ------------------------------------------------------------------------- Theorem 2
def test_reynolds_is_oblique_in_euclidean_and_selfadjoint_in_the_gauge():
    """Exactly where the orthogonal-action generalisation proofs break."""
    euclid = reynolds_selfadjointness_defect(reynolds_quadratic_exact(metric="euclidean"))
    gauge = reynolds_selfadjointness_defect(reynolds_quadratic_exact(metric="M"))
    assert euclid["relative_idempotency_error"] < 1e-10  # still a projection ...
    assert euclid["relative_asymmetry"] > 0.1            # ... but not self-adjoint
    assert gauge["relative_asymmetry"] < 1e-12           # self-adjoint in the M metric
    assert np.isclose(euclid["trace"], gauge["trace"])   # same rank either way
    assert np.isclose(euclid["trace"], 26.0)             # = 1 + 25, matching Theorem 1


def test_exact_reynolds_agrees_with_monte_carlo():
    exact = reynolds_quadratic_exact(metric="euclidean")
    mc = reynolds_quadratic(n_samples=20000, seed=1, metric="euclidean")
    assert np.abs(exact - mc).max() < 0.05


def test_metric_condition_number_exceeds_one():
    assert metric_condition_number() > 1.0


@pytest.mark.parametrize("m,k,expected", [(1, 1, 5), (1, 2, 16), (1, 3, 40), (2, 2, 58)])
def test_invariant_dimensions_are_the_expected_integers(m, k, expected):
    dims = invariant_dimension_series(m, 5 * m, k)
    assert abs(dims[k] - expected) < 1e-6
    assert dims[k] < total_dimension_series(m, 5 * m, k)[k], "symmetry must remove dimensions"


def test_weyl_integration_agrees_with_haar_monte_carlo():
    for m, k in ((1, 2), (2, 3)):
        exact = invariant_dimension_series(m, 5 * m, k)[k]
        mc = monte_carlo_invariant_dimension(m, 5 * m, k, n_samples=20000)
        assert abs(exact - mc) / exact < 0.02


# ------------------------------------------------------------------------- Theorem 3
def test_gram_matrix_is_invariant_and_triple_products_are_equivariant():
    V = RNG.normal(size=(3, 20))
    R = random_rotation(RNG)
    assert np.allclose(gram_invariant(R @ V), gram_invariant(V), atol=1e-10)
    idx = np.array([[0, 1, 2], [3, 4, 5]])
    assert np.allclose(triple_products(R @ V, idx), triple_products(V, idx), atol=1e-10)


def test_pose_is_exactly_recoverable_for_a_full_rank_trajectory():
    V = RNG.normal(size=(3, 40))
    R = random_rotation(RNG)
    assert np.rad2deg(geodesic_distance(R, recover_rotation(V, R @ V))) < 1e-4


def test_collinear_trajectory_has_an_unidentifiable_pose():
    """Rank-1 stratum: an SO(2) stabiliser makes one pose angle unrecoverable."""
    direction = RNG.normal(size=(3, 1))
    V = direction @ RNG.normal(size=(1, 40))
    errors = []
    for _ in range(32):
        R = random_rotation(RNG)
        errors.append(np.rad2deg(geodesic_distance(R, recover_rotation(V, R @ V))))
    assert np.mean(errors) > 30.0, "rank-1 pose must not be identifiable"


def test_planar_trajectory_is_pose_identifiable_but_achiral():
    """Rank-2 stratum: pose recoverable, but every triple product vanishes."""
    basis = np.linalg.qr(RNG.normal(size=(3, 3)))[0][:, :2]
    V = basis @ RNG.normal(size=(2, 40))
    R = random_rotation(RNG)
    assert np.rad2deg(geodesic_distance(R, recover_rotation(V, R @ V))) < 1e-4
    idx = np.array([[0, 1, 2], [5, 9, 13]])
    assert np.abs(triple_products(V, idx)).max() < 1e-10
