"""Architecture-level guarantees.

The point of building equivariance into the architecture rather than encouraging it
with a loss is that it can be *checked*.  These tests assert exact invariance to
float32 round-off, and -- equally important -- assert that the controls are NOT
invariant, so a test suite that trivially passes everything would be caught.
"""

import numpy as np
import pytest
import torch

from eqecg.data.synthetic import (
    DipoleECGSimulator,
    SimulatorConfig,
    lead_reversal_matrix,
)
from eqecg.group import random_rotation
from eqecg.leads import GEOMETRY as geo
from eqecg.leads import lead_identity_matrix
from eqecg.models.equivariant import EquivariantECGNet, gram_features
from eqecg.train import MODEL_ZOO, make_model

RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def batch() -> torch.Tensor:
    sim = DipoleECGSimulator(SimulatorConfig(rotation="none"))
    return torch.tensor(sim.sample(6, np.random.default_rng(0))["x"])


def _relative_logit_change(model, x, R) -> float:
    model.eval()
    with torch.no_grad():
        base = model(x)["logits"]
        rho = torch.tensor(geo.rho_ext(R), dtype=torch.float32)
        out = model(torch.einsum("ij,bjt->bit", rho, x))["logits"]
    return float((out - base).abs().max() / base.abs().max().clamp_min(1e-12))


def test_equivariant_model_is_invariant_to_machine_precision(batch):
    torch.manual_seed(0)
    model = make_model("equivariant", 4)
    for _ in range(5):
        assert _relative_logit_change(model, batch, random_rotation(RNG)) < 1e-5


def test_naive_orthogonal_lift_is_measurably_not_invariant(batch):
    """The control: assuming the action is orthogonal breaks equivariance."""
    torch.manual_seed(0)
    model = make_model("equivariant-naive-lift", 4)
    worst = max(_relative_logit_change(model, batch, random_rotation(RNG)) for _ in range(5))
    assert worst > 1e-3


def test_plain_cnn_is_not_invariant(batch):
    torch.manual_seed(0)
    model = make_model("resnet", 4)
    worst = max(_relative_logit_change(model, batch, random_rotation(RNG)) for _ in range(5))
    assert worst > 1e-2


def test_pose_head_is_equivariant_not_invariant(batch):
    torch.manual_seed(0)
    model = EquivariantECGNet(4, predict_pose=True).eval()
    with torch.no_grad():
        base = model(batch)["rotation"]
        for _ in range(5):
            R = random_rotation(RNG)
            rho = torch.tensor(geo.rho_ext(R), dtype=torch.float32)
            out = model(torch.einsum("ij,bjt->bit", rho, batch))["rotation"]
            expected = torch.tensor(R, dtype=torch.float32) @ base
            assert (out - expected).abs().max() < 1e-4


def test_predicted_pose_is_a_rotation_matrix(batch):
    model = EquivariantECGNet(4, predict_pose=True).eval()
    with torch.no_grad():
        R = model(batch)["rotation"]
    assert torch.allclose(R.transpose(1, 2) @ R, torch.eye(3).expand_as(R), atol=1e-4)
    assert torch.allclose(torch.det(R), torch.ones(len(R)), atol=1e-4)


def test_gram_features_are_invariant_under_rotation_of_the_vector_channels():
    v = torch.randn(4, 5, 3, 32)
    R = torch.tensor(random_rotation(RNG), dtype=torch.float32)
    rotated = torch.einsum("ij,bcjt->bcit", R, v)
    assert torch.allclose(gram_features(v), gram_features(rotated), atol=1e-4)


def test_gram_features_keep_trajectory_structure():
    """The trajectory Gram must distinguish loops that the mean-pooled Gram cannot."""
    # exclude the duplicated endpoint so a full period has exactly zero mean
    t = torch.linspace(0, 2 * np.pi, 65)[:-1]
    loop = torch.stack([torch.cos(t), torch.sin(t), torch.zeros_like(t)])[None, None]
    flat = torch.stack([torch.cos(t), torch.zeros_like(t), torch.zeros_like(t)])[None, None]
    assert not torch.allclose(gram_features(loop), gram_features(flat), atol=1e-3)
    # Both have zero mean, so a mean-pooled descriptor alone would call them identical.
    assert loop.mean(-1).abs().max() < 1e-6 and flat.mean(-1).abs().max() < 1e-6


def test_pose_aware_head_breaks_invariance_only_through_the_named_channel(batch):
    torch.manual_seed(0)
    strict = make_model("equivariant", 4)
    aware = make_model("equivariant-pose-aware", 4)
    R = random_rotation(RNG)
    assert _relative_logit_change(strict, batch, R) < 1e-5
    assert _relative_logit_change(aware, batch, R) > 1e-5


@pytest.mark.parametrize("name", sorted(MODEL_ZOO))
def test_every_model_runs_and_is_parameter_matched(name, batch):
    model = make_model(name, 4)
    out = model(batch)
    assert out["logits"].shape == (len(batch), 4)
    n = sum(p.numel() for p in model.parameters())
    assert 9e4 < n < 2.2e5, f"{name} has {n} parameters, outside the matched band"


def test_simulator_output_satisfies_the_limb_lead_identities():
    sim = DipoleECGSimulator(SimulatorConfig(nondipolar_fraction=0.3))
    x = sim.sample(8, np.random.default_rng(1))["x"]
    assert np.abs(lead_identity_matrix() @ x).max() < 1e-5


def test_lead_reversal_matrices_are_involutions():
    for kind in ("LA-RA", "LA-LL"):
        T = lead_reversal_matrix(kind)
        assert np.allclose(T @ T, np.eye(12), atol=1e-12), "swapping twice must be identity"


def test_lead_reversal_is_consistent_with_the_limb_lead_definitions():
    """Applying the reversal to a realisable signal must stay realisable."""
    sim = DipoleECGSimulator(SimulatorConfig(rotation="none"))
    x = sim.sample(4, np.random.default_rng(2))["x"]
    for kind in ("LA-RA", "LA-LL"):
        y = np.einsum("ij,bjt->bit", lead_reversal_matrix(kind), x)
        assert np.abs(lead_identity_matrix() @ y).max() < 1e-4
