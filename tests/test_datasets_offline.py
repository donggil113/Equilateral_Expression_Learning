"""Offline tests for the corpus parsing logic.

The network phase of each loader cannot run in every environment, but the parts
that actually decide label correctness -- SCP-code mapping, official fold
assignment, patient-disjoint splitting, lead reordering -- are pure functions of
files on disk, and are tested here against fixtures.
"""

import numpy as np
import pandas as pd
import pytest

from eqecg.data.datasets import Code15, PTBXL, PTBXL_SUPERCLASSES
from eqecg.data.preprocess import reorder_leads, standardise_record
from eqecg.leads import LEAD_NAMES, lead_identity_matrix


def test_ptbxl_superclass_mapping_handles_zero_likelihood():
    db = pd.DataFrame(
        {
            "scp_codes": [
                "{'NORM': 100.0, 'SR': 0.0}",   # NORM only
                "{'IMI': 80.0, 'ASMI': 20.0}",  # both map to MI
                "{'NDT': 100.0, 'LVH': 50.0}",  # STTC + HYP
                "{'AMI': 0.0}",                 # explicitly absent -> no label
            ],
            "strat_fold": [1, 9, 10, 3],
        }
    )
    st = pd.DataFrame(
        {
            "diagnostic": [1, 1, 1, 1, 1, 1],
            "diagnostic_class": ["NORM", "MI", "MI", "STTC", "HYP", "MI"],
        },
        index=["NORM", "IMI", "ASMI", "NDT", "LVH", "AMI"],
    )
    Y = PTBXL.assign_superclasses(db, st)
    col = {c: i for i, c in enumerate(PTBXL_SUPERCLASSES)}
    assert Y[0, col["NORM"]] == 1 and Y[0].sum() == 1
    assert Y[1, col["MI"]] == 1 and Y[1].sum() == 1
    assert Y[2, col["STTC"]] == 1 and Y[2, col["HYP"]] == 1
    assert Y[3].sum() == 0, "likelihood-0 codes must not produce a positive label"


def test_ptbxl_official_fold_split_is_exactly_8_1_1():
    db = pd.DataFrame({"strat_fold": np.repeat(np.arange(1, 11), 10)})
    s = PTBXL.fold_split(db)
    assert len(s["train"]) == 80 and len(s["val"]) == 10 and len(s["test"]) == 10
    assert set(s["train"]) & set(s["val"]) == set()
    assert set(s["train"]) & set(s["test"]) == set()


def test_code15_split_never_shares_a_patient_across_folds():
    exams = pd.DataFrame({"patient_id": np.repeat(np.arange(200), 3)})
    s = Code15.patient_disjoint_split(exams, seed=1)
    pid = exams["patient_id"].to_numpy()
    groups = {k: set(pid[v]) for k, v in s.items()}
    assert groups["train"] & groups["val"] == set()
    assert groups["train"] & groups["test"] == set()
    assert groups["val"] & groups["test"] == set()
    assert sum(len(v) for v in s.values()) == len(exams)


def test_lead_reordering_is_name_driven_not_positional():
    x = np.arange(12 * 5, dtype=np.float32).reshape(12, 5)
    names = ["V3", "DI", "AVF", "V6", "DIII", "V1", "AVR", "V5", "DII", "V2", "AVL", "V4"]
    out = reorder_leads(x, names)
    for i, canon in enumerate(LEAD_NAMES):
        src = names.index({"I": "DI", "II": "DII", "III": "DIII",
                           "aVR": "AVR", "aVL": "AVL", "aVF": "AVF"}.get(canon, canon))
        assert np.array_equal(out[i], x[src])


def test_unknown_lead_name_is_an_error_not_a_silent_guess():
    with pytest.raises(KeyError):
        reorder_leads(np.zeros((12, 5)), ["X"] * 12)


def test_standardise_enforces_limb_lead_identities():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 5000)).astype(np.float32)
    out, corr = standardise_record(x, 500, list(LEAD_NAMES))
    assert out.shape == (12, 1000)
    assert np.abs(lead_identity_matrix() @ out).max() < 1e-5
    assert 0.0 <= corr <= 1.0
