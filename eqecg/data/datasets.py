"""Acquisition and preparation of the four public corpora named in the study plan.

Each loader has the same two-phase shape:

``fetch``    network phase -- download the archive/files into ``root``.
``prepare``  pure-local phase -- parse the metadata, standardise every record via
             :mod:`eqecg.data.preprocess`, and write a single memory-mapped
             ``(N, 12, 1000)`` float32 array plus a label table.

The split is deliberate.  The parsing and labelling logic -- which is where silent
mistakes actually happen (SCP code mapping, lead order, fold assignment) -- is a
pure function of files on disk, so it is unit-tested offline against synthetic
fixtures without touching the network.

.. note::
   The PhysioNet and Zenodo hosts are not reachable from every execution
   environment.  :func:`require_network` fails with an explicit, actionable message
   naming the blocked host rather than producing a confusing partial download.
"""

from __future__ import annotations

import ast
import json
import os
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eqecg.data.preprocess import TARGET_LENGTH, standardise_record

__all__ = [
    "DatasetSpec",
    "require_network",
    "PTBXL",
    "Code15",
    "MimicIVECG",
    "load_prepared",
    "REGISTRY",
]


@dataclass
class DatasetSpec:
    name: str
    host: str
    url: str
    citation: str
    n_records: int
    note: str = ""


def require_network(host: str, timeout: float = 20.0) -> None:
    """Fail fast, and informatively, when egress to a data host is unavailable."""
    try:
        urllib.request.urlopen(f"https://{host}/", timeout=timeout).close()
    except Exception as exc:  # noqa: BLE001 - we re-raise with context
        raise RuntimeError(
            f"cannot reach {host!r}: {exc}.\n"
            "This environment's egress policy may not allow this host. Run the "
            "`fetch` step where the host is reachable (or mount a pre-downloaded "
            "copy at --root); everything after `fetch` is purely local."
        ) from exc


def _download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Resumable download with a progress line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    pos = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url)
    if pos:
        req.add_header("Range", f"bytes={pos}-")
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "ab") as fh:
        total = int(resp.headers.get("Content-Length", 0)) + pos
        done = pos
        while True:
            block = resp.read(chunk)
            if not block:
                break
            fh.write(block)
            done += len(block)
            if total:
                print(f"\r  {dest.name}: {100*done/total:5.1f}%", end="", flush=True)
    print()
    tmp.rename(dest)
    return dest


# --------------------------------------------------------------------------------------
# PTB-XL
# --------------------------------------------------------------------------------------

PTBXL_SPEC = DatasetSpec(
    name="ptbxl",
    host="physionet.org",
    url=(
        "https://physionet.org/static/published-projects/ptb-xl/"
        "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    ),
    citation="Wagner et al., Sci Data 2020",
    n_records=21837,
    note="official 10-fold split; folds 1-8 train, 9 val, 10 test",
)

#: PTB-XL diagnostic superclasses, the standard 5-way benchmark.
PTBXL_SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


class PTBXL:
    spec = PTBXL_SPEC

    def __init__(self, root: str | Path):
        self.root = Path(root) / "ptbxl"

    # -- phase 1 ---------------------------------------------------------------------
    def fetch(self) -> Path:
        require_network(self.spec.host)
        zip_path = self.root / "ptbxl.zip"
        _download(self.spec.url, zip_path)
        marker = self.root / "extracted"
        if not marker.exists():
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(self.root)
            marker.write_text("ok")
        return self.root

    def _base(self) -> Path:
        """Locate the directory holding ``ptbxl_database.csv`` after extraction."""
        hits = list(self.root.rglob("ptbxl_database.csv"))
        if not hits:
            raise FileNotFoundError(
                f"ptbxl_database.csv not found under {self.root}; run fetch() first"
            )
        return hits[0].parent

    # -- phase 2 (pure local, unit-tested) -------------------------------------------
    @staticmethod
    def assign_superclasses(
        database: pd.DataFrame, statements: pd.DataFrame
    ) -> np.ndarray:
        """Multi-hot ``(N, 5)`` diagnostic superclass matrix.

        A record is positive for a superclass if any of its SCP codes with non-zero
        likelihood maps to it.  Codes with likelihood ``0`` mean "explicitly absent"
        in PTB-XL and are dropped; records whose codes carry no likelihood field at
        all are kept, matching the reference benchmark implementation.
        """
        diag = statements[statements["diagnostic"] == 1] if "diagnostic" in statements else statements
        code_to_class = diag["diagnostic_class"].to_dict()
        Y = np.zeros((len(database), len(PTBXL_SUPERCLASSES)), dtype=np.float32)
        col = {c: i for i, c in enumerate(PTBXL_SUPERCLASSES)}
        for row, raw in enumerate(database["scp_codes"]):
            codes = ast.literal_eval(raw) if isinstance(raw, str) else dict(raw)
            for code, likelihood in codes.items():
                if likelihood == 0:
                    continue
                cls = code_to_class.get(code)
                if cls in col:
                    Y[row, col[cls]] = 1.0
        return Y

    @staticmethod
    def fold_split(database: pd.DataFrame) -> dict[str, np.ndarray]:
        """Official PTB-XL split: folds 1-8 train, fold 9 validation, fold 10 test."""
        fold = database["strat_fold"].to_numpy()
        return {
            "train": np.where(fold <= 8)[0],
            "val": np.where(fold == 9)[0],
            "test": np.where(fold == 10)[0],
        }

    def prepare(self, limit: int | None = None) -> Path:
        import wfdb

        base = self._base()
        db = pd.read_csv(base / "ptbxl_database.csv", index_col="ecg_id")
        st = pd.read_csv(base / "scp_statements.csv", index_col=0)
        if limit:
            db = db.iloc[:limit]

        Y = self.assign_superclasses(db, st)
        splits = self.fold_split(db)

        out_dir = self.root / "prepared"
        out_dir.mkdir(parents=True, exist_ok=True)
        X = np.lib.format.open_memmap(
            out_dir / "signals.npy", mode="w+", dtype=np.float32,
            shape=(len(db), 12, TARGET_LENGTH),
        )
        corrections = np.empty(len(db), dtype=np.float32)
        for i, fname in enumerate(db["filename_lr"]):
            sig, meta = wfdb.rdsamp(str(base / fname))
            X[i], corrections[i] = standardise_record(
                sig.T, int(meta["fs"]), list(meta["sig_name"])
            )
            if i % 500 == 0:
                print(f"\r  ptbxl {i}/{len(db)}", end="", flush=True)
        print()
        X.flush()
        np.savez(
            out_dir / "labels.npz",
            Y=Y, classes=np.array(PTBXL_SUPERCLASSES),
            corrections=corrections,
            **{f"split_{k}": v for k, v in splits.items()},
        )
        (out_dir / "meta.json").write_text(
            json.dumps({"name": "ptbxl", "n": len(db), "task": "multilabel"}, indent=2)
        )
        return out_dir


# --------------------------------------------------------------------------------------
# CODE-15%
# --------------------------------------------------------------------------------------

CODE15_SPEC = DatasetSpec(
    name="code15",
    host="zenodo.org",
    url="https://zenodo.org/records/4916206/files",
    citation="Ribeiro et al., Nat Commun 2020 (CODE-15% subset)",
    n_records=345779,
    note="18 HDF5 shards, 4096 samples at 400 Hz, zero-padded",
)
CODE15_LABELS = ("1dAVb", "RBBB", "LBBB", "SB", "ST", "AF")
CODE15_LEAD_NAMES = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class Code15:
    spec = CODE15_SPEC

    def __init__(self, root: str | Path, n_shards: int = 18):
        self.root = Path(root) / "code15"
        self.n_shards = n_shards

    def fetch(self) -> Path:
        require_network(self.spec.host)
        _download(f"{self.spec.url}/exams.csv?download=1", self.root / "exams.csv")
        for i in range(self.n_shards):
            _download(
                f"{self.spec.url}/exams_part{i}.hdf5?download=1",
                self.root / f"exams_part{i}.hdf5",
            )
        return self.root

    @staticmethod
    def patient_disjoint_split(
        exams: pd.DataFrame, seed: int = 0, fractions=(0.8, 0.1, 0.1)
    ) -> dict[str, np.ndarray]:
        """Split by ``patient_id`` so no subject appears in two folds.

        CODE-15 contains repeat examinations of the same patient; a record-level
        split leaks subject identity and inflates every metric.
        """
        rng = np.random.default_rng(seed)
        pids = exams["patient_id"].to_numpy()
        uniq = np.unique(pids)
        rng.shuffle(uniq)
        n_tr = int(fractions[0] * len(uniq))
        n_va = int(fractions[1] * len(uniq))
        groups = {
            "train": set(uniq[:n_tr]),
            "val": set(uniq[n_tr : n_tr + n_va]),
            "test": set(uniq[n_tr + n_va :]),
        }
        return {k: np.where(np.isin(pids, list(v)))[0] for k, v in groups.items()}

    def prepare(self, limit: int | None = None) -> Path:
        import h5py

        exams = pd.read_csv(self.root / "exams.csv")
        out_dir = self.root / "prepared"
        out_dir.mkdir(parents=True, exist_ok=True)

        wanted = {}
        for i in range(self.n_shards):
            path = self.root / f"exams_part{i}.hdf5"
            if path.exists():
                wanted[i] = path
        if not wanted:
            raise FileNotFoundError(f"no HDF5 shards under {self.root}; run fetch() first")

        ids, rows = [], []
        for path in wanted.values():
            with h5py.File(path, "r") as fh:
                ids.append(np.asarray(fh["exam_id"]))
        ids = np.concatenate(ids)
        if limit:
            ids = ids[:limit]
        index = {int(e): r for r, e in enumerate(ids)}

        X = np.lib.format.open_memmap(
            out_dir / "signals.npy", mode="w+", dtype=np.float32,
            shape=(len(ids), 12, TARGET_LENGTH),
        )
        corrections = np.zeros(len(ids), dtype=np.float32)
        for path in wanted.values():
            with h5py.File(path, "r") as fh:
                exam_ids = np.asarray(fh["exam_id"])
                tracings = fh["tracings"]
                for j, e in enumerate(exam_ids):
                    r = index.get(int(e))
                    if r is None:
                        continue
                    X[r], corrections[r] = standardise_record(
                        np.asarray(tracings[j]).T, 400, CODE15_LEAD_NAMES
                    )
        X.flush()

        exams = exams.set_index("exam_id").loc[[int(e) for e in ids]].reset_index()
        Y = exams[list(CODE15_LABELS)].to_numpy(dtype=np.float32)
        splits = self.patient_disjoint_split(exams)
        np.savez(
            out_dir / "labels.npz", Y=Y, classes=np.array(CODE15_LABELS),
            corrections=corrections, **{f"split_{k}": v for k, v in splits.items()},
        )
        (out_dir / "meta.json").write_text(
            json.dumps({"name": "code15", "n": len(ids), "task": "multilabel"}, indent=2)
        )
        return out_dir


# --------------------------------------------------------------------------------------
# MIMIC-IV-ECG
# --------------------------------------------------------------------------------------

MIMIC_SPEC = DatasetSpec(
    name="mimic-iv-ecg",
    host="physionet.org",
    url="https://physionet.org/files/mimic-iv-ecg/1.0",
    citation="Gow et al., PhysioNet 2023",
    n_records=800035,
    note="credentialed access; used unlabelled for pre-training and for the H2 probe",
)


class MimicIVECG:
    spec = MIMIC_SPEC

    def __init__(self, root: str | Path):
        self.root = Path(root) / "mimic-iv-ecg"

    def fetch(self, max_records: int | None = None) -> Path:
        """Download the record list and the waveform files it names.

        PhysioNet requires credentialed access for this corpus; set
        ``PHYSIONET_USER`` / ``PHYSIONET_PASSWORD`` before calling.
        """
        require_network(self.spec.host)
        user, password = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASSWORD")
        if user and password:
            mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            mgr.add_password(None, f"https://{self.spec.host}/", user, password)
            opener = urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(mgr))
            urllib.request.install_opener(opener)
        _download(f"{self.spec.url}/record_list.csv", self.root / "record_list.csv")
        rl = pd.read_csv(self.root / "record_list.csv")
        if max_records:
            rl = rl.iloc[:max_records]
        for path in rl["path"]:
            for ext in (".hea", ".dat"):
                _download(f"{self.spec.url}/{path}{ext}", self.root / f"{path}{ext}")
        return self.root

    def prepare(self, limit: int | None = None) -> Path:
        import wfdb

        rl = pd.read_csv(self.root / "record_list.csv")
        if limit:
            rl = rl.iloc[:limit]
        out_dir = self.root / "prepared"
        out_dir.mkdir(parents=True, exist_ok=True)
        X = np.lib.format.open_memmap(
            out_dir / "signals.npy", mode="w+", dtype=np.float32,
            shape=(len(rl), 12, TARGET_LENGTH),
        )
        corrections = np.empty(len(rl), dtype=np.float32)
        for i, path in enumerate(rl["path"]):
            sig, meta = wfdb.rdsamp(str(self.root / path))
            X[i], corrections[i] = standardise_record(
                sig.T, int(meta["fs"]), list(meta["sig_name"])
            )
        X.flush()
        np.savez(out_dir / "labels.npz", corrections=corrections,
                 subject_id=rl["subject_id"].to_numpy())
        (out_dir / "meta.json").write_text(
            json.dumps({"name": "mimic-iv-ecg", "n": len(rl), "task": "unlabelled"}, indent=2)
        )
        return out_dir


REGISTRY = {"ptbxl": PTBXL, "code15": Code15, "mimic-iv-ecg": MimicIVECG}


def load_prepared(prepared_dir: str | Path) -> dict:
    """Load a prepared corpus without reading the signal array into memory."""
    prepared_dir = Path(prepared_dir)
    out = {"X": np.load(prepared_dir / "signals.npy", mmap_mode="r")}
    labels = np.load(prepared_dir / "labels.npz", allow_pickle=True)
    out.update({k: labels[k] for k in labels.files})
    out["meta"] = json.loads((prepared_dir / "meta.json").read_text())
    return out
