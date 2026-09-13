#!/usr/bin/env python3
"""Download and prepare a public ECG corpus.

Two phases, run separately when the machine with network access is not the machine
that does the compute::

    python scripts/prepare_data.py ptbxl --root data --fetch
    python scripts/prepare_data.py ptbxl --root data --prepare

``--fetch`` is the only phase that needs egress.  If the host is blocked the command
fails immediately with the host named, rather than leaving a half-written archive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eqecg.data.datasets import REGISTRY, load_prepared  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(REGISTRY))
    ap.add_argument("--root", default="data", help="directory holding raw + prepared data")
    ap.add_argument("--fetch", action="store_true", help="download (needs network)")
    ap.add_argument("--prepare", action="store_true", help="parse and standardise (local)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N records")
    args = ap.parse_args()

    dataset = REGISTRY[args.dataset](args.root)
    spec = dataset.spec
    print(f"{spec.name}: {spec.n_records} records, host {spec.host}\n  {spec.citation}\n"
          f"  {spec.note}")

    if not (args.fetch or args.prepare):
        ap.error("choose at least one of --fetch / --prepare")

    if args.fetch:
        try:
            dataset.fetch()
        except RuntimeError as exc:
            print(f"\nFETCH FAILED\n{exc}", file=sys.stderr)
            return 2
    if args.prepare:
        out = dataset.prepare(limit=args.limit)
        data = load_prepared(out)
        print(f"prepared -> {out}\n  signals {data['X'].shape}")
        if "corrections" in data:
            import numpy as np

            c = np.asarray(data["corrections"])
            print(f"  limb-lead identity correction: mean {c.mean():.4f}, "
                  f"p95 {np.percentile(c, 95):.4f}")
            print("  (this is how far real recordings sit from the exact rank-8 "
                  "subspace; large values would undermine the decomposition)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
