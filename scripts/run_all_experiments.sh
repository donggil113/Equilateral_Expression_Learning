#!/usr/bin/env bash
# Runs the full experiment suite sequentially and logs each stage.
set -u
cd "$(dirname "$0")/.."
mkdir -p results logs
export EQECG_THREADS="${EQECG_THREADS:-4}"

run () {
  local name="$1"; shift
  echo "=== $name  ($(date -u +%H:%M:%S)) ==="
  if python3 "experiments/$name.py" "$@" > "logs/$name.log" 2>&1; then
    echo "--- $name done ($(date -u +%H:%M:%S))"
  else
    echo "!!! $name FAILED (see logs/$name.log)"; tail -20 "logs/$name.log"
  fi
}

run exp_theory
run exp_equivariance
run exp_h1_sample_efficiency --seeds 3 --steps 800
run exp_h2_h3_probe_robustness --seeds 3 --steps 900
run exp_h4_dipole_breakdown --seeds 2 --steps 700
run exp_capacity_ablation --seeds 2
run exp_lr_fairness
echo "ALL DONE $(date -u +%H:%M:%S)"
