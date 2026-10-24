#!/usr/bin/env bash
set -euo pipefail
# Run after the original timing sweep, so compilation cannot contend with it.
cd "$(dirname "$0")/../.."
python3 benchmarks/matched_accuracy/build.py --extensions-only
python3 benchmarks/matched_accuracy/check_integral.py
python3 benchmarks/matched_accuracy/run_incremental.py
python3 benchmarks/matched_accuracy/run_analytic.py
python3 benchmarks/matched_accuracy/run_compliance.py
bash benchmarks/matched_accuracy/finalize_corrected.sh
