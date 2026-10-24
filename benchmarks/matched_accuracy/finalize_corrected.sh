#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 benchmarks/matched_accuracy/build.py --repair-only
python3 benchmarks/matched_accuracy/repair_streams.py
python3 benchmarks/matched_accuracy/audit.py
python3 benchmarks/matched_accuracy/analyze.py
python3 benchmarks/matched_accuracy/holdout.py
python3 benchmarks/matched_accuracy/confirm_conservative.py
python3 benchmarks/matched_accuracy/report.py
python3 benchmarks/matched_accuracy/plot.py
python3 benchmarks/matched_accuracy/plot_bunny.py
