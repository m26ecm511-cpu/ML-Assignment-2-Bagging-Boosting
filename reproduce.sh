#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r requirements.txt
python run_experiments.py
python make_report.py
