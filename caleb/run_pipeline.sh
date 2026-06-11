#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
echo "Starting pipeline.py → pipeline_run.log"
python3 -u pipeline.py "$@" 2>&1 | tee pipeline_run.log
