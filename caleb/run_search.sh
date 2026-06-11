#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
echo "Starting search.py → search_run.log"
python3 -u search.py "$@" 2>&1 | tee search_run.log
