#!/bin/bash
# Docker entrypoint for the LLM Eval Harness.
# Reads MODE env var to decide what to run:
#   eval      — run the evaluation harness (default)
#   dashboard — launch the Streamlit dashboard

set -euo pipefail

MODE="${MODE:-eval}"

case "$MODE" in
  eval)
    echo "Starting LLM Eval Harness runner..."
    exec llm-eval run --split "${SPLIT:-smoke}"
    ;;
  dashboard)
    echo "Starting Streamlit dashboard on port 8501..."
    exec llm-eval dashboard
    ;;
  *)
    echo "Unknown MODE='$MODE'. Use 'eval' or 'dashboard'."
    exit 1
    ;;
esac
