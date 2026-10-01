#!/usr/bin/env bash
# Linux / Windows-WSL2 installer. Uses the NVIDIA GPU simulator automatically when nvidia-smi works.
set -euo pipefail
cd "$(dirname "$0")"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
echo "Using $($PY --version) at $PY"
$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU found - installing the GPU simulator"
    pip install -r Code/requirements-gpu.txt
else
    echo "No NVIDIA GPU visible - installing the CPU simulator"
    pip install -r Code/requirements.txt
fi
python Code/run_all.py --check
cat <<'MSG'

Installed. Next time run:
    source .venv/bin/activate
    python Code/run_all.py               # simulation + figures
    python Code/run_all.py --hardware    # also the real IBM quantum computer
MSG
