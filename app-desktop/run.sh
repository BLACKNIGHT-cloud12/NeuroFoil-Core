#!/usr/bin/env bash
# NeuroFoil Flight Deck — one-command local launch.
# Electron spawns the FastAPI sidecar itself; nothing else is required.
# ELECTRON_DISABLE_SANDBOX: the chrome-sandbox SUID helper is not
# configured on this machine; acceptable for a purely local app that
# loads only 127.0.0.1 content.
set -euo pipefail
cd "$(dirname "$0")"
export ELECTRON_DISABLE_SANDBOX=1
exec npm start
