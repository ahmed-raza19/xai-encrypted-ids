#!/usr/bin/env bash
# Start the isolated timing batch only once the background analysis jobs have exited.
cd "$(dirname "$0")"
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '04_e2_fidelity|05_e3_drift' }) { exit 0 } else { exit 1 }"; do
  sleep 30
done
bash run_isolated.sh
