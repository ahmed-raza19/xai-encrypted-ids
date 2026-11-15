#!/usr/bin/env bash
# Runs once the isolated batch has logged BATCH COMPLETE: trigger-parameter sensitivity
# (timing-sensitive, so still alone on the machine), then the subset-removal infidelity.
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
LOG=../results/isolated_log.txt
until grep -q "BATCH COMPLETE" "$LOG" 2>/dev/null; do sleep 30; done
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
$PY scripts/06b_e4_sensitivity.py cic2017 > ../results/e4sens_log_cic2017.txt 2>&1; log "sensitivity cic2017 done"
$PY scripts/06b_e4_sensitivity.py hikari > ../results/e4sens_log_hikari.txt 2>&1; log "sensitivity hikari done"
$PY scripts/04b_infidelity.py > ../results/infidelity_log.txt 2>&1; log "infidelity done"
log "AFTER COMPLETE"
