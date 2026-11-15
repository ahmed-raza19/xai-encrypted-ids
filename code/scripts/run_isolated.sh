#!/usr/bin/env bash
# Timing-sensitive experiments, run one after another on an otherwise idle machine.
# (The laptop's shared power budget couples cores: background load slows the timing core.)
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
log() { echo "[$(date +%H:%M:%S)] $*" >> ../results/isolated_log.txt; }
log "start"
XAI_E1_DIR=../results/e1_isolated $PY scripts/03_e1_cost.py run cic2017 hikari --tiers T3,T1,T2 >> ../results/isolated_log.txt 2>&1; log "E1 isolated done"
$PY scripts/06_e4_stream.py cic2017 > ../results/e4_log_cic2017.txt 2>&1; log "E4 cic2017 done"
$PY scripts/06_e4_stream.py hikari > ../results/e4_log_hikari.txt 2>&1; log "E4 hikari done"
$PY scripts/10_ablation_extra.py cic2017 hikari > ../results/ablation_extra_log.txt 2>&1; log "ablation extra done"
$PY scripts/08_robustness.py repeat > ../results/timing_repeat_log.txt 2>&1; log "timing repeat done"
log "ALL DONE"
# not timing-sensitive, run last
$PY scripts/08_robustness.py perturb cic2017 hikari > ../results/robustness_log.txt 2>&1; log "robustness perturb done"
log "BATCH COMPLETE"
