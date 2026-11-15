#!/usr/bin/env bash
# Resumable download of the improved CSE-CIC-IDS2018 (10.4 GB); retries on connection resets.
URL="https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip"
OUT="data/raw/cicids2018/CSECICIDS2018_improved.zip"
TOTAL=10426851729
for attempt in $(seq 1 60); do
  have=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  if [ "$have" -ge "$TOTAL" ]; then echo "[$(date +%H:%M:%S)] complete ($have bytes)"; exit 0; fi
  echo "[$(date +%H:%M:%S)] attempt $attempt, resuming at $have bytes"
  curl -sS -L --retry 3 --retry-delay 5 -C - -A "research-download" -o "$OUT" "$URL"
  sleep 5
done
echo "[$(date +%H:%M:%S)] gave up after 60 attempts"
exit 1
