#!/usr/bin/env bash
# Download the public datasets used in Phase 3 (resumable). Run from Iteration_3/.
# CIRA-CIC-DoHBrw-2020 must be downloaded manually via the CIC form (see README).
set -u
RAW="data/raw"
get() {  # url  dest
  echo "[$(date +%H:%M:%S)] downloading $2"
  curl -L --fail --retry 5 --retry-delay 10 -C - -A "research-download" -o "$2" "$1" \
    && echo "[$(date +%H:%M:%S)] done $2 ($(du -h "$2" | cut -f1))" \
    || echo "[$(date +%H:%M:%S)] FAILED $2"
}
Z="https://zenodo.org/api/records/6463389/files"
get "$Z/ALLFLOWMETER_HIKARI2021.csv.zip/content"                 "$RAW/hikari/ALLFLOWMETER_HIKARI2021.csv.zip"
get "$Z/ground-truth.zip/content"                                "$RAW/hikari/ground-truth.zip"
get "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip" "$RAW/cicids2017/CICIDS2017_improved.zip"
get "$Z/BRUTEFORCE_XML/pcap/Monday_2021-04-12_0611.pcap/content"   "$RAW/pcap/hikari_BRUTEFORCE_XML_2021-04-12.pcap"
get "$Z/BRUTEFORCE_HTTPS/pcap/Sunday_2021-04-11_2154.pcap/content" "$RAW/pcap/hikari_BRUTEFORCE_HTTPS_2021-04-11.pcap"
get "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip" "$RAW/cicids2018/CSECICIDS2018_improved.zip"
echo "[$(date +%H:%M:%S)] ALL DOWNLOADS FINISHED"
