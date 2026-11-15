"""Packet-level validation of payload stripping (Phase 2, Step 3.1) with NFStream.

Runs inside WSL (NFStream is Linux-only):
  ~/nfsenv/bin/python code/scripts/07_pcap_pipeline.py
For every PCAP:
  1. strip: rewrite the capture so each packet keeps only its link, IP and TCP/UDP headers
     (original wire length preserved) -> no application byte survives;
  2. extract flows with NFStream from the full and the stripped capture (DPI on for full,
     off for stripped; IP-layer size accounting; statistical + early-packet (SPLT) features);
  3. compare: share of flows whose size/timing/direction features are identical, and share of
     flows for which payload-derived fields (DPI application, TLS SNI, JA3/JA4) exist.
Outputs: results/pcap_validation.json and data/processed/pcap_flows_stripped.parquet
"""
import json
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from nfstream import NFStreamer

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "pcap"
OUT = ROOT / "data" / "processed"
RES = ROOT / "results"
BENIGN = ["Monday_2021-03-29", "Tuesday_2021-03-30"]


def extract_benign():
    out = []
    z = zipfile.ZipFile(ROOT / "data" / "raw" / "hikari" / "ground-truth.zip")
    for name in z.namelist():
        if any(b in name for b in BENIGN):
            dst = RAW / ("hikari_BENIGN_" + Path(name).name.replace(".anonymized", ""))
            if not dst.exists():
                dst.write_bytes(z.read(name))
            out.append(dst)
    return out


def header_len(pkt):
    """Length of link + network + transport headers for an Ethernet frame."""
    off = 14
    if len(pkt) < off:
        return len(pkt)
    etype = struct.unpack("!H", pkt[12:14])[0]
    while etype in (0x8100, 0x88A8) and len(pkt) >= off + 4:      # VLAN tags
        etype = struct.unpack("!H", pkt[off + 2:off + 4])[0]
        off += 4
    if etype == 0x0800 and len(pkt) >= off + 20:                  # IPv4
        ihl = (pkt[off] & 0x0F) * 4
        proto = pkt[off + 9]
        l4 = off + ihl
    elif etype == 0x86DD and len(pkt) >= off + 40:                # IPv6 (no extension parsing)
        proto = pkt[off + 6]
        l4 = off + 40
    else:
        return min(len(pkt), off)
    if proto == 6 and len(pkt) >= l4 + 13:                        # TCP: data offset
        return min(len(pkt), l4 + ((pkt[l4 + 12] >> 4) * 4))
    if proto == 17:                                               # UDP
        return min(len(pkt), l4 + 8)
    return min(len(pkt), l4)


def strip(src: Path, dst: Path):
    kept = removed = packets = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        gh = fi.read(24)
        endian = "<" if gh[:4] in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"
        linktype = struct.unpack(endian + "I", gh[20:24])[0]
        if linktype != 1:
            raise ValueError(f"{src.name}: link type {linktype} not Ethernet")
        fo.write(gh)
        rec = struct.Struct(endian + "IIII")
        while True:
            h = fi.read(16)
            if len(h) < 16:
                break
            ts_s, ts_us, incl, orig = rec.unpack(h)
            pkt = fi.read(incl)
            n = header_len(pkt)
            fo.write(rec.pack(ts_s, ts_us, n, orig))
            fo.write(pkt[:n])
            kept += n
            removed += incl - n
            packets += 1
    return {"packets": packets, "header_bytes_kept": kept, "payload_bytes_removed": removed}


def flows(path, dpi):
    t0 = time.perf_counter()
    df = NFStreamer(source=str(path), decode_tunnels=True, n_dissections=20 if dpi else 0,
                    statistical_analysis=True, splt_analysis=10, accounting_mode=1,
                    idle_timeout=120, active_timeout=1800).to_pandas()
    return df, time.perf_counter() - t0


KEY = ["src_ip", "src_port", "dst_ip", "dst_port", "protocol", "bidirectional_first_seen_ms"]
DPI_FIELDS = ["application_name", "application_category_name", "requested_server_name",
              "client_fingerprint", "server_fingerprint", "user_agent", "content_type"]


def compare(full, stripped):
    meta_cols = [c for c in stripped.columns if c.startswith(("bidirectional_", "src2dst_", "dst2src_", "splt_"))
                 and c not in KEY and pd.api.types.is_numeric_dtype(stripped[c])]
    m = full.merge(stripped, on=KEY, suffixes=("_full", "_strip"))
    same = {}
    for c in meta_cols:
        a, b = m[c + "_full"], m[c + "_strip"]
        same[c] = float(np.mean(np.isclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-9)))
    dpi = {}
    for c in DPI_FIELDS:
        if c in full.columns:
            nonempty = lambda s: float(np.mean(s.astype(str).str.len().gt(0) & ~s.astype(str).isin(["Unknown", "nan", "None"])))
            dpi[c] = {"full": nonempty(full[c]),
                      "stripped": nonempty(stripped[c]) if c in stripped.columns else 0.0}
    return {"matched_flows": len(m), "flows_full": len(full), "flows_stripped": len(stripped),
            "metadata_columns": len(meta_cols), "metadata_identical_share_min": float(min(same.values())),
            "metadata_identical_share_mean": float(np.mean(list(same.values()))),
            "least_identical": sorted(same.items(), key=lambda kv: kv[1])[:5], "dpi_fields": dpi}


def main():
    pcaps = sorted(RAW.glob("hikari_BRUTEFORCE_*.pcap")) + extract_benign()
    report, all_flows = {}, []
    for p in pcaps:
        sp = RAW / "stripped" / p.name
        sp.parent.mkdir(exist_ok=True)
        st = strip(p, sp)
        full, t_full = flows(p, dpi=True)
        strp_dpi, _ = flows(sp, dpi=True)          # same extractor settings, payload removed
        strp, t_strip = flows(sp, dpi=False)       # deployment setting: DPI disabled
        cmp = compare(full, strp_dpi)
        r = {**st, "size_full_mb": p.stat().st_size / 2**20, "size_stripped_mb": sp.stat().st_size / 2**20,
             "nfstream_full_s": t_full, "nfstream_stripped_s": t_strip,
             "packets_per_s_stripped": st["packets"] / t_strip, "flows_per_s_stripped": len(strp) / t_strip, **cmp}
        report[p.name] = r
        strp["source_pcap"] = p.name
        all_flows.append(strp)
        print(p.name, json.dumps({k: v for k, v in r.items() if k not in ("dpi_fields", "least_identical")}), flush=True)
        print("   DPI fields full vs stripped:", r["dpi_fields"], flush=True)
    (RES / "pcap_validation.json").write_text(json.dumps(report, indent=1, default=str))
    pd.concat(all_flows, ignore_index=True).to_parquet(OUT / "pcap_flows_stripped.parquet", index=False)


if __name__ == "__main__":
    main()
