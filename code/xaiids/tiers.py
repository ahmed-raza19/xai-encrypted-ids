"""Feature-visibility tiers (Phase 2, Step 1 and Table 5).

T1  full visibility (plaintext-era reference): every extractor feature except identifiers
    (IP addresses, flow IDs, timestamps).
T2  TLS 1.3 over TCP: T1 minus ports/protocol and any payload- or DPI-derived attribute;
    cleartext TCP-header fields (flags, window sizes, header lengths) remain.
T3  QUIC-strict, protocol-invariant metadata: T2 minus every transport-header field, leaving
    only size, timing, direction, volume and burst statistics.

Rules are expressed as regular expressions over column names so the same tier definition is
applied identically to every row and every dataset.
"""
import re

ID_PATTERNS = [
    r"^unnamed", r"^uid$", r"^id$", r"flow.?id", r"^src.?ip$", r"^dst.?ip$", r"source.?ip", r"destination.?ip",
    r"^originh$", r"^responh$", r"timestamp", r"^ts$", r"^label", r"traffic_category", r"attempted",
    r"^family$", r"^env$", r"^order$", r"^group$", r"^tool$", r"^attack$", r"^split$",
]
PORT_PATTERNS = [r"port", r"^originp$", r"^responp$", r"^protocol$", r"icmp"]
# cleartext transport-header information (visible for TLS-over-TCP, encrypted for QUIC)
TRANSPORT_PATTERNS = [
    r"flag", r"window", r"win.?bytes", r"win.?size", r"header.?(size|len)", r"hdr", r"seg.?size.?min",
    r"tcp",
]


def _match(col, patterns):
    c = col.lower()
    return any(re.search(p, c) for p in patterns)


def feature_columns(columns):
    """All candidate model features (identifiers and labels removed)."""
    return [c for c in columns if not _match(c, ID_PATTERNS)]


def tier_columns(columns, tier):
    feats = feature_columns(columns)
    if tier == "T1":
        return feats
    no_port = [c for c in feats if not _match(c, PORT_PATTERNS)]
    if tier == "T2":
        return no_port
    if tier == "T3":
        return [c for c in no_port if not _match(c, TRANSPORT_PATTERNS)]
    raise ValueError(tier)


# Semantic groups used for grouped attributions and the T3 feature-group ablation (A8, A10).
# Order matters: timing is tested first (e.g. "fwd_iat.max"), then explicit size words
# (e.g. "fwd_pkts_payload.avg" is a size even though it contains "pkts"), then counts.
GROUP_PATTERNS = [
    ("timing", [r"iat", r"time", r"duration", r"active", r"idle", r"rate", r"per.?sec", r"/s$"]),
    ("size", [r"len", r"size", r"payload", r"bytes", r"byts"]),
    ("volume_direction", [r"pkt", r"packet", r"down.?up", r"ratio", r"subflow", r"bulk", r"count", r"sent", r"received"]),
]


def feature_group(col):
    c = col.lower()
    if _match(c, TRANSPORT_PATTERNS):
        return "transport_header"
    if _match(c, PORT_PATTERNS):
        return "port_protocol"
    # timing first: e.g. "fwd_iat.max" is timing even though it contains "max"
    for name, pats in GROUP_PATTERNS:
        if any(re.search(p, c) for p in pats):
            return name
    return "other"
