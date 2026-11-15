"""Dataset loading, cleaning, label harmonisation and leakage-safe splitting (Phase 2, Steps 2-3).

Every loader returns a DataFrame with the model features (all candidate T1 columns) plus:
  label   0 = benign, 1 = malicious
  family  harmonised attack family (common taxonomy across datasets)
  group   capture session / file used for time-ordered splitting
  order   within-group temporal order (timestamp when available, else file order)
  env     dataset / network environment identifier
and, where available, ``tool`` (DoH tunnelling tool).
"""
from __future__ import annotations

import io
import re
import zipfile

import numpy as np
import pandas as pd

from .config import DATA_PROC, DATA_RAW, MAX_ROWS, SEED, SPLIT
from .tiers import feature_columns

META = ["label", "family", "attack", "group", "order", "env", "split", "tool"]

# ----------------------------------------------------------------------------- harmonisation

FAMILY_MAP = [
    (r"benign|background|normal", "Benign"),
    (r"ddos|loic|hoic", "DDoS"),
    (r"dos|hulk|goldeneye|slowloris|slowhttp", "DoS"),
    (r"patator|brute|ftp|ssh|bruteforce", "BruteForce"),
    (r"web|xss|sql", "WebAttack"),
    (r"bot|ares", "Botnet"),
    (r"portscan|probing|scan", "Reconnaissance"),
    (r"infiltration|infilteration", "Infiltration"),
    (r"miner|xmrig", "Cryptomining"),
    (r"heartbleed", "Heartbleed"),
    (r"malicious|doh|dns2tcp|dnscat|iodine|tunnel", "Tunnelling"),
]


def harmonise_family(label: str) -> str:
    s = str(label).lower()
    for pat, fam in FAMILY_MAP:
        if re.search(pat, s):
            return fam
    return "Other"


# ----------------------------------------------------------------------------- cleaning

def clean_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Coerce to float32; rates undefined for zero-duration flows (inf/NaN) are set to 0
    rather than dropping the flow, so single-packet scans are kept (Phase 2, Step 3.4)."""
    X = df[cols].apply(pd.to_numeric, errors="coerce").astype("float64")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.clip(-1e12, 1e12).astype("float32")
    df = df.drop(columns=cols).join(X)
    return df


def cap_rows(df: pd.DataFrame, cap: int, seed=SEED) -> pd.DataFrame:
    """Keep every malicious flow (up to half the cap per family proportionally) and
    uniformly subsample benign flows to fit the memory budget."""
    if len(df) <= cap:
        return df
    rng = np.random.default_rng(seed)
    mal = df[df.label == 1]
    ben = df[df.label == 0]
    mal_cap = min(len(mal), cap // 2)
    if len(mal) > mal_cap:
        # water-filling over fine-grained attack labels: small attacks are kept in full and
        # only the largest ones are subsampled, so rare families survive the cap
        key = "attack" if "attack" in mal.columns else "family"
        sizes = mal[key].value_counts().sort_values()
        alloc, left, k = {}, mal_cap, len(sizes)
        for lab, size in sizes.items():
            share = left // k
            alloc[lab] = int(min(size, share))
            left -= alloc[lab]
            k -= 1
        mal = pd.concat([g.sample(n=alloc[lab], random_state=seed) for lab, g in mal.groupby(key)])
    ben_n = min(len(ben), cap - len(mal))
    ben = ben.sample(n=ben_n, random_state=int(rng.integers(1e9)))
    return pd.concat([mal, ben]).sort_index()


def assign_splits(df: pd.DataFrame, fracs=SPLIT) -> pd.DataFrame:
    """Time-ordered split inside every capture group: first 60% train, next 20% validation,
    last 20% test, so every family appears in all partitions and no future flow trains a
    model evaluated on the past (Phase 2, Step 3.6)."""
    df = df.sort_values(["group", "order"]).copy()
    pos = df.groupby("group").cumcount()
    n = df.groupby("group")["order"].transform("size")
    q = pos / n
    df["split"] = np.where(q < fracs[0], "train", np.where(q < fracs[0] + fracs[1], "val", "test"))
    return df


def drop_leakage(df: pd.DataFrame, feats) -> pd.DataFrame:
    """De-duplicate within each split and remove val/test rows whose exact feature vector
    also occurs in training (Phase 2, Step 3.4)."""
    h = pd.util.hash_pandas_object(df[feats], index=False)
    df = df.assign(_h=h.values)
    df = df.drop_duplicates(subset=["split", "_h", "label"])
    train_h = set(df.loc[df.split == "train", "_h"])
    leak = (df.split != "train") & df._h.isin(train_h)
    return df.loc[~leak].drop(columns="_h")


def finalise(df: pd.DataFrame, name: str) -> pd.DataFrame:
    feats = feature_columns([c for c in df.columns if c not in META])
    df = clean_numeric(df, feats)
    df = cap_rows(df, MAX_ROWS.get(name, 400_000))
    df = assign_splits(df)
    df = drop_leakage(df, feats)
    for c in ("tool", "attack"):
        if c not in df.columns:
            df[c] = ""
    df = df[feats + META].reset_index(drop=True)
    df.to_parquet(DATA_PROC / f"{name}.parquet", index=False)
    return df


# ----------------------------------------------------------------------------- loaders

def load_hikari() -> pd.DataFrame:
    z = zipfile.ZipFile(DATA_RAW / "hikari" / "ALLFLOWMETER_HIKARI2021.csv.zip")
    df = pd.read_csv(z.open("ALLFLOWMETER_HIKARI2021.csv"), low_memory=False)
    df["label"] = df["Label"].astype(int)
    df["family"] = df["traffic_category"].map(harmonise_family)
    df["attack"] = df["traffic_category"].astype(str)
    # HIKARI flow CSVs carry no timestamp; attack and benign traffic were captured in separate
    # sessions, so the traffic category is the capture group and file order is the time proxy.
    df["group"] = df["traffic_category"].astype(str)
    df["order"] = np.arange(len(df))
    df["env"] = "hikari"
    return df


def load_cic(year: int, days=None) -> pd.DataFrame:
    """Improved CIC-IDS2017 / CSE-CIC-IDS2018 (Engelen et al. 2021; Liu et al. 2022)."""
    folder = DATA_RAW / ("cicids2017" if year == 2017 else "cicids2018")
    zpath = next(folder.glob("*.zip"))
    z = zipfile.ZipFile(zpath)
    members = [m for m in z.namelist() if m.lower().endswith(".csv")]
    if days:
        members = [m for m in members if any(d.lower() in m.lower() for d in days)]
    parts = []
    for m in members:
        part = pd.read_csv(z.open(m), low_memory=False)
        part["group"] = m.split("/")[-1].rsplit(".", 1)[0]
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    lab = [c for c in df.columns if c.strip().lower() == "label"][0]
    raw_label = df[lab].astype(str)
    # "Attempted" flows carry no attack payload; the authors advise relabelling them benign.
    attempted = raw_label.str.contains("attempted", case=False)
    raw_label = raw_label.where(~attempted, "BENIGN")
    df["family"] = raw_label.map(harmonise_family)
    df["label"] = (df["family"] != "Benign").astype(int)
    df["attack"] = raw_label.str.strip()
    ts = [c for c in df.columns if c.strip().lower() == "timestamp"]
    df["order"] = pd.to_datetime(df[ts[0]], errors="coerce").astype("int64") if ts else np.arange(len(df))
    # Each attack occupies one time window of its day, so splitting a whole day by time would
    # put some families only in train or only in test. Splitting is therefore time-ordered
    # inside every (day, family) group.
    df["group"] = df["group"] + "|" + df["family"]
    df["env"] = f"cic{year}"
    return df


def load_cic_sampled(year: int, days, cap: int, chunksize=400_000, seed=SEED) -> pd.DataFrame:
    """Memory-bounded loader for the very large CSE-CIC-IDS2018 release.

    Pass 1 counts every raw label (label column only). Pass 2 streams the CSVs in chunks and
    keeps each row with a label-specific probability so that the expected sample follows the
    same water-filling allocation as ``cap_rows`` (attacks <= cap/2, small classes complete,
    benign filling the remainder)."""
    folder = DATA_RAW / ("cicids2017" if year == 2017 else "cicids2018")
    z = zipfile.ZipFile(next(folder.glob("*.zip")))
    members = [m for m in z.namelist() if m.lower().endswith(".csv")
               and (not days or any(d.lower() in m.lower() for d in days))]

    def norm(lbl):
        lbl = lbl.astype(str).str.strip()
        return lbl.where(~lbl.str.contains("attempted", case=False), "BENIGN")

    counts = {}
    for m in members:
        head = pd.read_csv(z.open(m), nrows=0)
        lab = [c for c in head.columns if c.strip().lower() == "label"][0]
        for ch in pd.read_csv(z.open(m), usecols=[lab], chunksize=2_000_000):
            for k, v in norm(ch[lab]).value_counts().items():
                counts[(m, k)] = counts.get((m, k), 0) + int(v)
    per_label = pd.Series(counts).groupby(level=1).sum()
    is_benign = per_label.index.map(lambda s: harmonise_family(s) == "Benign")
    attacks = per_label[~is_benign].sort_values()
    alloc, left, k = {}, cap // 2, len(attacks)
    for lbl, size in attacks.items():
        alloc[lbl] = min(size, left // k)
        left -= alloc[lbl]
        k -= 1
    ben_total = per_label[is_benign].sum()
    ben_target = cap - sum(alloc.values())
    prob = {lbl: alloc[lbl] / attacks[lbl] for lbl in attacks.index}
    for lbl in per_label[is_benign].index:
        prob[lbl] = min(1.0, ben_target / max(ben_total, 1))

    rng = np.random.default_rng(seed)
    parts = []
    for m in members:
        for ch in pd.read_csv(z.open(m), chunksize=chunksize, low_memory=False):
            lab = [c for c in ch.columns if c.strip().lower() == "label"][0]
            lbl = norm(ch[lab])
            keep = rng.random(len(ch)) < lbl.map(prob).fillna(0.0).to_numpy()
            if keep.any():
                part = ch.loc[keep].copy()
                part[lab] = lbl[keep]
                part["group"] = m.split("/")[-1].rsplit(".", 1)[0]
                parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    lab = [c for c in df.columns if c.strip().lower() == "label"][0]
    raw_label = df[lab].astype(str)
    df["family"] = raw_label.map(harmonise_family)
    df["label"] = (df["family"] != "Benign").astype(int)
    df["attack"] = raw_label.str.strip()
    ts = [c for c in df.columns if c.strip().lower() == "timestamp"]
    df["order"] = pd.to_datetime(df[ts[0]], errors="coerce").astype("int64")
    df["group"] = df["group"] + "|" + df["family"]
    df["env"] = f"cic{year}"
    return df


def load_dohbrw() -> pd.DataFrame:
    """CIRA-CIC-DoHBrw-2020 layer 2 (benign DoH vs malicious DoH tunnels)."""
    folder = DATA_RAW / "dohbrw"
    files = [p for p in folder.rglob("*.csv")]
    if not files:
        for zp in folder.rglob("*.zip"):
            with zipfile.ZipFile(zp) as z:
                z.extractall(folder)
        files = [p for p in folder.rglob("*.csv")]
    benign = [p for p in files if re.search(r"l2-benign", p.name, re.I)]
    malicious = [p for p in files if re.search(r"l2-malicious", p.name, re.I)]
    if not benign or not malicious:
        raise FileNotFoundError(f"DoHBrw layer-2 CSVs not found in {folder}: {[p.name for p in files]}")
    parts = []
    for p in benign + malicious:
        part = pd.read_csv(p, low_memory=False)
        part["group"] = p.stem
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    lab = [c for c in df.columns if c.strip().lower() == "label"][0]
    df["label"] = df[lab].astype(str).str.contains("malicious", case=False).astype(int)
    df["family"] = np.where(df.label == 1, "Tunnelling", "Benign")
    ts = [c for c in df.columns if c.strip().lower() == "timestamp"]
    df["order"] = pd.to_datetime(df[ts[0]], errors="coerce").astype("int64")
    df["env"] = "dohbrw"
    return df


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_PROC / f"{name}.parquet")
