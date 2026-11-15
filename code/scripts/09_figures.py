"""Generate every Phase 3 figure from the result files (figures/*.pdf and *.png)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

from xaiids.config import FIGURES, RESULTS            # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linestyle": ":"})
TIER_COL = {"T1": "#4C72B0", "T2": "#DD8452", "T3": "#55A868"}
DS_NAME = {"cic2017": "CIC-IDS2017", "hikari": "HIKARI-2021", "dohbrw": "DoHBrw-2020", "cic2018": "CSE-CIC-IDS2018"}



def load_detection():
    """detection.csv, with per-flow latency replaced by the isolated re-timing when available."""
    d = pd.read_csv(RESULTS / "detection.csv")
    iso = RESULTS / "detection_latency_isolated.csv"
    if iso.exists():
        lat = pd.read_csv(iso)[["dataset", "tier", "model", "lat_p50_ms", "lat_p95_ms"]]
        d = d.drop(columns=["lat_p50_ms", "lat_p95_ms"]).merge(lat, on=["dataset", "tier", "model"], how="left")
    return d

def save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("saved", name)


def label(method, budget):
    return method if pd.isna(budget) or budget in (None, "None") else f"{method}-{int(budget)}"


# --------------------------------------------------------------------------------------- detection
def fig_detection():
    d = load_detection()
    dss = [x for x in DS_NAME if x in set(d.dataset)]
    fig, axes = plt.subplots(2, len(dss), figsize=(3.3 * len(dss), 4.2), squeeze=False)
    for j, ds in enumerate(dss):
        for i, metric in enumerate(("f1", "pr_auc")):
            ax = axes[i, j]
            sub = d[d.dataset == ds]
            mods = [m for m in ("LR", "EBM", "RF", "XGB", "MLP") if m in set(sub.model)]
            w = 0.26
            for k, tier in enumerate(("T1", "T2", "T3")):
                vals = [sub[(sub.model == m) & (sub.tier == tier)][metric].mean() for m in mods]
                ax.bar(np.arange(len(mods)) + (k - 1) * w, vals, w, color=TIER_COL[tier], label=tier)
            ax.set_xticks(range(len(mods)), mods)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{DS_NAME[ds]}: {'F1 at FPR$\\leq$1%' if metric == 'f1' else 'PR-AUC'}")
            if j == 0 and i == 0:
                ax.legend(ncol=3, loc="lower right")
    save(fig, "fig_detection_tiers")


# --------------------------------------------------------------------------------------- cost
def load_e1():
    src = RESULTS / "e1_isolated" if (RESULTS / "e1_isolated").exists() else RESULTS / "e1"
    rows = [json.loads(p.read_text()) for p in src.glob("*.json")]   # timings measured on an idle machine
    e1 = pd.DataFrame(rows)
    e1["label"] = [label(m, b) for m, b in zip(e1.method, e1.budget)]
    return e1


def fig_cost():
    e1 = load_e1()
    det = load_detection()
    t3 = e1[e1.tier == "T3"]
    dss = [x for x in DS_NAME if x in set(t3.dataset)]
    order_models = ["LR", "EBM", "XGB", "RF", "MLP"]
    colors = {"LR": "#8172B3", "EBM": "#937860", "XGB": "#4C72B0", "RF": "#55A868", "MLP": "#C44E52"}
    # one common row order for all panels (shared y axis): model, then median cost over datasets
    t3 = t3.assign(key=t3.model + ": " + t3.label, mo=t3.model.map({m: i for i, m in enumerate(order_models)}))
    order = (t3.groupby(["key", "mo"]).p50_ms.median().reset_index().sort_values(["mo", "p50_ms"]).key.tolist())
    pos = {k: i for i, k in enumerate(order)}
    # height grows with the number of rows so the tick labels never overlap
    fig, axes = plt.subplots(1, len(dss), figsize=(3.6 * len(dss), 0.13 * len(order) + 1.0), squeeze=False,
                             sharey=True)
    for j, ds in enumerate(dss):
        ax = axes[0, j]
        sub = t3[t3.dataset == ds]
        for r in sub.itertuples():
            ax.errorbar(r.p50_ms, pos[r.key], xerr=[[0.0], [r.p95_ms - r.p50_ms]], fmt="o", ms=3.5,
                        color=colors[r.model], capsize=2, lw=0.8)
        d3 = det[(det.dataset == ds) & (det.tier == "T3") & (det.model == "XGB")].lat_p50_ms
        if len(d3):
            ax.axvline(float(d3.iloc[0]), color="k", ls="--", lw=0.8)
            ax.text(float(d3.iloc[0]) * 1.15, len(order) - 0.6, "XGB detection", fontsize=7)
        ax.set_xscale("log")
        ax.set_yticks(range(len(order)), order, fontsize=7)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_title(f"{DS_NAME[ds]} (T3, one core)")
    fig.supxlabel("time per explanation (ms, log scale; dot = p50, bar = p95)", fontsize=9)
    save(fig, "fig_cost_latency")

    # memory
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    piv = t3.groupby("label").peak_mem_increment_mb.median().sort_values()
    ax.barh(range(len(piv)), piv.values, color="#4C72B0")
    ax.set_yticks(range(len(piv)), piv.index)
    ax.set_xlabel("peak resident-memory increment (MB, median over datasets/models)")
    save(fig, "fig_cost_memory")


# --------------------------------------------------------------------------------------- fidelity
def fig_cost_fidelity():
    e1 = load_e1()
    fid = pd.read_csv(RESULTS / "e2_fidelity.csv")
    fid["budget"] = fid.budget.astype(object)
    e1["key"] = e1.dataset + "|" + e1.tier + "|" + e1.model + "|" + e1.label
    fid["label"] = [label(m, b) for m, b in zip(fid.method, fid.budget)]
    fid["key"] = fid.dataset + "|" + fid.tier + "|" + fid.model + "|" + fid.label
    m = fid.merge(e1[["key", "p50_ms"]], on="key")
    m = m[m.tier == "T3"]
    dss = [x for x in DS_NAME if x in set(m.dataset)]
    fig, axes = plt.subplots(1, len(dss), figsize=(3.6 * len(dss), 3.2), squeeze=False)
    mk = {"TreeSHAP-path": "s", "TreeSHAP-interv": "D", "KernelSHAP": "o", "LIME": "^", "IG": "v",
          "DeepSHAP": "P", "LinearSHAP": "X", "EBM-native": "*", "PartitionSHAP": "h"}
    colors = {"LR": "#8172B3", "EBM": "#937860", "XGB": "#4C72B0", "RF": "#55A868", "MLP": "#C44E52"}
    for j, ds in enumerate(dss):
        ax = axes[0, j]
        sub = m[m.dataset == ds]
        for r in sub.itertuples():
            ax.scatter(r.p50_ms, r.faith_median, marker=mk.get(r.method, "o"), color=colors[r.model], s=28,
                       edgecolor="k", lw=0.3)
        # Pareto frontier (min cost, max faithfulness)
        pts = sub.sort_values("p50_ms")[["p50_ms", "faith_median"]].values
        front, best = [], -np.inf
        for x, y in pts:
            if y > best:
                front.append((x, y))
                best = y
        if front:
            fx, fy = zip(*front)
            ax.step(fx, fy, where="post", color="k", lw=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("median time per explanation (ms, log)")
        ax.set_ylabel("faithfulness correlation")
        ax.set_title(f"{DS_NAME[ds]} (T3)")
    handles = [plt.Line2D([], [], marker=v, ls="", color="gray", label=k) for k, v in mk.items()]
    handles += [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig_cost_fidelity_pareto")


def fig_fidelity_tiers():
    fid = pd.read_csv(RESULTS / "e2_fidelity.csv")
    fid["label"] = [label(m, b) for m, b in zip(fid.method, fid.budget)]
    keep = {("XGB", "TreeSHAP-path"), ("XGB", "TreeSHAP-interv"), ("XGB", "KernelSHAP-500"), ("XGB", "LIME-1000"),
            ("MLP", "IG-32"), ("MLP", "DeepSHAP"), ("MLP", "KernelSHAP-500"), ("MLP", "LIME-1000")}
    fid = fid[[(m, l) in keep for m, l in zip(fid.model, fid.label)]]
    dss = [x for x in DS_NAME if x in set(fid.dataset)]
    fig, axes = plt.subplots(2, len(dss), figsize=(3.6 * len(dss), 4.2), squeeze=False, sharex="col")
    for j, ds in enumerate(dss):
        sub = fid[fid.dataset == ds]
        labs = sorted({f"{m}:{l}" for m, l in zip(sub.model, sub.label)})
        for i, (col, ttl) in enumerate((("faith_median", "faithfulness correlation"),
                                        ("aopc_gain", "deletion AOPC $-$ random"))):
            ax = axes[i, j]
            if col == "aopc_gain":
                sub = sub.assign(aopc_gain=sub.aopc_median - sub.aopc_median_rand)
            w = 0.26
            for k, tier in enumerate(("T1", "T2", "T3")):
                vals = [sub[(sub.model + ":" + sub.label == lab) & (sub.tier == tier)][col].mean() for lab in labs]
                ax.bar(np.arange(len(labs)) + (k - 1) * w, vals, w, color=TIER_COL[tier], label=tier)
            ax.set_xticks(range(len(labs)), labs, rotation=60, ha="right")
            ax.set_title(f"{DS_NAME[ds]}: {ttl}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    save(fig, "fig_fidelity_tiers", tight=False)


def fig_agreement():
    ag = pd.read_csv(RESULTS / "e2_agreement.csv")
    ag = ag[ag.tier == "T3"]
    dss = [ds for ds in DS_NAME if len(ag[ag.dataset == ds])]
    mos = [mo for mo in ("XGB", "MLP") if len(ag[ag.model == mo])]
    # rows = datasets, columns = detectors; one shared colour bar to the right of the grid
    fig, axes = plt.subplots(len(dss), len(mos), figsize=(3.6 * len(mos) + 0.6, 3.4 * len(dss)), squeeze=False,
                             constrained_layout=True)
    for i, ds in enumerate(dss):
        for j, mo in enumerate(mos):
            ax = axes[i, j]
            sub = ag[(ag.dataset == ds) & (ag.model == mo)]
            if sub.empty:
                ax.set_visible(False)
                continue
            labs = sorted(set(sub.a) | set(sub.b))
            M = np.eye(len(labs))
            for r in sub.itertuples():
                a, b = labs.index(r.a), labs.index(r.b)
                M[a, b] = M[b, a] = r.jaccard5
            im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(labs)), labs, rotation=60, ha="right", fontsize=7)
            ax.set_yticks(range(len(labs)), labs, fontsize=7)
            ax.grid(False)
            ax.set_title(f"{DS_NAME[ds]}, {mo}", fontsize=9)
            for a in range(len(labs)):
                for b in range(len(labs)):
                    ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=6,
                            color="w" if M[a, b] < 0.5 else "k")
    fig.colorbar(im, ax=axes, shrink=0.5, label="Jaccard@5")
    save(fig, "fig_agreement", tight=False)


# --------------------------------------------------------------------------------------- drift
def fig_drift():
    p = RESULTS / "e3_crossnet.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fam = d[d.part == "family"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.9, 1]})
    ax = axes[0]
    cols = [("J5_halves", "within network: random halves"), ("J5_seed", "within network: retrained (seed)"),
            ("J5_crossnet_same_model", "cross-network: same model"), ("J5_crossnet_own_models", "cross-network: own models")]
    lab = [f"{r.family} {r.tier}\n{'17→18' if r.source == 'cic2017' else '18→17'}" for r in fam.itertuples()]
    w = 0.2
    for k, (c, n) in enumerate(cols):
        ax.bar(np.arange(len(fam)) + (k - 1.5) * w, fam[c].values, w, label=n)
    ax.set_xticks(range(len(fam)), lab, rotation=90, fontsize=6)
    ax.set_ylabel("Jaccard@5 of family profiles")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax = axes[1]
    for tier, mk in (("T1", "o"), ("T3", "s")):
        s = fam[fam.tier == tier]
        ax.scatter(s.recall_drop, s.drift_same_model, marker=mk, label=tier, s=26, edgecolor="k", lw=0.3)
    ax.set_xlabel("recall drop (source $-$ target)")
    ax.set_ylabel("explanation drift $\\Delta_c$ = 1 $-$ J@5")
    ax.legend()
    cpath = RESULTS / "e3_crossnet_corr.json"
    if cpath.exists():
        c = json.loads(cpath.read_text())
        ax.set_title(f"Spearman $\\rho$={c['spearman_drift_vs_recall_drop']:.2f} (p={c['p']:.3f}, n={c['n']})")
    save(fig, "fig_drift")


# --------------------------------------------------------------------------------------- streaming
POL_ORDER = ["B1 Explain-none", "B2a Explain-all (fast)", "B2b Explain-all (deep)", "B3 Static global",
             "B4 Random (matched rate)", "B5 Alert-only", "B6 Amortised (all flows)", "Proposed"]


def fig_stream():
    for p in sorted(RESULTS.glob("e4_*_XGB_T3.csv")) + sorted(RESULTS.glob("e4_*_MLP_T3.csv")):
        d = pd.read_csv(p)
        tag = p.stem.replace("e4_", "")
        base = d[d.policy.isin(POL_ORDER)]
        fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
        for i, scen in enumerate(("mixed", "burst")):
            s = base[base.scenario == scen]
            for j, (col, yl, logy) in enumerate((("coverage_fast_incl_deferred@1s", "alerts explained $\\leq$1 s", False),
                                                   ("O_xai_sync", "sync. XAI overhead $O$ (log)", True),
                                                   ("verdict_latency_p99_ms", "p99 verdict latency ms (log)", True))):
                ax = axes[i, j]
                for pol in POL_ORDER:
                    q = s[s.policy == pol].sort_values("load")
                    if len(q):
                        ax.plot(q.load, q[col].clip(lower=1e-3), marker="o", ms=3, lw=1.6 if pol == "Proposed" else 0.9,
                                color="k" if pol == "Proposed" else None, label=pol)
                if logy:
                    ax.set_yscale("log")
                if col == "O_xai_sync":
                    ax.axhline(0.10, color="r", ls=":", lw=0.8)
                ax.set_xlabel("offered load (fraction of detection capacity)")
                ax.set_ylabel(yl)
                ax.set_title(f"{scen} traffic")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.tight_layout()
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0), frameon=False,
                   fontsize=7.5)
        save(fig, f"fig_stream_{tag}", tight=False)

        abl = d[(d.load == 0.75) & (~d.policy.isin(POL_ORDER[:-1]))]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        for j, (col, yl) in enumerate((("coverage_fast@1s", "alerts explained sync. $\\leq$1 s"),
                                       ("O_xai_sync", "sync. XAI overhead $O$"), ("cache_hit_rate", "cache hit rate"))):
            ax = axes[j]
            pols = list(dict.fromkeys(abl.policy))
            w = 0.38
            for k, scen in enumerate(("mixed", "burst")):
                v = [abl[(abl.policy == pp) & (abl.scenario == scen)][col].mean() for pp in pols]
                ax.bar(np.arange(len(pols)) + (k - 0.5) * w, v, w, label=scen)
            ax.set_xticks(range(len(pols)), [pp.replace("beta=", "$\\beta$=") for pp in pols], rotation=65, ha="right",
                          fontsize=6.5)
            ax.set_ylabel(yl)
            if col == "O_xai_sync":
                ax.axhline(0.10, color="r", ls=":", lw=0.8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06), frameon=False)
        save(fig, f"fig_ablation_{tag}")


# --------------------------------------------------------------------------------------- robustness / pcap
def fig_robustness():
    p = RESULTS / "robustness_perturbation.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for j, (col, yl) in enumerate((("f1", "detection F1"), ("expl_jaccard5", "explanation stability J@5"))):
        ax = axes[j]
        for (ds, mo, kind), g in d.groupby(["dataset", "model", "perturbation"]):
            g = g.sort_values("level")
            ax.plot(g.level, g[col], marker="o" if kind == "timing_jitter" else "s", ms=3,
                    ls="-" if mo == "XGB" else "--", label=f"{DS_NAME.get(ds, ds)} {mo} {kind.replace('_', ' ')}")
        ax.set_xlabel("perturbation level")
        ax.set_ylabel(yl)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=6.5)
    save(fig, "fig_robustness", tight=False)


def fig_pcap():
    p = RESULTS / "pcap_validation.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())
    att = {k: v for k, v in r.items() if "BRUTEFORCE" in k}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6))
    names = [k.replace("hikari_", "").replace(".pcap", "").split("_2021")[0].replace("_", "-") for k in att]
    removed = [v["payload_bytes_removed"] / (v["payload_bytes_removed"] + v["header_bytes_kept"]) for v in att.values()]
    same = [v["metadata_identical_share_min"] for v in att.values()]
    x = np.arange(len(names))
    axes[0].bar(x - 0.2, removed, 0.4, label="bytes removed by stripping")
    axes[0].bar(x + 0.2, same, 0.4, label="metadata features identical")
    axes[0].set_xticks(x, names, fontsize=7)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("share")
    axes[0].legend(fontsize=7, loc="lower center")
    fields = ["requested_server_name", "client_fingerprint", "server_fingerprint", "application_name", "user_agent"]
    full = [np.mean([v["dpi_fields"][f]["full"] for v in att.values()]) for f in fields]
    strp = [np.mean([v["dpi_fields"][f]["stripped"] for v in att.values()]) for f in fields]
    x = np.arange(len(fields))
    axes[1].bar(x - 0.2, full, 0.4, label="full capture")
    axes[1].bar(x + 0.2, strp, 0.4, label="header-only capture")
    axes[1].set_xticks(x, ["TLS SNI", "client fp.", "server fp.", "DPI app.", "user agent"], fontsize=7)
    axes[1].set_ylabel("share of flows with field")
    axes[1].legend(fontsize=7)
    save(fig, "fig_pcap_stripping")


if __name__ == "__main__":
    for fn in (fig_detection, fig_cost, fig_cost_fidelity, fig_fidelity_tiers, fig_agreement, fig_drift,
               fig_stream, fig_robustness, fig_pcap):
        try:
            fn()
        except FileNotFoundError as e:
            print("skip", fn.__name__, e)
