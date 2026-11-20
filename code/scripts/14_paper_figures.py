"""Figures 2-4 of the Phase 4 paper, drawn at their printed size (two-column Springer layout, 6.3 in text width).

Same result files and values as 09_figures.py; only the size, fonts and labels differ, so that every
label prints at 6-7.5 pt and the wording matches the paper (normal traffic / port-scan flood, policy names
as in the policy table).

usage: python 14_paper_figures.py [output_dir]      (default: figures/paper)
"""
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

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else FIGURES / "paper"
OUT.mkdir(parents=True, exist_ok=True)
TEXTWIDTH = 6.3   # inches (455 pt)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linestyle": ":", "grid.linewidth": 0.5, "legend.frameon": False,
    "pdf.fonttype": 42,
})
MODEL_COL = {"LR": "#8172B3", "EBM": "#937860", "XGB": "#4C72B0", "RF": "#55A868", "MLP": "#C44E52"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"saved {name} ({w:.2f} x {h:.2f} in before trimming)")


def label(method, budget):
    return method if pd.isna(budget) or budget in (None, "None") else f"{method}-{int(budget)}"


# ------------------------------------------------------------------ Fig. 2: cost vs faithfulness
def fig_pareto():
    src = RESULTS / "e1_isolated" if (RESULTS / "e1_isolated").exists() else RESULTS / "e1"
    e1 = pd.DataFrame([json.loads(p.read_text()) for p in src.glob("*.json")])
    e1["label"] = [label(m, b) for m, b in zip(e1.method, e1.budget)]
    e1["key"] = e1.dataset + "|" + e1.tier + "|" + e1.model + "|" + e1.label
    fid = pd.read_csv(RESULTS / "e2_fidelity.csv")
    fid["budget"] = fid.budget.astype(object)
    fid["label"] = [label(m, b) for m, b in zip(fid.method, fid.budget)]
    fid["key"] = fid.dataset + "|" + fid.tier + "|" + fid.model + "|" + fid.label
    m = fid.merge(e1[["key", "p50_ms"]], on="key")
    m = m[m.tier == "T3"]
    mk = {"TreeSHAP-path": "s", "TreeSHAP-interv": "D", "KernelSHAP": "o", "LIME": "^", "IG": "v",
          "DeepSHAP": "P", "LinearSHAP": "X", "EBM-native": "*", "PartitionSHAP": "h"}
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.8), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.775, bottom=0.19, top=0.9, wspace=0.08)
    for ax, ds, name in zip(axes, ("cic2017", "hikari"), ("CIC-IDS2017", "HIKARI-2021")):
        sub = m[m.dataset == ds]
        for r in sub.itertuples():
            ax.scatter(r.p50_ms, r.faith_median, marker=mk.get(r.method, "o"), color=MODEL_COL[r.model], s=16,
                       edgecolor="k", lw=0.25, zorder=3)
        front, best = [], -np.inf
        for x, y in sub.sort_values("p50_ms")[["p50_ms", "faith_median"]].values:
            if y > best:
                front.append((x, y))
                best = y
        fx, fy = zip(*front)
        ax.step(fx, fy, where="post", color="k", lw=0.7, alpha=0.7, zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel("median time per explanation (ms, log scale)")
        ax.set_title(f"{name} ($T_3$)")
    axes[0].set_ylabel("faithfulness")
    handles = [plt.Line2D([], [], marker=v, ls="", color="0.45", mec="k", mew=0.25, ms=4, label=k)
               for k, v in mk.items()]
    handles += [plt.Line2D([], [], marker="o", ls="", color=c, mec="k", mew=0.25, ms=4, label=k)
                for k, c in MODEL_COL.items()]
    handles += [plt.Line2D([], [], color="k", lw=0.7, label="best trade-off")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.785, 0.53), ncol=1, handletextpad=0.3,
               labelspacing=0.32, borderaxespad=0)
    save(fig, "fig_cost_fidelity_pareto")


# ------------------------------------------------------------------ Fig. 3: explanation drift
def fig_drift():
    d = pd.read_csv(RESULTS / "e3_crossnet.csv")
    fam = d[d.part == "family"].copy()
    short = {"DoS": "DoS", "DDoS": "DDoS", "BruteForce": "Brute", "Botnet": "Bot"}
    fam["dir"] = np.where(fam.source == "cic2017", 0, 1)
    fam["tord"] = fam.tier.map({"T1": 0, "T3": 1})
    fam["ford"] = fam.family.map({k: i for i, k in enumerate(short)})
    fam = fam.sort_values(["dir", "tord", "ford"]).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.72), gridspec_kw={"width_ratios": [2.35, 1]})
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.21, top=0.76, wspace=0.28)
    ax = axes[0]
    cols = [("J5_halves", "within network: random halves"), ("J5_seed", "within network: retrained model"),
            ("J5_crossnet_same_model", "across networks: same model"),
            ("J5_crossnet_own_models", "across networks: own models")]
    colors = ["#9ecae1", "#3182bd", "#fdae6b", "#d94801"]
    x = np.arange(len(fam), dtype=float) + np.repeat([0.0, 0.6], 8)   # small gap between the two directions
    w = 0.2
    for k, ((c, n), col) in enumerate(zip(cols, colors)):
        ax.bar(x + (k - 1.5) * w, fam[c].values, w, label=n, color=col, lw=0)
    ax.set_xticks(x, [f"{short[r.family]}\n{r.tier.replace('T', '$T_') + '$'}" for r in fam.itertuples()],
                  fontsize=5.8)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_ylabel("top-5 agreement (J@5)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    ax.axvline((x[7] + x[8]) / 2, color="0.4", lw=0.6, ls="--")
    for xs, txt in ((x[:8].mean(), "CIC-IDS2017 $\\rightarrow$ CSE-CIC-IDS2018"),
                    (x[8:].mean(), "CSE-CIC-IDS2018 $\\rightarrow$ CIC-IDS2017")):
        ax.text(xs, 1.02, txt, ha="center", va="bottom", fontsize=6, transform=ax.get_xaxis_transform())
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.09), ncol=2, columnspacing=1.2, handlelength=1.2,
              handletextpad=0.4, borderaxespad=0)
    ax = axes[1]
    for tier, mk, col in (("T1", "o", "#4C72B0"), ("T3", "s", "#DD8452")):
        s = fam[fam.tier == tier]
        ax.scatter(s.recall_drop, s.drift_same_model, marker=mk, label=f"${tier[0]}_{tier[1]}$", s=14,
                   color=col, edgecolor="k", lw=0.25, zorder=3)
    ax.set_xlabel("drop in recall (training $-$ new network)")
    ax.set_ylabel("explanation drift ($1-$J@5)")
    ax.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.2)
    c = json.loads((RESULTS / "e3_crossnet_corr.json").read_text())
    ax.set_title(f"Spearman $\\rho_s$ = {c['spearman_drift_vs_recall_drop']:.2f} "
                 f"($p$ = {c['p']:.3f}, $n$ = {c['n']})", fontsize=6.8)
    save(fig, "fig_drift")


# ------------------------------------------------------------------ Fig. 4: policies under load
POLICIES = [("B1 Explain-none", "B1 none"), ("B2a Explain-all (fast)", "B2a all, TreeSHAP"),
            ("B2b Explain-all (deep)", "B2b all, KernelSHAP"), ("B3 Static global", "B3 global"),
            ("B4 Random (matched rate)", "B4 random"), ("B5 Alert-only", "B5 every alert"),
            ("B6 Amortised (all flows)", "B6 learned"), ("Proposed", "Proposed trigger")]
POL_COL = ["#8c8c8c", "#4C72B0", "#64B5CD", "#937860", "#8172B3", "#DD8452", "#55A868", "k"]


def fig_stream(tag="cic2017_XGB_T3"):
    d = pd.read_csv(RESULTS / f"e4_{tag}.csv")
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 2.25), sharex=True)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.23, top=0.925, wspace=0.3, hspace=0.2)
    metrics = (("coverage_fast_incl_deferred@1s", "alerts explained within 1 s", False),
               ("O_xai_sync", "overhead $O_{\\mathrm{sync}}$ (log scale)", True),
               ("verdict_latency_p99_ms", "99th-percentile verdict delay (ms, log scale)", True))
    for i, (scen, rowname) in enumerate((("mixed", "normal traffic"), ("burst", "port-scan flood"))):
        s = d[d.scenario == scen]
        for j, (col, title, logy) in enumerate(metrics):
            ax = axes[i, j]
            for (pol, name), pc in zip(POLICIES, POL_COL):
                q = s[s.policy == pol].sort_values("load")
                if len(q):
                    ax.plot(q.load, q[col].clip(lower=1e-3), marker="o", ms=2.2, color=pc,
                            lw=1.5 if pol == "Proposed" else 0.8, label=name, zorder=4 if pol == "Proposed" else 3)
            if logy:
                ax.set_yscale("log")
            else:
                ax.set_ylim(-0.05, 1.05)
            if col == "O_xai_sync":
                ax.axhline(0.10, color="r", ls=":", lw=0.8)
            if i == 0:
                ax.set_title(title)
            if i == 1:
                ax.set_xlabel("offered load (share of detection capacity)")
            ax.set_xticks([0.25, 0.75, 1.5])
        axes[i, 0].set_ylabel(rowname, fontsize=7.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, bbox_to_anchor=(0.53, 0.0), columnspacing=0.9,
               handlelength=1.4, handletextpad=0.3, borderaxespad=0)
    save(fig, f"fig_stream_{tag}")


if __name__ == "__main__":
    fig_pareto()
    fig_drift()
    fig_stream()
