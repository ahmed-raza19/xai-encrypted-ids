"""Statistical analysis (Phase 2, Step 9) and LaTeX tables for the Phase 3 report.

Reads every result file and writes report/generated/*.tex plus results/stats.json, so every
number in the report is produced by code from the raw results.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

from xaiids import stats                            # noqa: E402
from xaiids.config import RESULTS                   # noqa: E402

GEN = ROOT.parent / "report" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
DS = {"cic2017": "CIC-IDS2017", "hikari": "HIKARI-2021", "dohbrw": "DoHBrw-2020", "cic2018": "CSE-CIC-IDS2018"}
S = {}   # collected statistics -> results/stats.json



def load_detection():
    """detection.csv, with per-flow latency replaced by the isolated re-timing when available."""
    d = pd.read_csv(RESULTS / "detection.csv")
    iso = RESULTS / "detection_latency_isolated.csv"
    if iso.exists():
        lat = pd.read_csv(iso)[["dataset", "tier", "model", "lat_p50_ms", "lat_p95_ms"]]
        d = d.drop(columns=["lat_p50_ms", "lat_p95_ms"]).merge(lat, on=["dataset", "tier", "model"], how="left")
    return d

def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def lab(m, b):
    return m if (b is None or (isinstance(b, float) and np.isnan(b)) or b in ("None", "nan")) else f"{m}-{int(float(b))}"


def write(name, body):
    (GEN / f"{name}.tex").write_text(body, encoding="utf-8")
    print("wrote", name)


def longtable(caption, label, colspec, header, rows, size):
    """Page-breaking table (header repeated) for tables too long for one float."""
    head = " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\"
    n = len(header)
    out = [r"\begingroup", size, r"\setlength{\tabcolsep}{2.5pt}", rf"\begin{{longtable}}{{{colspec}}}",
           rf"\caption{{{caption}}}\label{{{label}}} \\", r"\toprule", head, r"\midrule", r"\endfirsthead",
           rf"\multicolumn{{{n}}}{{l}}{{\textit{{(continued)}}}} \\", r"\toprule", head, r"\midrule", r"\endhead",
           r"\bottomrule", r"\endlastfoot"]
    for r in rows:
        out.append(r"\midrule" if r == "MID" else " & ".join(str(c) for c in r) + r" \\")
    out += [r"\end{longtable}", r"\endgroup", ""]
    return "\n".join(out)


def table(caption, label, colspec, header, rows, size=r"\footnotesize", note=None):
    if len(rows) > 40 and note is None:
        return longtable(caption, label, colspec, header, rows, size)
    out = [r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", size,
           r"\setlength{\tabcolsep}{3.5pt}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
           " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\", r"\midrule"]
    for r in rows:
        out.append(r"\midrule" if r == "MID" else " & ".join(str(c) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if note:
        out += [r"\par\smallskip", rf"\raggedright\scriptsize {note}"]
    out += [r"\end{table}", ""]
    return "\n".join(out)


def f(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{d}f}"


# ------------------------------------------------------------------------------------------ data
def t_data():
    rows = []
    for ds in DS:
        p = RESULTS / f"data_summary_{ds}.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        sp = s["split_rows"]
        fam = {k: sum(v.values()) for k, v in s["family_by_split"].items()}
        attacks = ", ".join(f"{k} {v:,}" for k, v in sorted(fam.items(), key=lambda kv: -kv[1]) if k != "Benign")
        rows.append([DS[ds], f"{s['rows']:,}", f"{sp.get('train', 0):,} / {sp.get('val', 0):,} / {sp.get('test', 0):,}",
                     f"{100 * s['malicious_share_by_split'].get('test', 0):.1f}\\%",
                     f"{s['features_T1']} / {s['features_T2']} / {s['features_T3']}", attacks])
    write("tab_data", table("Processed datasets: size, time-ordered splits, attack share of the test split, features per tier and attack families",
                            "tab:data", r">{\raggedright\arraybackslash}p{1.9cm}rlrl>{\raggedright\arraybackslash}p{5.3cm}",
                            ["Dataset", "Flows", "Train / Val / Test", "Attack", "T1/T2/T3", "Attack families (flows)"], rows))


# ------------------------------------------------------------------------------------------ detection
def t_detection():
    d = load_detection()
    rows = []
    for ds in DS:
        sub = d[d.dataset == ds]
        if sub.empty:
            continue
        if rows:
            rows.append("MID")
        for tier in ("T1", "T3"):            # T2 lies between the two and is shown in Fig. fig:detection
            for m in ("LR", "EBM", "RF", "XGB", "MLP"):
                r = sub[(sub.tier == tier) & (sub.model == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append([DS[ds] if (tier == "T1" and m == "LR") else "", tier, m, r.features, f(r.precision),
                             f(r.recall), f(r.f1), f(r.pr_auc), f(r.roc_auc), f(r.fpr, 4), f(r.mcc), f(r.lat_p50_ms, 3)])
    write("tab_detection", table(
        r"Detection baselines on full visibility ($T_1$) and QUIC-strict metadata ($T_3$); $T_2$ is shown in Fig.~\ref{fig:detection} (test set; threshold set on validation for FPR$\leq$1\%; latency = one flow, one core)",
        "tab:detection", "llllrrrrrrrr", ["Dataset", "Tier", "Model", "$d$", "Prec.", "Recall", "F1", "PR-AUC", "ROC-AUC",
                                          "FPR", "MCC", "ms/flow"], rows, size=r"\scriptsize"))
    # T1 vs T3 detection drop per model
    out = {}
    for ds in DS:
        for m in ("LR", "RF", "XGB", "MLP"):
            a = d[(d.dataset == ds) & (d.model == m) & (d.tier == "T1")]
            b = d[(d.dataset == ds) & (d.model == m) & (d.tier == "T3")]
            if len(a) and len(b):
                out[f"{ds}|{m}"] = {"f1_T1": float(a.f1.iloc[0]), "f1_T3": float(b.f1.iloc[0]),
                                    "prauc_T1": float(a.pr_auc.iloc[0]), "prauc_T3": float(b.pr_auc.iloc[0])}
    S["detection_tier_drop"] = out


# ------------------------------------------------------------------------------------------ E1 cost
def load_e1():
    src = RESULTS / "e1_isolated" if (RESULTS / "e1_isolated").exists() else RESULTS / "e1"
    rows = [json.loads(p.read_text()) for p in src.glob("*.json")]   # timings measured on an idle machine
    e = pd.DataFrame(rows)
    e["label"] = [lab(m, b) for m, b in zip(e.method, e.budget)]
    return e


def t_cost():
    e = load_e1()
    det = load_detection()
    rows = []
    for ds in DS:
        sub = e[(e.dataset == ds) & (e.tier == "T3")]
        if sub.empty:
            continue
        if rows:
            rows.append("MID")
        first = True
        for m in ("LR", "EBM", "XGB", "RF", "MLP"):
            dl = det[(det.dataset == ds) & (det.tier == "T3") & (det.model == m)].lat_p50_ms
            dl = float(dl.iloc[0]) if len(dl) else np.nan
            for r in sub[sub.model == m].sort_values("p50_ms").itertuples():
                rows.append([DS[ds] if first else "", m, esc(r.label), r.n, f(r.p50_ms, 3 if r.p50_ms < 0.1 else 2), f(r.p95_ms, 3 if r.p95_ms < 0.1 else 2), f(r.p99_ms, 3 if r.p99_ms < 0.1 else 2),
                             f(r.p50_ms / dl, 1) + r"$\times$" if dl == dl else "--", f(r.peak_mem_increment_mb, 1),
                             f(r.throughput_per_s, 1)])
                first = False
    write("tab_cost_full", table(          # every column; kept for the extended version, not used in the report
        r"E1 --- per-flow explanation cost on the QUIC-strict tier T3 (one pinned core). Ratio = median explanation time / median detection time of the same model.",
        "tab:costfull", "lllrrrrrrr", ["Dataset", "Model", "Explainer", "$n$", "p50 ms", "p95 ms", "p99 ms", "ratio", "MB", "expl./s"], rows,
        size=r"\scriptsize"))

    # compact version for the report: one row per detector + explainer, datasets side by side
    t3 = e[e.tier == "T3"]
    dsets = [ds for ds in ("cic2017", "hikari") if ds in set(t3.dataset)]
    rows = []
    for m in ("LR", "EBM", "XGB", "RF", "MLP"):
        sub = t3[t3.model == m]
        if sub.empty:
            continue
        if rows:
            rows.append("MID")
        ref = sub[sub.dataset == dsets[0]].sort_values("p50_ms")
        first = True
        for lbl in ref.label:
            row = [m if first else "", esc(lbl)]
            mem = []
            for ds in dsets:
                r = sub[(sub.dataset == ds) & (sub.label == lbl)]
                dl = det[(det.dataset == ds) & (det.tier == "T3") & (det.model == m)].lat_p50_ms
                dl = float(dl.iloc[0]) if len(dl) else np.nan
                if r.empty:
                    row += ["--", "--"]
                    continue
                r = r.iloc[0]
                row += [f(r.p50_ms, 3 if r.p50_ms < 0.1 else 2), f(r.p50_ms / dl, 1) + r"$\times$" if dl == dl else "--"]
                mem.append(r.peak_mem_increment_mb)
            rows.append(row + [f(max(mem), 1) if mem else "--"])
            first = False
    hdr = ["Model", "Explainer"]
    for ds in dsets:
        short = "CIC" if ds == "cic2017" else "HIKARI"
        hdr += [rf"{short} ms", rf"{short} ratio"]
    write("tab_cost", table(
        r"E1 --- median per-flow explanation cost on the QUIC-strict tier $T_3$ (one pinned core) for both datasets. "
        r"Ratio = median explanation time / median detection time of the same detector; MB = peak memory increment "
        r"(larger of the two datasets). Tail latencies and throughput are summarised in the text.",
        "tab:cost", "llrrrrr", hdr + ["MB"], rows, size=r"\scriptsize"))
    S["cost_T3"] = e[e.tier == "T3"][["dataset", "model", "label", "p50_ms", "p95_ms", "p99_ms", "peak_mem_increment_mb"]].to_dict("records")


# ------------------------------------------------------------------------------------------ E2 fidelity
def t_fidelity():
    fid = pd.read_csv(RESULTS / "e2_fidelity.csv")
    fid["label"] = [lab(m, b) for m, b in zip(fid.method, fid.budget)]
    per = pd.read_parquet(RESULTS / "e2_perflow.parquet")
    per["label"] = [lab(m, b) for m, b in zip(per.method, per.budget)]
    rows = []
    for ds in DS:
        sub = fid[(fid.dataset == ds) & (fid.tier == "T3")]
        if sub.empty:
            continue
        if rows:
            rows.append("MID")
        first = True
        for m in ("LR", "EBM", "XGB", "RF", "MLP"):
            for r in sub[sub.model == m].sort_values("faith_median", ascending=False).itertuples():
                rows.append([DS[ds] if first else "", m, esc(r.label), f(r.faith_median), f(r.faith_benign),
                             f(r.aopc_median - r.aopc_median_rand), f(r.aopc_benign - r.aopc_benign_rand),
                             f(getattr(r, "infid_norm", np.nan), 2) if "infid_norm" in sub else f"{r.infidelity:.2e}",
                             f(r.max_sensitivity, 2), f(r.gini, 2)])
                first = False
    write("tab_fidelity_full", table(      # every column; kept for the extended version, not used in the report
        r"E2 --- explanation fidelity on tier T3. Faithfulness: median over flows with the training-median (med) and benign-sample (ben) removal baselines; AOPC$^+$ = deletion AOPC minus random-order AOPC; infidelity = normalised subset-removal infidelity (0 = attributions predict every removal effect after optimal rescaling, 1 = no linear relation); infidelity and max-sensitivity: median, lower is better.",
        "tab:fidelityfull", "lllrrrrrrr", ["Dataset", "Model", "Explainer", r"Faith.\,med", r"Faith.\,ben", r"AOPC$^+$\,med",
                                           r"AOPC$^+$\,ben", "Infid.", "Max-sens.", "Gini"], rows, size=r"\scriptsize"))

    # compact version for the report: one row per detector + explainer, datasets side by side
    t3 = fid[fid.tier == "T3"].assign(aopc_gain=lambda x: x.aopc_median - x.aopc_median_rand)
    dsets = [ds for ds in ("cic2017", "hikari") if ds in set(t3.dataset)]
    short = {"cic2017": "CIC", "hikari": "HIK"}
    rows = []
    for m in ("LR", "EBM", "XGB", "RF", "MLP"):
        sub = t3[t3.model == m]
        if sub.empty:
            continue
        if rows:
            rows.append("MID")
        ref = sub[sub.dataset == dsets[0]].sort_values("faith_median", ascending=False)
        first = True
        for lbl in ref.label:
            row = [m if first else "", esc(lbl)]
            for col, d in (("faith_median", 3), ("aopc_gain", 3), ("infid_norm", 2)):
                for ds in dsets:
                    r = sub[(sub.dataset == ds) & (sub.label == lbl)]
                    row.append(f(float(r[col].iloc[0]), d) if len(r) and col in r else "--")
            rows.append(row)
            first = False
    hdr = ["Model", "Explainer"]
    for name in (r"Faith.", r"AOPC$^+$", r"Infid."):
        hdr += [rf"{name}\,{short[ds]}" for ds in dsets]
    write("tab_fidelity", table(
        r"E2 --- explanation fidelity on tier $T_3$ (median over flows, training-median removal baseline; CIC = CIC-IDS2017, "
        r"HIK = HIKARI-2021). Faith.\ = faithfulness correlation and AOPC$^+$ = deletion AOPC minus random-order AOPC "
        r"(higher is better); Infid.\ = normalised subset-removal infidelity (0 = attributions predict every removal effect "
        r"after optimal rescaling, 1 = no linear relation; lower is better). The benign-sample baseline, max-sensitivity "
        r"and sparsity are summarised in the text.",
        "tab:fidelity", "llrrrrrr", hdr, rows, size=r"\scriptsize"))

    # Friedman-Nemenyi over explainers (blocks = flows) per dataset x model on T3 faithfulness
    fr = {}
    for (ds, m), g in per[per.tier == "T3"].groupby(["dataset", "model"]):
        piv = g.pivot_table(index="flow", columns="label", values="faith_median")
        piv = piv.dropna(axis=0)
        if piv.shape[1] >= 3 and piv.shape[0] >= 10:
            r = stats.friedman_nemenyi(piv.values)
            n_b, k_m = piv.shape
            fr[f"{ds}|{m}"] = {"chi2": r["chi2"], "p": r["p"], "cd": r["cd"], "n": int(n_b), "k": int(k_m),
                               "kendall_w": float(r["chi2"] / (n_b * (k_m - 1))),
                               "mean_ranks": dict(zip(piv.columns, map(float, r["mean_ranks"])))}
    S["friedman_faithfulness_T3"] = fr

    # tier effect: paired Wilcoxon T1 vs T3 on the same flows for each explainer, Holm-corrected
    tests = []
    for (ds, m, l), g in per.groupby(["dataset", "model", "label"]):
        a = g[g.tier == "T1"].set_index("flow").faith_median
        b = g[g.tier == "T3"].set_index("flow").faith_median
        common = a.index.intersection(b.index)
        if len(common) >= 10:
            w = stats.wilcoxon_rb(a.loc[common].values, b.loc[common].values)
            tests.append({"dataset": ds, "model": m, "explainer": l, "n": len(common),
                          "faith_T1": float(a.loc[common].mean()), "faith_T3": float(b.loc[common].mean()), **w})
    if tests:
        t = pd.DataFrame(tests)
        t["p_holm"] = stats.holm(t.p.values)
        t.to_csv(RESULTS / "stats_tier_wilcoxon.csv", index=False)
        S["tier_wilcoxon"] = t.to_dict("records")
        rows = [[DS[r.dataset], r.model, esc(r.explainer), r.n, f(r.faith_T1), f(r.faith_T3), f(r.faith_T3 - r.faith_T1),
                 f(r.rank_biserial, 2), f"{r.p_holm:.1e}"] for r in t.itertuples()]
        write("tab_tier_tests", table(
            r"RQ2 --- faithfulness under full visibility (T1) vs.\ QUIC-strict metadata (T3) on the same flows (paired Wilcoxon, Holm-corrected; effect size = matched-pairs rank-biserial)",
            "tab:tiertests", "lllrrrrrr", ["Dataset", "Model", "Explainer", "$n$", "T1", "T3", r"$\Delta$", "$r_{rb}$", "$p_{Holm}$"],
            rows, size=r"\scriptsize"))

    col = pd.read_csv(RESULTS / "e2_collinearity.csv")
    rows = [[DS[r.dataset], r.tier, r.features, f(r.mean_abs_spearman), f(r["share_pairs_abs_rho_gt_0.9"])] for _, r in col.iterrows()]
    write("tab_collinearity", table(r"Feature collinearity per visibility tier (training data)", "tab:collinearity", "llrrr",
                                    ["Dataset", "Tier", "Features", r"mean $|\rho_s|$", r"share $|\rho_s|>0.9$"], rows))

    ag = pd.read_csv(RESULTS / "e2_agreement.csv")
    S["agreement_T3"] = ag[ag.tier == "T3"].to_dict("records")


# ------------------------------------------------------------------------------------------ E3 drift
def t_drift():
    p = RESULTS / "e3_crossnet.csv"
    if p.exists():
        d = pd.read_csv(p)
        fam = d[d.part == "family"]
        rows = [[r.tier, "17$\\rightarrow$18" if r.source == "cic2017" else "18$\\rightarrow$17", r.family,
                 f(r.recall_source), f(r.recall_target), f(r.J5_halves, 2), f(r.J5_seed, 2),
                 f(r.J5_crossnet_same_model, 2), f(r.J5_crossnet_own_models, 2)] for r in fam.itertuples()]
        write("tab_drift", table(
            r"E3 --- cross-network explanation consistency for shared attack families (Jaccard@5 of mean $|\phi|$ profiles; TreeSHAP-interventional, XGBoost). Noise floors: random halves of the same flows and a model retrained with another seed.",
            "tab:drift", "lllrrrrrr", ["Tier", "Direction", "Family", "Rec.\\,src", "Rec.\\,tgt", "Halves", "Seed",
                                       "Same model", "Own models"], rows, size=r"\scriptsize"))
        ov = d[d.part == "overall"]
        S["crossnet_overall"] = ov.to_dict("records")
        # paired test: within-network (halves) vs cross-network (same model) agreement
        if len(fam) >= 6:
            S["crossnet_wilcoxon_halves_vs_crossnet"] = stats.wilcoxon_rb(fam.J5_halves.values, fam.J5_crossnet_same_model.values)
            S["crossnet_wilcoxon_seed_vs_crossnet"] = stats.wilcoxon_rb(fam.J5_seed.values, fam.J5_crossnet_same_model.values)
        c = RESULTS / "e3_crossnet_corr.json"
        if c.exists():
            S["crossnet_corr"] = json.loads(c.read_text())
    p = RESULTS / "e3_temporal.csv"
    if p.exists():
        S["temporal"] = pd.read_csv(p).to_dict("records")
    p = RESULTS / "e3_p9_replication.csv"
    if p.exists():
        S["p9_replication"] = pd.read_csv(p).to_dict("records")


# ------------------------------------------------------------------------------------------ E4 streaming
POL = ["B1 Explain-none", "B2a Explain-all (fast)", "B2b Explain-all (deep)", "B3 Static global",
       "B4 Random (matched rate)", "B5 Alert-only", "B6 Amortised (all flows)", "Proposed"]


def t_stream():
    for p in sorted(RESULTS.glob("e4_*_T3.csv")):
        tag = p.stem.replace("e4_", "")
        d = pd.read_csv(p)
        ds = d.dataset.iloc[0]
        rows = []
        for scen in ("mixed", "burst"):
            if rows:
                rows.append("MID")
            for pol in POL:
                r = d[(d.scenario == scen) & (d.policy == pol) & (d.load == 0.75)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append([scen if pol == POL[0] else "", esc(pol), f(r.trigger_rate), f(r.O_xai_sync), f(r.O_async, 2),
                             f(r["coverage_fast@1s"]), f(r["coverage_fast_incl_deferred@1s"]), f(r["coverage_deep@30s"]),
                             f(r.verdict_latency_p99_ms, 1), f(r.cache_hit_rate, 2)])
        write(f"tab_stream_{tag}", table(
            rf"E4 --- explanation policies at 75\% offered load on {DS[ds]} ({d.model.iloc[0]}, T3). $O_{{\mathrm{{sync}}}}$ = all synchronous XAI work (trigger, cache, explanation) relative to detection time; Cov.\,sync = true alerts with an exact local explanation on the detection path within 1\,s; +def.\ adds the deferred lane; deep = escalations served within 30\,s.",
            f"tab:stream:{tag}", "llrrrrrrrr", ["Traffic", "Policy", "Expl.\\,rate", r"$O_{\mathrm{sync}}$", r"$O_{\mathrm{async}}$",
                                                "Cov.\\,sync", "+def.", "Deep", "p99 ms", "Cache"], rows, size=r"\scriptsize"))
        ab = d[(d.load == 0.75) & (~d.policy.isin(POL[:-1]))]
        rows = []
        for scen in ("mixed", "burst"):
            if rows:
                rows.append("MID")
            for pol in dict.fromkeys(ab.policy):
                r = ab[(ab.scenario == scen) & (ab.policy == pol)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append([scen if pol == "Proposed" else "", esc(pol).replace("beta=", r"$\beta$="), f(r.O_xai_sync), f(r.O_async, 2), f(r["coverage_fast@1s"]),
                             f(r["coverage_fast_incl_deferred@1s"]), f(r["coverage_deep@30s"]), f(r.verdict_latency_p99_ms, 1),
                             f(r.cache_hit_rate, 2), int(r.deep_shed)])
        write(f"tab_ablation_{tag}", table(
            rf"Trigger ablations A1--A6 on {DS[ds]} at 75\% load (each row removes or changes one component of the proposed trigger)",
            f"tab:ablation:{tag}", "llrrrrrrrr", ["Traffic", "Variant", r"$O_{\mathrm{sync}}$", r"$O_{\mathrm{async}}$", "Cov.\\,sync", "+def.",
                                                  "Deep", "p99 ms", "Cache", "Shed"], rows, size=r"\scriptsize"))
        # paired window-level tests: Proposed vs B5 / B2a (same stream windows)
        w = pd.read_parquet(RESULTS / f"e4_{tag}_windows.parquet")
        res = {}
        for scen in ("mixed", "burst"):
            for other in ("B5 Alert-only", "B2a Explain-all (fast)"):
                a = w[(w.scenario == scen) & (w.load == 0.75) & (w.policy == "Proposed")].sort_values("window")
                b = w[(w.scenario == scen) & (w.load == 0.75) & (w.policy == other)].sort_values("window")
                for col in ("verdict_p99_ms", "O_xai_sync"):
                    res[f"{scen}|{other}|{col}"] = stats.wilcoxon_rb(a[col].values, b[col].values)
        S[f"stream_window_tests_{tag}"] = res
        fp = RESULTS / f"e4_{tag}_fidelity.csv"
        if fp.exists():
            fd = pd.read_csv(fp)
            S[f"delivered_fidelity_{tag}"] = fd.groupby(["scenario", "policy", "delivered"]).agg(
                n=("i", "size"), faith=("faith", "mean"), faith_exact=("faith_exact", "mean"),
                j5=("jaccard5_vs_exact", "mean")).reset_index().to_dict("records")
            # H4 equivalence: delivered explanations of the proposed trigger vs exact explanations of the same flows
            pr = fd[fd.policy == "Proposed"]
            if len(pr) >= 10:
                S[f"tost_H4_{tag}"] = stats.tost_paired(pr.faith.values, pr.faith_exact.values, 0.05)


# ------------------------------------------------------------------------------------------ ablations / robustness
def t_extra():
    p = RESULTS / "ablation_extra.csv"
    if p.exists():
        d = pd.read_csv(p)
        rows = [[DS[r.dataset], esc(r.ablation), esc(r.variant), f(r.get("f1")) if "f1" in d else "--",
                 f(r.faith), f(r.aopc_gain), f(r.max_sensitivity, 2), f(r.mean_ms, 1)] for _, r in d.iterrows()]
        write("tab_ablation_extra", table(r"Ablations A8 (feature groups within T3), A10 (grouped vs independent attributions) and A11 (KernelSHAP background size)",
                                          "tab:ablextra", "lp{2.8cm}p{4.2cm}rrrrr",
                                          ["Dataset", "Ablation", "Variant", "F1", "Faith.", r"AOPC$^+$", "Max-sens.", "ms"], rows,
                                          size=r"\scriptsize"))
    p = RESULTS / "robustness_perturbation.csv"
    if p.exists():
        d = pd.read_csv(p)
        lv = [0.0, 0.05, 0.25, 0.5]
        rows = []
        for (ds, mo, kind), g in d.groupby(["dataset", "model", "perturbation"], sort=False):
            g = g.set_index("level")
            rows.append([DS[ds], mo, kind.replace("_", " ")] + [f(g.f1.get(a)) for a in lv]
                        + [f(g.fpr.get(a), 4) for a in (0.0, 0.5)] + [f(g.expl_jaccard5.get(a), 2) for a in lv[1:]])
        S["robustness"] = d.to_dict("records")
        write("tab_robustness", table(
            r"Robustness on T3: each test flow's timing features are scaled by $U(1-a, 1+a)$ (jitter) or its size features by $U(1, 1+a)$ (padding); thresholds stay fixed. J@5 compares explanations of the same 200 flows before and after perturbation (XGBoost: interventional TreeSHAP; MLP: IG-32).",
            "tab:robust", "lllrrrrrrrrr",
            ["Dataset", "Model", "Perturbation", "F1 $a$=0", "0.05", "0.25", "0.5", "FPR $a$=0", "0.5", "J@5 0.05", "0.25", "0.5"],
            rows, size=r"\scriptsize"))
    p = RESULTS / "robustness_timing_repeat.csv"
    if p.exists():
        d = pd.read_csv(p)
        s = d.groupby("config").mean_ms.agg(["mean", "std"]).reset_index()
        s["cv"] = s["std"] / s["mean"]
        S["timing_repeat"] = s.to_dict("records")
        def cfg(c):          # "cic2017|T3|XGB|KernelSHAP|500" -> dataset, tier, model, explainer-budget
            ds, tier, mo, me, bu = c.split("|")
            return [DS.get(ds, ds), tier, mo, esc(me if bu in ("None", "nan", "") else f"{me}-{bu}")]
        rows = [cfg(r.config) + [f(r["mean"], 2), f(r["std"], 2), f(100 * r.cv, 1) + r"\%"] for _, r in s.iterrows()]
        write("tab_timing_repeat", table(r"Timing repeatability: three fresh-process repetitions per configuration",
                                         "tab:timingrep", "llllrrr", ["Dataset", "Tier", "Model", "Explainer", "mean ms",
                                                                      "sd ms", "CV"], rows, size=r"\scriptsize"))


# ------------------------------------------------------------------------------------------ cost vs fidelity
def t_cost_fidelity():
    """Spearman association between explanation cost and faithfulness across T3 configurations
    (Phase 2, Step 9: does paying more buy a more faithful explanation?)."""
    e = load_e1()
    fid = pd.read_csv(RESULTS / "e2_fidelity.csv")
    fid["label"] = [lab(m, b) for m, b in zip(fid.method, fid.budget)]
    m = e.merge(fid, on=["dataset", "tier", "model", "label"], suffixes=("", "_f"))
    m = m[m.tier == "T3"]
    out = {}
    for key, g in [("all", m)] + [(ds, g) for ds, g in m.groupby("dataset")]:
        if len(g) >= 5:
            from scipy.stats import spearmanr
            r, p = spearmanr(np.log10(g.p50_ms), g.faith_median)
            out[key] = {"rho": float(r), "p": float(p), "n": int(len(g))}
    S["cost_fidelity_spearman_T3"] = out


# ------------------------------------------------------------------------------------------ A6 parameter sensitivity
def t_sensitivity():
    names = {"default": "Proposed (defaults)", "band_frac": r"uncertainty band (share of val.\ flows)",
             "novelty_q": r"novelty quantile $\kappa$", "rho": r"audit rate $\rho$"}
    for p in sorted(RESULTS.glob("e4sens_*.csv")):
        d = pd.read_csv(p)
        tag = p.stem.replace("e4sens_", "")
        ds = d.dataset.iloc[0]
        rows = []
        for scen in ("mixed", "burst"):
            if rows:
                rows.append("MID")
            for i, r in enumerate(d[d.scenario == scen].to_dict("records")):
                val = "--" if r["param"] == "default" else f"{r['value']:g}"
                rows.append([scen if i == 0 else "", names.get(r["param"], esc(r["param"])), val, f(r["trigger_rate"]),
                             f(r["O_xai_sync"]), f(r["O_async"], 2), f(r["coverage_fast@1s"]),
                             f(r["coverage_fast_incl_deferred@1s"]), f(r["coverage_deep@30s"]), f(r["verdict_latency_p99_ms"], 1)])
        write(f"tab_sensitivity_{tag}", table(
            rf"Ablation A6 (rule parameters) on {DS[ds]} at 75\% load: one parameter changed at a time on identical streams",
            f"tab:sens:{tag}", "llrrrrrrrr", ["Traffic", "Parameter", "Value", r"Expl.\,rate", r"$O_{\mathrm{sync}}$", r"$O_{\mathrm{async}}$",
                                               r"Cov.\,sync", "+def.", "Deep", "p99 ms"], rows, size=r"\scriptsize"))


# ------------------------------------------------------------------------------------------ compact report tables
def side_by_side(caption, label, colspec, header, parts, size=r"\scriptsize", sep="2.5pt", width=0.495):
    """One table float with two sub-tables next to each other. parts = [(title, rows), (title, rows)]."""
    n = len(header)
    head = " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\"
    out = [r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", size,
           rf"\setlength{{\tabcolsep}}{{{sep}}}"]
    for k, (title, rows) in enumerate(parts):
        out += [rf"\begin{{minipage}}[t]{{{width}\textwidth}}", r"\centering", rf"\begin{{tabular}}{{{colspec}}}",
                r"\toprule", rf"\multicolumn{{{n}}}{{c}}{{\textbf{{{title}}}}} \\", r"\midrule", head, r"\midrule"]
        out += [r"\midrule" if r == "MID" else " & ".join(str(c) for c in r) + r" \\" for r in rows]
        out += [r"\bottomrule", r"\end{tabular}", r"\end{minipage}"]
        if k == 0:
            out.append(r"\hfill")
    out += [r"\end{table}", ""]
    return "\n".join(out)


def t_compact():
    """Short, space-saving tables used by the final Phase 3 report."""
    det = load_detection()
    short_ds = {"cic2017": "CIC-IDS2017", "hikari": "HIKARI-2021"}

    # 1) detection: CIC | HIKARI
    parts = []
    for ds in ("cic2017", "hikari"):
        rows = []
        for tier in ("T1", "T3"):
            if rows:
                rows.append("MID")
            for m in ("LR", "EBM", "RF", "XGB", "MLP"):
                r = det[(det.dataset == ds) & (det.tier == tier) & (det.model == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append([tier, m, f(r.precision), f(r.recall), f(r.f1), f(100 * r.fpr, 1), f(r.roc_auc), f(r.pr_auc)])
        parts.append((short_ds[ds], rows))
    write("c_detection", side_by_side(
        r"Detection results on the test set ($T_1$ = all features, $T_3$ = encrypted-traffic view). Alert threshold set on validation data for at most 1\% false positives; FPR in \%.",
        "tab:det", "llrrrrrr", ["Tier", "Model", "Prec.", "Recall", "F1", "FPR", "ROC", "PR"], parts))

    # 2) cost + fidelity in one table (T3)
    e = load_e1()
    e = e[e.tier == "T3"]
    fid = pd.read_csv(RESULTS / "e2_fidelity.csv")
    fid["label"] = [lab(m, b) for m, b in zip(fid.method, fid.budget)]
    fid = fid[fid.tier == "T3"]
    rows = []
    for m in ("LR", "EBM", "XGB", "RF", "MLP"):
        sub = e[(e.model == m) & (e.dataset == "cic2017")].sort_values("p50_ms")
        if sub.empty:
            continue
        if rows:
            rows.append("MID")
        first = True
        for lbl in sub.label:
            row = [m if first else "", esc(lbl)]
            for ds in ("cic2017", "hikari"):
                c = e[(e.dataset == ds) & (e.model == m) & (e.label == lbl)]
                q = fid[(fid.dataset == ds) & (fid.model == m) & (fid.label == lbl)]
                dl = det[(det.dataset == ds) & (det.tier == "T3") & (det.model == m)].lat_p50_ms
                dl = float(dl.iloc[0]) if len(dl) else np.nan
                if c.empty:
                    row += ["--"] * 4
                    continue
                c = c.iloc[0]
                row += [f(c.p50_ms, 3 if c.p50_ms < 0.1 else 2), f(c.p50_ms / dl, 1) + r"$\times$",
                        f(float(q.faith_median.iloc[0]), 2) if len(q) else "--",
                        f(float(q.infid_norm.iloc[0]), 2) if len(q) and "infid_norm" in q else "--"]
            rows.append(row)
            first = False
    body = [r"\begin{table}[!htbp]", r"\centering",
            r"\caption{Cost and quality of every explainer on the encrypted-traffic view $T_3$ (one CPU core). "
            r"ms = median time per explanation; $\times$ = explanation time divided by the detector's own time; "
            r"Faith.\ = faithfulness (higher is better); Infid.\ = infidelity (lower is better).}",
            r"\label{tab:costfid}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{llrrrr@{\hspace{10pt}}rrrr}", r"\toprule",
            r" & & \multicolumn{4}{c}{\textbf{CIC-IDS2017}} & \multicolumn{4}{c}{\textbf{HIKARI-2021}} \\",
            r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
            r"\textbf{Model} & \textbf{Explainer} & \textbf{ms} & $\boldsymbol{\times}$ & \textbf{Faith.} & \textbf{Infid.} & "
            r"\textbf{ms} & $\boldsymbol{\times}$ & \textbf{Faith.} & \textbf{Infid.} \\", r"\midrule"]
    body += [r"\midrule" if r == "MID" else " & ".join(str(c) for c in r) + r" \\" for r in rows]
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("c_costfid", "\n".join(body))

    # 3) T1 vs T3 faithfulness tests: CIC | HIKARI
    t = pd.read_csv(RESULTS / "stats_tier_wilcoxon.csv")
    parts = []
    for ds in ("cic2017", "hikari"):
        rows = [[r.model, esc(r.explainer), f(r.faith_T1, 2), f(r.faith_T3, 2), f(r.faith_T3 - r.faith_T1, 2),
                 f"{r.p_holm:.0e}" if r.p_holm < 0.01 else f(r.p_holm, 2)] for r in t[t.dataset == ds].itertuples()]
        parts.append((short_ds[ds], rows))
    write("c_tiertests", side_by_side(
        r"Faithfulness with all features ($T_1$) and with the encrypted-traffic view ($T_3$) on the same flows. $p$ = paired Wilcoxon test, Holm-corrected.",
        "tab:tier", "llrrrr", ["Model", "Explainer", "$T_1$", "$T_3$", r"$\Delta$", "$p$"], parts))

    # 4) streaming policies and 5) trigger ablations, CIC-IDS2017 XGB T3 at 75 % load: mixed | burst
    d = pd.read_csv(RESULTS / "e4_cic2017_XGB_T3.csv")
    d = d[d.load == 0.75]
    names = {"B1 Explain-none": "B1 explain none", "B2a Explain-all (fast)": "B2a all, TreeSHAP",
             "B2b Explain-all (deep)": "B2b all, KernelSHAP", "B3 Static global": "B3 one global expl.",
             "B4 Random (matched rate)": "B4 random", "B5 Alert-only": "B5 every alert",
             "B6 Amortised (all flows)": "B6 learned explainer", "Proposed": r"\textbf{Proposed}"}
    parts = []
    for scen, title in (("mixed", "Normal (mixed) traffic"), ("burst", "With port-scan flood")):
        rows = []
        for pol in POL:
            r = d[(d.scenario == scen) & (d.policy == pol)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([names[pol], f(r.O_xai_sync, 2), f(r["coverage_fast_incl_deferred@1s"], 2),
                         f(r.verdict_latency_p99_ms / 1000, 2) if r.verdict_latency_p99_ms >= 1000 else f(r.verdict_latency_p99_ms / 1000, 3)])
        parts.append((title, rows))
    write("c_stream", side_by_side(
        r"Explanation policies on CIC-IDS2017 at 75\% load. Overhead = explanation work as a share of detection time; "
        r"Covered = share of true alerts with an exact explanation within 1\,s; Delay = 99th-percentile time to a verdict in seconds.",
        "tab:stream", "lrrr", ["Policy", "Overhead", "Covered", "Delay (s)"], parts, sep="4pt"))

    vnames = {"Proposed": r"\textbf{Full trigger}", "A1 no uncertainty rule": "A1 no uncertainty rule",
              "A2 no novelty rule": "A2 no novelty rule", "A3 no cache": "A3 no cache",
              "A4 no budget guard": "A4 no budget limit", "A5a fast-only (no deep tier)": "A5a no deep lane",
              "A5b deep-only (sync deep)": "A5b KernelSHAP only", "A6 beta=0.02": r"A6 budget 2\%",
              "A6 beta=0.05": r"A6 budget 5\%", "A6 beta=0.2": r"A6 budget 20\%"}
    order = ["Proposed", "A1 no uncertainty rule", "A2 no novelty rule", "A3 no cache", "A4 no budget guard",
             "A5a fast-only (no deep tier)", "A5b deep-only (sync deep)", "A6 beta=0.02", "A6 beta=0.05", "A6 beta=0.2"]
    parts = []
    for scen, title in (("mixed", "Normal (mixed) traffic"), ("burst", "With port-scan flood")):
        rows = []
        for pol in order:
            r = d[(d.scenario == scen) & (d.policy == pol)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([vnames[pol], f(r.O_xai_sync, 2), f(r["coverage_fast@1s"], 2),
                         f(r["coverage_deep@30s"], 2) if r["coverage_deep@30s"] == r["coverage_deep@30s"] else "--",
                         f(r.verdict_latency_p99_ms, 0)])
        parts.append((title, rows))
    write("c_ablation", side_by_side(
        r"Trigger ablations on CIC-IDS2017 at 75\% load (each row removes or changes one part). Instant = share of alerts "
        r"explained on the detection path within 1\,s; Deep = share of KernelSHAP requests finished within 30\,s; Delay in ms.",
        "tab:abl", "lrrrr", ["Variant", "Overhead", "Instant", "Deep", "Delay"], parts, sep="3pt"))

    # 6) A8 / A10 / A11: CIC | HIKARI
    a = pd.read_csv(RESULTS / "ablation_extra.csv")
    vn = {"timing": "A8 timing only", "size": "A8 sizes only", "volume_direction": "A8 volume/direction only",
          "all_T3": r"A8 all $T_3$ features", "KernelSHAP-500, k=10": "A11 background 10",
          "KernelSHAP-500, k=50": "A11 background 50", "KernelSHAP-500, k=100": "A11 background 100",
          "independent KernelSHAP-500": "A10 per feature (KernelSHAP)",
          "grouped PartitionSHAP-500 (Owen)": "A10 grouped (PartitionSHAP)"}
    parts = []
    for ds in ("cic2017", "hikari"):
        rows = [[vn.get(r.variant, esc(r.variant)), f(r.f1) if r.f1 == r.f1 else "--", f(r.faith, 2), f(r.mean_ms, 1)]
                for r in a[a.dataset == ds].itertuples()]
        parts.append((short_ds[ds], rows))
    # 7) known-answer sanity test: XGB | MLP
    p = RESULTS / "sanity_planted.csv"
    if p.exists():
        s = pd.read_csv(p)
        S["sanity_planted"] = s.to_dict("records")
        sparts = []                       # separate name: `parts` still holds the A8/A10/A11 table below
        for mo in ("XGB", "MLP"):
            g = s[s.model == mo]
            srows = [[esc(r.explainer), f(r.hit_top1_A, 2), f(r.mass_share_A, 2), f(r.false_top1_B, 2)] for r in g.itertuples()]
            sparts.append((("XGBoost" if mo == "XGB" else "MLP"), srows))
        write("c_sanity", side_by_side(
            r"Known-answer test (CIC-IDS2017, $T_3$). Found = share of the 100 flows that are attacks only because of the planted "
            r"feature where it is ranked first (should be 1); Weight = its share of the explanation for those flows; "
            r"Wrong = share of 100 real attacks where the irrelevant planted feature is ranked first (should be 0).",
            "tab:sanity", "lrrr", ["Explainer", "Found", "Weight", "Wrong"], sparts, sep="5pt"))

    write("c_extra", side_by_side(
        r"Ablations of the features and the explainer (XGBoost, $T_3$, 100 flows): A8 trains on one feature group, "
        r"A10 explains groups of related features, A11 changes the KernelSHAP background size.",
        "tab:extra", "lrrr", ["Variant", "F1", "Faith.", "ms"], parts, sep="4pt"))


if __name__ == "__main__":
    for fn in (t_data, t_detection, t_cost, t_fidelity, t_cost_fidelity, t_drift, t_stream, t_sensitivity, t_extra,
               t_compact):
        try:
            fn()
        except FileNotFoundError as e:
            print("skip", fn.__name__, e)
    (RESULTS / "stats.json").write_text(json.dumps(S, indent=1, default=float))
    print("stats.json written")
