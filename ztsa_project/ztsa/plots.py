"""Figures and LaTeX tables from the frozen results.

    python -m ztsa.plots --paper ../paper/ztsa_ieee
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
COL = {"additive & static RL": "#8c8c8c", "additive & dynamic RL": "#4d4d4d",
       "SL & static RL": "#7fa7d8", "SL & dynamic RL": "#1f5aa6", "ZT-CMS (ours)": "#d1495b"}
SHORT = {"Additive-static": "Add-S", "Additive-dynamic": "Add-D", "SL-static": "SL-S", "SL-dynamic": "SL-D",
         "SL-dynamic+CV": "+CV", "SL-dynamic+MS": "+MS", "SL-dynamic+LP": "+LP",
         "SL-dynamic+CV+MS": "+CV+MS", "ZT-CMS (full)": "ZT-CMS"}
ATT = ["credential_theft", "compromised_device", "lateral_movement", "slow_exfiltration"]
ATT_LABEL = ["Credential\ntheft", "Compromised\ndevice", "Lateral\nmovement", "Slow\nexfiltration"]


def load_perf():
    rows = list(csv.DictReader(open(ROOT / "results/perf/decision_time.csv")))
    agg = defaultdict(list)
    for r in rows:
        agg[(r["cache"] == "True", r["config"], int(r["concurrency"]))].append(
            (float(r["median_ms"]), float(r["q1_ms"]), float(r["q3_ms"])))
    # median over repetitions, as in the paper
    return {k: tuple(np.median(np.array(v), axis=0)) for k, v in agg.items()}


def fig_decision_time(perf, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    conc = sorted({k[2] for k in perf})
    for ax, cache, title in zip(axes, (False, True), ("(a) PIP queries DB (paper Fig. 2)",
                                                     "(b) PIP serves from cache (paper Fig. 3)")):
        for cfg, c in COL.items():
            y = [perf[(cache, cfg, n)][0] for n in conc]
            lo = [perf[(cache, cfg, n)][1] for n in conc]
            hi = [perf[(cache, cfg, n)][2] for n in conc]
            ls = "--" if "static" in cfg else "-"
            ax.plot(range(len(conc)), y, ls, color=c, marker="o", ms=2.5, lw=1.1, label=cfg)
            ax.fill_between(range(len(conc)), lo, hi, color=c, alpha=0.08)
        ax.set_xticks(range(len(conc)), conc)
        ax.set_xlabel("Concurrent decision requests (PEP instances)")
        ax.set_ylabel("Decision-making time (ms)")
        ax.set_title(title, fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(fontsize=6.2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "decision_time.pdf")
    plt.close(fig)


def fig_accuracy(s, out):
    res = s["results"]["calibrated_1pct"]
    names = list(res)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for ax, key, label in zip(axes, ("auc", "malicious_recall", "episode_detection"),
                              ("ROC AUC", "Malicious-request recall @1% benign denial",
                               "Attack episodes detected @1%")):
        m = [res[n][key][0] for n in names]
        sd = [res[n][key][1] for n in names]
        colors = ["#8c8c8c"] * 2 + ["#1f5aa6"] * 2 + ["#e6a1aa"] * 4 + ["#d1495b"]
        ax.bar(range(len(names)), m, yerr=sd, color=colors, capsize=1.5, error_kw={"lw": 0.6})
        ax.set_xticks(range(len(names)), [SHORT[n] for n in names], rotation=60, ha="right", fontsize=6.5)
        ax.set_title(label, fontsize=7.5)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        if key == "auc":
            ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(out / "accuracy.pdf")
    plt.close(fig)


def fig_attacks(s, out):
    res = s["results"]["calibrated_1pct"]
    sel = ["Additive-dynamic", "SL-dynamic", "ZT-CMS (full)"]
    colors = ["#8c8c8c", "#1f5aa6", "#d1495b"]
    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    w = 0.26
    for i, (n, c) in enumerate(zip(sel, colors)):
        v = [res[n][f"recall_{a}"][0] for a in ATT]
        e = [res[n][f"recall_{a}"][1] for a in ATT]
        ax.bar(np.arange(4) + (i - 1) * w, v, w, yerr=e, color=c, label=SHORT[n], capsize=1.2,
               error_kw={"lw": 0.5})
    ax.set_xticks(range(4), ATT_LABEL, fontsize=6.5)
    ax.set_ylabel("Recall @1% benign denial")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out / "attacks.pdf")
    plt.close(fig)


def tables(perf, s, out):
    # ---- Table: paper vs reproduction at 64 concurrent requests
    paper = {(False, "additive & static RL"): 296, (False, "SL & static RL"): 294,
             (False, "additive & dynamic RL"): 317, (False, "SL & dynamic RL"): 316,
             (True, "additive & static RL"): 126, (True, "SL & static RL"): 126,
             (True, "additive & dynamic RL"): 133, (True, "SL & dynamic RL"): 133}
    lines = [r"\begin{table}[t]\caption{Decision time at 64 concurrent requests: paper vs.\ our reproduction}"
             r"\label{tab:repro}\centering\small\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{llrr}\toprule", r"PIP & Configuration & Paper (ms) & Ours (ms)\\\midrule"]
    for cache in (False, True):
        for cfg in ("additive & static RL", "SL & static RL", "additive & dynamic RL", "SL & dynamic RL",
                    "ZT-CMS (ours)"):
            p = paper.get((cache, cfg))
            lines.append(f"{'Cache' if cache else 'DB'} & {cfg.replace('&', chr(92) + '&')} & "
                         f"{p if p else '--'} & {perf[(cache, cfg, 64)][0]:.2f}\\\\")
        if not cache:
            lines.append(r"\midrule")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (out / "repro_table.tex").write_text("\n".join(lines) + "\n")

    # ---- Table: claims check
    def ratio(cache, a, b):
        return perf[(cache, a, 64)][0] / perf[(cache, b, 64)][0]
    claims = {
        "sl_vs_add_static_db": ratio(False, "SL & static RL", "additive & static RL"),
        "sl_vs_add_static_cache": ratio(True, "SL & static RL", "additive & static RL"),
        "dyn_vs_static_add_db": ratio(False, "additive & dynamic RL", "additive & static RL"),
        "dyn_vs_static_sl_db": ratio(False, "SL & dynamic RL", "SL & static RL"),
        "dyn_vs_static_add_cache": ratio(True, "additive & dynamic RL", "additive & static RL"),
        "dyn_vs_static_sl_cache": ratio(True, "SL & dynamic RL", "SL & static RL"),
        "cache_vs_db_sl_static": perf[(True, "SL & static RL", 64)][0] / perf[(False, "SL & static RL", 64)][0],
        "cache_vs_db_sl_dynamic": perf[(True, "SL & dynamic RL", 64)][0] / perf[(False, "SL & dynamic RL", 64)][0],
        "ztcms_vs_sl_dynamic_db": ratio(False, "ZT-CMS (ours)", "SL & dynamic RL"),
        "ztcms_vs_sl_dynamic_cache": ratio(True, "ZT-CMS (ours)", "SL & dynamic RL"),
    }
    json.dump(claims, open(out / "claims.json", "w"), indent=2)

    # ---- Table: accuracy
    res = s["results"]["calibrated_1pct"]
    pr = s["results"]["paper_rule"]
    lines = [r"\begin{table*}[t]\caption{Decision accuracy on five unseen test enterprises "
             r"(mean $\pm$ s.d.). Calibrated: threshold fixed on a separate enterprise to deny 1\% of benign requests.}"
             r"\label{tab:acc}\centering\small\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lcccccccc}\toprule",
             r" & \multicolumn{2}{c}{Paper rule (trust exceeds risk)} & \multicolumn{6}{c}{Calibrated (1\% benign denial target)}\\"
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-9}",
             r"Algorithm & Recall & Benign deny & AUC & Recall & Benign deny & F1 & Episodes det. & Exposure\\\midrule"]
    for n in res:
        f = lambda d, k, p=3: f"{d[n][k][0]:.{p}f}"  # noqa: E731
        name = n.replace("SL-dynamic+", "SL-dyn + ").replace("ZT-CMS (full)", r"\textbf{ZT-CMS (ours)}")
        lines.append(f"{name} & {f(pr,'malicious_recall')} & {f(pr,'benign_deny')} & "
                     f"{f(res,'auc')}$\\pm${res[n]['auc'][1]:.3f} & {f(res,'malicious_recall')} & "
                     f"{f(res,'benign_deny')} & {f(res,'f1')} & {f(res,'episode_detection')} & "
                     f"{f(res,'mean_exposure',1)}\\\\")
        if n in ("Additive-dynamic", "SL-dynamic", "SL-dynamic+CV+MS"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule\end{tabular}\end{table*}"]
    (out / "accuracy_table.tex").write_text("\n".join(lines) + "\n")


def numbers(perf, s, out):
    """Every number quoted in the paper text, as LaTeX macros."""
    m = {}
    tag = {"additive & static RL": "AddS", "additive & dynamic RL": "AddD", "SL & static RL": "SLS",
           "SL & dynamic RL": "SLD", "ZT-CMS (ours)": "Z"}
    for cache, c in ((False, "DB"), (True, "C")):
        for cfg, t in tag.items():
            m[f"T{t}{c}"] = f"{perf[(cache, cfg, 64)][0]:.2f}"
            m[f"Tone{t}{c}"] = f"{perf[(cache, cfg, 1)][0]:.2f}"
            m[f"Tmax{t}{c}"] = f"{perf[(cache, cfg, 128)][0]:.2f}"
    pct = lambda a, b: f"{100 * (a / b - 1):.1f}"  # noqa: E731
    g = lambda cache, cfg: perf[(cache, cfg, 64)][0]  # noqa: E731
    for cache, c in ((False, "DB"), (True, "C")):
        m[f"DynAdd{c}"] = pct(g(cache, "additive & dynamic RL"), g(cache, "additive & static RL"))
        m[f"DynSL{c}"] = pct(g(cache, "SL & dynamic RL"), g(cache, "SL & static RL"))
        m[f"SLvsAddS{c}"] = pct(g(cache, "SL & static RL"), g(cache, "additive & static RL"))
        m[f"SLvsAddD{c}"] = pct(g(cache, "SL & dynamic RL"), g(cache, "additive & dynamic RL"))
        m[f"ZvsSLD{c}"] = pct(g(cache, "ZT-CMS (ours)"), g(cache, "SL & dynamic RL"))
    m["CacheSaveS"] = f"{100 * (1 - g(True, 'SL & static RL') / g(False, 'SL & static RL')):.1f}"
    m["CacheSaveD"] = f"{100 * (1 - g(True, 'SL & dynamic RL') / g(False, 'SL & dynamic RL')):.1f}"
    m["CacheSaveAddS"] = f"{100 * (1 - g(True, 'additive & static RL') / g(False, 'additive & static RL')):.1f}"
    g1 = lambda cache, cfg: perf[(cache, cfg, 1)][0]  # noqa: E731
    m["CacheSaveOneS"] = f"{100 * (1 - g1(True, 'SL & static RL') / g1(False, 'SL & static RL')):.0f}"
    m["CacheSaveOneAddS"] = f"{100 * (1 - g1(True, 'additive & static RL') / g1(False, 'additive & static RL')):.0f}"

    res, pr = s["results"]["calibrated_1pct"], s["results"]["paper_rule"]
    t2 = {"Additive-static": "AddS", "Additive-dynamic": "AddD", "SL-static": "SLS", "SL-dynamic": "SLD",
          "SL-dynamic+CV": "CV", "SL-dynamic+MS": "MS", "SL-dynamic+LP": "LP", "SL-dynamic+CV+MS": "CVMS",
          "ZT-CMS (full)": "Z"}
    for n, t in t2.items():
        r = res[n]
        m[f"Auc{t}"] = f"{r['auc'][0]:.3f}"
        m[f"Rec{t}"] = f"{100 * r['malicious_recall'][0]:.1f}"
        m[f"Bd{t}"] = f"{100 * r['benign_deny'][0]:.2f}"
        m[f"Fone{t}"] = f"{r['f1'][0]:.3f}"
        m[f"Ep{t}"] = f"{100 * r['episode_detection'][0]:.1f}"
        m[f"Exp{t}"] = f"{r['mean_exposure'][0]:.1f}"
        m[f"Hi{t}"] = f"{r['hi_tier_allowed_per_episode'][0]:.1f}"
        m[f"PrRec{t}"] = f"{100 * pr[n]['malicious_recall'][0]:.1f}"
        m[f"PrBd{t}"] = f"{100 * pr[n]['benign_deny'][0]:.2f}"
        m[f"Us{t}"] = f"{s['compute_us_median'][n]:.0f}"
        for a, ta in zip(ATT, ("Cred", "Comp", "Lat", "Slow")):
            m[f"R{ta}{t}"] = f"{100 * r[f'recall_{a}'][0]:.1f}"
    p = s["paired_ztcms_vs_sl_dynamic"]
    for k, t in (("auc", "Auc"), ("malicious_recall", "Rec"), ("f1", "Fone"), ("episode_detection", "Ep"),
                 ("mean_exposure", "Exp")):
        sc = 100 if k in ("malicious_recall", "episode_detection") else 1
        nd = 1 if sc == 100 or k == "mean_exposure" else 3
        m[f"D{t}"] = f"{sc * p[k]['mean_diff']:.{nd}f}"
        m[f"D{t}Lo"] = f"{sc * p[k]['ci95'][0]:.{nd}f}"
        m[f"D{t}Hi"] = f"{sc * p[k]['ci95'][1]:.{nd}f}"
    cnt = s["counts"]
    m["NTest"] = f"{sum(c['test'] for c in cnt):,}"
    m["NMal"] = f"{sum(c['malicious'] for c in cnt):,}"
    m["NBen"] = f"{sum(c['test'] - c['malicious'] for c in cnt):,}"
    m["NEp"] = f"{sum(c['episodes'] for c in cnt):,}"
    m["NWarm"] = f"{sum(c['warmup'] for c in cnt):,}"
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (out / "numbers.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=str(ROOT.parent / "paper" / "ztsa_ieee"))
    a = ap.parse_args()
    out = Path(a.paper)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    perf = load_perf()
    s = json.load(open(ROOT / "results/accuracy/summary.json"))
    fig_decision_time(perf, out / "figures")
    fig_accuracy(s, out / "figures")
    fig_attacks(s, out / "figures")
    tables(perf, s, out)
    numbers(perf, s, out)
    print("figures and tables written to", out)


if __name__ == "__main__":
    main()
