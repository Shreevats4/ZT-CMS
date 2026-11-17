"""One-page visual summary for the presentation, plus a live attack trace.

    python -m ztsa.showcase

Runs the paper's algorithms (additive, SL) and ZT-CMS on an unseen test enterprise
(seed 201) with the calibrated thresholds from results/accuracy, replays one
lateral-movement attack request by request, and combines it with the stored
reproduction and accuracy results. Writes results/showcase/*.png.
"""
from __future__ import annotations

import copy
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .algorithms import ZTCMS, Additive, SubjectiveLogic  # noqa: E402
from .scenarios import SEGMENTS, Enterprise  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "showcase"
TEAL, RED, GREY, NAVY, GOLD = "#0E7C86", "#C8102E", "#9AA5AE", "#0F2A3D", "#D99A1E"
PAPER = {  # base paper Sec. IX, 64 concurrent requests (ms)
    (False, "additive & static RL"): 296, (False, "SL & static RL"): 294,
    (False, "additive & dynamic RL"): 317, (False, "SL & dynamic RL"): 316,
    (True, "additive & static RL"): 126, (True, "SL & static RL"): 126,
    (True, "additive & dynamic RL"): 133, (True, "SL & dynamic RL"): 133,
}
CFG_SHORT = {"additive & static RL": "Add.\nstatic", "SL & static RL": "SL\nstatic",
             "additive & dynamic RL": "Add.\ndynamic", "SL & dynamic RL": "SL\ndynamic"}
ATT = ["credential_theft", "compromised_device", "lateral_movement", "slow_exfiltration"]
ATT_LABEL = ["Credential\ntheft", "Compromised\ndevice", "Lateral\nmovement", "Slow\nexfiltration"]
SHOW = [("Additive-dynamic", "Additive (base paper)", GREY), ("SL-dynamic", "SL (base paper)", NAVY),
        ("ZT-CMS (full)", "ZT-CMS (ours)", RED)]


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11.5,
                         "axes.titlelocation": "left"})


def load_perf():
    agg = defaultdict(list)
    for r in csv.DictReader(open(ROOT / "results/perf/decision_time.csv")):
        agg[(r["cache"] == "True", r["config"], int(r["concurrency"]))].append(float(r["median_ms"]))
    return {k: float(np.median(v)) for k, v in agg.items()}


def panel_repro(ax, perf):
    cfgs = list(CFG_SHORT)
    x = np.arange(len(cfgs))
    w = 0.2
    bars = [("Paper, database", [PAPER[(False, c)] for c in cfgs], NAVY, None),
            ("Paper, cache", [PAPER[(True, c)] for c in cfgs], NAVY, "//"),
            ("Ours, database", [perf[(False, c, 64)] for c in cfgs], TEAL, None),
            ("Ours, cache", [perf[(True, c, 64)] for c in cfgs], TEAL, "//")]
    for i, (lab, ys, col, hatch) in enumerate(bars):
        b = ax.bar(x + (i - 1.5) * w, ys, w, label=lab, color=col if not hatch else "white",
                   edgecolor=col, hatch=hatch, lw=1.2)
        ax.bar_label(b, fmt="%.0f", fontsize=7.5, padding=1)
    ax.set_xticks(x, [CFG_SHORT[c] for c in cfgs])
    ax.set_ylabel("Decision time at 64 PEPs (ms)")
    ax.set_title("A. Reproduction: paper vs. our testbed")
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", handlelength=1.4, columnspacing=1)
    ax.set_ylim(0, 400)


def panel_scaling(ax, perf):
    conc = sorted({k[2] for k in perf})
    lines = [("additive & dynamic RL", "Additive (base paper)", GREY), ("SL & dynamic RL", "SL (base paper)", NAVY),
             ("ZT-CMS (ours)", "ZT-CMS (ours)", RED)]
    for cfg, lab, col in lines:
        ax.plot(range(len(conc)), [perf[(True, cfg, n)] for n in conc], "-o", color=col, ms=4, lw=1.8,
                label=lab + " (cache)")
        ax.plot(range(len(conc)), [perf[(False, cfg, n)] for n in conc], "--", color=col, lw=1.1,
                label=lab + " (DB)")
    ax.set_xticks(range(len(conc)), conc)
    ax.set_xlabel("Concurrent requests (PEP instances)")
    ax.set_ylabel("Median decision time (ms)")
    ax.set_title("B. Our testbed: decision time vs. load (dynamic risk)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5)


def panel_accuracy(ax, s):
    res = s["results"]["calibrated_1pct"]
    mets = [("auc", "AUC (x100)", 100), ("malicious_recall", "Malicious\nrequests denied", 100),
            ("episode_detection", "Attacks\ndetected", 100)]
    x = np.arange(len(mets))
    w = 0.26
    for i, (name, lab, col) in enumerate(SHOW):
        ys = [res[name][k][0] * sc for k, _, sc in mets]
        b = ax.bar(x + (i - 1) * w, ys, w, color=col, label=lab)
        ax.bar_label(b, labels=[f"{y/100:.2f}" if j == 0 else f"{y:.0f}%" for j, y in enumerate(ys)],
                     fontsize=8, padding=1)
    ax.set_xticks(x, [m[1] for m in mets])
    ax.set_title("C. Accuracy at the same 1% false-deny rate")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=3)
    ax.set_ylim(0, 112)


def panel_attacks(ax, s):
    res = s["results"]["calibrated_1pct"]
    x = np.arange(len(ATT))
    w = 0.26
    for i, (name, lab, col) in enumerate(SHOW):
        ys = [res[name][f"recall_{a}"][0] * 100 for a in ATT]
        b = ax.bar(x + (i - 1) * w, ys, w, color=col, label=lab)
        ax.bar_label(b, fmt="%.0f", fontsize=7.5, padding=1)
    ax.set_xticks(x, ATT_LABEL)
    ax.set_ylabel("Malicious requests denied (%)")
    ax.set_title("D. Per attack type")
    ax.set_ylim(0, 112)
    ax.legend(fontsize=8, frameon=False, loc="upper right")


def live_trace(s, seed=201):
    """Score one unseen enterprise and pick a lateral-movement episode to replay."""
    tau = s["calibration_tau"]
    warm, test = Enterprise(seed).generate()
    algos = {"SL-dynamic": SubjectiveLogic(True), "ZT-CMS (full)": ZTCMS(), "Additive-dynamic": Additive(True)}
    m = {}
    for name, a in algos.items():
        a = copy.deepcopy(a)
        if isinstance(a, ZTCMS):
            a.fit(warm)
        m[name] = np.array([a.margin(r) for r in test])
        print(f"  scored {len(test)} requests with {name}", flush=True)
    eps = defaultdict(list)
    for i, r in enumerate(test):
        if r["attack"] == "lateral_movement":
            eps[r["episode"]].append(i)

    def first_deny(name, idx):
        d = [i for i in idx if m[name][i] <= tau[name]]
        return idx.index(d[0]) if d else None

    best = None
    for ep, idx in eps.items():
        sl, zt, ad = (first_deny(n, idx) for n in ("SL-dynamic", "ZT-CMS (full)", "Additive-dynamic"))
        if zt is not None and sl is None and ad is None:
            key = (zt, -len(idx))
            if best is None or key < best[0]:
                best = (key, ep, idx)
    _, ep, idx = best
    user, t0 = test[idx[0]]["user"], test[idx[0]]["t"]
    ctx = [i for i, r in enumerate(test) if r["user"] == user and not r["malicious"] and r["t"] < t0][-10:]
    return test, m, tau, ep, ctx, idx


def panel_trace(ax, trace):
    test, m, tau, ep, ctx, idx = trace
    seq = ctx + idx
    xs = np.arange(len(seq))
    for name, lab, col in SHOW[1:]:
        y = np.array([m[name][i] - tau[name] for i in seq])
        ax.plot(xs, y, "-o", color=col, ms=4, lw=1.8, label=lab)
        den = y <= 0
        ax.scatter(xs[den], y[den], s=70, facecolor="none", edgecolor=col, lw=1.6, zorder=5)
    ax.axhline(0, color="#333", lw=1)
    ax.axvspan(len(ctx) - 0.5, len(seq) - 0.5, color=RED, alpha=0.06)
    tr = ax.get_xaxis_transform()
    ax.text(len(ctx) / 2 - 0.5, 1.01, "same user's normal work", ha="center", va="bottom", fontsize=9,
            color="#555", transform=tr)
    ax.text(len(ctx), 1.01, "attack: lateral movement across segments  ->", va="bottom", fontsize=9,
            color=RED, transform=tr)
    ax.text(-0.4, 0.015, "ALLOW above / DENY below", ha="left", va="bottom", fontsize=8, color="#333")
    names = [SEGMENTS[test[i]["segment"]][0] for i in seq]
    ax.set_xticks(xs, names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Trust minus risk\n(relative to threshold)")
    sl_den = sum(m["SL-dynamic"][i] <= tau["SL-dynamic"] for i in idx)
    zt_den = sum(m["ZT-CMS (full)"][i] <= tau["ZT-CMS (full)"] for i in idx)
    ax.set_title(f"E. Live replay of one unseen attack ({len(idx)} requests): base paper (SL and additive) "
                 f"denies {sl_den}, ZT-CMS denies {zt_den}  (circles = denied)", pad=18)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=0.25, lw=0.5)


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    s = json.load(open(ROOT / "results/accuracy/summary.json"))
    perf = load_perf()
    print("replaying an unseen enterprise (seed 201) ...", flush=True)
    trace = live_trace(s)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=0.6, wspace=0.18, top=0.93)
    panel_repro(fig.add_subplot(gs[0, 0]), perf)
    panel_scaling(fig.add_subplot(gs[0, 1]), perf)
    panel_accuracy(fig.add_subplot(gs[1, 0]), s)
    panel_attacks(fig.add_subplot(gs[1, 1]), s)
    panel_trace(fig.add_subplot(gs[2, :]), trace)
    fig.suptitle("ZT-CMS on top of Bradatsch et al. (IEEE TrustCom 2023): reproduction and novelty",
                 fontsize=15, fontweight="bold", color=NAVY, x=0.08, ha="left", y=0.975)
    fig.savefig(OUT / "dashboard.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    for fname, fn, args, size in [("A_reproduction.png", panel_repro, (perf,), (8, 4.5)),
                                  ("B_scaling.png", panel_scaling, (perf,), (8, 4.5)),
                                  ("C_accuracy.png", panel_accuracy, (s,), (8, 4.5)),
                                  ("D_attacks.png", panel_attacks, (s,), (8, 4.5)),
                                  ("E_live_attack.png", panel_trace, (trace,), (14, 5))]:
        fig, ax = plt.subplots(figsize=size)
        fn(ax, *args)
        fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
