"""Decision-accuracy experiment (the paper's stated future work, Sec. X).

All algorithms see identical labelled traffic. Two operating points:
  paper rule   tau = 0 (allow iff trust > risk, Eq. 3 / Eq. 5)
  calibrated   tau chosen per algorithm on a separate calibration enterprise so that
               exactly 1% of benign requests are denied; then applied unchanged to
               five unseen test enterprises.

    python -m ztsa.accuracy --out results/accuracy
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from .algorithms import ZTCMS, all_algorithms
from .scenarios import ATTACKS, Enterprise

ROOT = Path(__file__).resolve().parents[1]
CAL_SEED, TEST_SEEDS = 101, (201, 202, 203, 204, 205)
BENIGN_DENY_TARGET = 0.01


def score(algo, warm, test):
    if isinstance(algo, ZTCMS):
        algo.fit(warm)
    m = np.empty(len(test))
    t0 = time.perf_counter()
    for i, r in enumerate(test):
        m[i] = algo.margin(r)
    us = (time.perf_counter() - t0) / len(test) * 1e6
    return m, us


def auc(m, y):
    """ROC AUC of the denial score (-margin) for malicious vs benign (Mann-Whitney U)."""
    s = -m
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    npos, nneg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def metrics(m, tau, test):
    y = np.array([r["malicious"] for r in test])
    deny = m <= tau
    tp, fp = int((deny & y).sum()), int((deny & ~y).sum())
    fn = int((~deny & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {"benign_deny": fp / int((~y).sum()), "malicious_recall": rec, "precision": prec,
           "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    # episode view: detected if any request denied; exposure = allowed requests before first denial
    eps = {}
    for i, r in enumerate(test):
        if r["malicious"]:
            eps.setdefault(r["episode"], []).append(i)
    det, exposure, hi_allowed = [], [], []
    for idx in eps.values():
        d = [deny[i] for i in idx]
        det.append(any(d))
        exposure.append(d.index(True) if any(d) else len(d))
        hi_allowed.append(sum(1 for i in idx if not deny[i] and test[i]["tier"] >= 2))
    out.update(episode_detection=float(np.mean(det)), mean_exposure=float(np.mean(exposure)),
               hi_tier_allowed_per_episode=float(np.mean(hi_allowed)))
    for a in ATTACKS:
        mask = np.array([r["attack"] == a for r in test])
        out[f"recall_{a}"] = float(deny[mask].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/accuracy")
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=len(TEST_SEEDS))
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    algos = all_algorithms()
    names = [x.name for x in algos]

    warm, test = Enterprise(CAL_SEED, a.users, a.episodes).generate()
    y = np.array([r["malicious"] for r in test])
    tau = {}
    for algo in algos:
        m, _ = score(copy.deepcopy(algo), warm, test)
        tau[algo.name] = float(np.quantile(m[~y], BENIGN_DENY_TARGET))
        print(f"calibrated {algo.name:24} tau={tau[algo.name]:.4f}", flush=True)

    rows, timing, counts = [], {n: [] for n in names}, []
    for seed in TEST_SEEDS[:a.seeds]:
        warm, test = Enterprise(seed, a.users, a.episodes).generate()
        y = np.array([r["malicious"] for r in test])
        counts.append({"seed": seed, "warmup": len(warm), "test": len(test), "malicious": int(y.sum()),
                       "episodes": len({r["episode"] for r in test if r["malicious"]})})
        for algo in algos:
            m, us = score(copy.deepcopy(algo), warm, test)
            timing[algo.name].append(us)
            for point, t in (("paper_rule", 0.0), ("calibrated_1pct", tau[algo.name])):
                rows.append({"seed": seed, "algorithm": algo.name, "operating_point": point,
                             "auc": auc(m, y), **metrics(m, t, test)})
            print(f"seed {seed} {algo.name:24} AUC={rows[-1]['auc']:.4f} "
                  f"recall@1%={rows[-1]['malicious_recall']:.4f} benign_deny={rows[-1]['benign_deny']:.4f}", flush=True)

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # summary: mean and std over seeds; paired difference of ZT-CMS vs SL-dynamic
    keys = [k for k in rows[0] if k not in ("seed", "algorithm", "operating_point")]
    summary = {"calibration_tau": tau, "counts": counts,
               "compute_us_median": {n: float(np.median(v)) for n, v in timing.items()}, "results": {}}
    for point in ("paper_rule", "calibrated_1pct"):
        summary["results"][point] = {}
        for n in names:
            sel = [r for r in rows if r["algorithm"] == n and r["operating_point"] == point]
            summary["results"][point][n] = {k: [float(np.mean([r[k] for r in sel])), float(np.std([r[k] for r in sel]))]
                                            for k in keys}
    paired = {}
    for k in ("auc", "malicious_recall", "f1", "episode_detection", "mean_exposure"):
        base = [r[k] for r in rows if r["algorithm"] == "SL-dynamic" and r["operating_point"] == "calibrated_1pct"]
        ours = [r[k] for r in rows if r["algorithm"] == "ZT-CMS (full)" and r["operating_point"] == "calibrated_1pct"]
        d = np.array(ours) - np.array(base)
        half = 2.776 * d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")  # t(0.975, 4)
        paired[k] = {"mean_diff": float(d.mean()), "ci95": [float(d.mean() - half), float(d.mean() + half)]}
    summary["paired_ztcms_vs_sl_dynamic"] = paired
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
