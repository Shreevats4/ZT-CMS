"""Reproduce the paper's Sec. IX decision-time experiment (Figs. 2 and 3).

Paper: 1..128 parallel PEP instances x 1000 requests, 10 repetitions, median and
quartiles of the PEP-measured decision time, PIP without cache (Fig. 2) and with
cache (Fig. 3), additive vs SL, static vs dynamic risk level. We run the same
grid (plus our ZT-CMS algorithm) on one machine; requests per instance and
repetitions are reduced and stated in the output manifest.

    python -m ztsa.perf --out results/perf [--reqs 50 --reps 3]
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .services import CERTS, PG_PORT, ROOT, Client

RUNTIME = ROOT / "runtime"
PG_BIN = Path(os.environ.get("ZTSA_PG_BIN", r"C:\Program Files\PostgreSQL\16\bin"))
PORTS = {"pap": 56101, "pip": 56102, "pdp": 56103}
CONCURRENCY = [int(x) for x in os.environ.get("ZTSA_CONC", "1,2,4,8,16,32,64,128").split(",")]
CONFIGS = [  # (label, policy_set)
    ("additive & static RL", "add-static"), ("additive & dynamic RL", "add-dynamic"),
    ("SL & static RL", "sl-static"), ("SL & dynamic RL", "sl-dynamic"),
    ("ZT-CMS (ours)", "ztcms"),
]


def make_certs():
    if (CERTS / "pep.crt").exists():
        return
    CERTS.mkdir(parents=True, exist_ok=True)
    ossl = shutil.which("openssl") or r"C:\Program Files\Git\mingw64\bin\openssl.exe"
    run = lambda *a: subprocess.run([ossl, *a], cwd=CERTS, check=True, capture_output=True)  # noqa: E731
    run("req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes", "-days", "365",
        "-keyout", "ca.key", "-out", "ca.crt", "-subj", "/CN=ztsa-ca")
    ext = CERTS / "ext.cnf"
    ext.write_text("subjectAltName=DNS:localhost,IP:127.0.0.1\nextendedKeyUsage=serverAuth,clientAuth\n")
    for n in ("pep", "pdp", "pip", "pap"):
        run("req", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes",
            "-keyout", f"{n}.key", "-out", f"{n}.csr", "-subj", f"/CN={n}")
        run("x509", "-req", "-in", f"{n}.csr", "-CA", "ca.crt", "-CAkey", "ca.key", "-CAcreateserial",
            "-days", "365", "-out", f"{n}.crt", "-extfile", "ext.cnf")


def start_postgres():
    data = RUNTIME / "pgdata"
    if not data.exists():
        subprocess.run([PG_BIN / "initdb", "-D", data, "-U", "ztsa", "-A", "trust", "--encoding=UTF8",
                        "--locale=C"], check=True, capture_output=True)
    st = subprocess.run([PG_BIN / "pg_ctl", "-D", data, "status"], capture_output=True)
    if st.returncode != 0:
        # no captured pipes: the server inherits them on Windows and would block run()
        subprocess.run([PG_BIN / "pg_ctl", "-D", data, "-l", RUNTIME / "postgres.log", "-w", "-o",
                        f"-p {PG_PORT} -c max_connections=400 -c listen_addresses=127.0.0.1", "start"],
                       check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([PG_BIN / "createdb", "-h", "127.0.0.1", "-p", str(PG_PORT), "-U", "ztsa", "ztsa"],
                   capture_output=True)
    subprocess.run([sys.executable, "-m", "ztsa.services", "seed"], cwd=ROOT, check=True)


def stop_postgres():
    subprocess.run([PG_BIN / "pg_ctl", "-D", RUNTIME / "pgdata", "-m", "fast", "stop"], capture_output=True)


class Stack:
    def __init__(self, cache: bool):
        self.cache, self.procs = cache, []

    def __enter__(self):
        py = [sys.executable, "-m", "ztsa.services"]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.procs.append(subprocess.Popen(py + ["pap", "--port", str(PORTS["pap"])], cwd=ROOT, creationflags=flags))
        self.procs.append(subprocess.Popen(py + ["pip", "--port", str(PORTS["pip"])] + (["--cache"] if self.cache else []),
                                           cwd=ROOT, creationflags=flags))
        self.procs.append(subprocess.Popen(py + ["pdp", "--port", str(PORTS["pdp"]), "--pip", str(PORTS["pip"]),
                                                 "--pap", str(PORTS["pap"])], cwd=ROOT, creationflags=flags))
        c = Client("pep", PORTS["pdp"])
        for _ in range(100):
            try:
                c.post("/decide", _query("add-static", 0, 0))
                return self
            except OSError:
                time.sleep(0.2)
        raise RuntimeError("stack did not start")

    def __exit__(self, *a):
        for p in self.procs:
            p.kill()
        for p in self.procs:
            p.wait()


def _query(policy_set, i, k):
    return {"policy_set": policy_set, "user": f"u{(i * 7 + k) % 200}", "device": f"d{(i * 7 + k) % 200}",
            "resource": f"r{(i + k) % 10}", "t": float(k)}


def _worker(args):
    """One load-generator process running `threads` PEP instances."""
    policy_set, first, threads, reqs, start_at = args
    client = Client("pep", PORTS["pdp"])
    lat = [[] for _ in range(threads)]
    barrier = threading.Barrier(threads)

    def pep(j):
        client.post("/decide", _query(policy_set, first + j, -1))  # establish mTLS session
        barrier.wait()
        while time.time() < start_at:
            time.sleep(0.001)
        for k in range(reqs):
            t0 = time.perf_counter()
            client.post("/decide", _query(policy_set, first + j, k))
            lat[j].append((time.perf_counter() - t0) * 1000.0)

    ts = [threading.Thread(target=pep, args=(j,)) for j in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return [x for xs in lat for x in xs]


def measure(pool, policy_set, n, reqs, procs):
    p = min(n, procs)
    split = [n // p + (1 if i < n % p else 0) for i in range(p)]
    start_at = time.time() + 0.6
    jobs, first = [], 0
    for s in split:
        jobs.append((policy_set, first, s, reqs, start_at))
        first += s
    return [x for part in pool.map(_worker, jobs) for x in part]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/perf")
    ap.add_argument("--reqs", type=int, default=50)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    make_certs()
    start_postgres()
    rows = []
    pool = mp.Pool(a.procs)
    try:
        # Configurations are interleaved inside every (rep, n) cell and the cache/DB stacks
        # alternate per repetition, so slow drift (e.g. laptop thermal throttling over a
        # long run) affects all configurations equally instead of whichever runs last.
        for rep in range(a.reps):
            for cache in ((False, True) if rep % 2 == 0 else (True, False)):
                with Stack(cache):
                    for _, ps in CONFIGS:
                        measure(pool, ps, 16, 10, a.procs)  # warm connections / interpreter
                    for n in CONCURRENCY:
                        order = CONFIGS if (n + rep) % 2 == 0 else CONFIGS[::-1]
                        for label, ps in order:
                            lat = measure(pool, ps, n, a.reqs, a.procs)
                            q1, med, q3 = np.percentile(lat, [25, 50, 75])
                            rows.append({"cache": cache, "config": label, "concurrency": n, "rep": rep,
                                         "median_ms": med, "q1_ms": q1, "q3_ms": q3, "n": len(lat)})
                            print(f"rep={rep} cache={cache!s:5} {label:22} n={n:3} median={med:8.3f} ms", flush=True)
    finally:
        pool.close()
        stop_postgres()
    with open(out / "decision_time.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump({"requests_per_instance": a.reqs, "repetitions": a.reps, "load_processes": a.procs,
               "concurrency": CONCURRENCY, "cpu_count": os.cpu_count(),
               "paper": {"requests_per_instance": 1000, "repetitions": 10}},
              open(out / "manifest.json", "w"), indent=2)


if __name__ == "__main__":
    main()
