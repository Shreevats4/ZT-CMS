# ZT-CMS: reproduction of an IEEE TrustCom 2023 zero-trust paper + our extension

**Base paper (implemented and reproduced):**
L. Bradatsch, O. Miroshkin, N. Trkulja, F. Kargl, *"Zero Trust Score-based Network-level
Access Control in Enterprise Networks,"* Proc. IEEE TrustCom 2023 (22nd Int. Conf. on Trust,
Security and Privacy in Computing and Communications). Preprint: arXiv:2402.08299.
Paper: https://arxiv.org/abs/2402.08299 (results we compare against: Sec. IX, page 7).
Authors' code: https://github.com/zintpavowj/ZT_Score-based_Network_Level_AC (Go, 5 services).

**Project topic:** Zero Trust Architecture as an engineering discipline — continuous
verification, micro-segmentation and least-privilege enforcement in enterprise networks.

## What is implemented

| Part | File | What it is |
|---|---|---|
| SL operators | [ztsa/sl.py](ztsa/sl.py) | Opinions, weighted belief fusion (WBF), cumulative belief fusion (CBF), projected probability |
| 29 trust + 9 risk attributes | [ztsa/attributes.py](ztsa/attributes.py) | The paper's attribute lists (Sec. IV, V) with one documented expert weight table |
| Paper algorithms | [ztsa/algorithms.py](ztsa/algorithms.py) | Additive (Sec. VI, Eq. 1-3) and Subjective Logic (Sec. VII, Eq. 5), static or dynamic risk level |
| **Our novelty: ZT-CMS** | [ztsa/algorithms.py](ztsa/algorithms.py) | SL + **C**ontinuous verification (per-entity decaying CUSUM of disbelief) + **M**icro-**S**egmentation risk agents (segment novelty, lateral fan-out) + least-privilege entitlement agent |
| Testbed (reproduction) | [ztsa/services.py](ztsa/services.py) | PEP → PDP(PE+TA) → PAP / PIP → PostgreSQL, all REST over mutual TLS 1.3, like the paper's Fig. 1 |
| Decision-time experiment | [ztsa/perf.py](ztsa/perf.py) | Paper Sec. IX: 1-128 parallel PEPs, PIP with/without cache, static/dynamic RL |
| Enterprise simulator | [ztsa/scenarios.py](ztsa/scenarios.py) | 200 users, 10 micro-segments, benign noise, 4 attacks |
| Accuracy experiment | [ztsa/accuracy.py](ztsa/accuracy.py) | The paper's stated future work (FP/FN rates) for all algorithms |
| Tests | [ztsa/tests.py](ztsa/tests.py) | Paper worked examples (P(ut)=0.3, P(rod)=0.1, 5>10 deny) and SL identities |

## Run

```powershell
cd ztsa_project
python -m pip install -r ../requirements.txt
python -m unittest -v ztsa.tests
python -m ztsa.perf --out results/perf --reqs 50 --reps 3      # needs PostgreSQL 16 + openssl
python -m ztsa.accuracy --out results/accuracy
python -m ztsa.plots                                           # figures + LaTeX tables for the paper
python -m ztsa.showcase                                        # one-page visual summary (results/showcase)
python -m ztsa.demo                                            # start the live PEP/PDP/PIP/PAP testbed
```

Set `ZTSA_PG_BIN` if PostgreSQL is not in `C:\Program Files\PostgreSQL\16\bin`.

## Protocol notes (honest scope)

* The paper reports **decision time only** (Sec. IX). We reproduce those measurements and check
  its three claims (SL ≈ additive; dynamic RL ≈ +7 %; cache ≈ −42 %) on one desktop, not on the
  paper's four 32-core servers with 100 GbE, so absolute milliseconds differ.
  We use 50 requests per PEP instance and 3 repetitions (paper: 1000 and 10). Configurations
  are interleaved within every (repetition, concurrency) cell: a first, config-by-config run
  (kept in `results/perf_sequential_run/`) showed that whichever configuration ran last looked
  slowest because the laptop throttled over the 40-minute run.
* The no-cache PIP uses a bounded pool of 16 PostgreSQL connections; one connection per
  thread exhausted Windows process resources at 128 parallel PEPs.
* The paper reports **no accuracy**; it lists FP/FN evaluation as future work (Sec. X). Our
  accuracy numbers for the paper's algorithms are therefore *our measurements of the paper's
  method*, on our synthetic enterprise, not numbers printed in the paper.
* ZT-CMS design choices (β, step cap, agent strengths) were set on a development enterprise
  (seed 301). Thresholds were calibrated on seed 101. Results are reported on unseen seeds 201-205.
* The same expert weight table is used by every algorithm.
