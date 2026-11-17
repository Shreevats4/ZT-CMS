"""Start the whole ZTA testbed and keep it running for a live demo.

    python -m ztsa.demo [--no-cache]

Starts PostgreSQL, then PAP, PIP and PDP (REST over mutual TLS 1.3), sends one
sample access request through every algorithm as a PEP would, prints the
decisions, and keeps the services up until Ctrl+C.
"""
from __future__ import annotations

import argparse
import time

from .perf import PORTS, Stack, make_certs, start_postgres, stop_postgres
from .services import Client

ALGOS = [("Additive, static risk (paper)", "add-static"), ("Additive, dynamic risk (paper)", "add-dynamic"),
         ("SL, static risk (paper)", "sl-static"), ("SL, dynamic risk (paper)", "sl-dynamic"),
         ("ZT-CMS (ours)", "ztcms")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true", help="PIP reads PostgreSQL on every request")
    a = ap.parse_args()
    make_certs()
    print("starting PostgreSQL ...", flush=True)
    start_postgres()
    try:
        with Stack(cache=not a.no_cache):
            print(f"PAP :{PORTS['pap']}  PIP :{PORTS['pip']} (cache={'off' if a.no_cache else 'on'})  "
                  f"PDP :{PORTS['pdp']}  - all mTLS 1.3", flush=True)
            pep = Client("pep", PORTS["pdp"])
            req = {"user": "u7", "device": "d7", "resource": "r3", "t": 0.0}
            print(f"\nsample request: user={req['user']} device={req['device']} resource={req['resource']}")
            for label, ps in ALGOS:
                t0 = time.perf_counter()
                r = pep.post("/decide", dict(req, policy_set=ps))
                ms = (time.perf_counter() - t0) * 1000
                print(f"  {label:<32} {'ALLOW' if r['allow'] else 'DENY ':<5}  margin {r['margin']:+.3f}  {ms:6.1f} ms")
            print("\nstack is running; send requests to the PDP at https://localhost:"
                  f"{PORTS['pdp']}/decide. Press Ctrl+C to stop.", flush=True)
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_postgres()
        print("stopped.")


if __name__ == "__main__":
    main()
