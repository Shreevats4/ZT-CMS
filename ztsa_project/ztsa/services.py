"""Reproduction testbed of the paper's Sec. IX (Fig. 1 components).

PEP (load generator) -> PDP (policy engine + trust algorithm) -> PAP (policy) and
PIP (attributes) -> PostgreSQL attribute DB. All component links are REST over
mutual TLS; the PIP reaches the DB over SQL, exactly as in the paper. The PIP
either queries the DB for every request (no cache, Fig. 2) or serves attributes
from its in-memory cache after first load (Fig. 3).

Run one component:  python -m ztsa.services pdp|pip|pap --port N [--cache]
"""
from __future__ import annotations

import argparse
import http.client
import json
import socket
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .algorithms import ZTCMS, Additive, SubjectiveLogic
from .attributes import RISK, TRUST
from .scenarios import ROLES, SEGMENTS, TIER

ROOT = Path(__file__).resolve().parents[1]
CERTS = ROOT / "runtime" / "certs"
PG_PORT = 55440
N_USERS, N_DEVICES = 200, 200


# ----------------------------------------------------------------------------- TLS helpers
def server_ctx(name):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(CERTS / f"{name}.crt", CERTS / f"{name}.key")
    ctx.load_verify_locations(CERTS / "ca.crt")
    ctx.verify_mode = ssl.CERT_REQUIRED          # mutual TLS
    return ctx


def client_ctx(name):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(CERTS / f"{name}.crt", CERTS / f"{name}.key")
    ctx.load_verify_locations(CERTS / "ca.crt")
    return ctx


class Client:
    """Thread-local keep-alive mTLS connection to one component."""

    def __init__(self, me, port):
        self.ctx, self.port, self.local = client_ctx(me), port, threading.local()

    def _conn(self):
        c = getattr(self.local, "c", None)
        if c is None:
            c = http.client.HTTPSConnection("localhost", self.port, context=self.ctx, timeout=60)
            for k in range(20):
                try:
                    c.connect()
                    break
                except OSError:
                    if k == 19:
                        raise
                    time.sleep(0.05 * (k + 1))
            c.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.local.c = c
        return c

    def post(self, path, obj):
        body = json.dumps(obj).encode()
        for attempt in (0, 1):
            c = self._conn()
            try:
                c.request("POST", path, body, {"Content-Type": "application/json"})
                resp = c.getresponse()
                return json.loads(resp.read())
            except (ConnectionError, http.client.HTTPException, OSError):
                self.local.c = None
                if attempt:
                    raise


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    route = None  # set per server: callable(path, obj) -> obj

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        obj = json.loads(self.rfile.read(n)) if n else {}
        out = json.dumps(type(self).route(self.path, obj)).encode()
        # one write: avoids Nagle/delayed-ACK stalls between header and body
        self.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                         + str(len(out)).encode() + b"\r\n\r\n" + out)

    def log_message(self, *a):
        pass


def serve(name, port, route):
    class H(Handler):
        pass
    H.route = staticmethod(route)
    class S(ThreadingHTTPServer):
        request_queue_size = 512      # must be a class attribute: listen() runs in __init__
        daemon_threads = True
    srv = S(("127.0.0.1", port), H)
    # handshake lazily in the handler thread, not serially in the accept loop
    srv.socket = server_ctx(name).wrap_socket(srv.socket, server_side=True, do_handshake_on_connect=False)
    print(f"{name} listening on {port}", flush=True)
    srv.serve_forever()


# ----------------------------------------------------------------------------- database
def pg_dsn():
    return f"host=127.0.0.1 port={PG_PORT} user=ztsa dbname=ztsa"


def seed_database():
    """Fill attribute DB with dummy values, as the paper did for all 29 + 9 attributes."""
    import random

    import psycopg
    rnd = random.Random(7)
    with psycopg.connect(pg_dsn(), autocommit=True) as con:
        con.execute("DROP TABLE IF EXISTS attrs")
        con.execute("CREATE TABLE attrs (etype text, eid text, attr text, value double precision)")
        rows = []
        for i in range(N_USERS):
            for k, (ent, *_r) in TRUST.items():
                if ent == "user":
                    rows.append(("user", f"u{i}", k, rnd.uniform(0.6, 1.0)))
            role = list(ROLES)[i % len(ROLES)]
            rows.append(("user", f"u{i}", "role_idx", float(list(ROLES).index(role))))
        for i in range(N_DEVICES):
            for k, (ent, *_r) in TRUST.items():
                if ent == "device":
                    rows.append(("device", f"d{i}", k, rnd.uniform(0.6, 1.0)))
        for k, (ent, *_r) in TRUST.items():
            if ent == "cc":
                rows.append(("cc", "mtls", k, 1.0))
        for s in range(len(SEGMENTS)):
            for k in RISK:
                rows.append(("resource", f"r{s}", k, TIER[s] / 3 if k == "data_sensitivity" else 0.0))
        with con.cursor() as cur:
            cur.executemany("INSERT INTO attrs VALUES (%s,%s,%s,%s)", rows)
        con.execute("CREATE INDEX attrs_idx ON attrs (etype, eid)")
        con.execute("ANALYZE attrs")


# ----------------------------------------------------------------------------- PIP
PIP_POOL = 16


def pip_route_factory(cache: bool):
    import queue

    import psycopg
    pool: queue.Queue = queue.Queue()
    for _ in range(PIP_POOL):      # bounded pool (like Go database/sql); one backend per thread
        pool.put(None)             # exhausts Windows desktop heap at 128 parallel PEPs
    store = {}
    sql_static = ("SELECT etype, attr, value FROM attrs WHERE (etype, eid) IN "
                  "(('user', %s), ('device', %s), ('cc', 'mtls'))")
    sql_dynamic = ("SELECT etype, attr, value FROM attrs WHERE (etype, eid) IN "
                   "(('user', %s), ('device', %s), ('cc', 'mtls'), ('resource', %s))")

    def query(sql, args):
        c = pool.get()
        try:
            if c is None or c.closed:
                c = psycopg.connect(pg_dsn(), autocommit=True)
            return c.execute(sql, args).fetchall()
        except Exception:
            c = None
            raise
        finally:
            pool.put(c)

    if cache:  # Fig. 3 scenario: PIP has loaded all attributes and serves from memory
        with psycopg.connect(pg_dsn()) as con:
            for etype, eid, attr, val in con.execute("SELECT etype, eid, attr, value FROM attrs"):
                store.setdefault((etype, eid), []).append((etype, attr, val))

    def route(path, q):
        if cache:
            keys = [("user", q["user"]), ("device", q["device"]), ("cc", "mtls")]
            if q["risk"]:
                keys.append(("resource", q["resource"]))
            rows = [r for k in keys for r in store.get(k, ())]
        # Fig. 2 scenario: one DB batch per request; dynamic risk adds the resource's risk attributes
        elif q["risk"]:
            rows = query(sql_dynamic, (q["user"], q["device"], q["resource"]))
        else:
            rows = query(sql_static, (q["user"], q["device"]))
        obs, risk, ctx = {}, {}, {}
        for etype, attr, val in rows:
            if etype == "resource":
                risk[attr] = val
            elif attr == "role_idx":
                ctx["role_idx"] = val
            else:
                obs[attr] = val
        return {"obs": obs, "risk": risk, "ctx": ctx}
    return route


# ----------------------------------------------------------------------------- PAP
POLICY_SETS = {
    "add-static": ("additive", False), "add-dynamic": ("additive", True),
    "sl-static": ("sl", False), "sl-dynamic": ("sl", True), "ztcms": ("ztcms", True),
}


def pap_route(path, q):
    algo, dyn = POLICY_SETS[q["policy_set"]]
    return {"algorithm": algo, "dynamic_risk": dyn, "resource": q["resource"], "threshold": 0.0}


# ----------------------------------------------------------------------------- PDP
def pdp_route_factory(pip_port, pap_port):
    pip, pap = Client("pdp", pip_port), Client("pdp", pap_port)
    ztcms, lock = ZTCMS(), threading.Lock()
    algos = {("additive", False): Additive(False), ("additive", True): Additive(True),
             ("sl", False): SubjectiveLogic(False), ("sl", True): SubjectiveLogic(True)}
    roles = list(ROLES)

    def route(path, q):
        policy = pap.post("/policy", q)                                   # PE <- PAP
        attrs = pip.post("/attributes", {"user": q["user"], "device": q["device"],
                                         "resource": q["resource"], "risk": policy["dynamic_risk"]})
        seg = int(q["resource"][1:])
        req = {"t": q["t"], "user": q["user"], "device": q["device"], "segment": seg, "tier": TIER[seg],
               "obs": attrs["obs"], "risk": attrs["risk"]}
        if policy["algorithm"] == "ztcms":
            role = roles[int(attrs["ctx"].get("role_idx", 0))]
            req["entitled"] = seg in (ROLES[role][0] | ROLES[role][1])
            with lock:                                                    # shared temporal state
                m = ztcms.margin(req)
        else:
            m = algos[(policy["algorithm"], policy["dynamic_risk"])].margin(req)
        return {"allow": m > policy["threshold"], "margin": m}
    return route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("component", choices=["pdp", "pip", "pap", "seed"])
    ap.add_argument("--port", type=int)
    ap.add_argument("--pip", type=int, default=0)
    ap.add_argument("--pap", type=int, default=0)
    ap.add_argument("--cache", action="store_true")
    a = ap.parse_args()
    if a.component == "seed":
        seed_database()
    elif a.component == "pip":
        serve("pip", a.port, pip_route_factory(a.cache))
    elif a.component == "pap":
        serve("pap", a.port, pap_route)
    else:
        serve("pdp", a.port, pdp_route_factory(a.pip, a.pap))


if __name__ == "__main__":
    main()
