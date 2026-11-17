"""Trust algorithms.

Paper baselines (Bradatsch et al., TrustCom 2023):
  Additive   Sec. VI : TS_total = sum of met weights; allow iff TS_total > RL_total (Eq. 1-3)
  SL         Sec. VII: per-entity WBF opinions, CBF risk opinion; allow iff
                       P(ut) > P(rod) and P(dt) > P(rod) and P(cct) > P(rod) (Eq. 5)
Both support a static or dynamic risk level (threshold).

Our extension, ZT-CMS (Continuous verification, Micro-Segmentation, least-privilege Scoping),
keeps the SL algorithm and adds three switchable components:
  cv  continuous verification: per-entity decaying CUSUM of excess SL disbelief,
      measured against each user's / device's own warm-up norm, so weak evidence
      accumulates across requests instead of being judged one request at a time
  ms  micro-segmentation: two risk agents, segment novelty for this user and
      lateral fan-out (distinct segments in the last hour beyond the user's norm)
  lp  least privilege: a role-entitlement risk agent; out-of-entitlement access
      is evidence that permitting the request causes damage

Every algorithm returns a margin; the access decision is margin > tau. The paper's
rule is tau = 0. Calibrated comparisons choose tau per algorithm on benign data.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

from .attributes import NOMINAL_RISK, RISK, TRUST, TRUST_BY_ENTITY, quantize
from .sl import cbf, opinion_from_degree, projected, wbf

STATIC_RL_ADD = sum(RISK[k][0] * v for k, v in NOMINAL_RISK.items())
STATIC_RL_SL = projected(cbf([opinion_from_degree(v, RISK[k][1]) for k, v in NOMINAL_RISK.items()]))


# ----------------------------------------------------------------------------- paper
class Additive:
    def __init__(self, dynamic: bool):
        self.dynamic = dynamic
        self.name = f"Additive-{'dynamic' if dynamic else 'static'}"

    def margin(self, req) -> float:
        obs = req["obs"]
        ts = sum(TRUST[k][1] * quantize(g) for k, g in obs.items())                  # Eq. 1
        if self.dynamic:
            rl = sum(RISK[k][0] * lvl for k, lvl in req["risk"].items())              # Eq. 2
        else:
            rl = STATIC_RL_ADD
        return ts - rl                                                                # Eq. 3


class SubjectiveLogic:
    def __init__(self, dynamic: bool):
        self.dynamic = dynamic
        self.name = f"SL-{'dynamic' if dynamic else 'static'}"

    @staticmethod
    def entity_opinions(obs, u_scale=None):
        out = {}
        for ent, names in TRUST_BY_ENTITY.items():
            ops = []
            for k in names:
                if k in obs:
                    u = TRUST[k][2] * (u_scale(k) if u_scale else 1.0)
                    ops.append(opinion_from_degree(obs[k], u))
            out[ent] = wbf(ops)
        return out

    @staticmethod
    def risk_opinion(risk, extra=()):
        ops = [opinion_from_degree(lvl, RISK[k][1]) for k, lvl in risk.items()]
        ops.extend(extra)
        return cbf(ops)

    def margin(self, req) -> float:
        ents = self.entity_opinions(req["obs"])
        rod = projected(self.risk_opinion(req["risk"])) if self.dynamic else STATIC_RL_SL
        return min(projected(o) for o in ents.values()) - rod                         # Eq. 5


# ----------------------------------------------------------------------------- ours
class ZTCMS(SubjectiveLogic):
    TAU_MIN = 120.0     # CUSUM decay time constant (minutes)
    BETA = 0.5          # trust penalty per unit of accumulated excess disbelief
    STEP = 0.05         # max CUSUM increment per request: large one-off shifts are the
                        # per-request scorer's job; CUSUM targets small persistent shifts
    PROBATION = 10      # unseen entities learn their own norm from their first requests
    WINDOW = 60.0       # fan-out window (minutes)

    def __init__(self, cv=True, ms=True, lp=True, beta=None):
        super().__init__(dynamic=True)
        self.cv, self.ms, self.lp = cv, ms, lp
        self.beta = self.BETA if beta is None else beta
        tags = [t for t, on in (("CV", cv), ("MS", ms), ("LP", lp)) if on]
        self.name = "ZT-CMS (full)" if len(tags) == 3 else "SL-dynamic+" + "+".join(tags)
        self.usual = defaultdict(set)
        self.fan_norm = defaultdict(lambda: 2)
        self.mu = {"user": 0.0, "device": 0.0}   # fallback for entities unseen in warm-up
        self.mu_entity = {}                      # per-entity benign disbelief norm
        self.reset_state()

    def reset_state(self):
        self.cusum = defaultdict(float)
        self.last_t = {}
        self.recent = defaultdict(deque)
        self.online = defaultdict(list)   # first observations of entities unseen in warm-up

    def _fan_out(self, req):
        q = self.recent[req["user"]]
        q.append((req["t"], req["segment"]))
        while q and q[0][0] < req["t"] - self.WINDOW:
            q.popleft()
        return len({s for _, s in q})

    def fit(self, warmup):
        """Learn per-user segment norms and per-entity benign disbelief from benign warm-up traffic."""
        counts = defaultdict(lambda: defaultdict(int))
        fans = defaultdict(list)
        dis = {"user": [], "device": []}
        per = defaultdict(list)
        self.reset_state()
        for r in warmup:
            counts[r["user"]][r["segment"]] += 1
            fans[r["user"]].append(self._fan_out(r))
            ents = self.entity_opinions(r["obs"])
            for e, key in (("user", r["user"]), ("device", r["device"])):
                dis[e].append(ents[e][1])
                per[(e, key)].append(ents[e][1])
        for user, c in counts.items():
            tot = sum(c.values())
            self.usual[user] = {s for s, n in c.items() if n >= 2 and n / tot >= 0.03}
            f = sorted(fans[user])
            self.fan_norm[user] = f[int(0.95 * (len(f) - 1))]

        def norm(xs):  # slack: only disbelief above mean + 1 sd of the entity's own norm accumulates
            m = sum(xs) / len(xs)
            return m + math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
        for e, xs in dis.items():   # unseen entities get the strict global median, not a lenient norm
            self.mu[e] = sorted(xs)[len(xs) // 2]
        self.mu_entity = {k: norm(xs) for k, xs in per.items() if len(xs) >= 20}
        self.reset_state()

    def margin(self, req) -> float:
        ents = self.entity_opinions(req["obs"])
        p = {e: projected(o) for e, o in ents.items()}
        if self.cv:   # continuous verification: decaying CUSUM of excess disbelief per entity
            for e, key in (("user", req["user"]), ("device", req["device"])):
                ck = (e, key)
                last = self.last_t.get(ck)
                decay = math.exp(-(req["t"] - last) / self.TAU_MIN) if last is not None else 1.0
                mu = self.mu_entity.get(ck)
                if mu is None:   # unseen entity: self-baseline during probation
                    obs_d = self.online[ck]
                    if len(obs_d) < self.PROBATION:
                        obs_d.append(ents[e][1])
                        self.last_t[ck] = req["t"]
                        continue
                    m = sum(obs_d) / len(obs_d)
                    mu = self.mu_entity[ck] = m + math.sqrt(sum((x - m) ** 2 for x in obs_d) / len(obs_d))
                inc = min(ents[e][1] - mu, self.STEP)
                self.cusum[ck] = max(0.0, self.cusum[ck] * decay + inc)
                self.last_t[ck] = req["t"]
                p[e] -= self.beta * self.cusum[ck]
        extra = []
        if self.ms:   # micro-segmentation: segment novelty and lateral fan-out risk agents
            novel = req["segment"] not in self.usual.get(req["user"], set())
            extra.append(opinion_from_degree(0.8 if novel else 0.0, 0.10))
            excess = max(0, self._fan_out(req) - self.fan_norm[req["user"]])
            extra.append(opinion_from_degree(min(1.0, excess / 3.0), 0.10))
        if self.lp:   # least privilege: out-of-entitlement access is evidence of damage risk
            extra.append(opinion_from_degree(0.0 if req["entitled"] else 1.0, 0.10))
        rod = projected(self.risk_opinion(req["risk"], extra))
        return min(p.values()) - rod


def all_algorithms():
    return [Additive(False), Additive(True), SubjectiveLogic(False), SubjectiveLogic(True),
            ZTCMS(cv=True, ms=False, lp=False), ZTCMS(cv=False, ms=True, lp=False),
            ZTCMS(cv=False, ms=False, lp=True), ZTCMS(cv=True, ms=True, lp=False),
            ZTCMS(cv=True, ms=True, lp=True)]
