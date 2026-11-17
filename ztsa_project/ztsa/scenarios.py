"""Synthetic enterprise: users, devices, micro-segments, benign noise and four attacks.

The paper evaluates only decision time and leaves accuracy (false allow / false deny)
as future work (Sec. X). This generator provides labelled resource access requests
(RARs) carrying the paper's 29 trust attributes and 9 risk attributes, so every
algorithm can be scored on identical traffic.

Days 0-6 are benign warm-up (used only to fit ZT-CMS norms); days 7-13 are the
evaluation period, where attack episodes are interleaved with benign traffic.
"""
from __future__ import annotations

import numpy as np

SEGMENTS = [  # (name, tier)
    ("intranet", 0), ("mail", 0), ("gitlab", 1), ("ci-cd", 1), ("hr-portal", 2),
    ("finance-erp", 2), ("customer-db", 3), ("prod-k8s", 3), ("ad-admin", 3), ("backup", 2),
]
TIER = [t for _, t in SEGMENTS]
ROLES = {  # role: (usual segments, extra entitled segments)
    "dev":     ({0, 1, 2, 3}, {7}),
    "hr":      ({0, 1, 4}, {5}),
    "finance": ({0, 1, 5}, {6}),
    "ops":     ({0, 1, 3, 7}, {2, 8, 9}),
    "admin":   ({0, 1, 8, 9}, {6, 7}),
    "sales":   ({0, 1, 6}, {5}),
}
ATTACKS = ("credential_theft", "compromised_device", "lateral_movement", "slow_exfiltration")
WARMUP_DAYS, TOTAL_DAYS = 7, 14
DAY = 1440.0


def _clip(x):
    return float(min(1.0, max(0.0, x)))


class Enterprise:
    def __init__(self, seed: int, n_users: int = 200, episodes_per_attack: int = 40):
        self.rng = np.random.default_rng(seed)
        self.n_users = n_users
        self.episodes_per_attack = episodes_per_attack
        r = self.rng
        roles = list(ROLES)
        self.users = []
        for i in range(n_users):
            role = roles[r.integers(len(roles))]
            usual_role, extra = ROLES[role]
            usual = set(r.choice(sorted(usual_role), size=max(2, len(usual_role) - int(r.integers(0, 2))),
                                 replace=False).tolist()) | {0}
            self.users.append({
                "id": f"u{i}", "role": role, "usual": usual, "entitled": usual_role | extra,
                "mfa": r.random() < 0.8, "byod": r.random() < 0.2,
                "shift": float(r.normal(9 * 60, 60)), "devices": [f"d{i}"],
            })
        # Per-day environment state: network threat level, segment threat/patch/maintenance
        self.env = []
        for _ in range(TOTAL_DAYS):
            nt = r.choice([0.0, 0.5, 1.0], p=[0.88, 0.10, 0.02])
            self.env.append({
                "network_threat": float(nt), "network_state": float(r.random() < 0.05) * 0.5,
                "system_threat": [float(r.random() < 0.05) * 0.5 for _ in SEGMENTS],
                "svc_patch": [float(r.random() < 0.10) * 0.5 for _ in SEGMENTS],
                "sys_patch": [float(r.random() < 0.08) * 0.5 for _ in SEGMENTS],
                "system_state": [float(r.random() < 0.04) * 0.5 for _ in SEGMENTS],
            })
        # Per-user-day benign anomalies
        self.anom = {}
        for u in self.users:
            for d in range(TOTAL_DAYS):
                self.anom[(u["id"], d)] = {
                    "travel": r.random() < 0.04, "new_device": r.random() < 0.01,
                    "patch_lag": r.random() < 0.06, "typo": r.random() < 0.05,
                    "late": r.random() < 0.06,
                }

    # --------------------------------------------------------------- attributes
    def _risk(self, day, seg, action, protocol):
        e = self.env[day]
        return {
            "protocol": protocol, "action": action, "data_sensitivity": TIER[seg] / 3,
            "svc_patch": e["svc_patch"][seg], "system_state": e["system_state"][seg],
            "system_threat": e["system_threat"][seg], "sys_patch": e["sys_patch"][seg],
            "network_state": e["network_state"], "network_threat": e["network_threat"],
        }

    def _service_usage(self, u, seg):
        if seg in u["usual"]:
            return 1.0
        return 0.5 if seg in u["entitled"] else 0.2

    def _benign_obs(self, u, seg, minute_of_day, day, byod_now, rate_g):
        r = self.rng
        a = self.anom[(u["id"], day)]
        hi = lambda: _clip(r.beta(12, 1))  # noqa: E731
        in_shift = u["shift"] - 60 <= minute_of_day <= u["shift"] + 9.5 * 60
        o = {
            "auth_factors": 1.0 if u["mfa"] else 0.6,
            "auth_patterns": 0.5 if a["typo"] else hi(),
            "enterprise_presence": hi(), "trust_history": _clip(r.normal(0.9, 0.04)),
            "input_behavior": _clip(r.beta(9, 2)), "service_usage": self._service_usage(u, seg),
            "device_usage": hi(), "access_time": hi() if in_shift else 0.4,
            "access_rate": rate_g, "db_update": hi(),
            "connection_security": 1.0, "sw_patch": 0.5 if a["patch_lag"] else hi(),
            "sys_patch": 0.5 if a["patch_lag"] else hi(), "type": 1.0, "fingerprint": hi(),
            "setup_date": hi(), "location": 0.3 if a["travel"] else hi(), "health": _clip(r.beta(10, 1.5)),
            "managed": 1.0, "vuln_scan": hi(), "dev_auth": 1.0, "dev_presence": hi(),
            "dev_trust_history": _clip(r.normal(0.9, 0.04)), "dev_service_usage": hi(),
            "user_usage": hi(), "dev_db_update": hi(),
            "cc_auth": 1.0, "cc_confidentiality": 1.0, "cc_integrity": 1.0,
        }
        if byod_now:  # legitimate personal device: unmanaged, user certificate only
            o.update(managed=0.0, dev_auth=0.5, type=0.5, vuln_scan=0.3, sys_patch=0.5,
                     fingerprint=0.8, dev_trust_history=0.7)
        if a["new_device"]:
            o.update(fingerprint=0.2, device_usage=0.2, user_usage=0.3, dev_trust_history=0.3,
                     dev_presence=0.3)
        return o

    # --------------------------------------------------------------- traffic
    def _benign(self, out):
        r = self.rng
        for u in self.users:
            for day in range(TOTAL_DAYS):
                if day % 7 in (5, 6) and r.random() < 0.85:
                    continue  # weekends are quiet
                a = self.anom[(u["id"], day)]
                for _ in range(int(r.integers(1, 4))):
                    start = u["shift"] + r.uniform(0, 8 * 60)
                    if a["late"]:
                        start = u["shift"] + r.uniform(11 * 60, 14 * 60)
                    byod_now = u["byod"] and r.random() < 0.4
                    t = day * DAY + start
                    burst = r.random() < 0.05
                    for _ in range(int(r.poisson(12)) + 1):
                        x = r.random()
                        if x < 0.93:
                            seg = int(r.choice(sorted(u["usual"])))
                        elif x < 0.98:
                            seg = int(r.choice(sorted(u["entitled"] | {0})))
                        else:
                            seg = int(r.integers(len(SEGMENTS)))
                        action = float(r.choice([0.0, 0.5, 1.0], p=[0.75, 0.22, 0.03]))
                        protocol = 0.5 if (seg in (7, 8) and r.random() < 0.5) else 0.0
                        obs = self._benign_obs(u, seg, (t % DAY), day, byod_now, 0.5 if burst else 1.0)
                        out.append(self._req(t, u, f"{u['id']}-byod" if byod_now else u["devices"][0],
                                             seg, obs, self._risk(day, seg, action, protocol), None, None))
                        t += float(r.exponential(4.0)) + 0.2

    def _req(self, t, u, device, seg, obs, risk, attack, episode):
        return {"t": float(t), "user": u["id"], "device": device, "segment": seg, "tier": TIER[seg],
                "entitled": seg in u["entitled"], "obs": obs, "risk": risk,
                "malicious": attack is not None, "attack": attack, "episode": episode}

    def _attacks(self, out):
        r = self.rng
        ep = 0
        for kind in ATTACKS:
            for _ in range(self.episodes_per_attack):
                u = self.users[int(r.integers(self.n_users))]
                day = int(r.integers(WARMUP_DAYS, TOTAL_DAYS))
                getattr(self, "_" + kind)(out, u, day, f"{kind}-{ep}")
                ep += 1

    def _targets(self, u, n, novel_bias):
        r = self.rng
        pool_hi = [s for s in range(len(SEGMENTS)) if TIER[s] >= 2]
        segs = []
        for _ in range(n):
            x = r.random()
            if x < novel_bias:
                segs.append(int(r.choice(pool_hi)))
            elif x < novel_bias + 0.2:
                segs.append(int(r.choice(sorted(u["entitled"]))))
            else:
                segs.append(int(r.choice(sorted(u["usual"]))))
        return segs

    def _credential_theft(self, out, u, day, ep):
        r = self.rng
        t = day * DAY + r.uniform(0, DAY)
        vpn_masked = r.random() < 0.4
        dev = f"atk-{ep}"
        for seg in self._targets(u, int(r.integers(15, 41)), 0.5):
            obs = self._benign_obs(u, seg, t % DAY, day, False, _clip(r.uniform(0.5, 1.0)))
            obs.update(auth_factors=1.0 if r.random() < 0.5 else 0.6, auth_patterns=0.5,
                       enterprise_presence=0.6, input_behavior=_clip(r.beta(3, 3)), device_usage=0.1,
                       managed=0.0, dev_auth=0.3, fingerprint=0.1, location=0.7 if vpn_masked else 0.2,
                       type=0.5, vuln_scan=0.3, sys_patch=0.5, dev_presence=0.3,
                       dev_trust_history=0.3, user_usage=0.2)
            if not (u["shift"] - 60 <= t % DAY <= u["shift"] + 570):
                obs["access_time"] = 0.4
            out.append(self._req(t, u, dev, seg, obs, self._risk(day, seg, float(r.choice([0.0, 0.5])), 0.0),
                                 "credential_theft", ep))
            t += float(r.uniform(1, 5))

    def _compromised_device(self, out, u, day, ep):
        r = self.rng
        t = day * DAY + u["shift"] + r.uniform(0, 6 * 60)
        for seg in self._targets(u, int(r.integers(30, 61)), 0.35):
            obs = self._benign_obs(u, seg, t % DAY, day, False, 0.6)
            obs.update(input_behavior=_clip(r.beta(5, 3)), health=_clip(r.beta(5, 3)),
                       vuln_scan=0.7, sw_patch=0.7,
                       connection_security=0.5 if r.random() < 0.3 else 1.0)
            out.append(self._req(t, u, u["devices"][0], seg, obs,
                                 self._risk(day, seg, float(r.choice([0.0, 0.5])), 0.0), "compromised_device", ep))
            t += float(r.uniform(2, 15))

    def _lateral_movement(self, out, u, day, ep):
        r = self.rng
        t = day * DAY + u["shift"] + r.uniform(0, 6 * 60)
        for seg in self._targets(u, int(r.integers(20, 41)), 0.55):
            obs = self._benign_obs(u, seg, t % DAY, day, False, 0.8 if r.random() < 0.5 else 1.0)
            out.append(self._req(t, u, u["devices"][0], seg, obs,
                                 self._risk(day, seg, float(r.choice([0.0, 0.5])), 0.0), "lateral_movement", ep))
            t += float(r.uniform(1, 4))

    def _slow_exfiltration(self, out, u, day, ep):
        r = self.rng
        hi_ent = [s for s in u["entitled"] if TIER[s] >= 2 and s not in u["usual"]]
        target = int(r.choice(hi_ent)) if hi_ent else int(r.choice([4, 5, 6, 9]))
        for d in range(day, min(TOTAL_DAYS, day + 3)):
            t = d * DAY + u["shift"] + r.uniform(8 * 60, 12 * 60)
            for _ in range(int(r.integers(8, 16))):
                seg = target if r.random() < 0.7 else int(r.choice(sorted(u["usual"])))
                obs = self._benign_obs(u, seg, t % DAY, d, False, 0.8)
                if r.random() < 0.6:
                    obs["access_time"] = 0.4
                out.append(self._req(t, u, u["devices"][0], seg, obs, self._risk(d, seg, 0.0, 0.0),
                                     "slow_exfiltration", f"{ep}"))
                t += float(r.uniform(5, 25))

    def generate(self):
        out = []
        self._benign(out)
        self._attacks(out)
        out.sort(key=lambda q: q["t"])
        warm = [q for q in out if q["t"] < WARMUP_DAYS * DAY]
        test = [q for q in out if q["t"] >= WARMUP_DAYS * DAY]
        return warm, test
