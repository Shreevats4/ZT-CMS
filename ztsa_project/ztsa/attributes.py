"""The paper's 29 trust attributes (Sec. IV) and 9 risk attributes (Sec. V).

The paper leaves weights to security experts and uses dummy values in its
prototype. The values below are our single, documented expert setting; every
compared algorithm uses the same table. Each trust attribute has:
  w     additive weight added when met (Sec. VI, Eq. 1)
  u     SL agent uncertainty, which the paper defines as the SL weight (Sec. VII-A)
  crit  attribute treated as security-critical by our least-privilege extension
"""

# name: (entity, w, u, crit)
TRUST = {
    # ---- user (10) ----
    "auth_factors":        ("user", 10, 0.15, True),
    "auth_patterns":       ("user", 4, 0.50, False),
    "enterprise_presence": ("user", 3, 0.60, False),
    "trust_history":       ("user", 5, 0.40, False),
    "input_behavior":      ("user", 4, 0.50, False),
    "service_usage":       ("user", 5, 0.40, False),
    "device_usage":        ("user", 5, 0.40, True),
    "access_time":         ("user", 3, 0.60, False),
    "access_rate":         ("user", 3, 0.60, False),
    "db_update":           ("user", 1, 0.80, False),
    # ---- device (16) ----
    "connection_security": ("device", 5, 0.40, True),
    "sw_patch":            ("device", 4, 0.50, False),
    "sys_patch":           ("device", 5, 0.40, True),
    "type":                ("device", 2, 0.70, False),
    "fingerprint":         ("device", 6, 0.30, True),
    "setup_date":          ("device", 2, 0.70, False),
    "location":            ("device", 4, 0.50, True),
    "health":              ("device", 3, 0.60, False),
    "managed":             ("device", 8, 0.20, True),
    "vuln_scan":           ("device", 5, 0.40, True),
    "dev_auth":            ("device", 8, 0.20, True),
    "dev_presence":        ("device", 2, 0.70, False),
    "dev_trust_history":   ("device", 4, 0.50, False),
    "dev_service_usage":   ("device", 3, 0.60, False),
    "user_usage":          ("device", 4, 0.50, False),
    "dev_db_update":       ("device", 1, 0.80, False),
    # ---- communication channel (3) ----
    "cc_auth":             ("cc", 8, 0.20, True),
    "cc_confidentiality":  ("cc", 8, 0.20, True),
    "cc_integrity":        ("cc", 8, 0.20, True),
}
assert len(TRUST) == 29

# name: (additive risk weight, SL agent uncertainty)
RISK = {
    "protocol":         (5, 0.50),
    "action":           (8, 0.30),
    "data_sensitivity": (15, 0.15),
    "svc_patch":        (6, 0.40),
    "system_state":     (4, 0.50),
    "system_threat":    (12, 0.20),
    "sys_patch":        (6, 0.40),
    "network_state":    (3, 0.50),
    "network_threat":   (12, 0.20),
}
assert len(RISK) == 9

ENTITIES = ("user", "device", "cc")
TRUST_BY_ENTITY = {e: [k for k, v in TRUST.items() if v[0] == e] for e in ENTITIES}

# Static thresholds used by the "static risk level" variants. The paper calls the
# literature's static threshold "a relatively static value"; we use the expected
# dynamic risk level of a nominal request (HTTPS read of tier-1 data, all systems
# normal) so static and dynamic variants start from the same operating point.
NOMINAL_RISK = {"protocol": 0.0, "action": 0.0, "data_sensitivity": 1 / 3, "svc_patch": 0.0,
                "system_state": 0.0, "system_threat": 0.0, "sys_patch": 0.0,
                "network_state": 0.0, "network_threat": 0.0}


def quantize(g: float) -> float:
    """Additive algorithm: discrete degrees of fulfilment -> weight fraction (W_tv -> W_wv)."""
    if g >= 0.8:
        return 1.0
    if g >= 0.4:
        return 0.5
    return 0.0
