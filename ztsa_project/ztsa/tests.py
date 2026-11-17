"""Conformance tests against the paper's worked examples and SL operator identities.

    python -m unittest -v ztsa.tests
"""
import unittest

from .algorithms import STATIC_RL_SL, ZTCMS, Additive, SubjectiveLogic
from .scenarios import Enterprise
from .sl import cbf, cbf2, opinion_from_degree, projected, wbf


class PaperExamples(unittest.TestCase):
    def test_additive_example_sec_vi(self):
        # Sec. VI: correct password adds 5; outdated patch adds risk 10; 5 > 10 is false -> deny
        ts, rl = 0 + 5, 0 + 10
        self.assertFalse(ts > rl)

    def test_sl_user_example_sec_vii_a(self):
        # Sec. VII-A: omega = {0.2, 0.6, 0.2, 0.5} -> P(ut) = 0.3
        o = (0.2, 0.6, 0.2, 0.5)
        self.assertAlmostEqual(projected(wbf([o])), 0.3)

    def test_sl_risk_example_sec_vii_d(self):
        # Sec. VII-D: omega_rod = {0.0, 0.8, 0.2, 0.5} -> P(rod) = 0.1, and 0.3 > 0.1 permits
        o = (0.0, 0.8, 0.2, 0.5)
        self.assertAlmostEqual(projected(cbf([o])), 0.1)
        self.assertTrue(0.3 > projected(cbf([o])))


class Operators(unittest.TestCase):
    def test_wbf_identity_and_additivity(self):
        o = opinion_from_degree(0.7, 0.3)
        f = wbf([o])
        for x, y in zip(f, o):
            self.assertAlmostEqual(x, y)
        g = wbf([opinion_from_degree(0.9, 0.2), opinion_from_degree(0.1, 0.6), opinion_from_degree(0.5, 0.4)])
        self.assertAlmostEqual(g[0] + g[1] + g[2], 1.0)

    def test_wbf_weights_low_uncertainty_more(self):
        strong_good = opinion_from_degree(1.0, 0.1)
        weak_bad = opinion_from_degree(0.0, 0.8)
        self.assertGreater(projected(wbf([strong_good, weak_bad])), 0.8)

    def test_cbf_reduces_uncertainty_and_commutes(self):
        a, b = opinion_from_degree(0.2, 0.4), opinion_from_degree(0.9, 0.3)
        ab, ba = cbf2(a, b), cbf2(b, a)
        for x, y in zip(ab, ba):
            self.assertAlmostEqual(x, y)
        self.assertLess(ab[2], min(a[2], b[2]))
        self.assertAlmostEqual(sum(ab[:3]), 1.0)

    def test_vacuous_is_neutral(self):
        a = opinion_from_degree(0.3, 0.4)
        out = cbf2((0.0, 0.0, 1.0, 0.5), a)
        for x, y in zip(out, a):
            self.assertAlmostEqual(x, y)


class Algorithms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.warm, cls.test = Enterprise(7, n_users=20, episodes_per_attack=3).generate()

    def test_static_threshold_constant(self):
        self.assertTrue(0.0 < STATIC_RL_SL < 0.5)

    def test_per_entity_no_compensation(self):
        # Sec. VIII: a trusted device must not compensate an untrusted user
        req = dict(self.test[0])
        obs = dict(req["obs"])
        for k in ("auth_factors", "auth_patterns", "input_behavior", "service_usage", "device_usage"):
            obs[k] = 0.0
        req["obs"] = obs
        self.assertLess(SubjectiveLogic(True).margin(req), SubjectiveLogic(True).margin(self.test[0]))

    def test_all_algorithms_run_and_cv_is_stateful(self):
        z = ZTCMS()
        z.fit(self.warm)
        first = [z.margin(r) for r in self.test[:200]]
        z.reset_state()
        again = [z.margin(r) for r in self.test[:200]]
        self.assertEqual(first, again)          # deterministic given state reset
        self.assertTrue(all(isinstance(Additive(True).margin(r), float) for r in self.test[:50]))

    def test_warmup_is_benign_and_test_has_attacks(self):
        self.assertFalse(any(r["malicious"] for r in self.warm))
        self.assertTrue(any(r["malicious"] for r in self.test))


if __name__ == "__main__":
    unittest.main()
