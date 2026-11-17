"""Subjective Logic operators used by the paper (Josang 2016; van der Heijden et al. 2018).

Binary opinions are tuples (b, d, u, a) with b + d + u = 1.
"""
from __future__ import annotations

U_MIN = 1e-6  # avoid dogmatic opinions (u = 0), for which WBF is undefined


def opinion_from_degree(g: float, u: float, a: float = 0.5):
    """Agent opinion from a degree of fulfilment g in [0, 1] and an uncertainty u.

    The paper defines the trust-attribute weight as the agent's uncertainty:
    the lower u, the more the agent counts in weighted belief fusion.
    """
    u = min(max(u, U_MIN), 1.0)
    g = min(max(g, 0.0), 1.0)
    return (g * (1.0 - u), (1.0 - g) * (1.0 - u), u, a)


def projected(o) -> float:
    """Projected probability P = b + u * a."""
    return o[0] + o[2] * o[3]


def wbf(ops):
    """Weighted belief fusion (van der Heijden, Kopp, Kargl 2018), N sources.

    Closed form with prod(u) factored out for numerical stability:
      b = sum b_i (1-u_i)/u_i / (sum 1/u_i - N)
      u = (N - sum u_i)      / (sum 1/u_i - N)
      a = sum a_i (1-u_i)    / (N - sum u_i)
    Vacuous opinions (u = 1) carry no weight and are skipped.
    """
    ops = [o for o in ops if o[2] < 1.0]
    if not ops:
        return (0.0, 0.0, 1.0, 0.5)
    n = len(ops)
    inv = sum(1.0 / o[2] for o in ops)
    su = sum(o[2] for o in ops)
    den = inv - n
    b = sum(o[0] * (1.0 - o[2]) / o[2] for o in ops) / den
    u = (n - su) / den
    a = sum(o[3] * (1.0 - o[2]) for o in ops) / (n - su)
    return (b, max(0.0, 1.0 - b - u), u, a)


def cbf2(x, y):
    """Cumulative belief fusion of two opinions (Josang 2016, Def. 12.6)."""
    ua, ub = x[2], y[2]
    k = ua + ub - ua * ub
    if k <= 0.0:  # both dogmatic: average
        return ((x[0] + y[0]) / 2, (x[1] + y[1]) / 2, 0.0, (x[3] + y[3]) / 2)
    b = (x[0] * ub + y[0] * ua) / k
    u = ua * ub / k
    da = ua + ub - 2 * ua * ub
    a = (x[3] * ub + y[3] * ua - (x[3] + y[3]) * ua * ub) / da if da > 0 else (x[3] + y[3]) / 2
    return (b, max(0.0, 1.0 - b - u), u, a)


def cbf(ops):
    """Sequential cumulative fusion of all opinions (associative operator)."""
    out = (0.0, 0.0, 1.0, 0.5)
    for o in ops:
        out = cbf2(out, o)
    return out
