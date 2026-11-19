"""Forecast GTA VI trophy earn rates on Exophase as five quantiles per trophy.

Counts-space, refresh-step model (v2, written from scratch).

Why this shape.  Exophase does not move a game's counts continuously: the
house saw the page frozen -- 638 players tracked, every rate identical --
from its first probe through its 04:00Z read on launch night, then one
refresh by 04:30Z (3,868 players, every rate but the 100% one lower).  A
rate only changes when Exophase refreshes, and at a refresh it changes as a
ratio of two counts: earners over players tracked.  So this model works in
counts, not in rates:

  * Collapse the history into distinct page states (consecutive identical
    reads are one state).  A transition between states is a refresh.
  * Per refresh j, players jump by dP_j and trophy i's earners by dE_ij.
    Earners split into two channels, fitted per trophy by a ridge
    regression over the observed refreshes (priors are tier medians):

        dE_ij = q_i * dP_j  +  h_i * (P_{j-1} * dt_j)

    q_i is the earn rate among newly tracked players (the dilution
    channel: new players hold far fewer trophies), h_i the per-player
    per-hour rate at which already-tracked players who do NOT hold
    the trophy yet keep earning it (the progression channel, applied
    to the non-holder pool P - E, not to all players).  The two are
    separated because refreshes vary in size and spacing: a big jump
    with a short gap is mostly new players, a small jump after a long
    gap mostly progression.
  * Players are projected refresh by refresh: the next refreshes arrive
    at the fitted rate, each adding the players the fitted arrival
    rate expects at that time (the arrival rate itself is fitted as
    log-linear in time from dP/dt over the observed refreshes, so a
    decaying launch surge is extrapolated as a decay, not a constant).
    Earners follow by the same step rule.  Rate = E / P.
  * Uncertainty is honest about counts: binomial noise on the new
    cohort, Poisson noise on progression, parameter uncertainty from the
    regression, uncertainty on the number and size of future refreshes,
    and a drift term that grows with the horizon.  All of it is combined
    on the logit scale.
  * Between refreshes the page is exactly frozen, so the forecast is a
    mixture, not a random walk: with probability p0 no refresh lands
    before the target and the rate is exactly the last read (spread 0.03
    logit, just rounding); with probability 1 - p0 at least one refresh
    lands and the rate follows the projected counts.  p0 comes from a
    renewal model of the inter-refresh interval, conditioned on how long
    the page has already been flat.  Quantiles are read off the mixture
    CDF by bisection.  This is what makes a short horizon sharp and a
    long horizon wide, instead of spreading a random walk everywhere.

Priors used only when no refresh has ever been observed (m = 0).  They
come from the competition's own published rules (RULES.txt): the first
refresh took about 4.5 h, multiplied players by 6.06 (638 -> 3,868), and
the 100% trophy stayed at 100% -- so all 3,230 new players held it and
only 103 other trophies were earned among them, i.e. new players earn
other trophies at about 6.7% of the tracked players' rates.  Those
numbers are a prior template for the first refresh, not a claim about
any single trophy: the mixture spread carries the allocation
uncertainty.  The flat time behind the renewal model counts half an
hour of pre-history, because the page predates the first read (the
last refresh most likely fell within the first hour after release).
Once refreshes are observed, everything is fitted and the priors stop
mattering.

Everything is deterministic: no randomness, no wall clock, no network.
Python 3.13 standard library only.  Input from $MAKEFUN_INPUT_DIR
(else /input), output to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_VALUES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
LO = 0.01
HI = 99.99

# --- Priors for the no-refresh-observed case (see module docstring) ---
PRIOR_INTERVAL_H = 4.5        # median hours between Exophase refreshes
PRIOR_INTERVAL_SIGMA = 0.30   # log-sigma of the inter-refresh interval
PRIOR_JUMP_FACTOR = 6.06      # players multiplier of the first refresh
PRIOR_JUMP_ACCUM_H = 4.5      # hours of accumulation behind that jump
PRIOR_NEW_RATE = 0.067        # new players' earn rate, as a share of the
                              # tracked players' rate (other trophies)
PRIOR_ALLOC_SIGMA = 0.25      # logit spread for systematic per-trophy
                              # allocation away from proportionality
PRIOR_JUMP_CV = 0.35          # relative sd of the first-refresh jump size
PRE_HISTORY_H = 0.5           # the page predates the first read: the last
                              # refresh most likely fell within the first
                              # hour after release, so the true flat time
                              # is the observed flat time plus ~0.5 h
# --- Model constants ---
DECAY_PER_REFRESH = 0.95      # future refresh sizes shrink (growth slows)
SIGMA_FROZEN = 0.03           # logit spread when no refresh lands
SIGMA_DRIFT_PER_SQRT_DAY = 0.10  # q/h drift over the horizon, logit
SIGMA_FLOOR = 0.12            # floor on the projected (refresh) sigma
SIGMA_CAP = 2.0               # cap on the projected (refresh) sigma
RIDGE_LAMBDA = 0.5            # half a pseudo-observation of prior weight
RECENCY_HALF_LIFE_H = 48.0    # refreshes older than this count half


def clamp_rate(pct):
    if pct is None:
        return None
    return LO if pct < LO else HI if pct > HI else pct


def to_logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def from_logit(value):
    if value > 20.0:
        return HI
    if value < -20.0:
        return LO
    return 100.0 / (1.0 + math.exp(-value))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def inv_normal(level):
    """Standard normal quantile (Acklam's rational approximation)."""
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416)
    p_low = 0.02425
    p_high = 1.0 - p_low
    if level < p_low:
        q = math.sqrt(-2.0 * math.log(level))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if level <= p_high:
        q = level - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - level))
    num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
    den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    return -num / den


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def round4(x):
    """Round to 4 decimals, half up (matches the starter's rounding)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_moment(text):
    if text is None:
        return None
    if isinstance(text, str) and text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def hours_between(a, b):
    return (b - a).total_seconds() / 3600.0


# ---------------------------------------------------------------- states

def extract_states(case):
    """Collapse history into distinct page states.

    Returns (states, checkpoint, n_trophies) where each state is
    (moment, players_or_None, [rate_or_None per trophy]).  Consecutive
    reads with identical players and rates are one state; its moment is
    the first read showing it.
    """
    trophies = case.get("trophies") or []
    n = len(trophies)
    states = []
    for read in case.get("history") or []:
        moment = parse_moment(read.get("at"))
        if moment is None:
            continue
        players = read.get("players_tracked")
        if not isinstance(players, (int, float)) or players <= 0:
            players = None
        rates = read.get("rates") or []
        row = []
        for i in range(n):
            rate = rates[i] if i < len(rates) else None
            row.append(rate if isinstance(rate, (int, float)) else None)
        key = (players, tuple(row))
        if states and states[-1][2] == key:
            continue
        states.append((moment, players, key))
    checkpoint = parse_moment(case.get("at"))
    if checkpoint is None and states:
        checkpoint = states[-1][0]
    return states, checkpoint, n


def last_known(states, i):
    """(rate, players) of trophy i at the latest state that lists it."""
    for _moment, players, key in reversed(states):
        if key[1][i] is not None:
            return key[1][i], players
    return None, None


# ------------------------------------------------------- refresh fitting

def sd_q_prior(q, exposure_dp):
    """Prior sd for a new-cohort earn rate: 50% relative, plus one
    earner of absolute uncertainty over the exposure (total new
    players).  Never an absolute floor: for a rare trophy q itself is
    ~0.001, and a 0.02 floor would pretend its new earners could be
    thirty times the point estimate."""
    return 0.5 * q + 1.0 / max(exposure_dp, 1.0)


def ridge2(pairs, ys, prior, weights=None, lam=RIDGE_LAMBDA):
    """Ridge fit of y ~ a*u1 + b*u2 toward `prior`, with covariance.

    `weights` (optional, same length as pairs) down-weight older
    refreshes so the fit tracks slow drift in q and h.  Non-negativity
    is enforced by refitting one-parameter when a coefficient goes
    negative.
    """
    if weights is None:
        weights = [1.0] * len(pairs)
    s11 = sum(w * u1 * u1 for w, (u1, _) in zip(weights, pairs)) + lam
    s12 = sum(w * u1 * u2 for w, (u1, u2) in zip(weights, pairs))
    s22 = sum(w * u2 * u2 for w, (_, u2) in zip(weights, pairs)) + lam
    t1 = sum(w * u1 * y for w, (u1, _), y in zip(weights, pairs, ys)) \
        + lam * prior[0]
    t2 = sum(w * u2 * y for w, (_, u2), y in zip(weights, pairs, ys)) \
        + lam * prior[1]
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        return prior, (abs(prior[0]) * 0.5 + 0.1, abs(prior[1]) * 0.5 + 0.1)
    a = (t1 * s22 - t2 * s12) / det
    b = (s11 * t2 - s12 * t1) / det
    resid = [(y - (a * u1 + b * u2)) for (u1, u2), y in zip(pairs, ys)]
    s2 = max(sum(w * r * r for w, r in zip(weights, resid))
             / max(sum(weights), 1.0), 1.0)
    inv11 = s22 / det
    inv22 = s11 / det
    if a < 0.0 or b < 0.0:
        # Refit with the negative channel pinned at zero.
        if a < 0.0 and b < 0.0:
            return (0.0, 0.0), (math.sqrt(s2 * inv11), math.sqrt(s2 * inv22))
        if a < 0.0:
            den = s22
            b1 = t2 / den if den > 0 else prior[1]
            b1 = max(0.0, b1)
            return (0.0, b1), (math.sqrt(s2 * inv11), math.sqrt(s2 / den))
        den = s11
        a1 = t1 / den if den > 0 else prior[0]
        a1 = max(0.0, a1)
        return (a1, 0.0), (math.sqrt(s2 / den), math.sqrt(s2 * inv22))
    return (a, b), (math.sqrt(s2 * inv11), math.sqrt(s2 * inv22))


def fit_arrival_rate(trans):
    """Fit the new-player arrival rate nu(t) = exp(a + b t) from the
    observed refreshes, where nu_j = dP_j / dt_j (players per hour).
    Dividing by the gap removes the refresh-interval noise from the
    jump sizes: a jump after a long gap is big because more players
    arrived, not because arrivals sped up.  Returns (a, b, sd_log)
    with b clamped to [-0.08, +0.02]/h, or None when too few
    positive jumps exist.  Times are hours since the first state."""
    pts = []
    for dt, dP, _p_before, _dE, t_mid, _e_before in trans:
        if dP > 0 and dt > 0:
            pts.append((t_mid, math.log(dP / dt)))
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx <= 0:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    b = max(-0.08, min(0.02, sxy / sxx))
    a = my - b * mx
    resid = [p[1] - (a + b * p[0]) for p in pts]
    sd = math.sqrt(sum(r * r for r in resid) / max(n - 2, 1))
    return (a, b, min(0.6, max(0.05, sd)))


def fit_refresh_model(states, n, trophies):
    """Fit the refresh-step parameters from the observed transitions.

    Returns a dict with lam (refreshes/hour), interval_median_h,
    dp_base (typical players added per refresh), dp_sd, per-trophy
    (q, h, sd_q, sd_h), and m (number of observed refreshes).
    """
    m = len(states) - 1
    result = {"m": m, "lam": None, "interval_median_h": PRIOR_INTERVAL_H,
              "dp_base": None, "dp_sd": None, "per_trophy": None,
              "arrival": None, "t_last_h": 0.0}
    if m < 1:
        return result
    # Transitions: (dt_h, dP, P_before, [dE per trophy], t_mid_h).
    trans = []
    for j in range(1, len(states)):
        t0, p0, key0 = states[j - 1]
        t1, p1, key1 = states[j]
        dt = max(hours_between(t0, t1), 1e-6)
        if p0 is None or p1 is None:
            continue
        dP = p1 - p0
        dE = []
        e_before = []
        for i in range(n):
            r0 = key0[1][i]
            r1 = key1[1][i]
            if r0 is None or r1 is None:
                dE.append(None)
                e_before.append(None)
            else:
                dE.append(r1 / 100.0 * p1 - r0 / 100.0 * p0)
                e_before.append(r0 / 100.0 * p0)
        t_mid = hours_between(states[0][0], t0) + 0.5 * dt
        trans.append((dt, dP, p0, dE, t_mid, e_before))
    if not trans:
        result["m"] = 0
        return result
    span_h = max(hours_between(states[0][0], states[-1][0]), 1e-6)
    result["lam"] = len(trans) / span_h
    result["t_last_h"] = span_h
    result["arrival"] = fit_arrival_rate(trans)
    dts = [t[0] for t in trans]
    result["interval_median_h"] = median(dts) or PRIOR_INTERVAL_H
    dps = [t[1] for t in trans if t[1] > 0]
    recent_dps = [t[1] for t in trans[-3:] if t[1] > 0]
    base = median(recent_dps) if recent_dps else (median(dps) if dps else 0.0)
    result["dp_base"] = base
    if len(dps) >= 2:
        mean_dp = sum(dps) / len(dps)
        var_dp = sum((d - mean_dp) ** 2 for d in dps) / (len(dps) - 1)
        result["dp_sd"] = math.sqrt(var_dp)
    else:
        result["dp_sd"] = 0.5 * base if base > 0 else 0.0

    # Normalisation scale for the new-players regressor.  The
    # progression regressor is normalised per trophy below: its pool
    # is the players who do NOT hold that trophy yet, which varies
    # hugely between a 1% trophy and a 30% one.
    x1s = [t[1] for t in trans]
    X1 = median([abs(v) for v in x1s if v != 0]) or 1.0

    # One-parameter per-trophy estimates, for tier-median priors.
    # Progression exposure is (players - earners) x hours: only
    # players who do not hold the trophy yet can earn it.
    q1 = []
    h1 = []
    for i in range(n):
        num = 0.0
        den_dp = 0.0
        den_ex = 0.0
        for dt, dP, p_before, dE, _t_mid, e_before in trans:
            if dE[i] is None or e_before[i] is None:
                continue
            num += dE[i]
            den_dp += max(dP, 0.0)
            den_ex += (p_before - e_before[i]) * dt
        q1.append(num / den_dp if den_dp > 0 else None)
        h1.append(num / den_ex if den_ex > 0 else None)
    tiers = [t.get("tier") for t in trophies]

    def tier_median(values, tier, fallback):
        pool = [v for v, t in zip(values, tiers)
                if v is not None and t == tier]
        if len(pool) >= 2:
            return median(pool)
        pool = [v for v in values if v is not None]
        got = median(pool) if pool else None
        return got if got is not None else fallback

    per_trophy = []
    tot_dp = sum(max(t[1], 0.0) for t in trans)
    t_last = max(t[4] for t in trans)
    for i in range(n):
        rate_last, _ = last_known(states, i)
        saturated = rate_last is not None and rate_last >= 99.995
        # Neutral prior: the one-parameter estimates q1 (all growth to
        # new players) and h1 (all growth to progression) are the two
        # ends of the unidentified tradeoff segment; the truth lies
        # between them, so the prior sits at its midpoint.  Centring
        # on (q1, h1) instead would pull both channels up at once.
        q_prior = 0.5 * tier_median(q1, tiers[i], 0.05)
        h_prior = 0.5 * tier_median(h1, tiers[i], 0.0)
        q_prior = min(1.0, max(0.0, q_prior))
        h_prior = max(0.0, h_prior)
        if saturated:
            # Everyone already holds it; every new player holds it too.
            per_trophy.append((1.0, 0.0, 0.005, 0.0))
            continue
        if m < 2:
            # One refresh cannot separate the channels: attribute the
            # jump to new players, keep progression at zero.
            qi = q1[i] if q1[i] is not None else q_prior
            qi = min(1.0, max(0.0, qi))
            per_trophy.append((qi, 0.0, sd_q_prior(qi, tot_dp), 0.0))
            continue
        pairs = []
        ys = []
        x2_raw = []
        weights = []
        for dt, dP, p_before, dE, t_mid, e_before in trans:
            if dE[i] is None or e_before[i] is None:
                continue
            x2_raw.append((p_before - e_before[i]) * dt)
            pairs.append((dP / X1, x2_raw[-1]))
            ys.append(dE[i])
            weights.append(0.5 ** ((t_last - t_mid) / RECENCY_HALF_LIFE_H))
        if not pairs:
            per_trophy.append((q_prior, h_prior,
                               sd_q_prior(q_prior, tot_dp),
                               max(h_prior, 1e-6)))
            continue
        X2 = median([abs(v) for v in x2_raw if v != 0]) or 1.0
        pairs = [(u1, u2 / X2) for (u1, u2) in pairs]
        (a, b), (sd_a, sd_b) = ridge2(
            pairs, ys, (q_prior * X1, h_prior * X2), weights)
        qi = min(1.0, max(0.0, a / X1))
        hi = max(0.0, b / X2)
        per_trophy.append((qi, hi, sd_a / X1, sd_b / X2))
    result["per_trophy"] = per_trophy
    return result


# ------------------------------------------------------------- projection

def poisson_moments(lam_h, conditional):
    """Mean and variance of a Poisson(lam_h) refresh count; when
    `conditional`, conditioned on at least one refresh (the mixture's
    refresh branch: the count is then 1 with high probability, not
    lam_h with Poisson spread).  Summed exactly to k = 40."""
    if lam_h <= 0:
        return (1.0, 0.0) if conditional else (0.0, 0.0)
    probs = []
    total = 0.0
    term = math.exp(-lam_h)
    for k in range(0, 41):
        if k > 0:
            term *= lam_h / k
        probs.append(term)
        total += term
    if conditional:
        denom = 1.0 - probs[0]
        if denom <= 0:
            return 1.0, 0.0
        mean = sum(k * p for k, p in enumerate(probs) if k > 0) / denom
        var = sum((k - mean) ** 2 * p for k, p in enumerate(probs)
                  if k > 0) / denom
        return mean, var
    mean = sum(k * p for k, p in enumerate(probs))
    var = sum((k - mean) ** 2 * p for k, p in enumerate(probs))
    return mean, var


def project_counts(p_last, e_last, q, h, lam, dp_seq, dp_sd,
                   horizon_h, n_refresh, conditional_count):
    """Step the counts forward over `n_refresh` refreshes.

    Returns (P, E, dP_total, exposure_P_hours, var_dP_total,
    var_E_count) where var_dP_total is the variance of the total new
    players (refresh sizes vary, and the refresh count varies) and
    var_E_count is the count noise on earners (binomial on the new
    cohort plus Poisson on progression).  Parameter uncertainty is
    NOT in here: the caller folds it in through the sensitivities
    of log(E/P), because a bigger jump moves E and P together --
    treating them as independent would overstate the rate's spread
    several-fold.
    """
    if horizon_h <= 0 or n_refresh <= 0:
        return p_last, e_last, 0.0, 0.0, 0.0, 0.0
    dt = 1.0 / lam if lam and lam > 0 else horizon_h
    P = float(p_last)
    E = float(e_last)
    dp_total = 0.0
    exposure = 0.0  # integral of (P - E) over time: the non-holder
    # player-hours that progression can draw on.  Only players who do
    # not hold the trophy yet can earn it; using P instead would
    # over-project earners badly once rates are high.
    t = 0.0
    steps = 0
    for k in range(n_refresh):
        dp = dp_seq[k] if k < len(dp_seq) else 0.0
        dp = max(0.0, dp)
        seg = min(dt, horizon_h - t)
        if seg <= 0:
            break
        exposure += (P - E) * seg
        E += h * (P - E) * seg + q * dp
        P += dp
        dp_total += dp
        t += seg
        steps += 1
    if t < horizon_h:
        exposure += (P - E) * (horizon_h - t)
        E += h * (P - E) * (horizon_h - t)
    mean_n, var_n = poisson_moments(lam * horizon_h if lam else 0.0,
                                    conditional_count)
    mean_dp = dp_total / steps if steps > 0 else 0.0
    var_dp_total = mean_n * dp_sd ** 2 + var_n * mean_dp ** 2
    var_E_count = dp_total * q * (1.0 - q) + h * exposure
    return P, E, dp_total, exposure, var_dp_total, var_E_count


def survival_flat(x, interval_median_h):
    """P(no refresh for x hours) under a lognormal interval model.

    Computed with erfc so it does not underflow to exactly 0 for long
    flat stretches (the naive 1 - Phi form rounds to 0 past z ~ 8,
    which turns the renewal ratio 0/0 into a crash)."""
    if x <= 0:
        return 1.0
    z = (math.log(x) - math.log(interval_median_h)) / PRIOR_INTERVAL_SIGMA
    if z >= 0:
        return 0.5 * math.erfc(z / math.sqrt(2.0))
    return 1.0 - 0.5 * math.erfc(-z / math.sqrt(2.0))


def mixture_quantiles(mu_a, sd_a, mu_b, sd_b, p_no_refresh, levels):
    """Quantiles of a two-mode normal mixture on the logit scale."""
    def cdf(x):
        return (p_no_refresh * norm_cdf((x - mu_a) / sd_a)
                + (1.0 - p_no_refresh) * norm_cdf((x - mu_b) / sd_b))
    lo = min(mu_a, mu_b) - 8.0 * max(sd_a, sd_b) - 1.0
    hi = max(mu_a, mu_b) + 8.0 * max(sd_a, sd_b) + 1.0
    out = []
    for level in levels:
        a, b = lo, hi
        for _ in range(80):
            mid = 0.5 * (a + b)
            if cdf(mid) < level:
                a = mid
            else:
                b = mid
        out.append(0.5 * (a + b))
    return out


# ---------------------------------------------------------------- forecast

def forecast_case(case):
    levels = case.get("levels") or list(DEFAULT_LEVELS)
    horizon_min = case.get("horizon_minutes") or 0
    horizon_h = max(0.0, horizon_min / 60.0)
    horizon_d = horizon_h / 24.0
    trophies = case.get("trophies") or []
    n = len(trophies)
    states, checkpoint, _n = extract_states(case)
    if not states or n == 0:
        raise ValueError("no usable history")
    fit = fit_refresh_model(states, n, trophies)
    m = fit["m"]

    last_moment, p_last, _key = states[-1]
    if p_last is None:
        raise ValueError("no players-tracked count")
    if checkpoint is not None:
        t_since_first = max(0.0, hours_between(states[0][0], checkpoint))
        t_since_refresh = max(0.0, hours_between(last_moment, checkpoint))
    else:
        t_since_first = max(0.0, hours_between(states[0][0], last_moment))
        t_since_refresh = 0.0
    # How long the page has been flat.  With refreshes observed it is
    # the time since the last refresh; with none observed it is the
    # whole observed history plus the pre-history allowance (the page
    # predates the first read).
    if m >= 1:
        flat_h = t_since_refresh
    else:
        flat_h = t_since_first + PRE_HISTORY_H

    if m >= 1:
        lam = fit["lam"] or (1.0 / PRIOR_INTERVAL_H)
        interval_med = fit["interval_median_h"]
        dp_base = fit["dp_base"] or 0.0
        dp_sd = fit["dp_sd"] or (0.5 * dp_base if dp_base > 0 else 0.0)
        per_trophy = fit["per_trophy"]
    else:
        lam = 1.0 / PRIOR_INTERVAL_H
        interval_med = PRIOR_INTERVAL_H
        # Prior first refresh: players multiply by ~6.06 over ~4.5 h of
        # accumulation; scale the jump by the accumulation behind the
        # refresh we would see (flat so far + mid-horizon).
        accum = flat_h + 0.5 * horizon_h
        scale = min(2.0, max(0.25, accum / PRIOR_JUMP_ACCUM_H))
        dp_base = (PRIOR_JUMP_FACTOR - 1.0) * p_last * scale
        dp_sd = PRIOR_JUMP_CV * dp_base
        per_trophy = None

    expected_refreshes = lam * horizon_h
    denom = survival_flat(flat_h, interval_med)
    if horizon_h <= 0:
        p_no_refresh = 1.0
    elif denom <= 0.0:
        # The page is already flat far past any plausible interval: a
        # refresh before the target is (numerically) certain.
        p_no_refresh = 0.0
    else:
        p_no_refresh = survival_flat(flat_h + horizon_h, interval_med) / denom
    p_no_refresh = min(1.0, max(0.0, p_no_refresh))
    use_mixture = horizon_h > 0 and expected_refreshes < 2.0
    if use_mixture:
        n_b = max(1, int(round(expected_refreshes / (1.0 - p_no_refresh)))) \
            if p_no_refresh < 1.0 else 1
        # Conditioned on a refresh landing inside a short horizon there
        # can only be one or two of them (intervals are hours long).
        n_b = min(n_b, int(horizon_h / interval_med) + 2, 50)
    else:
        n_b = max(1, int(round(expected_refreshes))) if horizon_h > 0 else 0
        n_b = min(n_b, 500)

    # Future refresh sizes.  With enough observed refreshes, the
    # arrival rate nu(t) = exp(a + b t) is fitted from dP/dt and each
    # future jump is nu at its own time over one interval; otherwise
    # (few or no refreshes) the typical jump decays by 5% a refresh.
    if m >= 1 and fit.get("arrival"):
        a_nu, b_nu, sd_log_nu = fit["arrival"]
        dp_seq = []
        for k in range(n_b):
            mid = t_since_first + (k + 0.5) / lam
            dp_seq.append(max(0.0, math.exp(a_nu + b_nu * mid) / lam))
        # Never trust the arrival fit to 5%: the future rate is an
        # extrapolation, and real arrivals are lumpy.
        cv = max(sd_log_nu, 0.15)
        dp_sd = cv * (sum(dp_seq) / len(dp_seq) if dp_seq else 0.0)
    else:
        dp_seq = [dp_base * (DECAY_PER_REFRESH ** k) for k in range(n_b)]

    rows = []
    for i, trophy in enumerate(trophies):
        rate_last, _players = last_known(states, i)
        if rate_last is None:
            rate_last = 1.0
        e_last = rate_last / 100.0 * p_last
        saturated = rate_last >= 99.995
        if per_trophy is not None:
            q, h, sd_q, sd_h = per_trophy[i]
        elif saturated:
            q, h, sd_q, sd_h = 1.0, 0.0, 0.005, 0.0
        else:
            q = min(1.0, PRIOR_NEW_RATE * rate_last / 100.0)
            h, sd_q, sd_h = 0.0, sd_q_prior(q, dp_base), 0.0
        mu_a = to_logit(rate_last)
        if horizon_h <= 0 or (use_mixture and p_no_refresh >= 1.0):
            quantiles_logit = [mu_a + inv_normal(lv) * SIGMA_FROZEN
                               for lv in levels]
        else:
            P_t, E_t, dp_tot, expos, var_dp, var_e_count = project_counts(
                p_last, e_last, q, h, lam, dp_seq, dp_sd,
                horizon_h, n_b, use_mixture)
            if saturated and q >= 0.999 and h == 0.0:
                mu_b = to_logit(99.99)
                sd_b = 0.05
            else:
                E_safe = max(E_t, 0.05)
                P_safe = max(P_t, 1.0)
                mu_b = to_logit(100.0 * E_safe / P_safe)
                # Delta method on log(E/P): a bigger jump raises E and
                # P together, so its net effect is (q/E - 1/P), not
                # the sum of two independent errors.
                sens_dp = q / E_safe - 1.0 / P_safe
                var_logit = (
                    sens_dp ** 2 * var_dp
                    + (dp_tot / E_safe) ** 2 * sd_q ** 2
                    + (expos / E_safe) ** 2 * sd_h ** 2
                    + var_e_count / (E_safe ** 2)
                    + (SIGMA_DRIFT_PER_SQRT_DAY
                       * math.sqrt(horizon_d)) ** 2
                )
                if m == 0:
                    var_logit += PRIOR_ALLOC_SIGMA ** 2
                sd_b = min(SIGMA_CAP, max(SIGMA_FLOOR, math.sqrt(var_logit)))
            if use_mixture:
                quantiles_logit = mixture_quantiles(
                    mu_a, SIGMA_FROZEN, mu_b, sd_b, p_no_refresh, levels)
            else:
                quantiles_logit = [mu_b + inv_normal(lv) * sd_b
                                   for lv in levels]
        quantiles = []
        previous = None
        for value in quantiles_logit:
            pct = min(HI, max(LO, round4(from_logit(value))))
            if previous is not None and pct < previous:
                pct = previous
            quantiles.append(pct)
            previous = pct
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def persistence_fallback(case):
    """Last-rate forecast with a wide spread; only used if the model fails."""
    levels = case.get("levels") or list(DEFAULT_LEVELS)
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    last_rates = history[-1].get("rates") if history else None
    rows = []
    for i, trophy in enumerate(trophies):
        rate = None
        if last_rates and i < len(last_rates):
            rate = last_rates[i]
        centre = to_logit(rate if rate is not None else 1.0)
        quantiles = []
        previous = None
        for level in levels:
            value = min(HI, max(LO, round4(
                from_logit(centre + inv_normal(level) * 1.0))))
            if previous is not None and value < previous:
                value = previous
            quantiles.append(value)
            previous = value
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = persistence_fallback(case)
        out_path = OUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    run()
