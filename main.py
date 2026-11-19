"""Forecast GTA VI trophy earn rates on Exophase as five quantiles per trophy.

Model, per trophy, in plain words:

  * Work on the logit scale.  A rate r (percent) becomes ln(r / (100 - r)),
    after clamping r into [0.01, 99.99] so 0% and 100% still have a logit.
  * Time runs on ln(1 + hours since the release day began).  Log time keeps a
    4.5-day extrapolation bounded instead of letting a straight line in days
    run away.
  * Fit a least-squares line of logit(rate) against log time over the most
    recent 24 hours of reads, then shrink that trophy's slope toward the
    median slope across trophies: a trophy with few reads borrows the global
    pattern, one with many reads keeps its own.  The intercept is refitted
    to the shrunk slope so the line still passes through the trophy's own
    recent level.
  * A second, structural view: fit log(earners) and log(players tracked)
    against log time and divide the two forecasts.  When new players flood
    in faster than earners, that view predicts the rate falling -- the
    dilution a refresh brings.  The final median is the average of the two
    views in logit space, so neither can dominate.
  * Spread: a random-walk term growing with sqrt(days of horizon), a
    refresh-jump term of the same shape, and the trophy's own fit residual
    (shrunk toward the global residual).  Combined in quadrature, floored so
    even a flat history admits that a refresh can move rates, and capped.

Everything is deterministic: no randomness, no wall clock, no network.
Only the standard library.  Input comes from $MAKEFUN_INPUT_DIR (else
/input), output goes to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
LO = 0.01
HI = 99.99

# Random-walk spread per sqrt(day) on the logit scale, and the extra spread
# allowed for Exophase refresh jumps, which arrive as steps, not smoothly.
WALK_SIGMA = 0.5
JUMP_SIGMA = 0.45
SIGMA_FLOOR = 0.30
SIGMA_CAP = 2.2
# How many reads a trophy's own slope is worth against the global slope.
SHRINK_PRIOR = 2.0
RESID_FLOOR = 0.08
RESID_CAP = 0.80
# Blend weight of the direct rate curve vs the earners/players view.
CURVE_WEIGHT = 0.75
# Recent window for the fits, in hours.
WINDOW_HOURS = 24.0


def clamp_rate(pct):
    """Keep a percent inside the range where the logit is defined."""
    if pct is None:
        return None
    return LO if pct < LO else HI if pct > HI else pct


def to_logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def from_logit(value):
    return 100.0 / (1.0 + math.exp(-value))


def parse_moment(text):
    """ISO 8601 moment -> aware datetime in UTC; 'Z' included."""
    if text is None:
        return None
    if isinstance(text, str) and text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def release_origin(case):
    """Midnight UTC of the game's release day, from the case itself."""
    game = case.get("game") or {}
    day = game.get("release_date")
    if isinstance(day, str) and len(day) >= 10:
        try:
            base = datetime.fromisoformat(day[:10])
            return base.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    reads = case.get("history") or []
    if reads:
        first = parse_moment(reads[0].get("at"))
        if first is not None:
            return first.replace(hour=0, minute=0, second=0, microsecond=0)
    return parse_moment(case.get("at"))


def log_hours(moment, origin):
    """ln(1 + hours from origin to moment); reads before origin stay valid."""
    hours = (moment - origin).total_seconds() / 3600.0
    if hours < -0.99:
        hours = -0.99
    return math.log1p(hours)


def inv_normal(level):
    """Standard normal quantile by Acklam's rational approximation."""
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


def least_squares(xs, ys):
    """Slope, intercept and residual RMS of y on x; None when degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = math.sqrt(sum((y - (intercept + slope * x)) ** 2
                          for x, y in zip(xs, ys)) / n)
    return slope, intercept, resid


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def collect_series(case):
    """Per-trophy (x, logit rate) points plus the players series.

    Returns (origin, target_x, horizon_days, trophies, per_trophy, players)
    where per_trophy[i] is a list of (x, logit, earners_or_None) and
    players is a list of (x, log players).  Reads lacking a trophy's rate
    are skipped for that trophy only.
    """
    origin = release_origin(case)
    target = parse_moment(case.get("target"))
    if target is None:
        target = parse_moment(case.get("at"))
    target_x = log_hours(target, origin)
    horizon_days = max(0.0, (case.get("horizon_minutes") or 0) / 1440.0)
    trophies = case.get("trophies") or []
    per_trophy = [[] for _ in trophies]
    players = []
    for read in case.get("history") or []:
        moment = parse_moment(read.get("at"))
        if moment is None:
            continue
        x = log_hours(moment, origin)
        tracked = read.get("players_tracked")
        if isinstance(tracked, (int, float)) and tracked > 0:
            players.append((x, math.log(tracked)))
        rates = read.get("rates") or []
        for i in range(len(trophies)):
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                continue
            earners = None
            if isinstance(tracked, (int, float)) and tracked > 0:
                earners = max(0.0, clamp_rate(rate) / 100.0 * tracked)
            per_trophy[i].append((x, to_logit(rate), earners))
    return origin, target_x, horizon_days, trophies, per_trophy, players


def recent(points, last_x, window=WINDOW_HOURS):
    """Points from the recent window; all points when too few fall in it."""
    if not points:
        return points
    # x is ln(1 + hours), so a 24 h window is not a fixed x gap; recover
    # hours from x and keep reads whose hour is within the window of the
    # latest read's hour.
    last_h = math.expm1(last_x)
    kept = [p for p in points if last_h - math.expm1(p[0]) <= window]
    return kept if len(kept) >= 3 else points


def refresh_break(players):
    """x of the first read after the last refresh jump, or None.

    A refresh shows up as one interval's players-tracked count leaping
    (organic growth between two 10-30 minute reads is a few percent at
    most).  Everything before the leap belongs to the old denominator
    regime: a rate curve fitted across it reads the step as a slope.
    """
    if len(players) < 3:
        return None
    moves = [players[i + 1][1] - players[i][1]
             for i in range(len(players) - 1)]
    typical = median([m for m in moves if m > 0]) if any(m > 0 for m in moves) else 0.0
    cut = None
    for i, move in enumerate(moves):
        if move > 0.08 and move > 2.5 * max(typical, 0.01):
            cut = players[i + 1][0]
    return cut


def forecast_trophy(points, players, target_x, horizon_days,
                    global_slope, global_resid, jump_scale, fallback_logit,
                    break_x=None):
    """Median logit and spread for one trophy at the target."""
    if not points:
        return fallback_logit, max(SIGMA_FLOOR, 1.0)
    if break_x is not None:
        after = [p for p in points if p[0] >= break_x]
        if len(after) >= 3:
            points = after
        players = [p for p in players if p[0] >= (break_x or -1e9)]
    last_x = points[-1][0]
    windowed = recent(points, last_x)
    windowed_players = recent(players, last_x)
    xs = [p[0] for p in windowed]
    ys = [p[1] for p in windowed]
    fit = least_squares(xs, ys)
    n = len(xs)
    if fit is None:
        # One distinct time only: no slope of its own, borrow the global one.
        slope = global_slope
        level = ys[-1]
        resid = global_resid
        median_logit = level + slope * (target_x - xs[-1])
    else:
        slope, _intercept, resid = fit
        weight = n / (n + SHRINK_PRIOR)
        shrunk = weight * slope + (1.0 - weight) * global_slope
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        intercept = mean_y - shrunk * mean_x
        median_logit = intercept + shrunk * target_x
        resid = (n * resid + SHRINK_PRIOR * global_resid) / (n + SHRINK_PRIOR)
    resid = min(RESID_CAP, max(RESID_FLOOR, resid))

    # Structural view: earners over players.  Fit log earners and log
    # players against log time, extrapolate both, divide.
    earners_view = None
    earned = [(p[0], math.log(p[2])) for p in windowed if p[2] and p[2] > 0]
    if len(earned) >= 2 and len(windowed_players) >= 2:
        e_fit = least_squares([p[0] for p in earned], [p[1] for p in earned])
        p_fit = least_squares([p[0] for p in windowed_players],
                              [p[1] for p in windowed_players])
        if e_fit is not None and p_fit is not None:
            e_pred = e_fit[1] + e_fit[0] * target_x
            p_pred = p_fit[1] + p_fit[0] * target_x
            rate_view = 100.0 * math.exp(e_pred - p_pred)
            if 0.0 < rate_view < 100.0:
                earners_view = to_logit(rate_view)
    if earners_view is not None:
        median_logit = (CURVE_WEIGHT * median_logit
                        + (1.0 - CURVE_WEIGHT) * earners_view)

    days = horizon_days
    spread = math.sqrt(
        (WALK_SIGMA * math.sqrt(days)) ** 2
        + (JUMP_SIGMA * math.sqrt(days)) ** 2
        + resid ** 2
        + (0.5 * jump_scale) ** 2
    )
    spread = min(SIGMA_CAP, max(SIGMA_FLOOR, spread))
    return median_logit, spread


def forecast_case(case):
    """Full forecast object for one case."""
    levels = case.get("levels") or list(DEFAULT_LEVELS)
    origin, target_x, horizon_days, trophies, per_trophy, players = \
        collect_series(case)

    # Global slope: median of the per-trophy slopes that exist.
    slopes = []
    resids = []
    for points in per_trophy:
        if len(points) < 2:
            continue
        fit = least_squares([p[0] for p in points], [p[1] for p in points])
        if fit is not None:
            slopes.append(fit[0])
            resids.append(fit[2])
    global_slope = median(slopes) if slopes else 0.0
    global_resid = median(resids) if resids else 0.15

    # Jump scale: RMS of consecutive-read logit moves across trophies.
    jumps = []
    for points in per_trophy:
        for a, b in zip(points, points[1:]):
            jumps.append(b[1] - a[1])
    jump_scale = math.sqrt(sum(j * j for j in jumps) / len(jumps)) \
        if jumps else 0.0

    # Fallback level: the median of the trophies' last rates.
    lasts = [p[-1][1] for p in per_trophy if p]
    fallback_logit = median(lasts) if lasts else to_logit(1.0)
    break_x = refresh_break(players)

    rows = []
    for i, trophy in enumerate(trophies):
        med, spread = forecast_trophy(
            per_trophy[i], players, target_x, horizon_days,
            global_slope, global_resid, jump_scale, fallback_logit,
            break_x=break_x)
        quantiles = []
        previous = None
        for level in levels:
            value = from_logit(med + inv_normal(level) * spread)
            value = min(HI, max(LO, value))
            value = round(value, 4)
            if previous is not None and value < previous:
                value = previous
            quantiles.append(value)
            previous = value
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
            value = min(HI, max(LO, round(
                from_logit(centre + inv_normal(level) * 1.0), 4)))
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
