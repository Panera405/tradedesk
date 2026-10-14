#!/usr/bin/env python3
"""
AI Trade Desk - swing-trade scanner (US stocks/ETFs + liquid options)
=====================================================================
Implements a 0-100 Trade Quality Score:
  Trend & structure 20 | Entry location 15 | Volume 10 | Relative strength 10
  Momentum 10 | Catalyst 10 | Risk/reward 15 | Options quality 10
  90+ Exceptional, 80-89 High quality, 70-79 Watchlist, <70 No trade.
Never manufactures a trade: below 80 it says NO TRADE and prints the prices
that would create a setup.

Setups: breakout + retest, high-volume breakout, failed breakout/breakdown, 20/50 EMA pullback,
support bounce / resistance rejection at pivot-cluster zones and unfilled gaps, relative-strength
continuation, plus pattern detection (bull/bear flag, cup and handle, flat base, triangles,
double bottom/top). Options: compares the standard monthly with the nearest weekly (~45 DTE).

Install:   pip install yfinance pandas numpy plotly
Run once:  python trade_desk.py
Hourly:    python trade_desk.py --watch            (alerts on new 80+ scores)
Options:   --equity 50000 --risk 0.01 --flow-max 25 --tickers-file list.txt --no-options

Alerts (optional, set environment variables):
  Telegram: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  Email/SMS-gateway: SMTP_HOST, SMTP_PORT(587), SMTP_USER, SMTP_PASS, ALERT_TO

DATA LIMITS (read this):
  * yfinance is free, delayed and unofficial. Fine for research, not for live execution.
  * "Whale flow" here is a PROXY built from free option-chain volume vs open interest,
    premium traded, stock volume and price action. True block trades and dark-pool
    prints need a paid feed - plug one into dark_pool_prints() below.
  * Fundamentals ("good company") are NOT scored by code.
This is a research tool, not financial advice.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import smtplib
import time
import urllib.parse
import urllib.request
from datetime import date, datetime
from email.message import EmailMessage
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

ET = ZoneInfo("America/New_York")

# --------------------------- configuration ---------------------------------
UNIVERSE = """
AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA LLY JPM V UNH XOM MA COST HD PG JNJ
ABBV NFLX BAC CRM ORCL AMD CVX KO MRK WMT PEP ADBE TMO LIN CSCO ACN MCD ABT
DIS QCOM TXN GE CAT IBM AMGN INTU ISRG GS MS NOW UBER BKNG SPGI HON PFE LOW
""".split()

SECTOR_ETF = {}
for _etf, _names in {
    "XLK": "AAPL MSFT NVDA AVGO ORCL AMD ADBE CRM CSCO ACN QCOM TXN IBM INTU NOW",
    "XLC": "GOOGL META NFLX DIS",
    "XLY": "AMZN TSLA HD MCD LOW BKNG",
    "XLF": "JPM V MA BAC GS MS SPGI",
    "XLV": "LLY UNH JNJ ABBV MRK TMO ABT AMGN ISRG PFE",
    "XLE": "XOM CVX",
    "XLP": "COST PG KO PEP WMT",
    "XLI": "GE CAT HON UBER",
    "XLB": "LIN",
}.items():
    for _t in _names.split():
        SECTOR_ETF[_t] = _etf

MIN_PRICE = 10.0
MIN_AVG_DOLLAR_VOLUME = 25_000_000
MAX_POSITION_PCT = 0.20
STAGE1_PREFILTER = 52          # of 80 technical points, before earnings/options lookups
SCORE_ACTIONABLE = 80
RISK_FREE = 0.04

NAMES = {
    1: {"pb": "20/50 EMA pullback", "bo_rt": "Breakout + retest",
        "bo_hv": "High-volume breakout", "bo": "Breakout (volume unconfirmed)",
        "rs": "Relative-strength continuation"},
    -1: {"pb": "Resistance rejection (20/50 EMA)", "bo_rt": "Breakdown + retest",
         "bo_hv": "High-volume breakdown", "bo": "Breakdown (volume unconfirmed)",
         "rs": "Relative-weakness continuation"},
}


# --------------------------- indicators ------------------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(c, n=14):
    dlt = c.diff()
    up = dlt.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-dlt.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def macd_hist(c):
    m = ema(c, 12) - ema(c, 26)
    return m - ema(m, 9)


def pivots(arr, k=4, mode="high"):
    out = []
    for i in range(k, len(arr) - k):
        w = arr[i - k:i + k + 1]
        if (mode == "high" and arr[i] == w.max()) or (mode == "low" and arr[i] == w.min()):
            out.append(i)
    return out


def clip01(x):
    return float(min(max(x, 0.0), 1.0))


def ret(s, n):
    return float(s.iloc[-1] / s.iloc[-n - 1] - 1)


def session_fraction(now=None):
    """Fraction of the regular US session elapsed (to project partial-day volume)."""
    now = now or datetime.now(ET)
    if now.weekday() >= 5:
        return 1.0
    o = now.replace(hour=9, minute=30, second=0, microsecond=0)
    c = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now < o or now >= c:
        return 1.0
    return max((now - o).total_seconds() / (6.5 * 3600), 0.15)


def market_open(now=None):
    now = now or datetime.now(ET)
    return now.weekday() < 5 and now.replace(hour=9, minute=30, second=0) <= now < now.replace(hour=16, minute=0, second=0)


# --------------------------- support/resistance zones & patterns -------------
NAMES[1].update({"fb": "Failed breakdown (bear trap)", "sb": "Support bounce (zone)",
                 "gb": "Gap support bounce"})
NAMES[-1].update({"fb": "Failed breakout (bull trap)", "sb": "Resistance rejection (zone)",
                  "gb": "Gap resistance rejection"})
PATTERN_NAMES = {
    1: {"flag": "Bull flag", "cup": "Cup and handle", "base": "Flat base breakout",
        "tri_asc": "Ascending triangle", "tri_sym": "Symmetrical triangle", "double": "Double bottom"},
    -1: {"flag": "Bear flag", "cup": "Inverted cup and handle", "base": "Flat base breakdown",
         "tri_asc": "Descending triangle", "tri_sym": "Symmetrical triangle", "double": "Double top"},
}


def sr_zones(df, a, lookback=250, k=4):
    """Cluster swing highs and lows (within 0.6 ATR) into zones. A zone needs 2+ touches.
    Highs and lows are pooled because broken resistance becomes support and vice versa."""
    h, l = df["High"].values[-lookback:], df["Low"].values[-lookback:]
    n = len(h)
    pts = sorted([(float(h[i]), i) for i in pivots(h, k, "high")]
                 + [(float(l[i]), i) for i in pivots(l, k, "low")])
    zones, cur = [], []

    def flush(cl):
        if len(cl) < 2:
            return
        prices = [p for p, _ in cl]
        lo, hi = min(prices), max(prices)
        if hi - lo < 0.3 * a:
            mid = (lo + hi) / 2
            lo, hi = mid - 0.15 * a, mid + 0.15 * a
        ages = [n - 1 - i for _, i in cl]
        zones.append({"lo": lo, "hi": hi, "mid": (lo + hi) / 2, "touches": len(cl),
                      "weight": float(sum(math.exp(-g / 100) for g in ages)), "kind": "zone"})

    for p in pts:
        if cur and p[0] - cur[0][0] > 0.6 * a:
            flush(cur)
            cur = []
        cur.append(p)
    flush(cur)
    return zones


def gap_zones(df, a, lookback=120):
    """Unfilled gaps of at least 0.6 ATR in the last ~6 months act as support/resistance."""
    h, l = df["High"].values, df["Low"].values
    out = []
    for i in range(max(1, len(h) - lookback), len(h)):
        if l[i] - h[i - 1] >= 0.6 * a and l[i:].min() > h[i - 1]:
            lo, hi = float(h[i - 1]), float(l[i])
        elif l[i - 1] - h[i] >= 0.6 * a and h[i:].max() < l[i - 1]:
            lo, hi = float(h[i]), float(l[i - 1])
        else:
            continue
        out.append({"lo": lo, "hi": hi, "mid": (lo + hi) / 2, "touches": 1, "weight": 1.0, "kind": "gap"})
    return out


def _near(cx_last, trig, ax):
    """Price may sit up to 1.5 ATR below the trigger (entry-location points shrink with distance)
    or up to 1.5 ATR beyond it (already broke out)."""
    dlt = cx_last - trig
    return (-1.5 * ax <= dlt <= 1.5 * ax), dlt > 0


# All detectors work in "x-space", where a bullish pattern always points up:
#   long: x = ln(price);  short: x = -ln(price) (so bear patterns become bull patterns).
def pat_flag(Hx, Lx, Cx, V, ax):
    n = len(Cx)
    if n < 60:
        return None
    top_idx = n - 22 + int(np.argmax(Hx[-22:]))
    flen = n - 1 - top_idx
    if not 4 <= flen <= 15:
        return None
    lo_start = max(0, top_idx - 15)
    low_idx = lo_start + int(np.argmin(Lx[lo_start:top_idx]))
    top, low = float(Hx[top_idx]), float(Lx[low_idx])
    pole = top - low
    if pole < max(0.12, 5 * ax) or top_idx - low_idx < 3:
        return None
    fH, fL = Hx[top_idx + 1:], Lx[top_idx + 1:]
    if not 0.08 <= (top - float(fL.min())) / pole <= 0.45:
        return None
    if np.polyfit(np.arange(flen), Cx[top_idx + 1:], 1)[0] > 0.1 * ax:
        return None
    if V[top_idx + 1:].mean() > 0.9 * V[low_idx:top_idx + 1].mean():
        return None
    # trigger = the flag's upper trendline projected one bar ahead (never above the highest flag high)
    sl, ic = np.polyfit(np.arange(flen), fH, 1)
    trig = float(min(fH.max(), sl * flen + ic))
    ok, broke = _near(Cx[-1], trig, ax)
    if not ok:
        return None
    return {"trig": trig, "sup": float(fL.min()), "height": float(pole), "broke": broke,
            "seg": [(low_idx, low, top_idx, top, "pole")], "hx": []}


def pat_cup(Hx, Lx, Cx, ax):
    n = len(Cx)
    for hl in (5, 8, 12, 16, 20):
        for W in range(40, min(220, n - hl - 1), 10):
            end = n - hl
            st = end - W
            sH, sL, sC = Hx[st:end], Lx[st:end], Cx[st:end]
            q = W // 4
            li = int(np.argmax(sH[:q]))
            left = float(sH[li])
            rr = int(np.argmax(sH[-q:]))
            right = float(sH[-q:][rr])
            lo_idx = q // 2 + int(np.argmin(sL[q // 2:W - q // 2]))
            low = float(sL[lo_idx])
            depth = left - low
            if not 0.12 <= depth <= 0.45 or abs(right - left) > 0.3 * depth:
                continue
            if not 0.2 * W <= lo_idx <= 0.8 * W:
                continue
            if float((sC < low + 0.3 * depth).mean()) < 0.12:      # rounded bottom, not a V
                continue
            hmin = float(Lx[end:].min())
            rim = max(left, right)
            if rim - hmin > 0.5 * depth or hmin < low + 0.5 * depth:
                continue
            ok, broke = _near(Cx[-1], rim, ax)
            if not ok:
                continue
            return {"trig": rim, "sup": hmin, "height": depth, "broke": broke,
                    "seg": [(st + li, left, st + W - q + rr, right, "cup rim")],
                    "hx": [(hmin, "handle low")]}
    return None


def pat_base(Hx, Lx, Cx, ax):
    for L in (25, 35):
        top, bot = float(Hx[-L:].max()), float(Lx[-L:].min())
        rng = top - bot
        if not max(0.02, 2 * ax) <= rng <= 0.10 or Cx[-1] < bot + 0.5 * rng:
            continue
        ok, broke = _near(Cx[-1], top, ax)
        if ok:
            return {"trig": top, "sup": bot, "height": rng, "broke": broke, "seg": [], "hx": []}
    return None


def pat_triangle(Hx, Lx, Cx, ax):
    n = len(Cx)
    st = n - 70
    if st < 0:
        return None
    ph = [st + i for i in pivots(Hx[st:], 3, "high")][-3:]
    pl = [st + i for i in pivots(Lx[st:], 3, "low")][-3:]
    if len(ph) < 2 or len(pl) < 2 or ph[-1] - ph[0] < 12 or pl[-1] - pl[0] < 12:
        return None
    mh, bh = np.polyfit(ph, Hx[ph], 1)
    ml, bl = np.polyfit(pl, Lx[pl], 1)
    first, last = min(ph[0], pl[0]), n - 1
    hs, he = mh * first + bh, mh * last + bh
    ls, le = ml * first + bl, ml * last + bl
    w0, w1 = hs - ls, he - le
    if w0 <= 0 or w1 <= 0 or w1 > 0.7 * w0:
        return None
    dh, dl = he - hs, le - ls
    if abs(dh) <= 0.015 and dl >= 0.02:
        sub, trig = "tri_asc", float(Hx[ph].max())
    elif dh <= -0.02 and dl >= 0.02:
        sub, trig = "tri_sym", float(he)
    else:
        return None
    ok, broke = _near(Cx[-1], trig, ax)
    if not ok:
        return None
    return {"sub": sub, "trig": trig, "sup": float(le), "height": float(w0), "broke": broke,
            "seg": [(first, float(hs), last, float(he), "upper"), (first, float(ls), last, float(le), "lower")],
            "hx": []}


def pat_double(Hx, Lx, Cx, ax):
    n = len(Cx)
    st = max(0, n - 100)
    pl = [st + i for i in pivots(Lx[st:], 4, "low")]
    for j in range(len(pl) - 1, 0, -1):
        i2 = pl[j]
        if n - 1 - i2 > 30:
            break
        for m in range(j - 1, -1, -1):
            i1 = pl[m]
            if not 10 <= i2 - i1 <= 80 or abs(float(Lx[i1] - Lx[i2])) > 0.03:
                continue
            neck = float(Hx[i1:i2 + 1].max())
            lowest = float(min(Lx[i1], Lx[i2]))
            if neck - max(float(Lx[i1]), float(Lx[i2])) < 0.06:
                continue
            ok, broke = _near(Cx[-1], neck, ax)
            if ok:
                return {"trig": neck, "sup": lowest, "height": neck - lowest, "broke": broke,
                        "seg": [(i1, float(Lx[i1]), i2, float(Lx[i2]), "double low")], "hx": []}
    return None


def detect_patterns(df, d, a):
    """Return setup candidates for flags, cup & handle, flat base, triangles, double bottom/top."""
    h, l, c, v = (df[x].values for x in ("High", "Low", "Close", "Volume"))
    n = len(c)
    Hx = np.log(h) if d == 1 else -np.log(l)
    Lx = np.log(l) if d == 1 else -np.log(h)
    Cx = d * np.log(c)
    ax = a / float(c[-1])
    px = lambda x: float(math.exp(d * x))
    names = PATTERN_NAMES[d]
    out = []
    dets = [("flag", pat_flag(Hx, Lx, Cx, v, ax)), ("cup", pat_cup(Hx, Lx, Cx, ax)),
            ("base", pat_base(Hx, Lx, Cx, ax)), ("tri", pat_triangle(Hx, Lx, Cx, ax)),
            ("double", pat_double(Hx, Lx, Cx, ax))]
    for key, det in dets:
        if not det:
            continue
        name = names[det.get("sub", key)]
        trig, sup = px(det["trig"]), px(det["sup"])
        out.append({"key": "pat_" + key, "name": name, "kind": "breakout", "ref": trig, "stop_ref": sup,
                    "pre_break": not det["broke"], "measured": px(det["trig"] + det["height"]),
                    "vol": "expand" if det["broke"] else "dry", "trig_label": name,
                    "geom": {"h": [(trig, "trigger"), (sup, "pattern support")]
                                  + [(px(x), lab) for x, lab in det["hx"]],
                             "seg": [(n - 1 - i0, px(x0), n - 1 - i1, px(x1), lab)
                                     for i0, x0, i1, x1, lab in det["seg"]]}})
    return out


# --------------------------- core technical scoring -------------------------
def score_stock(df, d, spy, qqq, sec, now=None):
    """Score one ticker in direction d (+1 long, -1 short) on the 80 technical points.
    Every valid setup candidate is scored; the best one is reported (others listed as 'also').
    Returns a dict (setup may be None) or None if the name fails hard filters."""
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(df) < 260:
        return None
    now = now or datetime.now(ET)
    frac = session_fraction(now)
    if frac < 1 and df.index[-1].date() == now.date():
        df.iloc[-1, df.columns.get_loc("Volume")] = df["Volume"].iloc[-1] / frac

    c, h, l, o, v = (df[x] for x in ("Close", "High", "Low", "Open", "Volume"))
    close = float(c.iloc[-1])
    e20, s50, s200 = ema(c, 20), c.rolling(50).mean(), c.rolling(200).mean()
    e20v, s50v, s200v = float(e20.iloc[-1]), float(s50.iloc[-1]), float(s200.iloc[-1])
    a = float(atr(df).iloc[-1])
    avg_dv = float((c * v).rolling(20).mean().iloc[-1])
    if not a > 0 or close < MIN_PRICE or avg_dv < MIN_AVG_DOLLAR_VOLUME:
        return None

    # ---- Trend & structure (20) ----
    t = 0.0
    t += 3 * (d * (close - e20v) > 0)
    t += 3 * (d * (e20v - s50v) > 0)
    t += 3 * (d * (s50v - s200v) > 0)
    t += 2 * (d * (s50v - float(s50.iloc[-11])) > 0)
    t += 2 * (d * (s200v - float(s200.iloc[-21])) > 0)
    wk = c.resample("W").last().dropna()
    w10, w30 = wk.rolling(10).mean(), wk.rolling(30).mean()
    t += 2 * (d * (float(wk.iloc[-1]) - float(w10.iloc[-1])) > 0)
    t += 2 * (d * (float(w10.iloc[-1]) - float(w30.iloc[-1])) > 0)
    hv, lv = h.values[-160:], l.values[-160:]
    ph, pl = pivots(hv, 4, "high"), pivots(lv, 4, "low")
    t += 1.5 * (len(ph) >= 2 and d * (hv[ph[-1]] - hv[ph[-2]]) > 0)
    t += 1.5 * (len(pl) >= 2 and d * (lv[pl[-1]] - lv[pl[-2]]) > 0)

    # ---- Support/resistance zones (pivot clusters + unfilled gaps) ----
    zones = sr_zones(df, a) + gap_zones(df, a)
    near_z = sorted([z for z in zones if z["hi"] > close - 10 * a and z["lo"] < close + 10 * a],
                    key=lambda z: -z["weight"])[:8]
    below = [z for z in near_z if z["hi"] <= close + 0.2 * a]
    above = [z for z in near_z if z["lo"] >= close - 0.2 * a]
    sup_zone = max(below, key=lambda z: z["hi"]) if below else None
    res_zone = min(above, key=lambda z: z["lo"]) if above else None

    hi20, lo20 = float(h.tail(20).max()), float(l.tail(20).min())
    base = {"dir": d, "close": close, "atr": a, "trend": round(t, 1), "e20": e20v,
            "s50": s50v, "s200": s200v, "avg_dv": avg_dv, "setup": None,
            "zones": near_z, "sup_zone": sup_zone, "res_zone": res_zone,
            "watch_pullback": (round(min(e20v, s50v), 2), round(max(e20v, s50v), 2)),
            "watch_break": round(hi20 if d == 1 else lo20, 2)}
    if t < 11:
        return None
    ext = d * (close - e20v) / a
    if ext > 2.5:
        base["flags"] = ["EXTENDED from 20 EMA"]
        return base

    # ---- Relative strength (10) ----
    rs_spy = d * (ret(c, 63) - ret(spy, 63))
    rs_sec = d * (ret(c, 63) - ret(sec, 63))
    rs_qqq = d * (ret(c, 21) - ret(qqq, 21))
    rs_pts = 4 * clip01(rs_spy / 0.10) + 4 * clip01(rs_sec / 0.08) + 2 * clip01(rs_qqq / 0.05)

    # ---- Volume context, momentum ----
    vr = float(v.iloc[-1] / v.iloc[-21:-1].mean())
    r3 = float(v.iloc[-3:].mean() / v.iloc[-21:-1].mean())
    seg = df.tail(20)
    dd = seg["Close"].diff() * d
    acc = float(seg["Volume"][dd > 0].sum() / (seg["Volume"][dd < 0].sum() + 1))
    acc_pts = 5 * clip01((acc - 0.8) / 0.7)

    r = float(rsi(c).iloc[-1])
    r = r if d == 1 else 100 - r
    flags = []
    mom = 0.0
    if 45 <= r <= 68:
        mom += 5
    elif 40 <= r < 45 or 68 < r <= 75:
        mom += 2.5
    elif r > 75:
        flags.append("OVEREXTENDED momentum (RSI)")
    mh = macd_hist(c)
    mom += 3 * (d * (float(mh.iloc[-1]) - float(mh.iloc[-4])) > 0)
    mom += 2 * (d * ret(c, 21) > 0)

    # ---- Setup candidates ----
    lows3, highs3 = float(l.iloc[-3:].min()), float(h.iloc[-3:].max())
    bar_up, bar_dn = close > float(o.iloc[-1]), close < float(o.iloc[-1])
    cands = []

    level = float(h.iloc[-23:-3].max()) if d == 1 else float(l.iloc[-23:-3].min())
    broke = bool((d * (c.iloc[-3:] - level) > 0).any()) and d * (close - level) > 0
    ext_lvl = d * (close - level) / a
    if broke and ext_lvl <= 1.5:
        retest = lows3 <= level + 0.5 * a if d == 1 else highs3 >= level - 0.5 * a
        key = "bo_rt" if retest else ("bo_hv" if vr >= 1.5 else "bo")
        cands.append({"key": key, "name": NAMES[d][key], "kind": "breakout", "ref": level, "vol": "expand"})

    # failed breakout (short) / failed breakdown (long): a fresh trap that reversed
    if d == -1:
        ref_ = float(h.iloc[-26:-4].max())
        poke = float(h.iloc[-4:].max())
        if poke > ref_ and close < ref_ and close < float(c.iloc[-2]) and ref_ - close <= 1.5 * a:
            cands.append({"key": "fb", "name": NAMES[d]["fb"], "kind": "breakout", "ref": ref_,
                          "stop_ref": poke, "vol": "reversal"})
    else:
        ref_ = float(l.iloc[-26:-4].min())
        poke = float(l.iloc[-4:].min())
        if poke < ref_ and close > ref_ and close > float(c.iloc[-2]) and close - ref_ <= 1.5 * a:
            cands.append({"key": "fb", "name": NAMES[d]["fb"], "kind": "breakout", "ref": ref_,
                          "stop_ref": poke, "vol": "reversal"})

    def touched(series):
        lvl = series.iloc[-3:]
        return lows3 <= lvl.max() + 0.3 * a if d == 1 else highs3 >= lvl.min() - 0.3 * a

    pulled = ((hi20 - close) if d == 1 else (close - lo20)) / a
    if (touched(e20) or touched(s50)) and d * (close - s50v) > 0 and -1.0 <= ext <= 1.2 and pulled >= 1.0:
        ref = e20v if abs(close - e20v) <= abs(close - s50v) else s50v
        cands.append({"key": "pb", "name": NAMES[d]["pb"], "kind": "pullback", "ref": ref, "vol": "dry"})

    # bounce / rejection at a horizontal zone (pivot cluster or unfilled gap)
    zl = []
    for z in near_z:
        if d == 1 and z["mid"] <= close and lows3 <= z["hi"] + 0.25 * a and close >= z["mid"] \
                and close - z["hi"] <= 1.0 * a and bar_up:
            zl.append((close - z["mid"], z))
        elif d == -1 and z["mid"] >= close and highs3 >= z["lo"] - 0.25 * a and close <= z["mid"] \
                and z["lo"] - close <= 1.0 * a and bar_dn:
            zl.append((z["mid"] - close, z))
    if zl:
        z = min(zl, key=lambda x: x[0])[1]
        cands.append({"key": "sb", "name": NAMES[d]["gb" if z["kind"] == "gap" else "sb"], "kind": "pullback",
                      "ref": z["mid"], "stop_ref": z["lo"] if d == 1 else z["hi"], "vol": "dry"})

    cands += detect_patterns(df, d, a)

    hi252, lo252 = float(h.tail(252).max()), float(l.tail(252).min())
    near_extreme = close >= 0.97 * hi252 if d == 1 else close <= 1.03 * lo252
    tight = (float(h.tail(10).max()) - float(l.tail(10).min())) / close <= 0.07
    if near_extreme and tight and rs_spy > 0.03 and ext <= 1.5:
        cands.append({"key": "rs", "name": NAMES[d]["rs"], "kind": "rs", "ref": e20v, "vol": "rs"})

    if not cands:
        base["flags"] = ["no valid setup"]
        return base

    prev_extreme = float(h.iloc[-2]) if d == 1 else float(l.iloc[-2])

    def make_plan(cd):
        ref = cd["ref"]
        dist = abs(close - ref) / a
        entry_pts = 15 * clip01(1 - (dist - 0.5) / 2.5)
        rule = cd["vol"]
        if rule == "expand":
            vp = 5 * clip01((vr - 1) / 0.6)
        elif rule == "dry":
            vp = 5 * clip01((1.1 - r3) / 0.4)
        elif rule == "reversal":
            vp = 5 * clip01((vr - 1) / 0.5)
        else:
            vp = 5 * clip01((vr - 0.8) / 0.6)
        vol_pts = acc_pts + vp

        pre = bool(cd.get("pre_break"))
        entry = ref if pre else close
        if not pre and d * (prev_extreme - close) > 0 and d * (prev_extreme - close) / a <= 1.0:
            entry = prev_extreme
        if cd.get("stop_ref") is not None:
            stop = cd["stop_ref"] - d * 0.25 * a
        else:
            stop = (float(l.tail(8).min()) - 0.25 * a) if d == 1 else (float(h.tail(8).max()) + 0.25 * a)
        risk = d * (entry - stop)
        if risk <= 0:
            return None
        if risk < 0.75 * a:
            stop, risk = entry - d * 0.75 * a, 0.75 * a
        if risk > 3.5 * a:
            return None

        # targets: nearest S/R zone edge, then pattern measured move / next zone / 3R
        if d == 1:
            zt = sorted(z["lo"] for z in near_z if z["lo"] > entry + 0.5 * risk)
        else:
            zt = sorted((z["hi"] for z in near_z if z["hi"] < entry - 0.5 * risk), reverse=True)
        piv = ([hv[i] for i in ph if hv[i] > entry + 0.5 * risk] if d == 1
               else [lv[i] for i in pl if lv[i] < entry - 0.5 * risk])
        if zt:
            t1 = zt[0]
        elif piv:
            t1 = min(piv) if d == 1 else max(piv)
        else:
            t1 = entry + d * 2 * risk
        meas, t2 = cd.get("measured"), None
        if meas is not None and d * (meas - t1) > 0.2 * risk and d * (meas - entry) <= 8 * risk:
            t2 = meas
        if t2 is None:
            beyond = [z for z in zt[1:] if d * (z - t1) > 0.5 * risk]
            t2 = beyond[0] if beyond and d * (beyond[0] - entry) <= 6 * risk else entry + d * 3 * risk
        if d * (t2 - t1) <= 0:
            t2 = t1 + d * risk
        stretch = entry + d * 5 * risk
        if d * (stretch - t2) <= 0:
            stretch = t2 + d * risk
        rr = d * (t1 - entry) / risk
        rr_pts = max(0.0, min(15.0, (rr - 1) * 10 if rr < 2 else 10 + (min(rr, 3) - 2) * 5))

        est = d * (t1 - entry) / a / 0.4
        hold = f"{max(2, round(est * 0.7))}-{min(20, round(est * 1.4) + 2)} trading days"
        if pre:
            zone = (entry, entry + 0.3 * a) if d == 1 else (entry - 0.3 * a, entry)
            trig = (f"{'Buy' if d == 1 else 'Short'} on a break {'above' if d == 1 else 'below'} "
                    f"{ref:.2f} ({cd['trig_label']}) with volume (stop order)")
        else:
            zone = (entry - 0.4 * a, entry + 0.1 * a) if d == 1 else (entry - 0.1 * a, entry + 0.4 * a)
            trig = (f"Buy on a trade above {prev_extreme:.2f} (prior-day high) with volume"
                    if d == 1 else f"Short on a trade below {prev_extreme:.2f} (prior-day low) with volume")
        stage1 = t + entry_pts + vol_pts + rs_pts + mom + rr_pts
        return {"setup": cd["name"], "key": cd["key"], "kind": cd["kind"], "entry": entry, "zone": zone,
                "trigger": trig, "stop": stop, "t1": t1, "t2": t2, "stretch": stretch, "rr": rr,
                "risk_ps": risk, "hold": hold, "level": ref, "geom": cd.get("geom"),
                "comp": {"trend": round(t, 1), "entry": round(entry_pts, 1), "volume": round(vol_pts, 1),
                         "rs": round(rs_pts, 1), "momentum": round(mom, 1), "rr": round(rr_pts, 1)},
                "stage1": round(stage1, 1)}

    plans = [p for p in (make_plan(cd) for cd in cands) if p]
    if not plans:
        base["flags"] = ["setup found but stop too wide for a clean trade"]
        return base
    best = max(plans, key=lambda p: p["stage1"])
    base.update(best)
    base.update({"also": [p["setup"] for p in plans if p is not best], "vr": vr, "rsi": r, "ext": ext,
                 "rs_spy": rs_spy, "rs_sec": rs_sec, "rs_qqq": rs_qqq, "flags": flags,
                 "hv30": float(c.pct_change().tail(30).std() * math.sqrt(252))})
    return base


# --------------------------- catalysts & options ----------------------------
def next_earnings(tk):
    try:
        cal = yf.Ticker(tk).calendar
        eds = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if eds:
            fut = sorted((x.date() if isinstance(x, datetime) else x) for x in eds)
            fut = [x for x in fut if x >= date.today()]
            return fut[0] if fut else None
    except Exception:
        pass
    return None


def catalyst_points(ed):
    if ed is None:
        return 5.0, None, "earnings date unknown"
    days = (ed - date.today()).days
    if days <= 5:
        return 0.0, days, f"EARNINGS IN {days}d - binary risk"
    if days <= 14:
        return 2.0, days, f"earnings in {days}d (inside hold window)"
    if days <= 28:
        return 6.0, days, f"earnings in {days}d (late in hold window)"
    return 10.0, days, f"no earnings for {days}d"


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _npdf(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def bs_greeks(S, K, T, iv, call=True, r=RISK_FREE):
    if T <= 0 or iv <= 0:
        return None
    sq = iv * math.sqrt(T)
    d1 = (math.log(S / K) + (r + iv * iv / 2) * T) / sq
    d2 = d1 - sq
    decay = -S * _npdf(d1) * iv / (2 * math.sqrt(T))
    if call:
        return _ncdf(d1), (decay - r * K * math.exp(-r * T) * _ncdf(d2)) / 365
    return abs(_ncdf(d1) - 1), (decay + r * K * math.exp(-r * T) * _ncdf(-d2)) / 365


def _is_monthly(exp_str):
    dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
    return dt.weekday() == 4 and 15 <= dt.day <= 21     # third Friday


def _eval_expiry(t, exp, dte, d, S, hv30, earn, delta_goal):
    """Find the contract nearest the target delta in one expiry and grade its liquidity (10 pts)."""
    ch = t.option_chain(exp)
    tab = ch.calls if d == 1 else ch.puts
    best = None
    for r in tab.itertuples():
        iv, bid, ask = r.impliedVolatility, r.bid, r.ask
        if not (bid > 0 and ask > 0 and iv >= 0.05):
            continue
        g = bs_greeks(S, r.strike, dte / 365, iv, call=(d == 1))
        if not g:
            continue
        gap = abs(g[0] - delta_goal)
        if best is None or gap < best[0]:
            best = (gap, r, g)
    if not best:
        return None
    _, r, (delta, theta) = best
    mid = (r.bid + r.ask) / 2
    sp = (r.ask - r.bid) / mid
    oi = float(r.openInterest) if pd.notna(r.openInterest) else 0.0
    vol = float(r.volume) if pd.notna(r.volume) else 0.0
    iv_hv = r.impliedVolatility / hv30 if hv30 > 0 else 1.3
    pts = (4 * clip01((0.12 - sp) / 0.09) + 3 * clip01(oi / 500)
           + 1 * clip01(vol / 50) + 2 * clip01((1.6 - iv_hv) / 0.6))
    notes = []
    illiquid = oi < 100 or sp > 0.08
    if illiquid:
        notes.append("ILLIQUID contract - do not trade it; use shares or pick another strike")
    if earn and earn <= datetime.strptime(exp, "%Y-%m-%d").date():
        notes.append("earnings before expiry: expect IV crush")
    return {"exp": exp, "dte": dte, "monthly": _is_monthly(exp), "strike": float(r.strike),
            "type": "CALL" if d == 1 else "PUT", "delta": round(delta, 2),
            "theta_day": round(theta * 100, 2), "mid": round(mid, 2), "spread_pct": round(sp * 100, 1),
            "oi": int(oi), "volume": int(vol), "iv": round(r.impliedVolatility * 100, 1),
            "iv_vs_hv": round(iv_hv, 2), "exp_move": round(S * r.impliedVolatility * math.sqrt(dte / 365), 2),
            "pts": round(pts, 1), "illiquid": illiquid, "note": "; ".join(notes), "alt": ""}


def option_plan(tk, d, S, hv30, earn=None, dte_min=35, dte_goal=45, delta_goal=0.60):
    """Compare the expiry nearest the target with the nearest standard monthly (third Friday)
    within 14 days of it, and keep whichever has the better liquidity score."""
    try:
        t = yf.Ticker(tk)
        today = date.today()
        exps = []
        for e in t.options:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
            if dte >= dte_min:
                exps.append((e, dte))
        if not exps:
            return None
        picks = [min(exps, key=lambda x: abs(x[1] - dte_goal))]
        monthlies = [x for x in exps if _is_monthly(x[0]) and abs(x[1] - dte_goal) <= 14]
        if monthlies:
            picks.append(min(monthlies, key=lambda x: abs(x[1] - dte_goal)))
        plans = []
        for e, dte in {p[0]: p for p in picks}.values():
            p = _eval_expiry(t, e, dte, d, S, hv30, earn, delta_goal)
            if p:
                plans.append(p)
        if not plans:
            return None
        best = max(plans, key=lambda p: (p["pts"], p["monthly"]))
        for o in plans:
            if o is not best:
                best["alt"] = (f"also checked {o['exp']} ({'monthly' if o['monthly'] else 'weekly'}), "
                               f"liquidity {o['pts']}/10")
        return best
    except Exception:
        return None


# --------------------------- whale-flow proxy -------------------------------
def dark_pool_prints(ticker):
    """HOOK: return recent block/dark-pool prints from a paid feed (Unusual Whales,
    Polygon, Tradier, FlowAlgo...). Free data has none, so this returns None."""
    return None


def accumulation_zone(df, a):
    seg = df.tail(10)
    avg = df["Volume"].rolling(20).mean().shift(1).tail(10)
    mask = (seg["Volume"] > 1.2 * avg) & (seg["Close"] > seg["Open"])
    use = seg[mask] if mask.sum() >= 2 else seg
    tp = (use["High"] + use["Low"] + use["Close"]) / 3
    w = float((tp * use["Volume"]).sum() / use["Volume"].sum())
    return w - 0.5 * a, w + 0.5 * a


def flow_scan(tk, S, max_exp=3, max_dte=60):
    try:
        t = yf.Ticker(tk)
        today = date.today()
        exps = [e for e in t.options
                if 0 <= (datetime.strptime(e, "%Y-%m-%d").date() - today).days <= max_dte][:max_exp]
        call_p = put_p = 0.0
        n, top = 0, []
        seen = with_vol = near = 0
        for e in exps:
            ch = t.option_chain(e)
            for side, tab in (("C", ch.calls), ("P", ch.puts)):
                for r in tab.itertuples():
                    vol = float(r.volume) if pd.notna(r.volume) else 0.0
                    oi = float(r.openInterest) if pd.notna(r.openInterest) else 0.0
                    px = float(r.lastPrice) if pd.notna(r.lastPrice) else 0.0
                    otm = r.strike > S if side == "C" else r.strike < S
                    prem = vol * px * 100
                    seen += 1
                    with_vol += vol > 0
                    near += bool(otm and vol >= 100 and prem >= 50_000)
                    if otm and vol >= 200 and vol >= max(oi, 1) and prem >= 100_000:
                        n += 1
                        if side == "C":
                            call_p += prem
                        else:
                            put_p += prem
                        top.append((prem, e, side, float(r.strike), int(vol), int(oi)))
        return {"call_prem": call_p, "put_prem": put_p, "net": call_p - put_p,
                "count": n, "top": sorted(top, reverse=True)[:3],
                "seen": seen, "with_vol": with_vol, "near": near}
    except Exception:
        return None


def flow_rank(data, tickers, flow_max, now):
    """Top 10 'accumulation' names by a transparent confidence formula (0-100)."""
    frac = session_fraction(now)
    cands = []
    for tk in tickers:
        try:
            df = data[tk].dropna(subset=["Close"])
            v = df["Volume"].copy()
            if frac < 1 and df.index[-1].date() == now.date():
                v.iloc[-1] /= frac
            vr = float(v.iloc[-1] / v.iloc[-21:-1].mean())
            dv = float((df["Close"] * df["Volume"]).tail(20).mean())
            cands.append((vr, dv, tk))
        except Exception:
            continue
    active = [x for x in cands if x[0] >= 1.15]
    cands = sorted(active, key=lambda x: -x[1])[:flow_max]
    diag = {"tickers_checked": len(tickers), "active_volume": len(active), "chains_scanned": 0,
            "chain_failures": 0, "contracts_seen": 0, "contracts_with_volume": 0,
            "near_misses": 0, "with_unusual": 0, "net_bullish": 0}
    rows = []
    for vr, dv, tk in cands:
        df = data[tk].dropna(subset=["Close"])
        S = float(df["Close"].iloc[-1])
        fl = flow_scan(tk, S)
        time.sleep(0.3)
        if not fl:
            diag["chain_failures"] += 1
            continue
        diag["chains_scanned"] += 1
        diag["contracts_seen"] += fl["seen"]
        diag["contracts_with_volume"] += fl["with_vol"]
        diag["near_misses"] += fl["near"]
        diag["with_unusual"] += fl["count"] > 0
        if fl["net"] <= 0:
            continue
        diag["net_bullish"] += 1
        a = float(atr(df).iloc[-1])
        last = df.iloc[-1]
        rng = float(last["High"] - last["Low"])
        upper = rng > 0 and (float(last["Close"]) - float(last["Low"])) / rng >= 0.6
        above = S > float(ema(df["Close"], 20).iloc[-1])
        conf = (35 * clip01(fl["net"] / 2e6) + 20 * clip01(fl["count"] / 6)
                + 20 * clip01((vr - 1) / 1.0) + 15 * upper + 10 * above)
        lo, hi = accumulation_zone(df, a)
        rows.append({"ticker": tk, "confidence": round(conf), "price": round(S, 2),
                     "net_premium": fl["net"], "unusual_contracts": fl["count"],
                     "vol_vs_avg": round(vr, 2), "zone": (round(lo, 2), round(hi, 2)),
                     "top_prints": fl["top"], "dark_pool": dark_pool_prints(tk)})
    return sorted(rows, key=lambda r: -r["confidence"])[:10], diag


# --------------------------- pipeline ---------------------------------------
def regime(closes):
    spy = closes["SPY"].dropna()
    s50, s200 = spy.rolling(50).mean(), spy.rolling(200).mean()
    px = float(spy.iloc[-1])
    off = px < float(s200.iloc[-1]) and px < float(s50.iloc[-1])
    rising = float(s50.iloc[-1]) > float(s50.iloc[-10])
    vix = closes["^VIX"].dropna().iloc[-1] if "^VIX" in closes else float("nan")
    if off:
        txt = "RISK-OFF: SPY below 50d and 200d. Longs need 85+; shorts get priority."
    elif px > float(s200.iloc[-1]) and rising:
        txt = "RISK-ON: SPY above a rising 50d and above 200d."
    else:
        txt = "NEUTRAL/CHOPPY: SPY trend not clean. Be selective."
    return off, f"{txt} (SPY {px:.2f}, VIX {vix:.1f})"


def run_scan(tickers, equity, risk_pct, flow_max, use_options, now=None, dte_min=35, dte_goal=45):
    now = now or datetime.now(ET)
    sec_etfs = set(SECTOR_ETF.get(t, "SPY") for t in tickers)
    allt = sorted(set(tickers) | {"SPY", "QQQ", "^VIX"} | sec_etfs)
    print(f"[{now:%Y-%m-%d %H:%M ET}] downloading {len(allt)} symbols...")
    data = yf.download(allt, period="2y", auto_adjust=True, group_by="ticker",
                       progress=False, threads=True)
    closes = pd.DataFrame({t: data[t]["Close"] for t in allt if t in data.columns.get_level_values(0)})
    risk_off, reg_txt = regime(closes)
    spy, qqq = closes["SPY"].dropna(), closes["QQQ"].dropna()

    stage1, watch = [], []
    for tk in tickers:
        try:
            df = data[tk]
            sec = closes[SECTOR_ETF.get(tk, "SPY")].dropna()
            for d in (1, -1):
                r = score_stock(df, d, spy, qqq, sec, now)
                if not r:
                    continue
                r["ticker"] = tk
                (stage1 if r.get("setup") else watch).append(r)
        except Exception:
            continue

    plans = []
    for r in sorted(stage1, key=lambda x: -x["stage1"]):
        if r["stage1"] < STAGE1_PREFILTER:
            r["total"] = r["stage1"]
            r["tier"] = "No trade"
            plans.append(r)
            continue
        ed = next_earnings(r["ticker"])
        cat_pts, days, cat_txt = catalyst_points(ed)
        if days is not None and days <= 5:
            r["flags"].append("EARNINGS WITHIN 5 DAYS")
        op = (option_plan(r["ticker"], r["dir"], r["close"], r["hv30"], ed, dte_min=dte_min, dte_goal=dte_goal)
              if use_options else None)
        opt_pts = op["pts"] if op else 0.0
        r["comp"]["catalyst"], r["comp"]["options"] = cat_pts, opt_pts
        r["catalyst_txt"], r["earnings"], r["option"] = cat_txt, ed, op
        r["total"] = round(r["stage1"] + cat_pts + opt_pts, 1)
        r["tier"] = ("Exceptional" if r["total"] >= 90 else "High quality" if r["total"] >= 80
                     else "Watchlist" if r["total"] >= 70 else "No trade")
        shares = min(math.floor(equity * risk_pct / r["risk_ps"]),
                     math.floor(equity * MAX_POSITION_PCT / r["entry"]))
        r["shares"], r["position"] = shares, shares * r["entry"]
        r["risk_dollars"] = shares * r["risk_ps"]
        need = 85 if (risk_off and r["dir"] == 1) else SCORE_ACTIONABLE
        r["actionable"] = r["total"] >= need and not (days is not None and days <= 5)
        plans.append(r)
        time.sleep(0.2)
    plans.sort(key=lambda x: -x["total"])

    flow, flow_diag = (flow_rank(data, tickers, flow_max, now)
                       if use_options and flow_max > 0 else ([], None))
    return {"time": now, "regime": reg_txt, "risk_off": risk_off, "plans": plans,
            "watch": sorted(watch, key=lambda x: -x["trend"]), "flow": flow,
            "flow_diag": flow_diag, "data": data}


# --------------------------- reporting --------------------------------------
def fmt_plan(p):
    d = "LONG" if p["dir"] == 1 else "SHORT"
    o = p.get("option")
    cm = p["comp"]
    chart = "GOOD CHART" if p["trend"] >= 14 else "MIXED CHART"
    trade = "GOOD TRADE" if p["total"] >= 80 else "NOT YET A GOOD TRADE"
    L = [f"{p['ticker']} - {d} - {p['total']:.1f}/100 ({p['tier']})   [{chart} | {trade}]",
         f"  Price {p['close']:.2f} | Setup: {p['setup']}",
         f"  Why: trend {cm['trend']}/20, entry {cm['entry']}/15, volume {cm['volume']}/10, RS {cm['rs']}/10 "
         f"(vs SPY {p['rs_spy']*100:+.1f}%, sector {p['rs_sec']*100:+.1f}%), momentum {cm['momentum']}/10, "
         f"catalyst {cm.get('catalyst', 0)}/10, R/R {cm['rr']}/15, options {cm.get('options', 0)}/10",
         f"  Entry zone {p['zone'][0]:.2f}-{p['zone'][1]:.2f} | Trigger: {p['trigger']}",
         f"  Stop/invalidation {p['stop']:.2f} | T1 {p['t1']:.2f} | T2 {p['t2']:.2f} | Stretch {p['stretch']:.2f}",
         f"  Reward/risk to T1: {p['rr']:.1f}:1 | Hold ~{p['hold']} | Catalyst: {p.get('catalyst_txt', 'n/a')}",
         f"  Abandon if: closes beyond {p['stop']:.2f}, loses the 50 SMA ({p['s50']:.2f}), "
         f"SPY breaks its 200d, or earnings/news arrive inside the hold window.",
         f"  Shares: {p.get('shares', 0)} sh (~${p.get('position', 0):,.0f}, risking ${p.get('risk_dollars', 0):,.0f})"]
    if o:
        L.append(f"  Option: {o['exp']} {o['strike']:.0f} {o['type']} | delta {o['delta']} | mid {o['mid']} | "
                 f"theta/day {o['theta_day']} per contract | spread {o['spread_pct']}% | OI {o['oi']} | "
                 f"vol {o['volume']} | IV {o['iv']}% ({o['iv_vs_hv']}x 30d realized) | "
                 f"1-sigma move to expiry +/-{o['exp_move']} {o['note']}")
    else:
        L.append("  Option: none graded (no liquid chain found or options disabled)")
    st = []
    if p.get("sup_zone"):
        st.append(f"support {_zone_txt(p['sup_zone'])}")
    if p.get("res_zone"):
        st.append(f"resistance {_zone_txt(p['res_zone'])}")
    if st:
        L.append("  Structure: " + " | ".join(st))
    if p.get("also"):
        L.append("  Also matches: " + ", ".join(p["also"]))
    if p["flags"]:
        L.append("  FLAGS: " + "; ".join(p["flags"]))
    return "\n".join(L)


def watch_line(w):
    side = "long" if w["dir"] == 1 else "short"
    pb = w["watch_pullback"]
    act = "buy" if w["dir"] == 1 else "short a rally into"
    brk = "close above" if w["dir"] == 1 else "close below"
    z = w.get("sup_zone") if w["dir"] == 1 else w.get("res_zone")
    zt = f" or {'support' if w['dir'] == 1 else 'resistance'} zone {_zone_txt(z)}" if z else ""
    fl = f" [{'; '.join(w['flags'])}]" if w.get("flags") else ""
    return (f"{w['ticker']} ({side}, trend {w['trend']}/20) at {w['close']:.2f}: {act} {pb[0]}-{pb[1]} "
            f"(20EMA/50SMA){zt} or {brk} {w['watch_break']} on volume{fl}")


def flow_lines(res):
    out = [f"{f['ticker']} conf {f['confidence']} | px {f['price']} | net unusual call premium "
           f"${f['net_premium']/1e6:.2f}M | {f['unusual_contracts']} unusual contracts | "
           f"vol {f['vol_vs_avg']}x | est. accumulation zone {f['zone'][0]}-{f['zone'][1]}"
           for f in res["flow"]]
    d = res.get("flow_diag")
    if d is None:
        out.append("Flow scan disabled.")
    else:
        out.append(f"Diagnostic: {d['active_volume']}/{d['tickers_checked']} tickers had elevated volume; "
                   f"{d['chains_scanned']} option chains scanned ({d['chain_failures']} failed); "
                   f"{d['contracts_seen']} contracts seen, {d['contracts_with_volume']} with nonzero volume; "
                   f"{d['with_unusual']} tickers with unusual prints, {d['net_bullish']} net bullish; "
                   f"{d['near_misses']} near-miss contracts at half-size thresholds.")
    return out


def print_report(res, detail=False):
    plans = res["plans"]
    act = [p for p in plans if p.get("actionable")]
    print("\n" + "=" * 78)
    print(f"TRADE DESK  {res['time']:%Y-%m-%d %H:%M ET}\nMarket: {res['regime']}")
    print("=" * 78)
    if not act:
        print("\n#1 TRADE TODAY: NO TRADE - nothing scores 80+ at current prices.")
    else:
        print(f"\n#1 TRADE TODAY:\n{fmt_plan(act[0])}")
    for label, pred in (("BEST PULLBACK SETUP", lambda p: p["dir"] == 1 and p["kind"] == "pullback"),
                        ("BEST BREAKOUT SETUP", lambda p: p["kind"] == "breakout" and p["dir"] == 1),
                        ("BEST SHORT SETUP", lambda p: p["dir"] == -1)):
        pool = [p for p in plans if pred(p) and "total" in p]
        top = pool[0] if pool else None
        if top and top.get("actionable"):
            print(f"\n{label}:\n{fmt_plan(top)}")
        elif top:
            print(f"\n{label}: none at 80+. Closest: {top['ticker']} {top['total']:.1f}/100 "
                  f"({top['setup']}), entry {top['zone'][0]:.2f}-{top['zone'][1]:.2f}, stop {top['stop']:.2f}")
        else:
            print(f"\n{label}: none found.")
    print("\nTOP 5 BY SCORE:")
    for i, p in enumerate(plans[:5], 1):
        print(f"  {i}. {p['ticker']} {'LONG' if p['dir']==1 else 'SHORT'} {p['total']:.1f} - {p['setup']} ({p['tier']})")
    if detail:
        print("\nDETAIL (full plan and component scores for the top 5):")
        for p in plans[:5]:
            print("\n" + fmt_plan(p))
    print("\nNO-TRADE WATCHLIST (good chart, no acceptable entry yet):")
    for w in res["watch"][:8]:
        print("  " + watch_line(w))
    print("\nWHALE-FLOW PROXY - top accumulation candidates (free-data proxy, not true dark-pool data):")
    for line in flow_lines(res):
        print("  " + line)


# --------------------------- dashboard --------------------------------------
def _fmt_money(x):
    return f"${x:,.0f}"


def _zone_txt(z):
    tag = "gap" if z["kind"] == "gap" else f"{z['touches']} touches"
    return f"{z['lo']:.2f}-{z['hi']:.2f} ({tag})"


def plan_alert_text(p):
    """Short, labeled phone-friendly text for Telegram/email alerts."""
    d = "LONG" if p["dir"] == 1 else "SHORT"
    o, L = p.get("option"), []
    L += [f"{p['ticker']}  {d}  |  {p['total']:.1f}/100 ({p['tier']})",
          f"Setup: {p['setup']}", f"Price now: {p['close']:.2f}", "",
          f"Entry zone: {p['zone'][0]:.2f} - {p['zone'][1]:.2f}",
          f"Trigger: {p['trigger']}",
          f"Stop: {p['stop']:.2f}",
          f"Target 1: {p['t1']:.2f}  |  Target 2: {p['t2']:.2f}  |  Stretch: {p['stretch']:.2f}",
          f"Reward/risk to T1: {p['rr']:.1f}:1  |  Hold: ~{p['hold']}",
          f"Catalyst: {p.get('catalyst_txt', 'n/a')}", ""]
    st = []
    if p.get("sup_zone"):
        st.append(f"Support {_zone_txt(p['sup_zone'])}")
    if p.get("res_zone"):
        st.append(f"Resistance {_zone_txt(p['res_zone'])}")
    if st:
        L.append("Structure: " + " | ".join(st))
    if p.get("also"):
        L.append("Also matches: " + ", ".join(p["also"]))
    if p.get("shares"):
        L.append(f"Shares: {p['shares']} (~{_fmt_money(p['position'])}), risking {_fmt_money(p['risk_dollars'])}")
    if o:
        L.append(f"Option: {o['exp']} {o['strike']:.0f} {o['type']}, delta {o['delta']}, mid {o['mid']}, "
                 f"spread {o['spread_pct']}%, OI {o['oi']}, IV {o['iv']}%"
                 + (f" ({o['note']})" if o["note"] else ""))
    L += ["", f"Abandon if: closes beyond {p['stop']:.2f}, loses the 50 SMA ({p['s50']:.2f}), "
              f"SPY breaks its 200d, or earnings/news hit inside the hold."]
    if p["flags"]:
        L.append("WARNING: " + "; ".join(p["flags"]))
    return "\n".join(L)


def plan_card_html(p):
    """One readable card per trade plan for the dashboard."""
    import html as H
    long_ = p["dir"] == 1
    side_col = "#2ecc71" if long_ else "#e74c3c"
    tier_col = {"Exceptional": "#2ecc71", "High quality": "#2ecc71", "Watchlist": "#f5a623"}.get(p["tier"], "#888")
    comp = p["comp"]

    bars = ""
    for lab, key, mx in (("Trend", "trend", 20), ("Entry", "entry", 15), ("Volume", "volume", 10),
                         ("Rel. strength", "rs", 10), ("Momentum", "momentum", 10),
                         ("Catalyst", "catalyst", 10), ("Reward/risk", "rr", 15), ("Options", "options", 10)):
        if key not in comp:
            bars += (f"<div class='bar'><span>{lab}</span><div class='track'></div>"
                     f"<b>n/a</b></div>")
            continue
        v = float(comp[key])
        pct = 100 * v / mx
        c = "#2ecc71" if pct >= 70 else "#f5a623" if pct >= 40 else "#e74c3c"
        bars += (f"<div class='bar'><span>{lab}</span><div class='track'><i style='width:{pct:.0f}%;"
                 f"background:{c}'></i></div><b>{v:.1f}/{mx}</b></div>")

    zone_mid = (p["zone"][0] + p["zone"][1]) / 2
    levels = [("Stretch", p["stretch"], f"{p['stretch']:.2f}", "#27ae60"),
              ("Target 2", p["t2"], f"{p['t2']:.2f}", "#27ae60"),
              ("Target 1", p["t1"], f"{p['t1']:.2f}", "#2ecc71"),
              ("Entry zone", zone_mid, f"{p['zone'][0]:.2f} - {p['zone'][1]:.2f}", "#4a90e2"),
              ("Price now", p["close"], f"{p['close']:.2f}", "#dddddd"),
              ("Stop", p["stop"], f"{p['stop']:.2f}", "#e74c3c")]
    levels.sort(key=lambda x: -x[1])
    ladder = "".join(
        f"<div class='lvl' style='border-left:4px solid {c}'><span>{lab}</span><b>{txt}</b>"
        f"<em>{(v / p['close'] - 1) * 100:+.1f}%</em></div>" for lab, v, txt, c in levels)

    facts = [("Setup", p["setup"]), ("Trigger", p["trigger"]), ("Expected hold", p["hold"]),
             ("Reward/risk to T1", f"{p['rr']:.1f} : 1"), ("Catalyst", p.get("catalyst_txt", "not evaluated")),
             ("Chart / Trade", f"{'GOOD' if p['trend'] >= 14 else 'MIXED'} chart, "
                               f"{'GOOD' if p['total'] >= 80 else 'not yet a good'} trade")]
    if p.get("sup_zone"):
        facts.append(("Nearest support", _zone_txt(p["sup_zone"])))
    if p.get("res_zone"):
        facts.append(("Nearest resistance", _zone_txt(p["res_zone"])))
    if p.get("also"):
        facts.append(("Also matches", ", ".join(p["also"])))
    grid = "".join(f"<div><small>{H.escape(k)}</small><br>{H.escape(str(v))}</div>" for k, v in facts)

    warn = ""
    if p["flags"]:
        warn = "<div class='warn'>&#9888; " + H.escape("; ".join(p["flags"])) + "</div>"

    shares = ""
    if p.get("shares"):
        shares = (f"<div class='box'><small>SHARES</small><br><b>{p['shares']}</b> sh "
                  f"(~{_fmt_money(p['position'])})<br>risking {_fmt_money(p['risk_dollars'])}</div>")
    o = p.get("option")
    opt = ""
    if o:
        warn_o = ""
        if o.get("illiquid"):
            warn_o = f"<br><span style='color:#e74c3c'><b>{H.escape(o['note'])}</b></span>"
        elif o["note"]:
            warn_o = f"<br><span style='color:#f5a623'>{H.escape(o['note'])}</span>"
        opt = (f"<div class='box'><small>OPTION ({'monthly' if o.get('monthly') else 'weekly'})</small><br>"
               f"<b>{o['exp']} {o['strike']:.0f} {o['type']}</b><br>"
               f"delta {o['delta']} | mid {o['mid']} | spread {o['spread_pct']}%<br>"
               f"OI {o['oi']} | vol {o['volume']} | IV {o['iv']}% ({o['iv_vs_hv']}x realized)<br>"
               f"theta {o['theta_day']}/day | 1-sigma move +/-{o['exp_move']}" + warn_o
               + (f"<br><small>{H.escape(o['alt'])}</small>" if o.get("alt") else "") + "</div>")
    elif "options" in comp:
        opt = "<div class='box'><small>OPTION</small><br>No liquid contract found</div>"

    return (f"<div class='card'><div class='head'><h4>{p['ticker']}</h4>"
            f"<span class='badge' style='background:{side_col}'>{'LONG' if long_ else 'SHORT'}</span>"
            f"<span class='score'>{p['total']:.1f}<small>/100</small></span>"
            f"<span class='badge' style='background:{tier_col}'>{p['tier']}</span></div>{warn}"
            f"<div class='sec'>Score breakdown</div>{bars}"
            f"<div class='sec'>Price ladder</div>{ladder}"
            f"<div class='sec'>Key facts</div><div class='facts'>{grid}</div>"
            f"<div class='abandon'><b>Abandon if:</b> closes beyond {p['stop']:.2f}, loses the 50 SMA "
            f"({p['s50']:.2f}), SPY breaks its 200-day, or earnings/news arrive inside the hold window.</div>"
            f"<div class='boxes'>{shares}{opt}</div></div>")


def make_dashboard(res, path):
    import html as H
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data, plans = res["data"], res["plans"]
    shown = [p for p in plans if p.get("actionable")][:5]
    for p in plans:                       # always chart at least the top 3 by score
        if len(shown) >= 3:
            break
        if "zone" in p and all(p is not q for q in shown):
            shown.append(p)
    parts, first = [], True
    for p in shown:
        df = data[p["ticker"]].dropna(subset=["Close"])
        c = df["Close"]
        view = df.tail(130)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=view.index, open=view["Open"], high=view["High"], low=view["Low"],
                                     close=view["Close"], name=p["ticker"]), 1, 1)
        for name, ser, col in (("EMA20", ema(c, 20), "#f5a623"), ("SMA50", c.rolling(50).mean(), "#4a90e2"),
                               ("SMA200", c.rolling(200).mean(), "#bd10e0")):
            fig.add_trace(go.Scatter(x=view.index, y=ser.loc[view.index], name=name,
                                     line=dict(width=1.3, color=col)), 1, 1)
        fig.add_hrect(y0=p["zone"][0], y1=p["zone"][1], fillcolor="#2ecc71", opacity=0.18, line_width=0, row=1, col=1)
        for z in p.get("zones", []):
            zc = "#9b59b6" if z["kind"] == "gap" else ("#2ecc71" if z["mid"] < p["close"] else "#e74c3c")
            fig.add_hrect(y0=z["lo"], y1=z["hi"], fillcolor=zc, opacity=0.10, line_width=0, row=1, col=1,
                          annotation_text="gap" if z["kind"] == "gap" else f"S/R x{z['touches']}",
                          annotation_position="top left", annotation_font_size=9)
        g = p.get("geom") or {}
        for gp, lab in g.get("h", []):
            fig.add_hline(y=gp, line_dash="dot", line_color="#f1c40f", annotation_text=lab,
                          annotation_position="bottom right", annotation_font_size=9, row=1, col=1)
        for bb0, p0, bb1, p1, lab in g.get("seg", []):
            if bb0 < len(view) and bb1 < len(view):
                fig.add_trace(go.Scatter(x=[df.index[len(df) - 1 - bb0], df.index[len(df) - 1 - bb1]],
                                         y=[p0, p1], mode="lines", line=dict(color="#f1c40f", width=2),
                                         name=lab), 1, 1)
        for lab, y, col in (("STOP", p["stop"], "#e74c3c"), ("T1", p["t1"], "#2ecc71"),
                            ("T2", p["t2"], "#27ae60")):
            fig.add_hline(y=y, line_dash="dash", line_color=col, annotation_text=f"{lab} {y:.2f}", row=1, col=1)
        fig.add_trace(go.Bar(x=view.index, y=view["Volume"], name="Volume", marker_color="#7f8c8d"), 2, 1)
        d = "LONG" if p["dir"] == 1 else "SHORT"
        fig.update_layout(template="plotly_dark", height=620, xaxis_rangeslider_visible=False,
                          title=f"{p['ticker']} {d} - {p['total']:.1f}/100 ({p['tier']}) - {p['setup']} "
                                f"(R/R {p['rr']:.1f}:1)", margin=dict(l=40, r=20, t=60, b=30))
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False))
        first = False

    def row(p):
        return (f"<tr><td>{p['ticker']}</td><td>{'LONG' if p['dir']==1 else 'SHORT'}</td><td>{p['total']:.1f}</td>"
                f"<td>{p['tier']}</td><td>{p['setup']}</td><td>{p['zone'][0]:.2f}-{p['zone'][1]:.2f}</td>"
                f"<td>{p['stop']:.2f}</td><td>{p['t1']:.2f}</td><td>{p['rr']:.1f}</td>"
                f"<td>{H.escape(p.get('catalyst_txt', ''))}</td></tr>")

    table = "".join(row(p) for p in plans[:10] if "zone" in p)
    banner = ("" if any(p.get("actionable") for p in plans)
              else "<h2 style='color:#e67e22'>NO TRADE: nothing scores 80+ right now.</h2>")
    detail = "".join(plan_card_html(p) for p in plans[:5] if "zone" in p)
    watch = "".join(f"<li>{H.escape(watch_line(w))}</li>" for w in res["watch"][:8])
    flow = "".join(f"<li>{H.escape(x)}</li>" for x in flow_lines(res))
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Trade Desk</title>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<style>body{{background:#111;color:#ddd;font-family:system-ui,sans-serif;margin:16px;max-width:1200px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #333;padding:6px 8px;text-align:left}}
th{{color:#999}}pre{{background:#1b1b1b;padding:12px;white-space:pre-wrap;font-size:13px}}
.wrap{{overflow-x:auto}}h3{{margin-top:28px}}li{{margin:4px 0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}}
.card{{background:#1b1b1b;border:1px solid #2c2c2c;border-radius:10px;padding:14px}}
.head{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}.head h4{{margin:0;font-size:22px}}
.badge{{color:#111;font-weight:700;font-size:12px;padding:3px 8px;border-radius:12px}}
.score{{margin-left:auto;font-size:26px;font-weight:700}}.score small{{font-size:13px;color:#888}}
.sec{{color:#888;font-size:12px;text-transform:uppercase;letter-spacing:.06em;margin:14px 0 6px}}
.bar{{display:grid;grid-template-columns:96px 1fr 58px;gap:8px;align-items:center;font-size:13px;margin:3px 0}}
.bar b{{text-align:right;font-weight:600}}.track{{background:#2a2a2a;height:8px;border-radius:4px;overflow:hidden}}
.track i{{display:block;height:100%}}
.lvl{{display:grid;grid-template-columns:90px 1fr auto;gap:8px;padding:5px 8px;margin:3px 0;background:#222;font-size:14px}}
.lvl em{{color:#999;font-style:normal;font-size:12px}}
.facts{{display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:13px}}.facts small{{color:#888}}
.warn{{background:#3a2a10;border:1px solid #f5a623;color:#f5c26b;padding:8px;border-radius:6px;margin-top:10px;font-size:13px}}
.abandon{{background:#2a1717;border:1px solid #5a2a2a;padding:8px;border-radius:6px;margin-top:12px;font-size:13px}}
.boxes{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:12px}}
.box{{background:#222;padding:8px;border-radius:6px;font-size:13px}}.box small{{color:#888}}</style></head><body>
<h1>Trade Desk</h1><p>Updated {res['time']:%Y-%m-%d %H:%M ET}. {H.escape(res['regime'])}</p>{banner}
<div class='wrap'><table><tr><th>Ticker</th><th>Dir</th><th>Score</th><th>Tier</th><th>Setup</th><th>Entry zone</th>
<th>Stop</th><th>T1</th><th>R/R</th><th>Catalyst</th></tr>{table}</table></div>
<h3>Top plans in detail</h3><div class='cards'>{detail}</div>
<h3>No-trade watchlist (prices that would create a setup)</h3><ul>{watch}</ul>
<h3>Whale-flow proxy (not true dark-pool data)</h3><ul>{flow}</ul>
<h3>Charts</h3>{''.join(parts)}
<p style='color:#777'>Research tool, not financial advice. Free delayed data.</p></body></html>"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard written to {path}")


# --------------------------- alerts & watch loop -----------------------------
def notify(subject, body):
    print(f"\n*** ALERT: {subject} ***\n{body}\n")
    tok, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if tok and chat:
        try:
            q = urllib.parse.urlencode({"chat_id": chat, "text": f"{subject}\n\n{body}"[:4000]}).encode()
            urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", q, timeout=15)
        except Exception as e:
            print("telegram failed:", e)
    host, to = os.getenv("SMTP_HOST"), os.getenv("ALERT_TO")
    if host and to:
        try:
            m = EmailMessage()
            m["Subject"], m["From"], m["To"] = subject, os.getenv("SMTP_USER", to), to
            m.set_content(body)
            with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=20) as s:
                s.starttls()
                s.login(os.getenv("SMTP_USER", ""), os.getenv("SMTP_PASS", ""))
                s.send_message(m)
        except Exception as e:
            print("email failed:", e)


def check_alerts(res, state_path="alerts_state.json"):
    try:
        state = json.load(open(state_path))
    except Exception:
        state = {}
    today = f"{res['time']:%Y-%m-%d}"
    state = {k: v for k, v in state.items() if k.startswith(today)}
    for p in res["plans"]:
        if not p.get("actionable"):
            continue
        key = f"{today}|{p['ticker']}|{p['dir']}|{p['setup']}"
        if key in state:
            continue
        state[key] = f"{res['time']:%H:%M}"
        notify(f"{p['ticker']} {'LONG' if p['dir']==1 else 'SHORT'} scores {p['total']:.1f}", plan_alert_text(p))
    json.dump(state, open(state_path, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description="AI Trade Desk")
    ap.add_argument("--equity", type=float, default=25_000)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--flow-max", type=int, default=25, help="max tickers to option-flow scan (0=off)")
    ap.add_argument("--tickers-file")
    ap.add_argument("--no-options", action="store_true")
    ap.add_argument("--watch", action="store_true", help="run hourly during market hours and alert on 80+")
    ap.add_argument("--interval-min", type=int, default=60)
    ap.add_argument("--out", default="dashboard.html")
    ap.add_argument("--dte-goal", type=int, default=45, help="target days to option expiration")
    ap.add_argument("--dte-min", type=int, default=35, help="minimum days to option expiration")
    ap.add_argument("--detail", action="store_true", help="print full plan + component scores for top 5")
    ap.add_argument("--state", default="alerts_state.json", help="alert de-duplication file")
    ap.add_argument("--only-market-hours", action="store_true", help="exit quietly if the US market is closed")
    a = ap.parse_args()
    tickers = open(a.tickers_file).read().split() if a.tickers_file else UNIVERSE
    tickers = [t.upper() for t in tickers]

    def once():
        res = run_scan(tickers, a.equity, a.risk, a.flow_max, not a.no_options,
                       dte_min=a.dte_min, dte_goal=a.dte_goal)
        print_report(res, a.detail)
        try:
            make_dashboard(res, a.out)
        except ImportError:
            print("pip install plotly to enable the chart dashboard")
        check_alerts(res, a.state)

    if not a.watch:
        if a.only_market_hours and not market_open():
            print('US market closed - skipping scan.')
            return
        once()
        return
    print("Watch mode: scanning every", a.interval_min, "min during US market hours. Ctrl+C to stop.")
    while True:
        if market_open():
            try:
                once()
            except Exception as e:
                print("scan failed:", e)
            time.sleep(a.interval_min * 60)
        else:
            time.sleep(300)


if __name__ == "__main__":
    main()
