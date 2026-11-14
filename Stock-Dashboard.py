#!/usr/bin/env python3
"""
NSE MARKET BREADTH DASHBOARD - SINGLE-FILE COLAB VERSION.

Paste this whole file into one Google Colab cell and run it. Everything is
included: pip install, the NSE downloads, the breadth maths, the backtested
zone levels and the rendered dashboard. No other cells, no local files.

WHAT YOU GET (rendered as an HTML dashboard right in the notebook)

  1. TODAY'S BREADTH  - advancers vs decliners (Total Market or MidSmallcap 400),
     moves by market-cap tier, Nifty 500 distribution days, FII/DII flows.
  2. MARKET INTERNALS - participation (% above 9/21/50/200 EMA), new highs -
     lows, advance/decline and 4% movers for All stocks / Nifty 50 / 500 /
     Midcap 150 / Smallcap 250, charted over 3M/6M/1Y with a Nifty 50 underlay.
     Hover the chart to read every metric for that day.
  3. TODAY'S ACTION   - volume breakouts, pivot breakouts, new 52-week highs and
     RS leaders slipping below their 50 EMA.
  4. SECTOR BREADTH   - NSE industry groups ranked by participation, with
     1d / 5d / 21d equal-weighted returns.

Two more tabs on the same page (switch them off in CONFIG):

  * IPO SCREENER  - recent main-board NSE IPOs near their post-listing high or
    low, consolidating in a tight range on drying volume (relaxed variant).
  * RVOL SCREENER - Total Market stocks with a >= 1000% relative-volume green
    spike in the last 15 sessions that has held, split into near-high, rest,
    volume dry-up, inside-mother-candle and pre-breakout tables.

  Both tabs end with a 6-month candlestick chart per match (EMA 20/50/200 and a
  volume pane); click a ticker in any table to jump to its chart.

A 2-year download is enough to seed a 200 EMA and a 52-week high.

CAVEAT: the universe is today's NIFTY TOTAL MARKET list, so names delisted in
the last two years are missing.

Edit the CONFIG block near the bottom before running.
"""

# ---------------------------------------------------------------------------
# 1. Dependencies (safe to re-run; no-ops if already installed)
# ---------------------------------------------------------------------------
import sys
import pandas as pd
import numpy as np
...

import base64
import csv
import html
import io
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from matplotlib.figure import Figure

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

NSE_BASE = "https://nsearchives.nseindia.com/content/indices"
TOTAL_MARKET = "ind_niftytotalmarket_list"
SIZE_SEGMENTS = {
    "Nifty 50": "ind_nifty50list",
    "Nifty 100 (large)": "ind_nifty100list",
    "Midcap 150": "ind_niftymidcap150list",
    "Smallcap 250": "ind_niftysmallcap250list",
    "Microcap 250": "ind_niftymicrocap250_list",
}
CACHE_DIR = ".breadth_cache"
IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# 2. NSE metadata
# ---------------------------------------------------------------------------
def _nse_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                                    "Chrome/120.0 Safari/537.36"})
    s.get("https://www.nseindia.com", timeout=20)
    return s


def fetch_index_csv(slug, session=None):
    """Rows of an NSE index constituent CSV, cached for the day."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{slug}.csv")
    fresh = (os.path.exists(path)
             and datetime.fromtimestamp(os.path.getmtime(path)).date() == datetime.now().date())
    if fresh:
        text = open(path).read()
    else:
        session = session or _nse_session()
        resp = session.get(f"{NSE_BASE}/{slug}.csv", timeout=20)
        resp.raise_for_status()
        text = resp.text
        with open(path, "w") as f:
            f.write(text)
    return [r for r in csv.DictReader(io.StringIO(text))
            if r.get("Symbol") and not r["Symbol"].strip().startswith("DUMMY")]


def fetch_universe():
    """(sorted symbols, {symbol: industry}) for the NIFTY TOTAL MARKET list."""
    rows = fetch_index_csv(TOTAL_MARKET)
    industries = {r["Symbol"].strip(): r.get("Industry", "Unknown").strip() for r in rows}
    return sorted(industries), industries


def fetch_members(slug, session=None):
    return {r["Symbol"].strip() for r in fetch_index_csv(slug, session)}


# ---------------------------------------------------------------------------
# 3. Prices
# ---------------------------------------------------------------------------
def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _drop_partial_sessions(closes, volumes, min_frac=0.5):
    # yfinance can append a session that only a handful of tickers have printed yet, and it
    # sometimes carries rows for exchange holidays (e.g. 15 Jan 2026) with little or no volume
    counts = closes.notna().sum(axis=1)
    keep = counts >= counts.max() * min_frac
    traded = (volumes.reindex(index=closes.index, columns=closes.columns) > 0).sum(axis=1)
    quiet = traded < traded[keep].median() * min_frac
    for d in closes.index[~keep][-3:]:
        print(f"  [warn] dropped {d:%d %b}: only {counts[d]} of {counts.max()} stocks have a close yet "
              f"- rerun with REFRESH = True later", file=sys.stderr)
    for d in closes.index[keep & quiet]:
        print(f"  [warn] dropped {d:%d %b %Y}: only {traded[d]} stocks traded - looks like a holiday row",
              file=sys.stderr)
    keep &= ~quiet
    return closes[keep], volumes.reindex(index=closes.index[keep], columns=closes.columns)


def _match_calendar(closes, volumes, index):
    """Keep only sessions the Nifty 50 traded (plus any after its last print).

    Yahoo sometimes has stock rows for NSE holidays, e.g. 15 Jan 2026 (Maharashtra civic
    polls); one such row breaks every 40/65-day window it touches.
    """
    if index is None or index.empty:
        return closes, volumes
    keep = closes.index.isin(index.index) | (closes.index > index.index.max())
    for d in closes.index[~keep]:
        print(f"  [warn] dropped {d:%d %b %Y}: Nifty 50 did not trade (exchange holiday)", file=sys.stderr)
    return closes[keep], volumes.reindex(index=closes.index[keep], columns=closes.columns)


def _prices_fresh(path):
    """True if the cache was written after the latest NSE close (16:00 IST cut-off, weekdays)."""
    if not os.path.exists(path):
        return False
    now = datetime.now(IST)
    cutoff = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now < cutoff:
        cutoff -= timedelta(days=1)
    while cutoff.weekday() >= 5:
        cutoff -= timedelta(days=1)
    return datetime.fromtimestamp(os.path.getmtime(path), IST) >= cutoff


def load_prices(symbols, period="2y", refresh=False, batch_size=60, verbose=True, tag=""):
    """(closes, volumes) [dates x tickers] panels, cached as gzipped CSV so re-runs are instant.

    Plain symbols get ".NS"; symbols already ending in ".NS" / ".BO" are used as they are.
    `tag` keeps a second universe's cache apart from the main one.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = {k: os.path.join(CACHE_DIR, f"{k}{'_' + tag if tag else ''}_{period}.csv.gz")
             for k in ("closes", "volumes", "opens", "highs", "lows")}   # o/h/l feed the Manas screener
    fresh = all(_prices_fresh(p) for p in paths.values())
    if fresh and not refresh:
        if verbose:
            print(f"Using cached prices from {CACHE_DIR}")
        return _drop_partial_sessions(*(pd.read_csv(paths[k], index_col=0, parse_dates=True)
                                        for k in ("closes", "volumes")))

    tickers = [s if s.endswith((".NS", ".BO")) else f"{s}.NS" for s in symbols]
    if verbose:
        print(f"Downloading {period} of daily closes and volumes for {len(tickers)} stocks "
              f"(one-off; cached for the rest of the day)...")

    close_frames, volume_frames = [], []
    ohl_frames = {"Open": [], "High": [], "Low": []}
    for n, batch in enumerate(_chunks(tickers, batch_size), 1):
        data = None
        for attempt in (1, 2):
            try:
                data = yf.download(batch, period=period, interval="1d", auto_adjust=True,
                                   group_by="column", threads=True, progress=False)
            except Exception as e:
                print(f"  [warn] batch {n} attempt {attempt} failed: {e}", file=sys.stderr)
            if data is not None and not data.empty:
                break
            time.sleep(5)
        if data is None or data.empty:
            continue
        multi = isinstance(data.columns, pd.MultiIndex)
        close_frames.append(data["Close"] if multi else data[["Close"]].set_axis(batch, axis=1))
        volume_frames.append(data["Volume"] if multi else data[["Volume"]].set_axis(batch, axis=1))
        for k in ohl_frames:
            ohl_frames[k].append(data[k] if multi else data[[k]].set_axis(batch, axis=1))
        if verbose:
            print(f"  batch {n}: {close_frames[-1].shape[1]} symbols", flush=True)

    closes = pd.concat(close_frames, axis=1).sort_index()
    closes = closes.loc[:, ~closes.columns.duplicated()].dropna(axis=1, how="all")
    volumes = pd.concat(volume_frames, axis=1).sort_index()
    volumes = volumes.loc[:, ~volumes.columns.duplicated()].reindex(index=closes.index, columns=closes.columns)
    closes.to_csv(paths["closes"])
    volumes.to_csv(paths["volumes"])
    for k, key in (("Open", "opens"), ("High", "highs"), ("Low", "lows")):
        f = pd.concat(ohl_frames[k], axis=1).sort_index()
        f = f.loc[:, ~f.columns.duplicated()].reindex(index=closes.index, columns=closes.columns)
        f.to_csv(paths[key])
    if verbose:
        print(f"Cached {closes.shape[1]} tickers x {closes.shape[0]} days")
    return _drop_partial_sessions(closes, volumes)


def load_index(ticker="^NSEI", period="2y", refresh=False):
    """Daily closes for an index (Nifty 50 by default), cached like the stocks; None on failure."""
    path = os.path.join(CACHE_DIR, f"index_{ticker.lstrip('^')}_{period}.csv.gz")
    if _prices_fresh(path) and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    try:
        s = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.dropna().rename("close")
        if s.empty:
            raise ValueError("no rows returned")
        os.makedirs(CACHE_DIR, exist_ok=True)
        s.to_frame().to_csv(path)
        return s
    except Exception as e:
        print(f"  [warn] {ticker} unavailable: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# 4. Breadth
# ---------------------------------------------------------------------------
def load_ohl(period="2y", tag=""):
    """(opens, highs, lows) saved by load_prices for the same period / tag; None if missing."""
    out = []
    for key in ("opens", "highs", "lows"):
        path = os.path.join(CACHE_DIR, f"{key}{'_' + tag if tag else ''}_{period}.csv.gz")
        if not os.path.exists(path):
            return None
        out.append(pd.read_csv(path, index_col=0, parse_dates=True))
    return tuple(out)


def load_index_ohlc(ticker="^NSEI", period="2y", refresh=False):
    """Daily Open / High / Low / Close for an index, cached; None on failure."""
    path = os.path.join(CACHE_DIR, f"index_ohlc_{ticker.lstrip('^')}_{period}.csv.gz")
    if _prices_fresh(path) and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].dropna(subset=["Close"])
        if df.empty:
            raise ValueError("no rows returned")
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(path)
        return df
    except Exception as e:
        print(f"  [warn] {ticker} OHLC unavailable: {e}", file=sys.stderr)
        return None


def compute_breadth(closes, min_stocks=20):
    valid = closes.notna()
    out = pd.DataFrame(index=closes.index)
    out["n_stocks"] = valid.sum(axis=1)

    for span in (9, 20, 21, 50, 200):
        ema = closes.ewm(span=span, adjust=False).mean()
        seeded = valid & (valid.cumsum() >= span)  # EMA needs history to mean anything
        out[f"pct_above_{span}ema"] = ((closes > ema) & seeded).sum(axis=1) / \
            seeded.sum(axis=1).replace(0, np.nan) * 100.0

    chg = closes.diff()
    adv, dec = (chg > 0).sum(axis=1), (chg < 0).sum(axis=1)
    out["adv"], out["dec"] = adv, dec
    total = (adv + dec).replace(0, np.nan)
    out["adv_pct"] = adv / total
    out["thrust_10d"] = out["adv_pct"].ewm(span=10, adjust=False).mean()

    net_ratio = (adv - dec) / total * 100.0
    out["mcclellan_osc"] = (net_ratio.ewm(span=19, adjust=False).mean()
                            - net_ratio.ewm(span=39, adjust=False).mean())

    roll_max = closes.rolling(252, min_periods=200).max()
    roll_min = closes.rolling(252, min_periods=200).min()
    tested = roll_max.notna().sum(axis=1).replace(0, np.nan)
    out["pct_new_52w_high"] = (closes >= roll_max).sum(axis=1) / tested * 100.0
    out["pct_new_52w_low"] = (closes <= roll_min).sum(axis=1) / tested * 100.0
    out["net_new_highs"] = out["pct_new_52w_high"] - out["pct_new_52w_low"]
    out["pct_near_52w_high"] = (closes >= roll_max * 0.95).sum(axis=1) / tested * 100.0
    out["new_52w_high_n"] = (closes >= roll_max).sum(axis=1)
    out["new_52w_low_n"] = (closes <= roll_min).sum(axis=1)

    rets = closes.pct_change()
    out["up_4pct"] = (rets >= 0.04).sum(axis=1)
    out["down_4pct"] = (rets <= -0.04).sum(axis=1)
    out["ad_ratio"] = adv / dec.replace(0, np.nan)

    return out[out["n_stocks"] >= min_stocks]


# ---------------------------------------------------------------------------
# 4b. Stockbee Market Monitor (Pradeep Bonde's breadth sheet, NSE edition)
# ---------------------------------------------------------------------------
MM_RANGES = (  # (column, lookback sessions, move) - Stockbee's Telechart scans use min/max close
    ("q25", 65, 0.25),   # 25%+ in a quarter
    ("m25", 20, 0.25),   # 25%+ in a month
    ("m50", 20, 0.50),   # 50%+ in a month
    ("d34", 34, 0.13),   # 13%+ in 34 days
)


def market_monitor(closes, volumes, min_volume=100000, ma_days=40, thrust_pctile=90, lookback=252):
    """One row per session with every column of the Stockbee Market Monitor.

    Thrust levels are NSE's own: a buying (selling) thrust is an up-4% (down-4%) count in
    the top `100 - thrust_pctile`% of the previous `lookback` sessions - as rare on NSE as
    Stockbee's 300 is in the US. Today is not part of its own yardstick.

    Up/down 4%: close >= 4% from yesterday on volume above yesterday's and at least
    `min_volume` shares. Up X% in N days: close / lowest close of the last N sessions
    >= 1 + X (down: close / highest close <= 1 - X). T2108: % of stocks above their
    40-day simple moving average.
    """
    vol = volumes.reindex(index=closes.index, columns=closes.columns)
    ret = closes / closes.shift(1) - 1
    on_vol = (vol > vol.shift(1)) & (vol >= min_volume)
    out = pd.DataFrame(index=closes.index)
    out["universe"] = closes.notna().sum(axis=1)
    out["up4"] = ((ret >= 0.04) & on_vol).sum(axis=1)
    out["dn4"] = ((ret <= -0.04) & on_vol).sum(axis=1)
    for col, thr in (("up4", "thr_up"), ("dn4", "thr_dn")):
        out[thr] = np.ceil(out[col].shift(1).rolling(lookback, min_periods=120)
                           .quantile(thrust_pctile / 100.0))
    for days in (5, 10):
        out[f"r{days}"] = (out["up4"].rolling(days).sum()
                           / out["dn4"].rolling(days).sum().replace(0, np.nan))
    for key, n, move in MM_RANGES:
        lo = closes.rolling(n, min_periods=n - 3).min()   # a few gaps allowed (suspensions,
        hi = closes.rolling(n, min_periods=n - 3).max()   # missing Yahoo days)
        out[f"{key}_up"] = (closes / lo >= 1 + move).sum(axis=1)
        out[f"{key}_dn"] = (closes / hi <= 1 - move).sum(axis=1)
    sma = closes.rolling(ma_days, min_periods=ma_days - 3).mean()
    out["t2108"] = (closes > sma).sum(axis=1) / sma.notna().sum(axis=1).replace(0, np.nan) * 100.0
    # NSE's own "normal" for the ratios: on NSE more stocks rise 4% than fall on most days,
    # so Stockbee's fixed 2 / 0.5 would read nearly every day as strong buying.
    for days in (5, 10):
        past = out[f"r{days}"].shift(1).rolling(lookback, min_periods=120)
        out[f"r{days}_hi"], out[f"r{days}_lo"] = past.quantile(0.8), past.quantile(0.2)
        out[f"r{days}_med"] = past.median()
    return out.iloc[65:]   # first quarter has no 65-day lookback yet


def fetch_nse_equities(series=("EQ", "BE")):
    """Every NSE main-board stock as yfinance tickers: EQ (normal) and BE (trade-for-trade).

    BZ, the NSE Z category, and the SME platform are left out.
    """
    resp = _nse_session().get(EQUITY_LIST_CSV_URL, timeout=30)
    resp.raise_for_status()
    out = set()
    for raw in csv.DictReader(io.StringIO(resp.text)):
        row = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
        if row.get("SERIES") in series and row.get("SYMBOL"):
            out.add(f"{row['SYMBOL']}.NS")
    return sorted(out)


# ---------------------------------------------------------------------------
# 5. Groupings - size segments and sectors
# ---------------------------------------------------------------------------
def segment_panels(closes):
    panels = {"TOTAL MARKET": closes}
    session = _nse_session()
    members = {}
    for label, slug in SIZE_SEGMENTS.items():
        try:
            members[label] = fetch_members(slug, session)
        except Exception as e:
            print(f"  [warn] {slug}: {e}", file=sys.stderr)

    def sub(symbols):
        return closes[[c for c in closes.columns if c.removesuffix(".NS") in symbols]]

    for label, mem in members.items():
        panel = sub(mem)
        if panel.shape[1] >= 20:
            panels[label] = panel
    if "Midcap 150" in members and "Smallcap 250" in members:
        panels["MID + SMALLCAP"] = sub(members["Midcap 150"] | members["Smallcap 250"])
        if "Nifty 100 (large)" in members:
            # Nifty 500 is by definition Nifty 100 + Midcap 150 + Smallcap 250
            panels["Nifty 500"] = sub(members["Nifty 100 (large)"] | members["Midcap 150"]
                                      | members["Smallcap 250"])
    return panels


def sector_rows(closes, industries, min_stocks=5):
    """Participation and momentum for every NSE industry group."""
    valid = closes.notna()
    above = {}
    for span in (20, 50, 200):
        ema = closes.ewm(span=span, adjust=False).mean()
        seeded = valid & (valid.cumsum() >= span)
        above[span] = ((closes > ema) & seeded).iloc[-1], seeded.iloc[-1]

    rets = closes.pct_change()
    rets = rets.where(rets.abs() < 0.35)

    groups = {}
    for col in closes.columns:
        groups.setdefault(industries.get(col.removesuffix(".NS"), "Unknown"), []).append(col)

    rows = []
    for name, cols in groups.items():
        if len(cols) < min_stocks:
            continue
        row = {"sector": name, "stocks": len(cols)}
        for span in (20, 50, 200):
            hit, seeded = above[span]
            n = int(seeded[cols].sum())
            row[f"p{span}"] = float(hit[cols].sum()) / n * 100.0 if n else np.nan
        for days, key in ((1, "ret_1d"), (5, "ret_5d"), (21, "ret_21d")):
            grp = rets[cols].mean(axis=1).iloc[-days:]
            row[key] = float((1.0 + grp.fillna(0.0)).prod() - 1.0) * 100.0
        rows.append(row)

    return pd.DataFrame(rows).sort_values("p50", ascending=False).reset_index(drop=True)


def rs_ratings(closes):
    """IBD-style 1-99 relative-strength rating from weighted 3/6/9/12-month returns."""
    def ret(n):
        return closes.iloc[-1] / closes.iloc[-1 - n] - 1.0
    raw = 0.4 * ret(63) + 0.2 * ret(126) + 0.2 * ret(189) + 0.2 * ret(252)
    return (raw.rank(pct=True) * 99).round().clip(1, 99)


def cap_tier_rows(panels):
    """Today's advances, declines and median move per size tier."""
    tiers = [("Large #1-100", "Nifty 100 (large)"), ("Mid #101-250", "Midcap 150"),
             ("Small #251-500", "Smallcap 250"), ("Micro #501+", "Microcap 250")]
    rows = []
    for name, key in tiers:
        if key not in panels:
            continue
        r = panels[key].pct_change().iloc[-1].dropna() * 100.0
        rows.append({"tier": name, "up": int((r > 0).sum()), "down": int((r < 0).sum()),
                     "median": float(r.median())})
    return rows


def distribution_days(closes, volumes, window=25):
    """Sessions in the last `window` where the universe fell 0.2%+ on higher total volume."""
    rets = closes.pct_change()
    ew = rets.where(rets.abs() < 0.35).mean(axis=1) * 100.0
    vol = volumes.sum(axis=1)
    return int(((ew <= -0.2) & (vol > vol.shift(1))).iloc[-window:].sum())


def daily_action(closes, volumes, rs):
    """Today's volume breakouts, pivot breakouts, 52-week highs and leaders slipping."""
    vol_x = volumes.iloc[-1] / volumes.shift(1).rolling(20, min_periods=20).mean().iloc[-1]
    df = pd.DataFrame({
        "close": closes.iloc[-1],
        "ret": closes.pct_change().iloc[-1] * 100.0,
        "vol_x": vol_x,
        "pivot": closes.shift(1).rolling(20, min_periods=20).max().iloc[-1],
        "high52": closes.rolling(252, min_periods=200).max().iloc[-1],
        "ema50": closes.ewm(span=50, adjust=False).mean().iloc[-1],
        "rs": rs.reindex(closes.columns),
    }).dropna(subset=["ret"])
    df.index = [c.removesuffix(".NS") for c in df.index]
    return {
        "vol_up": df[(df["ret"] >= 4) & (df["vol_x"] >= 2)].sort_values("vol_x", ascending=False),
        "breakouts": df[(df["close"] > df["pivot"]) & (df["ret"] >= 2) & (df["vol_x"] >= 1.5)]
            .sort_values("rs", ascending=False),
        "new_highs": df[df["close"] >= df["high52"]].sort_values("ret", ascending=False),
        "slipping": df[(df["rs"] >= 80) & (df["close"] < df["ema50"])].sort_values("rs", ascending=False),
    }


def fetch_fii_dii():
    """Today's FII/DII cash-market buy/sell/net (Cr) from NSE, or None if unavailable."""
    def num(x):
        return float(str(x).replace(",", ""))
    try:
        resp = _nse_session().get("https://www.nseindia.com/api/fiidiiTradeReact", timeout=20,
                                  headers={"Accept": "application/json",
                                           "Referer": "https://www.nseindia.com/"})
        resp.raise_for_status()
        return [{"category": str(r["category"]).replace("*", "").strip(), "date": r.get("date", ""),
                 "buy": num(r["buyValue"]), "sell": num(r["sellValue"]), "net": num(r["netValue"])}
                for r in resp.json()]
    except Exception as e:
        print(f"  [warn] FII/DII flows unavailable: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# 6. IPO consolidation screener (relaxed)
# ---------------------------------------------------------------------------
EQUITY_LIST_CSV_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"


def fetch_recent_ipo_universe(months=6, include_sme=False, max_same_day_listings=3):
    """[{'symbol', 'listing_date'}] for NSE listings in the last `months`, newest first.

    Dates shared by more than `max_same_day_listings` symbols are series-migration
    artifacts, not IPOs, and are dropped.
    """
    resp = _nse_session().get(EQUITY_LIST_CSV_URL, timeout=20)
    resp.raise_for_status()

    cutoff = datetime.now().date() - timedelta(days=30 * months)
    universe = []
    for raw_row in csv.DictReader(io.StringIO(resp.text)):
        row = {k.strip(): (v.strip() if v else v) for k, v in raw_row.items()}
        symbol, series, listed = row.get("SYMBOL"), row.get("SERIES"), row.get("DATE OF LISTING")
        if not symbol or not listed or (not include_sme and series != "EQ"):
            continue
        try:
            listing_date = datetime.strptime(listed, "%d-%b-%Y").date()
        except ValueError:
            continue
        if listing_date >= cutoff:
            universe.append({"symbol": symbol, "listing_date": listing_date})

    if max_same_day_listings is not None:
        counts = Counter(r["listing_date"] for r in universe)
        bad = {d for d, c in counts.items() if c > max_same_day_listings}
        if bad:
            print(f"  [info] dropping {sum(counts[d] for d in bad)} symbol(s) on {len(bad)} mass-listing "
                  f"date(s): {sorted(str(d) for d in bad)}", file=sys.stderr)
        universe = [r for r in universe if r["listing_date"] not in bad]

    universe.sort(key=lambda r: r["listing_date"], reverse=True)
    return universe


def check_ipo_consolidation(daily_df, near_low_pct=0.05, near_high_pct=0.20, consolidation_days=10,
                            range_pct_max=0.25, volume_lookback=10, min_avg_volume=100000,
                            avg_volume_days=5, require_inside_bar=False):
    """(is_match, details) for one post-IPO High/Low/Close/Volume history, oldest first.

    Young IPOs (fewer than 2 * volume_lookback bars) skip the volume-trend check and
    are flagged young_ipo.
    """
    if len(daily_df) < max(consolidation_days, avg_volume_days):
        return False, None
    is_young_ipo = len(daily_df) < volume_lookback * 2

    high_52wk = float(daily_df["High"].max())
    low_52wk = float(daily_df["Low"].min())
    current_close = float(daily_df["Close"].iloc[-1])
    pct_from_low = (current_close - low_52wk) / low_52wk
    pct_from_high = (high_52wk - current_close) / high_52wk

    at_low = pct_from_low <= near_low_pct
    near_high = pct_from_high <= near_high_pct
    if not (at_low or near_high):
        return False, None

    recent = daily_df.iloc[-consolidation_days:]
    range_pct = (float(recent["High"].max()) - float(recent["Low"].min())) / current_close
    if range_pct > range_pct_max:
        return False, None

    recent_avg_vol = prior_avg_vol = None
    if not is_young_ipo:
        recent_avg_vol = float(daily_df["Volume"].iloc[-volume_lookback:].mean())
        prior_avg_vol = float(daily_df["Volume"].iloc[-volume_lookback * 2:-volume_lookback].mean())
        if prior_avg_vol <= 0 or recent_avg_vol >= prior_avg_vol:
            return False, None

    avg_volume_5d = float(daily_df["Volume"].iloc[-avg_volume_days:].mean())
    if avg_volume_5d < min_avg_volume:
        return False, None

    is_inside_bar = inside_bar_volume_change_pct = None
    if len(daily_df) >= 2:
        last, prev = daily_df.iloc[-1], daily_df.iloc[-2]
        is_inside_bar = bool(last["High"] <= prev["High"] and last["Low"] >= prev["Low"])
        if prev["Volume"] > 0:
            inside_bar_volume_change_pct = round((last["Volume"] - prev["Volume"]) / prev["Volume"] * 100, 1)

    if require_inside_bar and not (is_inside_bar and inside_bar_volume_change_pct is not None
                                   and inside_bar_volume_change_pct < 0):
        return False, None

    return True, {
        "position": "near_52wk_low" if at_low else "near_52wk_high",
        "current_price": round(current_close, 2),
        "52wk_high": round(high_52wk, 2),
        "52wk_low": round(low_52wk, 2),
        "pct_from_low": round(pct_from_low * 100, 1),
        "pct_from_high": round(pct_from_high * 100, 1),
        "consolidation_range_pct": round(range_pct * 100, 1),
        "volume_change_pct": (round((recent_avg_vol - prior_avg_vol) / prior_avg_vol * 100, 1)
                              if recent_avg_vol is not None else None),
        "avg_volume_5d": int(avg_volume_5d),
        "inside_bar": is_inside_bar,
        "young_ipo": is_young_ipo,
    }


def run_ipo_screener(months=6, include_sme=False, max_same_day_listings=3, suffix=".NS", **params):
    """(matches DataFrame, {ticker: OHLCV frame}) for recent-IPO consolidation matches."""
    universe = fetch_recent_ipo_universe(months, include_sme, max_same_day_listings)
    print(f"IPO screener: {len(universe)} NSE IPO(s) from the last {months} months...")
    matches, frames = [], {}
    for entry in universe:
        ticker = f"{entry['symbol']}{suffix}"
        try:
            daily = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=False)
            if daily.empty:
                continue
            daily = daily[["Open", "High", "Low", "Close", "Volume"]].dropna()
            ok, info = check_ipo_consolidation(daily, **params)
        except Exception as e:
            print(f"  [warn] {ticker}: {e}", file=sys.stderr)
            continue
        if ok:
            matches.append({"ticker": entry["symbol"], "listing_date": entry["listing_date"].isoformat(), **info})
            frames[entry["symbol"]] = daily
    print(f"  {len(matches)} IPO match(es)")
    return pd.DataFrame(matches), frames


# ---------------------------------------------------------------------------
# 7. Relative-volume (RVOL) spike screener
# ---------------------------------------------------------------------------
def _corporate_action_dates(daily_df, special_div_yield_pct=5.0, max_daily_move_pct=35.0):
    """Dates distorted by splits/bonuses, special dividends, or >35% raw moves (unreported splits)."""
    dates = set()
    close = daily_df["Close"].astype(float)
    if "Stock Splits" in daily_df.columns:
        splits = daily_df["Stock Splits"].fillna(0).astype(float)
        dates.update(splits[splits != 0].index)
    if "Dividends" in daily_df.columns:
        divs = daily_df["Dividends"].fillna(0).astype(float)
        yield_pct = divs / close.replace(0, np.nan) * 100.0
        dates.update(yield_pct[yield_pct >= special_div_yield_pct].index)
    move_pct = close.pct_change().abs() * 100.0
    dates.update(move_pct[move_pct > max_daily_move_pct].index)
    return dates


def check_relative_volume(daily_df, avg_window=20, lookback_days=15, min_rvol_pct=1000.0,
                          min_avg_volume=0, min_close=0.0, exclude_corporate_actions=True,
                          special_div_yield_pct=5.0, max_daily_move_pct=35.0,
                          require_green_candle=True, min_green_body_pct=5.0,
                          max_retrace_pct=60.0, weeks_52_days=252):
    """(is_match, details) for the highest-RVOL qualifying spike in the last `lookback_days` bars."""
    if len(daily_df) < avg_window + lookback_days:
        return False, None

    volume = daily_df["Volume"].astype(float)
    close = daily_df["Close"].astype(float)
    open_ = daily_df["Open"].astype(float)
    high = daily_df["High"].astype(float)
    low = daily_df["Low"].astype(float)

    avg_volume = volume.rolling(avg_window).mean().shift(1)
    rvol_pct = volume / avg_volume * 100.0
    window = rvol_pct.iloc[-lookback_days:].dropna()
    if window.empty:
        return False, None
    hits = window[window >= min_rvol_pct]

    body_pct = (close - open_) / open_ * 100.0
    if require_green_candle and not hits.empty:
        hits = hits[body_pct.loc[hits.index] >= min_green_body_pct]

    if exclude_corporate_actions and not hits.empty:
        ca_pos = [daily_df.index.get_loc(d) for d in
                  _corporate_action_dates(daily_df, special_div_yield_pct, max_daily_move_pct)]
        if ca_pos:
            # an action inside the baseline puts the pre-action volumes on a different share count
            def tainted(date):
                pos = daily_df.index.get_loc(date)
                return any(pos - avg_window <= p <= pos for p in ca_pos)
            hits = hits[[not tainted(d) for d in hits.index]]

    def retrace_floor(pos):
        hi, lo = float(high.iloc[pos]), float(low.iloc[pos])
        return hi - max_retrace_pct / 100.0 * (hi - lo)

    if max_retrace_pct < 100.0 and not hits.empty:
        def held(date):
            pos = daily_df.index.get_loc(date)
            after = low.iloc[pos + 1:]
            return after.empty or float(after.min()) >= retrace_floor(pos)
        hits = hits[[held(d) for d in hits.index]]

    if hits.empty:
        return False, None

    best_date = hits.idxmax()
    best_avg_volume = float(avg_volume.loc[best_date])
    if best_avg_volume < min_avg_volume or float(close.iloc[-1]) < min_close:
        return False, None

    pos = daily_df.index.get_loc(best_date)
    best_close = float(close.loc[best_date])
    prev_close = float(close.iloc[pos - 1]) if pos > 0 else best_close
    day_change_pct = (best_close - prev_close) / prev_close * 100.0 if prev_close else 0.0

    spike_high, spike_low = float(high.iloc[pos]), float(low.iloc[pos])
    spike_range = spike_high - spike_low
    lows_after, highs_after = low.iloc[pos + 1:], high.iloc[pos + 1:]
    if lows_after.empty:
        low_since = retraced_pct = None
    else:
        low_since = float(lows_after.min())
        retraced_pct = (spike_high - low_since) / spike_range * 100.0 if spike_range > 0 else 0.0

    high_52w = float(high.iloc[-weeks_52_days:].max())
    current_close = float(close.iloc[-1])
    base_volumes = volume.iloc[pos + 1:]
    closed_above_mother = bool((close.iloc[pos + 1:] > spike_high).any())
    above_mother_pct = ((float(highs_after.max()) - spike_high) / spike_high * 100.0
                        if not highs_after.empty else None)

    if base_volumes.empty:
        base_avg_volume = last_volume = last_vol_vs_base_pct = inside_days = None
        lowest_vol_in_base = inside_mother = False
    else:
        base_avg_volume = float(base_volumes.mean())
        last_volume = float(base_volumes.iloc[-1])
        last_vol_vs_base_pct = round(last_volume / base_avg_volume * 100.0, 1) if base_avg_volume else None
        lowest_vol_in_base = bool(last_volume <= float(base_volumes.min()))
        inside_bars = (highs_after <= spike_high) & (lows_after >= spike_low)
        inside_days = f"{int(inside_bars.sum())}/{len(inside_bars)}"
        inside_mother = bool(inside_bars.all())

    return True, {
        "rvol_date": pd.Timestamp(best_date).strftime("%Y-%m-%d"),
        "rvol_pct": round(float(hits.max()), 1),
        "volume": int(volume.loc[best_date]),
        "avg_volume": int(best_avg_volume),
        "open_on_rvol_day": round(float(open_.loc[best_date]), 2),
        "close_on_rvol_day": round(best_close, 2),
        "body_pct": round(float(body_pct.loc[best_date]), 2),
        "day_change_pct": round(day_change_pct, 2),
        "retraced_pct": round(max(retraced_pct, 0.0), 1) if retraced_pct is not None else None,
        "days_ago": int(len(daily_df) - 1 - pos),
        "current_close": round(current_close, 2),
        "high_52w": round(high_52w, 2),
        "pct_from_52w_high": round((high_52w - current_close) / high_52w * 100.0, 1) if high_52w else None,
        "mother_high": round(spike_high, 2),
        "above_mother_pct": round(above_mother_pct, 1) if above_mother_pct is not None else None,
        "to_trigger_pct": round((spike_high - current_close) / current_close * 100.0, 1),
        "pre_breakout": bool(not closed_above_mother and current_close < spike_high),
        "base_avg_volume": int(base_avg_volume) if base_avg_volume is not None else None,
        "last_volume": int(last_volume) if last_volume is not None else None,
        "last_vol_vs_base_pct": last_vol_vs_base_pct,
        "lowest_vol_in_base": lowest_vol_in_base,
        "inside_days": inside_days,
        "inside_mother": inside_mother,
    }


def _extract_ticker_frame(data, symbol):
    if isinstance(data.columns, pd.MultiIndex):
        if symbol not in data.columns.get_level_values(0):
            return None
        df = data[symbol]
    else:
        df = data
    required = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    optional = [c for c in ("Dividends", "Stock Splits") if c in df.columns]
    df = df[required + optional].dropna(subset=required)
    return df if not df.empty else None


def run_rvol_screener(symbols, suffix=".NS", near_high_pct=35.0, batch_size=50, **params):
    """(matches DataFrame sorted by RVOL with a near_52w_high flag, {ticker: OHLCV frame})."""
    tickers = [f"{s}{suffix}" for s in symbols]
    period_days = max(int((params.get("avg_window", 20) + params.get("lookback_days", 15)) * 1.6) + 15, 400)
    print(f"RVOL screener: {len(tickers)} tickers, unadjusted OHLCV with corporate actions...")
    matches, frames = [], {}
    for n, batch in enumerate(_chunks(tickers, batch_size), 1):
        try:
            data = yf.download(batch, period=f"{period_days}d", interval="1d", auto_adjust=False,
                               actions=True, group_by="ticker", threads=True, progress=False)
        except Exception as e:
            print(f"  [warn] RVOL batch {n} failed: {e}", file=sys.stderr)
            continue
        if data is None or data.empty:
            continue
        for t in batch:
            try:
                df = _extract_ticker_frame(data, t)
                if df is None:
                    continue
                ok, info = check_relative_volume(df, **params)
                if ok:
                    matches.append({"ticker": t.removesuffix(suffix), **info})
                    frames[t.removesuffix(suffix)] = df
            except Exception as e:
                print(f"  [warn] {t}: {e}", file=sys.stderr)
    df = pd.DataFrame(matches)
    if not df.empty:
        df = df.sort_values("rvol_pct", ascending=False).reset_index(drop=True)
        df["near_52w_high"] = df["pct_from_52w_high"] <= near_high_pct
    print(f"  {len(df)} RVOL match(es)")
    return df, frames


# ---------------------------------------------------------------------------
# 8. HTML rendering
# ---------------------------------------------------------------------------
CSS = """
:root{--bg:#f6f7f9;--card:#fff;--line:#e5e7eb;--txt:#111827;--dim:#6b7280;
--grn:#16a34a;--red:#dc2626;--amb:#d97706;--blu:#2563eb;--bar:#e5e7eb}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
font:14px/1.5 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:20px}
h1{font-size:20px;margin:0 0 2px}
h2{font-size:18px;font-weight:700;margin:34px 0 2px}
.secsub{color:var(--dim);font-size:13px;margin:0 0 14px}
.sub{color:var(--dim);font-size:12px;margin:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px}
.ttl{font-size:13px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:#374151}
.grid{display:grid;gap:16px}
.g3{grid-template-columns:repeat(3,1fr)}
.g4{grid-template-columns:repeat(4,1fr)}
.g2{grid-template-columns:1fr 1fr}
.big{font-size:30px;font-weight:700;line-height:1.2}
.split{display:flex;height:8px;border-radius:4px;overflow:hidden;background:var(--bar);margin:12px 0 6px}
.split>span{display:block;height:100%}
.row{display:flex;justify-content:space-between;align-items:baseline;gap:8px;padding:3px 0}
.d{font-size:11px;margin-left:3px}
.tag{display:inline-block;padding:1px 8px;border-radius:4px;font-size:12px;font-weight:600;vertical-align:middle}
.tag.ok{background:#dcfce7;color:#15803d}.tag.bad{background:#fee2e2;color:#b91c1c}
.cnt{color:var(--dim);font-size:12px}
.arow{display:grid;grid-template-columns:1fr auto 72px;gap:10px;align-items:center;padding:7px 0;
border-top:1px solid #f1f2f4;font-size:13px}
.chip{font-size:11px;background:#f3f4f6;color:#374151;border-radius:4px;padding:1px 6px;text-align:center}
.chip.hi{background:#dbeafe;color:#1d4ed8}
details.more summary{cursor:pointer;color:var(--blu);font-size:12px;padding-top:8px}
.morelist{max-height:360px;overflow:auto}
.intl{display:grid;grid-template-columns:260px 1fr;background:#fff;border:1px solid var(--line);
border-radius:12px;overflow:hidden}
.side .si{display:block;padding:14px 18px;border-bottom:1px solid var(--line);
border-left:3px solid transparent;cursor:pointer}
.side .si.on{background:#eff4ff;border-left-color:#2563eb}
.side .si.on .k{color:#2563eb}
.side .k,.mh .k{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#4b5563}
.side .v{font-size:18px;font-weight:700}
.side .s{font-size:13px}
.main{padding:18px 20px;border-left:1px solid var(--line);min-width:0}
.mtop{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}
.tabs{display:flex;gap:8px;flex-wrap:wrap}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{font:inherit;font-size:13px;padding:4px 12px;border:0;border-left:1px solid var(--line);
background:#fff;color:#374151;cursor:pointer}
.seg button:first-child{border-left:0}
.seg button.on{background:#2563eb;color:#fff}
.shead{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap}
.usel{font:inherit;font-size:13px;padding:5px 8px;border:1px solid var(--line);border-radius:6px;
background:#fff;color:var(--txt);margin-bottom:14px}
@media(max-width:900px){.g4,.g3{grid-template-columns:1fr 1fr}.intl{grid-template-columns:1fr}
.main{border-left:none}}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:right;color:var(--dim);font-weight:600;font-size:11px;
text-transform:uppercase;letter-spacing:.05em;padding:7px 9px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
td{text-align:right;padding:7px 9px;border-bottom:1px solid #f1f2f4}
tr:last-child td{border-bottom:none}
tr.hl td{background:#f9fafb}
.pos{color:var(--grn)}.neg{color:var(--red)}.amb{color:var(--amb)}.dim{color:var(--dim)}
.note{color:var(--dim);font-size:12px;margin-top:10px}
.tabbar{display:flex;gap:4px;border-bottom:1px solid var(--line);margin:8px 0 0}
.tabbtn{font:inherit;font-size:14px;font-weight:600;padding:9px 16px;border:0;
border-bottom:2px solid transparent;background:none;color:var(--dim);cursor:pointer}
.tabbtn.on{color:var(--blu);border-bottom-color:var(--blu)}
.tscroll{overflow-x:auto}
.tscroll table{white-space:nowrap}
.tscroll a{color:var(--blu);text-decoration:none;font-weight:600}
.charts{display:grid;gap:16px}
.chartcard{padding:10px;scroll-margin-top:12px}
.chartcard img{width:100%;display:block}
.mm td{padding:6px 9px;font-variant-numeric:tabular-nums}
.mm th{text-align:center;vertical-align:bottom}
.mm .mm-h1,.mm .mm-h2{font-size:12px;color:#111827;border-bottom:none;border-radius:6px 6px 0 0}
.mm .mm-h1{background:#cffafe}.mm .mm-h2{background:#fef9c3}
.mm td.mm-g{background:#bbf7d0}.mm td.mm-g2{background:#16a34a;color:#fff;font-weight:600}
.mm td.mm-r{background:#f08080}.mm td.mm-p{background:#fde2e2}
.mmread{margin-bottom:16px}
.mmc{margin-bottom:4px}.mmc .mtop{align-items:center;margin-bottom:6px}
.mminfo{font-size:13px;line-height:1.5}.mmsvg{width:100%;height:auto;display:block}
.mmscroll{max-height:640px;overflow:auto;padding-top:0}
.mmscroll tr:nth-child(2) th{position:sticky;top:0;background:#fff;z-index:1}
.mmread .card{padding:14px 16px}
"""


def _esc(x):
    return html.escape(str(x))


def _num(v, dp=1, pct=False, signed=False, colour=False):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "<td>-</td>"
    fmt = f"{{:+.{dp}f}}" if signed else f"{{:.{dp}f}}"
    text = fmt.format(float(v)) + ("%" if pct else "")
    cls = ""
    if colour:
        cls = ' class="pos"' if float(v) > 0 else (' class="neg"' if float(v) < 0 else "")
    return f"<td{cls}>{text}</td>"


def _sgn(v, dp=2):
    return f"{v:+,.{dp}f}".replace("-", "&minus;")


def _missing(v):
    return v is None or (not isinstance(v, (str, bool, np.bool_)) and pd.isna(v))


def _fmt_volume(v):
    v = float(v)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"{v / div:,.2f}{unit}"
    return f"{v:,.0f}"


def _html_table(df, columns, fmt, links=None):
    cols = [(k, label) for k, label in columns if k in df.columns]
    head = "".join(f"<th>{_esc(label)}</th>" for _, label in cols)

    def cell(k, v):
        text = _esc(fmt(k, v))
        if k == "ticker" and links and v in links:
            return f'<td><a href="#{links[v]}">{text}</a></td>'
        return f"<td>{text}</td>"

    body = "".join("<tr>" + "".join(cell(k, r[k]) for k, _ in cols) + "</tr>" for _, r in df.iterrows())
    return f'<div class="tscroll"><table><tr>{head}</tr>{body}</table></div>'


def _section(title, df, columns, fmt, subtitle="", links=None):
    sub = f'<p class="secsub">{_esc(subtitle)}</p>' if subtitle else '<div style="height:10px"></div>'
    table = _html_table(df, columns, fmt, links) if not df.empty else '<div class="note">None.</div>'
    return f'<h2>{_esc(title)} ({len(df)})</h2>{sub}<div class="card">{table}</div>'


CHART_KEYS = ("show_charts", "chart_months", "chart_emas", "max_charts")


def _candle_png(daily_df, title, months=6, emas=(20, 50, 200), levels=(), spike_date=None, shade_bars=0):
    """Base64 PNG: candles + EMAs + volume pane, with optional levels, spike bar and shaded window.

    EMAs use the full history before slicing, so the 200-EMA is right at the left edge.
    """
    full = daily_df.dropna(subset=["Open", "High", "Low", "Close"])
    ema_series = {s: full["Close"].astype(float).ewm(span=s, adjust=False).mean()
                  for s in emas if len(full) >= s}
    df = full.iloc[-max(int(months * 21), 20):]
    x = np.arange(len(df))
    o, h, l, c = (df[k].astype(float).values for k in ("Open", "High", "Low", "Close"))
    colors = np.where(c >= o, "#26a69a", "#ef5350")

    fig = Figure(figsize=(12, 6.2), dpi=80)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)
    fig.suptitle(title, fontsize=12)

    ax.vlines(x, l, h, color=colors, linewidth=1, zorder=2)
    # doji bodies have zero height, so give them a thin visible floor
    min_body = max(float(np.mean(h - l)) * 0.03, 1e-6)
    ax.bar(x, np.maximum(np.abs(c - o), min_body), bottom=np.minimum(o, c), width=0.6,
           color=colors, edgecolor=colors, linewidth=0.8, zorder=3)
    for i, (span, ser) in enumerate(sorted(ema_series.items())):
        ax.plot(x, ser.iloc[-len(df):].values, linewidth=1.3, zorder=4,
                color=("#1f77b4", "#ff9800", "#7b1fa2", "#455a64")[i % 4], label=f"EMA {span}")
    for value, colour, label in levels:
        ax.axhline(float(value), color=colour, linestyle="--", linewidth=1, alpha=0.75, label=label)
    if spike_date:
        pos = np.nonzero(df.index.strftime("%Y-%m-%d") == spike_date)[0]
        if len(pos):
            ax.axvline(int(pos[0]), color="#fbc02d", alpha=0.45, linewidth=6, zorder=1)
    if shade_bars:
        ax.axvspan(len(df) - shade_bars - 0.5, len(df) - 0.5, color="#90caf9", alpha=0.18, zorder=0,
                   label=f"Last {shade_bars} sessions")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    ax.tick_params(labelbottom=False)

    axv.bar(x, df["Volume"].astype(float).values / 1e6, color=colors, alpha=0.65)
    axv.set_ylabel("Vol (M)")
    axv.grid(alpha=0.25)
    step = max(1, len(df) // 12)
    axv.set_xticks(x[::step])
    axv.set_xticklabels([d.strftime("%d-%b-%y") for d in df.index[::step]], rotation=45, ha="right", fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _charts(df, frames, cfg, prefix, annotate):
    """(charts HTML, {ticker: anchor id}) for the matches; annotate(row) -> _candle_png kwargs."""
    if not cfg.get("show_charts", True) or df.empty:
        return "", {}
    cap = cfg.get("max_charts", 0)
    months, emas = cfg.get("chart_months", 6), tuple(cfg.get("chart_emas", (20, 50, 200)))
    cards, links = [], {}
    for _, r in (df.head(cap) if cap else df).iterrows():
        frame = frames.get(r["ticker"])
        if frame is None:
            continue
        try:
            png = _candle_png(frame, months=months, emas=emas, **annotate(r))
        except Exception as e:
            print(f"  [warn] chart {r['ticker']}: {e}", file=sys.stderr)
            continue
        anchor = f"{prefix}-{r['ticker']}"
        links[r["ticker"]] = anchor
        cards.append(f'<div class="card chartcard" id="{_esc(anchor)}">'
                     f'<img alt="{_esc(r["ticker"])}" src="data:image/png;base64,{png}"></div>')
    head = (f'<h2>Charts ({len(cards)} of {len(df)})</h2><p class="secsub">Last {months} months, '
            f'EMA {" / ".join(str(e) for e in emas)}. Click a ticker in any table above to jump here.</p>')
    return head + f'<div class="charts">{"".join(cards)}</div>', links


IPO_COLUMNS = [("ticker", "Ticker"), ("listing_date", "Listed"), ("current_price", "Price"),
               ("52wk_high", "52w High"), ("52wk_low", "52w Low"), ("pct_from_low", "From Low"),
               ("pct_from_high", "From High"), ("consolidation_range_pct", "Range"),
               ("volume_change_pct", "Vol Chg"), ("avg_volume_5d", "Avg Vol 5d"),
               ("inside_bar", "Inside Bar"), ("young_ipo", "Young IPO")]


def _ipo_fmt(key, v):
    if _missing(v):
        return "-"
    if key in ("pct_from_low", "pct_from_high", "consolidation_range_pct", "volume_change_pct"):
        return f"{float(v):.1f}%"
    if key == "avg_volume_5d":
        return _fmt_volume(v)
    if key in ("current_price", "52wk_high", "52wk_low"):
        return f"{float(v):,.2f}"
    if isinstance(v, (bool, np.bool_)):
        return "Yes" if v else "No"
    return str(v)


def _ipo_tab(df, frames, cfg):
    intro = (f'<h2>IPO consolidation (relaxed)</h2><p class="secsub">Main-board NSE IPOs from the last '
             f'{cfg.get("months", 6)} months, within {cfg.get("near_low_pct", 0.05) * 100:.0f}% of the '
             f'post-listing low or {cfg.get("near_high_pct", 0.20) * 100:.0f}% of the high, ranging within '
             f'{cfg.get("range_pct_max", 0.25) * 100:.0f}% over {cfg.get("consolidation_days", 10)} days on '
             f'falling volume (skipped for young IPOs), with a 5-day average volume of at least '
             f'{cfg.get("min_avg_volume", 100000):,.0f}.</p>')
    if df.empty:
        return intro + '<div class="card note">No matches today.</div>'

    def annotate(r):
        where = "near post-listing high" if r["position"] == "near_52wk_high" else "near post-listing low"
        return dict(title=f"{r['ticker']}  |  listed {r['listing_date']}  |  {where}  |  "
                          f"{r['consolidation_range_pct']:.1f}% range",
                    levels=[(r["52wk_high"], "#2e7d32", "Post-listing high"),
                            (r["52wk_low"], "#d32f2f", "Post-listing low")],
                    shade_bars=cfg.get("consolidation_days", 10))

    charts, links = _charts(df, frames, cfg, "ipo", annotate)
    return intro + "".join(
        _section(title, df[df["position"] == pos].reset_index(drop=True), IPO_COLUMNS, _ipo_fmt, links=links)
        for pos, title in (("near_52wk_high", "Near post-listing high"),
                           ("near_52wk_low", "Near post-listing low"))) + charts


RVOL_COLUMNS = [("ticker", "Ticker"), ("rvol_date", "Spike Date"), ("rvol_pct", "RVOL"),
                ("volume", "Volume"), ("avg_volume", "Avg Vol"), ("open_on_rvol_day", "Open"),
                ("close_on_rvol_day", "Close"), ("body_pct", "Body"), ("day_change_pct", "Day"),
                ("retraced_pct", "Retraced"), ("base_avg_volume", "Base Vol"), ("last_volume", "Last Vol"),
                ("last_vol_vs_base_pct", "vs Base"), ("mother_high", "Mother Hi"),
                ("inside_days", "Inside"), ("above_mother_pct", "Above Hi"),
                ("to_trigger_pct", "To Trigger"), ("current_close", "Now"), ("high_52w", "52w High"),
                ("pct_from_52w_high", "Off High"), ("days_ago", "Age")]


# ---------------------------------------------------------------------------
# 7b. Manas Arora screener (Minervini-style swing set-ups, daily-chart version)
# ---------------------------------------------------------------------------
def run_manas_screener(closes, volumes, opens, highs, lows, min_price=30.0, near_high_pct=25.0,
                       above_low_pct=50.0, purple_move=0.05, purple_volume=1_000_000, prior_run_pct=50.0,
                       tight_range_pct=3.0, max_ema_dist_pct=5.0, pullback_min_pct=4.0, pullback_max_pct=20.0,
                       breakout_vol_x=1.5, pre_breakout_pct=3.0, base_range_pct=10.0, **_):
    """(tables {name: DataFrame}, frames {ticker: OHLCV}) for the latest session.

    Built from secondhand write-ups of Manas Arora's process (his own posts on X could not be read);
    intraday parts are approximated from daily bars: "strong start" = opened above yesterday's high,
    closed above the open, on more volume than yesterday.
    Trend filter (his interview template): close > min_price, within near_high_pct of the 52-week high,
    close > 50 SMA > 200 SMA, 200 SMA rising for 3 months, >= above_low_pct above the 52-week low,
    52-week high made in the last 6 months.
    """
    C, O, H, L = closes, opens.reindex_like(closes), highs.reindex_like(closes), lows.reindex_like(closes)
    V = volumes.reindex_like(closes)
    C, O, H, L, V = (x.iloc[-320:] for x in (C, O, H, L, V))
    sma20, sma50 = C.rolling(20).mean(), C.rolling(50).mean()
    sma200 = C.rolling(200, min_periods=190).mean()
    ema21 = C.ewm(span=21, adjust=False).mean()
    hi52, lo52 = H.rolling(252, min_periods=200).max(), L.rolling(252, min_periods=200).min()
    template = ((C > min_price) & (C >= hi52 * (1 - near_high_pct / 100)) & (C > sma50) & (sma50 > sma200)
                & (sma200 > sma200.shift(63)) & (sma200 > sma200.shift(21))
                & (C >= lo52 * (1 + above_low_pct / 100)) & (H.rolling(126).max() >= hi52 * 0.999))
    avg_v = V.shift(1).rolling(20).mean()
    rng = (H - L) / C * 100
    avg_rng = rng.shift(1).rolling(20).mean()
    pos = ((C - L) / (H - L).replace(0, np.nan)).fillna(0.5)            # where it closed in the day's range
    ret = C / C.shift(1) - 1
    purple = ((ret >= purple_move) & (V >= purple_volume)).astype(int)    # his "purple dot": fast mover day
    purple60 = purple.rolling(60, min_periods=1).max() > 0
    run = (C.rolling(60).max() / L.rolling(126, min_periods=60).min() - 1) * 100
    inside = (H <= H.shift(1)) & (L >= L.shift(1))
    ema_dist = (C / ema21 - 1) * 100
    strong = (O > H.shift(1)) & (C > O) & (V > V.shift(1))
    shifted = lambda x: x.shift(1, fill_value=False).astype(bool)

    # 1. contraction (SVRO): prior run + purple dot, tight inside days just above the 21 EMA
    svro = (template & purple60 & (run >= prior_run_pct) & (rng.rolling(3).mean() <= tight_range_pct)
            & (rng.rolling(3).max() <= tight_range_pct + 1) & (inside | shifted(inside))
            & (ema_dist >= 0) & (ema_dist <= max_ema_dist_pct))
    svro_go = shifted(svro) & strong
    # 2. 21 EMA pullback (2A)
    depth = (1 - C / H.rolling(15).max()) * 100
    zone = (L <= ema21 * 1.02) & (C >= ema21 * 0.98) & (C > sma50)
    pull = template & zone & (depth >= pullback_min_pct) & (depth <= pullback_max_pct)
    pull_dry = pull & (V.rolling(3).mean() < 0.8 * avg_v)
    ss = shifted(pull) & strong & template
    sv = pull & (V >= 1.5 * avg_v) & (rng <= 0.75 * avg_rng) & (pos >= 0.5)
    cv = (template & (L <= ema21) & (C > sma50) & (depth >= pullback_min_pct) & (V >= 2.5 * avg_v)
          & (rng >= 1.5 * avg_rng) & (pos >= 0.6))
    # 3. consolidation breakout / 4. pre-breakout
    hi20 = H.shift(1).rolling(20).max()
    bo = template & (C > hi20) & (V >= breakout_vol_x * avg_v) & (pos >= 0.67)
    base10 = (H.rolling(10).max() - L.rolling(10).min()) / C * 100
    pre = (template & (C >= hi20 * (1 - pre_breakout_pct / 100)) & (C <= hi20) & (base10 <= base_range_pct)
           & (C > sma20))

    d = C.index[-1]
    last = lambda x: x.loc[d]

    def table(mask, trigger, kind, level=None):
        t = last(mask)
        names = t[t.fillna(False).astype(bool)].index
        rows = []
        for n in names:
            c, lo_, hi_ = C.at[d, n], L.at[d, n], H.at[d, n]
            lv = level.at[d, n] if level is not None else hi_
            stop = lo_
            rows.append({"ticker": n.removesuffix(".NS"), "close": c,
                         "day_change_pct": ret.at[d, n] * 100,
                         "trigger": trigger(n) if callable(trigger) else trigger,
                         "vol_x": V.at[d, n] / avg_v.at[d, n] if avg_v.at[d, n] else np.nan,
                         "range_pct": rng.at[d, n], "ema21_dist_pct": ema_dist.at[d, n],
                         "off_high_pct": (c / hi52.at[d, n] - 1) * 100, "run_pct": run.at[d, n],
                         "trigger_level": lv if kind == "watch" else np.nan,
                         "to_trigger_pct": (lv / c - 1) * 100 if kind == "watch" else np.nan,
                         "stop": stop, "risk_pct": (c - stop) / c * 100 if kind == "go" else
                         (lv - stop) / lv * 100})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("vol_x" if kind == "go" else "to_trigger_pct",
                              ascending=kind != "go").reset_index(drop=True)

    tag2a = lambda n: " + ".join(k for k, m in (("Strong start", ss), ("Stop volume", sv),
                                                 ("Climax volume", cv)) if bool(m.at[d, n]))
    tables = {
        "svro_go": table(svro_go, "Strong start from contraction", "go"),
        "svro_watch": table(svro & ~svro_go, "Coiling", "watch"),
        "pb_go": table(ss | sv | cv, tag2a, "go"),
        "pb_watch": table(pull_dry & ~(ss | sv | cv), "Pulling back, volume drying", "watch"),
        "bo_go": table(bo, "Breakout", "go"),
        "pre_watch": table(pre & ~bo, "Tight base", "watch", level=hi20),
    }
    # funnel for the run log: how many stocks survive each filter on the latest session
    cnt = lambda mask: int(mask.loc[d].fillna(False).astype(bool).sum())
    tmpl = template.loc[d].fillna(False).astype(bool)
    print("Manas screener funnel (latest session):")
    for label, n in (("stocks with a close", int(C.loc[d].notna().sum())),
                     ("pass trend filter", int(tmpl.sum())),
                     ("  + purple dot in 60 days", cnt(template & purple60)),
                     ("  + prior run >= %g%%" % prior_run_pct, cnt(template & purple60 & (run >= prior_run_pct))),
                     ("  + tight 3 days", cnt(template & purple60 & (run >= prior_run_pct)
                                              & (rng.rolling(3).mean() <= tight_range_pct))),
                     ("  SVRO setting up / triggered", f"{cnt(svro & ~svro_go)} / {cnt(svro_go)}"),
                     ("  at 21 EMA, 4-20% pullback", cnt(pull)),
                     ("  2A setting up (volume drying)", cnt(pull_dry & ~(ss | sv | cv))),
                     ("  2A triggered: strong start / stop vol / climax vol",
                      f"{cnt(ss)} / {cnt(sv)} / {cnt(cv)}"),
                     ("  breakouts today", cnt(bo)),
                     ("  pre-breakout (tight base)", cnt(pre & ~bo))):
        print(f"  {label:<52} {n}")
    names = {t + ".NS" for df in tables.values() if not df.empty for t in df["ticker"]}
    frames = {n.removesuffix(".NS"): pd.DataFrame({"Open": opens[n], "High": highs[n], "Low": lows[n],
                                                   "Close": closes[n], "Volume": volumes[n]}).dropna()
              for n in names if n in closes.columns}
    return tables, frames


MANAS_COLUMNS = [("ticker", "Ticker"), ("trigger", "Trigger"), ("close", "Close"), ("day_change_pct", "Day"),
                 ("vol_x", "Vol x avg"), ("range_pct", "Range"), ("ema21_dist_pct", "vs 21 EMA"),
                 ("run_pct", "Prior run"), ("off_high_pct", "Off 52w high"), ("trigger_level", "Trigger above"),
                 ("to_trigger_pct", "To trigger"), ("stop", "Stop (day low)"), ("risk_pct", "Risk")]


def _manas_fmt(key, v):
    if _missing(v):
        return "-"
    if key in ("close", "trigger_level", "stop"):
        return f"{float(v):,.2f}"
    if key == "vol_x":
        return f"{float(v):.1f}x"
    if key in ("day_change_pct", "ema21_dist_pct", "off_high_pct", "to_trigger_pct"):
        return f"{float(v):+.1f}%"
    if key in ("range_pct", "risk_pct"):
        return f"{float(v):.1f}%"
    if key == "run_pct":
        return f"{float(v):,.0f}%"
    return str(v)


def _manas_tab(tables, frames, cfg, market_note=""):
    intro = ('<h2>Manas Arora set-ups</h2><p class="secsub">Minervini-style swing set-ups as Manas Arora '
             'describes them, screened on daily bars across all NSE stocks that pass his trend filter (close '
             f'above &#8377;{cfg.get("min_price", 30):.0f}, within {cfg.get("near_high_pct", 25):.0f}% of the '
             '52-week high, close &gt; 50 SMA &gt; 200 SMA, 200 SMA rising 3 months, 50%+ above the 52-week '
             'low, 52-week high in the last 6 months). Rules come from secondhand write-ups of his posts and '
             'interviews; his intraday checks (opening range, first-minutes volume) are approximated: '
             '<i>strong start</i> = opened above yesterday&rsquo;s high and closed green on more volume.</p>'
             + (f'<div class="card note" style="margin-bottom:6px">{market_note}</div>' if market_note else ""))
    groups = [
        ("Contraction breakout (SVRO) &middot; triggered today", "svro_go",
         "Prior 50%+ run with a purple-dot day (5%+ on 1M+ shares), tight inside days 0-5% above the 21 EMA, "
         "then a strong start."),
        ("Contraction (SVRO) &middot; setting up", "svro_watch",
         "Coiling now. Trigger = a strong start above today's high; stop = today's low."),
        ("21 EMA pullback (2A) &middot; triggered today", "pb_go",
         "Strong start = bounce off the EMA; Stop volume = heavy volume on a narrow bar at the EMA (enter next "
         "open); Climax volume = wide, heavy washout through the EMA that closed off the lows."),
        ("21 EMA pullback (2A) &middot; setting up", "pb_watch",
         "4-20% off the 15-day high, at the 21 EMA, volume drying up."),
        ("Consolidation breakout &middot; triggered today", "bo_go",
         f"Closed above the 20-day high on {cfg.get('breakout_vol_x', 1.5):g}x+ average volume, in the top third "
         "of the day's range."),
        ("Pre-breakout (tight base) &middot; setting up", "pre_watch",
         f"Within {cfg.get('pre_breakout_pct', 3):g}% of the 20-day high, 10-day range "
         f"{cfg.get('base_range_pct', 10):g}% or less, above the 20 MA. Trigger = the 20-day high."),
    ]
    order = ["svro_go", "pb_go", "bo_go", "svro_watch", "pb_watch", "pre_watch"]   # charts: triggers first
    seen, chart_rows = set(), []
    for key in order:
        df = tables.get(key, pd.DataFrame())
        for _, r in df.iterrows():
            if r["ticker"] not in seen:
                seen.add(r["ticker"])
                chart_rows.append(r)
    chart_df = pd.DataFrame(chart_rows)

    def annotate(r):
        lv = [(r["stop"], "#d32f2f", "Stop (day low)")]
        if not _missing(r.get("trigger_level")):
            lv.append((r["trigger_level"], "#2e7d32", "Trigger"))
        return dict(title=f"{r['ticker']}  |  {r['trigger']}  |  close {r['close']:,.2f}", levels=lv)

    charts, links = _charts(chart_df, frames, cfg, "manas", annotate) if not chart_df.empty else ("", {})
    body = "".join(_section(t.replace("&middot;", "-"), tables.get(k, pd.DataFrame()), MANAS_COLUMNS,
                            _manas_fmt, sub, links) for t, k, sub in groups)
    return intro + body + charts


def _rvol_fmt(key, v):
    if _missing(v):
        return "-"
    if key in ("volume", "avg_volume", "base_avg_volume", "last_volume"):
        return _fmt_volume(v)
    if key == "rvol_pct":
        return f"{float(v):,.0f}%"
    if key in ("body_pct", "retraced_pct", "pct_from_52w_high", "above_mother_pct", "last_vol_vs_base_pct"):
        return f"{float(v):.1f}%"
    if key in ("day_change_pct", "to_trigger_pct"):
        return f"{float(v):+.1f}%"
    if key in ("open_on_rvol_day", "close_on_rvol_day", "current_close", "high_52w", "mother_high"):
        return f"{float(v):,.2f}"
    if key == "days_ago":
        return "today" if int(v) == 0 else f"{int(v)}d"
    return str(v)


def _rvol_tab(df, frames, cfg):
    nh = cfg.get("near_high_pct", 35.0)
    intro = (f'<h2>Relative-volume spikes</h2><p class="secsub">RVOL &ge; {cfg.get("min_rvol_pct", 1000):,.0f}% '
             f'vs the prior {cfg.get("avg_window", 20)} days, in the last {cfg.get("lookback_days", 15)} sessions, '
             f'on a green candle with a body of at least {cfg.get("min_green_body_pct", 5.0):.0f}%, not retraced '
             f'more than {cfg.get("max_retrace_pct", 60.0):.0f}% of the spike range since. Splits, bonuses and '
             f'special dividends excluded. The last three tables are subsets of the first two.</p>')
    if df.empty:
        return intro + '<div class="card note">No matches today.</div>'

    def annotate(r):
        return dict(title=f"{r['ticker']}  |  spike {r['rvol_date']} @ {r['rvol_pct']:,.0f}% RVOL  |  "
                          f"mother high {r['mother_high']:,.2f}",
                    levels=[(r["mother_high"], "#d32f2f", "Mother high")],
                    spike_date=r["rvol_date"])

    charts, links = _charts(df, frames, cfg, "rvol", annotate)
    near = df["near_52w_high"]
    sections = [
        (f"Near 52-week high - within {nh:.0f}%", df[near], ""),
        (f"Rest - more than {nh:.0f}% off the 52-week high", df[~near], ""),
        ("Volume dry-up - last session is the lowest volume of the base", df[df["lowest_vol_in_base"]],
         "Supply drying up while price holds."),
        ("Inside mother candle - still coiling within the spike bar's range", df[df["inside_mother"]],
         "Breakout trigger = the mother candle's High."),
        ("Pre-breakout - never closed above the mother high, still below it", df[df["pre_breakout"]],
         "To Trigger = how far price must rise to take out the mother candle's High."),
    ]
    return intro + "".join(_section(t, d.reset_index(drop=True), RVOL_COLUMNS, _rvol_fmt, s, links)
                           for t, d, s in sections) + charts


_TAB_JS = ("var w=this.closest('.tabwrap'),k=this.dataset.t;"
           "w.querySelectorAll('.tabbtn').forEach(function(b){b.classList.toggle('on',b.dataset.t===k)});"
           "w.querySelectorAll('.tabpane').forEach(function(p){p.style.display=p.dataset.t===k?'':'none'})")


_SWITCH_JS = ("var w=this.closest('.uwrap'),v=this.value;"
              "w.querySelectorAll('.ublk').forEach(function(e){e.style.display=e.dataset.u===v?'':'none'})")


def _switch(items):
    """(select, blocks) showing one block per (label, html) item; caller wraps both in .uwrap."""
    if len(items) < 2:
        return "", items[0][1] if items else ""
    opts = "".join(f'<option value="{i}">{_esc(label)}</option>' for i, (label, _) in enumerate(items))
    blocks = "".join(
    '<div class="ublk" data-u="{}"{}>{}</div>'.format(
        i,
        '' if i == 0 else ' style="display:none"',
        h
    )
    for i, (_, h) in enumerate(items)
)
    return f'<select class="usel" onchange="{_SWITCH_JS}">{opts}</select>', blocks


def _market_internals(b, nifty):
    """Skeleton + JSON data for one universe; MB_JS draws the chart and handles hover."""
    view = b.tail(252)

    def arr(s):
        return [round(float(v), 2) if pd.notna(v) and np.isfinite(v) else None for v in s]

    last = view.iloc[-1]
    data = {
        "d": [d.strftime("%Y-%m-%d") for d in view.index],
        "p": {str(w): arr(view[f"pct_above_{w}ema"]) for w in PART_WINDOWS},
        "nh": arr(view["new_52w_high_n"] - view["new_52w_low_n"]),
        "ad": arr(view["ad_ratio"]),
        "m4": arr(view["up_4pct"] - view["down_4pct"]),
        "ix": arr(nifty.reindex(view.index)) if nifty is not None else [None] * len(view),
        "n": f"{int(last['n_stocks']):,}",
    }
    payload = html.escape(json.dumps(data, allow_nan=False), quote=True)
    side = "".join(f'<div class="si"><div class="k">{t}</div><span class="v"></span> '
                   f'<span class="s"></span></div>'
                   for t in ("Participation", "New highs &minus; lows", "Advance / decline", "4% movers"))
    wt = "".join(f"<button>{w}D</button>" for w in PART_WINDOWS)
    rt = "".join(f"<button>{r}</button>" for r in ("3M", "6M", "1Y"))
    ix_note = "Nifty 50 (scaled)" if nifty is not None else "Nifty 50 unavailable"
    return (f'<div class="mbi" data-d="{payload}">'
            f'<div class="intl"><div class="side">{side}</div><div class="main">'
            f'<div class="mtop"><div class="mh"><div class="k"></div>'
            f'<div><span class="big"></span> <span class="s"></span></div>'
            f'<div class="note mn" style="margin-top:2px"></div></div>'
            f'<div class="tabs"><div class="seg wt">{wt}</div><div class="seg rt">{rt}</div></div></div>'
            f'<div class="chart" style="margin-top:12px"></div>'
            f'<div class="row note"><span><b style="color:#2563eb">&#9644;</b> metric &nbsp;'
            f'<b style="color:#9ca3af">&#9644;</b> {ix_note} &middot; '
            f'McClellan {last["mcclellan_osc"]:+.1f} &middot; Zweig thrust {last["thrust_10d"]:.2f}</span>'
            f'<span class="foot"></span></div>'
            f'</div></div></div>')


PART_WINDOWS = (9, 21, 50, 200)

# Draws every .mbi block: zone bands, Nifty 50 underlay, hover read-out on the left list.
MM_JS = r"""
(function(){
var NS='http://www.w3.org/2000/svg',MON=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
var MMID=0,W=1200,H=340,L=64,R=14,T=12,B=26,PW=W-L-R,PH=H-T-B;
function ok(v){return v!==null&&v!==undefined&&isFinite(v);}
function mk(p,t,a){var e=document.createElementNS(NS,t);for(var k in a)e.setAttribute(k,a[k]);p.appendChild(e);return e;}
function txt(p,x,y,s,a){var e=mk(p,'text',Object.assign({x:x,y:y,'font-size':11,fill:'#6b7280'},a||{}));e.textContent=s;return e;}
function fd(s){var p=s.split('-');return (+p[2])+' '+MON[+p[1]-1]+' '+p[0];}
function fn(v,dp){return ok(v)?v.toLocaleString('en-IN',{minimumFractionDigits:dp||0,maximumFractionDigits:dp||0}):'\u2013';}
function info(box,D,i){var s=D.s[i],tag=s===2?['Turning up?','pos']:s===-2?['Turning down?','neg']:s>0?['Buying dominates','pos']:s<0?['Selling dominates','neg']:['Mixed','amb'];
 box.querySelector('.mminfo').innerHTML='<b>'+fd(D.d[i])+'</b> &middot; <b class="'+tag[1]+'">'+tag[0]+'</b>'
  +(ok(D.sc[i])?' <span class="dim">(score '+fn(D.sc[i])+')</span>':'')
  +' &middot; Nifty '+(box.dataset.mode==='candle'&&D.o&&ok(D.o[i])?'O '+fn(D.o[i],0)+' H '+fn(D.h[i],0)+' L '+fn(D.l[i],0)+' C ':'')+fn(D.ix[i],0)+' &middot; up 4% <b class="pos">'+fn(D.up[i])+'</b> / down 4% <b class="neg">'+fn(D.dn[i])+'</b>'
  +' <span class="dim">(thrust '+fn(D.tu[i])+' / '+fn(D.td[i])+')</span> &middot; 5d '+fn(D.r5[i],2)+' &middot; 10d '+fn(D.r10[i],2)
  +' &middot; T2108 '+fn(D.t[i],1)
  +(box._cv&&box._b0!==null&&ok(box._cv(i))?'<br><span class="dim">Since '+fd(D.d[box._b0])+': Nifty 50 '
    +(D.ix[i]/D.ix[box._b0]*100-100).toFixed(1)+'% &middot; <span style="color:#ea580c">'
    +(box.dataset.cmp==='md'?'Midcap 150':'Smallcap 250')+' (equal-weighted, dashed) '
    +(box._cv(i)/box._cv(box._b0)*100-100).toFixed(1)+'%</span></span>':'');}
function draw(box,D,span){
 var svg=box.querySelector('svg');while(svg.firstChild)svg.removeChild(svg.firstChild);
 svg.setAttribute('viewBox','0 0 '+W+' '+H);
 var uid=box.dataset.uid||(box.dataset.uid=String(++MMID));  // unique per chart: a hidden chart's pattern doesn't paint
 var defs=mk(svg,'defs',{});[['mmTu'+uid,'#86efac'],['mmTd'+uid,'#fca5a5']].forEach(function(p){
   var pt=mk(defs,'pattern',{id:p[0],width:8,height:8,patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'});
   mk(pt,'rect',{width:8,height:8,fill:'#ffffff'});mk(pt,'rect',{width:4,height:8,fill:p[1]});});
 var N=D.d.length,i0=Math.max(0,N-span),n=N-i0,step=PW/n;
 function X(i){return L+(i-i0+0.5)*step;}
 var ck=box.dataset.cmp===undefined?'sm':box.dataset.cmp,C=ck?D[ck]:null,b0=null,base=null;
 if(C){for(var i=i0;i<N;i++){if(ok(C[i])&&ok(D.ix[i])){b0=i;break;}}if(b0!==null)base=D.ix[b0]/C[b0];}
 function CV(i){return (base!==null&&ok(C[i]))?C[i]*base:null;}
 box._cv=CV;box._b0=b0;
 var CD=box.dataset.mode==='candle'&&D.o&&D.h&&D.l;
 function okc(i){return CD&&ok(D.o[i])&&ok(D.h[i])&&ok(D.l[i])&&ok(D.ix[i]);}
 var lo=Infinity,hi=-Infinity;for(var i=i0;i<N;i++){[D.ix[i],CV(i),okc(i)?D.h[i]:null,okc(i)?D.l[i]:null].forEach(function(v){if(ok(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}});}
 if(!(hi>lo)){lo=0;hi=1;}var pad=(hi-lo)*0.12;lo-=pad;hi+=pad;
 function Y(v){return T+(1-(v-lo)/(hi-lo))*PH;}
 // background = daily verdict
 for(var i=i0;i<N;i++){var s=D.s[i];mk(svg,'rect',{x:X(i)-step/2,y:T,width:step+0.5,height:PH,
   fill:s===2?'url(#mmTu'+uid+')':s===-2?'url(#mmTd'+uid+')':s>0?'#dcfce7':s<0?'#fee2e2':'#fef9c3'});}
 // grid + y labels
 for(var k=0;k<=4;k++){var v=lo+(hi-lo)*k/4,yy=Y(v);mk(svg,'line',{x1:L,x2:L+PW,y1:yy,y2:yy,stroke:'#ffffff','stroke-width':1});
   txt(svg,L-6,yy+4,fn(v,0),{'text-anchor':'end'});}
 // month ticks
 var last='';for(var i=i0;i<N;i++){var m=D.d[i].slice(0,7);if(m!==last&&i>i0){var p=D.d[i].split('-');
   mk(svg,'line',{x1:X(i)-step/2,x2:X(i)-step/2,y1:T+PH,y2:T+PH+4,stroke:'#9ca3af'});
   txt(svg,X(i)-step/2,H-8,MON[+p[1]-1]+(p[1]==='01'?' '+p[0].slice(2):''),{'text-anchor':'middle'});}last=m;}
 // Nifty line
 var d='';for(var i=i0;i<N;i++){if(ok(D.ix[i]))d+=(d?'L':'M')+X(i).toFixed(1)+','+Y(D.ix[i]).toFixed(1);}
 if(base!==null){var dc='';for(var i=i0;i<N;i++){var v=CV(i);if(ok(v))dc+=(dc?'L':'M')+X(i).toFixed(1)+','+Y(v).toFixed(1);}
   mk(svg,'path',{d:dc,fill:'none',stroke:'#ea580c','stroke-width':1.6,'stroke-dasharray':'5,3'});}
 if(CD){var bw=Math.max(1,Math.min(10,step*0.66));        // neutral candles: hollow up, filled down
   for(var i=i0;i<N;i++){if(!okc(i))continue;var x=X(i),o=D.o[i],c=D.ix[i],up=c>=o;
     mk(svg,'line',{x1:x,x2:x,y1:Y(D.h[i]),y2:Y(D.l[i]),stroke:'#374151','stroke-width':1});
     var yt=Y(Math.max(o,c)),yb=Y(Math.min(o,c));
     mk(svg,'rect',{x:x-bw/2,y:yt,width:bw,height:Math.max(1,yb-yt),fill:up?'#ffffff':'#374151',stroke:'#374151','stroke-width':1});}}
 else mk(svg,'path',{d:d,fill:'none',stroke:'#111827','stroke-width':1.8});
 // thrust days: buying below the candle low (or the line), selling above the high
 var sz=Math.max(4,Math.min(7,step*0.9));
 for(var i=i0;i<N;i++){if(!ok(D.ix[i]))continue;var x=X(i),u=D.up[i],dn=D.dn[i];
   var yl=Y(okc(i)?D.l[i]:D.ix[i]),yh=Y(okc(i)?D.h[i]:D.ix[i]);
   if(ok(D.tu[i])&&u>dn&&u>=D.tu[i])mk(svg,'path',{d:'M'+x+','+(yl+5)+'l'+(-sz)+','+(sz*1.6)+'h'+(2*sz)+'z',fill:'#15803d'});
   if(ok(D.td[i])&&dn>u&&dn>=D.td[i])mk(svg,'path',{d:'M'+x+','+(yh-5)+'l'+(-sz)+','+(-sz*1.6)+'h'+(2*sz)+'z',fill:'#b91c1c'});}
 mk(svg,'rect',{x:L,y:T,width:PW,height:PH,fill:'none',stroke:'#e5e7eb'});
 // hover
 var cross=mk(svg,'line',{x1:0,x2:0,y1:T,y2:T+PH,stroke:'#111827',opacity:0});
 var ov=mk(svg,'rect',{x:L,y:T,width:PW,height:PH,fill:'transparent'});
 ov.addEventListener('mousemove',function(e){var r=svg.getBoundingClientRect(),vx=(e.clientX-r.left)*W/r.width;
   var i=Math.max(i0,Math.min(N-1,i0+Math.floor((vx-L)/step)));
   cross.setAttribute('x1',X(i));cross.setAttribute('x2',X(i));cross.setAttribute('opacity',0.4);info(box,D,i);});
 ov.addEventListener('mouseleave',function(){cross.setAttribute('opacity',0);info(box,D,N-1);});
 info(box,D,N-1);
}
document.querySelectorAll('.mmc').forEach(function(box){
 var D=JSON.parse(box.dataset.d),btns=box.querySelectorAll('.mmrng button');
 btns.forEach(function(b){b.addEventListener('click',function(){btns.forEach(function(x){x.classList.toggle('on',x===b);});
   box._span=+b.dataset.n;draw(box,D,box._span);});});
 var mb=box.querySelectorAll('.mmmode button');
 mb.forEach(function(b){b.addEventListener('click',function(){mb.forEach(function(x){x.classList.toggle('on',x===b);});
   box.dataset.mode=b.dataset.k;draw(box,D,box._span||126);});});
 if(!D.o){var gm=box.querySelector('.mmmode');if(gm)gm.style.display='none';}
 var cb=box.querySelectorAll('.mmcmp button');
 cb.forEach(function(b){b.addEventListener('click',function(){cb.forEach(function(x){x.classList.toggle('on',x===b);});
   box.dataset.cmp=b.dataset.k;draw(box,D,box._span||126);});});
 if(!(D.sm||[]).some(ok)){box.dataset.cmp='';var g=box.querySelector('.mmcmp');if(g)g.style.display='none';}
 draw(box,D,126);
});
document.querySelectorAll('.mmrows').forEach(function(seg){
 var tbl=seg.closest('.mmtw').querySelector('table'),btns=seg.querySelectorAll('button');
 btns.forEach(function(b){b.addEventListener('click',function(){var n=+b.dataset.n;
   btns.forEach(function(x){x.classList.toggle('on',x===b);});
   tbl.querySelectorAll('tr[data-i]').forEach(function(r){r.style.display=+r.dataset.i<n?'':'none';});});});
});
})();
"""


MB_JS = r"""
(function(){
var W=1000,H=300,PL=52,PR=8,PT=10,PB=26,PW=W-PL-PR,PH=H-PT-PB,RNG=[63,126,252];
var GRN='#dcfce7',RED='#fee2e2',MON=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
function isF(v){return v!==null&&v!==undefined&&isFinite(v);}
function fInt(v){return (v<0?'\u2212':'')+Math.abs(Math.round(v)).toLocaleString('en-US');}
function fDate(s,y){var p=s.split('-');return (+p[2])+' '+MON[+p[1]-1]+(y?' '+p[0]:'');}
function sym(a){var m=0;a.forEach(function(v){if(isF(v))m=Math.max(m,Math.abs(v));});
  var r=Math.max(1,Math.ceil(m)),h=r*1.1;
  return {lo:-h,hi:h,refs:[[-r,fInt(-r)],[0,'0'],[r,fInt(r)]],bands:[[0,h,GRN],[-h,0,RED]]};}
function metrics(D,w){return [
 {t:'Participation',s:D.p[w],f:function(v){return v.toFixed(0)+'%';},
  st:function(v){return v>=80?['Stretched','amb']:v>=50?['Broad','pos']:v>=20?['Narrow','neg']:['Washed out','neg'];},
  note:'Share of stocks closing above their '+w+'-day EMA. Above 80% is stretched, below 20% washed out.',
  sc:function(){return {lo:0,hi:100,refs:[[0,'0%'],[20,'20%'],[50,'50%'],[80,'80%'],[100,'100%']],
    bands:[[80,100,GRN],[0,20,RED]]};}},
 {t:'New highs \u2212 lows',s:D.nh,f:fInt,
  st:function(v){return v>0?['More highs than lows','pos']:v<0?['More lows than highs','neg']:['Even','dim'];},
  note:'Stocks at a 52-week high minus stocks at a 52-week low. Green = more highs, red = more lows.',sc:sym},
 {t:'Advance / decline',s:D.ad,f:function(v){return v.toFixed(2);},
  st:function(v){return v>=1?['More risers','pos']:['More fallers','neg'];},
  note:'Advancers divided by decliners. Green = above 1 (more risers), red = below 1.',
  sc:function(a){var m=0;a.forEach(function(v){if(isF(v))m=Math.max(m,v);});var h=Math.min(Math.max(2.2,m*1.1),6);
    return {lo:0,hi:h,refs:[[0.5,'0.5'],[1,'1.0'],[2,'2.0']],bands:[[1,h,GRN],[0,1,RED]]};}},
 {t:'4% movers',s:D.m4,f:fInt,
  st:function(v){return v>=0?['More up 4%+','pos']:['More down 4%+','neg'];},
  note:'Stocks up 4% or more minus stocks down 4% or more. Green = net up, red = net down.',sc:sym}];}
function setS(el,m,v){el.querySelector('.s').textContent=isF(v)?m.st(v)[0]:'';
  el.querySelector('.s').className='s '+(isF(v)?m.st(v)[1]:'');}
function init(root){
 var D=JSON.parse(root.getAttribute('data-d')),WIN=Object.keys(D.p).map(Number).sort(function(a,b){return a-b;});
 var st={m:0,w:Math.min(2,WIN.length-1),r:1},N=D.d.length;
 var side=root.querySelectorAll('.si'),chart=root.querySelector('.chart'),mh=root.querySelector('.mh'),
     foot=root.querySelector('.foot'),wt=root.querySelector('.wt');
 function show(i){var ms=metrics(D,WIN[st.w]);
  ms.forEach(function(m,k){var v=m.s[i];side[k].querySelector('.v').textContent=isF(v)?m.f(v):'\u2013';setS(side[k],m,v);});
  var m=ms[st.m],v=m.s[i];mh.querySelector('.k').textContent=m.t+(st.m===0?' \u00b7 '+WIN[st.w]+'D':'');
  mh.querySelector('.big').textContent=isF(v)?m.f(v):'\u2013';setS(mh,m,v);
  foot.textContent=(i===N-1?'Latest \u00b7 ':'')+fDate(D.d[i],true)+' \u00b7 '+D.n+' stocks';}
 function draw(){
  var m=metrics(D,WIN[st.w])[st.m],n=Math.min(RNG[st.r],N),o=N-n,a=m.s.slice(o),sc=m.sc(a);
  function x(i){return PL+(n>1?i/(n-1):0)*PW;}
  function y(v){v=Math.min(Math.max(v,sc.lo),sc.hi);return PT+(sc.hi-v)/(sc.hi-sc.lo)*PH;}
  function path(arr,yf){var p='',on=false;arr.forEach(function(v,i){if(!isF(v)){on=false;return;}
    p+=(on?'L':'M')+x(i).toFixed(1)+','+yf(v).toFixed(1);on=true;});return p;}
  var h=['<svg viewBox="0 0 '+W+' '+H+'" style="display:block;width:100%;height:auto;cursor:crosshair;touch-action:none">'];
  sc.bands.forEach(function(b){h.push('<rect x="'+PL+'" y="'+y(b[1])+'" width="'+PW+'" height="'+(y(b[0])-y(b[1]))+'" fill="'+b[2]+'" fill-opacity=".75"/>');});
  sc.refs.forEach(function(r){h.push('<line x1="'+PL+'" x2="'+(W-PR)+'" y1="'+y(r[0])+'" y2="'+y(r[0])+'" stroke="#d1d5db"/>'+
    '<text x="'+(PL-8)+'" y="'+(y(r[0])+4)+'" fill="#6b7280" font-size="12" text-anchor="end">'+r[1]+'</text>');});
  var ix=D.ix.slice(o),lo=Infinity,hi=-Infinity;ix.forEach(function(v){if(isF(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}});
  if(hi>lo)h.push('<path d="'+path(ix,function(v){return PT+6+(hi-v)/(hi-lo)*(PH-12);})+'" fill="none" stroke="#9ca3af" stroke-width="1.4" stroke-opacity=".85"/>');
  h.push('<path d="'+path(a,y)+'" fill="none" stroke="#2563eb" stroke-width="2"/>');
  [[0,'start'],[Math.floor((n-1)/2),'middle'],[n-1,'end']].forEach(function(t){
    h.push('<text x="'+x(t[0])+'" y="'+(H-6)+'" fill="#6b7280" font-size="12" text-anchor="'+t[1]+'">'+fDate(D.d[o+t[0]])+'</text>');});
  h.push('<line class="gl" y1="'+PT+'" y2="'+(PT+PH)+'" stroke="#111827" stroke-opacity=".35" visibility="hidden"/>'+
    '<circle class="gd" r="4" fill="#2563eb" stroke="#fff" stroke-width="1.5" visibility="hidden"/></svg>');
  chart.innerHTML=h.join('');
  var svg=chart.firstChild,gl=svg.querySelector('.gl'),gd=svg.querySelector('.gd');
  function hover(e){var r=svg.getBoundingClientRect(),vx=(e.clientX-r.left)/r.width*W;
    var i=Math.max(0,Math.min(n-1,Math.round((vx-PL)/PW*(n-1))));
    gl.setAttribute('x1',x(i));gl.setAttribute('x2',x(i));gl.setAttribute('visibility','visible');
    if(isF(a[i])){gd.setAttribute('cx',x(i));gd.setAttribute('cy',y(a[i]));gd.setAttribute('visibility','visible');}
    else gd.setAttribute('visibility','hidden');
    show(o+i);}
  svg.addEventListener('pointermove',hover);svg.addEventListener('pointerdown',hover);
  svg.addEventListener('pointerleave',function(){gl.setAttribute('visibility','hidden');gd.setAttribute('visibility','hidden');show(N-1);});
  mh.querySelector('.mn').textContent=m.note;
  side.forEach(function(el,k){el.classList.toggle('on',k===st.m);});
  wt.style.display=st.m===0?'':'none';
  wt.querySelectorAll('button').forEach(function(b,k){b.classList.toggle('on',k===st.w);});
  root.querySelectorAll('.rt button').forEach(function(b,k){b.classList.toggle('on',k===st.r);});
  show(N-1);}
 side.forEach(function(el,k){el.addEventListener('click',function(){st.m=k;draw();});});
 wt.querySelectorAll('button').forEach(function(b,k){b.addEventListener('click',function(){st.w=k;draw();});});
 root.querySelectorAll('.rt button').forEach(function(b,k){b.addEventListener('click',function(){st.r=k;draw();});});
 draw();}
document.querySelectorAll('.mbi').forEach(function(r){try{init(r);}catch(e){
  r.querySelector('.chart').textContent='Chart error: '+e;}});
})();
"""


def _sector_table(rows):
    body = []
    for _, r in rows.iterrows():
        body.append(
            f"<tr><td>{_esc(r['sector'])}</td><td>{int(r['stocks'])}</td>"
            + _num(r["p20"], pct=True) + _num(r["p50"], pct=True) + _num(r["p200"], pct=True)
            + _num(r["ret_1d"], pct=True, signed=True, colour=True)
            + _num(r["ret_5d"], pct=True, signed=True, colour=True)
            + _num(r["ret_21d"], pct=True, signed=True, colour=True) + "</tr>")
    return f"""<div class="card"><table>
<tr><th>Sector</th><th>Stocks</th><th>&gt;20 EMA</th><th>&gt;50 EMA</th><th>&gt;200 EMA</th>
<th>1D</th><th>5D</th><th>21D</th></tr>
{''.join(body)}</table>
<div class="note">Ranked by 50-EMA participation. Returns are equal weighted inside
each group, so they describe the average stock, not the index heavyweights.</div></div>"""


def _mm_cell(v, cls="", dp=0):
    txt = "&ndash;" if _missing(v) else (f"{v:,.{dp}f}" if dp else f"{int(v):,}")
    return f'<td class="{cls}">{txt}</td>' if cls else f"<td>{txt}</td>"


def _mm_pair(up, dn, thr_up=None, thr_dn=None):
    """Stockbee colouring, as in his sheets.

    More up than down: both cells green, the up cell dark green on a thrust day.
    More down than up: both cells pink, the down cell red on a selling-thrust day.
    Thrust levels only apply to the 4% columns; the longer-range pairs are just green / red.
    """
    if _missing(up) or _missing(dn) or up == dn:
        return _mm_cell(up) + _mm_cell(dn)
    if thr_up is None:
        return (_mm_cell(up, "mm-g") + _mm_cell(dn, "mm-g")) if up > dn else \
               (_mm_cell(up, "mm-r") + _mm_cell(dn, "mm-r"))
    if up > dn:
        hot = not _missing(thr_up) and up >= thr_up
        return _mm_cell(up, "mm-g2" if hot else "mm-g") + _mm_cell(dn, "mm-g")
    hot = not _missing(thr_dn) and dn >= thr_dn
    return _mm_cell(up, "mm-p") + _mm_cell(dn, "mm-r" if hot else "mm-p")


def _mm_streak(flags):
    """Length of the run of True values ending on the last row."""
    n = 0
    for f in reversed(list(flags)):
        if not f:
            break
        n += 1
    return n


def _prank(x, n=252, min_n=120):
    """Percentile rank (0-1) of each value against the previous `n` values."""
    def rank(a):
        past, now = a[:-1], a[-1]
        past = past[~np.isnan(past)]
        if np.isnan(now) or len(past) < min_n:
            return np.nan
        return ((past < now).sum() + 0.5 * (past == now).sum()) / len(past)
    return x.rolling(n + 1, min_periods=min_n + 1).apply(rank, raw=True)


def mm_regime(mm, lookback=252):
    """Day-by-day sentiment: score 0-100 and verdict +1 buying / -1 selling / 0 mixed (our rules).

    Every part is ranked against its own previous year, because NSE breadth is lopsided
    (more stocks rise 4% than fall on most days) and fixed US levels don't carry over:
      1. 4% balance   - up-4% vs down-4% counts over 10 sessions
      2. T2108 level  - % of stocks above their 40-day average
      3. T2108 trend  - its change over 10 sessions
      4. Primary trend - change over 10 sessions in (up 25% - down 25%) / (up + down) in a quarter
    Score = average rank x 100. 60+ = buying dominates, 40 or less = selling dominates.
    The two trend parts catch a market that is slipping while many small stocks still rise.
    """
    bal = np.log((mm["up4"].rolling(10).sum() + 10) / (mm["dn4"].rolling(10).sum() + 10))
    prim = (mm["q25_up"] - mm["q25_dn"]) / (mm["q25_up"] + mm["q25_dn"]).replace(0, np.nan)
    parts = pd.DataFrame({"bal": _prank(bal, lookback), "t2108": _prank(mm["t2108"], lookback),
                          "t2108_trend": _prank(mm["t2108"].diff(10), lookback),
                          "prim_trend": _prank(prim.diff(10), lookback)}, index=mm.index)
    score = parts.mean(axis=1, skipna=False) * 100
    buy, sell = mm["up4"] > mm["dn4"], mm["dn4"] > mm["up4"]
    bthr = (buy & (mm["up4"] >= mm["thr_up"])).astype(int)
    sthr = (sell & (mm["dn4"] >= mm["thr_dn"])).astype(int)
    state = _mm_states(score, bthr, sthr)
    return pd.concat([parts * 100, pd.DataFrame({
        "score": score, "state": state,
        "b10": bthr.rolling(10, min_periods=1).sum(), "s10": sthr.rolling(10, min_periods=1).sum(),
        "bthr": bthr, "sthr": sthr},
        index=mm.index)], axis=1)


def _mm_states(score, bthr, sthr, hold=10, enter=(60, 40), leave=(55, 45), oversold=30, overbought=70):
    """Daily verdict: 1 buying, -1 selling, 0 mixed, 2 turning up?, -2 turning down?

    The score is slow (10-day windows), so two faster rules sit on top of it:
      * Thrust confirmation - 2+ buying thrusts within 5 sessions, the latest thrust being a
        buying one, switches to buying at once and holds for `hold` sessions unless a selling
        thrust comes. Selling is the mirror image.
      * Turning - a buying thrust while the score was oversold (<= 30 the day before) flags
        "turning up?" until confirmed, cancelled by a selling thrust, or `hold` sessions pass.
        It is an early warning only: in Mar 2026 it fired 2-3 weeks before the real low.
    Otherwise the score decides, with a buffer: buying starts at 60 and lasts until it drops
    below 55 (selling: starts at 40, lasts until above 45). Checked on Feb-Nov 2026 NSE data.
    """
    last = pd.Series(np.where(bthr == 1, 1, np.where(sthr == 1, -1, np.nan)), index=score.index).ffill()
    b_ok = ((bthr.rolling(5, min_periods=1).sum() >= 2) & (last == 1)).values
    s_ok = ((sthr.rolling(5, min_periods=1).sum() >= 2) & (last == -1)).values
    sc, prev, b, s_ = score.values, score.shift(1).values, bthr.values, sthr.values
    out, st, lock, turn, t0 = np.zeros(len(sc), int), 0, -1, 0, 0
    for i, x in enumerate(sc):
        if b_ok[i]:
            st, lock = 1, i + hold
        elif s_ok[i]:
            st, lock = -1, i + hold
        elif lock >= i and ((st == 1 and s_[i]) or (st == -1 and b[i])):
            lock = -1                                    # an opposite thrust ends the hold
        if lock < i and not np.isnan(x):
            if st == 1:
                st = -1 if x <= enter[1] else (0 if x < leave[0] else 1)
            elif st == -1:
                st = 1 if x >= enter[0] else (0 if x > leave[1] else -1)
            else:
                st = 1 if x >= enter[0] else (-1 if x <= enter[1] else 0)
        if turn == 1 and (s_[i] or st == 1 or i - t0 >= hold):
            turn = 0
        if turn == -1 and (b[i] or st == -1 or i - t0 >= hold):
            turn = 0
        if b[i] and prev[i] <= oversold and st != 1:
            turn, t0 = 1, i
        if s_[i] and prev[i] >= overbought and st != -1:
            turn, t0 = -1, i
        out[i] = 2 * turn if turn and st != turn else st
    return pd.Series(out, index=score.index)


def mm_green_type(mm, reg, washed=15, late=45):
    """What kind of green is today's? (label, cls, note) or None when not green / turning up.

    From the Feb-Nov 2026 back-check (small sample - a guide, not a rule):
      fresh - the green run began within 10 sessions of T2108 at or below `washed` and has 2+
              buying thrusts: 24 Mar, 1 Apr, 27 Oct and 2 Nov 2026 all led to 7-11% Nifty moves.
      late  - began with T2108 already at `late`+ and at most 1 buying thrust in the 10 sessions
              before: 26 May, 10 Jul, 3 Aug, 26 Aug 2026 - Nifty 2-3% lower 10 sessions later.
      steady - anything else (e.g. 12 Jun 2026: +3.4% over 16 sessions).
    """
    st = reg["state"]
    if st.iloc[-1] not in (1, 2):
        return None
    on = (st > 0).values
    i0 = len(on) - 1
    while i0 > 0 and on[i0 - 1]:
        i0 -= 1
    start = st.index[i0]
    t = mm["t2108"]
    t_low, t_start = t.iloc[max(0, i0 - 10):i0 + 1].min(), t.iloc[i0]
    thr_since = int(reg["bthr"].iloc[max(0, i0 - 9):].sum())
    thr_before = int(reg["bthr"].iloc[max(0, i0 - 9):i0 + 1].sum())
    if t_low <= washed and thr_since >= 2:
        return ("Fresh green &middot; multi-week potential", "ok",
                f"Began {start:%d %b} from a washed-out market (T2108 low {t_low:.0f}) with {thr_since} buying "
                f"thrusts. Starts like this led the Apr and Nov 2026 rallies. Hold while green; the first "
                f"warning is Nifty 750 turning red.")
    if t_low <= washed:
        return ("Fresh but unconfirmed", "amb",
                f"From a washed-out market (T2108 low {t_low:.0f}) but only {thr_since} buying thrust so far. "
                f"A second thrust confirms it; in Mar 2026 two such starts failed before the real low.")
    if st.iloc[-1] == 2:
        return ("Bounce attempt &middot; short trades", "bad",
                f"A buying thrust, but T2108 only fell to {t_low:.0f} (not washed out). Turns like this on "
                f"10 Mar, 18 Sep and 6 Oct 2026 mostly failed &mdash; quick trades only until it confirms.")
    if t_start >= late and thr_before <= 1:
        return ("Late green &middot; short trades", "bad",
                f"Began {start:%d %b} with T2108 already at {t_start:.0f} and {thr_before} buying thrust"
                f"{'' if thr_before == 1 else 's'} before it. Greens like this faded within two weeks in May, "
                f"Jul and Aug 2026 &mdash; take quick profits, keep stops tight.")
    return ("Steady green", "",
            f"Began {start:%d %b} (T2108 {t_start:.0f}, {thr_since} buying thrusts since). Neither washed-out "
            f"nor late: hold while green and watch Nifty 750 for the first warning.")


def _mm_read(mm, pctile, other=None, nifty=None, seg=None):
    """Today's read: sentiment verdict, its score and parts, thrust days and the ratios."""
    last = mm.iloc[-1]
    reg = mm_regime(mm)
    now = reg.iloc[-1]
    b10, s10 = int(now["b10"]), int(now["s10"])
    r5, r10 = last["r5"], last["r10"]
    state = int(now["state"])   # 2 / -2 = turning up / down (early, unconfirmed)
    since = reg.index[-_mm_streak(reg["state"] == state)]   # when sentiment last changed
    if state == 2:
        verdict, cls, msg = "Turning up?", "ok", ("A buying thrust from oversold &mdash; an early sign funds are "
                                               "stepping in. Not confirmed yet: wait for a second thrust or a "
                                               "rising score, and expect false starts.")
    elif state == -2:
        verdict, cls, msg = "Turning down?", "bad", ("A selling thrust from an overbought market &mdash; an early "
                                                   "warning. Not confirmed yet: tighten stops, go easy on new buys.")
    elif state > 0:
        verdict, cls, msg = "Buying dominates", "ok", "Funds are buying. Breakouts tend to work and follow through."
    elif state < 0:
        verdict, cls, msg = "Selling dominates", "bad", "Funds are selling. Breakouts tend to fail; protect capital."
    else:
        verdict, cls, msg = "Mixed", "amb", "Neither side in control. Breakouts are less reliable; be selective."
    tag = f'<span class="tag {cls}">{verdict}</span>' if cls != "amb" else \
          f'<span class="tag" style="background:#fef3c7;color:#b45309">{verdict}</span>'
    extra = ""
    gt = mm_green_type(mm, reg)
    if gt:
        lab, gcls, note = gt
        style = {"ok": "background:#dcfce7;color:#15803d", "bad": "background:#fee2e2;color:#b91c1c",
                 "amb": "background:#fef3c7;color:#b45309"}.get(gcls, "background:#f3f4f6;color:#374151")
        extra += (f'<div style="margin-top:10px"><span class="tag" style="{style}">{lab}</span></div>'
                  f'<div class="note" style="margin:4px 0 0">{note}</div>')
    if other is not None and state > 0:
        o_name, o_mm = other
        o_state = int(mm_regime(o_mm)["state"].iloc[-1])
        if o_state < 0:
            extra += (f'<div class="note" style="margin:8px 0 0;color:#b91c1c"><b>Warning:</b> {o_name} is '
                      f'already {"turning down" if o_state == -2 else "red"} &mdash; larger stocks are weakening '
                      f'first (in Aug 2026 it turned red 4 weeks before this list). Tighten exits.</div>')
    lvl = lambda v: "&ndash;" if _missing(v) else f"{int(v):,}"
    num = lambda v, dp=2: "&ndash;" if _missing(v) else f"{v:.{dp}f}"
    part = lambda k, lab: (f'<div class="row"><span class="dim">{lab}</span>'
                           f'<b class="{"pos" if now[k] >= 60 else "neg" if now[k] <= 40 else ""}">'
                           f'{num(now[k], 0)}</b></div>')
    sc = now["score"]
    rc = lambda v, hi, lo: ("" if _missing(v) or _missing(hi) or _missing(lo) else
                            "pos" if v >= hi else "neg" if v <= lo else "")
    return f"""<div class="grid g4 mmread">
<div class="card"><div class="ttl">Today's read</div><div style="margin:10px 0 6px">{tag}
<span class="cnt">since {since:%d %b}</span></div>
<div class="note" style="margin:0">{msg}</div>{extra}</div>
<div class="card"><div class="ttl">Sentiment score</div>
<div class="big {"pos" if state == 1 else "neg" if state == -1 else "amb"}">{num(sc, 0)}
<span class="dim" style="font-size:14px;font-weight:400">/ 100</span></div>
<div class="note" style="margin:0 0 4px">60+ buying &middot; 40 or less selling &middot; each part ranked vs its past year</div>
{part("bal", "4% balance, 10 days")}{part("t2108", "T2108 level")}
{part("t2108_trend", "T2108 trend")}{part("prim_trend", "Primary trend")}</div>
{_mm_divergence(mm, reg, nifty, seg)}
<div class="card"><div class="ttl">Thrusts &amp; ratios</div>
<div class="row" style="margin-top:8px"><span class="dim">Thrust days, last 10</span>
<b><span class="pos">{b10}</span> <span class="dim">vs</span> <span class="neg">{s10}</span></b></div>
<div class="note" style="margin:0 0 8px">Today's &ldquo;300&rdquo;: <b>{lvl(last['thr_up'])}+</b> up 4% (buying),
<b>{lvl(last['thr_dn'])}+</b> down 4% (selling) &middot; top {100 - pctile:g}% of the last year</div>
<div class="row"><span class="dim">5-day ratio</span><b class="{rc(r5, last['r5_hi'], last['r5_lo'])}">{num(r5)}</b></div>
<div class="row"><span class="dim">10-day ratio</span><b class="{rc(r10, last['r10_hi'], last['r10_lo'])}">{num(r10)}</b></div>
<div class="note" style="margin:4px 0 0">Normal is about <b>{num(last['r5_med'], 1)}</b> / <b>{num(last['r10_med'], 1)}</b>.
5-day: strong buying &ge; {num(last['r5_hi'], 1)}, strong selling &le; {num(last['r5_lo'], 1)} (top / bottom 20% of the last year)</div></div>
</div>"""


def _mm_divergence(mm, reg, nifty, seg, gap_lim=15, lead_gap=10, off_high=3.0, near_high=1.0):
    """Card 3: Nifty 50 vs mid / small caps. Flags a divergence, else shows who is leading.

    Participation = % of stocks above their 50-day EMA in each index; the gap is
    Nifty 50 minus Smallcap 250. Rules, back-checked on Nov 2025 - Nov 2026 NSE data:
      narrow   - Nifty within 1% of its 20-day high and the gap >= +15: large caps carry the
                 index. 11 of 11 past cases (all Nov 2025 - Feb 2026) had the Nifty lower
                 20 sessions later. Shown as a caution: one stretch of market, small sample.
      lead_dn  - Nifty 3%+ below its 20-day high, gap <= -10, breadth not red, green within
                 20 sessions, no washout: large caps lead the decline (4 of 7 fell further).
      lead_up  - Nifty 3%+ below its high, gap <= -10 after a washout (T2108 <= 15 within 15
                 sessions): small caps lead out of the low (2 of 2 rose 3-6% in 20 sessions).
    """
    rows = [(k, v) for k, v in (seg or {}).items() if v is not None]
    if nifty is None or not rows or "Nifty 50" not in (seg or {}) or "Smallcap 250" not in seg:
        return '<div class="card"><div class="ttl">Large vs small caps</div><div class="note">Index data unavailable today.</div></div>'
    ix = nifty.reindex(mm.index).ffill()
    dd = (ix / ix.rolling(20, min_periods=10).max() - 1) * 100
    p50 = {k: v["p50"].reindex(mm.index) for k, v in rows}
    gap = p50["Nifty 50"] - p50["Smallcap 250"]
    sm_chg = p50["Smallcap 250"] - p50["Smallcap 250"].shift(10)
    st = reg["state"]
    washed = mm["t2108"].rolling(15, min_periods=1).min() <= 15
    green20 = (st == 1).rolling(20, min_periods=1).max() > 0
    d, g, sc_ = dd.iloc[-1], gap.iloc[-1], sm_chg.iloc[-1]
    alert = None
    if not _missing(d) and not _missing(g):
        if d >= -near_high and g >= gap_lim:
            trend = (f"Small caps are catching up ({p50['Smallcap 250'].iloc[-11]:.0f}% &rarr; "
                     f"{p50['Smallcap 250'].iloc[-1]:.0f}% in 10 days)." if sc_ >= 0 else
                     f"Small caps are falling behind ({p50['Smallcap 250'].iloc[-11]:.0f}% &rarr; "
                     f"{p50['Smallcap 250'].iloc[-1]:.0f}%).")
            alert = ("Large caps leading the index", "amb",
                     f"Nifty near its high with Nifty 50 participation {g:.0f} points above small caps. {trend} "
                     f"In Nov 2025 &ndash; Feb 2026 this set-up came before a lower Nifty 20 days later "
                     f"in all 11 cases (one stretch of market &mdash; treat as a caution).")
        elif d <= -off_high and g <= -lead_gap and washed.iloc[-1]:
            alert = ("Small caps leading out of the low", "ok",
                     f"Nifty {d:.1f}% off its 20-day high but small caps are {-g:.0f} points ahead in "
                     f"participation after a washed-out market. Both times in 2026 (25 Mar, 1 Apr) the Nifty "
                     f"was 3&ndash;6% higher 20 days later.")
        elif d <= -off_high and g <= -lead_gap and st.iloc[-1] in (0, 1, 2) and green20.iloc[-1]:
            alert = ("Large caps leading the decline", "bad",
                     f"Nifty {d:.1f}% off its 20-day high while breadth still holds (small caps {-g:.0f} points "
                     f"ahead). In 2026 this came before further falls 4 times out of 7 (e.g. 8 Sep). Watch "
                     f"Nifty 750 and tighten exits.")
    style = {"ok": "background:#dcfce7;color:#15803d", "bad": "background:#fee2e2;color:#b91c1c",
             "amb": "background:#fef3c7;color:#b45309"}
    head = (f'<div style="margin:10px 0 4px"><span class="tag" style="{style[alert[1]]}">{alert[0]}</span></div>'
            f'<div class="note" style="margin:0 0 8px">{alert[2]}</div>') if alert else \
           ('<div style="margin:10px 0 4px"><span class="tag" style="background:#f3f4f6;color:#374151">'
            'No divergence</span></div>')
    def ret20(lvl):
        x = lvl.reindex(mm.index).ffill()
        return (x.iloc[-1] / x.iloc[-21] - 1) * 100 if len(x) > 21 and x.iloc[-21] > 0 else np.nan
    perf = {k: ret20(v["lvl"]) for k, v in rows}
    body = "".join(
        f'<div class="row"><span class="dim">{k}</span><span><b>{p50[k].iloc[-1]:.0f}%</b> '
        f'<span class="cnt">({"+" if p50[k].iloc[-1] - p50[k].iloc[-11] >= 0 else "&minus;"}'
        f'{abs(p50[k].iloc[-1] - p50[k].iloc[-11]):.0f})</span> &middot; '
        f'<b class="{"pos" if perf[k] >= 0 else "neg"}">{perf[k]:+.1f}%</b></span></div>'
        for k, _ in rows if not _missing(p50[k].iloc[-1]) and not _missing(perf[k]))
    lead = max(((k, perf[k]) for k, _ in rows if not _missing(perf[k])), key=lambda t: t[1], default=None)
    lead_txt = f"Leading over 20 days: <b>{lead[0]}</b>." if lead else ""
    return f"""<div class="card"><div class="ttl">Large vs small caps</div>{head}
<div class="note" style="margin:0 0 2px">% above 50-day average (10-day change) &middot; 20-day return, equal-weighted</div>
{body}<div class="note" style="margin:6px 0 0">{lead_txt}</div></div>"""


ON, HIDE = ' class="on"', ' style="display:none"'   # kept outside f-strings: Python < 3.12 forbids backslashes in them


def _mm_chart(mm, nifty, days=252, seg=None, ohlc=None):
    """Skeleton + JSON for the stacked sentiment charts; MM_JS draws them."""
    view = mm.tail(days)
    reg = mm_regime(mm).reindex(view.index)
    ix = nifty.reindex(view.index) if nifty is not None else pd.Series(np.nan, index=view.index)

    def arr(x, dp=2):
        return [round(float(v), dp) if pd.notna(v) and np.isfinite(v) else None for v in x]

    data = {"d": [d.strftime("%Y-%m-%d") for d in view.index], "ix": arr(ix),
            "up": arr(view["up4"], 0), "dn": arr(view["dn4"], 0),
            "tu": arr(view["thr_up"], 0), "td": arr(view["thr_dn"], 0),
            "r5": arr(view["r5"]), "r10": arr(view["r10"]),
            "qu": arr(view["q25_up"], 0), "qd": arr(view["q25_dn"], 0),
            "t": arr(view["t2108"], 1), "s": [int(v) for v in reg["state"]],
            "sc": arr(reg["score"], 0),
            "sm": arr((seg or {}).get("Smallcap 250", {}).get("lvl", pd.Series(dtype=float)).reindex(view.index), 3),
            "md": arr((seg or {}).get("Midcap 150", {}).get("lvl", pd.Series(dtype=float)).reindex(view.index), 3)}
    if ohlc is not None:
        o = ohlc.reindex(view.index)
        data.update({"o": arr(o["Open"]), "h": arr(o["High"]), "l": arr(o["Low"])})
    payload = html.escape(json.dumps(data, allow_nan=False), quote=True)
    rng = "".join(f'<button data-n="{n}"{ON if n == 126 else ""}>{lab}</button>'
                  for lab, n in (("3M", 63), ("6M", 126), ("1Y", 252)))
    return f"""<div class="card mmc" data-d="{payload}">
<div class="mtop"><div class="mminfo"></div><div class="tabs"><div class="seg mmmode"><button data-k="line" class="on">Line</button><button data-k="candle">Candles</button></div><div class="seg mmcmp"><button data-k="sm" class="on">Smallcap 250</button><button data-k="md">Midcap 150</button><button data-k="">Off</button></div><div class="seg mmrng">{rng}</div></div></div>
<svg class="mmsvg" preserveAspectRatio="xMidYMid meet"></svg>
<div class="note">Nifty 50 against the daily verdict: <span class="pos">green</span> background = buying
dominates, <span class="neg">red</span> = selling dominates, yellow = mixed, <span class="pos">green stripes</span> =
turning up? (buying thrust from oversold, not yet confirmed), <span class="neg">red stripes</span> = turning down?. &#9650; = buying-thrust day,
&#9660; = selling-thrust day. <span style="color:#ea580c">Dashed orange</span> = Smallcap 250 or
Midcap 150 (equal-weighted from their member stocks, rebased to the Nifty at the start of the view) &mdash;
when the two lines split, large and small caps are diverging. A change of background colour is a change in sentiment. Hover any day to
read its numbers.</div></div>"""


def _mm_table(mm, index, rows=252, show=21):
    view = mm.tail(rows).iloc[::-1]
    idx = index.reindex(view.index) if index is not None else pd.Series(np.nan, index=view.index)
    body = []
    for i, (d, r) in enumerate(view.iterrows()):
        ratio = lambda v, hi, lo: _mm_cell(v, "mm-g" if not _missing(hi) and v >= hi else
                                           "mm-r" if not _missing(lo) and v <= lo else "", dp=2)
        t = r["t2108"]
        body.append(
            f'<tr data-i="{i}"{"" if i < show else HIDE}><td>{d:%d %b %Y}</td>' + _mm_pair(r["up4"], r["dn4"], r["thr_up"], r["thr_dn"])
            + ratio(r["r5"], r["r5_hi"], r["r5_lo"]) + ratio(r["r10"], r["r10_hi"], r["r10_lo"])
            + "".join(_mm_pair(r[f"{k}_up"], r[f"{k}_dn"]) for k, _, _ in MM_RANGES)
            + _mm_cell(r["universe"])
            + _mm_cell(t, "mm-g" if t <= 20 else "mm-r" if t >= 80 else "", dp=2)
            + _mm_cell(idx.get(d), dp=2) + "</tr>")
    btns = "".join(f'<button data-n="{n}"{ON if n == show else ""}>{lab}</button>'
                   for lab, n in (("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252)) if n <= max(rows, 21))
    return f"""<div class="mmtw"><div class="shead" style="margin-top:18px"><div class="ttl">Daily sheet</div>
<div class="seg mmrows" style="margin-bottom:8px">{btns}</div></div>
<div class="card tscroll mm mmscroll"><table>
<tr><th></th><th colspan="6" class="mm-h1">Primary breadth indicators</th>
<th colspan="6" class="mm-h2">Secondary breadth indicators</th><th colspan="3"></th></tr>
<tr><th>Date</th><th>Up 4%+<br>today</th><th>Down 4%+<br>today</th><th>5-day<br>ratio</th>
<th>10-day<br>ratio</th><th>Up 25%+<br>in a quarter</th><th>Down 25%+<br>in a quarter</th>
<th>Up 25%+<br>in a month</th><th>Down 25%+<br>in a month</th><th>Up 50%+<br>in a month</th>
<th>Down 50%+<br>in a month</th><th>Up 13%+<br>in 34 days</th><th>Down 13%+<br>in 34 days</th>
<th>Stock<br>universe</th><th>T2108<br>(% &gt; 40 MA)</th><th>Nifty 50</th></tr>
{''.join(body)}</table></div></div>"""


def _mm_tab(monitors, nifty, cfg, seg=None, ohlc=None):
    """monitors = [(dropdown label, mm frame, subtitle note)]; the first one shows by default."""
    tp = cfg.get("thrust_pctile", 90)
    n750 = next(((label, mm) for label, mm, _ in monitors if label == "Nifty 750"), None)
    sel, blocks = _switch([(label, f"""<p class="secsub">Pradeep Bonde (Stockbee) style breadth sheet
across {int(mm['universe'].iloc[-1]):,} stocks{note} &middot; newest session first</p>
{_mm_read(mm, tp, n750 if label != "Nifty 750" else None, nifty, seg)}{_mm_chart(mm, nifty, seg=seg, ohlc=ohlc)}{_mm_table(mm, nifty, cfg.get("rows", 252))}""")
                           for label, mm, note in monitors])
    return f"""<div class="uwrap">
<div class="shead"><h2>Monitor</h2>{sel}</div>
{blocks}
<div class="note"><b>Up / down 4%+ today</b>: closed 4% or more above / below yesterday on
volume higher than yesterday's and at least {cfg.get('min_volume', 100000):,} shares.
<b>5 / 10-day ratio</b>: up-4% count &divide; down-4% count summed over 5 / 10 sessions; green
when in the top 20% of the last year, red in the bottom 20%. (On NSE more stocks rise 4% than fall
on most days, so Stockbee's fixed 2 and 0.5 would show nearly every day as strong buying.)
<b>Up / down X% in N days</b>: close vs the lowest / highest close of the last N sessions
(quarter = 65, month = 20).
<b>T2108</b>: % of stocks above their 40-day moving average (&le; 20 oversold, &ge; 80 overbought).
Colours follow Stockbee's sheet: a 4% pair is green when more stocks rose than fell and pink
when more fell; the up cell turns dark green (buying thrust) or the down cell red (selling thrust)
when the count is in the top {100 - tp:g}% of the previous {cfg.get('lookback', 252)} sessions.
That is NSE's version of his 300: a count rare enough that only funds can cause it. NSE stocks
move 4% far more often than US stocks, so a fixed number or a straight share of the universe
would not carry over. The longer-range pairs are green or red by which side is bigger.
<b>Sentiment score / background colour</b>: the average of four parts, each ranked 0&ndash;100
against its own last year &mdash; the 10-day 4% balance, T2108, T2108's 10-day change and the
10-day change in the quarter's up-25% vs down-25% balance. 60+ = buying dominates, 40 or less =
selling dominates; once in, a side holds until the score crosses 55 / 45. Because the score is
slow, thrusts act faster: two buying thrusts within 5 sessions (the latest one a buying thrust)
turn it green at once for at least 10 sessions, and a single buying thrust from an oversold market
shows <i>Turning up?</i> as an early warning (expect false starts). Selling is the mirror image.
These rules are ours, built on Stockbee's indicators and checked on Feb&ndash;Nov 2026 NSE data.
<b>Type of green</b>: <i>Fresh</i> = began within 10 sessions of T2108 at 15 or below, with 2+ buying
thrusts (multi-week moves in Apr and Nov 2026); <i>Late</i> = began with T2108 at 45+ and at most one
buying thrust before it (faded within two weeks in May, Jul, Aug 2026: short trades); <i>Bounce attempt</i> =
a turning-up from a market that was not washed out (mostly failed in 2026); <i>Steady</i> = in between. Based on one year of data &mdash; a guide, not a guarantee.
<b>Reading it</b>: funds can't hide. Day after day of buying, with back-to-back thrust days, is a
breadth thrust &mdash; breakouts work and follow through. Day after day of selling is the mirror
image. Use it to time breakouts, not to pick stocks.</div>
</div>"""


def _action_card(title, note, df, chip, n=8):
    head = (f'<div class="row"><span class="ttl">{_esc(title)}</span>'
            f'<span class="cnt">{len(df):,} stocks</span></div>'
            f'<div class="note" style="margin:4px 0 10px">{_esc(note)}</div>')
    if df.empty:
        return f'<div class="card">{head}<div class="note">None today.</div></div>'

    def rows(part):
        return "".join(
            f'<div class="arow"><b style="font-weight:600">{_esc(sym)}</b>'
            f'<span class="{"pos" if r["ret"] >= 0 else "neg"}">{_sgn(r["ret"])}%</span>{chip(r)}</div>'
            for sym, r in part.iterrows())

    more = ""
    if len(df) > n:
        more = (f'<details class="more"><summary>Show all {len(df):,}</summary>'
                f'<div class="morelist">{rows(df.iloc[n:])}</div></details>')
    return f'<div class="card">{head}{rows(df.head(n))}{more}</div>'


def _vol_chip(r):
    return f'<span class="chip">{r["vol_x"]:.1f}&times; vol</span>'


def _rs_chip(r):
    if pd.isna(r["rs"]):
        return '<span class="chip">RS &ndash;</span>'
    return f'<span class="chip{" hi" if r["rs"] >= 80 else ""}">RS {r["rs"]:.0f}</span>'


def _delta(now, prev, good_up=True):
    d = int(now) - int(prev)
    if d == 0:
        return '<span class="d dim">&ndash;</span>'
    cls = "pos" if (d > 0) == good_up else "neg"
    return f'<span class="d {cls}">{"&#9650;" if d > 0 else "&#9660;"}{abs(d):,}</span>'


def _tier_card(rows):
    parts = []
    for r in rows:
        name, rank = r["tier"].split(" ", 1)
        tot = max(r["up"] + r["down"], 1)
        parts.append(
            f'<div style="margin-top:12px"><div class="row"><span>{_esc(name)} '
            f'<span class="cnt">{_esc(rank)}</span></span>'
            f'<span class="{"pos" if r["median"] >= 0 else "neg"}">median {_sgn(r["median"])}%</span></div>'
            f'<div class="split" style="margin:4px 0"><span style="width:{r["up"] / tot * 100:.1f}%;'
            f'background:var(--grn)"></span><span style="flex:1;background:var(--red)"></span></div>'
            f'<div class="row cnt" style="padding:0"><span>{r["up"]:,} up</span>'
            f'<span>{r["down"]:,} down</span></div></div>')
    body = "".join(parts) or '<div class="note">NSE size-index lists unavailable.</div>'
    return (f'<div class="card"><div class="ttl">By market cap</div>'
            f'<div class="note" style="margin-top:4px">Where the buying or selling was. '
            f'Tiers follow NSE size-index membership.</div>{body}</div>')


def _flows_card(flows):
    head = '<div class="ttl">FII / DII flows</div>'
    if not flows:
        return (f'<div class="card">{head}<div class="note">NSE did not return provisional '
                f'flow data this run.</div></div>')
    rows = []
    for f in flows:
        name = "FII / FPI" if "FII" in f["category"].upper() else f["category"]
        sign = "+" if f["net"] >= 0 else "&minus;"
        rows.append(
            f'<div style="margin-top:10px"><div class="row"><span>{_esc(name)}</span>'
            f'<b class="{"pos" if f["net"] >= 0 else "neg"}">{sign}&#8377;{abs(f["net"]):,.0f} Cr</b></div>'
            f'<div class="cnt" style="text-align:right">bought {f["buy"]:,.0f} &middot; '
            f'sold {f["sell"]:,.0f} Cr</div></div>')
    return (f'<div class="card">{head}<div class="note" style="margin-top:4px">Cash market, '
            f'provisional &middot; {_esc(flows[0]["date"])}</div>{"".join(rows)}</div>')


def _ad_body(b):
    last, prev = b.iloc[-1], b.iloc[-2]
    adv, dec = int(last["adv"]), int(last["dec"])
    unch = max(int(last["n_stocks"]) - adv - dec, 0)
    tot = max(adv + dec + unch, 1)

    def row(label, col, good_up):
        return (f'<div class="row"><span>{label}</span><span>{int(last[col]):,}'
                f'{_delta(last[col], prev[col], good_up)}</span></div>')

    return f"""<div class="split"><span style="width:{adv / tot * 100:.1f}%;background:var(--grn)"></span>
      <span style="width:{unch / tot * 100:.1f}%;background:#d1d5db"></span>
      <span style="flex:1;background:var(--red)"></span></div>
    <div class="row"><span class="pos">{adv:,} up{_delta(adv, prev['adv'])}</span>
      <span class="neg">{dec:,} down{_delta(dec, prev['dec'], False)}</span></div>
    <div style="margin-top:10px">{row("Up 4%+", "up_4pct", True)}{row("Down 4%+", "down_4pct", False)}
      {row("At 52W highs", "new_52w_high_n", True)}{row("At 52W lows", "new_52w_low_n", False)}</div>
    <div class="note">{int(last['n_stocks']):,} stocks traded &middot; &#9650;&#9660; vs the previous session</div>"""


def _breadth_today(extras):
    sel, blocks = _switch([(label, _ad_body(bb)) for label, bb in extras["ad"].items()])
    dist = extras["dist"]
    tag = ('<span class="tag ok">Normal</span>' if dist < 3 else
           '<span class="tag" style="background:#fef3c7;color:#92400e">Elevated</span>' if dist < 5 else
           '<span class="tag bad">Under pressure</span>')
    return f"""<div class="grid g4">
  <div class="card uwrap"><div class="ttl" style="margin-bottom:8px">Advancers vs decliners</div>
    {sel.replace('class="usel"', 'class="usel" style="margin-bottom:0"')}{blocks}</div>
  {_tier_card(extras['tiers'])}
  <div class="card"><div class="ttl">Distribution days</div>
    <div class="note" style="margin-top:4px">{_esc(extras['dist_label'])} (equal-weighted) down 0.2%+
      on higher volume than the day before &mdash; institutions selling. Last 25 sessions.</div>
    <div style="margin:14px 0"><span class="big">{dist}</span> &nbsp;{tag}</div>
    <div class="note">5 or more = the uptrend is under pressure. A day drops off after 25
      sessions.</div></div>
  {_flows_card(extras['flows'])}
</div>"""


def build_dashboard(b, sec_df, universe, extras):
    act = extras["action"]
    today = b.index[-1]
    isel, iblocks = _switch([(label, _market_internals(bb, extras.get("nifty")))
                             for label, bb in extras["internals"].items()])
    breadth_html = f"""
<h2>Today's breadth</h2>
<p class="secsub">How many stocks took part, in which size bracket, and who was buying</p>
{_breadth_today(extras)}

<div class="uwrap">
<div class="shead"><div><h2>Market internals</h2>
<p class="secsub">How broad the move is underneath the index &middot; pick one to see it large
&middot; hover the chart to read that day on all four</p></div>
{isel}</div>
{iblocks}
</div>

<h2>Today's action</h2>
<p class="secsub">RS = our 1&ndash;99 relative-strength rank on weighted 3/6/9/12-month returns</p>
<div class="grid g4">
{_action_card('Up 4%+ on volume', 'Rose 4% or more on at least 2x the 20-day average volume.', act['vol_up'], _vol_chip)}
{_action_card('Breakouts today', 'Cleared their 20-day pivot high, up 2%+ on 1.5x volume.', act['breakouts'], _rs_chip)}
{_action_card('New 52-week highs', 'Closed above their previous 52-week high today.', act['new_highs'], _rs_chip)}
{_action_card('Leaders slipping', 'RS 80+ stocks that closed below their 50-day EMA today.', act['slipping'], _rs_chip)}
</div>

<h2>Sector Breadth</h2>
{_sector_table(sec_df)}

<p class="note">Data from {b.index[0]:%b %Y} to {today:%d %b %Y} ({len(b):,} sessions).
Action lists use the {_esc(universe)} universe.</p>"""

    # tab order: Monitor first, the screeners, Market breadth last
    tabs = [(name, h) for name, h in (("Monitor", extras.get("mm_html")),
                                      ("IPO screener", extras.get("ipo_html")),
                                      ("RVOL screener", extras.get("rvol_html")),
                                      ("Manas Arora", extras.get("manas_html")),
                                      ("Market breadth", breadth_html)) if h]
    tabbar = "".join(f'<button class="tabbtn{" on" if i == 0 else ""}" data-t="{i}" onclick="{_TAB_JS}">'
                     f'{_esc(name)}</button>' for i, (name, _) in enumerate(tabs))
    panes = "".join(f'<div class="tabpane" data-t="{i}"{"" if i == 0 else HIDE}>{h}</div>'
                    for i, (_, h) in enumerate(tabs))

    return f"""<!doctype html><meta charset="utf-8">
<title>NSE Market Breadth - {today:%d %b %Y}</title><style>{CSS}</style>
<div class="wrap tabwrap">
<p class="sub">NSE market breadth &middot; data to {today:%A, %d %B %Y} &middot;
last updated {datetime.now(IST):%d %b %Y, %I:%M %p} IST</p>
<div class="tabbar">{tabbar}</div>
{panes}
</div>
<script>{MB_JS}{MM_JS}</script>"""


# ---------------------------------------------------------------------------
# 9. Run
# ---------------------------------------------------------------------------
def _in_notebook():
    return "ipykernel" in sys.modules or "google.colab" in sys.modules


INTERNALS_UNIVERSES = (("All stocks", "TOTAL MARKET"), ("Nifty 50", "Nifty 50"),
                       ("Nifty 500", "Nifty 500"), ("Nifty Midcap 150", "Midcap 150"),
                       ("Nifty Smallcap 250", "Smallcap 250"))
AD_UNIVERSES = (("Nifty Total Market", "TOTAL MARKET"), ("Nifty MidSmallcap 400", "MID + SMALLCAP"))


def run_dashboard(period="2y", universe="all", refresh=False, output="breadth_dashboard.html",
                  ipo=None, rvol=None, monitor=None, manas=None):
    """Builds the dashboard, displays it in the notebook and writes the HTML file.

    `ipo` / `rvol` are screener settings dicts; None skips that tab.
    """
    symbols, industries = fetch_universe()
    closes, volumes = load_prices(symbols, period=period, refresh=refresh)
    nifty_ohlc = load_index_ohlc("^NSEI", period=period, refresh=refresh)
    nifty = (nifty_ohlc["Close"].rename("close") if nifty_ohlc is not None
             else load_index("^NSEI", period=period, refresh=refresh))
    closes, volumes = _match_calendar(closes, volumes, nifty)

    panels = segment_panels(closes)
    keys = {"all": "TOTAL MARKET", "n50": "Nifty 50", "large": "Nifty 100 (large)",
            "n500": "Nifty 500", "mid": "Midcap 150", "small": "Smallcap 250",
            "micro": "Microcap 250", "midsmall": "MID + SMALLCAP"}
    label = keys.get(universe, "TOTAL MARKET")
    panel = panels.get(label, closes)
    vol = volumes[panel.columns]

    print(f"Building dashboard for {label} ({panel.shape[1]} stocks)...")
    breadths = {}

    def breadth_of(key):
        if key not in breadths:
            breadths[key] = compute_breadth(panels[key])
        return breadths[key]

    b = compute_breadth(panel)
    dist_key = "Nifty 500" if "Nifty 500" in panels else label
    dist_panel = panels.get(dist_key, panel)
    extras = {
        "tiers": cap_tier_rows(panels),
        "dist": distribution_days(dist_panel, volumes[dist_panel.columns]),
        "dist_label": dist_key,
        "flows": fetch_fii_dii(),
        "action": daily_action(panel, vol, rs_ratings(closes)),
        "internals": {name: breadth_of(key) for name, key in INTERNALS_UNIVERSES if key in panels},
        "ad": {name: breadth_of(key) for name, key in AD_UNIVERSES if key in panels},
        "nifty": nifty,
    }

    mm_c = mm_v = None
    monitors = []
    if monitor is not None:
        try:
            params = dict(min_volume=monitor.get("min_volume", 100000),
                          thrust_pctile=monitor.get("thrust_pctile", 90),
                          lookback=monitor.get("lookback", 252))
            total = ("Nifty 750", market_monitor(closes, volumes, **params),
                     " in the Nifty Total Market (the Nifty 750)")
            monitors = [total]
            if monitor.get("universe", "full") == "full":
                tickers = fetch_nse_equities()
                extra = [t for t in tickers if t not in closes.columns]
                print(f"Monitor universe: {len(tickers):,} NSE stocks "
                      f"({len(extra):,} beyond the Nifty 750 download)")
                xc, xv = (load_prices(extra, period=period, refresh=refresh, tag="fullmkt") if extra
                          else (pd.DataFrame(index=closes.index), pd.DataFrame(index=closes.index)))
                mm_c = pd.concat([closes, xc.reindex(closes.index)], axis=1)
                mm_v = pd.concat([volumes, xv.reindex(closes.index)], axis=1)
                mm_c = mm_c.loc[:, ~mm_c.columns.duplicated()]
                mm_v = mm_v.loc[:, ~mm_v.columns.duplicated()]
                monitors.insert(0, ("All NSE stocks", market_monitor(mm_c, mm_v, **params),
                                    " listed on NSE (main board; Z category and SME excluded)"))
            seg = {}
            for name, key in (("Nifty 50", "Nifty 50"), ("Midcap 150", "Midcap 150"),
                              ("Smallcap 250", "Smallcap 250")):
                if key in panels:
                    r = panels[key].pct_change(fill_method=None).clip(-0.5, 0.5).mean(axis=1)
                    seg[name] = {"lvl": (1 + r.fillna(0)).cumprod() * 100,      # equal-weighted
                                 "p50": breadth_of(key)["pct_above_50ema"]}
            extras["mm_html"] = _mm_tab(monitors, extras["nifty"], monitor, seg, nifty_ohlc)
        except Exception as e:
            print(f"  [warn] Market Monitor failed: {e}", file=sys.stderr)
            extras["mm_html"] = f'<div class="card note" style="margin-top:16px">Market Monitor failed: {_esc(e)}</div>'

    if manas is not None:
        try:
            ohl = load_ohl(period)
            if ohl is None:
                raise RuntimeError("open/high/low cache missing - run once with REFRESH = True")
            o, h_, l_ = (x.reindex(index=closes.index, columns=closes.columns) for x in ohl)
            uc, uv = closes, volumes
            if mm_c is not None:                                  # all NSE stocks, as the Monitor
                xohl = load_ohl(period, tag="fullmkt")
                if xohl is not None:
                    uc, uv = mm_c, mm_v
                    o, h_, l_ = (pd.concat([a, b.reindex(closes.index)], axis=1)
                                 .loc[:, lambda f: ~f.columns.duplicated()].reindex(columns=uc.columns)
                                 for a, b in zip((o, h_, l_), xohl))
            m_params = {k: v for k, v in manas.items() if k not in CHART_KEYS}
            tables, frames = run_manas_screener(uc, uv, o, h_, l_, **m_params)
            note = ""
            if monitors:
                reg = mm_regime(monitors[0][1])
                stt = int(reg["state"].iloc[-1])
                verdict = {1: "Buying dominates", -1: "Selling dominates", 2: "Turning up?",
                           -2: "Turning down?"}.get(stt, "Mixed")
                note = (f"<b>Market ({monitors[0][0]}): {verdict}</b> (score {reg['score'].iloc[-1]:.0f}). "
                        f"Manas checks the market before any set-up and sizes down or sits out when "
                        f"it is weak &mdash; breakouts follow through far more often when buying dominates.")
            print(f"Manas screener: " + ", ".join(f"{k} {len(v)}" for k, v in tables.items()))
            extras["manas_html"] = _manas_tab(tables, frames, manas, note)
        except Exception as e:
            print(f"  [warn] Manas screener failed: {e}", file=sys.stderr)
            extras["manas_html"] = f'<div class="card note" style="margin-top:16px">Screener failed: {_esc(e)}</div>'


    for key, cfg, run, tab in (("ipo_html", ipo, lambda c: run_ipo_screener(**c), _ipo_tab),
                               ("rvol_html", rvol, lambda c: run_rvol_screener(symbols, **c), _rvol_tab)):
        if cfg is None:
            continue
        params = {k: v for k, v in cfg.items() if k not in CHART_KEYS}
        try:
            extras[key] = tab(*run(params), cfg)
        except Exception as e:
            print(f"  [warn] {key.split('_')[0].upper()} screener failed: {e}", file=sys.stderr)
            extras[key] = f'<div class="card note" style="margin-top:16px">Screener failed: {_esc(e)}</div>'

    page = build_dashboard(b, sec_df=sector_rows(closes, industries), universe=label,
                           extras=extras)

    if output:
        os.makedirs("docs", exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(page)
        print(f"Saved {output}")

    if _in_notebook():
        from IPython.display import HTML, display
        display(HTML(page))
    return page


# ---------------------------------------------------------------------------
# 10. CONFIG - edit before running
# ---------------------------------------------------------------------------
# 2y is the minimum that seeds a 200 EMA and a real 52-week high, and it keeps
# the download small. Nothing here backtests, so longer history buys nothing.
PERIOD = "2y"
UNIVERSE = "midsmall"               # all | n50 | large | n500 | mid | small | micro | midsmall
REFRESH = True                      # True to ignore the cached prices
OUTPUT_HTML = "docs/index.html"   # None to skip writing the file

# IPO screener tab (set to None to skip). Same defaults as the relaxed IPO notebook.
IPO_SCREENER = dict(
    months=6,                   # "recent IPO" = listed within this many months
    include_sme=False,          # also include SME-platform listings
    max_same_day_listings=3,    # drop dates shared by more symbols than this (artifact filter)
    near_low_pct=0.05,          # "at low" if within this fraction above the post-listing low
    near_high_pct=0.20,         # "near high" if within this fraction below the post-listing high
    consolidation_days=10,      # recent window checked for a tight range
    range_pct_max=0.25,         # max High-Low range over that window, as a fraction of price
    volume_lookback=10,         # recent vs prior average-volume window
    min_avg_volume=100000,      # liquidity filter over avg_volume_days
    avg_volume_days=5,
    require_inside_bar=False,   # latest day must be an inside bar on lower volume
    show_charts=True,           # candlestick chart per match at the bottom of the tab
    chart_months=6,
    chart_emas=(20, 50, 200),
    max_charts=0,               # 0 = chart every match
)

# Market Monitor tab - Stockbee breadth sheet (set to None to skip).
MARKET_MONITOR = dict(
    universe="full",            # full = All NSE stocks (default) + Nifty 750 in the dropdown
                                # total = Nifty 750 only (faster: no extra download)
    rows=252,                   # sessions in the daily sheet (1M shown first, expand to 1Y)
    min_volume=100000,          # shares; a 4% move only counts on at least this volume
    thrust_pctile=90,           # thrust day = 4% count in the top 10% of the last `lookback`
    lookback=252,               # sessions - NSE's own version of Stockbee's US "300"
)

# Manas Arora set-ups tab (set to None to skip). Uses the Monitor's All NSE download.
MANAS_SCREENER = dict(
    min_price=30.0,             # his trend filter: close above Rs 30 ...
    near_high_pct=25.0,         # ... within 25% of the 52-week high
    above_low_pct=50.0,         # ... at least 50% above the 52-week low
    purple_move=0.05,           # "purple dot" = a 5%+ day ...
    purple_volume=1_000_000,    # ... on 1M+ shares (within the last 60 sessions)
    prior_run_pct=50.0,         # SVRO needs a 50%+ prior run
    tight_range_pct=3.0,        # SVRO: average daily range of the last 3 days
    max_ema_dist_pct=5.0,       # SVRO: close 0-5% above the 21 EMA
    pullback_min_pct=4.0,       # 2A: 4-20% off the 15-day high, at the 21 EMA
    pullback_max_pct=20.0,
    breakout_vol_x=1.5,         # breakout volume vs the 20-day average
    pre_breakout_pct=3.0,       # pre-breakout: within 3% of the 20-day high ...
    base_range_pct=10.0,        # ... with a 10-day range of 10% or less
    show_charts=True,
    chart_months=6,
    chart_emas=(10, 21, 50),    # his working averages
    max_charts=60,              # triggers first, then set-ups
)

# RVOL screener tab (set to None to skip). Same defaults as the RVOL notebook.
RVOL_SCREENER = dict(
    min_rvol_pct=1000.0,        # 1000% = 10x the prior average volume
    lookback_days=15,           # spike must be within the last N sessions
    avg_window=20,              # baseline days before the spike
    min_green_body_pct=5.0,     # (Close - Open) / Open on the spike day
    max_retrace_pct=60.0,       # no Low since the spike may retrace more of its range
    near_high_pct=35.0,         # first table = within this % of the 52-week high
    exclude_corporate_actions=True,
    require_green_candle=True,
    min_avg_volume=0,
    min_close=0.0,
    show_charts=True,           # candlestick chart per match at the bottom of the tab
    chart_months=6,
    chart_emas=(20, 50, 200),
    max_charts=0,               # 0 = chart every match
)


if __name__ == "__main__":
    run_dashboard(period=PERIOD, universe=UNIVERSE, refresh=REFRESH, output=OUTPUT_HTML,
                  ipo=IPO_SCREENER, rvol=RVOL_SCREENER, monitor=MARKET_MONITOR, manas=MANAS_SCREENER)
