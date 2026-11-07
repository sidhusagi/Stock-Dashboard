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
    # yfinance can append a session that only a handful of tickers have printed yet
    counts = closes.notna().sum(axis=1)
    keep = counts >= counts.max() * min_frac
    for d in closes.index[~keep][-3:]:
        print(f"  [warn] dropped {d:%d %b}: only {counts[d]} of {counts.max()} stocks have a close yet "
              f"- rerun with REFRESH = True later", file=sys.stderr)
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


def load_prices(symbols, period="2y", refresh=False, batch_size=60, verbose=True):
    """(closes, volumes) [dates x tickers] panels, cached as gzipped CSV so re-runs are instant."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = {k: os.path.join(CACHE_DIR, f"{k}_{period}.csv.gz") for k in ("closes", "volumes")}
    fresh = all(_prices_fresh(p) for p in paths.values())
    if fresh and not refresh:
        if verbose:
            print(f"Using cached prices from {CACHE_DIR}")
        return _drop_partial_sessions(*(pd.read_csv(paths[k], index_col=0, parse_dates=True)
                                        for k in ("closes", "volumes")))

    tickers = [f"{s}.NS" for s in symbols]
    if verbose:
        print(f"Downloading {period} of daily closes and volumes for {len(tickers)} NSE stocks "
              f"(one-off; cached for the rest of the day)...")

    close_frames, volume_frames = [], []
    for n, batch in enumerate(_chunks(tickers, batch_size), 1):
        try:
            data = yf.download(batch, period=period, interval="1d", auto_adjust=True,
                               group_by="column", threads=True, progress=False)
        except Exception as e:
            print(f"  [warn] batch {n} failed: {e}", file=sys.stderr)
            continue
        if data is None or data.empty:
            continue
        multi = isinstance(data.columns, pd.MultiIndex)
        close_frames.append(data["Close"] if multi else data[["Close"]])
        volume_frames.append(data["Volume"] if multi else data[["Volume"]])
        if verbose:
            print(f"  batch {n}: {close_frames[-1].shape[1]} symbols", flush=True)

    closes = pd.concat(close_frames, axis=1).sort_index()
    closes = closes.loc[:, ~closes.columns.duplicated()].dropna(axis=1, how="all")
    volumes = pd.concat(volume_frames, axis=1).sort_index()
    volumes = volumes.loc[:, ~volumes.columns.duplicated()].reindex(index=closes.index, columns=closes.columns)
    closes.to_csv(paths["closes"])
    volumes.to_csv(paths["volumes"])
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
    tabs = [("Market breadth", None)] + [(name, h) for name, h in
                                         (("IPO screener", extras.get("ipo_html")),
                                          ("RVOL screener", extras.get("rvol_html"))) if h]
    tabbar = "".join(f'<button class="tabbtn{" on" if i == 0 else ""}" data-t="{i}" onclick="{_TAB_JS}">'
                     f'{_esc(name)}</button>' for i, (name, _) in enumerate(tabs))
    other_panes = "".join(f'<div class="tabpane" data-t="{i}" style="display:none">{h}</div>'
                          for i, (_, h) in enumerate(tabs) if i)

    return f"""<!doctype html><meta charset="utf-8">
<title>NSE Market Breadth - {today:%d %b %Y}</title><style>{CSS}</style>
<div class="wrap tabwrap">
<p class="sub">NSE market breadth &middot; {today:%A, %d %B %Y}</p>
<div class="tabbar">{tabbar}</div>
<div class="tabpane" data-t="0">

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
Action lists use the {_esc(universe)} universe.</p>
</div>
{other_panes}
</div>
<script>{MB_JS}</script>"""


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
                  ipo=None, rvol=None):
    """Builds the dashboard, displays it in the notebook and writes the HTML file.

    `ipo` / `rvol` are screener settings dicts; None skips that tab.
    """
    symbols, industries = fetch_universe()
    closes, volumes = load_prices(symbols, period=period, refresh=refresh)

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
        "nifty": load_index("^NSEI", period=period, refresh=refresh),
    }

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
                  ipo=IPO_SCREENER, rvol=RVOL_SCREENER)
