#!/usr/bin/env python3
"""
Swing-trade candidate screener.

Pulls ~1 year of daily bars for a liquid US equity universe, computes trend /
momentum / volatility indicators, applies hard filters, scores what survives,
and writes a ranked JSON file plus an append-only pick log for hit-rate tracking.

This produces CANDIDATES, not recommendations. Nothing here is financial advice.

Usage:
    python screener.py              # run the scan
    python screener.py --review     # score how past picks actually performed
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------------
# Config -- tune these, they're the whole personality of the screener
# ----------------------------------------------------------------------------

CFG = {
    "cap_tier": "smallmicro",      # micro | small | smallmicro | large | all
    "min_price": 2.0,              # below ~$2 is mostly noise and dilution
    "max_price": 200.0,
    "min_dollar_volume": 3_000_000,  # $/day -- the right liquidity test for small caps
    "min_atr_pct": 2.5,
    "max_atr_pct": 12.0,           # small caps move more; ceiling is looser
    "min_adx": 20.0,               # trend strength
    "rsi_band": (35.0, 55.0),      # pullback-in-uptrend sweet spot
    "max_rsi": 65.0,               # hard reject: too extended to enter here
    "earnings_blackout_days": 5,
    "top_n": 15,
    "history_days": 400,
    "benchmark": "IWM",            # small-cap benchmark, not SPY
    "max_universe": 3000,          # cap the scan size; raise if you have patience
}

# Market cap bands in dollars
CAP_BANDS = {
    "micro":      (50e6, 300e6),
    "small":      (300e6, 2e9),
    "smallmicro": (50e6, 2e9),
    "large":      (10e9, 1e15),
    "all":        (0, 1e15),
}

OUT_DIR = "docs"          # GitHub Pages serves this folder for free
LOG_FILE = "picks_log.csv"

# ----------------------------------------------------------------------------
# Indicators -- Wilder's smoothing throughout, no TA-Lib dependency
# ----------------------------------------------------------------------------

def wilder(series: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing == EMA with alpha = 1/n."""
    return series.ewm(alpha=1.0 / n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilder(gain, n)
    avg_loss = wilder(loss, n)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return wilder(tr, n)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    atr_n = atr(df, n).replace(0, np.nan)
    plus_di = 100 * wilder(pd.Series(plus_dm, index=df.index), n) / atr_n
    minus_di = 100 * wilder(pd.Series(minus_dm, index=df.index), n) / atr_n

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return wilder(dx.fillna(0), n)


# ----------------------------------------------------------------------------
# Universe
# ----------------------------------------------------------------------------

FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD",
    "NFLX", "CRM", "ORCL", "ADBE", "INTC", "QCOM", "TXN", "MU", "PANW", "SNOW",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "AXP", "V", "MA", "PYPL",
    "XOM", "CVX", "COP", "SLB", "OXY", "MPC", "PSX", "DVN", "HAL",
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "AMGN", "GILD", "BMY",
    "WMT", "COST", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD", "DIS", "CMG",
    "CAT", "DE", "BA", "GE", "HON", "UPS", "FDX", "LMT", "RTX", "UNP",
    "F", "GM", "RIVN", "UBER", "ABNB", "DAL", "UAL", "CCL", "MAR",
    "COIN", "XYZ", "SHOP", "SPOT", "ROKU", "PLTR", "SOFI", "HOOD", "DKNG",
    "T", "VZ", "CMCSA", "PEP", "KO", "PG", "MDLZ", "CL", "KMB",
]


def get_universe() -> list:
    """
    Full US-listed common stock universe from the NASDAQ Trader symbol
    directory -- free, no key, includes every small and micro cap.
    Filters out ETFs, test issues, warrants, units and preferreds.
    """
    import io
    import urllib.request

    sources = [
        ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
        ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol"),
    ]
    tickers = set()

    for url, symcol in sources:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                text = r.read().decode("utf-8", errors="ignore")

            df = pd.read_csv(io.StringIO(text), sep="|")
            df = df[~df[symcol].astype(str).str.contains("File Creation", na=False)]

            if "ETF" in df.columns:
                df = df[df["ETF"].astype(str).str.upper() != "Y"]
            if "Test Issue" in df.columns:
                df = df[df["Test Issue"].astype(str).str.upper() != "Y"]

            for sym in df[symcol].astype(str):
                sym = sym.strip()
                # $ = warrants/units/rights, . = preferreds and share classes,
                # 5-letter NASDAQ symbols ending in W/R/U/P are derivatives
                if not sym or "$" in sym or "." in sym or len(sym) > 5:
                    continue
                if len(sym) == 5 and sym[-1] in "WRUPQ":
                    continue
                tickers.add(sym)
        except Exception as e:
            print(f"  universe fetch failed for {url}: {e}", file=sys.stderr)

    if len(tickers) < 100:
        print("  falling back to built-in universe", file=sys.stderr)
        return FALLBACK_UNIVERSE

    out = sorted(tickers)
    if len(out) > CFG["max_universe"]:
        print(f"  trimming {len(out)} -> {CFG['max_universe']} (max_universe)")
        out = out[:CFG["max_universe"]]
    return out


def get_market_caps(tickers: list) -> dict:
    """
    Fetch market cap for a SHORT list of tickers -- one call each, so only
    run this on names that already survived the technical filters.
    """
    caps = {}
    for i, t in enumerate(tickers):
        if i and i % 25 == 0:
            print(f"    caps {i}/{len(tickers)}...")
        try:
            fi = yf.Ticker(t).fast_info
            cap = None
            for key in ("market_cap", "marketCap"):
                try:
                    cap = fi[key] if not hasattr(fi, key) else getattr(fi, key)
                except Exception:
                    continue
                if cap:
                    break
            if cap:
                caps[t] = float(cap)
        except Exception:
            continue
    return caps


# ----------------------------------------------------------------------------
# Earnings blackout (optional -- needs a free Finnhub key)
# ----------------------------------------------------------------------------

def earnings_blackout(days: int) -> set:
    key = os.environ.get("FINNHUB_KEY")
    if not key:
        print("  no FINNHUB_KEY set -- skipping earnings blackout", file=sys.stderr)
        return set()
    try:
        import urllib.request
        today = datetime.now(timezone.utc).date()
        url = (f"https://finnhub.io/api/v1/calendar/earnings?"
               f"from={today}&to={today + timedelta(days=days)}&token={key}")
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.load(r)
        blocked = {e["symbol"] for e in data.get("earningsCalendar", [])}
        print(f"  earnings blackout: {len(blocked)} tickers reporting soon")
        return blocked
    except Exception as e:
        print(f"  earnings lookup failed: {e}", file=sys.stderr)
        return set()


# ----------------------------------------------------------------------------
# Data fetch
# ----------------------------------------------------------------------------

def fetch_bars(tickers: list, days: int) -> dict:
    """Batch-download daily bars. Returns {ticker: DataFrame}."""
    start = datetime.now(timezone.utc).date() - timedelta(days=days)
    out = {}
    chunk_size = 100

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        print(f"  downloading {i + 1}-{i + len(chunk)} of {len(tickers)}...")
        try:
            raw = yf.download(chunk, start=start, interval="1d",
                              group_by="ticker", auto_adjust=False,
                              threads=True, progress=False)
        except Exception as e:
            print(f"    chunk failed: {e}", file=sys.stderr)
            continue

        for t in chunk:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.dropna()
                if len(df) >= 120:
                    out[t] = df
            except Exception:
                continue
    return out


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def evaluate(ticker: str, df: pd.DataFrame, bench_ret: dict) -> dict | None:
    close = df["Close"]
    px = float(close.iloc[-1])

    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(df) >= 200 else np.nan
    avg_vol = float(df["Volume"].rolling(20).mean().iloc[-1])
    rel_vol = float(df["Volume"].iloc[-1] / avg_vol) if avg_vol else 0.0
    dollar_vol = avg_vol * px

    # largest single-day move over the last month, as a % -- gap risk proxy
    max_gap = float(close.pct_change().iloc[-21:].abs().max() * 100)

    atr_val = float(atr(df).iloc[-1])
    atr_pct = 100 * atr_val / px
    rsi_val = float(rsi(close).iloc[-1])
    adx_val = float(adx(df).iloc[-1])

    ret20 = 100 * (px / float(close.iloc[-21]) - 1) if len(close) > 21 else 0.0
    ret60 = 100 * (px / float(close.iloc[-61]) - 1) if len(close) > 61 else 0.0
    rs20 = ret20 - bench_ret.get("ret20", 0.0)
    rs60 = ret60 - bench_ret.get("ret60", 0.0)

    # ---- hard filters: fail any one and you're out -----------------------
    fails = []
    if not (CFG["min_price"] <= px <= CFG["max_price"]):
        fails.append("price")
    if dollar_vol < CFG["min_dollar_volume"]:
        fails.append("liquidity")
    if not (CFG["min_atr_pct"] <= atr_pct <= CFG["max_atr_pct"]):
        fails.append("volatility")
    if adx_val < CFG["min_adx"]:
        fails.append("no_trend")
    # Trend must be intact (20 above 50, price above 50), but allow price to
    # dip modestly below the 20-day -- that dip IS the pullback we want.
    if not (sma20 > sma50 and px > sma50 and px > 0.95 * sma20):
        fails.append("not_uptrend")
    if rsi_val > CFG["max_rsi"]:
        fails.append("extended")
    # Small caps gap violently; a recent 25%+ single-day move means your stop
    # is decoration. Skip anything that just did that.
    if max_gap > 25:
        fails.append("gappy")
    if fails:
        return None

    # ---- score the survivors, 0-100 --------------------------------------
    lo, hi = CFG["rsi_band"]
    # pullback: peak score in the middle of the band, decaying outside it
    mid = (lo + hi) / 2
    pullback = max(0.0, 1 - abs(rsi_val - mid) / 25) * 25

    trend = min(1.0, (adx_val - CFG["min_adx"]) / 25) * 15
    if not np.isnan(sma200) and px > sma200:
        trend += 10

    strength = (np.clip(rs20 / 15, 0, 1) * 15) + (np.clip(rs60 / 25, 0, 1) * 15)
    volume = np.clip((rel_vol - 1.0) / 1.0, 0, 1) * 10
    room = np.clip((atr_pct - CFG["min_atr_pct"]) / 3, 0, 1) * 10

    score = round(float(pullback + trend + strength + volume + room), 1)

    # ---- suggested levels -------------------------------------------------
    stop = px - 1.5 * atr_val
    # target is the greater of an ATR projection and the recent swing high,
    # so R:R actually varies with how much overhead room the name has
    target = max(px + 2.0 * atr_val, float(df["High"].iloc[-20:].max()))

    return {
        "ticker": ticker,
        "score": score,
        "price": round(px, 2),
        "cap_m": None,              # filled in later by get_market_caps
        "rsi": round(rsi_val, 1),
        "adx": round(adx_val, 1),
        "atr_pct": round(atr_pct, 2),
        "rel_vol": round(rel_vol, 2),
        "rs20": round(rs20, 1),
        "rs60": round(rs60, 1),
        "dollar_vol_m": round(dollar_vol / 1e6, 1),
        "max_gap": round(max_gap, 1),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "rr": round((target - px) / (px - stop), 1) if px > stop else 0,
        "why": build_reason(rsi_val, adx_val, rs20, rel_vol, px, sma200),
    }


def build_reason(rsi_val, adx_val, rs20, rel_vol, px, sma200) -> str:
    bits = []
    lo, hi = CFG["rsi_band"]
    if lo <= rsi_val <= hi:
        bits.append(f"pullback to RSI {rsi_val:.0f} inside an uptrend")
    if adx_val >= 25:
        bits.append(f"strong trend (ADX {adx_val:.0f})")
    if rs20 > 3:
        bits.append(f"outperforming SPY by {rs20:.0f}% over 20d")
    if rel_vol > 1.5:
        bits.append(f"volume {rel_vol:.1f}x normal")
    if not np.isnan(sma200) and px > sma200:
        bits.append("above 200-day")
    return "; ".join(bits) if bits else "passed all filters"


# ----------------------------------------------------------------------------
# Pick logging -- this is the part that keeps you honest
# ----------------------------------------------------------------------------

def log_picks(picks: list):
    today = datetime.now(timezone.utc).date().isoformat()
    rows = [{
        "date": today,
        "ticker": p["ticker"],
        "score": p["score"],
        "entry": p["price"],
        "stop": p["stop"],
        "target": p["target"],
    } for p in picks]
    new = pd.DataFrame(rows)

    if os.path.exists(LOG_FILE):
        old = pd.read_csv(LOG_FILE)
        old = old[~((old["date"] == today))]  # idempotent re-runs
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(LOG_FILE, index=False)
    print(f"  logged {len(rows)} picks to {LOG_FILE}")


def review():
    """Compare logged picks against what actually happened."""
    if not os.path.exists(LOG_FILE):
        print("No pick log yet -- run a scan first.")
        return

    log = pd.read_csv(LOG_FILE)
    log["date"] = pd.to_datetime(log["date"])
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=3)
    matured = log[log["date"] <= cutoff]
    if matured.empty:
        print("No picks old enough to judge yet.")
        return

    tickers = sorted(matured["ticker"].unique())
    bars = fetch_bars(tickers, 90)

    results = []
    for _, row in matured.iterrows():
        df = bars.get(row["ticker"])
        if df is None:
            continue
        after = df[df.index > row["date"]]
        if after.empty:
            continue
        hit_target = bool((after["High"] >= row["target"]).any())
        hit_stop = bool((after["Low"] <= row["stop"]).any())
        # if both, assume the stop came first -- the pessimistic assumption
        outcome = "stop" if hit_stop else ("target" if hit_target else "open")
        current = float(after["Close"].iloc[-1])
        results.append({
            "date": row["date"].date(), "ticker": row["ticker"],
            "outcome": outcome,
            "pct": round(100 * (current / row["entry"] - 1), 1),
        })

    res = pd.DataFrame(results)
    closed = res[res["outcome"] != "open"]
    print("\n=== PICK PERFORMANCE ===")
    print(f"Picks evaluated : {len(res)}")
    if len(closed):
        wins = (closed["outcome"] == "target").sum()
        print(f"Target hit      : {wins}/{len(closed)}  ({100*wins/len(closed):.0f}%)")
    print(f"Mean return     : {res['pct'].mean():.1f}%")
    print(f"Median return   : {res['pct'].median():.1f}%")
    print(f"\n{res.tail(25).to_string(index=False)}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def scan():
    print("Building universe...")
    universe = get_universe()
    print(f"  {len(universe)} tickers")

    blocked = earnings_blackout(CFG["earnings_blackout_days"])
    universe = [t for t in universe if t not in blocked]

    print("Fetching bars...")
    bars = fetch_bars(universe + [CFG["benchmark"]], CFG["history_days"])
    print(f"  usable history for {len(bars)} tickers")

    bench = bars.pop(CFG["benchmark"], None)
    bench_ret = {}
    if bench is not None and len(bench) > 61:
        c = bench["Close"]
        bench_ret = {
            "ret20": 100 * (float(c.iloc[-1]) / float(c.iloc[-21]) - 1),
            "ret60": 100 * (float(c.iloc[-1]) / float(c.iloc[-61]) - 1),
        }
        print(f"  {CFG['benchmark']} 20d {bench_ret['ret20']:+.1f}% / 60d {bench_ret['ret60']:+.1f}%")

    print("Scoring...")
    candidates = []
    for ticker, df in bars.items():
        try:
            r = evaluate(ticker, df, bench_ret)
            if r:
                candidates.append(r)
        except Exception as e:
            print(f"  {ticker} errored: {e}", file=sys.stderr)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    print(f"  {len(candidates)} passed technical filters")

    # ---- market cap filter, last because it costs one API call per name ----
    lo_cap, hi_cap = CAP_BANDS[CFG["cap_tier"]]
    if CFG["cap_tier"] != "all":
        # only price the best candidates -- no point capping the whole list
        shortlist = candidates[:80]
        print(f"Fetching market caps for top {len(shortlist)}...")
        caps = get_market_caps([c["ticker"] for c in shortlist])

        in_band = []
        for c in shortlist:
            cap = caps.get(c["ticker"])
            if cap is None:
                continue                      # unknown cap -> skip, don't guess
            c["cap_m"] = round(cap / 1e6, 1)
            if lo_cap <= cap <= hi_cap:
                in_band.append(c)
        print(f"  {len(in_band)} inside the {CFG['cap_tier']} cap band "
              f"(${lo_cap/1e6:.0f}M-${hi_cap/1e9:.1f}B)")
        candidates = in_band

    top = candidates[:CFG["top_n"]]

    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(bars),
        "passed_filters": len(candidates),
        "cap_tier": CFG["cap_tier"],
        "benchmark": bench_ret,
        "config": CFG,
        "picks": top,
        "disclaimer": "Screening output only. Not investment advice.",
    }
    with open(f"{OUT_DIR}/picks.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    log_picks(top)

    print(f"\nTop {len(top)}:\n")
    if top:
        print(pd.DataFrame(top)[
            ["ticker", "score", "price", "cap_m", "rsi", "adx", "atr_pct",
             "dollar_vol_m", "rs20", "max_gap", "stop", "target", "rr"]
        ].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", action="store_true",
                    help="score past picks instead of scanning")
    args = ap.parse_args()
    review() if args.review else scan()
