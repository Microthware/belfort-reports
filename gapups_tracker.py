
from __future__ import annotations
import csv, os, re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

HISTORY_DIR = "history"
CSV_PATH = os.path.join(HISTORY_DIR, "gapups-history.csv")
CSV_HEADERS = [
    "date","symbol","price",
    "next_day_price","next_day_pct",
    "week_price","week_pct",
    "month_price","month_pct",
    "mkt_day_pct","mkt_week_pct","mkt_month_pct",
    "streak"
]

def _ensure():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADERS)

def _read_rows() -> List[Dict[str,str]]:
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def _write_rows(rows: List[Dict[str,str]]):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADERS); w.writeheader()
        for r in rows: w.writerow(r)

def _last_close(symbol: str, on_date: str) -> Optional[float]:
    if yf is None:
        return None
    try:
        d0 = datetime.fromisoformat(on_date)
        # fetch +/- 3 days to find nearest close on/after date
        start = (d0 - timedelta(days=2)).date().isoformat()
        end = (d0 + timedelta(days=7)).date().isoformat()
        hist = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False)
        if hist is None or hist.empty:
            return None
        # choose first close on or after d0.date()
        for ts, row in hist.iterrows():
            d = ts.date().isoformat()
            if d >= on_date:
                px = float(row["Close"])
                if px > 0: return px
        # fallback latest
        px = float(hist["Close"].iloc[-1])
        return px if px>0 else None
    except Exception:
        return None

def record_today_from_finviz(tickers: List[str], today: Optional[str] = None):
    """Append today's tickers with entry price (close on or after today). Dedup by (date,symbol)."""
    _ensure()
    today = today or datetime.utcnow().date().isoformat()
    rows = _read_rows()
    seen = {(r["date"], r["symbol"]) for r in rows}
    for sym in tickers:
        sym = sym.strip().upper()
        key = (today, sym)
        if key in seen:
            continue
        price = _last_close(sym, today)
        rows.append({
            "date": today, "symbol": sym, "price": f"{price:.6f}" if price else "",
            "next_day_price":"", "next_day_pct":"",
            "week_price":"", "week_pct":"",
            "month_price":"", "month_pct":"",
            "mkt_day_pct":"", "mkt_week_pct":"", "mkt_month_pct":"",
            "streak": "1",
        })
        seen.add(key)
    _write_rows(rows)

def _series(symbol: str, start: str, end: Optional[str] = None):
    if yf is None: return []
    try:
        hist = yf.Ticker(symbol).history(start=start, end=end or None, auto_adjust=False)
        if hist is None or hist.empty: return []
        return [(ts.date().isoformat(), float(c)) for ts,c in hist["Close"].items() if float(c)>0]
    except Exception:
        return []

def _idx_on_or_after(dates: List[str], d0: str) -> Optional[int]:
    if d0 in dates: return dates.index(d0)
    for i,d in enumerate(dates):
        if d > d0: return i
    return None

def backfill_outcomes():
    """Compute T+1, T+5, T+21 outcomes and SPY market context. Also recompute streaks."""
    _ensure()
    rows = _read_rows()
    if not rows:
        return
    earliest = min(r["date"] for r in rows if r.get("date"))
    end = (datetime.utcnow().date() + timedelta(days=1)).isoformat()
    # preload SPY and all symbols
    spy_ser = _series("SPY", earliest, end)
    spy_dates = [d for d,_ in spy_ser]
    cache: Dict[str, List[tuple]] = {}
    for r in rows:
        sym = r["symbol"]
        if sym not in cache:
            cache[sym] = _series(sym, earliest, end)
    # compute outcomes
    for r in rows:
        d0 = r.get("date"); sym = r.get("symbol"); p0 = float(r.get("price") or 0) or None
        ser = cache.get(sym, [])
        dates = [d for d,_ in ser]
        i0 = _idx_on_or_after(dates, d0) if d0 else None
        if i0 is not None and p0:
            # T+1, +5, +21 (approx trg days)
            for k, off in (("next_day",1), ("week",5), ("month",21)):
                idx = i0 + off
                if idx < len(ser):
                    px = ser[idx][1]
                    r[f"{k}_price"] = f"{px:.6f}"
                    r[f"{k}_pct"] = f"{(px/p0 - 1.0):.6f}"
        # market context from SPY
        if d0 and spy_dates:
            j0 = _idx_on_or_after(spy_dates, d0)
            if j0 is not None:
                for k, off, fld in (("day",1,"mkt_day_pct"), ("week",5,"mkt_week_pct"), ("month",21,"mkt_month_pct")):
                    j = j0 + off
                    if j < len(spy_ser):
                        sp0 = spy_ser[j0][1]; sp1 = spy_ser[j][1]
                        r[fld] = f"{(sp1/sp0 - 1.0):.6f}"
    # recompute streaks (consecutive days)
    by_sym: Dict[str, List[str]] = {}
    for r in rows:
        by_sym.setdefault(r["symbol"], []).append(r["date"])
    for sym, dates in by_sym.items():
        dates.sort(reverse=True)
        # compute streaks descending
        prev = None; cnt = 0
        streak_map = {}
        for d in dates:
            if prev is None:
                cnt = 1
            else:
                # if today's date is exactly previous -1 day (calendar), increment; else reset
                dd = datetime.fromisoformat(prev).date() - timedelta(days=1)
                if d == dd.isoformat():
                    cnt += 1
                else:
                    cnt = 1
            streak_map[d] = cnt
            prev = d
        # assign back
        for r in rows:
            if r["symbol"]==sym:
                r["streak"] = str(streak_map.get(r["date"], 1))
    _write_rows(rows)

def load_rows_for_render() -> List[Dict[str, str]]:
    _ensure()
    rows = _read_rows()
    # sort newest first
    rows.sort(key=lambda x: (x.get("date") or ""), reverse=True)
    return rows
