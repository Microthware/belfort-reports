
from __future__ import annotations
import csv
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Iterable, Dict, List, Optional

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = HISTORY_DIR / "gapups-history.csv"

CSV_HEADERS = [
    "date", "name", "symbol", "price",
    "next_day_price", "next_day_pct",
    "week_later_price", "week_later_pct",
    "month_later_price", "month_later_pct",
    "mkt_day_pct", "mkt_week_pct", "mkt_month_pct",
]

TRADING_OFFSETS = {"day": 1, "week": 5, "month": 21}

def _ensure_csv():
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADERS)

def _today_iso_local() -> str:
    return datetime.now(timezone.utc).astimezone().date().isoformat()

def _fetch_series(symbol: str, start: str, end: Optional[str] = None):
    if yf is None:
        return None
    try:
        hist = yf.Ticker(symbol).history(start=start, end=end or None, auto_adjust=False)
        if hist is None or hist.empty:
            return None
        return [(ts.date().isoformat(), float(c)) for ts, c in hist["Close"].items() if float(c) > 0]
    except Exception:
        return None

def _idx_forward(dates: List[str], base_date: str, offset: int) -> Optional[int]:
    if base_date in dates:
        i0 = dates.index(base_date)
    else:
        i0 = next((i for i,d in enumerate(dates) if d > base_date), None)
        if i0 is None:
            return None
    i = i0 + offset
    return i if 0 <= i < len(dates) else None

def _pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or a == 0:
        return None
    return (b / a) - 1.0

def record_today_from_finviz(tickers: Iterable[str], names: Optional[Dict[str,str]] = None, run_date_iso: Optional[str] = None) -> None:
    """Append today's Finviz gap-ups that were already scraped elsewhere this run."""
    _ensure_csv()
    date_str = run_date_iso or _today_iso_local()

    existing = set()
    rows = []
    with CSV_PATH.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
            existing.add((row["date"], row["symbol"]))

    tickers = [ (t or "").upper().strip() for t in (tickers or []) if t ]
    if not tickers:
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_HEADERS); w.writeheader()
            for row in rows: w.writerow(row)
        return

    start = (datetime.fromisoformat(date_str) - timedelta(days=3)).date().isoformat()
    end   = (datetime.fromisoformat(date_str) + timedelta(days=2)).date().isoformat()

    for sym in tickers:
        if (date_str, sym) in existing:
            continue
        nm = (names or {}).get(sym)
        if nm is None and yf is not None:
            try:
                info = getattr(yf.Ticker(sym), "get_info", None)
                nm = (info() or {}).get("shortName") if callable(info) else None
            except Exception:
                nm = None
        nm = nm or sym

        series = _fetch_series(sym, start=start, end=end) or []
        dates = [d for d,_ in series]
        if not dates:
            continue
        base_idx = dates.index(date_str) if date_str in dates else max([i for i,d in enumerate(dates) if d < date_str], default=None)
        if base_idx is None:
            continue
        base_price = series[base_idx][1]

        rows.append({
            "date": date_str, "name": nm, "symbol": sym, "price": f"{base_price:.4f}",
            "next_day_price":"", "next_day_pct":"",
            "week_later_price":"", "week_later_pct":"",
            "month_later_price":"", "month_later_pct":"",
            "mkt_day_pct":"", "mkt_week_pct":"", "mkt_month_pct":""
        })

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADERS); w.writeheader()
        for row in rows: w.writerow(row)

def backfill_outcomes() -> None:
    """Fill forward outcomes (T+1/T+5/T+21) and SPY D/W/M market context where blank."""
    _ensure_csv()
    rows = list(csv.DictReader(CSV_PATH.open("r", encoding="utf-8")))
    if not rows or yf is None:
        return

    all_dates = sorted({r["date"] for r in rows})
    if not all_dates:
        return
    start = (datetime.fromisoformat(all_dates[0]) - timedelta(days=3)).date().isoformat()
    end   = (datetime.fromisoformat(all_dates[-1]) + timedelta(days=40)).date().isoformat()

    spy = _fetch_series("SPY", start=start, end=end) or []
    spy_dates = [d for d,_ in spy]

    cache: Dict[str, List[tuple]] = {}

    for r in rows:
        sym = r["symbol"]
        if sym not in cache:
            cache[sym] = _fetch_series(sym, start=start, end=end) or []
        ser = cache[sym]
        dates = [d for d,_ in ser]
        if not dates:
            continue

        d = r["date"]
        try:
            base = float(r["price"])
        except Exception:
            base = None

        def fill(off_key: str, price_key: str, pct_key: str):
            if r.get(pct_key) and r.get(price_key):
                return
            idx = _idx_forward(dates, d, TRADING_OFFSETS[off_key])
            if idx is None or base is None:
                return
            fwd = ser[idx][1]
            r[price_key] = f"{fwd:.4f}"
            r[pct_key]   = f"{_pct(base, fwd):.6f}"

        fill("day",   "next_day_price",   "next_day_pct")
        fill("week",  "week_later_price", "week_later_pct")
        fill("month", "month_later_price","month_later_pct")

        def fill_spy(off_key: str, key: str):
            if r.get(key):
                return
            # find base index on or before d
            if d in spy_dates:
                i0 = spy_dates.index(d)
            else:
                i0 = max([i for i,sd in enumerate(spy_dates) if sd < d], default=None)
            if i0 is None:
                return
            base_spy = spy[i0][1]
            i1 = i0 + TRADING_OFFSETS[off_key]
            if 0 <= i1 < len(spy):
                r[key] = f"{_pct(base_spy, spy[i1][1]):.6f}"

        fill_spy("day",   "mkt_day_pct")
        fill_spy("week",  "mkt_week_pct")
        fill_spy("month", "mkt_month_pct")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADERS); w.writeheader()
        for row in rows: w.writerow(row)

def load_rows_for_render() -> List[Dict[str,str]]:
    """Return newest-first rows with a computed 'streak' (>=2 shows 2d/3d/...)."""
    _ensure_csv()
    rows = list(csv.DictReader(CSV_PATH.open("r", encoding="utf-8")))
    for r in rows:
        for k in CSV_HEADERS:
            r[k] = r.get(k, "") or ""

    # Compute streaks based on consecutive *appearance days* per symbol.
    # Build date ladder (newest -> oldest) from the dataset (trading days we captured).
    unique_dates_desc = sorted({r["date"] for r in rows}, reverse=True)
    date_idx = {d:i for i,d in enumerate(unique_dates_desc)}
    by_sym = {}
    for r in rows:
        by_sym.setdefault(r["symbol"], set()).add(r["date"])

    for r in rows:
        sym = r["symbol"]
        if not sym or r["date"] == "":
            r["streak"] = ""
            continue
        sdates = by_sym.get(sym, set())
        j = date_idx.get(r["date"], 0)
        streak = 1
        while (j+1) < len(unique_dates_desc) and unique_dates_desc[j+1] in sdates:
            streak += 1
            j += 1
        r["streak"] = str(streak) if streak >= 2 else ""

    rows.sort(key=lambda x: (x["date"], x["name"]), reverse=True)
    return rows
