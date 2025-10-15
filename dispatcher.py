#!/usr/bin/env python3
import os, json, shutil, time, glob, re, sys, copy
import argparse

# === Run-mode argument parsing (CLI only) ===
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--only", type=str, help="Comma list of sections to run, e.g. gapups,portfolios")
_ap.add_argument("--gapups-only", action="store_true")
_ap.add_argument("--portfolios-only", action="store_true")
_ap.add_argument("--all", action="store_true", help="Explicitly run everything (same as no flags)")
_ap.add_argument("--skip-gapups", action="store_true")
_ap.add_argument("--skip-portfolios", action="store_true")
_ARGS, _ = _ap.parse_known_args()

RUN_ONLY = [s.strip().lower() for s in (_ARGS.only.split(",") if _ARGS.only else []) if s.strip()]
RUN_GAPUPS_ONLY = bool(_ARGS.gapups_only)
RUN_PORTFOLIOS_ONLY = bool(_ARGS.portfolios_only)
RUN_ALL = bool(_ARGS.all)
RUN_SKIP_GAPUPS = bool(_ARGS.skip_gapups)
RUN_SKIP_PORTFOLIOS = bool(_ARGS.skip_portfolios)

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape
from dotenv import load_dotenv
from gapups_tracker import record_today_from_finviz, backfill_outcomes, load_rows_for_render, compute_historical_streaks
from portfolios_tracker import fetch_and_update_all as port_fetch, compute_returns_all as port_compute, load_sections_for_render as port_sections

# ------------------------------------------------------------------
# Debug & runtime flags
# ------------------------------------------------------------------
DEBUG = True
DISABLE_SYNTH_HOLD = os.environ.get("DISABLE_SYNTH_HOLD", "1") == "1"
HIST_CONSOLIDATE = os.getenv("CONSOLIDATE_HISTORY", "1") == "1"  # consolidated per-bot file default ON

# Make sure local folder is importable (so stock.py works no matter CWD)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Local stock parser (preferred for candidates)
try:
    import stock as stockmod  # local file stock.py
    if DEBUG:
        print("[init] stock.py imported from:", getattr(stockmod, "__file__", "(unknown)"))
except Exception as e:
    stockmod = None
    if DEBUG:
        print("[init] stock.py not available:", repr(e))

# Optional price source (for PRICES_TODAY in prompt)
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None
    if DEBUG:
        print("[init] yfinance not available; PRICES_TODAY will be empty")

# Optional deps for Finviz fallback
try:
    import requests  # type: ignore
    from bs4 import BeautifulSoup as bs  # type: ignore
except Exception:
    requests = None
    bs = None
    if DEBUG:
        print("[init] requests/bs4 not available; Finviz fallback disabled")

TZ = os.environ.get("LOCAL_TZ", "America/New_York")
TEMPLATES_DIR = "templates"
REPORT_DIR = "report"
PROMPTS_DIR = "prompts"
HISTORY_DIR = "history"
DEFAULT_START_BAL = 10000.0
OVERVIEW_STATE_PATH = os.path.join(HISTORY_DIR, "overview_state.json")
PORTFOLIO_SECTIONS_JSON = os.path.join(HISTORY_DIR, "portfolios_sections.json")

def _load_overview_state() -> dict:
    try:
        if os.path.exists(OVERVIEW_STATE_PATH):
            return json.load(open(OVERVIEW_STATE_PATH, "r", encoding="utf-8"))
    except Exception as e:
        if DEBUG: print("[overview-state] load failed:", repr(e))
    return {"bots": [], "portfolios_sections": [], "gapups_meta": {}, "updated_at": None}

def _save_overview_state(state: dict):
    try:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        json.dump(state, open(OVERVIEW_STATE_PATH, "w", encoding="utf-8"), indent=2)
        if DEBUG: print("[overview-state] saved:", OVERVIEW_STATE_PATH)
    except Exception as e:
        if DEBUG: print("[overview-state] save failed:", repr(e))

def _load_portfolio_sections_from_file() -> list:
    """Load mini-donut section data from history/portfolios_sections.json or scrape docs/report portfolios.html."""
    try:
        if os.path.exists(PORTFOLIO_SECTIONS_JSON):
            with open(PORTFOLIO_SECTIONS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception as e:
        if DEBUG: print("[overview] couldn't read portfolios_sections.json:", repr(e))
    try:
        # Prefer published file, otherwise local run
        docs_pf = os.path.join(BASE_DIR, "docs", "portfolios.html")
        if not os.path.exists(docs_pf):
            docs_pf = os.path.join(REPORT_DIR, "portfolios.html")
        if os.path.exists(docs_pf):
            import re as _re
            html = open(docs_pf, "r", encoding="utf-8").read()
            m = _re.search(r"const\s+SECTIONS\s*=\s*(\[.*?\]);", html, _re.S)
            if m:
                return json.loads(m.group(1))
    except Exception as e:
        if DEBUG: print("[overview] scrape portfolios.html failed:", repr(e))
    return []

FINVIZ_SOURCES: List[Tuple[str,str]] = [
  ("gap_ups", "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o500,sh_relvol_o5,ta_changeopen_u5,ta_highlow20d_nh,ta_perf_d10o,ta_sma200_pa&ft=4&o=-changeopen"),
  ("pivots",  "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o500,sh_relvol_o1.5,ta_changeopen_u5,ta_highlow20d_nh,ta_sma200_pa&ft=4&ta=0&o=-changeopen"),
]
CANDIDATE_LIMIT_PER = int(os.environ.get("CANDIDATE_LIMIT_PER", "40"))
DEFAULT_LEADERS = ["SPY","QQQ","NVDA","AAPL","MSFT","META","AMD","AVGO","TSLA","SMCI"]

SCHEMA_SNIPPET = """
{
  "model_name": "<string>",
  "universe": ["SYM","..."] | "ALL",
  "starting_balance": <number>,
  "portfolio_analysis": "<string>",
  "trades": [
    {
      "time": "<ISO8601Z>",
      "symbol": "<string>",
      "side": "buy"|"sell"|"hold"|"rebalance",
      "qty": <number>,
      "price": <number>,
      "pnl_pct": <number|null>,
      "balance_after": <number>,
      "notes": "<string>"
    }
  ]
}
""".strip()

@dataclass
class Trade:
    time: str
    symbol: str
    side: str
    qty: float
    price: float
    pnl_pct: float | None
    balance_after: float | None
    notes: str = ""

@dataclass
class BotDay:
    model_name: str
    universe: List[str] | str | None
    starting_balance: float | None
    trades: List[Trade]
    portfolio_analysis: str = ""

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip()).strip("-").lower()
    return s or "bot"

# ------------------------------------------------------------------
# Consolidated history helpers (one file per bot) + legacy fallback
# ------------------------------------------------------------------
def _save_history_consolidated(model_slug: str, payload: Dict[str, Any], parsed: BotDay):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    path = os.path.join(HISTORY_DIR, f"{model_slug}.json")
    doc = {"slug": model_slug, "version": 1, "runs": []}
    try:
        if os.path.exists(path):
            doc = json.load(open(path, "r", encoding="utf-8"))
            if not isinstance(doc.get("runs"), list):
                doc["runs"] = []
    except Exception:
        pass

    run = {
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "ending_balance": _ending_balance(parsed),
        "last_trade_time": parsed.trades[0].time if parsed.trades else None,
        "raw": payload.get("response"),
        "prompt": payload.get("prompt"),
        "parsed": asdict(parsed),
    }
    doc["runs"].append(run)
    json.dump(doc, open(path, "w", encoding="utf-8"), indent=2)
    if DEBUG:
        print(f"[history] wrote (consolidated) {path} runs={len(doc['runs'])}")

def _load_yesterday_state_consolidated(model_slug: str) -> Dict[str, Any]:
    path = os.path.join(HISTORY_DIR, f"{model_slug}.json")
    if not os.path.exists(path):
        return {"balance": None, "last_time": None}
    try:
        doc = json.load(open(path, "r", encoding="utf-8"))
        runs = doc.get("runs") or []
        if not runs:
            return {"balance": None, "last_time": None}
        last = runs[-1]
        return {
            "balance": last.get("ending_balance"),
            "last_time": last.get("last_trade_time"),
        }
    except Exception as e:
        if DEBUG:
            print("[state] failed consolidated read:", repr(e))
        return {"balance": None, "last_time": None}

def _load_yesterday_state_legacy(model_slug: str) -> Dict[str, Any]:
    pattern = os.path.join(HISTORY_DIR, f"*__{model_slug}.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return {"balance": None, "last_time": None}
    last = files[-1]
    try:
        data = json.load(open(last,"r",encoding="utf-8"))
        balance = data.get("ending_balance")
        last_time = data.get("last_trade_time")
        return {"balance": balance, "last_time": last_time}
    except Exception as e:
        if DEBUG:
            print("[state] failed to read last history (legacy):", repr(e))
        return {"balance": None, "last_time": None}

def _load_yesterday_state(model_slug: str) -> Dict[str, Any]:
    if HIST_CONSOLIDATE:
        return _load_yesterday_state_consolidated(model_slug)
    return _load_yesterday_state_legacy(model_slug)

def _save_history_legacy(model_slug: str, payload: Dict[str, Any], parsed: BotDay):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "raw": payload,
        "parsed": asdict(parsed),
        "ending_balance": _ending_balance(parsed),
        "last_trade_time": parsed.trades[0].time if parsed.trades else None,  # newest first
        "saved_at": ts,
    }
    os.makedirs(HISTORY_DIR, exist_ok=True)
    path = os.path.join(HISTORY_DIR, f"{ts}__{model_slug}.json")
    json.dump(out, open(path,"w",encoding="utf-8"), indent=2)
    if DEBUG:
        print("[history] wrote (legacy)", path)

def _save_history(model_slug: str, payload: Dict[str, Any], parsed: BotDay):
    if HIST_CONSOLIDATE:
        _save_history_consolidated(model_slug, payload, parsed)
    else:
        _save_history_legacy(model_slug, payload, parsed)

# ------------------------------------------------------------------
# Parse & coerce
# ------------------------------------------------------------------
def _parse_ai_json(txt: str) -> Dict[str, Any]:
    # Find the first JSON object in the response
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in response.")
    return json.loads(m.group(0))

# Anchor identity to prompt file if provided
def _coerce_bot_day(d: Dict[str, Any], carry_balance: float | None, canonical_name: Optional[str] = None) -> BotDay:
    response_name = d.get("model_name") or d.get("name") or "Unnamed Bot"
    name = canonical_name or response_name  # anchor identity to prompt file if provided

    uni = d.get("universe") or d.get("symbols") or []
    starting = d.get("starting_balance")
    if starting is None:
        starting = carry_balance if carry_balance is not None else DEFAULT_START_BAL
    portfolio = d.get("portfolio_analysis") or ""
    trades: List[Trade] = []
    for t in (d.get("trades") or []):
        trades.append(Trade(
            time=str(t.get("time")),
            symbol=str(t.get("symbol")),
            side=str(t.get("side")).lower(),
            qty=float(t.get("qty") or 0),
            price=float(t.get("price") or 0),
            pnl_pct=(float(t.get("pnl_pct")) if t.get("pnl_pct") is not None else None),
            balance_after=(float(t.get("balance_after")) if t.get("balance_after") is not None else None),
            notes=str(t.get("notes") or ""),
        ))
    trades.sort(key=lambda x: x.time, reverse=True)  # newest first
    return BotDay(name, uni, starting, trades, portfolio_analysis=portfolio)

def _ending_balance(b: BotDay) -> float | None:
    for t in b.trades:
        if t.balance_after is not None:
            return t.balance_after
    return b.starting_balance

# ------------------------------------------------------------------
# Reconstruct carried portfolio (cash + FIFO lots) from history
# ------------------------------------------------------------------
def _read_consolidated_runs(model_slug: str) -> List[Dict[str, Any]]:
    path = os.path.join(HISTORY_DIR, f"{model_slug}.json")
    if not os.path.exists(path):
        return []
    try:
        doc = json.load(open(path, "r", encoding="utf-8"))
        return doc.get("runs") or []
    except Exception:
        return []

def _reconstruct_portfolio_state(model_slug: str, prices_today: Dict[str, float]) -> Dict[str, Any]:
    """Return {'cash': float, 'lots': {sym: [{'qty':q,'price':p,'time':t},...]}, 'last_px': {sym:px}}."""
    runs = _read_consolidated_runs(model_slug)
    # Find earliest starting balance
    start_cash = None
    all_trades: List[Dict[str, Any]] = []
    for run in runs:
        parsed = run.get("parsed") or {}
        if start_cash is None and parsed.get("starting_balance") is not None:
            try:
                start_cash = float(parsed["starting_balance"])
            except Exception:
                pass
        for td in (parsed.get("trades") or []):
            all_trades.append(td)

    if start_cash is None:
        start_cash = DEFAULT_START_BAL

    chron = sorted(all_trades, key=lambda x: str(x.get("time") or ""))

    cash = float(start_cash)
    lots: Dict[str, List[Dict[str, float]]] = {}
    last_px: Dict[str, float] = {}

    for t in chron:
        side = str(t.get("side") or "").lower()
        sym  = str(t.get("symbol") or "")
        qty  = float(t.get("qty") or 0)
        px   = float(t.get("price") or 0)
        if sym and px > 0:
            last_px[sym] = px

        if side in ("buy","rebalance"):
            if qty > 0 and px > 0:
                cash -= qty * px
                lots.setdefault(sym, []).append({"qty": qty, "price": px, "time": str(t.get("time") or "")})
        elif side == "sell":
            if qty > 0 and px > 0:
                # FIFO reduce
                held = lots.get(sym, [])
                remaining = qty
                while remaining > 0 and held:
                    take = min(remaining, held[0]["qty"])
                    held[0]["qty"] -= take
                    if held[0]["qty"] <= 1e-12:
                        held.pop(0)
                    remaining -= take
                cash += qty * px
                lots[sym] = [l for l in lots.get(sym, []) if l["qty"] > 1e-12]

    # Drop any empty symbols
    lots = {s:[l for l in L if l["qty"]>1e-12] for s,L in lots.items() if any(l["qty"]>1e-12 for l in L)}

    # Keep last_px fallback from history if no price today
    for sym in list(lots.keys()):
        if sym not in prices_today and sym not in last_px:
            last_px[sym] = 0.0

    if DEBUG:
        tot_qty = {s: sum(l["qty"] for l in L) for s,L in lots.items()}
        if lots:
            print("[carry] lots:", {k: round(v,6) for k,v in tot_qty.items()}, "| cash:", round(cash,2))
        else:
            print("[carry] no open lots | cash:", round(cash,2))

    return {"cash": cash, "lots": lots, "last_px": last_px}

# ---------- unified FIFO stats ----------
def _compute_fifo_stats(trades: List[Trade]) -> Dict[str, float]:
    chron = sorted(trades, key=lambda x: x.time or "")
    lots: Dict[str, List[Dict[str, Any]]] = {}
    closed = []

    for t in chron:
        side = (t.side or "").lower()
        sym = t.symbol
        qty = int(t.qty or 0)
        price = float(t.price or 0.0)
        if not sym or qty <= 0:
            continue

        if side in ("buy", "rebalance") and price > 0:
            lots.setdefault(sym, []).append({"qty": qty, "price": price, "time": t.time})
        elif side == "sell":
            remaining = qty
            cost = 0.0
            proceeds = price * qty
            first_buy_time = None
            while remaining > 0 and lots.get(sym):
                lot = lots[sym][0]
                take = min(remaining, lot["qty"])
                cost += take * lot["price"]
                if first_buy_time is None:
                    first_buy_time = lot.get("time")
                lot["qty"] -= take
                remaining -= take
                if lot["qty"] <= 0:
                    lots[sym].pop(0)
            pnl_pct = None
            if cost > 0:
                pnl_pct = (proceeds - cost) / cost
            elif t.pnl_pct is not None:
                pnl_pct = t.pnl_pct
            closed.append({
                "buy_time": first_buy_time,
                "sell_time": t.time,
                "symbol": sym,
                "qty": qty,
                "price": price,
                "balance_after": t.balance_after,
                "notes": t.notes,
                "pnl_pct": pnl_pct,
            })

    wins   = [c["pnl_pct"] for c in closed if c["pnl_pct"] is not None and c["pnl_pct"] > 0]
    losses = [c["pnl_pct"] for c in closed if c["pnl_pct"] is not None and c["pnl_pct"] < 0]
    total  = len(wins) + len(losses)
    win_rate_num = (len(wins) / total) if total else 0.0
    avg_win_num  = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss_num = (sum(losses) / len(losses)) if losses else 0.0
    return {
        "win_rate_num": win_rate_num,
        "avg_win_num": avg_win_num,
        "avg_loss_num": avg_loss_num,
        "trades_count": len(trades),
        "closed_sells": closed,
    }

# ---------- validator WITH carried state (no mutation of original) ----------
def _validate_trades_budget_with_state(
    trades: List[Trade],
    init_cash: float,
    init_lots: Dict[str, List[Dict[str, float]]],
    prices_today: Dict[str,float]
) -> List[str]:
    issues: List[str] = []
    EPS_BAL = 0.015  # 1.5% tolerance for balance_after checks

    chron = sorted(trades, key=lambda x: x.time or "")
    cash = float(init_cash or 0.0)
    lots: Dict[str, List[Dict[str, float]]] = copy.deepcopy(init_lots)  # deep copy; don't mutate input
    last_px: Dict[str, float] = {}

    def port_value() -> float:
        total = cash
        for sym, ls in lots.items():
            px = prices_today.get(sym.upper(), last_px.get(sym, 0.0))
            if px > 0:
                qty = sum(l["qty"] for l in ls)
                total += qty * px
        return total

    for t in chron:
        side = (t.side or "").lower()
        sym  = t.symbol
        qty  = float(t.qty or 0)   # allow fractional
        px   = float(t.price or 0.0)
        if sym and px > 0:
            last_px[sym] = px

        if side in ("buy","rebalance"):
            if px <= 0 or qty < 0:
                issues.append(f"{t.time} {sym} BUY invalid price/qty (price={px}, qty={qty}).")
            else:
                cost = qty * px
                if cost > cash + 1e-6:
                    issues.append(f"{t.time} {sym} BUY cost ${cost:,.2f} exceeds available cash ${cash:,.2f}.")
                else:
                    lots.setdefault(sym, []).append({"qty": qty, "price": px})
                    cash -= cost

        elif side == "sell":
            if px <= 0 or qty < 0:
                issues.append(f"{t.time} {sym} SELL invalid price/qty (price={px}, qty={qty}).")
            else:
                held = sum(l["qty"] for l in lots.get(sym, []))
                if qty > held + 1e-9:
                    issues.append(f"{t.time} {sym} SELL qty {qty} exceeds held {held}.")
                else:
                    remaining = qty
                    fifo_cost = 0.0
                    for lot in list(lots.get(sym, [])):
                        if remaining <= 1e-12: break
                        take = min(remaining, lot["qty"])
                        fifo_cost += take * lot["price"]
                        lot["qty"] -= take
                        remaining -= take
                        if lot["qty"] <= 1e-12:
                            lots[sym].pop(0)
                    cash += qty * px
                    if t.pnl_pct is not None and fifo_cost > 0:
                        exp = (qty*px - fifo_cost)/fifo_cost
                        if abs((t.pnl_pct or 0) - exp) > 0.005:
                            issues.append(f"{t.time} {sym} SELL pnl_pct {t.pnl_pct:.4f} != FIFO {exp:.4f} (tolerance 0.005).")

        # balance_after required & sanity check
        if t.balance_after is None:
            issues.append(f"{t.time} {sym or ''} missing balance_after.")
        else:
            est = port_value()
            if est > 0:
                ba = float(t.balance_after or 0.0)
                if abs(ba - est) / est > EPS_BAL:
                    issues.append(f"{t.time} balance_after ${ba:,.2f} != estimated ${est:,.2f} (>{int(EPS_BAL*100)}% diff).")

    return issues

# Backwards-compatible wrapper (no prior state)
def _validate_trades_budget(trades: List[Trade], starting_balance: float, prices_today: Dict[str,float]) -> List[str]:
    return _validate_trades_budget_with_state(trades, starting_balance, {}, prices_today)

def _build_correction_prompt(original_full_prompt: str, previous_json_text: str, issues: List[str], carry_note: str = "") -> str:
    lines = []
    lines.append(original_full_prompt)
    if carry_note:
        lines.append("\n--- CURRENT_PORTFOLIO_STATE (must honor) ---\n")
        lines.append(carry_note)
    lines.append("\n--- PREVIOUS JSON (for reference; fix it) ---\n")
    lines.append(previous_json_text.strip())
    lines.append("\n--- VALIDATION ERRORS (you must fix ALL) ---")
    for it in issues[:20]:  # cap to avoid ballooning
        lines.append(f"- {it}")
    lines.append("\nRe-output STRICT JSON ONLY (no commentary, no code fences), obeying all budget/share rules, NEVER exceeding available cash, NEVER selling more than held, and including 'balance_after' after EVERY action.")
    return "\n".join(lines)

# ------------------------------------------------------------------
# Candidates (stock.py preferred; Finviz + leaders fallback)
# ------------------------------------------------------------------
def _get_candidates_from_stock_py() -> Optional[Dict[str, List[str]]]:
    if stockmod is None:
        if DEBUG: print("[stock.py] not available")
        return None
    buckets: Dict[str, List[str]] = {}
    # gap_ups
    try:
        gaps = stockmod.Get_Stock_List(stockmod.gap_ups, "a", "tab-link") or []
        if gaps: buckets["gap_ups"] = gaps
        if DEBUG: print(f"[stock.py] gap_ups -> {len(gaps)} tickers: {', '.join(gaps[:10])}")
    except Exception as e:
        if DEBUG: print("[stock.py] gap_ups error:", repr(e))
    # pivots
    try:
        pivs = stockmod.Get_Stock_List(stockmod.pivots, "a", "tab-link") or []
        if pivs: buckets["pivots"] = pivs
        if DEBUG: print(f"[stock.py] pivots  -> {len(pivs)} tickers: {', '.join(pivs[:10])}")
    except Exception as e:
        if DEBUG: print("[stock.py] pivots error:", repr(e))
    if not buckets and DEBUG:
        print("[stock.py] no buckets returned")
    return buckets or None

def _fetch_finviz_tickers(url: str) -> List[str]:
    if requests is None or bs is None:
        if DEBUG: print("[finviz] scraping disabled (no deps)")
        return []
    try:
        if DEBUG: print("[finviz] GET", url)
        r = requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, timeout=20)
        if DEBUG: print("[finviz] status", r.status_code, "bytes", len(r.content))
        r.raise_for_status()
        soup = bs(r.content, 'lxml')
        results = soup.find_all("a", class_="tab-link")
        tickers: List[str] = []
        for op in results:
            href = op.get("href","")
            if "quote" in href:
                t = (op.get_text() or "").strip().upper()
                if t and t.isalnum():
                    tickers.append(t)
        seen = set(); out: List[str] = []
        for t in tickers:
            if t not in seen:
                seen.add(t); out.append(t)
        if DEBUG: print(f"[finviz] parsed {len(out)} tickers: {', '.join(out[:10])}")
        return out
    except Exception as e:
        if DEBUG: print("[finviz] error:", repr(e))
        return []

def _scan_candidates() -> Dict[str, List[str]]:
    buckets = _get_candidates_from_stock_py()
    if buckets:
        if DEBUG:
            print("[scan] using stock.py:", {k: len(v) for k, v in buckets.items()})
        return {k: v[:CANDIDATE_LIMIT_PER] for k, v in buckets.items()}

    if DEBUG: print("[scan] stock.py empty; trying finviz...")
    buckets2: Dict[str, List[str]] = {}
    for name, url in FINVIZ_SOURCES:
        arr = _fetch_finviz_tickers(url)
        if arr:
            buckets2[name] = arr[:CANDIDATE_LIMIT_PER]
    if buckets2:
        if DEBUG:
            print("[scan] finviz:", {k: len(v) for k, v in buckets2.items()})
        return buckets2

    if DEBUG: print("[scan] falling back to DEFAULT_LEADERS")
    return {"leaders": DEFAULT_LEADERS[:]}

def _format_candidates_for_prompt(buckets: Dict[str, List[str]]) -> str:
    lines: List[str] = []
    total = 0
    for name, arr in buckets.items():
        total += len(arr)
        lines.append(f"- {name}: {', '.join(arr)}")
    lines.append(f"(total candidates: {total})")
    return "\n".join(lines)

# ------------------------------------------------------------------
# Prices
# ------------------------------------------------------------------
def _fetch_prices(tickers: List[str]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    if yf is None or not tickers:
        if DEBUG: print("[prices] yfinance unavailable or empty list")
        return prices
    for t in tickers[:200]:
        try:
            df = yf.Ticker(t).history(period="1d", auto_adjust=False)
            if not df.empty:
                val = float(df["Close"].iloc[-1])
                if val > 0:
                    prices[t.upper()] = val
        except Exception:
            continue
    if DEBUG:
        sample = ", ".join(f"{k}={v:.2f}" for k, v in list(prices.items())[:10])
        print(f"[prices] fetched {len(prices)} prices; sample: {sample}")
    return prices

def _format_prices_for_prompt(prices: Dict[str, float]) -> str:
    if not prices:
        return "(none)"
    return ", ".join(f"{k}={v:.2f}" for k, v in sorted(prices.items()))

# ------------------------------------------------------------------
# Aggregation (consolidated-aware)
# ------------------------------------------------------------------
def _history_files_for(model_slug: str):
    pattern = os.path.join(HISTORY_DIR, f"*__{model_slug}.json")
    return sorted(glob.glob(pattern))

def _trade_key(td: Dict[str, Any]) -> tuple:
    return (
        str(td.get("time")),
        str(td.get("symbol")),
        str(td.get("side")).lower(),
        f'{float(td.get("qty", 0)):.8f}',
        f'{float(td.get("price", 0)):.8f}',
    )

def _aggregate_history(model_slug: str, latest_bot: BotDay) -> BotDay:
    seen = set()
    all_trades: List[Trade] = []
    earliest_start: float | None = None
    universe_latest = latest_bot.universe
    portfolio_latest = latest_bot.portfolio_analysis

    consolidated = os.path.join(HISTORY_DIR, f"{model_slug}.json")
    if os.path.exists(consolidated):
        try:
            doc = json.load(open(consolidated, "r", encoding="utf-8"))
            runs = doc.get("runs") or []
            for run in runs:
                parsed = run.get("parsed") or {}
                sb = parsed.get("starting_balance")
                if sb is not None and earliest_start is None:
                    try: earliest_start = float(sb)
                    except: pass

                uni = parsed.get("universe")
                if uni: universe_latest = uni

                pa = parsed.get("portfolio_analysis") or ""
                if pa: portfolio_latest = pa

                for td in (parsed.get("trades") or []):
                    k = _trade_key(td)
                    if k in seen:
                        continue
                    seen.add(k)
                    try:
                        all_trades.append(Trade(
                            time=str(td.get("time")),
                            symbol=str(td.get("symbol")),
                            side=str(td.get("side")).lower(),
                            qty=float(td.get("qty") or 0),
                            price=float(td.get("price") or 0),
                            pnl_pct=(float(td.get("pnl_pct")) if td.get("pnl_pct") is not None else None),
                            balance_after=(float(td.get("balance_after")) if td.get("balance_after") is not None else None),
                            notes=str(td.get("notes") or ""),
                        ))
                    except Exception:
                        continue
        except Exception as e:
            if DEBUG: print("[agg] consolidated read failed:", repr(e))

    else:
        # legacy timestamped history
        for path in _history_files_for(model_slug):
            try:
                blob = json.load(open(path, "r", encoding="utf-8"))
            except Exception:
                continue

            parsed = blob.get("parsed") or {}
            sb = parsed.get("starting_balance")
            if sb is not None and earliest_start is None:
                try:
                    earliest_start = float(sb)
                except Exception:
                    pass

            uni = parsed.get("universe")
            if uni:
                universe_latest = uni

            pa = parsed.get("portfolio_analysis") or ""
            if pa:
                portfolio_latest = pa

            for td in (parsed.get("trades") or []):
                k = _trade_key(td)
                if k in seen:
                    continue
                seen.add(k)
                try:
                    all_trades.append(Trade(
                        time=str(td.get("time")),
                        symbol=str(td.get("symbol")),
                        side=str(td.get("side")).lower(),
                        qty=float(td.get("qty") or 0),
                        price=float(td.get("price") or 0),
                        pnl_pct=(float(td.get("pnl_pct")) if td.get("pnl_pct") is not None else None),
                        balance_after=(float(td.get("balance_after") if td.get("balance_after") is not None else 0)),
                        notes=str(td.get("notes") or ""),
                    ))
                except Exception:
                    continue

    all_trades.sort(key=lambda x: x.time, reverse=True)
    start_balance = earliest_start if earliest_start is not None else latest_bot.starting_balance

    return BotDay(
        model_name=latest_bot.model_name,
        universe=universe_latest if universe_latest is not None else latest_bot.universe,
        starting_balance=start_balance,
        trades=all_trades,
        portfolio_analysis=portfolio_latest
    )

def _slim_sections(sections: list) -> list:
    """Return a compact [{name, holdings:[{ticker, weight}]}] for overview donuts."""
    slim = []
    for s in (sections or []):
        name = (s.get('name') or s.get('slug') or 'Portfolio')
        hs = []
        for h in (s.get('holdings') or []):
            try:
                hs.append({'ticker': h.get('ticker'), 'weight': float(h.get('weight') or 0.0)})
            except Exception:
                hs.append({'ticker': h.get('ticker'), 'weight': 0.0})
        slim.append({'name': name, 'holdings': hs})
    return slim
# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------

# def _render_overview(env, bots_rows: List[Dict[str, Any]]):
#     # Load prior state so we can preserve sections not updated in this run
#     state = _load_overview_state()

#     # Determine which sections are being updated in this run
#     update_bots = True  # if caller passed bots_rows, we merge it; otherwise keep prior
#     update_ports = (RUN_ALL or RUN_PORTFOLIOS_ONLY or ("portfolios" in RUN_ONLY if RUN_ONLY else False)) and not RUN_SKIP_PORTFOLIOS
#     update_gapups = (RUN_ALL or RUN_GAPUPS_ONLY or ("gapups" in RUN_ONLY if RUN_ONLY else False)) and not RUN_SKIP_GAPUPS

#     # ---- BOT SECTION ----
#     if bots_rows:
#         state["bots"] = bots_rows  # replace with latest computed
#     # else keep existing bots from prior state

#     # ---- PORTFOLIOS MINI DONUTS ----
#     # Build compact sections json (reuse portfolios data) *only* if updating; else keep prior
#     if update_ports:
#         try:
#             _secs = port_sections()
#         except Exception:
#             _secs = []
#         _slim = []
#         for _s in (_secs or []):
#             _name = (_s.get('name') or _s.get('slug') or 'Portfolio')
#             _hs = []
#             for _h in (_s.get('holdings') or []):
#                 try:
#                     _hs.append({'ticker': _h.get('ticker'), 'weight': float(_h.get('weight') or 0.0)})
#                 except Exception:
#                     _hs.append({'ticker': _h.get('ticker'), 'weight': 0.0})
#             _slim.append({'name': _name, 'holdings': _hs})
#         state["portfolios_sections"] = _slim

#     # ---- (Optional) GAPUPS summary placeholder ----
#     # If your overview.html uses gap-ups metadata, compute and set here.
#     # For now we just stamp last update time so the section doesn't appear stale.
#     if update_gapups:
#         state["gapups_meta"] = {"updated": time.strftime("%m/%d/%y %H:%M")}

#     # Persist merged state for the next partial run
#     state["updated_at"] = time.strftime("%m/%d/%y %H:%M")
#     _save_overview_state(state)

#     # ----- Render using merged state -----
#     tpl = env.get_template("overview.html")
#     gen_time = time.strftime("%m/%d/%y %H:%M")
#     bots_eff = state.get("bots") or []
#     total_trades = sum(r.get("total_actions", r.get("trades", 0)) for r in bots_eff)
#     avg_win_rate = sum((r.get("win_rate_num") or 0.0) for r in bots_eff)/len(bots_eff) if bots_eff else 0.0

#     sections_json = json.dumps(state.get("portfolios_sections") or [], separators=(',',':'))

#     html = tpl.render(
#         gen_time=gen_time,
#         tz=TZ,
#         bots=bots_eff,
#         total_trades=total_trades,
#         avg_win_rate=f"{avg_win_rate*100:.2f}%",
#         sections_json=sections_json
#     )
#     open(os.path.join(REPORT_DIR,"overview.html"),"w",encoding="utf-8").write(html)
def _render_overview(env, bots_rows: List[Dict[str, Any]], sections_override: list | None = None):
    # Load last state so partial jobs don't wipe other sections
    state = _load_overview_state() or {}
    state_bots = state.get("bots") or []
    state_sections = state.get("portfolios_sections") or []

    # Replace bots if provided
    if bots_rows:
        state_bots = bots_rows

    # Replace sections if override provided (e.g., portfolios-only run)
    if sections_override is not None:
        state_sections = _slim_sections(sections_override)

    # If still empty, try to scrape the latest portfolios.html (docs/ or report/)
    if not state_sections:
        state_sections = _load_portfolio_sections_from_file()

    # Save merged state for future runs
    out = {
        "bots": state_bots,
        "portfolios_sections": state_sections,
        "updated_at": time.strftime("%m/%d/%y %H:%M"),
    }
    _save_overview_state(out)

    # Render overview from merged state
    tpl = env.get_template("overview.html")
    gen_time = time.strftime("%m/%d/%y %H:%M")
    total_trades = sum((r.get("trades") or 0) for r in state_bots)
    avg_win_rate = (
        sum((r.get("win_rate_num") or 0.0) for r in state_bots)/len(state_bots) if state_bots else 0.0
    )
    html = tpl.render(
        gen_time=gen_time,
        tz=TZ,
        bots=state_bots,
        total_trades=total_trades,
        avg_win_rate=f"{avg_win_rate*100:.2f}%",
        sections_json=json.dumps(state_sections, separators=(",",":"))
    )
    open(os.path.join(REPORT_DIR,"overview.html"),"w",encoding="utf-8").write(html)


# Accept file_slug for stable filenames
def _render_model(env, b: BotDay, stats: Dict[str, Any], prompt_text: str = "", response_text: str = "", file_slug: Optional[str] = None):
    tpl = env.get_template("model_report.html")
    series = {"t":[], "v":[]}
    chron = list(sorted(b.trades, key=lambda x: x.time))
    carry = b.starting_balance or 0.0
    if not chron:
        # still draw something
        series["t"].append(datetime.now(timezone.utc).isoformat())
        series["v"].append(float(carry))
    else:
        for t in chron:
            if t.balance_after is not None:
                series["t"].append(t.time)
                series["v"].append(float(t.balance_after))
                carry = t.balance_after

    bal_text = f"${_ending_balance(b):,.2f}" if _ending_balance(b) is not None else "—"

    html = tpl.render(
        model_name=b.model_name,
        gen_time=time.strftime("%m/%d/%y %H:%M"),
        tz=TZ,
        balance_text=bal_text,
        win_rate_num=stats.get("win_rate_num", 0.0),
        avg_win_num=stats.get("avg_win_num", 0.0),
        avg_loss_num=stats.get("avg_loss_num", 0.0),
        symbols=(", ".join(b.universe) if isinstance(b.universe, list) and b.universe else (b.universe if isinstance(b.universe, str) else "—")),
        trades=[{
            "time": t.time,
            "symbol": t.symbol,
            "side": t.side,
            "qty": f"{t.qty:.6f}".rstrip("0").rstrip("."),   # more precision for fractional shares
            "qty_num": float(t.qty),
            "price": f"{t.price:.6f}",
            "price_num": float(t.price),
            "pnl_pct": (f"{(t.pnl_pct*100):.2f}%" if t.pnl_pct is not None else "—"),
            "pnl_pct_num": (t.pnl_pct if t.pnl_pct is not None else 0.0),
            "balance_after": (f"${t.balance_after:,.2f}" if t.balance_after is not None else "—"),
            "balance_after_num": (float(t.balance_after) if t.balance_after is not None else None),
            "balance_dir": 1 if (t.pnl_pct or 0) >= 0 else -1,
            "notes": t.notes,
        } for t in b.trades],
        balance_series_json=json.dumps(series, separators=(",",":")),
        portfolio_analysis=b.portfolio_analysis,
        prompt_text=prompt_text,
        response_text=response_text,
        start_balance=(b.starting_balance or 0.0),
    )

    fname = f"{(file_slug or _slug(b.model_name))}.html"
    open(os.path.join(REPORT_DIR, fname),"w",encoding="utf-8").write(html)
    return fname


# ------------------------------------------------------------------
# Gap-Ups page renderer
# ------------------------------------------------------------------
def _render_gapups(env):
    try:
        rows = load_rows_for_render()
    except Exception as e:
        if DEBUG: print("[gapups] load_rows_for_render failed:", repr(e))
        rows = []

    # NEW: compute historical streaks
    try:
        streaks = compute_historical_streaks(min_len=2)
    except Exception as e:
        if DEBUG: print("[gapups] compute_historical_streaks failed:", repr(e))
        streaks = []

    try:
        tpl = env.get_template("gapups.html")
    except Exception as e:
        if DEBUG: print("[gapups] missing template:", repr(e))
        return None

    html = tpl.render(
        gen_time=time.strftime("%m/%d/%y %H:%M"),
        rows_json=json.dumps(rows, separators=(",",":")),
        rows=rows,
        streaks_json=json.dumps(streaks, separators=(",",":")),
        streaks=streaks
    )
    out = os.path.join(REPORT_DIR, "gapups.html")
    open(out, "w", encoding="utf-8").write(html)
    if DEBUG: print("[gapups] rendered ->", out)
    return "gapups.html"



# ------------------------------------------------------------------
# Public Portfolios page renderer
# ------------------------------------------------------------------
def _render_portfolios(env):
    try:
        port_fetch(max_pages=3)
        port_compute()
    except Exception as e:
        if DEBUG: print("[portfolios] fetch/compute error:", repr(e))
    try:
        tpl = env.get_template("portfolios.html")
    except Exception as e:
        if DEBUG: print("[portfolios] missing template:", repr(e))
        return None
    sections = port_sections()
    html = tpl.render(gen_time=time.strftime("%m/%d/%y %H:%M"), sections=sections, sections_json=json.dumps(sections, separators=(",",":")))
    out = os.path.join(REPORT_DIR, "portfolios.html")
    open(out, "w", encoding="utf-8").write(html)
    if DEBUG: print("[portfolios] rendered ->", out)
    return "portfolios.html"



def _run_gapups_section(env, buckets):
    try:
        gap_up_tickers = []
        if isinstance(buckets, dict):
            gap_up_tickers = buckets.get("gap_ups", []) or []
        if gap_up_tickers:
            if DEBUG: print(f"[gapups] recording {len(gap_up_tickers)} Finviz tickers")
            record_today_from_finviz(gap_up_tickers)
        backfill_outcomes()
    except Exception as e:
        if DEBUG: print("[gapups] record/backfill failed:", repr(e))
    _render_gapups(env)

# ------------------------------------------------------------------
# Prompts & querying
# ------------------------------------------------------------------
def _load_prompts() -> Dict[str, str]:
    prompts = {}
    for p in sorted(glob.glob(os.path.join(PROMPTS_DIR, "*.txt"))):
        name = os.path.splitext(os.path.basename(p))[0]
        prompts[name] = open(p,"r",encoding="utf-8").read()
    if DEBUG:
        print("[prompts] found:", list(prompts.keys()))
    return prompts

def _format_carry_for_prompt(carry_state: Dict[str, Any], prices_today: Dict[str,float]) -> str:
    lots = carry_state.get("lots") or {}
    cash = float(carry_state.get("cash") or 0.0)
    # Collapse lots per symbol for shorter context (sum qty; show a ref price)
    lines = []
    lines.append(f"AVAILABLE_CASH_USD: {cash:.2f}")
    if lots:
        lines.append("OPEN_POSITIONS:")
        for sym, L in lots.items():
            qty = sum(l["qty"] for l in L)
            ref_px = prices_today.get(sym.upper(), (L[-1]["price"] if L else 0.0))
            first_time = L[0].get("time") if L else None
            lines.append(f"- {sym}: qty={qty:.6f}, ref_price={ref_px:.4f}, first_buy_time={first_time}")
    else:
        lines.append("OPEN_POSITIONS: []")
    return "\n".join(lines)

def _build_prompt(base_prompt: str, yesterday: Dict[str,Any], candidates_note: str, prices_note: str, carry_note: str) -> str:
    carry = yesterday.get("balance")
    last = yesterday.get("last_time")
    start_balance_hint = carry if carry is not None else DEFAULT_START_BAL

    parts: List[str] = []
    parts.append("You are an autonomous trading agent. Reply with STRICT JSON ONLY (no commentary, no code fences) following the schema documented below.")
    parts.append(f"- Starting balance for a new/first run: {start_balance_hint}.")
    parts.append(f"- Include only NEW actions since {last or 'yesterday'} but ensure the final 'balance_after' reflects current balance.")
    parts.append("- Today you MUST honor CURRENT_PORTFOLIO_STATE (see below). You may SELL or HOLD existing positions, and you may BUY new ones only if AVAILABLE_CASH_USD is sufficient. NEVER spend more than AVAILABLE_CASH_USD and NEVER sell more than currently held (FIFO).")
    parts.append("- You MUST output at least one action for the current day (first run included). Prefer buys from CANDIDATES_TODAY. If truly nothing qualifies, emit a single HOLD with 'balance_after'.")
    parts.append("- Allocate ~100% of capital across 1–4 positions using PRICES_TODAY (USD). Buys may be fractional shares (qty can be decimal, ≥ 4 places). For each buy, you may decide a dollar allocation and compute qty = allocation_dollars / price.")
    parts.append("- For EVERY action you output (buy/sell/hold/rebalance), include 'balance_after' = total portfolio value immediately after that action (cash + positions), valued using PRICES_TODAY.")
    parts.append("- For SELL actions, compute 'pnl_pct' from FIFO cost of prior BUYS when possible; keep it numeric (e.g., 0.0123 for +1.23), or null if not computable.")
    parts.append("- Keep numbers raw (no % signs in 'pnl_pct').")
    parts.append("- End-of-day you may hold 1–4 positions or cash if no setups qualify. Always output at least one action with 'balance_after'.")
    parts.append("PRICES_TODAY (USD):")
    parts.append(prices_note)
    parts.append("CANDIDATES_TODAY:")
    parts.append(candidates_note)
    parts.append("CURRENT_PORTFOLIO_STATE:")
    parts.append(carry_note)
    parts.append("Schema:")
    parts.append(SCHEMA_SNIPPET)
    parts.append("=== STRATEGY CONTEXT ===")
    parts.append(base_prompt)

    parts.append(
        'Example HOLD:\n'
        '{\n'
        '  "model_name": "example",\n'
        '  "universe": "ALL",\n'
        '  "starting_balance": 10000.0,\n'
        '  "portfolio_analysis": "No new setups; preserving capital.",\n'
        '  "trades": [\n'
        '    {"time":"2025-10-07T14:30:00Z","symbol":"CASH","side":"hold","qty":0,"price":0,"pnl_pct":null,"balance_after":10000.0,"notes":"No change today; waiting for setups."}\n'
        '  ]\n'
        '}\n'
    )
    return "\n\n".join(parts)

def query_model(model_name: str, prompt_text: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-5")

    def dev_sample():
        return json.dumps({
            "model_name": model_name,
            "universe": "ALL",
            "starting_balance": DEFAULT_START_BAL,
            "portfolio_analysis": "Dev mode sample (no API key).",
            "trades": []
        })

    # No key: use dev mode (or local sample if present)
    if not api_key or (isinstance(api_key, str) and api_key.strip() == ""):
        sample_path = os.path.join(PROMPTS_DIR, f"{model_name}.json")
        if os.path.exists(sample_path):
            if DEBUG: print("[openai] dev mode -> using local sample", sample_path)
            return open(sample_path, "r", encoding="utf-8").read()
        if DEBUG: print("[openai] dev mode -> returning empty sample")
        return dev_sample()

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # First try: Responses API without temperature (some models reject it)
        if DEBUG: print(f"[openai] responses.create model={model} len={len(prompt_text)}")
        resp = client.responses.create(model=model, input=prompt_text)  # no temperature
        text = getattr(resp, "output_text", None)
        if not text:
            # Build text from content parts if output_text missing
            try:
                parts = []
                for item in resp.output[0].content:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                text = "".join(parts)
            except Exception:
                text = ""
        if DEBUG: print("[openai] responses.create ok -> chars:", len(text))
        return text

    except Exception as e:
        if DEBUG: print("[openai] responses.create error:", repr(e))

        # Fallback: try Chat Completions API
        try:
            if DEBUG: print(f"[openai] chat.completions.create model={model}")
            chat = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                # don't set temperature; some models disallow
            )
            text = chat.choices[0].message.content or ""
            if DEBUG: print("[openai] chat.completions ok -> chars:", len(text))
            return text
        except Exception as e2:
            if DEBUG: print("[openai] chat.completions error:", repr(e2))

        # Local sample fallback
        sample_path = os.path.join(PROMPTS_DIR, f"{model_name}.json")
        if os.path.exists(sample_path):
            if DEBUG: print("[openai] fallback -> using local sample", sample_path)
            return open(sample_path,"r",encoding="utf-8").read()
        if DEBUG: print("[openai] fallback -> dev sample")
        return dev_sample()

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(RUN_ONLY, RUN_GAPUPS_ONLY, RUN_PORTFOLIOS_ONLY, RUN_ALL, RUN_SKIP_GAPUPS, RUN_SKIP_PORTFOLIOS):
    load_dotenv()

    # Clean report dir
    if os.path.exists(REPORT_DIR):
        for name in os.listdir(REPORT_DIR):
            p = os.path.join(REPORT_DIR, name)
            if os.path.isfile(p) or os.path.islink(p):
                os.unlink(p)
            else:
                shutil.rmtree(p)
    else:
        os.makedirs(REPORT_DIR, exist_ok=True)

    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=select_autoescape(["html"]))

    prompts = _load_prompts()
    bots_rows = []

    # Scan candidates & prices once per run (toggle with USE_SCAN, default ON)
    USE_SCAN = os.getenv("USE_SCAN", "1") == "1"
    buckets = _scan_candidates() if USE_SCAN else {}
    candidates_note = _format_candidates_for_prompt(buckets)
    uniq: List[str] = sorted({t.upper() for arr in buckets.values() for t in arr})
    prices = _fetch_prices(uniq) if uniq else {}
    prices_note = _format_prices_for_prompt(prices)
    if DEBUG:
        print("[scan] CANDIDATE buckets:", {k: len(v) for k, v in buckets.items()})
        print("[prices] count:", len(prices))

    # ---------- CLI gating ----------
    def _maybe_call(name, *a, **kw):
        fn = globals().get(name)
        return fn(*a, **kw) if callable(fn) else None

    if RUN_ONLY:
        if ("gapups" in RUN_ONLY) and (not RUN_SKIP_GAPUPS):
            _maybe_call("_run_gapups_section", env, buckets) or (_render_gapups(env))
        if ("portfolios" in RUN_ONLY) and (not RUN_SKIP_PORTFOLIOS):
            _maybe_call("_render_portfolios", env)
            try:
                _secs = port_sections()
            except Exception:
                _secs = []
        _render_overview(env, bots_rows=[], sections_override=_secs)
        return

    if RUN_GAPUPS_ONLY and not RUN_SKIP_GAPUPS:
        _maybe_call("_run_gapups_section", env, buckets) or (_render_gapups(env))
        _render_overview(env, bots_rows=[])
        return

    if RUN_PORTFOLIOS_ONLY and not RUN_SKIP_PORTFOLIOS:
        _maybe_call("_render_portfolios", env)
        try:
            _secs = port_sections()
        except Exception:
            _secs = []
        _render_overview(env, bots_rows=[], sections_override=_secs)
        try:
            with open(PORTFOLIO_SECTIONS_JSON, "w", encoding="utf-8") as f:
                json.dump(_slim_sections(_secs), f, indent=2)
        except Exception as e:
            if DEBUG: print("[overview] persist sections failed:", repr(e))
        return

    # RUN_ALL simply falls through into full pipeline below

    # === Gap-Ups & Public Portfolios pages ===
    try:
        gap_up_tickers = []
        if isinstance(buckets, dict):
            gap_up_tickers = buckets.get("gap_ups", []) or []
        if gap_up_tickers:
            if DEBUG: print(f"[gapups] recording {len(gap_up_tickers)} Finviz tickers")
            record_today_from_finviz(gap_up_tickers)
        backfill_outcomes()
    except Exception as e:
        if DEBUG: print("[gapups] record/backfill failed:", repr(e))
    _render_gapups(env)

    # Portfolios page
    _render_portfolios(env)

    for fname, base_prompt in prompts.items():
        slug = _slug(fname)

        # Carry-forward state from history (cash + lots) BEFORE building the prompt
        carry_state = _reconstruct_portfolio_state(slug, prices)
        carry_note = _format_carry_for_prompt(carry_state, prices)

        yest = _load_yesterday_state(slug)
        full_prompt = _build_prompt(base_prompt, yest, candidates_note, prices_note, carry_note)

        if DEBUG:
            print(f"\n[bot] {fname} | slug={slug} | carry_balance_hint={yest.get('balance')} | last={yest.get('last_time')}")
            print("[prompt] length:", len(full_prompt))
            print("[prompt] head:\n", full_prompt[:800])

        raw = query_model(fname, full_prompt)

        if DEBUG:
            print("[response] length:", len(raw))
            print("[response] head:\n", raw[:800])

        try:
            data = _parse_ai_json(raw)
        except Exception as e:
            if DEBUG:
                print("[parse] failed:", repr(e))
                print("[parse] raw snippet:", raw[:2000])
            data = {"model_name": fname, "universe": [], "starting_balance": yest.get("balance", DEFAULT_START_BAL), "portfolio_analysis": "", "trades": []}

        # Anchor to prompt name
        latest = _coerce_bot_day(data, yest.get("balance"), canonical_name=fname)
        if DEBUG:
            print(f"[parsed] trades={len(latest.trades)} starting_balance={latest.starting_balance}")

        # Validate AGAINST carried state (cash + lots)
        issues = _validate_trades_budget_with_state(
            latest.trades,
            carry_state.get("cash", DEFAULT_START_BAL),
            carry_state.get("lots", {}),
            prices
        )
        if issues:
            if DEBUG:
                print("[validate] issues found:", len(issues))
                for e in issues[:12]:
                    print("  -", e)
            correction_prompt = _build_correction_prompt(full_prompt, raw, issues, carry_note=carry_note)
            raw2 = query_model(fname, correction_prompt)
            try:
                data2 = _parse_ai_json(raw2)
                latest2 = _coerce_bot_day(data2, yest.get("balance"), canonical_name=fname)
                issues2 = _validate_trades_budget_with_state(
                    latest2.trades,
                    carry_state.get("cash", DEFAULT_START_BAL),
                    carry_state.get("lots", {}),
                    prices
                )
                if not issues2 and len(latest2.trades) > 0:
                    if DEBUG: print("[validate] correction succeeded")
                    latest = latest2
                    raw = raw2
                else:
                    if DEBUG:
                        print("[validate] correction still invalid; keeping original (unmodified).")
            except Exception as e:
                if DEBUG: print("[validate] correction parse failed:", repr(e))

        # Retry once if empty
        if len(latest.trades) == 0:
            retry_prompt = full_prompt + " IMPORTANT: Your previous response contained no trades. Use PRICES_TODAY and CURRENT_PORTFOLIO_STATE to produce 1–4 justified actions without exceeding AVAILABLE_CASH_USD, and include 'balance_after'. If you must hold, emit a single HOLD with 'balance_after'."
            raw2 = query_model(fname, retry_prompt)
            try:
                data2 = _parse_ai_json(raw2)
                latest2 = _coerce_bot_day(data2, yest.get("balance"), canonical_name=fname)
                if len(latest2.trades) > 0:
                    if DEBUG: print("[retry] succeeded with trades:", len(latest2.trades))
                    latest = latest2
                    raw = raw2
                    full_prompt = retry_prompt
                else:
                    if DEBUG: print("[retry] still no trades")
            except Exception as e:
                if DEBUG: print("[retry] parse failed:", repr(e))

        # Optional synthetic HOLD
        if len(latest.trades) == 0 and not DISABLE_SYNTH_HOLD:
            now = datetime.utcnow().isoformat() + "Z"
            if DEBUG:
                print("[synth] injecting HOLD at", now)
            synth = Trade(time=now, symbol="CASH", side="hold", qty=0.0, price=0.0, pnl_pct=None, balance_after=(latest.starting_balance or 0.0), notes="Auto HOLD: no trades returned; maintaining balance.")
            latest = BotDay(model_name=latest.model_name, universe=latest.universe, starting_balance=latest.starting_balance, trades=[synth], portfolio_analysis=latest.portfolio_analysis or "No trades — preserving capital.")
            raw = json.dumps({"model_name": latest.model_name, "universe": latest.universe if latest.universe is not None else "ALL", "starting_balance": latest.starting_balance or 0.0, "portfolio_analysis": latest.portfolio_analysis, "trades": [{
                "time": now, "symbol": "CASH", "side": "hold", "qty": 0, "price": 0, "pnl_pct": None, "balance_after": latest.starting_balance or 0.0, "notes": "Auto HOLD: no trades returned; maintaining balance."
            }]}, separators=(",",":"))

        _save_history(slug, {"prompt": full_prompt, "response": raw}, latest)

        # Aggregate full history for display & stats parity across runs
        agg = _aggregate_history(slug, latest)

        # Pretty display name derived from prompt filename (keep slug stable)
        pretty_name = fname.replace("-", " ").replace("_", " ").title()
        agg.model_name = pretty_name

        # unified stats (used by both pages)
        stats = _compute_fifo_stats(agg.trades)

        # Render model page (stable file name based on prompt slug)
        page = _render_model(env, agg, stats, prompt_text=full_prompt, response_text=raw, file_slug=slug)

        end_bal = _ending_balance(agg) or 0.0
        start_bal = agg.starting_balance or 0.0

        if isinstance(agg.universe, list):
            symbols_text = ", ".join(agg.universe) if agg.universe else "—"
        elif isinstance(agg.universe, str):
            symbols_text = agg.universe
        else:
            symbols_text = "—"

        if DEBUG:
            print(f"[stats] win_rate={stats['win_rate_num']:.2%} avg_win={stats['avg_win_num']:.2%} avg_loss={stats['avg_loss_num']:.2%} end_bal={end_bal:.2f}")

        bots_rows.append({
            "name": agg.model_name,
            "symbols": symbols_text,
            "win_rate": f"{stats['win_rate_num']*100:.2f}%",
            "win_rate_num": stats["win_rate_num"],
            "avg_win": f"{stats['avg_win_num']*100:.2f}%",
            "avg_loss": f"{stats['avg_loss_num']*100:.2f}%",
            "balance": f"${end_bal:,.2f}" if end_bal else "—",
            "balance_dir": 1 if end_bal >= start_bal else -1,
            "trades": len(agg.trades),
            "total_actions": len(agg.trades),
            "updated": time.strftime("%m/%d/%y %H:%M"),
            "link": page,
        })

    _render_overview(env, bots_rows)
    try:
        secs = _load_portfolio_sections_from_file()
        if secs:
            with open(PORTFOLIO_SECTIONS_JSON, "w", encoding="utf-8") as f:
                json.dump(secs, f, indent=2)
            if DEBUG: print("[overview] wrote", PORTFOLIO_SECTIONS_JSON)
    except Exception as e:
        if DEBUG: print("[overview] persist sections failed:", repr(e))
    print("Report generated")

if __name__ == '__main__':
    main(RUN_ONLY, RUN_GAPUPS_ONLY, RUN_PORTFOLIOS_ONLY, RUN_ALL, RUN_SKIP_GAPUPS, RUN_SKIP_PORTFOLIOS)
