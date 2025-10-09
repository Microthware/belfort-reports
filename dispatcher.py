#!/usr/bin/env python3
import os, json, shutil, time, glob, re, sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Debug & runtime flags
# ------------------------------------------------------------------
DEBUG = False
DISABLE_SYNTH_HOLD = os.environ.get("DISABLE_SYNTH_HOLD", "0") == "1"

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

def _load_yesterday_state(model_slug: str) -> Dict[str, Any]:
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
            print("[state] failed to read last history:", repr(e))
        return {"balance": None, "last_time": None}

def _save_history(model_slug: str, payload: Dict[str, Any], parsed: BotDay):
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
        print("[history] wrote", path)

def _parse_ai_json(txt: str) -> Dict[str, Any]:
    # Find the first JSON object in the response
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in response.")
    return json.loads(m.group(0))

def _coerce_bot_day(d: Dict[str, Any], carry_balance: float | None) -> BotDay:
    name = d.get("model_name") or d.get("name") or "Unnamed Bot"
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

def _stats_from_trades(trades: List[Trade]) -> Dict[str, float]:
    wins, losses = [], []
    for t in trades:
        if t.pnl_pct is None:
            continue
        if t.pnl_pct > 0: wins.append(t.pnl_pct)
        elif t.pnl_pct < 0: losses.append(t.pnl_pct)
    total = len(wins) + len(losses)
    win_rate = (len(wins)/total) if total else 0.0
    avg_win = sum(wins)/len(wins) if wins else 0.0
    avg_loss = sum(losses)/len(losses) if losses else 0.0
    return {"trades": total, "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss}

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
# Aggregation
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
                    balance_after=(float(td.get("balance_after")) if td.get("balance_after") is not None else None),
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

# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------
def _render_overview(env, bots_rows: List[Dict[str, Any]]):
    tpl = env.get_template("overview.html")
    gen_time = time.strftime("%m/%d/%y %H:%M")
    total_trades = sum(r["trades"] for r in bots_rows)
    avg_win_rate = sum(r["win_rate_num"] for r in bots_rows)/len(bots_rows) if bots_rows else 0.0
    html = tpl.render(
        gen_time=gen_time,
        tz=TZ,
        bots=bots_rows,
        total_trades=total_trades,
        avg_win_rate=f"{avg_win_rate*100:.2f}%",
    )
    open(os.path.join(REPORT_DIR,"overview.html"),"w",encoding="utf-8").write(html)

def _render_model(env, b: BotDay, prompt_text: str = "", response_text: str = ""):
    tpl = env.get_template("model_report.html")
    series = {"t":[], "v":[]}
    chron = list(sorted(b.trades, key=lambda x: x.time))
    for t in chron:
        if t.balance_after is not None:
            series["t"].append(t.time)
            series["v"].append(float(t.balance_after))
    bal_text = f"${_ending_balance(b):,.2f}" if _ending_balance(b) is not None else "—"
    stats = _stats_from_trades(b.trades)
    html = tpl.render(
        model_name=b.model_name,
        gen_time=time.strftime("%m/%d/%y %H:%M"),
        tz=TZ,
        balance_text=bal_text,
        win_rate_text=f"{stats['win_rate']*100:.2f}%",
        avg_win_text=f"{stats['avg_win']*100:.2f}%",
        avg_loss_text=f"{stats['avg_loss']*100:.2f}%",
        symbols=(", ".join(b.universe) if isinstance(b.universe, list) and b.universe else (b.universe if isinstance(b.universe, str) else "—")),
        trades=[{
            "time": t.time,
            "symbol": t.symbol,
            "side": t.side,
            "qty": f"{t.qty:.4f}".rstrip("0").rstrip("."),
            "price": f"{t.price:.4f}",
            "pnl_pct": (f"{t.pnl_pct*100:.2f}%" if t.pnl_pct is not None else "—"),
            "pnl_pct_num": (t.pnl_pct if t.pnl_pct is not None else 0.0),
            "balance_after": (f"${t.balance_after:,.2f}" if t.balance_after is not None else "—"),
            "balance_dir": 1 if (t.pnl_pct or 0) >= 0 else -1,
            "notes": t.notes,
        } for t in b.trades],
        balance_series_json=json.dumps(series, separators=(",",":")),
        portfolio_analysis=b.portfolio_analysis,
        prompt_text=prompt_text,
        response_text=response_text,
    )
    fname = f"{_slug(b.model_name)}.html"
    open(os.path.join(REPORT_DIR, fname),"w",encoding="utf-8").write(html)
    return fname

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

def _build_prompt(base_prompt: str, yesterday: Dict[str,Any], candidates_note: str, prices_note: str) -> str:
    carry = yesterday.get("balance")
    last = yesterday.get("last_time")
    start_balance_hint = carry if carry is not None else DEFAULT_START_BAL

    parts: List[str] = []
    parts.append("You are an autonomous trading agent. Reply with STRICT JSON ONLY (no commentary, no code fences) following the schema documented below.")
    parts.append(f"- Starting balance for a new/first run: {start_balance_hint}.")
    parts.append(f"- Include only NEW actions since {last or 'yesterday'} but ensure the final 'balance_after' reflects current balance.")
    parts.append("- You MUST output at least one action for the current day (first run included). Prefer buys from CANDIDATES_TODAY. If truly nothing qualifies, emit a single HOLD with 'balance_after'.")
    parts.append("- Allocate ~100% of capital across 1–4 positions using PRICES_TODAY (USD). For each buy, compute whole-share 'qty' = floor(allocation * starting_balance / price). Set 'balance_after' equal to total portfolio value after actions. PNL may be 0 on entry.")
    parts.append("- Use ISO8601 UTC timestamps (e.g., 2025-10-06T15:30:00Z).")
    parts.append("- Provide pnl_pct for closed trades when possible.")
    parts.append("- Keep numbers as raw numbers (no % symbols).")
    parts.append("- Include a concise 'portfolio_analysis' explaining the day's selection/rationale in 1–4 sentences.")
    parts.append("PRICES_TODAY (USD):")
    parts.append(prices_note)
    parts.append("CANDIDATES_TODAY:")
    parts.append(candidates_note)
    parts.append("Schema:")
    parts.append(SCHEMA_SNIPPET)
    parts.append("=== STRATEGY CONTEXT ===")
    parts.append(base_prompt)

    # Minimal HOLD example for clarity
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
    print(api_key)
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

    try:
        from openai import OpenAI  # lazy import
        client = OpenAI(api_key=api_key)
        if DEBUG: print(f"[openai] request -> model={model}, prompt_len={len(prompt_text)}")
        resp = client.responses.create(model=model, input=prompt_text, temperature=0.2)
        text = resp.output_text
        if DEBUG: print("[openai] response chars:", len(text))
        return text
    except Exception as e:
        if DEBUG: print("[openai] error:", repr(e))
        sample_path = os.path.join(PROMPTS_DIR, f"{model_name}.json")
        if os.path.exists(sample_path):
            return open(sample_path,"r",encoding="utf-8").read()
        return dev_sample()

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
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

    # Scan candidates & prices once per run
    # buckets = _scan_candidates()
    buckets = {}
    candidates_note = _format_candidates_for_prompt(buckets)
    uniq: List[str] = sorted({t.upper() for arr in buckets.values() for t in arr})
    prices = _fetch_prices(uniq)
    prices_note = _format_prices_for_prompt(prices)
    if DEBUG:
        print("[scan] CANDIDATE buckets:", {k: len(v) for k, v in buckets.items()})
        print("[prices] count:", len(prices))

    for fname, base_prompt in prompts.items():
        slug = _slug(fname)
        yest = _load_yesterday_state(slug)
        full_prompt = _build_prompt(base_prompt, yest, candidates_note, prices_note)

        if DEBUG:
            print(f"\n[bot] {fname} | slug={slug} | carry={yest.get('balance')} | last={yest.get('last_time')}")
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

        latest = _coerce_bot_day(data, yest.get("balance"))
        if DEBUG:
            print(f"[parsed] trades={len(latest.trades)} starting_balance={latest.starting_balance}")

        # Retry once if empty
        if len(latest.trades) == 0:
            retry_prompt = full_prompt + " IMPORTANT: Your previous response contained no trades. Use PRICES_TODAY to buy 1–4 names from CANDIDATES_TODAY, compute whole-share qty, and include 'balance_after'. If you truly must hold, emit a single HOLD with 'balance_after'."
            raw2 = query_model(fname, retry_prompt)
            try:
                data2 = _parse_ai_json(raw2)
                latest2 = _coerce_bot_day(data2, yest.get("balance"))
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

        agg = _aggregate_history(slug, latest)
        page = _render_model(env, agg, prompt_text=full_prompt, response_text=raw)
        stats = _stats_from_trades(agg.trades)
        end_bal = _ending_balance(agg) or 0.0
        start_bal = agg.starting_balance or 0.0

        if isinstance(agg.universe, list):
            symbols_text = ", ".join(agg.universe) if agg.universe else "—"
        elif isinstance(agg.universe, str):
            symbols_text = agg.universe
        else:
            symbols_text = "—"

        if DEBUG:
            print(f"[stats] win_rate={stats['win_rate']:.2%} avg_win={stats['avg_win']:.2%} avg_loss={stats['avg_loss']:.2%} end_bal={end_bal:.2f}")

        bots_rows.append({
            "name": agg.model_name,
            "symbols": symbols_text,
            "win_rate": f"{stats['win_rate']*100:.2f}%",
            "win_rate_num": stats["win_rate"],
            "avg_win": f"{stats['avg_win']*100:.2f}%",
            "avg_loss": f"{stats['avg_loss']*100:.2f}%",
            "balance": f"${end_bal:,.2f}" if end_bal else "—",
            "balance_dir": 1 if end_bal >= start_bal else -1,
            "trades": stats["trades"],
            "updated": time.strftime("%m/%d/%y %H:%M"),
            "link": page,
        })

    _render_overview(env, bots_rows)

    print("Report generated")

if __name__ == "__main__":
    main()
