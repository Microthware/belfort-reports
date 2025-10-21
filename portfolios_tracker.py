from __future__ import annotations
import re, csv, time, json, pdfminer
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup as bs
import yfinance as yf


HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
LOGO_CACHE_PATH = HISTORY_DIR / "logo_overrides.json"
_HP_STOCKS_CSV = HISTORY_DIR / "hp-stocks.csv"
CSV_HEADERS = [
    "published_date", "traded_date", "issuer", "ticker", "type", "size_range",
    "price", "shares", "pdf_url", "detail_url", "since_trade_pct", "since_trade_spy_pct"
]

@dataclass
class PortfolioSpec:
    slug: str
    name: str
    source: str
    identifier: str

PORTFOLIOS: List[PortfolioSpec] = [
    PortfolioSpec(slug="pelosi", name="Nancy Pelosi", source="capitoltrades", identifier="P000197"),
    PortfolioSpec(slug="tuberville", name="Tommy Tuberville", source="capitoltrades", identifier="T000278"),
    PortfolioSpec(slug="mccaul", name="Michael McCaul", source="capitoltrades", identifier="M001157"),
    PortfolioSpec(slug="gottheimer", name="Josh Gottheimer", source="capitoltrades", identifier="G000583"),
    PortfolioSpec(slug="hern", name="Kevin Hern", source="capitoltrades", identifier="H001082"),
]
    # PortfolioSpec(slug="vance", name="J. D. Vance", source="disclosure", identifier="V000137"),
    # PortfolioSpec(slug="mark-green", name="Mark E. Green", source="capitoltrades", identifier="G000590"),
    # PortfolioSpec(slug="blumenthal", name="Richard Blumenthal", source="capitoltrades", identifier="B001277"),
    # PortfolioSpec(slug="john-james", name="John James", source="capitoltrades", identifier="J000307"),

_DISCLOSURE_SOURCES = {
    "vance": [
        "https://www.whitehouse.gov/wp-content/uploads/2025/06/Vice-President-JD-Vance.pdf",
        "https://extapps2.oge.gov/201/Presiden.nsf/PAS%2BIndex/021DBF0DD058C1C185258CA9002C93BB/%24FILE/Vance%2C%20JD%202025%20Annual%20278.pdf",
        "https://www.documentcloud.org/documents/25041263-jd-vances-financial-disclosure/"
    ],
}

_DISCLOSURE_NAME_TO_TICKER = {
    "INVESCO QQQ": "QQQ",
    "QQQ": "QQQ",
    "SPDR S&P 500 ETF": "SPY",
    "SPDR S&P 500 ETF TRUST": "SPY",
    "SPY": "SPY",
    "SPDR DOW JONES INDUSTRIAL AVERAGE": "DIA",
    "DIA": "DIA",
    "ISHARES 20+ YEAR TREASURY BOND ETF": "TLT",
    "TLT": "TLT",
    "PROSHARES K-1 FREE CRUDE OIL STRATEGY ETF": "OILK",
    "OILK": "OILK",
    "SPDR GOLD TRUST": "GLD",
    "SPDR GOLD SHARES": "GLD",
    "GLD": "GLD",
    "RUMBLE": "RUM",
    "RUM": "RUM",
    "BITCOIN": "BTC",
    "BTC": "BTC",
}

def _csv_path(slug: str) -> Path:
    return HISTORY_DIR / f"{slug}-trades.csv"

def _ensure_csv(slug: str):
    p = _csv_path(slug)
    if not p.exists():
        with p.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADERS)

def _parse_text(el) -> str:
    if el is None: return ""
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()

def _get(url: str) -> str:
    if requests is None: raise RuntimeError("requests not available")
    r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    return r.text

def _get_soup(url: str):
    if bs is None: raise RuntimeError("bs4 not available")
    return bs(_get(url), "lxml")

def _ct_url(bioguide_id: str) -> str:
    return f"https://www.capitoltrades.com/politicians/{bioguide_id}"

def _ct_detail_links(bioguide_id: str, max_pages: int = 3) -> List[str]:
    links: List[str] = []
    base = _ct_url(bioguide_id)
    for p in range(1, max_pages+1):
        url = base if p == 1 else f"{base}?page={p}"
        soup = _get_soup(url)
        for a in soup.find_all("a"):
            txt = _parse_text(a)
            if "Goto trade detail page" in txt:
                href = a.get("href")
                if href and href.startswith("/"):
                    href = "https://www.capitoltrades.com" + href
                if href and href not in links:
                    links.append(href)
    return links

def _range_mid(s: str) -> float:
    s = s.replace(",", "").upper()
    m = re.search(r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*([KM]?)\s*[-–]\s*\$?\s*([0-9]+(?:\.[0-9]+)?)\s*([KM]?)", s)
    if not m:
        m2 = re.search(r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*([KM]?)\s*\+?", s)
        if not m2:
            return 0.0
        lo, lk = float(m2.group(1)), m2.group(2)
        mul = 1000.0 if lk=="K" else (1_000_000.0 if lk=="M" else 1.0)
        return lo*mul
    lo, lk, hi, hk = float(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
    mul_lo = 1000.0 if lk=="K" else (1_000_000.0 if lk=="M" else 1.0)
    mul_hi = 1000.0 if hk=="K" else (1_000_000.0 if hk=="M" else 1.0)
    return (lo*mul_lo + hi*mul_hi)/2.0

def _fetch_public_disclosure_text(slug: str) -> str:
    try:
        import requests
        from bs4 import BeautifulSoup
        import io
    except Exception:
        return ""
    urls = _DISCLOSURE_SOURCES.get(slug, [])
    text_blobs = []
    for u in urls:
        try:
            r = requests.get(u, timeout=20)
            ct = (r.headers.get("Content-Type") or "").lower()
            if "pdf" in ct or u.lower().endswith(".pdf"):
                try:
                    from pdfminer.high_level import extract_text
                    txt = extract_text(io.BytesIO(r.content))
                    if txt and len(txt) > 200:
                        text_blobs.append(txt)
                        continue
                except Exception:
                    print("EXCEPTION")
                    pass
            if "html" in ct or u.endswith("/"):
                soup = BeautifulSoup(r.text, "html.parser")
                txt = soup.get_text(" ", strip=True)
                if txt and len(txt) > 200:
                    text_blobs.append(txt)
                    continue
        except Exception:
            continue
    return "\n".join(text_blobs)

def _parse_disclosure_holdings(txt: str):
    if not txt:
        return []
    lines = [re.sub(r"\s+", " ", ln).strip().upper() for ln in txt.splitlines() if ln.strip()]
    buckets = {}
    for idx, ln in enumerate(lines):
        for name, tkr in _DISCLOSURE_NAME_TO_TICKER.items():
            if name in ln:
                m = re.search(r"\$\s*[0-9,]+(?:\s*[KM])?\s*[-–]\s*\$\s*[0-9,]+(?:\s*[KM])?", ln)
                amt = 0.0
                if not m and idx+1 < len(lines):
                    m = re.search(r"\$\s*[0-9,]+(?:\s*[KM])?\s*[-–]\s*\$\s*[0-9,]+(?:\s*[KM])?", lines[idx+1])
                if m:
                    amt = _range_mid(m.group(0))
                buckets[tkr] = buckets.get(tkr, 0.0) + float(amt)
    total = sum(buckets.values())
    if total <= 0:
        return []
    return [{"ticker": k, "weight": v/total} for k, v in sorted(buckets.items(), key=lambda x: -x[1])]

def disclosure_holdings_for_slug(slug: str):
    if slug != "vance":
        return []
    txt = _fetch_public_disclosure_text(slug)
    items = _parse_disclosure_holdings(txt)
    return items

def _ct_extract(url: str) -> Optional[Dict[str,str]]:
    soup = _get_soup(url)
    body = _parse_text(soup)

    def _find_date(label: str) -> Optional[str]:
        m = re.search(rf"{label}\s+(\d{{4}}-\d{{2}}-\d{{2}})", body, re.I)
        if m: return m.group(1)
        m2 = re.search(rf"{label}\s+(\d{{1,2}}\s+\w{{3}}\s+\d{{4}})", body, re.I)
        if m2:
            try:
                dt = datetime.strptime(m2.group(1), "%d %b %Y")
                return dt.strftime("%Y-%m-%d")
            except Exception: pass
        return None

    pub = _find_date("Published") or ""
    trd = _find_date("Traded") or ""

    m = re.search(r"\b([A-Z]{1,5}):US\b", body)
    ticker = (m.group(1) if m else "").upper()

    issuer = ""
    iss_el = soup.find("a", href=re.compile(r"/issuers/"))
    if iss_el: issuer = _parse_text(iss_el)
    if not issuer:
        h = soup.find(["h1","h2","h3"])
        if h: issuer = _parse_text(h)

    typ = ""
    mtyp = re.search(r"\b(buy|sell)\b", body, re.I)
    if mtyp: typ = mtyp.group(1).lower()

    size = ""
    msz = re.search(r"(\d[\w,\.]*\s*[–-]\s*\d[\w,\.]*)", body)
    if msz: size = msz.group(1).replace(" - ", "–")

    price = ""
    mpr = re.search(r"\$\s?([0-9,]+\.[0-9]{2}|[0-9,]+)", body)
    if mpr: price = mpr.group(0).replace(" ", "")

    shares = ""
    msh = re.search(r"\b([0-9]{1,3}(?:,[0-9]{3})+)\s+Shares\b", body, re.I)
    if msh: shares = msh.group(1)

    # Interpret "Purchased X call options ..." line if present
    ai = re.search(
        r"Additional Information Description:\s*Purchased\s+(\d{1,5})\s+call options?\s+with a strike price of\s*\$?\s*([0-9]{1,5}(?:\.[0-9]+)?)\s+and an expiration date of\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})",
        body, re.I
    )
    instrument = "option" if "option" in body.lower() else "stock"
    option_type = "call" if "call" in body.lower() else ("put" if "put" in body.lower() else "")
    is_exercise = bool(re.search(r"\bexercis", body, re.I) or re.search(r"\bassigned?\b", body, re.I))

    def _to_iso_date(s: str) -> str:
        s = (s or "").strip()
        fmts = ("%m/%d/%y","%m/%d/%Y","%b %d, %Y","%B %d, %Y","%Y-%m-%d")
        for fmt in fmts:
            try:
                dt = datetime.strptime(s, fmt)
                if fmt == "%m/%d/%y" and dt.year < 2000:
                    dt = dt.replace(year=dt.year + 2000)
                return dt.date().isoformat()
            except Exception:
                continue
        return ""

    if ai and typ == "buy":
        contracts = ai.group(1)
        strike = ai.group(2)
        expiry = _to_iso_date(ai.group(3))
        instrument = "option"; option_type = "call"; is_exercise = False
        shares = ""  # clear stray "50 Shares" for non-exercised options
    else:
        contracts = ""; strike = ""; expiry = ""

    pdf_url = ""
    a = soup.find("a", string=re.compile(r"View Original Filing", re.I))
    if a and a.get("href"):
        href = a.get("href")
        pdf_url = href if href.startswith("http") else ("https://" + href.lstrip("/"))

    return {
        "published_date":pub,"traded_date":trd,"issuer":issuer,"ticker":ticker,"type":typ,
        "size_range":size,"price":price,"shares":shares,"pdf_url":pdf_url,"detail_url":url,
        "since_trade_pct":"","since_trade_spy_pct":""
    }

def _load(slug: str) -> List[Dict[str,str]]:
    p = _csv_path(slug)
    if not p.exists(): return []
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def _save(slug: str, rows: List[Dict[str,str]]):
    p = _csv_path(slug)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({k: (r.get(k,"") or "") for k in CSV_HEADERS})

# ---------- Price history & returns ----------
def _series(symbol: str, start: str, end: Optional[str] = None):
    if yf is None: return []
    try:
        hist = yf.Ticker(symbol).history(start=start, end=end or None, auto_adjust=False)
        if hist is None or hist.empty: return []
        return [(ts.date().isoformat(), float(c)) for ts,c in hist["Close"].items() if float(c)>0]
    except Exception: return []

def _idx_on_or_after(dates: List[str], d0: str) -> Optional[int]:
    if d0 in dates: return dates.index(d0)
    for i,d in enumerate(dates):
        if d > d0: return i
    return None

def compute_returns_all():
    earliest = None
    for spec in PORTFOLIOS:
        for r in _load(spec.slug):
            d = r.get("traded_date") or ""
            if d and (earliest is None or d < earliest): earliest = d
    if earliest is None or yf is None: return
    end = (datetime.utcnow() + timedelta(days=1)).date().isoformat()
    spy = _series("SPY", earliest, end) or []
    spy_dates = [d for d,_ in spy]
    cache: Dict[str, List[tuple]] = {}
    for spec in PORTFOLIOS:
        rows = _load(spec.slug); changed=False
        for r in rows:
            sym = (r.get("ticker") or "").upper(); d0 = r.get("traded_date") or ""
            if not sym or not d0: continue
            if sym not in cache: cache[sym] = _series(sym, earliest, end) or []
            ser = cache[sym]; dates=[d for d,_ in ser]
            if not dates: continue
            i0 = _idx_on_or_after(dates, d0)
            if i0 is None: continue
            px0 = ser[i0][1]; px1 = ser[-1][1]
            if px0 and px1: r["since_trade_pct"] = f"{(px1/px0 - 1.0):.6f}"; changed=True
            if spy_dates:
                j0 = _idx_on_or_after(spy_dates, d0)
                if j0 is not None:
                    sp0 = spy[j0][1]; sp1 = spy[-1][1]
                    if sp0 and sp1: r["since_trade_spy_pct"]=f"{(sp1/sp0 - 1.0):.6f}"; changed=True
        if changed: _save(spec.slug, rows)

# ---------- Dollar parsing & holdings aggregation ----------
def _parse_dollar_token(tok: str) -> Optional[float]:
    """Parse tokens like $250K, 250K, $1.2M, 1000000 -> float dollars."""
    if tok is None: return None
    s = str(tok).strip().replace(",", "")
    m = re.fullmatch(r"\$?([0-9]+(?:\.[0-9]+)?)([kKmM]?)", s)
    if not m: return None
    num = float(m.group(1))
    suf = (m.group(2) or "").lower()
    mult = 1.0 if not suf else (1e3 if suf == "k" else 1e6)
    return num * mult

def _mid_range_dollars(txt: str) -> Optional[float]:
    """Return midpoint dollars from ranges like 250K–500K, $250K-$500K, 500000–1000000."""
    if not txt: return None
    s = str(txt).replace(",", "").replace("–", "-").strip()
    s = re.sub(r"\s*-\s*", "-", s)
    m = re.search(r"(\$?\d+(?:\.\d+)?[kKmM]?)-(\$?\d+(?:\.\d+)?[kKmM]?)", s)
    if not m: return None
    a = _parse_dollar_token(m.group(1))
    b = _parse_dollar_token(m.group(2))
    if a is None or b is None: return None
    return (a + b) / 2.0

def _num(s: str) -> Optional[float]:
    if not s: return None
    try:
        return float(re.sub(r"[^0-9\.]+","", s))
    except Exception:
        return None

def _estimate_row_dollars(r: Dict[str,str]) -> Optional[float]:
    """Best-effort dollars for this row: prefer size_range midpoint; else price*shares if both numeric."""
    mid = _mid_range_dollars(r.get("size_range") or "")
    if mid and mid > 0: return mid
    px = _num(r.get("price") or "")
    sh = _num((r.get("shares") or "").replace(",",""))
    if px and sh and px > 0 and sh > 0:
        return px * sh
    return None

# ---------- Logos (static + dynamic cache) ----------
LOGO_OVERRIDES_STATIC = {
    "NVDA": "https://logo.clearbit.com/nvidia.com",
    "AAPL": "https://logo.clearbit.com/apple.com",
    "MSFT": "https://logo.clearbit.com/microsoft.com",
    "AMZN": "https://logo.clearbit.com/amazon.com",
    "GOOGL":"https://logo.clearbit.com/abc.xyz",
    "META": "https://logo.clearbit.com/meta.com",
    "AVGO": "https://logo.clearbit.com/broadcom.com",
    "TSLA": "https://logo.clearbit.com/tesla.com",
    "AMD":  "https://logo.clearbit.com/amd.com",
    "NFLX": "https://logo.clearbit.com/netflix.com",
    "PANW": "https://logo.clearbit.com/paloaltonetworks.com",
    "CRWD": "https://logo.clearbit.com/crowdstrike.com",
    "VST":  "https://logo.clearbit.com/vistra.com",
    "TEM":  "https://logo.clearbit.com/tempus.com",
}

_LOGO_CACHE: Dict[str, str] = {}
_LOGO_CACHE_DIRTY = False

def _load_logo_cache() -> Dict[str,str]:
    global _LOGO_CACHE
    try:
        if LOGO_CACHE_PATH.exists():
            _LOGO_CACHE = json.loads(LOGO_CACHE_PATH.read_text(encoding="utf-8"))
        else:
            _LOGO_CACHE = {}
    except Exception:
        _LOGO_CACHE = {}
    return _LOGO_CACHE

def _save_logo_cache():
    global _LOGO_CACHE_DIRTY
    if not _LOGO_CACHE_DIRTY:
        return
    try:
        LOGO_CACHE_PATH.write_text(json.dumps(_LOGO_CACHE, indent=2, sort_keys=True), encoding="utf-8")
        _LOGO_CACHE_DIRTY = False
    except Exception:
        pass

def _domain_from_url(url: str) -> Optional[str]:
    try:
        netloc = urlparse(url).netloc or ""
        return netloc.lower() or None
    except Exception:
        return None

def _logo_for(ticker: str) -> str:
    """Return a logo URL for ticker. Order: cache.json > static overrides > yfinance website->clearbit > avatar fallback.
       Newly resolved entries are cached to history/logo_overrides.json so the map quickly grows toward 500+.
    """
    global _LOGO_CACHE_DIRTY, _LOGO_CACHE
    t = (ticker or "").upper()
    if not t:
        return "https://ui-avatars.com/api/?name=T&background=1f2937&color=ffffff&size=96&bold=true"

    cache = _load_logo_cache()
    if t in cache and cache[t]:
        return cache[t]

    if t in LOGO_OVERRIDES_STATIC:
        cache[t] = LOGO_OVERRIDES_STATIC[t]
        _LOGO_CACHE_DIRTY = True
        return cache[t]

    # yfinance website -> clearbit
    url = ""
    try:
        if yf is not None:
            info = {}
            try:
                info = yf.Ticker(t).info or {}
            except Exception:
                info = {}
            website = (info.get("website") or "") if isinstance(info, dict) else ""
            if isinstance(website, str) and website:
                dom = _domain_from_url(website)
                if dom:
                    url = f"https://logo.clearbit.com/{dom}"
    except Exception:
        url = ""

    if not url:
        url = f"https://ui-avatars.com/api/?name={t}&background=1f2937&color=ffffff&size=96&bold=true"

    cache[t] = url
    _LOGO_CACHE_DIRTY = True
    _LOGO_CACHE = cache
    return url

# ---------- Build holdings ----------
def _build_holdings(rows: List[Dict[str,str]]) -> List[Dict[str, object]]:
    if not rows: return []
    totals: Dict[str, float] = {}
    chg_num: Dict[str, float] = {}
    chg_den: Dict[str, float] = {}

    for r in rows:
        t = (r.get("ticker") or "").upper()
        if not t: continue
        dollars = _estimate_row_dollars(r)
        if dollars is None: continue
        side = (r.get("type") or "").lower()
        sign = -1.0 if side == "sell" else 1.0
        val = sign * dollars
        totals[t] = totals.get(t, 0.0) + val

        try:
            ch = float(r.get("since_trade_pct") or "")
        except Exception:
            ch = None
        if ch is not None:
            w = abs(dollars)
            chg_num[t] = chg_num.get(t, 0.0) + ch * w
            chg_den[t] = chg_den.get(t, 0.0) + w

    positives = {k:v for k,v in totals.items() if v > 0}
    denom = sum(positives.values())
    if denom <= 0: 
        return []

    items = []
    for t,v in sorted(positives.items(), key=lambda kv: kv[1], reverse=True):
        w = v / denom
        ch = ""
        if chg_den.get(t, 0.0) > 0:
            ch = chg_num[t] / chg_den[t]
        items.append({"ticker": t, "weight": w, "change": ch, "logo_url": _logo_for(t)})
    return items

# ---------- Sections for template ----------
def fetch_and_update_all(max_pages: int = 3):
    for spec in PORTFOLIOS:
        _ensure_csv(spec.slug)
        try:
            links = _ct_detail_links(spec.identifier, max_pages=max_pages) if spec.source=="capitoltrades" else []
        except Exception as e:
            print(f"[portfolios] list error {spec.slug}:", repr(e)); continue
        existing = {(r["pdf_url"], r["ticker"], r["traded_date"]) for r in _load(spec.slug)}
        rows = _load(spec.slug)
        for url in links:
            try: d = _ct_extract(url)
            except Exception: d = None
            if not d: continue
            key = (d["pdf_url"], d["ticker"], d["traded_date"])
            if key in existing: continue
            rows.append(d); existing.add(key); time.sleep(0.7)
        rows.sort(key=lambda x: ((x.get("published_date") or ""), (x.get("traded_date") or "")), reverse=True)
        _save(spec.slug, rows)

def _apply_seed_holdings(slug: str, derived: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Prefer public financial disclosure holdings if available; otherwise fall back
    to local seed json (history/seed_holdings/<slug>.json). If neither exists,
    return the originally derived holdings from trades.
    Each returned item must have: ticker, weight, change (optional), logo_url.
    """
    # Prefer disclosure-derived holdings (no hardcoding; parsed from public PDFs/pages)
    try:
        disc = disclosure_holdings_for_slug(slug)
    except Exception:
        disc = []
    if disc:
        out = []
        for it in disc:
            t = (it.get("ticker") or "").upper()
            w = float(it.get("weight") or 0.0)
            out.append({"ticker": t, "weight": w, "change": "", "logo_url": _logo_for(t)})
        s = sum(x.get("weight", 0.0) for x in out)
        if s > 0:
            for x in out:
                x["weight"] = x.get("weight", 0.0) / s
        return out

    return derived or []

def _hp_read_rows():
    p = _HP_STOCKS_CSV
    if not p.exists():
        return []
    out = []
    import csv as _csv
    with p.open("r", encoding="utf-8") as f:
        r = _csv.DictReader(f)
        for row in r:
            out.append({
                "date": (row.get("date") or "").strip(),
                "symbol": (row.get("symbol") or "").strip().upper(),
                "PNL": (row.get("PNL") or "").strip(),
                "gap ups": (row.get("gap ups") or "").strip(),
                "info": (row.get("info") or "").strip(),
            })
    return out

def _hp_write_rows(rows):
    p = _HP_STOCKS_CSV
    p.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    headers = ["date","symbol","PNL","gap ups","info"]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({
                "date": r.get("date",""),
                "symbol": r.get("symbol",""),
                "PNL": r.get("PNL",""),
                "gap ups": r.get("gap ups",""),
                "info": r.get("info",""),
            })

def _hp_parse_date(s: str) -> str | None:
    if not s: return None
    s = s.strip()
    from datetime import datetime as _dt
    for fmt in ("%Y-%m-%d","%m/%d/%Y","%m/%d/%y","%b %d, %Y","%B %d, %Y"):
        try:
            return _dt.strptime(s, fmt).date().isoformat()
        except Exception:
            pass
    return None

def _hp_yf_symbol(sym: str) -> str:
    s = (sym or "").upper().strip()
    if s == "BTC": return "BTC-USD"
    return s

def _hp_price_on_or_after(sym: str, iso_date: str) -> float | None:
    try:
        import yfinance as _yf
        df = _yf.Ticker(sym).history(start=iso_date, end=None, auto_adjust=False)
        if df is None or df.empty:
            return None
        for _, row in df.iterrows():
            c = float(row.get("Close") or 0)
            if c > 0: return c
    except Exception:
        pass
    return None

def _hp_last_close(sym: str) -> float | None:
    try:
        import yfinance as _yf
        df = _yf.Ticker(sym).history(period="5d", auto_adjust=False)
        if df is None or df.empty: return None
        v = float(df["Close"].iloc[-1])
        return v if v > 0 else None
    except Exception:
        return None

def load_hp_csv_for_render() -> list[dict]:
    """Return rows enriched with prices/pnl for templates; does not modify the CSV.
       Output: [{symbol,date,entry_price,current_price,pnl_float,pnl_text,gap_ups,info,logo_url}]
    """
    raw = _hp_read_rows()
    if not raw: return []
    symbols = sorted({r["symbol"] for r in raw if r["symbol"]})
    price_cache = {s: _hp_last_close(_hp_yf_symbol(s)) for s in symbols}
    out = []
    for r in raw:
        sym = r["symbol"]
        d_iso = _hp_parse_date(r["date"])
        entry = _hp_price_on_or_after(_hp_yf_symbol(sym), d_iso) if d_iso else None
        cur = price_cache.get(sym)
        pnl = (cur/entry - 1.0) if (entry and cur) else None
        pnl_text = (f"{pnl*100:.2f}%" if pnl is not None else "—")
        out.append({
            "symbol": sym,
            "date": r["date"],
            "entry_price": entry,
            "current_price": cur,
            "pnl_float": pnl,
            "pnl_text": pnl_text,
            "gap_ups": r.get("gap ups",""),
            "info": r.get("info",""),
            "logo_url": _logo_for(sym),
        })
    return out

def update_hp_csv_pnl():
    """Recompute PNL from 'date' to latest close and write back into the PNL column (as '%')."""
    rows = _hp_read_rows()
    if not rows: return
    symbols = sorted({r["symbol"] for r in rows if r["symbol"]})
    price_cache = {s: _hp_last_close(_hp_yf_symbol(s)) for s in symbols}
    for r in rows:
        sym = r["symbol"]
        d_iso = _hp_parse_date(r["date"])
        entry = _hp_price_on_or_after(_hp_yf_symbol(sym), d_iso) if d_iso else None
        cur = price_cache.get(sym)
        pnl = (cur/entry - 1.0) if (entry and cur) else None
        r["PNL"] = (f"{pnl*100:.2f}%" if pnl is not None else (r.get("PNL") or ""))
    _hp_write_rows(rows)

def load_sections_for_render() -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    _load_logo_cache()  # ensure cache is in memory
    for spec in PORTFOLIOS:
        rows = _load(spec.slug)
        for r in rows:
            for k in CSV_HEADERS: r[k] = r.get(k,"") or ""
        rows.sort(key=lambda x: ((x.get("published_date") or ""), (x.get("traded_date") or "")), reverse=True)
        holdings = _apply_seed_holdings(spec.slug, _build_holdings(rows))
        out.append({"slug": spec.slug, "name": spec.name, "rows": rows, "holdings": holdings})
    _save_logo_cache()
    return out
