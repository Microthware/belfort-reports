
from __future__ import annotations
import re, csv, time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

try:
    import requests
    from bs4 import BeautifulSoup as bs  # type: ignore
except Exception:
    requests = None
    bs = None

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

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
]

def _csv_path(slug: str) -> Path:
    return HISTORY_DIR / f"{slug}-trades.csv"

def _ensure_csv(slug: str):
    p = _csv_path(slug)
    if not p.exists():
        with p.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADERS)

def _parse_text(el) -> str:
    if el is None: return ""
    import re
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

def _ct_extract(url: str) -> Optional[Dict[str,str]]:
    soup = _get_soup(url)
    body = _parse_text(soup)

    def _find_date(label: str) -> Optional[str]:
        import re
        m = re.search(rf"{label}\s+(\d{{4}}-\d{{2}}-\d{{2}})", body, re.I)
        if m: return m.group(1)
        m2 = re.search(rf"{label}\s+(\d{{1,2}}\s+\w{{3}}\s+\d{{4}})", body, re.I)
        if m2:
            try:
                from datetime import datetime
                dt = datetime.strptime(m2.group(1), "%d %b %Y")
                return dt.strftime("%Y-%m-%d")
            except Exception: pass
        return None

    pub = _find_date("Published") or ""
    trd = _find_date("Traded") or ""

    import re
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

    pdf_url = ""
    a = soup.find("a", string=re.compile(r"View Original Filing", re.I))
    if a and a.get("href"):
        href = a.get("href")
        pdf_url = href if href.startswith("http") else ("https://" + href.lstrip("/"))

    return {"published_date":pub,"traded_date":trd,"issuer":issuer,"ticker":ticker,"type":typ,"size_range":size,"price":price,"shares":shares,"pdf_url":pdf_url,"detail_url":url,"since_trade_pct":"","since_trade_spy_pct":""}

def _load(slug: str) -> List[Dict[str,str]]:
    p = _csv_path(slug)
    if not p.exists(): return []
    import csv
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def _save(slug: str, rows: List[Dict[str,str]]):
    p = _csv_path(slug)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADERS); w.writeheader()
        for r in rows: w.writerow(r)

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
    # earliest trade date across all portfolios
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

def load_sections_for_render() -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for spec in PORTFOLIOS:
        rows = _load(spec.slug)
        for r in rows:
            for k in CSV_HEADERS: r[k] = r.get(k,"") or ""
        rows.sort(key=lambda x: ((x.get("published_date") or ""), (x.get("traded_date") or "")), reverse=True)
        out.append({"slug": spec.slug, "name": spec.name, "rows": rows})
    return out
