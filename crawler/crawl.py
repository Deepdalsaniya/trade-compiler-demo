# crawler/crawl.py
import hashlib, time, json, re, sys, io
import requests, yaml
from urllib.parse import urljoin, urlparse
from urllib import robotparser
from bs4 import BeautifulSoup
import pdfplumber

from storage import get_db, upsert_doc

UA = "TradeComplianceHarvester/0.1 (respecting robots; contact: demo@example.com)"
TIMEOUT = 20
SLEEP = 1.0

def sha256(b: bytes): return hashlib.sha256(b).hexdigest()

def allowed_by_robots(url, ua=UA):
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    rp = robotparser.RobotFileParser()
    rp.set_url(urljoin(base, "/robots.txt"))
    try:
        rp.read()
        return rp.can_fetch(ua, url)
    except:
        return True

def is_pdf(resp):
    ct = resp.headers.get("Content-Type","").lower()
    return "pdf" in ct or resp.url.lower().endswith(".pdf")

def extract_text(resp):
    if is_pdf(resp):
        try:
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            return text
        except:
            return ""
    # HTML
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script","style","nav","footer","header"]): tag.decompose()
    title = (soup.title.string.strip() if soup.title else "")
    text = " ".join(soup.get_text(separator=" ").split())
    return title, text

def crawl_source(source):
    con = get_db()
    seen = set()
    queue = list(source["seeds"])
    allowed_pat = [re.compile(p, re.I) for p in source.get("patterns_allow",[])]
    deny_pat    = [re.compile(p, re.I) for p in source.get("patterns_deny",[])]
    max_pages   = int(source.get("max_pages", 50))
    base        = source["base"]

    def ok(u):
        if not u.startswith("http"): u = urljoin(base, u)
        if not u.startswith(base): return False
        if any(p.search(u) for p in deny_pat): return False
        if allowed_pat and not any(p.search(u) for p in allowed_pat): return False
        return True

    processed = 0
    while queue and processed < max_pages:
        url = queue.pop(0)
        if url in seen: continue
        seen.add(url)
        if not ok(url): continue
        if not allowed_by_robots(url): continue

        try:
            resp = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
            status = resp.status_code
        except Exception as e:
            upsert_doc(con, dict(source=source["name"], url=url, fetched_at=int(time.time()),
                                 status=0, http_status=None, content_type=None,
                                 sha256=None, title=None, text_excerpt=str(e)[:500]))
            continue

        if status != 200:
            upsert_doc(con, dict(source=source["name"], url=url, fetched_at=int(time.time()),
                                 status=0, http_status=status, content_type=None,
                                 sha256=None, title=None, text_excerpt=""))
            continue

        digest = sha256(resp.content)
        ctype = resp.headers.get("Content-Type","")
        title, text = ("","")
        if is_pdf(resp):
            text = extract_text(resp)
        else:
            title, text = extract_text(resp)
            # enqueue more links (same host)
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = urljoin(resp.url, a["href"])
                if ok(href): queue.append(href)

        upsert_doc(con, dict(
            source=source["name"], url=resp.url, fetched_at=int(time.time()),
            status=1, http_status=status, content_type=ctype, sha256=digest,
            title=title, text_excerpt=text[:1500]
        ))
        processed += 1
        time.sleep(SLEEP)

if __name__ == "__main__":
    cfg = yaml.safe_load(open("crawler/sources.yml","r",encoding="utf-8"))
    for s in cfg["sources"]:
        crawl_source(s)

