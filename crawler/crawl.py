# crawler/crawl.py

# Import necessary libraries for handling hashing, time, JSON, regular expressions, system functions, and input/output
import hashlib, time, json, re, sys, io
# Import requests for HTTP requests, yaml for YAML file handling
import requests, yaml
# Import URL helpers
from urllib.parse import urljoin, urlparse
from urllib import robotparser
# Import BeautifulSoup for parsing HTML
from bs4 import BeautifulSoup
# Import pdfplumber for reading PDF files
import pdfplumber

# Import database functions from another file
from storage import get_db, upsert_doc

# User agent string for the crawler
UA = "TradeComplianceHarvester/0.1 (respecting robots; contact: demo@example.com)"
# Set HTTP request timeout and sleep time between requests
TIMEOUT = 20
SLEEP = 1.0

# Function to calculate SHA256 hash of bytes
def sha256(b: bytes): 
    return hashlib.sha256(b).hexdigest()

# Function to check if crawling a URL is allowed by robots.txt
def allowed_by_robots(url, ua=UA):
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"  # Get site base URL
    rp = robotparser.RobotFileParser()
    rp.set_url(urljoin(base, "/robots.txt"))  # Set robots.txt location
    try:
        rp.read()  # Read robots.txt
        return rp.can_fetch(ua, url)  # Check if our user agent can fetch the URL
    except:
        return True  # If robots.txt can't be read, allow by default

# Function to check if a response is a PDF file
def is_pdf(resp):
    ct = resp.headers.get("Content-Type","").lower()
    return "pdf" in ct or resp.url.lower().endswith(".pdf")

# Function to extract text from a response, handles both PDF and HTML
def extract_text(resp):
    if is_pdf(resp):
        try:
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                # Get text from all PDF pages
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            return text
        except:
            return ""
    # If not PDF, treat as HTML
    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove unwanted HTML tags
    for tag in soup(["script","style","nav","footer","header"]): 
        tag.decompose()
    # Get page title if available
    title = (soup.title.string.strip() if soup.title else "")
    # Get all visible text from the page
    text = " ".join(soup.get_text(separator=" ").split())
    return title, text

# Main function to crawl a single source
def crawl_source(source):
    con = get_db()  # Get database connection
    seen = set()  # Keep track of visited URLs
    queue = list(source["seeds"])  # Initialize queue with seed URLs
    # Compile allowed and denied URL patterns
    allowed_pat = [re.compile(p, re.I) for p in source.get("patterns_allow",[])]
    deny_pat    = [re.compile(p, re.I) for p in source.get("patterns_deny",[])]
    max_pages   = int(source.get("max_pages", 50))  # Limit number of pages to crawl
    base        = source["base"]  # Base URL for this source

    # Function to check if a URL is okay to crawl
    def ok(u):
        if not u.startswith("http"): 
            u = urljoin(base, u)  # Make sure it's absolute URL
        if not u.startswith(base): 
            return False  # Only crawl URLs starting with base
        if any(p.search(u) for p in deny_pat): 
            return False  # Skip if matches denied patterns
        if allowed_pat and not any(p.search(u) for p in allowed_pat): 
            return False  # If allowed patterns exist, skip if not matched
        return True

    processed = 0  # Count processed pages
    while queue and processed < max_pages:
        url = queue.pop(0)  # Get the next URL from the queue
        if url in seen: 
            continue  # Skip already seen URLs
        seen.add(url)
        if not ok(url): 
            continue  # Skip disallowed URLs
        if not allowed_by_robots(url): 
            continue  # Skip if not allowed by robots.txt

        try:
            # Try to fetch the URL
            resp = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
            status = resp.status_code
        except Exception as e:
            # If request fails, save error info to database
            upsert_doc(con, dict(
                source=source["name"], url=url, fetched_at=int(time.time()),
                status=0, http_status=None, content_type=None,
                sha256=None, title=None, text_excerpt=str(e)[:500]
            ))
            continue

        if status != 200:
            # If HTTP status is not OK, save info to database
            upsert_doc(con, dict(
                source=source["name"], url=url, fetched_at=int(time.time()),
                status=0, http_status=status, content_type=None,
                sha256=None, title=None, text_excerpt=""
            ))
            continue

        # Calculate hash and content type
        digest = sha256(resp.content)
        ctype = resp.headers.get("Content-Type","")
        title, text = ("","")
        if is_pdf(resp):
            text = extract_text(resp)  # Extract text from PDF
        else:
            title, text = extract_text(resp)  # Extract title and text from HTML
            # Find all links on the page and add them to the queue
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = urljoin(resp.url, a["href"])
                if ok(href): 
                    queue.append(href)

        # Save page data to database
        upsert_doc(con, dict(
            source=source["name"], url=resp.url, fetched_at=int(time.time()),
            status=1, http_status=status, content_type=ctype, sha256=digest,
            title=title, text_excerpt=text[:1500]
        ))
        processed += 1  # Increase processed count
        time.sleep(SLEEP)  # Wait before next request

# Run the crawler if this script is executed directly
if __name__ == "__main__":
    # Load sources from YAML config file
    cfg = yaml.safe_load(open("crawler/sources.yml","r",encoding="utf-8"))
    # For each source, start crawling
    for s in cfg["sources"]:
        crawl_source(s)
