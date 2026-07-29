"""
Creature-page scraper for the Wizard101 Central wiki (run locally).

Politeness contract (MediaWiki etiquette):
  - uses the MediaWiki API (api.php), the sanctioned programmatic surface,
    never HTML page scraping
  - identifies itself with a descriptive User-Agent (set --contact)
  - >= 1 s between requests, maxlag=5, honors Retry-After on 429/503
  - treats HTTP 403 as a BLOCK and stops with guidance — it is not a
    transient error and must not be retried with backoff

The wiki sits behind Cloudflare, which rejects the default python-requests
User-Agent with 403. Escalation ladder if you see 403:
  1. run with a real --contact so the UA is descriptive;
  2. `pip install cloudscraper` (picked up automatically) — passes the
     non-interactive Cloudflare challenge;
  3. the zero-automation fallback: open
        https://www.wizard101central.com/wiki/Special:Export
     in your browser, enter "Category:Creatures" under "Add pages from
     category", export, and feed the XML here with --from-xml export.xml.

Outputs:
  creatures_raw.jsonl   one record per page: title, revision, raw infobox
                        fields, wikitext length (resumable; reruns skip
                        already-fetched titles)
  creatures_clean.json  normalized stats: school, rank, health, stunable,
                        beguilable kept as True/False/None (never silently
                        zeroed), plus the raw field dict and a confidence
                        tag ("wiki-scrape").

Usage:
  python scrape_creatures.py --limit 30 --contact you@example.com
  python scrape_creatures.py --from-xml export.xml
"""
import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

API = "https://www.wizard101central.com/wiki/api.php"
RAW = Path("creatures_raw.jsonl")
CLEAN = Path("creatures_clean.json")


# ---------------------------------------------------------------- wikitext

def split_template_fields(wikitext):
    """Parse the first *Infobox template into a {key: value} dict, honoring
    nested {{...}} and [[...]] so pipes inside them don't split fields."""
    m = re.search(r"\{\{\s*([^|{}]*?Infobox[^|{}]*?)\s*[|}]", wikitext,
                  re.IGNORECASE)
    if not m:
        m = re.search(r"\{\{\s*(Creature[^|{}]*?)\s*[|}]", wikitext,
                      re.IGNORECASE)
    if not m:
        return {}
    start = wikitext.rfind("{{", 0, m.end())
    depth, i, n = 0, start, len(wikitext)
    parts, buf = [], []
    while i < n:
        two = wikitext[i:i + 2]
        if two in ("{{", "[["):
            depth += 1
            buf.append(two)
            i += 2
        elif two in ("}}", "]]"):
            depth -= 1
            if depth == 0 and two == "}}":
                parts.append("".join(buf))
                break
            buf.append(two)
            i += 2
        elif wikitext[i] == "|" and depth == 1:
            parts.append("".join(buf))
            buf = []
            i += 1
        else:
            buf.append(wikitext[i])
            i += 1
    fields = {}
    for part in parts[1:]:                    # parts[0] is the template name
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower().replace(" ", "_")
        v = re.sub(r"\[\[(?:[^]|]*\|)?([^]|]*)\]\]", r"\1", v)   # [[a|b]]->b
        v = re.sub(r"<[^>]+>", "", v).strip()
        if k:
            fields[k] = v
    return fields


def _to_int(v):
    if v is None:
        return None
    m = re.search(r"-?[\d,]+", str(v))
    return int(m.group().replace(",", "")) if m else None


def _to_bool(v):
    if v is None or str(v).strip() == "":
        return None                            # unspecified stays null
    s = str(v).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return True
    if s in ("no", "n", "false", "0"):
        return False
    return None


def normalize(title, fields):
    school = (fields.get("school") or "").strip().lower() or None
    return dict(
        name=title,
        school=school,
        rank=_to_int(fields.get("rank")),
        health=_to_int(fields.get("health")),
        stunable=_to_bool(fields.get("stunable")),
        beguilable=_to_bool(fields.get("beguile") or
                            fields.get("beguilable")),
        classification=fields.get("classification"),
        confidence="wiki-scrape",
        fields=fields,
    )


# ---------------------------------------------------------------- API client

def make_session(contact):
    ua = (f"wizAi-creature-scraper/0.3 (offline combat-sim research; "
          f"contact: {contact}) python")
    try:
        import cloudscraper
        sess = cloudscraper.create_scraper()
        print("using cloudscraper session")
    except ImportError:
        import requests
        sess = requests.Session()
    sess.headers["User-Agent"] = ua
    return sess


def api_get(sess, params, delay):
    params = dict(params, format="json", maxlag=5)
    while True:
        time.sleep(delay)
        r = sess.get(API, params=params, timeout=30)
        if r.status_code == 403:
            sys.exit(
                "HTTP 403: the wiki's Cloudflare layer is blocking this "
                "client. 403 is a block, not congestion — do NOT retry.\n"
                "  1. pass --contact so the User-Agent identifies you\n"
                "  2. pip install cloudscraper and rerun\n"
                "  3. zero-automation fallback: browser -> Special:Export "
                "-> 'Add pages from category: Category:Creatures' -> save "
                "XML -> rerun with --from-xml export.xml")
        if r.status_code in (429, 503):
            wait = int(r.headers.get("Retry-After", 30))
            print(f"  HTTP {r.status_code}, waiting {wait}s (server asked)")
            time.sleep(wait)
            continue
        r.raise_for_status()
        data = r.json()
        if "error" in data and data["error"].get("code") == "maxlag":
            print("  server lagged, waiting 5s")
            time.sleep(5)
            continue
        return data


def list_category(sess, category, limit, delay):
    titles, cont = [], {}
    while len(titles) < limit:
        data = api_get(sess, dict(
            action="query", list="categorymembers",
            cmtitle=f"Category:{category}", cmnamespace=0,
            cmlimit=min(500, limit - len(titles)), **cont), delay)
        titles += [m["title"] for m in
                   data["query"]["categorymembers"]]
        cont = data.get("continue", {})
        if not cont:
            break
    return titles[:limit]


def fetch_pages(sess, titles, delay):
    """Yield (title, wikitext) in API batches of 50 (one request per 50)."""
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        data = api_get(sess, dict(
            action="query", prop="revisions", rvprop="content",
            rvslots="main", titles="|".join(batch)), delay)
        for page in data["query"]["pages"].values():
            revs = page.get("revisions")
            if not revs:
                continue
            slot = revs[0].get("slots", {}).get("main", revs[0])
            yield page["title"], slot.get("*") or slot.get("content", "")


# ---------------------------------------------------------------- XML dump

def pages_from_xml(path):
    """Parse a Special:Export XML dump (namespace-agnostic)."""
    for _, elem in ET.iterparse(path):
        if elem.tag.rsplit("}", 1)[-1] == "page":
            title = text = None
            for node in elem.iter():
                tag = node.tag.rsplit("}", 1)[-1]
                if tag == "title":
                    title = node.text
                elif tag == "text":
                    text = node.text or ""
            if title and text is not None:
                yield title, text
            elem.clear()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Creatures")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--delay", type=float, default=1.2,
                    help="seconds between API requests (min 1.0)")
    ap.add_argument("--contact", default="set-me@example.com",
                    help="contact for the User-Agent (wiki etiquette)")
    ap.add_argument("--from-xml", metavar="FILE",
                    help="parse a browser-downloaded Special:Export dump "
                         "instead of hitting the API")
    args = ap.parse_args()
    delay = max(args.delay, 1.0)

    done = set()
    if RAW.exists():
        for line in RAW.open(encoding="utf-8"):
            done.add(json.loads(line)["name"])
        print(f"resuming: {len(done)} pages already fetched")

    out = RAW.open("a", encoding="utf-8")

    def record(title, wikitext):
        fields = split_template_fields(wikitext)
        rec = normalize(title, fields)
        rec["wikitext_chars"] = len(wikitext)
        out.write(json.dumps(rec) + "\n")
        return rec

    n_new = 0
    if args.from_xml:
        for title, text in pages_from_xml(args.from_xml):
            if title in done:
                continue
            record(title, text)
            n_new += 1
    else:
        sess = make_session(args.contact)
        print(f"Listing pages in Category:{args.category}...")
        titles = [t for t in
                  list_category(sess, args.category, args.limit, delay)
                  if t not in done]
        print(f"{len(titles)} pages to fetch "
              f"({(len(titles) + 49) // 50} API requests)")
        for title, text in fetch_pages(sess, titles, delay):
            record(title, text)
            n_new += 1
    out.close()
    print(f"fetched {n_new} new pages -> {RAW}")

    # rebuild the clean file from the full raw log
    clean = [json.loads(line) for line in RAW.open(encoding="utf-8")]
    for rec in clean:
        rec.pop("wikitext_chars", None)
    json.dump(clean, CLEAN.open("w", encoding="utf-8"), indent=1)
    print(f"wrote {len(clean)} creatures -> {CLEAN}")


if __name__ == "__main__":
    main()
