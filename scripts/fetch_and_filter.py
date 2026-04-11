#!/usr/bin/env python3
"""
Unified fetch + filter + rate pipeline — ONE LLM call.
Fetches today's arXiv papers, evaluates relevance + generates Chinese abstracts
in a single batch LLM call.
"""

import os, sys, json, time, logging, arxiv, requests
from datetime import date, datetime, timedelta, timezone
from typing import Optional

# Force unbuffered output so GitHub Actions logs appear immediately
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger("pipeline")

# ── API Config ──────────────────────────────────────────────────────────────
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
OPENAI_API_URL      = f"{OPENAI_API_BASE_URL}/chat/completions"
OPENAI_MODEL        = os.getenv("OPENAI_MODEL", "gpt-5.4")

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR       = os.path.join(PROJECT_ROOT, "daily_json")
HTML_DIR       = os.path.join(PROJECT_ROOT, "daily_html")
TEMPLATE_DIR   = os.path.join(PROJECT_ROOT, "templates")
TEMPLATE_NAME  = "paper_template.html"
REPORTS_PATH   = os.path.join(PROJECT_ROOT, "reports.json")
DATA_RETENTION = 60

TARGET_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG", "cs.SD", "eess.AS", "cs.MM", "cs.CV"]

# ── LLM Caller ───────────────────────────────────────────────────────────────

def llm_call(prompt: str, max_tokens: int = 3000, temperature: float = 0.1) -> Optional[str]:
    """One LLM call with retry on transient errors."""
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set"); return None

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt, wait in enumerate([0, 5, 15, 30]):
        if wait > 0:
            log.info(f"  Retry {attempt+1}/4 after {wait}s")
            time.sleep(wait)
        try:
            resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=300)
            if resp.status_code == 429:
                log.warning("429 rate-limit")
                continue
            if resp.status_code >= 500:
                log.warning(f"Server error {resp.status_code}")
                continue
            if resp.status_code != 200:
                log.warning(f"HTTP {resp.status_code}: {resp.text[:200]}")
                continue
            content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not content:
                log.warning("Empty response"); continue
            return content
        except requests.exceptions.Timeout:
            log.warning(f"Timeout attempt {attempt+1}")
        except Exception as e:
            log.warning(f"Error: {e}")
    log.error("All retries exhausted"); return None


# ── Fetch ───────────────────────────────────────────────────────────────────

def fetch_papers(target_date: date) -> list:
    end   = datetime.combine(target_date, datetime.min.time()) - timedelta(hours=6)
    start = end - timedelta(days=2)
    start_str, end_str = start.strftime("%Y%m%d%H%M"), end.strftime("%Y%m%d%H%M")
    log.info(f"Fetching {start_str} -> {end_str}")

    client = arxiv.Client()
    all_papers, seen = [], set()
    for cat in TARGET_CATEGORIES:
        query = f"cat:{cat} AND submittedDate:[{start_str} TO {end_str}]"
        try:
            search = arxiv.Search(query=query, max_results=15,
                                  sort_by=arxiv.SortCriterion.SubmittedDate)
            cnt = 0
            for r in client.results(search):
                if r.entry_id not in seen:
                    seen.add(r.entry_id)
                    all_papers.append({
                        "title":          r.title,
                        "summary":        r.summary.strip(),
                        "url":            r.entry_id,
                        "pdf_url":        r.pdf_url,
                        "published_date": r.published.isoformat(),
                        "categories":     r.categories,
                        "authors":        [a.name for a in r.authors],
                    })
                    cnt += 1
            log.info(f"  {cat}: +{cnt}")
        except Exception as e:
            log.warning(f"  {cat}: error - {e}")
    log.info(f"Total: {len(all_papers)} papers")
    return all_papers


# ── One-shot Evaluate (filter + rate + Chinese TLDR) ────────────────────────

UNIFIED_TMPL = """You are an expert AI research paper reviewer.

Research interests:
1. Audio Large Language Models (Audio LLM, Audio Foundation Models)
2. Audio/Speech Perception & Understanding
3. Audio/Speech Reasoning
4. Speech Synthesis (TTS, voice conversion, speech generation)
5. Omni Models (any-domain unified multimodal models including audio)

Evaluate each paper. Return a JSON array with one object per paper:

{{
  "idx": 1,
  "relevant": true/false,
  "tldr": "Chinese summary (must be Simplified Chinese, 2-3 sentences, start with 简要：或本文提出：)",
  "relevance_score": 8,
  "novelty_score": 7,
  "clarity_score": 8,
  "impact_score": 9,
  "overall_priority_score": 8
}}

IMPORTANT:
- "relevant" MUST be true only if the paper clearly matches at least one research interest above
- "tldr" MUST be in Simplified Chinese (简体中文) — write it yourself, do not copy from the abstract
- Only set relevant=true for papers that clearly match the topics
- For irrelevant papers, still output the JSON but set relevant=false and tldr=""
- Output EXACTLY one JSON object per paper below, no markdown fences

Papers:
{papers_block}

JSON array:"""

PAPERS_BLOCK_TMPL = """Paper {i}:
Title: {title}
Abstract: {abstract[:600]}"""


def evaluate_papers(papers: list) -> list:
    """Single LLM call: filter relevant papers + rate + generate Chinese TLDR."""
    if not OPENAI_API_KEY:
        log.warning("No API key"); return papers
    if not papers:
        return []

    block = "\n\n".join(
        PAPERS_BLOCK_TMPL.format(i=i+1, title=p["title"], abstract=p["summary"])
        for i, p in enumerate(papers)
    )
    prompt = UNIFIED_TMPL.format(papers_block=block)

    log.info(f"Sending {len(papers)} papers to LLM for evaluation...")
    resp = llm_call(prompt, max_tokens=4000, temperature=0.1)
    if not resp:
        log.error("LLM call failed"); return []

    # Parse JSON from response
    try:
        cleaned = resp
        for fence in ["```json", "```"]:
            if fence in cleaned:
                cleaned = cleaned.split(fence)[1].split("```")[0]
        items = json.loads(cleaned.strip())
        log.info(f"LLM raw response type: {type(items).__name__}, keys: {list(items.keys()) if isinstance(items, dict) else 'N/A (list)'}")
        # Handle nested: {"success":true,"data":[...]} or {"papers":[...]} etc
        if isinstance(items, dict):
            for key in ["papers", "results", "data", "items", "evaluations"]:
                if key in items:
                    items = items[key]
                    log.info(f"  Extracted nested key '{key}': {type(items).__name__}")
                    break
        # Single object -> list
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            raise ValueError(f"Expected list, got {type(items).__name__}: {str(items)[:200]}")
        # Extract idx field from each item
        smap = {}
        for item in items:
            if isinstance(item, dict) and "idx" in item:
                smap[item["idx"]] = item
        log.info(f"  Parsed {len(smap)} paper entries")
    except (json.JSONDecodeError, ValueError) as e:
        log.error(f"JSON parse error: {e}")
        log.info(f"Response (first 1000): {resp[:1000]}")
        return []

    rated = []
    for i, p in enumerate(papers):
        info = smap.get(i + 1, {})
        rel  = info.get("relevant", False)
        tldr = info.get("tldr", "")

        if not rel:
            log.info(f"  [{i+1}/{len(papers)}] SKIP {p['title'][:55]}")
            continue

        p["tldr"]                  = tldr
        p["relevance_score"]       = info.get("relevance_score", 5)
        p["novelty_score"]         = info.get("novelty_score", 5)
        p["clarity_score"]         = info.get("clarity_score", 5)
        p["impact_score"]          = info.get("impact_score", 5)
        p["overall_priority_score"] = info.get("overall_priority_score", 5)

        has_zh = any('\u4e00' <= c <= '\u9fff' for c in tldr)
        zh_mark = "ZH" if has_zh else "EN"
        log.info(f"  [{i+1}/{len(papers)}] STAR{ p['overall_priority_score']}/10 [{zh_mark}] {p['title'][:45]}")
        rated.append(p)

    log.info(f"Relevant: {len(rated)}/{len(papers)} papers")
    return rated


# ── Top 10 Cited ────────────────────────────────────────────────────────────

TOP_PROMPT = """Identify the TOP 10 most cited AI/ML papers from {prev_year}-{curr_year} related to:
1. Audio Large Language Models / Audio Foundation Models
2. Audio/Speech Perception & Understanding
3. Speech Synthesis (TTS, voice conversion)
4. Omni Models (unified multimodal models)

Output ONLY valid JSON array (no markdown, no explanation):
[
  {{"rank":1,"title":"<exact title>","authors":"<first author et al.>","year":{prev_year},"citations":"<number>","url":"<arxiv URL>","summary":"<1-sentence English summary>"}},
  ...9 more, sorted by citations descending
]

Only papers from {prev_year}-{curr_year}. If uncertain, provide best estimate."""


def get_top_cited() -> list:
    if not OPENAI_API_KEY:
        return []
    today = date.today()
    prompt = TOP_PROMPT.format(prev_year=today.year - 1, curr_year=today.year)
    resp = llm_call(prompt, max_tokens=2500, temperature=0.3)
    if not resp:
        log.warning("Top-cited: no response"); return []
    try:
        cleaned = resp
        for fence in ["```json", "```"]:
            if fence in cleaned:
                cleaned = cleaned.split(fence)[1].split("```")[0]
        data = json.loads(cleaned.strip())
        log.info(f"Top-cited response: {type(data).__name__}, keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")
        if isinstance(data, dict):
            for key in ["papers", "results", "data", "items", "top_papers"]:
                if key in data:
                    data = data[key]
                    log.info(f"  Extracted '{key}': {type(data).__name__}")
                    break
        if isinstance(data, dict):
            data = [data]
        return data if isinstance(data, list) else [data]
    except (json.JSONDecodeError, ValueError) as e:
        log.error(f"Top-cited parse error: {e}")
        log.info(f"Response: {resp[:500]}")
        return [resp.strip()]


# ── Cleanup ──────────────────────────────────────────────────────────────────

def cleanup(directory: str, days: int):
    if not os.path.exists(directory):
        return
    cutoff = datetime.now() - timedelta(days=days)
    removed = 0
    for fn in os.listdir(directory):
        fp = os.path.join(directory, fn)
        if os.path.isfile(fp) and datetime.fromtimestamp(os.path.getmtime(fp)) < cutoff:
            os.remove(fp); removed += 1
    log.info(f"Cleanup: removed {removed} from {directory}")


# ── HTML Generation ─────────────────────────────────────────────────────────

def generate_html(date_str: str, papers: list, top_cited: list):
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        log.warning("jinja2 not available"); return

    env  = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    tmpl = env.get_template(TEMPLATE_NAME)

    try:
        rdate = date.fromisoformat(date_str)
        fmt   = rdate.strftime("%Y_%m_%d")
        title = f"Audio/Speech AI Papers - {rdate.strftime('%B %d, %Y')}"
    except ValueError:
        fmt   = date.today().strftime("%Y_%m_%d")
        title = "Audio/Speech AI Daily"

    today_yr = date.today().year
    html = tmpl.render(
        papers=papers, title=title,
        report_date=rdate,
        generation_time=datetime.now(timezone.utc),
        top_cited_papers=top_cited,
        prev_year=today_yr - 1,
        curr_year=today_yr,
    )
    out = os.path.join(HTML_DIR, f"{fmt}.html")
    os.makedirs(HTML_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"Generated: {out}")


# ── Reports ──────────────────────────────────────────────────────────────────

def update_reports():
    files = sorted(
        [f for f in os.listdir(HTML_DIR) if f.endswith(".html")],
        reverse=True,
    )
    with open(REPORTS_PATH, "w") as f:
        json.dump(files, f, indent=4)
    log.info(f"reports.json: {len(files)} reports - {files}")


# ── Process one date ─────────────────────────────────────────────────────────

def process(target_date: date, force: bool = False):
    log.info(f"{'='*60}")
    log.info(f"  Processing: {target_date}")
    log.info(f"{'='*60}")
    jpath = os.path.join(JSON_DIR, f"{target_date.isoformat()}.json")
    os.makedirs(JSON_DIR, exist_ok=True)

    if os.path.exists(jpath) and not force:
        log.info("Loading from cache")
        with open(jpath) as f:
            papers = json.load(f)
    else:
        raw     = fetch_papers(target_date)
        papers  = evaluate_papers(raw)
        papers.sort(key=lambda x: x.get("overall_priority_score", 0), reverse=True)
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=4, ensure_ascii=False)
        log.info(f"Saved {jpath} ({len(papers)} papers)")

    is_today  = (target_date == date.today())
    top_cited = get_top_cited() if is_today else []

    generate_html(target_date.isoformat(), papers, top_cited)
    if is_today:
        update_reports()
    return papers


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    log.info(f"Model: {OPENAI_MODEL}")
    log.info(f"API: {OPENAI_API_URL}")
    log.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
    log.info(f"API key present: {bool(OPENAI_API_KEY)}")

    cleanup(JSON_DIR, DATA_RETENTION)
    cleanup(HTML_DIR, DATA_RETENTION)

    target = date.fromisoformat(args.date) if args.date else date.today()
    process(target, force=args.force)
    log.info("Pipeline complete!")
