#!/usr/bin/env python3
"""Unified fetch + filter + rate + generate pipeline."""

import os, sys, json, time, logging, arxiv, requests
from datetime import date, datetime, timedelta, timezone
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("pipeline")

# ── Config ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
# The gateway needs /v1/ prefix
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
OPENAI_API_URL      = f"{OPENAI_API_BASE_URL}/chat/completions"
OPENAI_MODEL        = os.getenv("OPENAI_MODEL", "gpt-5.4")

PROJECT_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR         = os.path.join(PROJECT_ROOT, "daily_json")
HTML_DIR         = os.path.join(PROJECT_ROOT, "daily_html")
TEMPLATE_DIR     = os.path.join(PROJECT_ROOT, "templates")
TEMPLATE_NAME    = "paper_template.html"
REPORTS_PATH     = os.path.join(PROJECT_ROOT, "reports.json")
DATA_RETENTION   = 60   # days

TARGET_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG", "cs.SD", "eess.AS", "cs.MM", "cs.CV"]

RESEARCH_TOPICS = """The user is interested in papers related to:
1. Audio Large Language Models (Audio LLM, Audio Foundation Models)
2. Audio Large Model Reasoning (Reasoning over audio, Audio chain-of-thought)
3. Audio/Speech Perception and Understanding (Speech recognition, Audio understanding, Multimodal audio perception)
4. Speech Synthesis (Text-to-Speech, Voice conversion, Speech generation)
5. Omni Models (any-domain Omni models, unified multimodal models that include audio)"""

# ── LLM caller ────────────────────────────────────────────────────────────────

def llm_call(prompt: str,
             max_tokens: int = 200,
             temperature: float = 0.1,
             retry_backoff: tuple = (1, 2, 4, 8)) -> Optional[str]:
    """Call LLM with exponential-backoff retry on transient errors."""
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set"); return None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt, wait in enumerate([0] + list(retry_backoff)):
        if wait > 0:
            log.info(f"  Retry {attempt+1} after {wait}s...")
            time.sleep(wait)
        try:
            resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                log.warning("  429 rate limit")
                continue
            if resp.status_code >= 500:
                log.warning(f"  Server error {resp.status_code}")
                continue
            if resp.status_code != 200:
                log.warning(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                return None
            result = resp.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not content:
                log.warning("  Empty response")
                return None
            return content
        except requests.exceptions.Timeout:
            log.warning(f"  Timeout")
        except Exception as e:
            log.warning(f"  Error: {e}")
    log.error("All retries exhausted")
    return None


# ── Step 1: Fetch papers ─────────────────────────────────────────────────────

def fetch_papers(specified_date: date) -> list:
    end_dt   = datetime.combine(specified_date, datetime.min.time()) - timedelta(hours=6)
    start_dt = end_dt - timedelta(days=3)
    start_str, end_str = start_dt.strftime("%Y%m%d%H%M"), end_dt.strftime("%Y%m%d%H%M")
    log.info(f"Fetching {start_str} -> {end_str}")

    client = arxiv.Client()
    all_papers, seen = [], set()
    for cat in TARGET_CATEGORIES:
        query = f"cat:{cat} AND submittedDate:[{start_str} TO {end_str}]"
        try:
            search = arxiv.Search(query=query, max_results=300,
                                   sort_by=arxiv.SortCriterion.SubmittedDate)
            cnt = 0
            for r in client.results(search):
                if r.entry_id not in seen:
                    seen.add(r.entry_id)
                    all_papers.append({
                        "title":          r.title,
                        "summary":        r.summary.strip(),
                        "url":            r.entry_id,
                        "published_date": r.published.isoformat(),
                        "updated_date":   r.updated.isoformat(),
                        "categories":     r.categories,
                        "authors":        [a.name for a in r.authors],
                    })
                    cnt += 1
            log.info(f"  {cat}: +{cnt}")
        except Exception as e:
            log.warning(f"  {cat}: {e}")
    log.info(f"Total unique: {len(all_papers)}")
    return all_papers


# ── Step 2: Filter (batches of 5) ─────────────────────────────────────────────

def filter_papers(papers: list) -> list:
    if not OPENAI_API_KEY: return papers
    if not papers: return []
    BATCH = 5
    filtered = []
    for i in range(0, len(papers), BATCH):
        batch = papers[i : i + BATCH]
        lines = [
            f"Paper {i+j+1}: {p['title']}\nAbstract: {p['summary'][:600]}"
            for j, p in enumerate(batch)
        ]
        prompt = f"""{RESEARCH_TOPICS}

Reply ONLY with the numbers of papers that match, separated by commas.
Example: 1, 3, 5

{chr(10).join(lines)}"""
        resp = llm_call(prompt, max_tokens=30)
        matched = set()
        if resp:
            for tok in resp.replace(" ", "").split(","):
                try:
                    matched.add(i + int(tok.strip()) - 1)
                except ValueError:
                    pass
        for j, p in enumerate(batch):
            idx = i + j
            ok = idx in matched
            log.info(f"  [{idx+1}/{len(papers)}] {'✓' if ok else '✗'} {p['title'][:50]}...")
            if ok:
                filtered.append(p)
        time.sleep(0.5)
    log.info(f"Filter: {len(filtered)}/{len(papers)} passed")
    return filtered


# ── Step 3: Rate (batches of 3, max 30 papers) ───────────────────────────────

RATING_TMPL = """You are an expert reviewer. Output a JSON array with one object per paper.

Research interests: Audio LLM, Audio/Speech Perception & Understanding, Speech Synthesis, Omni Models

{papers_block}

Single JSON array (no markdown):
[
  {{"idx":1,"tldr":"...","relevance_score":8,"novelty_score":7,"clarity_score":8,"impact_score":9,"overall_priority_score":8}},
  ... (one per paper)
]
TLDR must be in English only. Scores are integers 1-10."""

ITEMS_TMPL = "Paper {i}: Title: {title}\nAbstract: {abstract}"


def rate_papers(papers: list) -> list:
    if not OPENAI_API_KEY: return papers
    if not papers: return []
    papers = papers[:30]   # cap to avoid runaway API calls
    BATCH = 3
    rated = []
    for i in range(0, len(papers), BATCH):
        batch = papers[i : i + BATCH]
        block = "\n\n".join(
            ITEMS_TMPL.format(i=i+j+1, title=p["title"], abstract=p["summary"][:500])
            for j, p in enumerate(batch)
        )
        resp = llm_call(RATING_TMPL.format(papers_block=block), max_tokens=1500)
        if resp:
            try:
                cleaned = resp
                for fence in ["```json", "```"]:
                    if fence in cleaned:
                        cleaned = cleaned.split(fence)[1].split("```")[0]
                items = json.loads(cleaned)
                smap  = {item["idx"]: item for item in items}
                ok = True
            except (json.JSONDecodeError, KeyError) as e:
                log.warning(f"  Batch {(i//BATCH)+1}: parse error ({e})")
                ok = False
            if ok:
                for j, p in enumerate(batch):
                    info = smap.get(i + j + 1, {})
                    p.setdefault("tldr", info.get("tldr", ""))
                    p["relevance_score"]        = info.get("relevance_score", 5)
                    p["novelty_score"]          = info.get("novelty_score", 5)
                    p["clarity_score"]          = info.get("clarity_score", 5)
                    p["impact_score"]           = info.get("impact_score", 5)
                    p["overall_priority_score"] = info.get("overall_priority_score", 5)
                    log.info(f"  [{i+j+1}/{len(papers)}] score={p['overall_priority_score']}/10 "
                             f"{p['title'][:40]}...")
            else:
                for p in batch:
                    p.setdefault("tldr", ""); p.setdefault("overall_priority_score", 5)
        else:
            for p in batch:
                p.setdefault("tldr", ""); p.setdefault("overall_priority_score", 5)
        rated.extend(batch)
        time.sleep(1.5)
    return rated


# ── Step 4: Top cited papers ────────────────────────────────────────────────

TOP_CITED = """# Identify TOP 10 most cited papers from 2025-2026 related to:
1. Audio Large Language Models / Audio Foundation Models
2. Audio/Speech Perception and Understanding
3. Speech Synthesis (TTS, Voice Conversion)
4. Omni Models (unified multimodal models including audio)

Output a strict JSON array (no markdown):
[
  {{"rank":1,"title":"...","authors":"...","year":2025,"citations":1200,"venue":"arXiv","url":"https://...","summary":"..."}},
  ... (9 more)
]
Only 2025 or 2026 papers. Most cited first. Summaries in English."""


def get_top_cited() -> list:
    if not OPENAI_API_KEY: return []
    log.info("Fetching top cited papers...")
    resp = llm_call(TOP_CITED, max_tokens=2500, temperature=0.3,
                    retry_backoff=(2, 4, 8, 16))
    if not resp:
        log.warning("Top cited: no response"); return []
    try:
        cleaned = resp
        for fence in ["```json", "```"]:
            if fence in cleaned:
                cleaned = cleaned.split(fence)[1].split("```")[0]
        papers = json.loads(cleaned)
        log.info(f"Top cited: {len(papers)} retrieved")
        return papers
    except json.JSONDecodeError as e:
        log.error(f"Top cited parse error: {e}")
        return []


# ── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup(directory: str, days: int):
    if not os.path.exists(directory): return
    cutoff = datetime.now() - timedelta(days=days)
    removed = 0
    for fn in os.listdir(directory):
        fp = os.path.join(directory, fn)
        if os.path.isfile(fp) and datetime.fromtimestamp(os.path.getmtime(fp)) < cutoff:
            os.remove(fp); removed += 1
    log.info(f"Cleanup: removed {removed} from {directory}")


# ── HTML generation ──────────────────────────────────────────────────────────

def generate_html(date_str: str, papers: list, top_cited: list):
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        log.warning("jinja2 not available"); return
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    tmpl = env.get_template(TEMPLATE_NAME)
    try:
        rdate = date.fromisoformat(date_str)
        fmt   = rdate.strftime("%Y_%m_%d")
        title = f"Audio/Speech AI Papers - {rdate.strftime('%B %d, %Y')}"
    except ValueError:
        fmt   = date.today().strftime("%Y_%m_%d")
        title = "Audio/Speech AI Papers"
    html = tmpl.render(
        papers=papers, title=title, report_date=rdate,
        generation_time=datetime.now(timezone.utc),
        top_cited_papers=top_cited,
    )
    out = os.path.join(HTML_DIR, f"{fmt}.html")
    os.makedirs(HTML_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f: f.write(html)
    log.info(f"Generated: {out}")


# ── Index & list pages ───────────────────────────────────────────────────────

def generate_index_and_list():
    files = sorted([f for f in os.listdir(HTML_DIR) if f.endswith(".html")], reverse=True)
    list_html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Audio AI Daily</title>
<style>
  body{{font-family:system-ui;max-width:800px;margin:40px auto;padding:0 20px}}
  h1{{color:#0f172a}}a{{color:#14b8a6;text-decoration:none}}a:hover{{text-decoration:underline}}
  li{{margin:8px 0}}
</style>
</head>
<body>
  <h1>Audio/Speech AI Daily - Historical Reports</h1>
  <p><a href="index.html">Latest report</a></p>
  <ul>
    {"".join(f'<li><a href="./daily_html/{f}">{f}</a></li>' for f in files)}
  </ul>
</body>
</html>"""
    with open(os.path.join(PROJECT_ROOT, "list.html"), "w") as f:
        f.write(list_html)
    index_html = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Audio/Speech AI Daily</title>
<style>
  html,body{{margin:0;padding:0;height:100%;display:flex;flex-direction:column;font-family:system-ui}}
  #wrapper{{flex:1;overflow:hidden;position:relative}}
  iframe{{border:none;width:100%;height:100%;display:block}}
  #footer{{flex-shrink:0;height:40px;line-height:40px;text-align:center;border-top:1px solid #eee}}
  #footer a{{color:#14b8a6;text-decoration:none}}
  #loading,#error{{padding:40px;text-align:center;color:#64748b}}
</style>
<script>
  async function load(){{
    try{{
      const r = await fetch("reports.json");
      const files = await r.json();
      if(files && files.length){{
        document.getElementById("report").src = "./daily_html/" + files[0];
        document.getElementById("loading").style.display = "none";
        document.getElementById("report").style.display = "block";
      }}
    }}catch(e){{
      document.getElementById("loading").innerHTML = "No reports yet. <a href='list.html'>View list</a>";
    }}
  }}
  window.onload = load;
</script>
</head>
<body>
  <div id="wrapper">
    <div id="loading">Loading latest report...</div>
    <iframe id="report" style="display:none"></iframe>
  </div>
  <div id="footer"><a href="list.html">Historical Reports</a></div>
</body>
</html>"""
    with open(os.path.join(PROJECT_ROOT, "index.html"), "w") as f:
        f.write(index_html)
    log.info("Generated index.html and list.html")


def update_reports_and_pages():
    files = sorted([f for f in os.listdir(HTML_DIR) if f.endswith(".html")], reverse=True)
    with open(REPORTS_PATH, "w") as f: json.dump(files, f, indent=4)
    log.info(f"reports.json: {len(files)} reports")
    generate_index_and_list()


# ── Main ─────────────────────────────────────────────────────────────────────

def process(target_date: date, force: bool = False):
    log.info(f"\n{'='*60}\n{target_date}\n{'='*60}")
    jpath = os.path.join(JSON_DIR, f"{target_date.isoformat()}.json")
    os.makedirs(JSON_DIR, exist_ok=True)
    if os.path.exists(jpath) and not force:
        log.info("JSON exists - loading")
        with open(jpath) as f: papers = json.load(f)
    else:
        raw = fetch_papers(target_date)
        if not raw:
            with open(jpath, "w") as f: json.dump([], f)
            papers = []
        else:
            filtered = filter_papers(raw)
            papers   = rate_papers(filtered) if filtered else []
            papers.sort(key=lambda x: x.get("overall_priority_score", 0), reverse=True)
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(papers, f, indent=4, ensure_ascii=False)
            log.info(f"Saved {jpath} ({len(papers)} papers)")
    is_today = (target_date == date.today())
    top_cited = get_top_cited() if is_today else []
    generate_html(target_date.isoformat(), papers, top_cited)
    if is_today: update_reports_and_pages()
    return papers


def main():
    log.info(f"Model: {OPENAI_MODEL} | API: {OPENAI_API_URL}")
    cleanup(JSON_DIR, DATA_RETENTION)
    cleanup(HTML_DIR, DATA_RETENTION)
    today = date.today()
    for d in [today - timedelta(days=1), today]:
        process(d)
    log.info("Pipeline complete!")

if __name__ == "__main__":
    main()
