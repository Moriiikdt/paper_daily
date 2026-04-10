#!/usr/bin/env python3
"""
Unified fetch + filter + rate + generate pipeline.
Uses BATCHED LLM calls to avoid 4000+ sequential API calls.
Keeps only 2 months of data.
"""

import os
import sys
import json
import time
import logging
import arxiv
import requests
from datetime import date, datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("fetch_and_filter")

# ── Config ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE   = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
OPENAI_API_URL    = f"{OPENAI_API_BASE}/chat/completions"
OPENAI_MODEL      = "gpt-5.4"

PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR          = os.path.join(PROJECT_ROOT, "daily_json")
HTML_DIR          = os.path.join(PROJECT_ROOT, "daily_html")
TEMPLATE_DIR      = os.path.join(PROJECT_ROOT, "templates")
TEMPLATE_NAME     = "paper_template.html"
REPORTS_PATH      = os.path.join(PROJECT_ROOT, "reports.json")
DATA_RETENTION_DAYS = 60   # 2 months

# arXiv categories to search
TARGET_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG", "cs.SD", "eess.AS", "cs.MM", "cs.CV"]

# Research topics for filtering
RESEARCH_TOPICS = """The user is interested in papers related to:
1. Audio Large Language Models (Audio LLM, Audio Foundation Models)
2. Audio Large Model Reasoning (Reasoning over audio, Audio chain-of-thought)
3. Audio/Speech Perception and Understanding (Speech recognition, Audio understanding, Multimodal audio perception)
4. Speech Synthesis (Text-to-Speech, Voice conversion, Speech generation)
5. Omni Models (any-domain Omni models, unified multimodal models that include audio)"""

# ── LLM helpers ──────────────────────────────────────────────────────────────

def llm_call(prompt: str, max_tokens: int = 5, temperature: float = 0.1) -> str | None:
    """Single LLM API call."""
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set")
        return None
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        resp = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning(f"API error: {e}")
        return None


def batch_llm_calls(prompts: list[str], max_tokens: int = 800, temperature: float = 0.1) -> list[str | None]:
    """
    Call LLM with a batch of prompts in one request using a system prompt that
    wraps all items. Falls back to individual calls if batch is too large.
    """
    if not prompts:
        return []
    # Try batching everything at once
    combined = "\n\n---\n\n".join(f"[{i+1}]\n{p}" for i, p in enumerate(prompts))
    result = llm_call(combined, max_tokens=max_tokens, temperature=temperature)
    if result:
        return [result] * len(prompts)  # Not useful for structured output
    return [None] * len(prompts)


# ── Step 1: Fetch papers from arXiv ─────────────────────────────────────────

def fetch_papers(specified_date: date) -> list:
    """Fetch papers from multiple arXiv categories for a date range."""
    # Fetch papers from the past 3 days to catch papers across time zones
    end_dt   = datetime.combine(specified_date, datetime.min.time()) - timedelta(hours=6)
    start_dt = end_dt - timedelta(days=3)
    start_str = start_dt.strftime("%Y%m%d%H%M")
    end_str   = end_dt.strftime("%Y%m%d%H%M")

    log.info(f"Fetching papers from {start_str} to {end_str}")
    client = arxiv.Client()
    all_papers = []
    seen_ids = set()

    for cat in TARGET_CATEGORIES:
        query = f"cat:{cat} AND submittedDate:[{start_str} TO {end_str}]"
        try:
            search = arxiv.Search(query=query, max_results=300, sort_by=arxiv.SortCriterion.SubmittedDate)
            count = 0
            for result in client.results(search):
                if result.entry_id not in seen_ids:
                    seen_ids.add(result.entry_id)
                    all_papers.append({
                        "title":        result.title,
                        "summary":      result.summary.strip(),
                        "url":          result.entry_id,
                        "published_date": result.published.isoformat(),
                        "updated_date":   result.updated.isoformat(),
                        "categories":   result.categories,
                        "authors":      [a.name for a in result.authors],
                    })
                    count += 1
            log.info(f"  {cat}: +{count} papers")
        except Exception as e:
            log.warning(f"Error fetching {cat}: {e}")

    log.info(f"Total unique papers: {len(all_papers)}")
    return all_papers


# ── Step 2: Filter by topic (batched) ───────────────────────────────────────

def filter_papers(papers: list) -> list:
    """Filter papers by research topic using batched LLM calls."""
    if not OPENAI_API_KEY:
        log.warning("No API key — skipping filter, keeping all papers")
        return papers
    if not papers:
        return []

    # Batch into groups of 5 papers per LLM call
    BATCH = 5
    filtered = []

    for i in range(0, len(papers), BATCH):
        batch = papers[i:i+BATCH]
        lines = []
        for j, p in enumerate(batch):
            idx = i + j + 1
            lines.append(f"Paper {idx}: {p['title']}\nAbstract: {p['summary'][:800]}")

        prompt = f"""{RESEARCH_TOPICS}

Evaluate each paper and reply with ONLY the numbers of papers that match, separated by commas.
For example: 1, 3, 5

{chr(10).join(lines)}"""

        resp = llm_call(prompt, max_tokens=50)
        if not resp:
            log.warning(f"  Batch {i//BATCH+1}: API failed, keeping all papers in batch")
            filtered.extend(batch)
            continue

        matched = set()
        for token in resp.replace(" ", "").split(","):
            try:
                matched.add(i + int(token.strip()) - 1)
            except ValueError:
                pass

        for j, p in enumerate(batch):
            status = "✓" if (i+j) in matched else "✗"
            log.info(f"  [{i+j+1}/{len(papers)}] {status} {p['title'][:50]}...")
        filtered.extend(p for j, p in enumerate(batch) if (i+j) in matched)

    log.info(f"Filter result: {len(filtered)}/{len(papers)} papers matched")
    return filtered


# ── Step 3: Rate papers (batched) ───────────────────────────────────────────

RATING_PROMPT_TEMPLATE = """You are an expert reviewer. For each paper below, return a JSON object with your assessment.

Research interests: Audio LLM, Audio/Speech Perception & Understanding, Speech Synthesis, Omni Models

{papers_block}

Output a single JSON array (no markdown) with one object per paper, in the same order:
[
  {{"idx": 1, "tldr": "...", "relevance_score": 8, "novelty_score": 7, "clarity_score": 8, "impact_score": 9, "overall_priority_score": 8}},
  ...
]
Scores are 1-10 integers.
TLDR must be in English only.
"""

RATING_ITEMS_TEMPLATE = 'Paper {i}: Title: {title}\nAbstract: {abstract}'


def rate_papers(papers: list) -> list:
    """Rate papers in batches of 3 using structured LLM output."""
    if not OPENAI_API_KEY:
        log.warning("No API key — skipping rating")
        return papers
    if not papers:
        return []

    BATCH = 3
    rated = []

    for i in range(0, len(papers), BATCH):
        batch = papers[i:i+BATCH]
        papers_block = "\n\n".join(
            RATING_ITEMS_TEMPLATE.format(
                i=i + j + 1,
                title=p["title"],
                abstract=p["summary"][:600],
            )
            for j, p in enumerate(batch)
        )
        prompt = RATING_PROMPT_TEMPLATE.format(papers_block=papers_block)

        resp = llm_call(prompt, max_tokens=1500, temperature=0.1)
        if not resp:
            log.warning(f"  Rating batch {i//BATCH+1}: API failed, copying paper data")
            for p in batch:
                p.setdefault("tldr", "")
                p.setdefault("relevance_score", 5)
                p.setdefault("novelty_score", 5)
                p.setdefault("clarity_score", 5)
                p.setdefault("impact_score", 5)
                p.setdefault("overall_priority_score", 5)
            rated.extend(batch)
            continue

        try:
            # Strip markdown fences
            cleaned = resp
            for fence in ["```json", "```"]:
                if fence in cleaned:
                    cleaned = cleaned.split(fence)[1].split("```")[0]
            items = json.loads(cleaned)
            score_map = {item["idx"]: item for item in items}
            for j, p in enumerate(batch):
                info = score_map.get(i + j + 1, {})
                p.setdefault("tldr", info.get("tldr", ""))
                p["relevance_score"]       = info.get("relevance_score", 5)
                p["novelty_score"]         = info.get("novelty_score", 5)
                p["clarity_score"]         = info.get("clarity_score", 5)
                p["impact_score"]          = info.get("impact_score", 5)
                p["overall_priority_score"] = info.get("overall_priority_score", 5)
                log.info(f"  [{i+j+1}/{len(papers)}] score={p['overall_priority_score']}/10 — {p['title'][:40]}...")
        except (json.JSONDecodeError, KeyError) as e:
            log.warning(f"  Rating batch {i//BATCH+1}: parse error ({e}), copying defaults")
            for p in batch:
                p.setdefault("tldr", "")
                p.setdefault("overall_priority_score", 5)
            rated.extend(batch)
            continue

        rated.extend(batch)

    return rated


# ── Step 4: Top cited papers summary ──────────────────────────────────────

TOP_CITED_PROMPT = """# Task
Identify the TOP 10 most cited papers from 2025-2026 related to:
1. Audio Large Language Models (Audio LLM, Audio Foundation Models)
2. Audio/Speech Perception and Understanding
3. Speech Synthesis (TTS, Voice Conversion)
4. Omni Models (unified multimodal models including audio)

# Output Format (strict JSON array, no markdown)
[
  {{"rank": 1, "title": "...", "authors": "...", "year": 2025, "citations": 1200, "venue": "arXiv / NeurIPS 2025", "url": "https://...", "summary": "..."}},
  ... (9 more)
]

Requirements:
- Only 2025 or 2026 papers
- Most cited first
- Realistic citation counts
- High confidence only
"""


def get_top_cited() -> list:
    """Fetch top 10 most cited papers from recent years."""
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set")
        return []

    log.info("Fetching top cited papers summary...")
    resp = llm_call(TOP_CITED_PROMPT, max_tokens=2500, temperature=0.3)
    if not resp:
        log.warning("Top cited: no response from API")
        return []

    try:
        cleaned = resp
        for fence in ["```json", "```"]:
            if fence in cleaned:
                cleaned = cleaned.split(fence)[1].split("```")[0]
        papers = json.loads(cleaned)
        log.info(f"Retrieved {len(papers)} top papers")
        return papers
    except json.JSONDecodeError as e:
        log.error(f"Top cited JSON parse error: {e}")
        return []


# ── Step 5: Cleanup old files ──────────────────────────────────────────────

def cleanup_old_files(directory: str, days: int):
    if not os.path.exists(directory):
        return
    cutoff = datetime.now() - timedelta(days=days)
    removed = 0
    for fn in os.listdir(directory):
        fp = os.path.join(directory, fn)
        if os.path.isfile(fp) and datetime.fromtimestamp(os.path.getmtime(fp)) < cutoff:
            os.remove(fp)
            removed += 1
    log.info(f"Cleanup: removed {removed} files from {directory}")


# ── Step 6: Generate HTML ───────────────────────────────────────────────────

def generate_html(date_str: str, papers: list, top_cited: list):
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        log.warning("jinja2 not installed — skipping HTML generation")
        return

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template(TEMPLATE_NAME)

    try:
        rdate = date.fromisoformat(date_str)
        formatted = rdate.strftime("%Y_%m_%d")
        page_title = f"Audio/Speech AI Papers — {rdate.strftime('%B %d, %Y')}"
    except ValueError:
        formatted = date.today().strftime("%Y_%m_%d")
        page_title = "Audio/Speech AI Papers"

    html = template.render(
        papers=papers,
        title=page_title,
        report_date=rdate,
        generation_time=datetime.now(timezone.utc),
        top_cited_papers=top_cited,
    )

    out = os.path.join(HTML_DIR, f"{formatted}.html")
    os.makedirs(HTML_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"Generated: {out}")


# ── Step 7: Update reports.json ────────────────────────────────────────────

def update_reports():
    html_files = sorted(
        [f for f in os.listdir(HTML_DIR) if f.endswith(".html")],
        reverse=True,
    )
    with open(REPORTS_PATH, "w", encoding="utf-8") as f:
        json.dump(html_files, f, indent=4)
    log.info(f"reports.json updated: {len(html_files)} reports")


# ── Main pipeline ───────────────────────────────────────────────────────────

def process_date(target_date: date, force: bool = False):
    log.info(f"\n{'='*60}\nProcessing: {target_date.isoformat()}\n{'='*60}")

    json_path = os.path.join(JSON_DIR, f"{target_date.isoformat()}.json")
    os.makedirs(JSON_DIR, exist_ok=True)

    if os.path.exists(json_path) and not force:
        log.info(f"JSON exists — skipping fetch for {target_date}")
        with open(json_path, "r") as f:
            papers = json.load(f)
    else:
        raw = fetch_papers(target_date)
        if not raw:
            log.warning(f"No papers fetched for {target_date}")
            with open(json_path, "w") as f:
                json.dump([], f)
            papers = []
        else:
            filtered = filter_papers(raw)
            papers   = rate_papers(filtered) if filtered else []
            papers.sort(key=lambda x: x.get("overall_priority_score", 0), reverse=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(papers, f, indent=4, ensure_ascii=False)
            log.info(f"Saved: {json_path} ({len(papers)} papers)")

    # Generate HTML only for the most recent date (today)
    if target_date == date.today():
        top_cited = get_top_cited()
        generate_html(target_date.isoformat(), papers, top_cited)
        update_reports()
    else:
        # Still generate HTML for historical dates (they may be processed on day they're published)
        generate_html(target_date.isoformat(), papers, [])

    return papers


def main():
    target = date.today()

    # Cleanup
    cleanup_old_files(JSON_DIR, DATA_RETENTION_DAYS)
    cleanup_old_files(HTML_DIR, DATA_RETENTION_DAYS)

    # Process yesterday and today (catch papers across time zones)
    for d in [target - timedelta(days=1), target]:
        process_date(d)

    log.info("\nPipeline complete!")


if __name__ == "__main__":
    main()
