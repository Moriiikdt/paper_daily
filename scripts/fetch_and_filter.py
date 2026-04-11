#!/usr/bin/env python3
"""
Unified fetch + filter + rate + generate pipeline.
Daily-only: fetches today's papers, generates HTML report.
Top-10 cited papers from past year are included once per run.
"""

import os, sys, json, time, logging, arxiv, requests
from datetime import date, datetime, timedelta, timezone
import argparse
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pipeline")

# ── API Config ────────────────────────────────────────────────────────────────
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
OPENAI_API_URL      = f"{OPENAI_API_BASE}/chat/completions"
OPENAI_MODEL        = os.getenv("OPENAI_MODEL", "gpt-5.4")

PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR        = os.path.join(PROJECT_ROOT, "daily_json")
HTML_DIR        = os.path.join(PROJECT_ROOT, "daily_html")
TEMPLATE_DIR    = os.path.join(PROJECT_ROOT, "templates")
TEMPLATE_NAME   = "paper_template.html"
REPORTS_PATH    = os.path.join(PROJECT_ROOT, "reports.json")
DATA_RETENTION  = 60   # days

TARGET_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG", "cs.SD", "eess.AS", "cs.MM", "cs.CV"]

RESEARCH_TOPICS = """The user is interested in papers related to:
1. Audio Large Language Models (Audio LLM, Audio Foundation Models)
2. Audio/Speech Perception and Understanding (speech recognition, audio understanding, multimodal perception)
3. Audio/Speech Reasoning (reasoning over audio, audio chain-of-thought)
4. Speech Synthesis (TTS, voice conversion, speech generation)
5. Omni Models (any-domain unified multimodal models including audio)"""

# ── LLM caller with retry + backoff ─────────────────────────────────────────

def llm_call(prompt: str,
             max_tokens: int = 200,
             temperature: float = 0.1,
             retry_backoff: tuple = (3, 6, 12, 24, 48)) -> Optional[str]:
    """Call LLM with exponential-backoff retry on transient errors AND empty responses."""
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
            log.info(f"  ↻ Retry {attempt+1}/{len(retry_backoff)+1} after {wait}s…")
            time.sleep(wait)
        try:
            resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                log.warning("  ⚠ 429 rate-limit, backing off")
                continue
            if resp.status_code >= 500:
                log.warning(f"  ⚠ Server error {resp.status_code}")
                continue
            if resp.status_code != 200:
                log.warning(f"  ⚠ HTTP {resp.status_code}: {resp.text[:200]}")
                continue  # ← retry on non-200
            result = resp.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not content:
                log.warning(f"  ⚠ Empty response on attempt {attempt+1}")
                continue  # ← retry on empty (key fix for Chinese TLDR)
            return content
        except requests.exceptions.Timeout:
            log.warning(f"  ⚠ Timeout on attempt {attempt+1}")
        except Exception as e:
            log.warning(f"  ⚠ Error: {e}")
    log.error("All retries exhausted")
    return None


# ── Step 1: Fetch papers ─────────────────────────────────────────────────────

def fetch_papers(specified_date: date) -> list:
    """Fetch papers from arXiv for the given date (searches ±1 day window)."""
    end_dt   = datetime.combine(specified_date, datetime.min.time()) - timedelta(hours=6)
    start_dt = end_dt - timedelta(days=2)
    start_str, end_str = start_dt.strftime("%Y%m%d%H%M"), end_dt.strftime("%Y%m%d%H%M")
    log.info(f"Fetching {start_str} → {end_str}  ({specified_date})")

    client = arxiv.Client()
    all_papers, seen = [], set()
    for cat in TARGET_CATEGORIES:
        query = f"cat:{cat} AND submittedDate:[{start_str} TO {end_str}]"
        try:
            search = arxiv.Search(
                query=query,
                max_results=15,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
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
                        "updated_date":   r.updated.isoformat(),
                        "categories":     r.categories,
                        "authors":        [a.name for a in r.authors],
                    })
                    cnt += 1
            log.info(f"  {cat}: +{cnt}")
        except Exception as e:
            log.warning(f"  {cat}: error – {e}")
    log.info(f"Total unique papers: {len(all_papers)}")
    return all_papers


# ── Step 2: Filter papers (batch of 5) ────────────────────────────────────────

def filter_papers(papers: list) -> list:
    """Ask LLM which papers match the research topics. Batches of 5."""
    if not OPENAI_API_KEY:
        log.warning("No API key – skipping filter"); return papers
    if not papers:
        return []

    BATCH = 10
    filtered = []
    for i in range(0, len(papers), BATCH):
        batch = papers[i : i + BATCH]
        lines = [
            f"Paper {i+j+1}: {p['title']}\nAbstract: {p['summary'][:500]}"
            for j, p in enumerate(batch)
        ]
        prompt = f"""{RESEARCH_TOPICS}

Reply ONLY with the numbers of papers that match, separated by commas.
Example reply: 1, 3, 5

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
            ok  = idx in matched
            sym = "✓" if ok else "✗"
            log.info(f"  [{idx+1}/{len(papers)}] {sym} {p['title'][:55]}…")
            if ok:
                filtered.append(p)
        time.sleep(0.5)
    log.info(f"Filter: {len(filtered)}/{len(papers)} passed")
    # Cap at 12 for rate_papers to keep total API calls manageable
    filtered = filtered[:12]
    log.info(f"  → Capped to top 12 for rating")
    return filtered


# ── Helpers for Chinese TLDR ────────────────────────────────────────────────

def _has_chinese(text: str) -> bool:
    """Check if text contains at least one Chinese character."""
    return any('\u4e00' <= c <= '\u9fff' for c in text)


def _get_chinese_tldr(title: str, abstract: str) -> str:
    """
    Dedicated single-paper Chinese TLDR via a focused LLM call.
    Called as fallback when the batch rating returns empty or non-Chinese tldr.
    """
    prompt = f"""为以下论文写一个 2-3 句的中文摘要（简体中文）。摘要应该说明：论文提出了什么方法/模型，该方法的核心技术或贡献是什么，在什么任务上取得了什么效果。

标题: {title}
摘要: {abstract[:500]}

直接输出中文摘要，不要加引号，不要任何格式前缀，只输出纯中文文本。"""
    resp = llm_call(prompt, max_tokens=400, temperature=0.1)
    if resp:
        return resp.strip()[:400]
    log.warning(f"  ⚠ Dedicated Chinese TLDR call also returned nothing")
    return ""


# ── Step 3: Rate papers (batch of 2, cap at 30) ───────────────────────────────
# NOTE: tldr field = Chinese summary

RATING_TMPL = """You are an expert AI research paper reviewer.

Research interests:
1. Audio Large Language Models (Audio LLM, Audio Foundation Models)
2. Audio/Speech Perception & Understanding (speech recognition, audio understanding, multimodal perception)
3. Audio/Speech Reasoning (reasoning over audio, audio chain-of-thought)
4. Speech Synthesis (TTS, voice conversion, speech generation)
5. Omni Models (any-domain unified multimodal models including audio)

Papers to review:
{papers_block}

OUTPUT FORMAT — output ONLY valid JSON, no markdown fences, no explanation:
[
  {{"idx":1,"tldr":"中文摘要（必须！）：这篇论文提出...该方法...在...任务上取得了...效果。","relevance_score":8,"novelty_score":7,"clarity_score":8,"impact_score":9,"overall_priority_score":8}},
  {{"idx":2,"tldr":"中文摘要（必须！）：...","relevance_score":7,"novelty_score":8,"clarity_score":7,"impact_score":7,"overall_priority_score":7}}
]

CRITICAL RULES:
- "tldr" MUST be written in Simplified Chinese (简体中文), 2-3 sentences
- The tldr MUST contain Chinese characters (CJK unicode range) — English-only tlDRs will be rejected
- Output EXACTLY one JSON object per paper, in order
- Output ONLY the JSON array"""

ITEMS_TMPL = "Paper {i}: Title: {title}\\nAbstract: {abstract}"


def rate_papers(papers: list) -> list:
    """Score and rank papers. Batches of 2, capped at 30 total. Ensures Chinese tldr."""
    if not OPENAI_API_KEY:
        log.warning("No API key – skipping rating"); return papers
    if not papers:
        return []

    papers = papers[:6]
    BATCH  = 5  # rate batch = more reliable, especially for Chinese
    rated  = []
    for i in range(0, len(papers), BATCH):
        batch = papers[i : i + BATCH]
        block = "\n\n".join(
            ITEMS_TMPL.format(
                i=i+j+1,
                title=p["title"],
                abstract=p["summary"][:400],
            ) for j, p in enumerate(batch)
        )
        resp = llm_call(RATING_TMPL.format(papers_block=block), max_tokens=1500)

        if resp:
            try:
                cleaned = resp
                for fence in ["```json", "```"]:
                    if fence in cleaned:
                        cleaned = cleaned.split(fence)[1].split("```")[0]
                items = json.loads(cleaned)
                smap   = {item["idx"]: item for item in items}
                ok     = True
            except (json.JSONDecodeError, KeyError) as e:
                log.warning(f"  Batch {(i//BATCH)+1}: parse error ({e})")
                ok = False

            if ok:
                for j, p in enumerate(batch):
                    info     = smap.get(i + j + 1, {})
                    tldr_val = info.get("tldr", "")
                    # Fallback: if tldr is empty OR not Chinese, call dedicated endpoint
                    if not tldr_val or not _has_chinese(tldr_val):
                        log.info(f"  [{i+j+1}] No/invalid Chinese tldr — calling dedicated endpoint")
                        tldr_val = _get_chinese_tldr(p["title"], p["summary"])
                    p["tldr"]                     = tldr_val
                    p["relevance_score"]          = info.get("relevance_score", 5)
                    p["novelty_score"]            = info.get("novelty_score", 5)
                    p["clarity_score"]            = info.get("clarity_score", 5)
                    p["impact_score"]             = info.get("impact_score", 5)
                    p["overall_priority_score"]   = info.get("overall_priority_score", 5)
                    zh_mark = "✓ZH" if _has_chinese(tldr_val) else "✗"
                    log.info(f"  [{i+j+1}/{len(papers)}] {zh_mark} ★{p['overall_priority_score']}/10 {p['title'][:40]}…")
                    if _has_chinese(tldr_val):
                        log.info(f"       {tldr_val[:60]}…")
            else:
                # Parse failed — use dedicated endpoint for each paper
                for p in batch:
                    tldr = _get_chinese_tldr(p["title"], p["summary"])
                    p["tldr"] = tldr
                    p["relevance_score"] = 5; p["overall_priority_score"] = 5
        else:
            # No API response at all — use dedicated endpoint for each paper
            for p in batch:
                tldr = _get_chinese_tldr(p["title"], p["summary"])
                p["tldr"] = tldr
                p["relevance_score"] = 5; p["overall_priority_score"] = 5

        rated.extend(batch)
        time.sleep(2.0)  # longer delay for reliability
    return rated


# ── Step 4: Top-10 cited papers (past year) ─────────────────────────────────

TOP_CITED_PROMPT = """# Identify TOP 10 most cited papers from the past year related to:
1. Audio Large Language Models / Audio Foundation Models
2. Audio/Speech Perception and Understanding
3. Speech Synthesis (TTS, Voice Conversion)
4. Omni Models (unified multimodal models including audio)

The current date is {today}. The "past year" means papers from the previous full year
AND the current year (e.g. if today is 2026, look for 2025-2026 papers).

Output a strict JSON array (no markdown, no text outside the array):
[
  {{"rank":1,"title":"<exact title>","authors":"<first author et al.>","year":2025,"citations":1200,"venue":"arXiv","url":"https://arxiv.org/abs/...","summary":"<1-sentence English summary ONLY — do NOT write in Chinese>"}},
  ... (9 more, ranked by citations descending)
]

Only papers from {prev_year} or {curr_year}. Summaries MUST be in English.
If you are uncertain about exact citation counts, provide your best estimate."""

def get_top_cited() -> list:
    """Fetch top-10 cited papers via LLM (one call, ~30s)."""
    if not OPENAI_API_KEY:
        log.warning("No API key – skipping top-cited"); return []

    today   = date.today()
    prev_yr = today.year - 1
    curr_yr = today.year

    prompt  = TOP_CITED_PROMPT.format(today=today, prev_year=prev_yr, curr_year=curr_yr)
    resp    = llm_call(prompt, max_tokens=2500, temperature=0.3)

    if not resp:
        log.warning("Top-cited: no response"); return []
    try:
        cleaned = resp
        for fence in ["```json", "```"]:
            if fence in cleaned:
                cleaned = cleaned.split(fence)[1].split("```")[0]
        papers = json.loads(cleaned)
        log.info(f"Top-cited: {len(papers)} retrieved")
        return papers
    except json.JSONDecodeError as e:
        log.error(f"Top-cited parse error: {e}")
        try:
            with open(os.path.join(JSON_DIR, f"top_cited_fallback_{date.today().isoformat()}.txt"), "w") as _f:
                _f.write(resp.strip())
        except: pass
        return [resp.strip()]


# ── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup(directory: str, days: int):
    if not os.path.exists(directory):
        return
    cutoff = datetime.now() - timedelta(days=days)
    removed = 0
    for fn in os.listdir(directory):
        fp = os.path.join(directory, fn)
        if os.path.isfile(fp) and datetime.fromtimestamp(os.path.getmtime(fp)) < cutoff:
            os.remove(fp); removed += 1
    log.info(f"Cleanup: removed {removed} old file(s) from {directory}")


# ── HTML generation ────────────────────────────────────────────────────────────

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
        title = f"Audio/Speech AI Papers – {rdate.strftime('%B %d, %Y')}"
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


def update_reports_and_pages():
    files = sorted(
        [f for f in os.listdir(HTML_DIR) if f.endswith(".html")],
        reverse=True,
    )
    with open(REPORTS_PATH, "w") as f:
        json.dump(files, f, indent=4)
    log.info(f"reports.json: {len(files)} reports")


# ── Process one date ───────────────────────────────────────────────────────────

def process(target_date: date, force: bool = False):
    """Fetch, filter, rate, and generate HTML for one date."""
    log.info(f"\n{'='*60}\n  {target_date}\n{'='*60}")
    jpath = os.path.join(JSON_DIR, f"{target_date.isoformat()}.json")
    os.makedirs(JSON_DIR, exist_ok=True)

    if os.path.exists(jpath) and not force:
        log.info("JSON exists – loading from cache")
        with open(jpath) as f:
            papers = json.load(f)
    else:
        raw = fetch_papers(target_date)
        if not raw:
            with open(jpath, "w") as f:
                json.dump([], f)
            papers = []
        else:
            filtered = filter_papers(raw)
            papers   = rate_papers(filtered) if filtered else []
            papers.sort(key=lambda x: x.get("overall_priority_score", 0), reverse=True)
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(papers, f, indent=4, ensure_ascii=False)
            log.info(f"Saved {jpath} ({len(papers)} papers)")

    # Top-cited section only on today's report
    is_today  = (target_date == date.today())
    top_cited = get_top_cited() if is_today else []

    generate_html(target_date.isoformat(), papers, top_cited)
    if is_today:
        update_reports_and_pages()
    return papers


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    log.info(f"Model: {OPENAI_MODEL}  |  API: {OPENAI_API_URL}")
    cleanup(JSON_DIR, DATA_RETENTION)
    cleanup(HTML_DIR, DATA_RETENTION)

    today = date.today()
    process(today, force=args.force)
    log.info("Pipeline complete!")


if __name__ == "__main__":
    main()
