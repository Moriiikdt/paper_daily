import os
import requests
import time
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_API_URL = f"{OPENAI_API_BASE.rstrip('/')}/chat/completions"
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.4")


def call_llm_api(prompt: str, max_tokens: int = 5) -> str | None:
    """Call the configured LLM API and return the response."""
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY environment variable is not set.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return None
    except (KeyError, IndexError) as e:
        logging.error(f"Failed to parse API response: {e}")
        return None


# Topic descriptions for filtering
RESEARCH_TOPICS = """
The user is interested in papers related to:
1. Audio Large Language Models (Audio LLM, Audio Foundation Models)
2. Audio Large Model Reasoning (Reasoning over audio, Audio chain-of-thought)
3. Audio/Speech Perception and Understanding (Speech recognition, Audio understanding, Multimodal audio perception)
4. Speech Synthesis (Text-to-Speech, Voice conversion, Speech generation)
5. Omni Models (any-domain Omni models, unified multimodal models that include audio)
"""


def filter_papers_by_topic(papers: list, topic: str = "") -> list:
    """Use LLM to filter papers matching research interests."""
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY not set. Skipping filter.")
        return papers

    if not topic:
        topic = RESEARCH_TOPICS

    filtered = []
    logging.info(f"Filtering {len(papers)} papers...")

    for i, paper in enumerate(papers):
        title = paper.get('title', 'N/A')
        summary = paper.get('summary', 'N/A')

        prompt = f"""{topic}

Paper Title: {title}
Paper Abstract: {summary}

Question: Is this paper related to any of the topics above? Answer only 'yes' or 'no'."""

        resp = call_llm_api(prompt, max_tokens=5)
        if resp and 'yes' in resp.lower():
            filtered.append(paper)
            logging.info(f"[{i+1}/{len(papers)}] ✓ '{title[:50]}...'")
        else:
            logging.info(f"[{i+1}/{len(papers)}] ✗ '{title[:50]}...'")

    logging.info(f"Filter complete: {len(filtered)}/{len(papers)} papers matched.")
    return filtered


# Rating prompt - note: TLDR is in English only (no Chinese)
rating_prompt_template = """# Role
You are an experienced researcher evaluating AI/ML papers. Provide structured assessments in JSON format.

# Task
Evaluate the paper below and output a JSON object with your assessment.

# Paper
Title: %s
Abstract: %s

# Research Interests
Audio Large Language Models, Audio/Speech Perception & Understanding, Speech Synthesis, Omni Models

# Output Format (strict JSON, no markdown)
{
  "tldr": "<2-sentence summary in English>",
  "relevance_score": <int 1-10>,
  "novelty_score": <int 1-10>,
  "clarity_score": <int 1-10>,
  "impact_score": <int 1-10>,
  "overall_priority_score": <int 1-10>
}

Scoring guidelines:
- relevance: How directly does it relate to audio/speech/Omni AI?
- novelty: How novel is the approach compared to existing work?
- clarity: How clear and complete is the abstract?
- impact: What is the potential significance of this work?
- overall_priority: Combined reading priority score
"""


def rate_papers(papers: list) -> list:
    """Rate each paper using LLM."""
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY not set. Skipping rating.")
        return papers

    logging.info(f"Ranking {len(papers)} papers...")
    for i, paper in enumerate(papers):
        title = paper.get('title', 'N/A')
        summary = paper.get('summary', 'N/A')
        prompt = rating_prompt_template % (title, summary)

        for attempt in range(2):
            resp = call_llm_api(prompt, max_tokens=800)
            if resp:
                try:
                    # Clean markdown if present
                    cleaned = resp
                    if '```json' in cleaned:
                        cleaned = cleaned.split('```json')[1].split('```')[0]
                    rating = json.loads(cleaned)
                    paper.update(rating)
                    logging.info(f"[{i+1}/{len(papers)}] Rated '{title[:40]}...' -> {rating.get('overall_priority_score', '?')}/10")
                    break
                except json.JSONDecodeError:
                    logging.warning(f"JSON parse failed (attempt {attempt+1}): {resp[:100]}")
            else:
                logging.warning(f"[{i+1}/{len(papers)}] No response from API (attempt {attempt+1})")

        # Small delay between API calls
        time.sleep(0.5)

    return papers


# Google Scholar top papers summary prompt
top_papers_prompt_template = """# Task
You are a research assistant. Based on your knowledge, identify the TOP 10 most cited papers from 2025-2026 related to the following fields:

1. Audio Large Language Models (Audio LLM, Audio Foundation Models)
2. Audio/Speech Perception and Understanding
3. Speech Synthesis (TTS, Voice Conversion)
4. Omni Models (unified multimodal models including audio)

# Output Format (strict JSON array, no markdown)
[
  {
    "rank": 1,
    "title": "<paper title>",
    "authors": "<first author et al.>",
    "year": <publication year>,
    "citations": <estimated citation count>,
    "venue": "<conference/journal or arXiv>",
    "url": "<arxiv or paper URL>",
    "summary": "<1-2 sentence description>"
  },
  ... (9 more entries)
]

Requirements:
- Only include papers from 2025 or 2026
- Rank by citation count (most cited first)
- Provide realistic but approximate citation counts
- Include only works you have high confidence about
"""


def get_top_cited_papers_summary() -> list:
    """Get top 10 most cited papers from recent years via LLM."""
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY not set. Skipping top papers summary.")
        return []

    logging.info("Fetching top cited papers summary...")
    resp = call_llm_api(top_papers_prompt_template, max_tokens=2000)

    if not resp:
        return []

    try:
        # Clean markdown if present
        cleaned = resp
        if '```json' in cleaned:
            cleaned = resp.split('```json')[1].split('```')[0]
        elif '```' in cleaned:
            cleaned = cleaned.split('```')[1].split('```')[0]
        papers = json.loads(cleaned)
        logging.info(f"Retrieved {len(papers)} top papers")
        return papers
    except json.JSONDecodeError:
        logging.error(f"Failed to parse top papers JSON: {resp[:200]}")
        return []


if __name__ == '__main__':
    if OPENAI_API_KEY:
        test_papers = [
            {
                'title': 'GPT-4o: A Unified Multimodal Model Including Audio',
                'summary': 'We present GPT-4o, a model that can handle audio, vision, and text in a unified architecture...'
            },
            {
                'title': 'Deep Learning for Image Classification',
                'summary': 'We propose a new convolutional neural network architecture for image classification tasks...'
            },
        ]
        logging.info("Testing filter...")
        filtered = filter_papers_by_topic(test_papers)
        logging.info(f"Filtered: {len(filtered)} papers")
    else:
        logging.warning("Set OPENAI_API_KEY to run tests.")
