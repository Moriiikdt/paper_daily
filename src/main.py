import os
import sys
import json
import logging
import argparse
from datetime import date, datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scraper import fetch_papers_by_categories, TARGET_CATEGORIES
from filter import filter_papers_by_topic, rate_papers, get_top_cited_papers_summary
from html_generator import generate_html_from_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_JSON_DIR = os.path.join(PROJECT_ROOT, 'daily_json')
DEFAULT_HTML_DIR = os.path.join(PROJECT_ROOT, 'daily_html')
DEFAULT_TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'templates')
DEFAULT_TEMPLATE_NAME = 'paper_template.html'
DATA_RETENTION_DAYS = 60  # Keep 2 months of data


def cleanup_old_files(directory: str, days: int = DATA_RETENTION_DAYS):
    """Remove files older than specified days."""
    if not os.path.exists(directory):
        return

    cutoff = datetime.now() - timedelta(days=days)
    removed = 0

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            if mtime < cutoff:
                os.remove(filepath)
                removed += 1
                logging.info(f"Removed old file: {filename}")

    if removed > 0:
        logging.info(f"Cleanup complete: removed {removed} old files")


def process_date(target_date: date, force_refresh: bool = False):
    """Process papers for a single date."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Processing date: {target_date.isoformat()}")
    logging.info(f"{'='*60}")

    json_filename = f"{target_date.isoformat()}.json"
    json_filepath = os.path.join(DEFAULT_JSON_DIR, json_filename)

    # Determine if we need to fetch new papers
    should_fetch = force_refresh or not os.path.exists(json_filepath)

    if not should_fetch:
        logging.info(f"JSON file exists: {json_filepath}. Use --force to refresh.")
        papers = []
    else:
        # Step 1: Fetch papers from multiple categories
        logging.info("Step 1: Fetching papers from arXiv...")
        raw_papers = fetch_papers_by_categories(TARGET_CATEGORIES, specified_date=target_date)

        if not raw_papers:
            logging.warning(f"No papers found for {target_date.isoformat()}")
            # Create empty JSON
            os.makedirs(DEFAULT_JSON_DIR, exist_ok=True)
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump([], f)
            return

        logging.info(f"Fetched {len(raw_papers)} papers")

        # Step 2: Filter by topic
        logging.info("Step 2: Filtering papers by topic...")
        papers = filter_papers_by_topic(raw_papers)

        if not papers:
            logging.warning("No papers matched the research topics.")
            papers = []

        logging.info(f"Filtered to {len(papers)} relevant papers")

        # Step 3: Rate papers
        if papers:
            logging.info("Step 3: Rating papers...")
            papers = rate_papers(papers)
            papers.sort(key=lambda x: x.get('overall_priority_score', 0), reverse=True)

        # Save JSON
        os.makedirs(DEFAULT_JSON_DIR, exist_ok=True)
        for paper in papers:
            if isinstance(paper.get('published_date'), datetime):
                paper['published_date'] = paper['published_date'].isoformat()
            if isinstance(paper.get('updated_date'), datetime):
                paper['updated_date'] = paper['updated_date'].isoformat()

        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=4, ensure_ascii=False)

        logging.info(f"Saved JSON: {json_filepath}")

    # Step 4: Generate HTML
    logging.info("Step 4: Generating HTML report...")

    # Get top cited papers (only for today's report)
    top_cited = []
    if target_date == date.today():
        logging.info("Step 5: Fetching top cited papers summary...")
        top_cited = get_top_cited_papers_summary()

    generate_html_from_json(
        json_filepath,
        DEFAULT_TEMPLATE_DIR,
        DEFAULT_TEMPLATE_NAME,
        DEFAULT_HTML_DIR,
        top_cited_papers=top_cited
    )

    # Step 6: Update reports.json
    logging.info("Step 6: Updating reports list...")
    reports_path = os.path.join(PROJECT_ROOT, 'reports.json')
    html_files = sorted(
        [f for f in os.listdir(DEFAULT_HTML_DIR) if f.endswith('.html')],
        reverse=True
    )
    with open(reports_path, 'w', encoding='utf-8') as f:
        json.dump(html_files, f, indent=4)

    logging.info(f"Reports list updated: {len(html_files)} reports")


def main():
    parser = argparse.ArgumentParser(description='Fetch and process arXiv audio/speech papers.')
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format')
    parser.add_argument('--force', action='store_true', help='Force refresh even if JSON exists')
    args = parser.parse_args()

    # Determine date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            logging.error("Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        target_date = date.today()

    # Cleanup old files
    logging.info(f"Cleaning up files older than {DATA_RETENTION_DAYS} days...")
    cleanup_old_files(DEFAULT_JSON_DIR, DATA_RETENTION_DAYS)
    cleanup_old_files(DEFAULT_HTML_DIR, DATA_RETENTION_DAYS)

    # Process dates: yesterday, day before, and today
    dates_to_process = [
        target_date - timedelta(days=2),
        target_date - timedelta(days=1),
        target_date
    ]

    for d in dates_to_process:
        process_date(d, force_refresh=args.force)

    logging.info("\nAll done!")


if __name__ == '__main__':
    main()
