import arxiv
import logging
from datetime import date, timedelta, datetime, timezone
from typing import List, Dict, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Target arXiv categories for audio/speech/Omni papers
TARGET_CATEGORIES = [
    'cs.CL',      # Computation and Language (covers speech, NLP)
    'cs.AI',      # Artificial Intelligence
    'cs.LG',      # Machine Learning
    'cs.SD',      # Sound (Audio processing)
    'eess.AS',    # Audio and Speech Processing
    'cs.MM',      # Multimedia
    'cs.CV',      # Computer Vision
]


def fetch_papers_by_categories(
    categories: List[str],
    max_results_per_category: int = 300,
    specified_date: Optional[date] = None
) -> List[Dict[str, Any]]:
    """Fetches papers from multiple arXiv categories for a given date.

    Args:
        categories: List of arXiv category codes (e.g., ['cs.CL', 'cs.AI']).
        max_results_per_category: Max results per category.
        specified_date: Date to fetch papers for (UTC). Defaults to today.

    Returns:
        List of paper dictionaries with title, summary, url, dates, categories, authors.
    """
    if specified_date is None:
        specified_date = datetime.now(timezone.utc).date()
        logging.info(f"No date specified, defaulting to {specified_date.strftime('%Y-%m-%d')} UTC.")
    else:
        logging.info(f"Fetching papers for specified date: {specified_date.strftime('%Y-%m-%d')} UTC.")

    # Convert date to datetime for arXiv API
    end_dt = datetime.combine(specified_date, datetime.min.time())
    # Go back ~6 hours to handle arXiv's timezone
    end_dt = end_dt - timedelta(hours=6)
    start_dt = end_dt - timedelta(days=1)

    start_str = start_dt.strftime('%Y%m%d%H%M')
    end_str = end_dt.strftime('%Y%m%d%H%M')

    client = arxiv.Client()
    all_papers: List[Dict[str, Any]] = []
    seen_ids = set()

    for category in categories:
        query = f'cat:{category} AND submittedDate:[{start_str} TO {end_str}]'
        logging.info(f"Querying category '{category}': {query}")

        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results_per_category,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            results = client.results(search)
            count = 0
            for result in results:
                if result.entry_id not in seen_ids:
                    seen_ids.add(result.entry_id)
                    all_papers.append({
                        'title': result.title,
                        'summary': result.summary.strip(),
                        'url': result.entry_id,
                        'published_date': result.published,
                        'updated_date': result.updated,
                        'categories': result.categories,
                        'authors': [author.name for author in result.authors],
                    })
                    count += 1
            logging.info(f"  -> Found {count} new papers in {category}")
        except Exception as e:
            logging.warning(f"Error fetching from {category}: {e}")

    logging.info(f"Total unique papers fetched: {len(all_papers)}")
    return all_papers


def fetch_cv_papers(category: str = 'cs.CV', max_results: int = 500, specified_date: Optional[date] = None) -> List[Dict[str, Any]]:
    """Legacy function for backward compatibility - fetches from single category."""
    return fetch_papers_by_categories([category], max_results, specified_date)


if __name__ == '__main__':
    logging.info("Testing paper fetcher...")
    # Use yesterday's date to ensure results
    test_date = date.today() - timedelta(days=1)
    papers = fetch_papers_by_categories(TARGET_CATEGORIES, specified_date=test_date)
    print(f"\nFetched {len(papers)} papers for {test_date}")
    for i, p in enumerate(papers[:5]):
        print(f"  {i+1}. {p['title'][:60]}...")
