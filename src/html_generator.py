import json
import os
import logging
from datetime import date, datetime, timezone
from jinja2 import Environment, FileSystemLoader


def generate_html_from_json(
    json_file_path: str,
    template_dir: str,
    template_name: str,
    output_dir: str,
    top_cited_papers: list = None
):
    """Generate HTML from JSON data using Jinja2 template."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
            papers.sort(key=lambda x: x.get('overall_priority_score', 0), reverse=True)
    except FileNotFoundError:
        logging.error(f"JSON file not found: {json_file_path}")
        return
    except json.JSONDecodeError:
        logging.error(f"JSON decode error: {json_file_path}")
        return

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)

    try:
        filename = os.path.basename(json_file_path)
        date_str = filename.split('.')[0]
        report_date = date.fromisoformat(date_str)
        formatted_date = report_date.strftime("%Y_%m_%d")
        page_title = f"Audio/Speech AI Papers - {report_date.strftime('%B %d, %Y')}"
    except (IndexError, ValueError):
        today = date.today()
        formatted_date = today.strftime("%Y_%m_%d")
        page_title = f"Audio/Speech AI Papers - {today.strftime('%B %d, %Y')}"

    generation_time = datetime.now(timezone.utc)

    html_content = template.render(
        papers=papers,
        title=page_title,
        report_date=report_date,
        generation_time=generation_time,
        top_cited_papers=top_cited_papers or []
    )

    output_filename = f"{formatted_date}.html"
    output_filepath = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logging.info(f"Generated HTML: {output_filepath}")
    except IOError as e:
        logging.error(f"Error writing HTML: {e}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Quick test
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Project root: {project_root}")
