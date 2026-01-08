"""
News scraping using Event Registry API.
Based on fetch_data.py.
"""

import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from eventregistry import EventRegistry, QueryArticlesIter


def fetch_articles(
    topic: str,
    api_key: str = None,
    max_articles: int = 200,
    output_dir: str = None
) -> pd.DataFrame:
    """
    Fetch articles about a topic from Event Registry.

    Args:
        topic: Entity/topic to search for (e.g., "Draymond Green")
        api_key: Event Registry API key (or set EVENT_REGISTRY_API_KEY env var)
        max_articles: Maximum number of articles to fetch
        output_dir: Optional directory to save raw CSV

    Returns:
        DataFrame with columns: title, body_text, source, date, url
    """
    api_key = api_key or os.environ.get("EVENT_REGISTRY_API_KEY")
    if not api_key:
        raise ValueError("Event Registry API key required. Set EVENT_REGISTRY_API_KEY or pass api_key.")

    print(f"Fetching articles for: {topic}")
    er = EventRegistry(apiKey=api_key)

    articles = []

    # Strategy 1: Concept Search
    concept_uri = er.getConceptUri(topic)
    if concept_uri:
        print(f"  Using concept URI: {concept_uri}")
        q = QueryArticlesIter(conceptUri=concept_uri)
        for art in q.execQuery(er, maxItems=max_articles):
            articles.append(art)

    # Strategy 2: Keyword Search (fallback)
    if len(articles) < 10:
        print(f"  Fallback to keyword search: '{topic}'")
        existing_urls = {a.get('url') for a in articles}
        q = QueryArticlesIter(keywords=topic)
        for art in q.execQuery(er, maxItems=max_articles):
            if art.get('url') not in existing_urls:
                articles.append(art)
                if len(articles) >= max_articles:
                    break

    print(f"  Fetched {len(articles)} articles")

    if not articles:
        return pd.DataFrame()

    # Clean up
    clean_articles = []
    for art in articles:
        clean_articles.append({
            "title": art.get("title"),
            "body_text": art.get("body"),
            "source": art.get("source", {}).get("title"),
            "date": art.get("date"),
            "url": art.get("url")
        })

    df = pd.DataFrame(clean_articles)

    # Save if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "analysis_raw.csv"
        df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")

    return df
