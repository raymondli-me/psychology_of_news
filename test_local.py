#!/usr/bin/env python3
"""
Local test script for psychology_of_news package.
Separates data fetching from processing to save API calls.

Usage:
    python test_local.py fetch    # Fetch articles (uses Event Registry API)
    python test_local.py rate     # Rate sentences (uses LLM APIs)
    python test_local.py viz      # Create visualization only
    python test_local.py all      # Run everything
    python test_local.py test200  # Rate 200 existing sentences (timestamped output)

Setup:
    Create a .env file with your API keys:
        EVENT_REGISTRY_API_KEY=your_key
        OPENAI_API_KEY=your_key
        ANTHROPIC_API_KEY=your_key
        GOOGLE_API_KEY=your_key
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Load .env file if it exists
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        print(f"Loading API keys from {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    else:
        print("No .env file found. Using environment variables.")

load_env()

# Add package to path for local testing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psychology_of_news import Analyzer, Config

# Configuration
OUTPUT_DIR = "./test_output"
ARTICLES_CSV = f"{OUTPUT_DIR}/articles.csv"
RATINGS_CSV = f"{OUTPUT_DIR}/sentence_ratings.csv"

config = Config(
    topic="Draymond Green trade",
    output_dir=OUTPUT_DIR,

    # Sentence filtering
    max_sentences=20,  # Small test run
    keyword_filter=None,  # Auto: "Draymond Green"
    keyword_logic="any",

    # Rating task
    rating_question="How likely does this imply Draymond Green will be traded?",
    scale_low="No trade implication",
    scale_mid="Neutral/ambiguous",
    scale_high="Trade very likely",

    # Concurrency settings
    max_concurrent_per_model=5,
    batch_size=10,
)


def fetch_articles():
    """Step 1: Fetch articles from Event Registry (costs API calls)."""
    print("=" * 60)
    print("STEP 1: FETCHING ARTICLES")
    print("=" * 60)

    analyzer = Analyzer(config)
    df = analyzer.fetch(max_articles=100)

    # Save to a predictable location
    df.to_csv(ARTICLES_CSV, index=False)
    print(f"\nSaved {len(df)} articles to {ARTICLES_CSV}")
    return df


def rate_sentences():
    """Step 2: Rate sentences with LLMs (costs LLM API calls)."""
    print("=" * 60)
    print("STEP 2: RATING SENTENCES")
    print("=" * 60)

    if not os.path.exists(ARTICLES_CSV):
        print(f"ERROR: No articles found at {ARTICLES_CSV}")
        print("Run 'python test_local.py fetch' first")
        return None

    analyzer = Analyzer(config)
    analyzer.load_articles(ARTICLES_CSV)
    df = analyzer.rate()

    print(f"\nSaved ratings to {RATINGS_CSV}")
    return df


def create_visualization():
    """Step 3: Create visualization (no API calls)."""
    print("=" * 60)
    print("STEP 3: CREATING VISUALIZATION")
    print("=" * 60)

    if not os.path.exists(RATINGS_CSV):
        print(f"ERROR: No ratings found at {RATINGS_CSV}")
        print("Run 'python test_local.py rate' first")
        return None

    analyzer = Analyzer(config)

    # Load articles for URL mapping (optional)
    if os.path.exists(ARTICLES_CSV):
        analyzer.load_articles(ARTICLES_CSV)

    analyzer.load_ratings(RATINGS_CSV)
    viz_path = analyzer.visualize()

    print(f"\nVisualization saved to {viz_path}")
    return viz_path


def run_all():
    """Run the complete pipeline."""
    fetch_articles()
    rate_sentences()
    create_visualization()
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Results in: {OUTPUT_DIR}")
    print("=" * 60)


def test_200():
    """Test with 200 sentences from previous run (re-rates them)."""
    import pandas as pd
    import asyncio
    from datetime import datetime
    from psychology_of_news.rater import rate_sentences as rate_all

    OLD_CSV = "/Users/raymondli701/2026_01_07_workspace/attempts/20260107_235604_draymond_triple_llm/sentence_ratings.csv"

    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{OUTPUT_DIR}/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("TEST 200: Rating 200 sentences with current models")
    print(f"Run ID: {timestamp}")
    print(f"Output: {run_dir}")
    print("=" * 60)

    if not os.path.exists(OLD_CSV):
        print(f"ERROR: Old CSV not found at {OLD_CSV}")
        return

    # Load old sentences (ignore old scores)
    df = pd.read_csv(OLD_CSV)
    print(f"Loaded {len(df)} sentences")

    # Show models being used
    print("\nModels:")
    for m in config.models:
        print(f"  {m.name}: {m.model_id}")

    # Extract just the sentence data
    sentences = [
        {"text": row["text"], "source": row["source"], "article_title": row["article_title"]}
        for _, row in df.iterrows()
    ]

    print(f"\nRating {len(sentences)} sentences...")

    # Rate them
    rated = asyncio.run(rate_all(sentences, config))

    # Save results with timestamp
    result_df = pd.DataFrame(rated)
    result_path = f"{run_dir}/sentence_ratings.csv"
    result_df.to_csv(result_path, index=False)
    print(f"\nSaved to {result_path}")

    # Show score distribution
    print("\nScore distribution:")
    for m in config.models:
        col = f"{m.name}_score"
        if col in result_df.columns:
            print(f"  {m.name}: mean={result_df[col].mean():.2f}, min={result_df[col].min()}, max={result_df[col].max()}")

    # Create visualization in the same timestamped directory
    print("\nCreating visualization...")
    from psychology_of_news.visualizer import create_interactive_umap

    # Temporarily update config output_dir for this run
    original_output_dir = config.output_dir
    config.output_dir = run_dir
    viz_path = create_interactive_umap(result_df, config)
    config.output_dir = original_output_dir

    print(f"Visualization: {viz_path}")
    print(f"\nAll outputs in: {run_dir}")

    return result_df


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nCurrent status:")
        print(f"  Articles: {'EXISTS' if os.path.exists(ARTICLES_CSV) else 'NOT FOUND'}")
        print(f"  Ratings:  {'EXISTS' if os.path.exists(RATINGS_CSV) else 'NOT FOUND'}")
        print("\nAPI Keys:")
        print(f"  EVENT_REGISTRY: {'SET' if os.environ.get('EVENT_REGISTRY_API_KEY') else 'MISSING'}")
        print(f"  OPENAI:         {'SET' if os.environ.get('OPENAI_API_KEY') else 'MISSING'}")
        print(f"  ANTHROPIC:      {'SET' if os.environ.get('ANTHROPIC_API_KEY') else 'MISSING'}")
        print(f"  GOOGLE:         {'SET' if os.environ.get('GOOGLE_API_KEY') else 'MISSING'}")
        return

    cmd = sys.argv[1].lower()

    if cmd == "fetch":
        fetch_articles()
    elif cmd == "rate":
        rate_sentences()
    elif cmd == "viz":
        create_visualization()
    elif cmd == "all":
        run_all()
    elif cmd == "test200":
        test_200()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
