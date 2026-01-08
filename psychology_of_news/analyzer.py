"""
Main Analyzer class that orchestrates the full pipeline.
"""

import asyncio
import os
import sys
import warnings
import pandas as pd
import numpy as np
import nltk
from pathlib import Path
from datetime import datetime

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="litellm")

# Detect if we're in Jupyter/Colab
def _in_notebook():
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False

# Only apply nest_asyncio in notebooks where event loop is already running
if _in_notebook():
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

from .config import Config
from .scraper import fetch_articles
from .rater import rate_sentences
from .visualizer import create_interactive_umap


class Analyzer:
    """
    Main class for psychology of news analysis.

    Example:
        from psychology_of_news import Analyzer, Config

        config = Config(
            topic="Draymond Green trade",
            output_dir="./output"
        )

        analyzer = Analyzer(config)
        analyzer.fetch()     # Get articles from Event Registry
        analyzer.rate()      # Rate sentences with 3 LLMs
        analyzer.visualize() # Create interactive UMAP
        analyzer.run_haam()  # Run HAAM analysis
    """

    def __init__(self, config: Config):
        self.config = config
        self.articles_df = None
        self.sentences = []
        self.rated_df = None

        # Setup NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

    def fetch(
        self,
        api_key: str = None,
        max_articles: int = 200
    ) -> pd.DataFrame:
        """
        Fetch articles from Event Registry.

        Args:
            api_key: Event Registry API key (or use env var)
            max_articles: Max articles to fetch

        Returns:
            DataFrame of articles
        """
        self.articles_df = fetch_articles(
            topic=self.config.topic,
            api_key=api_key,
            max_articles=max_articles,
            output_dir=str(self.config.output_dir)
        )
        return self.articles_df

    def load_articles(self, csv_path: str) -> pd.DataFrame:
        """
        Load articles from existing CSV instead of fetching.

        Args:
            csv_path: Path to CSV with columns [title, body_text, source, date, url]

        Returns:
            DataFrame of articles
        """
        self.articles_df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.articles_df)} articles from {csv_path}")
        return self.articles_df

    def extract_sentences(self) -> list:
        """
        Extract sentences from loaded articles.

        Returns:
            List of sentence dicts
        """
        if self.articles_df is None:
            raise ValueError("No articles loaded. Call fetch() or load_articles() first.")

        keywords = self.config.topic_keywords
        logic = self.config.keyword_logic
        self.sentences = []

        for _, row in self.articles_df.iterrows():
            text = str(row.get('body_text', '')).replace('\n', ' ')
            source = row.get('source', 'Unknown')
            title = row.get('title', '')

            try:
                sents = nltk.sent_tokenize(text)
            except:
                sents = text.split('. ')

            for s in sents:
                s = s.strip()
                if len(s) < self.config.min_sentence_length:
                    continue
                if len(s) > self.config.max_sentence_length:
                    continue
                if not self.config.matches_keywords(s):
                    continue

                self.sentences.append({
                    "text": s,
                    "source": source,
                    "article_title": title
                })

                if len(self.sentences) >= self.config.max_sentences:
                    break

            if len(self.sentences) >= self.config.max_sentences:
                break

        kw_str = f" {logic.upper()} ".join(keywords)
        print(f"Extracted {len(self.sentences)} sentences matching: {kw_str}")
        return self.sentences

    def rate(self) -> pd.DataFrame:
        """
        Rate all extracted sentences with configured LLMs.

        Returns:
            DataFrame with sentences and scores
        """
        if not self.sentences:
            self.extract_sentences()

        if not self.sentences:
            raise ValueError("No sentences to rate.")

        print(f"\nRating {len(self.sentences)} sentences with {len(self.config.models)} models...")
        print(f"Models: {', '.join(m.name for m in self.config.models)}")
        print("-" * 60)

        # Run async rating
        rated = asyncio.run(rate_sentences(self.sentences, self.config))

        self.rated_df = pd.DataFrame(rated)

        # Save
        output_file = self.config.output_dir / "sentence_ratings.csv"
        self.rated_df.to_csv(output_file, index=False)
        print(f"\nSaved {len(self.rated_df)} rated sentences to {output_file}")

        # Print summary
        print("\nSummary:")
        for m in self.config.models:
            col = f'{m.name}_score'
            if col in self.rated_df.columns:
                print(f"  {m.name}: mean={self.rated_df[col].mean():.2f}, std={self.rated_df[col].std():.2f}")

        return self.rated_df

    def load_ratings(self, csv_path: str) -> pd.DataFrame:
        """Load existing ratings from CSV."""
        self.rated_df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.rated_df)} rated sentences from {csv_path}")
        return self.rated_df

    def visualize(self) -> str:
        """
        Create interactive 3D UMAP visualization.

        Returns:
            Path to saved HTML file
        """
        if self.rated_df is None:
            raise ValueError("No rated data. Call rate() or load_ratings() first.")

        # Get URL mapping if we have articles
        url_mapping = None
        if self.articles_df is not None:
            url_mapping = dict(zip(
                self.articles_df['title'].astype(str),
                self.articles_df['url'].astype(str)
            ))

        return create_interactive_umap(self.rated_df, self.config, url_mapping)

    def run_haam(self):
        """
        Run HAAM analysis on rated data.

        Returns:
            HAAM object
        """
        if self.rated_df is None:
            raise ValueError("No rated data. Call rate() or load_ratings() first.")

        # Add HAAM repo to path
        haam_path = os.environ.get("HAAM_PATH", "/Users/raymondli701/2026_01_07_workspace/haam_repo")
        if haam_path not in sys.path:
            sys.path.insert(0, haam_path)

        from haam import HAAM

        model_names = [m.name for m in self.config.models]
        if len(model_names) < 3:
            print("Warning: HAAM requires 3 models (criterion, human, ai)")
            return None

        # Map: first model = criterion (X), second = human (HU), third = AI
        criterion = self.rated_df[f'{model_names[0]}_score'].values.astype(float)
        human_judgment = self.rated_df[f'{model_names[1]}_score'].values.astype(float)
        ai_judgment = self.rated_df[f'{model_names[2]}_score'].values.astype(float)
        texts = self.rated_df['text'].tolist()

        print(f"\nRunning HAAM:")
        print(f"  Criterion (X):  {model_names[0]}")
        print(f"  Human (HU):     {model_names[1]}")
        print(f"  AI:             {model_names[2]}")

        n_components = min(50, len(self.rated_df) - 2)

        haam = HAAM(
            criterion=criterion,
            ai_judgment=ai_judgment,
            human_judgment=human_judgment,
            texts=texts,
            n_components=n_components,
            min_cluster_size=max(3, len(self.rated_df) // 20),
            min_samples=2,
            umap_n_components=3,
            standardize=True,
            sample_split_post_lasso=False,
            auto_run=True
        )

        # Save outputs
        haam_dir = str(self.config.output_dir / "haam_outputs")
        os.makedirs(haam_dir, exist_ok=True)

        try:
            haam.create_comprehensive_pc_analysis(
                k_topics=3,
                max_words=100,
                generate_wordclouds=True,
                generate_3d_umap=True,
                output_dir=haam_dir,
                display=False
            )
            print(f"HAAM outputs saved to {haam_dir}")
        except Exception as e:
            print(f"HAAM visualization error: {e}")

        return haam

    def run_all(self, event_registry_key: str = None) -> dict:
        """
        Run the complete pipeline.

        Args:
            event_registry_key: Event Registry API key

        Returns:
            Dict with paths to all outputs
        """
        print("=" * 60)
        print(f"PSYCHOLOGY OF NEWS: {self.config.topic}")
        print(f"Output: {self.config.output_dir}")
        print("=" * 60)

        # Step 1: Fetch
        print("\n[1/4] FETCHING ARTICLES...")
        self.fetch(api_key=event_registry_key)

        if self.articles_df is None or len(self.articles_df) == 0:
            print("ERROR: No articles found")
            return {}

        # Step 2: Rate
        print("\n[2/4] RATING SENTENCES...")
        self.rate()

        # Step 3: Visualize
        print("\n[3/4] CREATING VISUALIZATION...")
        viz_path = self.visualize()

        # Step 4: HAAM
        print("\n[4/4] RUNNING HAAM ANALYSIS...")
        try:
            self.run_haam()
        except Exception as e:
            print(f"HAAM error: {e}")

        print("\n" + "=" * 60)
        print("COMPLETE!")
        print(f"Results: {self.config.output_dir}")
        print("=" * 60)

        return {
            "ratings": str(self.config.output_dir / "sentence_ratings.csv"),
            "visualization": viz_path,
            "haam": str(self.config.output_dir / "haam_outputs")
        }
