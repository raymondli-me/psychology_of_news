#!/usr/bin/env python3
"""
Daily News Analysis Example

This script demonstrates a minimal daily workflow:
1. Fetch today's articles about a topic
2. Rate with 3 LLMs
3. Generate visualization

Run:
    python examples/daily_analysis.py
"""

import os
from datetime import datetime
from psychology_of_news import Analyzer, Config

# ============================================================================
# CONFIGURATION - Customize this!
# ============================================================================

TOPIC = "Draymond Green trade"  # Change to any topic/player

# Output directory with today's date
OUTPUT_DIR = f"./output/{datetime.now().strftime('%Y-%m-%d')}_{TOPIC.replace(' ', '_')}"

# API Keys - set these as environment variables or hardcode here
# os.environ["EVENT_REGISTRY_API_KEY"] = "your-key"
# os.environ["OPENAI_API_KEY"] = "sk-..."
# os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
# os.environ["GOOGLE_API_KEY"] = "AIza..."

# ============================================================================
# RUN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    config = Config(
        topic=TOPIC,
        output_dir=OUTPUT_DIR,
        max_sentences=100,  # Fewer for quick daily run
    )

    analyzer = Analyzer(config)

    # Option 1: Full pipeline
    results = analyzer.run_all()

    print("\nResults:")
    for key, path in results.items():
        print(f"  {key}: {path}")

    # Option 2: Step by step (useful for debugging)
    # analyzer.fetch()
    # analyzer.rate()
    # analyzer.visualize()
