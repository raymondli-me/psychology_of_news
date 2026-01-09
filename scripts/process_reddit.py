#!/usr/bin/env python3
"""Process Reddit data: collect, embed, cluster, rate, and prepare for visualization."""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.reddit import collect_reddit
from shared.pipeline import process_source


async def main():
    parser = argparse.ArgumentParser(description="Process Reddit data for visualization")
    parser.add_argument("--query", default="draymond green trade", help="Search query")
    parser.add_argument("--subreddits", nargs="+", default=["nba", "warriors", "nbadiscussion"],
                        help="Subreddits to search")
    parser.add_argument("--max-posts", type=int, default=50, help="Max posts per subreddit")
    parser.add_argument("--max-comments", type=int, default=20, help="Max comments per post")
    parser.add_argument("--time", default="month", choices=["hour", "day", "week", "month", "year", "all"])
    parser.add_argument("--skip-rating", action="store_true", help="Skip LLM rating (use for testing)")
    parser.add_argument("--output-dir", default="reddit/data", help="Output directory")

    args = parser.parse_args()

    output_dir = Path(__file__).parent.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("REDDIT SENTIMENT PROCESSING PIPELINE")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"Subreddits: {', '.join(args.subreddits)}")
    print(f"Time filter: {args.time}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Step 1: Collect Reddit data
    print("\n[STEP 1] Collecting Reddit data...")
    raw_data = collect_reddit(
        query=args.query,
        subreddits=args.subreddits,
        max_posts=args.max_posts,
        max_comments=args.max_comments,
        time_filter=args.time,
        output_path=str(output_dir / "raw_reddit.json")
    )

    if not raw_data:
        print("No data collected! Check your Reddit API credentials.")
        print("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")
        return

    print(f"\nCollected {len(raw_data)} items")

    # Step 2: Process through pipeline
    print("\n[STEP 2] Processing through pipeline...")
    processed = await process_source(
        source_type="reddit",
        items=raw_data,
        output_dir=str(output_dir),
        text_key="text",
        rate=not args.skip_rating
    )

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Processed {len(processed)} items")
    print(f"Output saved to: {output_dir}")
    print("\nTo start the Reddit visualization server:")
    print(f"  cd {Path(__file__).parent.parent}")
    print("  python reddit/server.py")
    print("  # Then visit http://localhost:8001")


if __name__ == "__main__":
    asyncio.run(main())
