#!/usr/bin/env python3
"""
Process Reddit/YouTube data with REAL LLM ratings (not mock keyword scores).
Uses gpt-5.1 with reasoning_effort="none", claude-sonnet-4-5, and gemini-2.5-flash.
"""

import json
import asyncio
import re
import random
import numpy as np
from pathlib import Path
import umap
from sklearn.cluster import KMeans
from litellm import acompletion

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.embeddings import generate_embeddings

# Model IDs for litellm
MODELS = {
    "gpt": "openai/gpt-5.1",
    "claude": "anthropic/claude-sonnet-4-5",
    "gemini": "gemini/gemini-2.5-flash"
}

RATING_PROMPT = """Rate this text on how strongly it implies Draymond Green will be traded.

Text: "{text}"

Score from 1-10:
1 = No trade implication at all
5 = Neutral/ambiguous
10 = Strongly implies trade will happen

Reply with ONLY a single number (1-10), nothing else."""

# Semaphores to rate-limit per model
MAX_CONCURRENT = 5
semaphores = {name: asyncio.Semaphore(MAX_CONCURRENT) for name in MODELS}


async def rate_single(text: str, model_key: str) -> int:
    """Rate one text with one model. Returns score 1-10."""
    model_id = MODELS[model_key]

    async with semaphores[model_key]:
        for attempt in range(3):
            try:
                # GPT-5.1 with reasoning disabled
                extra_params = {"reasoning_effort": "none"} if "gpt-5.1" in model_id else {}

                response = await acompletion(
                    model=model_id,
                    messages=[{"role": "user", "content": RATING_PROMPT.format(text=text[:500])}],
                    timeout=30,
                    max_tokens=10,
                    **extra_params
                )
                content = response.choices[0].message.content.strip()
                match = re.search(r'\b(\d+)\b', content)
                if match:
                    score = int(match.group(1))
                    return min(max(score, 1), 10)
                return 5
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"  {model_key} error: {e}")
                    return 5
    return 5


async def rate_text(text: str) -> dict:
    """Rate text with all 3 models in parallel."""
    tasks = [rate_single(text, model) for model in MODELS.keys()]
    results = await asyncio.gather(*tasks)
    return dict(zip(MODELS.keys(), results))


async def process_source(source: str, sample_size: int = 200):
    """Process Reddit or YouTube data with real LLM ratings."""

    # Set paths based on source
    if source == "reddit":
        data_dir = Path(__file__).parent.parent / "reddit" / "data"
        raw_file = data_dir / "raw_reddit_real.json"
    elif source == "youtube":
        data_dir = Path(__file__).parent.parent / "youtube" / "data"
        raw_file = data_dir / "raw_youtube.json"
    else:
        raise ValueError(f"Unknown source: {source}")

    # Load raw data
    with open(raw_file) as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} {source} items")

    # Sample randomly
    if len(raw_data) > sample_size:
        raw_data = random.sample(raw_data, sample_size)
        print(f"Sampled {sample_size} items for processing")

    # Extract texts
    texts = [item["text"] for item in raw_data]

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # UMAP for 3D projection
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        n_jobs=1,
        metric="cosine"
    )
    coords = reducer.fit_transform(embeddings)
    coords = coords * 2  # Scale for visualization
    print("UMAP complete")

    # Cluster
    print("Clustering...")
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Rate with LLMs
    print(f"Rating {len(texts)} texts with 3 LLMs...")
    print("  (This may take a few minutes)")

    # Process in batches
    batch_size = 10
    all_ratings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_tasks = [rate_text(text) for text in batch]
        batch_ratings = await asyncio.gather(*batch_tasks)
        all_ratings.extend(batch_ratings)

        done = min(i + batch_size, len(texts))
        print(f"  Rated {done}/{len(texts)} items")

    # Build points
    print("Building visualization data...")
    points = []
    for i, (item, coord, label, ratings) in enumerate(zip(raw_data, coords, labels, all_ratings)):
        point = {
            "id": i,
            "sentence": item["text"][:500],
            "text": item["text"][:500],
            "x": float(coord[0]),
            "y": float(coord[1]),
            "z": float(coord[2]),
            "cluster": int(label),
            "gpt": ratings["gpt"],
            "claude": ratings["claude"],
            "gemini": ratings["gemini"],
            "mean": (ratings["gpt"] + ratings["claude"] + ratings["gemini"]) / 3,
        }

        # Add source-specific fields
        if source == "reddit":
            point.update({
                "source": f"r/{item.get('subreddit', 'unknown')}",
                "subreddit": item.get("subreddit", "unknown"),
                "type": item.get("type", "comment"),
                "score": item.get("score", 0),
                "url": item.get("url", ""),
                "post_title": item.get("post_title", "")[:100]
            })
        elif source == "youtube":
            point.update({
                "source": f"YouTube: {item.get('channel_name', 'Unknown')[:20]}",
                "channel_name": item.get("channel_name", "Unknown"),
                "video_title": item.get("video_title", "")[:100],
                "type": item.get("type", "comment"),
                "like_count": item.get("like_count", 0),
                "url": item.get("url", ""),
                "video_id": item.get("video_id", "")
            })

        points.append(point)

    # Save points
    with open(data_dir / "points_data.json", "w") as f:
        json.dump(points, f)
    print(f"Saved {len(points)} points")

    # Generate cluster stats
    cluster_stats = {}
    for cluster_id in range(n_clusters):
        cluster_points = [p for p in points if p["cluster"] == cluster_id]
        if cluster_points:
            cx = np.mean([p["x"] for p in cluster_points])
            cy = np.mean([p["y"] for p in cluster_points])
            cz = np.mean([p["z"] for p in cluster_points])
            cluster_stats[str(cluster_id)] = {
                "count": len(cluster_points),
                "centroid": {"x": float(cx), "y": float(cy), "z": float(cz)},
                "gpt_mean": float(np.mean([p["gpt"] for p in cluster_points])),
                "claude_mean": float(np.mean([p["claude"] for p in cluster_points])),
                "gemini_mean": float(np.mean([p["gemini"] for p in cluster_points]))
            }

    with open(data_dir / "cluster_stats.json", "w") as f:
        json.dump(cluster_stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print(f"PROCESSING COMPLETE - {source.upper()}")
    print("=" * 60)
    print(f"Items processed: {len(points)}")
    print(f"Clusters: {n_clusters}")
    print(f"\nRating Summary:")
    print(f"  GPT mean:    {np.mean([p['gpt'] for p in points]):.2f}")
    print(f"  Claude mean: {np.mean([p['claude'] for p in points]):.2f}")
    print(f"  Gemini mean: {np.mean([p['gemini'] for p in points]):.2f}")
    print(f"\nRun scripts/label_{source}_topics.py to generate topic labels")
    print(f"Then restart: python {source}/server.py")

    return points


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process data with real LLM ratings")
    parser.add_argument("source", choices=["reddit", "youtube"], help="Data source")
    parser.add_argument("--sample", type=int, default=200, help="Sample size (default: 200)")

    args = parser.parse_args()

    asyncio.run(process_source(args.source, args.sample))
