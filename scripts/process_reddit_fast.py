#!/usr/bin/env python3
"""Process real Reddit data quickly (skip slow UMAP, use PCA instead)."""

import json
import random
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.embeddings import generate_embeddings


def main():
    data_dir = Path(__file__).parent.parent / "reddit" / "data"

    # Load real Reddit data
    with open(data_dir / "raw_reddit_real.json") as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} Reddit items")

    # Extract texts
    texts = [item["text"] for item in raw_data]

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Use PCA for 3D projection (faster than UMAP)
    print("Running PCA for 3D projection...")
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(embeddings)
    print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # Scale coordinates for better visualization
    coords = coords * 5

    # Cluster with KMeans (faster than DBSCAN tuning)
    print("Clustering...")
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Generate mock scores (based on keywords - real rating would use LLMs)
    print("Generating mock scores...")
    points = []
    for i, (item, coord, label) in enumerate(zip(raw_data, coords, labels)):
        text_lower = item["text"].lower()

        # Base score
        base = 5
        if any(w in text_lower for w in ["needs to be traded", "trade him", "trade asap", "ship him", "get rid"]):
            base += 2
        if any(w in text_lower for w in ["not going anywhere", "warrior for life", "keep him", "stay", "don't trade"]):
            base -= 2
        if any(w in text_lower for w in ["dynasty over", "implode", "rebuild", "tank"]):
            base += 1
        if any(w in text_lower for w in ["elite", "crucial", "valuable", "dpoy", "need him"]):
            base -= 1
        if any(w in text_lower for w in ["suspension", "punch", "ejected", "flagrant"]):
            base += 1

        gpt = max(1, min(10, base + random.randint(-1, 1)))
        claude = max(1, min(10, base + random.randint(-1, 1)))
        gemini = max(1, min(10, base + random.randint(-1, 1)))

        point = {
            "id": i,
            "sentence": item["text"][:500],  # Truncate long texts
            "x": float(coord[0]),
            "y": float(coord[1]),
            "z": float(coord[2]),
            "cluster": int(label),
            "gpt": gpt,
            "claude": claude,
            "gemini": gemini,
            "mean": (gpt + claude + gemini) / 3,
            "source": f"r/{item.get('subreddit', 'unknown')}",
            "subreddit": item.get("subreddit", "unknown"),
            "type": item.get("type", "comment"),
            "score": item.get("score", 0),
            "url": item.get("url", ""),
            "post_title": item.get("post_title", "")[:100]
        }
        points.append(point)

    # Save points
    with open(data_dir / "points_data.json", "w") as f:
        json.dump(points, f)
    print(f"Saved {len(points)} points")

    # Generate topic names based on cluster content
    print("Generating topic names...")
    topic_names = {}
    for model in ["gpt", "claude", "gemini"]:
        topic_names[model] = {}
        for cluster_id in range(n_clusters):
            cluster_points = [p for p in points if p["cluster"] == cluster_id]
            if cluster_points:
                # Simple topic naming based on common subreddit
                subreddits = [p["subreddit"] for p in cluster_points]
                most_common = max(set(subreddits), key=subreddits.count)

                # Check content for themes
                all_text = " ".join(p["sentence"].lower() for p in cluster_points[:20])
                if "steve kerr" in all_text or "kerr" in all_text:
                    name = "Kerr Discussion"
                elif "suspension" in all_text or "punch" in all_text:
                    name = "Suspensions"
                elif "trade" in all_text and "rumor" in all_text:
                    name = "Trade Rumors"
                elif "dynasty" in all_text or "championship" in all_text:
                    name = "Dynasty Talk"
                elif "contract" in all_text or "salary" in all_text:
                    name = "Contract Talk"
                elif "steph" in all_text or "curry" in all_text:
                    name = "Steph/Curry"
                elif "warriors" in most_common:
                    name = f"Warriors Fans"
                else:
                    name = f"r/{most_common}"

                topic_names[model][str(cluster_id)] = name

    with open(data_dir / "topic_names.json", "w") as f:
        json.dump(topic_names, f, indent=2)

    # Generate cluster stats
    print("Generating cluster stats...")
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
                "gpt_mean": np.mean([p["gpt"] for p in cluster_points]),
                "claude_mean": np.mean([p["claude"] for p in cluster_points]),
                "gemini_mean": np.mean([p["gemini"] for p in cluster_points])
            }

    with open(data_dir / "cluster_stats.json", "w") as f:
        json.dump(cluster_stats, f, indent=2)

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Processed {len(points)} Reddit items")
    print(f"Found {n_clusters} clusters")
    print("\nRestart the Reddit server to see the new data:")
    print("  pkill -f 'reddit/server.py'")
    print("  python reddit/server.py")
    print("  # Visit http://localhost:8001")


if __name__ == "__main__":
    main()
