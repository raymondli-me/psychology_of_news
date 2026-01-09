#!/usr/bin/env python3
"""Process YouTube data with UMAP (better clustering than PCA)."""

import json
import random
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import umap

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.embeddings import generate_embeddings


def main():
    data_dir = Path(__file__).parent.parent / "youtube" / "data"

    # Load raw YouTube data
    raw_path = data_dir / "raw_youtube.json"
    if not raw_path.exists():
        print(f"No data found at {raw_path}")
        return

    with open(raw_path) as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} YouTube items")

    # Extract texts
    texts = [item["text"] for item in raw_data]

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Use UMAP for 3D projection (better than PCA)
    print("Running UMAP for 3D projection...")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        n_jobs=1,
        metric="cosine"
    )
    coords = reducer.fit_transform(embeddings)
    print(f"UMAP complete, shape: {coords.shape}")

    # Scale coordinates for better visualization
    coords = coords * 2

    # Cluster with KMeans
    print("Clustering...")
    n_clusters = min(8, len(raw_data) // 10 + 1)
    n_clusters = max(3, n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Generate mock scores
    print("Generating mock scores...")
    points = []
    for i, (item, coord, label) in enumerate(zip(raw_data, coords, labels)):
        text_lower = item["text"].lower()

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
        if any(w in text_lower for w in ["mavs", "dallas", "mavericks"]):
            base += 1

        gpt = max(1, min(10, base + random.randint(-1, 1)))
        claude = max(1, min(10, base + random.randint(-1, 1)))
        gemini = max(1, min(10, base + random.randint(-1, 1)))

        point = {
            "id": i,
            "sentence": item["text"][:500],
            "text": item["text"][:500],
            "x": float(coord[0]),
            "y": float(coord[1]),
            "z": float(coord[2]),
            "cluster": int(label),
            "gpt": gpt,
            "claude": claude,
            "gemini": gemini,
            "mean": (gpt + claude + gemini) / 3,
            "source": f"YouTube: {item.get('channel_name', 'Unknown')[:20]}",
            "channel_name": item.get("channel_name", "Unknown"),
            "video_title": item.get("video_title", "")[:100],
            "type": item.get("type", "comment"),
            "like_count": item.get("like_count", 0),
            "url": item.get("url", ""),
            "video_id": item.get("video_id", "")
        }
        points.append(point)

    # Save points
    with open(data_dir / "points_data.json", "w") as f:
        json.dump(points, f)
    print(f"Saved {len(points)} points")

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
    print("PROCESSING COMPLETE (UMAP)")
    print("=" * 60)
    print(f"Processed {len(points)} YouTube items")
    print(f"Found {n_clusters} clusters")
    print("\nRun label_youtube_topics.py to generate per-model topic labels")
    print("Then restart the server: python youtube/server.py")


if __name__ == "__main__":
    main()
