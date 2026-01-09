"""UMAP dimensionality reduction and DBSCAN clustering."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import umap
from sklearn.cluster import DBSCAN


@dataclass
class ClusterStats:
    """Statistics for a cluster."""
    cluster_id: int
    count: int
    centroid: Tuple[float, float, float]
    indices: List[int]


def run_umap_3d(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """Project embeddings to 3D using UMAP."""
    n_samples = len(embeddings)
    print(f"Running UMAP on {n_samples} embeddings...")

    # Adjust n_neighbors for small datasets
    actual_neighbors = min(n_neighbors, n_samples - 1)
    if actual_neighbors < 2:
        actual_neighbors = 2

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=actual_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric="cosine",
        n_jobs=1,  # Single threaded for reproducibility
        low_memory=False  # Faster for small datasets
    )
    coords = reducer.fit_transform(embeddings)
    print("UMAP complete.")
    return coords


def run_dbscan(
    coords: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5
) -> np.ndarray:
    """Cluster points using DBSCAN."""
    print(f"Running DBSCAN (eps={eps}, min_samples={min_samples})...")
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = clusterer.fit_predict(coords)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"Found {n_clusters} clusters, {n_noise} noise points.")
    return labels


def compute_cluster_stats(
    coords: np.ndarray,
    labels: np.ndarray,
    scores: Optional[Dict[str, np.ndarray]] = None
) -> Dict[int, dict]:
    """Compute statistics for each cluster."""
    stats = {}
    unique_labels = set(labels)

    for label in unique_labels:
        mask = labels == label
        indices = np.where(mask)[0].tolist()
        cluster_coords = coords[mask]

        centroid = cluster_coords.mean(axis=0)

        stat = {
            "count": int(mask.sum()),
            "centroid": {
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "z": float(centroid[2])
            },
            "indices": indices
        }

        # Add score stats if provided
        if scores:
            for model, score_arr in scores.items():
                cluster_scores = score_arr[mask]
                stat[f"{model}_mean"] = float(cluster_scores.mean())
                stat[f"{model}_std"] = float(cluster_scores.std())

        stats[int(label)] = stat

    return stats


def auto_tune_dbscan(
    coords: np.ndarray,
    target_clusters: int = 10,
    eps_range: Tuple[float, float] = (0.3, 1.5),
    steps: int = 20
) -> Tuple[float, np.ndarray]:
    """Auto-tune DBSCAN eps to get close to target cluster count."""
    best_eps = 0.5
    best_labels = None
    best_diff = float("inf")

    for eps in np.linspace(eps_range[0], eps_range[1], steps):
        labels = run_dbscan(coords, eps=eps, min_samples=5)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        diff = abs(n_clusters - target_clusters)

        if diff < best_diff:
            best_diff = diff
            best_eps = eps
            best_labels = labels

        if n_clusters == target_clusters:
            break

    print(f"Best eps={best_eps:.2f} gives {len(set(best_labels)) - (1 if -1 in best_labels else 0)} clusters")
    return best_eps, best_labels
