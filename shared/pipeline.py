"""End-to-end processing pipeline for any source."""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from .embeddings import generate_embeddings
from .clustering import run_umap_3d, run_dbscan, compute_cluster_stats, auto_tune_dbscan
from .rating import rate_texts
from .topics import name_all_clusters


@dataclass
class ProcessedItem:
    """A fully processed text item."""
    id: int
    text: str
    source_type: str  # 'news', 'reddit', 'youtube'
    source_url: Optional[str]
    source_metadata: Dict

    # Coordinates
    x: float
    y: float
    z: float
    cluster: int

    # Scores
    gpt: int
    claude: int
    gemini: int
    mean: float


class ProcessingPipeline:
    """Process raw text items through the full pipeline."""

    def __init__(self, source_type: str, output_dir: str):
        self.source_type = source_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def process(
        self,
        items: List[Dict],
        text_key: str = "text",
        rate_texts_flag: bool = True,
        target_clusters: int = 10
    ) -> List[ProcessedItem]:
        """Process items through the full pipeline."""
        print(f"Processing {len(items)} {self.source_type} items...")

        # Extract texts
        texts = [item[text_key] for item in items]

        # 1. Generate embeddings
        print("\n[1/5] Generating embeddings...")
        embeddings = generate_embeddings(texts)

        # 2. UMAP projection
        print("\n[2/5] Running UMAP 3D projection...")
        coords = run_umap_3d(embeddings)

        # 3. Clustering
        print("\n[3/5] Clustering with DBSCAN...")
        _, labels = auto_tune_dbscan(coords, target_clusters=target_clusters)

        # 4. Rating (optional - can skip if already rated)
        if rate_texts_flag:
            print("\n[4/5] Rating with triple LLMs...")
            ratings = await rate_texts(texts, batch_size=5)
        else:
            print("\n[4/5] Skipping rating (using existing scores)...")
            ratings = [
                {"gpt": item.get("gpt", 5), "claude": item.get("claude", 5), "gemini": item.get("gemini", 5)}
                for item in items
            ]

        # 5. Topic naming
        print("\n[5/5] Naming topics...")
        topic_names = await name_all_clusters(texts, labels.tolist())

        # Build processed items
        processed = []
        for i, (item, coord, label, rating) in enumerate(zip(items, coords, labels, ratings)):
            mean_score = (rating["gpt"] + rating["claude"] + rating["gemini"]) / 3

            processed.append(ProcessedItem(
                id=i,
                text=item[text_key],
                source_type=self.source_type,
                source_url=item.get("url") or item.get("source_url"),
                source_metadata={k: v for k, v in item.items() if k != text_key},
                x=float(coord[0]),
                y=float(coord[1]),
                z=float(coord[2]),
                cluster=int(label),
                gpt=rating["gpt"],
                claude=rating["claude"],
                gemini=rating["gemini"],
                mean=mean_score
            ))

        # Save outputs
        self._save_outputs(processed, coords, labels, topic_names, ratings)

        print(f"\nProcessing complete! Saved to {self.output_dir}")
        return processed

    def _save_outputs(
        self,
        items: List[ProcessedItem],
        coords: np.ndarray,
        labels: np.ndarray,
        topic_names: Dict[int, str],
        ratings: List[Dict]
    ):
        """Save all outputs for the visualization."""
        # Points data (for Three.js)
        points_data = []
        for item in items:
            point = asdict(item)
            # Flatten source_metadata into point for easier access
            meta = point.pop("source_metadata", {})
            point.update(meta)
            # Rename text to sentence for compatibility
            point["sentence"] = point.pop("text")
            # Add source field
            point["source"] = meta.get("source") or meta.get("subreddit") or meta.get("video_title") or self.source_type
            points_data.append(point)

        with open(self.output_dir / "points_data.json", "w") as f:
            json.dump(points_data, f, indent=2)

        # Topic names (use same for all models for simplicity)
        topic_dict = {
            "gpt": {str(k): v for k, v in topic_names.items()},
            "claude": {str(k): v for k, v in topic_names.items()},
            "gemini": {str(k): v for k, v in topic_names.items()}
        }
        with open(self.output_dir / "topic_names.json", "w") as f:
            json.dump(topic_dict, f, indent=2)

        # Cluster stats
        score_arrays = {
            "gpt": np.array([r["gpt"] for r in ratings]),
            "claude": np.array([r["claude"] for r in ratings]),
            "gemini": np.array([r["gemini"] for r in ratings])
        }
        cluster_stats = compute_cluster_stats(coords, labels, score_arrays)
        with open(self.output_dir / "cluster_stats.json", "w") as f:
            json.dump(cluster_stats, f, indent=2)

        # CSV for reference
        df = pd.DataFrame([asdict(item) for item in items])
        df.to_csv(self.output_dir / "processed_data.csv", index=False)

        print(f"Saved: points_data.json, topic_names.json, cluster_stats.json, processed_data.csv")


async def process_source(
    source_type: str,
    items: List[Dict],
    output_dir: str,
    text_key: str = "text",
    rate: bool = True
) -> List[ProcessedItem]:
    """Convenience function to process a source."""
    pipeline = ProcessingPipeline(source_type, output_dir)
    return await pipeline.process(items, text_key=text_key, rate_texts_flag=rate)
