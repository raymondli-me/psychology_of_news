"""FastAPI server for YouTube sentiment visualization."""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Add parent to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.embeddings import EmbeddingModel

app = FastAPI(title="YouTube Sentiment Visualization")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class DataManager:
    """Manage YouTube visualization data."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.points_data = None
        self.df = None
        self.embeddings = None
        self.cluster_stats = None
        self.topic_names = None
        self.embedding_model = None

    def load_data(self):
        """Load processed YouTube data."""
        if not self.data_dir.exists():
            print(f"Data directory {self.data_dir} not found.")
            return

        # Load points data
        points_path = self.data_dir / "points_data.json"
        if points_path.exists():
            with open(points_path) as f:
                self.points_data = json.load(f)
            self.df = pd.DataFrame(self.points_data)
            self.df["id"] = self.df.index
            if "sentence" in self.df.columns and "text" not in self.df.columns:
                self.df["text"] = self.df["sentence"]
            print(f"Loaded {len(self.df)} YouTube items.")
        else:
            print("No points_data.json found.")
            return

        # Load topic names
        names_path = self.data_dir / "topic_names.json"
        if names_path.exists():
            with open(names_path) as f:
                self.topic_names = json.load(f)

        # Load cluster stats
        stats_path = self.data_dir / "cluster_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                self.cluster_stats = json.load(f)

        # Generate embeddings for search
        if self.df is not None and "text" in self.df.columns:
            print("Generating search embeddings...")
            self.embedding_model = EmbeddingModel.get_instance()
            self.embeddings = self.embedding_model.encode(self.df["text"].tolist())
            print("Ready.")

    def search(self, query: str, top_k: int = 15) -> List[Dict]:
        """Hybrid semantic + keyword search."""
        if self.embeddings is None:
            return []

        # Semantic search
        query_vec = self.embedding_model.encode_single(query)
        semantic_scores = np.dot(self.embeddings, query_vec) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8
        )

        # Keyword boost
        query_lower = query.lower()
        keywords = [w for w in query_lower.split() if len(w) > 3]
        keyword_scores = np.zeros(len(self.df))
        for idx, row in self.df.iterrows():
            text_lower = str(row.get("text", "")).lower()
            matches = sum(1 for kw in keywords if kw in text_lower)
            keyword_scores[idx] = matches / max(len(keywords), 1)

        # Combine
        combined = semantic_scores * 0.7 + keyword_scores * 0.3
        top_indices = np.argsort(combined)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                "id": int(idx),
                "text": row.get("text", row.get("sentence", "")),
                "score": float(combined[idx]),
                "mean_score": float(row.get("mean", 5)),
                "cluster": int(row.get("cluster", -1)),
                "channel_name": row.get("channel_name", ""),
                "video_title": row.get("video_title", ""),
                "type": row.get("type", "comment"),
                "like_count": int(row.get("like_count", 0))
            })
        return results


# Global manager
data_manager = DataManager(str(Path(__file__).parent / "data"))


@app.on_event("startup")
async def startup():
    data_manager.load_data()


@app.get("/")
def root():
    """Serve the visualization."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse("<h1>YouTube Sentiment Visualization</h1><p>Run the processing pipeline first.</p>")


@app.get("/api/data")
def get_data():
    """Return all visualization data."""
    if data_manager.points_data is None:
        raise HTTPException(503, "Data not loaded")
    return {
        "points": data_manager.points_data,
        "cluster_stats": data_manager.cluster_stats,
        "topic_names": data_manager.topic_names
    }


class ChatRequest(BaseModel):
    query: str
    model: str = "gemini"


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """RAG chat endpoint."""
    if data_manager.df is None:
        raise HTTPException(503, "Data not loaded")

    # Get context
    context = data_manager.search(request.query, top_k=10)

    # For now, return context directly (can add Director agent later)
    highlighted_ids = [item["id"] for item in context]

    # Simple response (can enhance with Director agent)
    answer = f"Found {len(context)} relevant YouTube videos/comments about '{request.query}'."

    return {
        "answer": answer,
        "actions": [{"type": "highlight_points", "target": highlighted_ids, "description": "Highlighting relevant items"}],
        "context": context,
        "highlighted_ids": highlighted_ids,
        "model_used": request.model
    }


@app.get("/api/status")
def status():
    return {
        "status": "online",
        "source": "youtube",
        "items_loaded": len(data_manager.df) if data_manager.df is not None else 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)  # Port 8002 for YouTube
