import os
import glob
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from psychology_of_news.config import Config

app = FastAPI(title="Narrative Vis Director")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Mount static directory (inside psychology_of_news package)
static_dir = os.path.join(os.path.dirname(__file__), "psychology_of_news", "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    """Serve the Narrative Vis client."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Narrative Vis Server</h1><p>Frontend not found. Place index.html in static/</p>")


import json
from psychology_of_news.director import DirectorAgent, DirectorResponse

class DataManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.df = None
        self.points_data = None
        self.embeddings = None
        self.cluster_stats = None
        self.topic_names = None
        self.embedding_model = None

    def find_latest_run(self) -> str:
        """Find the most recent run directory."""
        runs = glob.glob(os.path.join(self.output_dir, "run_*"))
        if not runs:
            return None
        return max(runs, key=os.path.getmtime)
        
    def load_data(self):
        """Load data from the latest run."""
        run_dir = self.find_latest_run()
        if not run_dir:
            print("No run data found.")
            return

        print(f"Loading data from: {run_dir}")

        # Load Points Data (with x,y,z coordinates from UMAP)
        points_path = os.path.join(run_dir, "points_data.json")
        if os.path.exists(points_path):
            with open(points_path) as f:
                self.points_data = json.load(f)
            # Create DataFrame for search
            self.df = pd.DataFrame(self.points_data)
            self.df["id"] = self.df.index
            # Rename 'sentence' to 'text' for consistency
            if 'sentence' in self.df.columns and 'text' not in self.df.columns:
                self.df['text'] = self.df['sentence']
            print(f"Loaded {len(self.df)} points with coordinates.")
        else:
            # Fallback to CSV (no coordinates)
            csv_path = os.path.join(run_dir, "sentence_ratings.csv")
            if os.path.exists(csv_path):
                self.df = pd.read_csv(csv_path)
                self.df["id"] = self.df.index
                self.points_data = self.df.to_dict(orient="records")
                print(f"Loaded {len(self.df)} sentences (no coords).")

        # Load Topic Names
        names_path = os.path.join(run_dir, "topic_names.json")
        if os.path.exists(names_path):
            with open(names_path) as f:
                self.topic_names = json.load(f)
            print("Loaded topic names.")

        # Load Cluster Stats
        stats_path = os.path.join(run_dir, "cluster_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                self.cluster_stats = json.load(f)
            print("Loaded cluster stats.")

        # Generate embeddings for semantic search
        if self.df is not None and 'text' in self.df.columns:
            print("Generating embeddings for search...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings = self.embedding_model.encode(self.df['text'].tolist(), show_progress_bar=True)
            print("Embeddings ready.")

    def search(self, query: str, top_k: int = 15):
        """Semantic search for RAG."""
        if self.embeddings is None:
            return []
            
        query_vec = self.embedding_model.encode([query])[0]
        scores = np.dot(self.embeddings, query_vec) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        )
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                "id": int(idx),
                "text": row["text"],
                "score": float(scores[idx]),
                "mean_score": float(row.get("mean_score", 5)),
                "cluster": int(row.get("cluster", -1)) if "cluster" in row else -1,
                "gpt_score": int(row.get("GPT_score", 5)),
                "claude_score": int(row.get("Claude_score", 5)),
                "gemini_score": int(row.get("Gemini_score", 5)),
            })
        return results

# Global Managers
data_manager = DataManager("./test_output")
director = DirectorAgent()

@app.on_event("startup")
async def startup_event():
    data_manager.load_data()

@app.get("/api/status")
def status():
    return {
        "status": "online", 
        "sentences_loaded": len(data_manager.df) if data_manager.df is not None else 0
    }
    
@app.get("/api/data")
def get_data():
    """Serve full data for the frontend visualization initialization."""
    if data_manager.points_data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    # Return everything needed for Three.js
    return {
        "points": data_manager.points_data,
        "cluster_stats": data_manager.cluster_stats,
        "topic_names": data_manager.topic_names
    }

class QueryRequest(BaseModel):
    query: str
    model: str = "gemini"  # Default to Gemini (cheapest/fastest)

@app.post("/api/chat")
async def chat(request: QueryRequest):
    if data_manager.df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    # 1. Retrieve Context
    context = data_manager.search(request.query, top_k=10)

    # 2. Get Topic Names for context (match the model being used)
    model_key = request.model.lower()
    topic_map = data_manager.topic_names.get(model_key, {}) if data_manager.topic_names else {}
    # Fallback to gpt if model not found
    if not topic_map and data_manager.topic_names:
        topic_map = data_manager.topic_names.get("gpt", {})

    # 3. Call Director Agent with selected model
    response = await director.direct(
        query=request.query,
        context_sentences=context,
        topic_names=topic_map,
        model=request.model
    )

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
