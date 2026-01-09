# Agent Handoff Document - Multi-Source Sentiment Visualization

## Project Overview
A Three.js-based interactive visualization comparing Draymond Green trade sentiments across **multiple sources**: News, Reddit, and YouTube (planned). Each source gets its own visualization with the same rating scale for direct comparison.

**Rating Question**: "How likely does this imply Draymond Green will be traded?" (1-10 scale)

**Models Used**:
- GPT: `gpt-4o-mini` (for rating/labeling)
- Claude: `claude-3-5-haiku-latest`
- Gemini: `gemini-2.0-flash`

---

## Current Status (Jan 8, 2026)

| Source | Status | Port | Items | URL |
|--------|--------|------|-------|-----|
| News | Working | 8000 | 200 | http://localhost:8000 |
| Reddit | Working | 8001 | 866 | http://localhost:8001 |
| YouTube | Planned | 8002 | - | - |

---

## Project Structure

```
psychology_of_news/
├── .env                          # API keys (OpenAI, Anthropic, Google, Reddit, EventRegistry)
├── server.py                     # News visualization server (port 8000)
├── AGENT_HANDOFF.md              # This file
├── MULTI_SOURCE_PLAN.md          # Detailed architecture planning doc
│
├── psychology_of_news/           # Original news package
│   ├── director.py               # Pydantic AI director agent
│   ├── static/
│   │   └── index.html            # News visualization UI
│   └── ...
│
├── reddit/                       # Reddit visualization (NEW)
│   ├── server.py                 # FastAPI server (port 8001)
│   ├── static/
│   │   └── index.html            # Reddit-themed UI (orange badges, subreddit tags)
│   └── data/
│       ├── raw_reddit_real.json  # Raw collected data (866 items)
│       ├── points_data.json      # Processed visualization data
│       ├── topic_names.json      # Per-model topic labels
│       └── cluster_stats.json    # Cluster centroids and stats
│
├── youtube/                      # YouTube visualization (PLANNED)
│   ├── server.py
│   ├── static/
│   │   └── index.html
│   └── data/
│
├── shared/                       # Shared components
│   ├── __init__.py
│   ├── embeddings.py             # Sentence-transformer wrapper (all-MiniLM-L6-v2)
│   ├── clustering.py             # UMAP + DBSCAN (has issues, use PCA instead)
│   ├── rating.py                 # Triple LLM rating
│   ├── topics.py                 # LLM topic naming
│   └── pipeline.py               # End-to-end processing (has UMAP issues)
│
├── collectors/                   # Data collection scripts
│   ├── __init__.py
│   └── reddit.py                 # PRAW-based Reddit collector
│
├── scripts/                      # Processing scripts
│   ├── process_reddit.py         # Full pipeline (UMAP issues)
│   ├── process_reddit_fast.py    # PCA-based processing (RECOMMENDED)
│   ├── quick_reddit_test.py      # Mock data generator for testing
│   ├── label_reddit_topics.py    # Generate per-model topic labels
│   └── test_reddit_mock.py       # Old mock test (UMAP hangs)
│
└── test_output/                  # News data output
    └── run_20260108_132100/
        ├── points_data.json
        ├── topic_names.json
        └── cluster_stats.json
```

---

## Key Features

### 1. 3D Scatter Plot (Three.js)
- Points colored by selected model's score (blue=1, yellow=5, red=10)
- OrbitControls for camera navigation
- Click point to open source URL
- Hover for detailed tooltip

### 2. Per-Model Topic Labels
Each model (GPT, Claude, Gemini) generates its own cluster labels. When switching models, labels change:
- GPT: More neutral ("Draymond Green Controversy")
- Claude: More dramatic ("Draymond Drama", "Player Meltdown")
- Gemini: More specific ("Mavs Trade Interest")

### 3. RAG Chat Sidebar
- Semantic search + keyword boost for context retrieval
- Director agent generates responses with citations
- Citation cards show relevant posts with scores
- Hover citation to see full tooltip, click to fly to point

### 4. Source-Specific UI
- **News**: Standard gold theme, "Click to open article"
- **Reddit**: Orange subreddit badges (r/nba, r/warriors), POST/COMMENT type labels, upvote scores

---

## Reddit Implementation Details

### Data Collection (`collectors/reddit.py`)
```python
from collectors.reddit import collect_reddit

data = collect_reddit(
    query="draymond green trade",
    subreddits=["nba", "warriors", "nbadiscussion"],
    max_posts=30,
    max_comments=15,
    time_filter="month"
)
```

**Credentials**: Set `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` in `.env`

**Output Fields**:
- `text`: Post body or comment text
- `subreddit`: Source subreddit
- `type`: "post" or "comment"
- `score`: Upvotes
- `url`: Permalink
- `post_title`: Parent post title

### Processing (`scripts/process_reddit_fast.py`)
Uses PCA instead of UMAP (UMAP hangs on this machine):
1. Generate embeddings (sentence-transformers)
2. PCA for 3D projection (fast, ~14% variance explained)
3. KMeans clustering (8 clusters)
4. Mock scores based on keywords (real LLM rating available but slow)

### Topic Labeling (`scripts/label_reddit_topics.py`)
Calls GPT, Claude, and Gemini to generate unique labels per cluster:
```bash
source venv/bin/activate
set -a && source .env && set +a
python scripts/label_reddit_topics.py
```

### Running Reddit Server
```bash
cd /Users/raymondli701/2026_01_07_workspace/psychology_of_news
source venv/bin/activate
set -a && source .env && set +a
python reddit/server.py
# Visit http://localhost:8001
```

---

## Known Issues & Quirks

### 1. UMAP Hangs on Small Datasets
**Problem**: `shared/clustering.py` UMAP takes forever (>5 min) even on 39 items.
**Workaround**: Use `process_reddit_fast.py` which uses PCA instead.
**TODO**: Debug UMAP or switch to different implementation.

### 2. Citation Tooltip Race Condition (FIXED)
**Problem**: Canvas mousemove handler hides chat citation tooltips immediately.
**Solution**: `window.chatTooltipActive` flag prevents canvas handler from hiding chat tooltips.

### 3. Google GenAI Deprecation Warning
```
FutureWarning: All support for the `google.generativeai` package has ended.
```
**TODO**: Migrate to `google.genai` package.

### 4. Mock Scores vs Real LLM Scores
Reddit currently uses keyword-based mock scores for speed. To use real LLM ratings:
- Set `rate=True` in `process_source()` call
- Will call GPT, Claude, Gemini for each item (slow, ~3 items/sec)

### 5. Port Conflicts
- News: 8000
- Reddit: 8001
- YouTube (planned): 8002
Kill existing processes: `pkill -f 'server.py'`

---

## API Keys Required (.env)

```
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-api03-...
GOOGLE_API_KEY=AIzaSy...
REDDIT_CLIENT_ID=qfA94D_...
REDDIT_CLIENT_SECRET=hsEtrUgUdfc9...
EVENT_REGISTRY_API_KEY=ca6db172-...
```

---

## Running Both Visualizations

```bash
cd /Users/raymondli701/2026_01_07_workspace/psychology_of_news
source venv/bin/activate
set -a && source .env && set +a

# Terminal 1: News (port 8000)
python server.py

# Terminal 2: Reddit (port 8001)
python reddit/server.py
```

---

## Data Flow

```
Raw Data (API/Collection)
    ↓
Embedding Generation (sentence-transformers all-MiniLM-L6-v2)
    ↓
Dimensionality Reduction (PCA 3D - faster than UMAP)
    ↓
Clustering (KMeans, 8 clusters)
    ↓
Topic Labeling (GPT, Claude, Gemini each generate labels)
    ↓
Score Rating (keyword-based mock or real LLM calls)
    ↓
JSON Output (points_data.json, topic_names.json, cluster_stats.json)
    ↓
FastAPI Server (serves data + RAG chat)
    ↓
Three.js Visualization (browser)
```

---

## Next Steps

### YouTube Integration
1. Create `collectors/youtube.py` using YouTube Data API v3
2. Search "draymond green trade" videos
3. Collect video comments
4. Process same as Reddit
5. Create `youtube/server.py` and `youtube/static/index.html`

### Improvements
- [ ] Fix UMAP performance or permanently switch to PCA
- [ ] Add real LLM rating (currently using keyword-based mock)
- [ ] Create comparison dashboard (side-by-side iframes)
- [ ] Add time-series if data has timestamps
- [ ] Migrate to `google.genai` package

---

## Useful Commands

```bash
# Collect Reddit data
REDDIT_CLIENT_ID=xxx REDDIT_CLIENT_SECRET=xxx python -c "
from collectors.reddit import collect_reddit
collect_reddit('draymond green trade', output_path='reddit/data/raw.json')
"

# Process Reddit data
python scripts/process_reddit_fast.py

# Generate topic labels
python scripts/label_reddit_topics.py

# Kill all servers
pkill -f 'server.py'

# Check what's running
lsof -i:8000 -i:8001 -i:8002
```

---

## File Checksums (for verification)

```
reddit/data/points_data.json: 866 items
reddit/data/topic_names.json: 8 clusters × 3 models
reddit/data/raw_reddit_real.json: 866 raw items
```

---

*Last updated: Jan 8, 2026*
