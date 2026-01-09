# Agent Handoff Document - Multi-Source Sentiment Visualization

## Project Overview
A Three.js-based interactive visualization comparing Draymond Green trade sentiments across **multiple sources**: News, Reddit, and YouTube. Each source gets its own visualization with the same rating scale for direct comparison.

**Rating Question**: "How likely does this imply Draymond Green will be traded?" (1-10 scale)

**Models Used** (via litellm):
- GPT: `openai/gpt-5-nano` (reasoning model - needs max_tokens=1000)
- Claude: `anthropic/claude-sonnet-4-5`
- Gemini: `gemini/gemini-2.5-flash`

**Note**: GPT-5-nano is a reasoning model. It uses tokens for internal "thinking" before output. With low max_tokens, all tokens go to reasoning, leaving nothing for visible output. Fix: set `max_tokens=1000`.

---

## Current Status (Jan 8, 2026)

| Source | Status | Port | Items | URL |
|--------|--------|------|-------|-----|
| News | Working | 8000 | 200 | http://localhost:8000 |
| Reddit | Working | 8001 | 866 | http://localhost:8001 |
| YouTube | Working | 8002 | 836 | http://localhost:8002 |

---

## Known Deficiencies (To Fix Eventually)

### 1. UMAP Not Working
- **Problem**: `shared/clustering.py` UMAP hangs (>5 min) even on small datasets
- **Current Workaround**: Using PCA instead (~15% variance explained)
- **Impact**: PCA produces less meaningful clusters than UMAP would
- **TODO**: Debug UMAP or try different implementation (umap-learn version issue?)

### 2. Mock Scores Instead of Real LLM Ratings
- **Problem**: Reddit and YouTube use keyword-based mock scores, not real LLM ratings
- **Current Workaround**: Scores based on keyword matching ("trade him" = +2, "keep him" = -2, etc.)
- **Impact**: Scores don't reflect actual LLM judgment
- **TODO**: Enable real rating by calling GPT/Claude/Gemini for each item (slow, ~3 items/sec)
- **How to enable**: Set `rate=True` in processing scripts, but expect ~5+ minutes for 800 items

### 3. Google GenAI Package Deprecated
- **Problem**: `google.generativeai` package shows deprecation warning
- **TODO**: Migrate to `google.genai` package

---

## Project Structure

```
psychology_of_news/
├── .env                          # API keys (OpenAI, Anthropic, Google, Reddit, YouTube)
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
├── reddit/                       # Reddit visualization
│   ├── server.py                 # FastAPI server (port 8001)
│   ├── static/
│   │   └── index.html            # Reddit-themed UI (orange badges, subreddit tags)
│   └── data/
│       ├── raw_reddit_real.json  # Raw collected data (866 items)
│       ├── points_data.json      # Processed visualization data
│       ├── topic_names.json      # Per-model topic labels
│       └── cluster_stats.json    # Cluster centroids and stats
│
├── youtube/                      # YouTube visualization
│   ├── server.py                 # FastAPI server (port 8002)
│   ├── static/
│   │   └── index.html            # YouTube-themed UI (red badges, channel names)
│   └── data/
│       ├── raw_youtube.json      # Raw collected data (836 items)
│       ├── points_data.json      # Processed visualization data
│       ├── topic_names.json      # Per-model topic labels
│       └── cluster_stats.json    # Cluster centroids and stats
│
├── shared/                       # Shared components
│   ├── __init__.py
│   └── embeddings.py             # Sentence-transformer wrapper (all-MiniLM-L6-v2)
│
├── collectors/                   # Data collection scripts
│   ├── __init__.py
│   ├── reddit.py                 # PRAW-based Reddit collector
│   └── youtube.py                # YouTube Data API v3 collector
│
├── scripts/                      # Processing scripts
│   ├── process_reddit_fast.py    # PCA-based Reddit processing
│   ├── process_youtube_fast.py   # PCA-based YouTube processing
│   ├── label_reddit_topics.py    # Generate per-model topic labels for Reddit
│   └── label_youtube_topics.py   # Generate per-model topic labels for YouTube
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
- GPT: More neutral ("Draymond Green Controversy", "Trade Speculation")
- Claude: More dramatic ("Draymond Drama Fatigue", "Warriors Drama Unfolding")
- Gemini: More specific ("Anti-Draymond Sentiment", "Kerr Outdated System")

### 3. RAG Chat Sidebar
- Semantic search + keyword boost for context retrieval
- Director agent generates responses with citations
- Citation cards show relevant posts with scores
- Hover citation to see full tooltip, click to fly to point

### 4. Source-Specific UI
- **News**: Gold theme, "Click to open article"
- **Reddit**: Orange theme, subreddit badges (r/nba, r/warriors), POST/COMMENT type labels, upvote scores
- **YouTube**: Red theme, channel name badges, VIDEO/COMMENT type labels, like counts

---

## YouTube Implementation Details

### Data Collection (`collectors/youtube.py`)
```python
from collectors.youtube import collect_youtube

data = collect_youtube(
    query="draymond green trade",
    max_videos=30,
    max_comments=50,
    output_path="youtube/data/raw_youtube.json"
)
```

**Credentials**: Set `YOUTUBE_API_KEY` in `.env`

**Output Fields**:
- `text`: Video title+description or comment text
- `video_id`: YouTube video ID
- `video_title`: Parent video title
- `channel_name`: Video uploader or commenter
- `type`: "video" or "comment"
- `like_count`: Likes on comment
- `url`: Direct YouTube link

### Processing (`scripts/process_youtube_fast.py`)
Same as Reddit - uses PCA instead of UMAP:
1. Generate embeddings (sentence-transformers)
2. PCA for 3D projection
3. KMeans clustering (8 clusters)
4. Mock scores based on keywords

### Topic Labeling (`scripts/label_youtube_topics.py`)
Calls GPT, Claude, and Gemini to generate unique labels per cluster.

### Running YouTube Server
```bash
cd /Users/raymondli701/2026_01_07_workspace/psychology_of_news
source venv/bin/activate
set -a && source .env && set +a
python youtube/server.py
# Visit http://localhost:8002
```

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

### Processing (`scripts/process_reddit_fast.py`)
Uses PCA instead of UMAP:
1. Generate embeddings (sentence-transformers)
2. PCA for 3D projection (~14% variance explained)
3. KMeans clustering (8 clusters)
4. Mock scores based on keywords

### Topic Labeling (`scripts/label_reddit_topics.py`)
Calls GPT, Claude, and Gemini to generate unique labels per cluster.

---

## API Keys Required (.env)

```
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-api03-...
GOOGLE_API_KEY=AIzaSy...
REDDIT_CLIENT_ID=qfA94D_...
REDDIT_CLIENT_SECRET=hsEtrUgUdfc9...
YOUTUBE_API_KEY=AIzaSyBd...
EVENT_REGISTRY_API_KEY=ca6db172-...
```

---

## Running All Three Visualizations

```bash
cd /Users/raymondli701/2026_01_07_workspace/psychology_of_news
source venv/bin/activate
set -a && source .env && set +a

# Terminal 1: News (port 8000)
python server.py

# Terminal 2: Reddit (port 8001)
python reddit/server.py

# Terminal 3: YouTube (port 8002)
python youtube/server.py
```

Or run all in background:
```bash
python server.py &
python reddit/server.py &
python youtube/server.py &
```

---

## Data Flow

```
Raw Data (API/Collection)
    ↓
Embedding Generation (sentence-transformers all-MiniLM-L6-v2)
    ↓
Dimensionality Reduction (PCA 3D - faster than UMAP but less accurate)
    ↓
Clustering (KMeans, 8 clusters)
    ↓
Topic Labeling (GPT, Claude, Gemini each generate labels)
    ↓
Score Rating (keyword-based mock - TODO: real LLM calls)
    ↓
JSON Output (points_data.json, topic_names.json, cluster_stats.json)
    ↓
FastAPI Server (serves data + RAG chat)
    ↓
Three.js Visualization (browser)
```

---

## Future Improvements

### High Priority (Deficiencies)
- [x] Fix UMAP performance - FIXED! Works fine, first run is slow due to JIT compilation
- [ ] Add real LLM rating instead of keyword-based mock scores
- [ ] Migrate to `google.genai` package (deprecation warning)

### Nice to Have
- [ ] YouTube video transcripts (not just comments) - use YouTube Transcript API
- [ ] Create comparison dashboard (side-by-side iframes)
- [ ] Add time-series analysis if data has timestamps
- [ ] Cross-source analysis (compare sentiments across News vs Reddit vs YouTube)

---

## Useful Commands

```bash
# Collect new YouTube data
python -c "from collectors.youtube import collect_youtube; collect_youtube('draymond green trade', output_path='youtube/data/raw_youtube.json')"

# Process YouTube data
python scripts/process_youtube_fast.py

# Generate YouTube topic labels
python scripts/label_youtube_topics.py

# Same for Reddit
python -c "from collectors.reddit import collect_reddit; collect_reddit('draymond green trade', output_path='reddit/data/raw_reddit_real.json')"
python scripts/process_reddit_fast.py
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

youtube/data/points_data.json: 836 items
youtube/data/topic_names.json: 8 clusters × 3 models
youtube/data/raw_youtube.json: 836 raw items
```

---

*Last updated: Jan 8, 2026*
