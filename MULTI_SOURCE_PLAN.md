# Multi-Source Sentiment Comparison: News vs Reddit vs YouTube

## Vision
Compare Draymond Green trade sentiments across three distinct platforms:
- **News** - Professional journalism (already have)
- **Reddit** - Community discussion (r/nba, r/warriors)
- **YouTube** - Video comments

Each gets its own interactive 3D visualization with RAG chat, using the same rating scale for direct comparison.

---

## Architecture Decision

### Option A: Three Separate Apps (RECOMMENDED)
```
psychology_of_news/
├── news/           # Current implementation
├── reddit/         # New - Reddit sentiment vis
└── youtube/        # New - YouTube sentiment vis
```

**Pros:**
- Cleanest separation, easiest to develop incrementally
- Each can run independently
- Source-specific customization (subreddit filters, video metadata)

**Cons:**
- Some code duplication (can mitigate with shared components)

### Future: Meta-Dashboard
Once all three exist, build a comparison dashboard that:
- Shows aggregate stats side-by-side
- Allows cross-source queries ("What does Reddit say vs News about Kerr?")
- Time-series comparison if data spans multiple dates

---

## Unified Rating Scale (Applied to All Sources)

**Question:** "How likely does this imply Draymond Green will be traded?"

| Score | Meaning |
|-------|---------|
| 1 | No trade implication at all |
| 2-3 | Minimal/unlikely trade signals |
| 4-5 | Neutral/ambiguous |
| 6-7 | Moderate trade signals |
| 8-9 | Strong trade indicators |
| 10 | Trade very likely/imminent |

**Models:**
- GPT-5-nano
- Claude Sonnet 4.5
- Gemini 2.5 Flash

---

## Data Collection Pipelines

### 1. Reddit Collection (adapt from variable_resolution)

**Source:** `variable_resolution/reddit_data_collector/reddit_search.py`

**Parameters:**
- Query: "draymond green trade"
- Subreddits: r/nba, r/warriors, r/NBATrade, r/GoldenStateWarriors
- Time filter: relevant period (e.g., last 3 months)
- N posts, K comments per post

**Output Fields:**
```python
{
    "id": int,                    # unique identifier
    "text": str,                  # post body or comment body
    "type": "post" | "comment",   # distinguish posts from comments
    "subreddit": str,             # r/nba, r/warriors, etc.
    "post_title": str,            # parent post title (for context)
    "author": str,
    "score": int,                 # upvotes
    "created_utc": datetime,
    "url": str,                   # permalink
    "parent_id": str | None       # for threading
}
```

**Collection Script Structure:**
```python
# collect_reddit.py
from praw import Reddit

def collect_reddit_data(
    query: str,
    subreddits: list[str],
    max_posts: int = 100,
    max_comments_per_post: int = 50,
    time_filter: str = "month"  # hour, day, week, month, year, all
) -> list[dict]:
    """
    1. Search each subreddit for query
    2. For each post, extract body + top comments
    3. Flatten to list of text items
    4. Return with metadata
    """
```

### 2. YouTube Collection (adapt from variable_resolution)

**Source:** `variable_resolution/youtube_data_collector/youtube_api.py`

**Parameters:**
- Query: "draymond green trade"
- Max videos: 50
- Max comments per video: 100

**Output Fields:**
```python
{
    "id": int,                    # unique identifier
    "text": str,                  # comment text
    "video_id": str,              # YouTube video ID
    "video_title": str,           # for context
    "author": str,
    "like_count": int,
    "published_at": datetime,
    "is_reply": bool,
    "parent_id": str | None       # for threading
}
```

**Collection Script Structure:**
```python
# collect_youtube.py
from googleapiclient.discovery import build

def collect_youtube_comments(
    query: str,
    max_videos: int = 50,
    max_comments_per_video: int = 100
) -> list[dict]:
    """
    1. Search YouTube for query videos
    2. For each video, fetch top comments
    3. Flatten to list of comment items
    4. Return with video metadata
    """
```

### 3. News Collection (existing)

Already have Event Registry integration. Current 200 sentences are from news articles.

---

## Processing Pipeline (Shared Across Sources)

```
Raw Text Items
    ↓
Sentence Segmentation (if needed)
    - News: Already sentence-level
    - Reddit: May need splitting for long posts
    - YouTube: Usually sentence-level comments
    ↓
Embedding Generation
    - Model: sentence-transformers/all-MiniLM-L6-v2
    - Output: 384-dim vectors
    ↓
UMAP 3D Projection
    - n_components=3
    - n_neighbors=15
    - min_dist=0.1
    ↓
DBSCAN Clustering
    - eps=0.5
    - min_samples=5
    ↓
Topic Naming (per cluster)
    - LLM summarizes cluster content
    - Returns short topic label
    ↓
Triple LLM Rating
    - GPT-5-nano rating
    - Claude Sonnet 4.5 rating
    - Gemini 2.5 Flash rating
    - All use same prompt/scale
    ↓
SQLite Database
    - Same schema for all sources
    - Source-specific metadata columns
```

### Database Schema (per source)

```sql
CREATE TABLE sentences (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    source_type TEXT,           -- 'news', 'reddit', 'youtube'
    source_url TEXT,
    source_metadata JSON,       -- source-specific fields
    created_at DATETIME,

    -- Embeddings & Projections
    embedding BLOB,             -- 384-dim vector
    umap_x REAL,
    umap_y REAL,
    umap_z REAL,
    cluster INTEGER,

    -- LLM Ratings
    gpt_score INTEGER,
    claude_score INTEGER,
    gemini_score INTEGER,
    mean_score REAL,

    -- Computed
    agreement_std REAL          -- std dev across models
);

CREATE TABLE topics (
    cluster_id INTEGER PRIMARY KEY,
    name TEXT,
    centroid_x REAL,
    centroid_y REAL,
    centroid_z REAL,
    count INTEGER
);
```

---

## Visualization Structure

Each source gets its own directory with identical structure:

```
psychology_of_news/
├── news/                       # EXISTING (rename from current)
│   ├── server.py
│   ├── director.py
│   ├── static/
│   │   └── index.html
│   └── data/
│       └── news_embeddings.db
│
├── reddit/                     # NEW
│   ├── server.py              # Copy, modify data source
│   ├── director.py            # Copy, modify prompts
│   ├── static/
│   │   └── index.html         # Copy, modify title/branding
│   └── data/
│       └── reddit_embeddings.db
│
├── youtube/                    # NEW
│   ├── server.py
│   ├── director.py
│   ├── static/
│   │   └── index.html
│   └── data/
│       └── youtube_embeddings.db
│
├── shared/                     # SHARED COMPONENTS
│   ├── embeddings.py          # Sentence transformer wrapper
│   ├── clustering.py          # UMAP + DBSCAN
│   ├── rating.py              # Triple LLM rating
│   └── db_schema.py           # SQLAlchemy models
│
├── collectors/                 # DATA COLLECTION
│   ├── collect_news.py        # Event Registry
│   ├── collect_reddit.py      # PRAW
│   └── collect_youtube.py     # YouTube API
│
└── scripts/                    # UTILITY SCRIPTS
    ├── process_source.py      # End-to-end pipeline
    └── compare_sources.py     # Cross-source analysis
```

---

## UI Customization Per Source

### News Visualization
- Title: "News Sentiment: Draymond Green Trade"
- Tooltip shows: Publication, Author, URL
- Citation cards show article source

### Reddit Visualization
- Title: "Reddit Sentiment: Draymond Green Trade"
- Color subreddit badges (r/nba = blue, r/warriors = gold)
- Tooltip shows: Subreddit, Score (upvotes), Author
- Citation cards show post title + subreddit

### YouTube Visualization
- Title: "YouTube Sentiment: Draymond Green Trade"
- Tooltip shows: Video title, Like count, Channel
- Citation cards show video thumbnail (small)

---

## Implementation Phases

### Phase 1: Reddit Integration
1. Create `collectors/collect_reddit.py` adapting from variable_resolution
2. Create `reddit/` directory structure
3. Run collection for "draymond green trade"
4. Process through shared pipeline
5. Launch Reddit visualization

### Phase 2: YouTube Integration
1. Create `collectors/collect_youtube.py` adapting from variable_resolution
2. Create `youtube/` directory structure
3. Run collection for "draymond green trade"
4. Process through shared pipeline
5. Launch YouTube visualization

### Phase 3: Comparison Dashboard (Future)
1. Side-by-side iframe layout
2. Cross-source aggregate stats
3. Unified search across all sources

---

## API Keys Required

| Service | Environment Variable | Purpose |
|---------|---------------------|---------|
| Reddit | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` | PRAW authentication |
| YouTube | `YOUTUBE_API_KEY` | YouTube Data API v3 |
| Event Registry | `EVENT_REGISTRY_API_KEY` | News collection |
| OpenAI | `OPENAI_API_KEY` | GPT ratings |
| Anthropic | `ANTHROPIC_API_KEY` | Claude ratings |
| Google | `GOOGLE_API_KEY` | Gemini ratings |

---

## Sample Queries Across Sources

Once all three visualizations exist, interesting comparisons:

1. **"What's the consensus on Steve Kerr's role?"**
   - News: Coach perspective, organizational quotes
   - Reddit: Fan opinions, hot takes
   - YouTube: Commentator analysis

2. **"What trade partners are mentioned?"**
   - News: Official reporting, rumored destinations
   - Reddit: Fan speculation, wish lists
   - YouTube: Trade proposal videos

3. **"Overall sentiment distribution"**
   - Compare mean scores across sources
   - Reddit likely more extreme (hot takes)
   - News likely more neutral (journalistic balance)

---

## Technical Notes

### Reddit Rate Limits
- PRAW handles rate limiting automatically
- ~60 requests/minute for authenticated users
- Batch collection recommended

### YouTube Quota
- 10,000 units/day default quota
- Search: 100 units
- Comments: 1 unit per request
- Plan collection carefully

### Embedding Consistency
- **Critical:** Use same embedding model across all sources
- Allows meaningful cross-source comparison
- Current: `sentence-transformers/all-MiniLM-L6-v2`

### Rating Consistency
- **Critical:** Use exact same prompt for all sources
- Temperature=0 for reproducibility
- Batch processing with rate limit handling

---

## File Sizes & Performance

Expected data volumes:
- News: ~200-500 sentences (current: 200)
- Reddit: ~500-2000 comments (depends on search depth)
- YouTube: ~500-1500 comments (50 videos × 30 comments)

Visualization performance:
- Three.js handles 2000+ points smoothly
- May need LOD for larger datasets
- SQLite scales fine for this volume

---

## Next Steps

1. **Set up credentials** - Reddit API, YouTube API
2. **Create shared/ directory** - Extract reusable components from current code
3. **Build collectors/** - Adapt from variable_resolution
4. **Test Reddit collection** - Start small, verify data quality
5. **Process & visualize Reddit** - First parallel source
6. **Repeat for YouTube**
7. **Build comparison dashboard**
