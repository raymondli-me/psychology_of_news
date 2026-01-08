# Narrative Vis: Chat-Driven 3D Visualization Architecture

> Interactive RAG system that lets users chat with their data while the AI animates the visualization

---

## The Vision

Imagine asking: *"What are the most controversial claims about the trade?"*

The AI:
1. Searches the 200 sentences semantically
2. Finds the high-variance clusters
3. **Animates the camera** to fly to that cluster
4. **Highlights the relevant points** in gold
5. Responds with citations: *"Cluster 4 contains the most divisive opinions..."*

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    3D UMAP Visualization                        │ │
│  │                    (Three.js Stage)                             │ │
│  │  - Points colored by score                                      │ │
│  │  - Topic labels at cluster centroids                            │ │
│  │  - Animated camera movements                                    │ │
│  │  - Highlight/dim effects                                        │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    Chat Interface                               │ │
│  │  "Show me sentences about Steve Kerr's comments"                │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               │ HTTP/WebSocket
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DIRECTOR SERVER (FastAPI)                       │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────────┐│
│  │ Data Manager  │  │   Semantic    │  │    Director Agent         ││
│  │               │  │   Search      │  │    (LLM-powered)          ││
│  │ - points_data │  │               │  │                           ││
│  │ - cluster_stats│  │ - Embeddings │  │ - Answers questions       ││
│  │ - topic_names │  │ - Cosine sim  │  │ - Generates commands:     ││
│  │               │  │   retrieval   │  │   • focus_topic(id)       ││
│  │               │  │               │  │   • highlight_points(ids) ││
│  │               │  │               │  │   • filter_score(range)   ││
│  │               │  │               │  │   • reset()               ││
│  └───────────────┘  └───────────────┘  └───────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                               │
                               │ Reads from disk
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA FILES (from Visualizer)                    │
│  test_output/run_YYYYMMDD_HHMMSS/                                   │
│  ├── points_data.json      # {x, y, z, sentence, scores, cluster}   │
│  ├── cluster_stats.json    # {centroid, count, means per model}     │
│  ├── topic_names.json      # {gpt: {0: "name"}, claude: {...}}      │
│  ├── sentence_ratings.csv  # Original rated data                    │
│  └── interactive_umap.html # Standalone visualization               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Stage (Frontend - `static/`)

The 3D visualization that receives commands from the Director.

**Files:**
- `static/index.html` - Main page with chat UI
- `static/js/stage.js` - Three.js visualization + Puppeteer commands
- `static/css/style.css` - Dark theme styling

**Puppeteer Commands (methods on StageController):**
```javascript
stage.focusTopic(clusterId)      // Fly camera to cluster centroid
stage.highlightPoints([ids])     // Highlight specific points, dim others
stage.filterScore({min, max})    // Hide points outside range (TODO)
stage.resetView()                // Reset all highlighting, fly to center
stage.setColors(model)           // Change coloring model (gpt/claude/gemini)
```

### 2. Director Agent (`psychology_of_news/director.py`)

LLM-powered agent that interprets user questions and generates visualization commands.

**Input:**
- User query
- Retrieved context sentences (from semantic search)
- Topic names dictionary

**Output (DirectorResponse):**
```json
{
  "answer": "Based on the data, there are 15 sentences discussing...",
  "actions": [
    {"type": "focus_topic", "target": 3, "description": "Focusing on trade rumors"},
    {"type": "highlight_points", "target": [12, 45, 67], "description": "Key sentences"}
  ]
}
```

**Action Types:**
| Type | Target | Effect |
|------|--------|--------|
| `focus_topic` | cluster_id (int) | Camera flies to cluster centroid |
| `highlight_points` | [point_ids] | Highlights points, dims others |
| `filter_score` | {min, max} | Hides points outside range |
| `reset` | null | Resets view and highlighting |

### 3. Data Manager (`server.py`)

Loads and indexes the visualization data for the Director.

**Data Sources:**
- `points_data.json` - Full point data with x,y,z coordinates
- `cluster_stats.json` - Cluster centroids and statistics
- `topic_names.json` - AI-generated topic names per model

**Semantic Search:**
- Uses `all-MiniLM-L6-v2` embeddings
- Cosine similarity retrieval
- Returns top-k most relevant sentences for RAG context

### 4. Visualizer Export (`psychology_of_news/visualizer.py`)

The existing UMAP visualizer now exports data for the server:

```python
# Saves alongside the HTML:
output_dir/
├── interactive_umap.html   # Standalone (works without server)
├── points_data.json        # For server
├── cluster_stats.json      # For server
└── topic_names.json        # For server
```

---

## Data Flow

### Startup
```
1. Server starts
2. DataManager finds latest run_* directory
3. Loads points_data.json, cluster_stats.json, topic_names.json
4. Generates embeddings for semantic search
5. Frontend fetches /api/data to initialize Three.js scene
```

### Chat Query
```
1. User types: "What do people think about Steve Kerr?"
2. Frontend POSTs to /api/chat
3. Server:
   a. Semantic search finds relevant sentences
   b. Director Agent receives query + context
   c. LLM generates answer + actions
4. Response sent to frontend
5. Frontend:
   a. Displays answer in chat
   b. Executes actions (focusTopic, highlightPoints, etc.)
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves frontend HTML |
| `/api/status` | GET | Server status, sentence count |
| `/api/data` | GET | Full points/clusters/topics for init |
| `/api/chat` | POST | Send query, get answer + actions |

### POST /api/chat
```json
// Request
{
  "query": "Which sources are most negative about the trade?",
  "model": "gpt-5-nano"
}

// Response
{
  "answer": "Based on my analysis, sportskeeda.com and TotalProSports tend to...",
  "actions": [
    {"type": "focus_topic", "target": 4, "description": "Trade speculation cluster"},
    {"type": "highlight_points", "target": [23, 45, 89], "description": "Negative sentences"}
  ]
}
```

---

## Running the System

### 1. Generate Data
```bash
source venv/bin/activate
python test_local.py test200
```

This creates:
- `test_output/run_YYYYMMDD_HHMMSS/points_data.json`
- `test_output/run_YYYYMMDD_HHMMSS/cluster_stats.json`
- `test_output/run_YYYYMMDD_HHMMSS/topic_names.json`

### 2. Start Server
```bash
source venv/bin/activate
python server.py
```

Server runs at `http://localhost:8000`

### 3. Open Browser
Navigate to `http://localhost:8000`

---

## Future Enhancements

### Phase 2: Smooth Animations
- Use Tween.js for smooth camera transitions
- Animated highlighting with glow effects
- Loading states during LLM thinking

### Phase 3: Rich Citations
- Clickable sentence references in answers
- "Show me this sentence" buttons
- Source links that open articles

### Phase 4: Advanced Queries
- "Compare what GPT and Claude think"
- "Show sentences where models disagree"
- Time-based filtering if dates available

### Phase 5: Voice Interface
- Speech-to-text input
- Text-to-speech narration
- Hands-free exploration

---

## Pro Tips

1. **Model Key Consistency**: Points use lowercase keys (`gpt`, `claude`, `gemini`), not `GPT_score`

2. **Coordinate System**: UMAP coords are arbitrary scale, centered around 0. Camera positioning uses these directly.

3. **Topic Names**: Each model generates its own topic names. The frontend switches based on selected color model.

4. **Embeddings**: Search embeddings are generated at server startup. For large datasets, consider pre-computing.

5. **Actions Array**: Director can return multiple actions. Frontend executes them in order.

---

## File Reference

```
psychology_of_news/
├── server.py                    # FastAPI server
├── psychology_of_news/
│   ├── director.py              # LLM Director Agent
│   ├── visualizer.py            # UMAP + data export
│   ├── config.py                # Model configs
│   ├── rater.py                 # Triple-LLM rating
│   └── static/
│       ├── index.html           # Frontend HTML
│       ├── js/stage.js          # Three.js Stage
│       └── css/style.css        # Styling
├── test_output/
│   └── run_YYYYMMDD_HHMMSS/
│       ├── points_data.json     # Point coordinates + data
│       ├── cluster_stats.json   # Cluster statistics
│       ├── topic_names.json     # AI topic names
│       └── interactive_umap.html # Standalone viz
└── docs/
    ├── LLM_MODEL_GUIDE.md       # Model quirks
    ├── AGENT_HANDOFF.md         # Session context
    └── NARRATIVE_VIS_ARCHITECTURE.md  # This file
```
