# Agent Handoff Document - Psychology of News Visualization

## Project Overview
A Three.js-based interactive visualization of 200 sentences about Draymond Green trade rumors, rated by 3 LLMs (GPT-5-nano, Claude Sonnet 4.5, Gemini 2.5 Flash) on a 1-10 scale for "how likely does this imply Draymond Green will be traded?"

## Key Features
1. **3D Scatter Plot**: Points colored by selected model's score (blue=low, yellow=mid, red=high)
2. **Topic Clustering**: UMAP-reduced embeddings with topic labels (percentile-ranked colors)
3. **RAG Chat Sidebar**: "Data Director" agent that answers questions about the data
4. **Citation System**: AI responses include clickable citations that highlight points on the visualization

## Architecture

### Frontend (`psychology_of_news/static/index.html`)
- Three.js for 3D visualization
- OrbitControls for camera navigation
- Chat sidebar with model selection (Gemini/Claude/GPT)
- Tooltip system (shared between canvas hover and chat citations)

### Backend
- **`server.py`**: FastAPI server with endpoints:
  - `GET /api/data` - Returns all points, cluster stats, topic names
  - `POST /api/chat` - RAG chat endpoint (semantic search + director agent)

- **`director.py`**: Pydantic AI agent that:
  - Takes user query + semantic search context
  - Generates natural language response with stage directions
  - Returns actions: `focus_topic`, `highlight_points`, `reset`

### Data Pipeline
- SQLite database with sentence embeddings
- Sentence-transformers for embedding generation
- Semantic search finds relevant sentences for RAG context

## Recent Bug Fixes (Jan 8, 2026)

### 1. Citation Tooltip Not Appearing
**Problem**: Canvas `mousemove` handler was racing with chat citation tooltip - hiding it immediately after showing.

**Solution**: Added `window.chatTooltipActive` flag:
- Set to `true` in `showCiteTooltip()`
- Set to `false` in `hideCiteTooltip()`
- Canvas mousemove handler returns early when flag is true

**Location**: `index.html` lines 322-327, 573-574, 611-612

### 2. Director Using Wrong IDs for Highlighting
**Problem**: Director was referencing context by loop index [0], [1] instead of actual database IDs.

**Solution**: Updated `director.py` to format context with actual IDs:
```python
actual_id = item.get('id', i)  # Use actual database ID
context_str += f"[ID:{actual_id}] Cluster: {cluster_name} | ..."
```

**Location**: `director.py` - context formatting section

### 3. Fallback Highlighting
**Problem**: If director didn't specify IDs to highlight, nothing was highlighted.

**Solution**: Added fallback in `server.py`:
```python
if not highlighted_ids:
    highlighted_ids = [item["id"] for item in context]
```

## Key Code Sections

### Tooltip CSS (`index.html:14`)
```css
#tooltip {
    position: fixed;
    z-index: 2000;
    max-width: 420px;
    /* ... */
}
```

### showCiteTooltip Function (`index.html:540-605`)
- Takes point index and DOM element
- Builds HTML with scores for all 3 models
- Positions tooltip to LEFT of chat sidebar
- Sets `chatTooltipActive` flag

### highlightPoints Function (`index.html:425-458`)
- Takes array of point IDs
- Enlarges and brightens matching points
- Dims non-matching points
- Adds white outline rings to highlighted points

### Director Agent (`director.py`)
- Uses Pydantic AI with structured output
- Model: `gemini-2.5-flash` (or selected chat model)
- Returns `DirectorResponse` with answer + actions

## File Structure
```
psychology_of_news/
├── server.py                 # FastAPI backend
├── psychology_of_news/
│   ├── director.py           # Pydantic AI director agent
│   ├── static/
│   │   └── index.html        # Main visualization UI
│   └── data/
│       └── draymond_embeddings.db  # SQLite with embeddings
└── venv/                     # Python virtual environment
```

## Running the Project
```bash
cd /Users/raymondli701/2026_01_07_workspace/psychology_of_news
source venv/bin/activate
python server.py
# Visit http://localhost:8000
```

## Model IDs Used
- GPT: `gpt-5-nano`
- Claude: `claude-sonnet-4-5`
- Gemini: `gemini-2.5-flash`

## Known Working State
- 3D visualization renders correctly
- Model switching works (GPT/Claude/Gemini/Agreement)
- Topic labels show with percentile-ranked colors
- Chat sidebar opens/closes
- RAG search returns relevant context
- Director generates responses with citations
- Citation cards appear below AI responses
- **Citation tooltip now appears to left of sidebar on hover**
- Point highlighting works when AI responds
- Click citation to fly camera to that point

## Next Steps / Potential Improvements
- Add more data points beyond 200
- Improve director's citation accuracy
- Add export/save functionality
- Consider adding time-series analysis if data has timestamps
- Add more sophisticated topic clustering
