# Agent Handoff: Psychology of News

> Last updated: January 2026
> Complete context for future agents working on this project

---

## Project Overview

**Purpose:** Analyze news articles about a topic using triple-LLM rating (GPT, Claude, Gemini) and visualize sentence-level sentiment/predictions in an interactive 3D UMAP.

**Core workflow:**
1. Fetch articles from Event Registry API
2. Extract sentences containing topic keywords
3. Rate each sentence with 3 LLMs (1-10 scale)
4. Generate interactive 3D UMAP visualization with topic clustering

---

## Key Files

| File | Purpose |
|------|---------|
| `psychology_of_news/config.py` | Model configs, prompts, settings |
| `psychology_of_news/rater.py` | Async LLM rating with retries |
| `psychology_of_news/visualizer.py` | 3D UMAP + topic clustering |
| `psychology_of_news/analyzer.py` | Main orchestrator class |
| `test_local.py` | Local testing script |
| `docs/LLM_MODEL_GUIDE.md` | Model-specific rules and fixes |

---

## Working Model Configuration (Jan 2026)

```python
models = [
    ModelConfig(name="GPT", model_id="openai/gpt-5-nano"),
    ModelConfig(name="Claude", model_id="anthropic/claude-sonnet-4-5"),
    ModelConfig(name="Gemini", model_id="gemini/gemini-2.5-flash"),
]
```

---

## Critical Pro Tips

### 1. GPT-5 Models Need `max_tokens=1000`

GPT-5 uses internal reasoning tokens. With low `max_tokens` (10-100), it returns **empty responses**.

```python
if "gpt-5" in model_id.lower():
    max_tokens = 1000
```

### 2. Gemini 2.5+ Needs `reasoning_effort="none"`

Gemini 2.5 has a "thinking" feature that causes litellm parsing bugs. Disable it:

```python
if "gemini-2.5" in model_id.lower() or "gemini-3" in model_id.lower():
    kwargs["reasoning_effort"] = "none"
```

**Bonus:** This also makes requests ~96% cheaper!

### 3. nest_asyncio Breaks Async Timeouts

**Never** apply `nest_asyncio.apply()` globally. Only use in Jupyter/Colab:

```python
def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

if _in_notebook():
    import nest_asyncio
    nest_asyncio.apply()
```

### 4. Don't Pass `reasoning_effort` to GPT via Chat Completions

GPT-5 reasoning_effort only works via Responses API, not chat completions. You'll get `BadRequestError: Unsupported value`.

### 5. URL Mapping for Clickable Points

The visualization needs URLs to make points clickable. When loading sentence data separately from articles:

```python
# Load articles with URLs
articles = pd.read_csv('articles.csv')
url_map = dict(zip(articles['title'], articles['url']))

# Add to rated sentences
df['url'] = df['article_title'].map(url_map)
```

---

## Visualization Features

### Topic Label Colors: Percentile-Based

Topic labels use **percentile ranking** among clusters, not raw scores. This spreads colors across the gradient even when all scores are low (e.g., 1-3).

```javascript
// Sort clusters by score, assign percentile-based color
clusterScores.sort((a, b) => a.score - b.score);
clusterScores.forEach((item, idx) => {
    percentileMap[item.clusterId] = 1 + (idx / (clusterScores.length - 1)) * 9;
});
```

Legend clarifies:
- **Points:** raw score (1-10)
- **Labels:** percentile rank (relative to other clusters)

### Model Buttons with ID Subtitles

Each model button shows the short model ID (e.g., `gpt-5-nano`) for clarity:

```python
@property
def short_model_id(self) -> str:
    if "/" in self.model_id:
        return self.model_id.split("/", 1)[1]
    return self.model_id
```

---

## Local Testing

### Setup

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy litellm nltk sentence-transformers umap-learn hdbscan eventregistry

# Create .env file
cat > .env << EOF
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
EVENT_REGISTRY_API_KEY=...
EOF
```

### Commands

```bash
python test_local.py fetch    # Fetch articles (Event Registry API)
python test_local.py rate     # Rate sentences (LLM APIs)
python test_local.py viz      # Create visualization only
python test_local.py test200  # Rate 200 existing sentences (timestamped)
```

### Timestamped Runs

`test200` creates timestamped output directories:
```
test_output/run_20260108_132100/
├── sentence_ratings.csv
└── interactive_umap.html
```

---

## Common Errors & Solutions

| Error | Cause | Fix |
|-------|-------|-----|
| GPT-5 returns empty | `max_tokens` too low | Set to 1000 |
| `cannot access local variable 'thought_signatures'` | Gemini 2.5 thinking feature | Pass `reasoning_effort="none"` |
| `Timeout context manager should b...` | `nest_asyncio` applied globally | Only apply in notebooks |
| `BadRequestError: Unsupported value` | `reasoning_effort` on GPT chat completions | Don't pass it |
| Clicking points doesn't open URL | No `url` column in data | Join from articles.csv |
| All topic labels same color | Raw scores clustered low | Use percentile ranking |

---

## API Keys

| Provider | Env Variable | Notes |
|----------|-------------|-------|
| OpenAI | `OPENAI_API_KEY` | |
| Anthropic | `ANTHROPIC_API_KEY` | |
| Google | `GOOGLE_API_KEY` | |
| Event Registry | `EVENT_REGISTRY_API_KEY` | For article fetching |

---

## Session Summary (Jan 8, 2026)

### What Was Done

1. **Debugged Gemini 2.5 parsing bug** - Fixed with `reasoning_effort="none"`
2. **Fixed GPT-5 empty responses** - Increased `max_tokens` to 1000
3. **Fixed async timeout bug** - Conditional `nest_asyncio.apply()`
4. **Updated to latest models** - gpt-5-nano, claude-sonnet-4-5, gemini-2.5-flash
5. **Added model ID subtitles** - Show actual model IDs in UI buttons
6. **Added color gradient legend** - Explains score scale
7. **Implemented percentile-based label colors** - Better visual differentiation
8. **Fixed clickable URLs** - Join URLs from articles.csv
9. **Created test200 command** - Rate 200 sentences with timestamped output
10. **Wrote documentation** - LLM_MODEL_GUIDE.md + this handoff

### Test Results (200 sentences)

| Model | Mean | Min | Max |
|-------|------|-----|-----|
| GPT (gpt-5-nano) | 1.43 | 1 | 8 |
| Claude (claude-sonnet-4-5) | 1.49 | 1 | 7 |
| Gemini (gemini-2.5-flash) | 2.17 | 1 | 9 |

10 topic clusters detected, AI-generated labels working for all 3 models.

---

## Next Steps / Ideas

- [ ] Add confidence intervals or agreement metrics to visualization
- [ ] Support for custom embedding models
- [ ] Batch processing for very large datasets
- [ ] Export cluster summaries as markdown reports
