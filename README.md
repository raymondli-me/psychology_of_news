# Psychology of News

Triple-LLM news sentiment analysis with interactive 3D UMAP visualization and HAAM analysis.

## What It Does

1. **Fetch** news articles about any topic from Event Registry
2. **Rate** each sentence using 3 LLMs (GPT, Claude, Gemini)
3. **Visualize** as interactive 3D UMAP with topic clusters
4. **Analyze** with HAAM (Human-AI Alignment Model)

## Quick Start

```python
from psychology_of_news import Analyzer, Config

# Configure
config = Config(
    topic="Draymond Green trade",
    output_dir="./output"
)

# Run full pipeline
analyzer = Analyzer(config)
results = analyzer.run_all(event_registry_key="YOUR_KEY")

# Or step by step:
analyzer.fetch()      # Get articles from Event Registry
analyzer.rate()       # Rate sentences with 3 LLMs
analyzer.visualize()  # Create interactive UMAP
analyzer.run_haam()   # Run HAAM analysis
```

## Configuration

```python
config = Config(
    # Topic to analyze
    topic="Draymond Green trade",

    # Custom prompt template
    prompt_template="""Rate this sentence on how strongly it implies {topic} will happen.
Sentence: "{text}"
Score from 1-10. Reply with ONLY a single number.""",

    # Models (default is GPT-5-mini, Claude-4.5, Gemini-2.5-Flash)
    models=[
        ModelConfig(name="GPT", model_id="openai/gpt-4o"),
        ModelConfig(name="Claude", model_id="anthropic/claude-sonnet-4-5"),
        ModelConfig(name="Gemini", model_id="gemini/gemini-2.5-flash-preview-09-2025"),
    ],

    # Sentence filtering
    max_sentences=200,
    require_topic_mention=True,  # Only sentences mentioning the topic

    # API keys (or use environment variables)
    openai_api_key="sk-...",
    anthropic_api_key="sk-ant-...",
    google_api_key="AIza...",
)
```

## Presets

```python
from psychology_of_news.config import trade_analysis, sentiment_analysis

# Trade likelihood analysis
config = trade_analysis("LeBron James")

# Sentiment analysis
config = sentiment_analysis("Tesla stock")
```

## Environment Variables

```bash
export EVENT_REGISTRY_API_KEY="your-key"
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
```

## Output

- `sentence_ratings.csv` - Rated sentences with scores from each model
- `interactive_umap.html` - 3D visualization with topic clusters
- `haam_outputs/` - HAAM analysis (wordclouds, PC analysis, UMAP)

## Requirements

- Python 3.10+
- Event Registry API key
- OpenAI, Anthropic, and Google API keys

## Installation

```bash
pip install -e .
```

Or install dependencies manually:

```bash
pip install pandas numpy nltk litellm eventregistry sentence-transformers umap-learn hdbscan scikit-learn
```

## HAAM Integration

This package uses [HAAM](https://github.com/raymondli-me/haam) for Human-AI Alignment analysis. Set the `HAAM_PATH` environment variable to point to your HAAM installation:

```bash
export HAAM_PATH="/path/to/haam_repo"
```

## License

MIT
