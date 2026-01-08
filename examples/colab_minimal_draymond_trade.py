# =============================================================================
# CELL 1: Install (run this cell first)
# =============================================================================
!pip install -q git+https://github.com/raymondli-me/psychology_of_news.git
!pip install -q litellm eventregistry sentence-transformers umap-learn hdbscan nest-asyncio

# =============================================================================
# CELL 2: Load API Keys from Colab Secrets
# =============================================================================
import os
from google.colab import userdata

os.environ["EVENT_REGISTRY_API_KEY"] = userdata.get("EVENT_REGISTRY_NEWS")
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = userdata.get("CLAUDE_API_KEY")
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

# =============================================================================
# CELL 3: Run Analysis
# =============================================================================
from psychology_of_news import Analyzer, Config

config = Config(
    # What to search for (Event Registry query)
    topic="Draymond Green trade",
    output_dir="/content/output",

    # Sentence filtering
    max_sentences=100,
    min_sentence_length=30,
    max_sentence_length=500,

    # Keyword filter: which sentences to include
    # - None: auto-extract from topic ("Draymond Green")
    # - str: single keyword ("Draymond")
    # - list: multiple keywords (["Draymond", "trade", "Warriors"])
    keyword_filter=None,
    keyword_logic="any",  # "any" (OR) or "all" (AND)

    # Examples:
    # keyword_filter=["Draymond", "trade"], keyword_logic="any"   -> has Draymond OR trade
    # keyword_filter=["Draymond", "trade"], keyword_logic="all"   -> has Draymond AND trade

    # Rating task (shown in visualization)
    rating_question="How likely does this imply Draymond Green will be traded?",
    scale_low="No trade implication",
    scale_mid="Neutral/ambiguous",
    scale_high="Trade very likely",
)

results = Analyzer(config).run_all()

# =============================================================================
# CELL 4: Display Visualization
# =============================================================================
from IPython.display import HTML
display(HTML(open(results["visualization"]).read()))
