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
    # What to analyze
    topic="Draymond Green trade",
    output_dir="/content/output",

    # Sentence filtering criteria
    max_sentences=100,           # Max sentences to rate (default: 200)
    min_sentence_length=30,      # Min characters (default: 30)
    max_sentence_length=500,     # Max characters (default: 500)
    require_topic_mention=True,  # Must contain keyword (default: True)
    keyword_filter=None,         # Custom keyword, or None for auto ("Draymond")

    # Rating task shown in visualization
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
