# =============================================================================
# CELL 1: Install (run once, then restart runtime)
# =============================================================================
!pip install -q git+https://github.com/raymondli-me/psychology_of_news.git nest-asyncio

# =============================================================================
# CELL 2: Load API Keys + Suppress Warnings
# =============================================================================
import os
import warnings
warnings.filterwarnings("ignore")

from google.colab import userdata

os.environ["EVENT_REGISTRY_API_KEY"] = userdata.get("EVENT_REGISTRY_NEWS")
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = userdata.get("CLAUDE_API_KEY")
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

# =============================================================================
# CELL 3: Run Analysis (start with fewer sentences to test)
# =============================================================================
from psychology_of_news import Analyzer, Config

config = Config(
    topic="Draymond Green trade",
    output_dir="/content/output",

    # START SMALL to test - increase once working
    max_sentences=20,  # Start with 20, increase to 100+ once confirmed working

    # Keyword filter
    keyword_filter=None,  # Auto: "Draymond Green"
    keyword_logic="any",

    # Rating task
    rating_question="How likely does this imply Draymond Green will be traded?",
    scale_low="No trade implication",
    scale_mid="Neutral/ambiguous",
    scale_high="Trade very likely",
)

print("Starting analysis with 20 sentences (test run)...")
results = Analyzer(config).run_all()

# =============================================================================
# CELL 4: Display Visualization
# =============================================================================
from IPython.display import HTML
display(HTML(open(results["visualization"]).read()))
