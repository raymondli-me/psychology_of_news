# =============================================================================
# CELL 1: Install (run this cell first)
# =============================================================================
!pip install -q git+https://github.com/raymondli-me/psychology_of_news.git
!pip install -q litellm eventregistry sentence-transformers umap-learn hdbscan

# =============================================================================
# CELL 2: API Keys (fill in your keys)
# =============================================================================
import os
os.environ["EVENT_REGISTRY_API_KEY"] = "YOUR_KEY"  # Get from eventregistry.org
os.environ["OPENAI_API_KEY"] = "sk-proj-..."
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-api03-..."
os.environ["GOOGLE_API_KEY"] = "AIzaSy..."

# =============================================================================
# CELL 3: Run Analysis
# =============================================================================
from psychology_of_news import Analyzer, Config

config = Config(
    topic="Draymond Green trade",
    output_dir="/content/output",
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
