"""
Minimal Colab Notebook Script

Copy this to a Colab notebook for daily analysis.
"""

# Cell 1: Install
# !pip install git+https://github.com/raymondli-me/psychology_of_news.git
# !pip install litellm eventregistry sentence-transformers umap-learn hdbscan

# Cell 2: Setup API Keys
import os
os.environ["EVENT_REGISTRY_API_KEY"] = "YOUR_KEY"  # Get from eventregistry.org
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
os.environ["GOOGLE_API_KEY"] = "AIza..."

# Cell 3: Run Analysis
from psychology_of_news import Analyzer, Config

config = Config(
    topic="Draymond Green trade",  # CHANGE THIS!
    output_dir="/content/output",
    max_sentences=100,
)

analyzer = Analyzer(config)
results = analyzer.run_all()

# Cell 4: Display Visualization
from IPython.display import HTML
with open(results["visualization"]) as f:
    display(HTML(f.read()))
