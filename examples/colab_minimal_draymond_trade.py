"""
Minimal Colab Script: Draymond Green Trade Analysis
====================================================
Copy to a Colab notebook. That's it.
"""

# =============================================================================
# CELL 1: Install
# =============================================================================
# !pip install -q git+https://github.com/raymondli-me/psychology_of_news.git

# =============================================================================
# CELL 2: API Keys
# =============================================================================
import os
os.environ["EVENT_REGISTRY_API_KEY"] = "YOUR_KEY"
os.environ["OPENAI_API_KEY"] = "sk-proj-..."
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-api03-..."
os.environ["GOOGLE_API_KEY"] = "AIzaSy..."

# =============================================================================
# CELL 3: Run Analysis
# =============================================================================
from psychology_of_news import Analyzer, Config

config = Config(topic="Draymond Green trade", output_dir="/content/output")
analyzer = Analyzer(config)
results = analyzer.run_all()

# =============================================================================
# CELL 4: Display
# =============================================================================
from IPython.display import HTML
with open(results["visualization"]) as f:
    display(HTML(f.read()))
