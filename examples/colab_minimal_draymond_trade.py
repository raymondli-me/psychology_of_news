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
# CELL 3: Configure & Run
# =============================================================================
from psychology_of_news import Analyzer, Config

config = Config(
    topic="Draymond Green trade",
    output_dir="/content/output",

    # Customize the rating task (shown in visualization)
    rating_question="How likely does this imply Draymond Green will be traded?",
    scale_low="No trade implication",
    scale_mid="Neutral/ambiguous",
    scale_high="Trade very likely",
)

analyzer = Analyzer(config)
results = analyzer.run_all()

# =============================================================================
# CELL 4: Display
# =============================================================================
from IPython.display import HTML
with open(results["visualization"]) as f:
    display(HTML(f.read()))


# =============================================================================
# ALTERNATIVE: Use a preset for common analysis types
# =============================================================================
# from psychology_of_news.config import trade_analysis, sentiment_analysis
#
# # Trade analysis preset
# config = trade_analysis("LeBron James")
#
# # Sentiment analysis preset
# config = sentiment_analysis("Tesla stock")
#
# # Then run: Analyzer(config).run_all()
