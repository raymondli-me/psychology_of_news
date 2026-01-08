"""
Psychology of News - Triple-LLM Sentiment Analysis

Analyze news sentiment using GPT, Claude, and Gemini,
with HAAM analysis and interactive 3D visualizations.
"""

from .config import Config
from .analyzer import Analyzer

__version__ = "0.1.0"
__all__ = ["Config", "Analyzer"]
