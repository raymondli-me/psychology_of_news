"""
Configuration for psychology_of_news analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass
class ModelConfig:
    """Config for a single LLM."""
    name: str
    model_id: str
    max_tokens: int = 10  # GPT-5-mini needs more, handled in rater


@dataclass
class Config:
    """
    Main configuration for news analysis.

    Example:
        config = Config(
            topic="Draymond Green trade",
            input_csv="/path/to/scraped_articles.csv",
            output_dir="/path/to/output"
        )
    """

    # What we're analyzing
    topic: str = "Draymond Green trade"

    # Human-readable description of what's being rated (shown in visualization)
    rating_question: str = "How strongly does this imply {topic} will happen?"
    scale_low: str = "No implication"
    scale_mid: str = "Neutral"
    scale_high: str = "Strongly implies"

    # The rating prompt - {topic} and {text} are replaced
    prompt_template: str = """Rate this sentence on how strongly it implies {topic} will happen.

Sentence: "{text}"

Score from 1-10:
1 = No implication at all
5 = Neutral/ambiguous
10 = Strongly implies it will happen

Reply with ONLY a single number (1-10), nothing else."""

    @property
    def rating_display(self) -> dict:
        """Get rating info for display in visualization."""
        return {
            "question": self.rating_question.format(topic=self.topic),
            "scale": {
                "low": f"1 = {self.scale_low}",
                "mid": f"5 = {self.scale_mid}",
                "high": f"10 = {self.scale_high}"
            }
        }

    # Input: CSV with columns [title, body_text, source, date, url]
    input_csv: Optional[str] = None

    # Output directory for all results
    output_dir: str = "./output"

    # Models - default triple-LLM setup
    models: list = field(default_factory=lambda: [
        ModelConfig(name="GPT", model_id="openai/gpt-5-mini"),
        ModelConfig(name="Claude", model_id="anthropic/claude-sonnet-4-5"),
        ModelConfig(name="Gemini", model_id="gemini/gemini-2.5-flash-preview-09-2025"),
    ])

    # Sentence extraction
    min_sentence_length: int = 30
    max_sentence_length: int = 500
    max_sentences: int = 200
    require_topic_mention: bool = True  # Only sentences mentioning keyword
    keyword_filter: str = None  # Custom keyword to filter by (default: auto from topic)

    # Rating settings
    max_concurrent_per_model: int = 5
    batch_size: int = 10

    # Visualization settings
    create_umap: bool = True
    create_haam: bool = True
    create_figures: bool = True

    # API keys - if None, uses environment variables
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    def __post_init__(self):
        """Set up API keys and output directory."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key
        if self.google_api_key:
            os.environ["GOOGLE_API_KEY"] = self.google_api_key

    def get_prompt(self, text: str) -> str:
        """Format prompt with topic and text."""
        return self.prompt_template.format(topic=self.topic, text=text)

    @property
    def topic_keyword(self) -> str:
        """Get keyword for sentence filtering."""
        if self.keyword_filter:
            return self.keyword_filter
        # Auto-extract: "Draymond Green trade" -> "Draymond"
        return self.topic.split()[0]


# Preset configs for common use cases
def trade_analysis(player: str) -> Config:
    """Preset for player trade analysis."""
    return Config(
        topic=f"{player} trade",
        rating_question=f"How likely does this imply {player} will be traded?",
        scale_low="No trade implication",
        scale_mid="Neutral/ambiguous",
        scale_high="Trade very likely",
        prompt_template=f"""Rate this sentence on how strongly it implies {player} will be traded.

Sentence: "{{text}}"

Score from 1-10:
1 = No trade implication at all
5 = Neutral/ambiguous
10 = Strongly implies trade will happen

Reply with ONLY a single number (1-10), nothing else."""
    )


def sentiment_analysis(subject: str) -> Config:
    """Preset for sentiment analysis."""
    return Config(
        topic=f"{subject} sentiment",
        rating_question=f"What is the sentiment about {subject}?",
        scale_low="Very negative",
        scale_mid="Neutral",
        scale_high="Very positive",
        prompt_template=f"""Rate the sentiment about {subject} in this sentence.

Sentence: "{{text}}"

Score from 1-10:
1 = Very negative
5 = Neutral
10 = Very positive

Reply with ONLY a single number (1-10), nothing else."""
    )


def impact_analysis(topic: str) -> Config:
    """Preset for news impact/significance analysis."""
    return Config(
        topic=topic,
        rating_question=f"How significant/impactful is this news about {topic}?",
        scale_low="Minor/routine",
        scale_mid="Moderate",
        scale_high="Major breaking news",
        prompt_template=f"""Rate how significant or impactful this news is regarding {topic}.

Sentence: "{{text}}"

Score from 1-10:
1 = Minor/routine news
5 = Moderately significant
10 = Major breaking news

Reply with ONLY a single number (1-10), nothing else."""
    )
